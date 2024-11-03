import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPT, GPTConfig, EncoderConfig, DecoderConfig

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------

# I/O
out_dir = 'out'
eval_interval = 10
log_interval = 1
eval_iters = 200
total_training_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' 

# WandB logging
wandb_log = False # disabled by default
wandb_project = 'AC_Transformer'
wandb_run_name = 'test_run_1'

# MDP
max_path_length = 10
max_relator_length = 256

# Data
data_dir = 'data'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1 this is micro batch size

# Model
encoder_args = dict(
    block_size = 4*max_relator_length,
    vocab_size = 8, # {-2, -1, 0, 1, 2} + 2 ==> pad=2, 5 rounded up to 8
    n_layer = 2,
    n_head = 4,
    n_embd = 128,
    n_segments = 4,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    causal = False
)

decoder_args = dict(
    block_size = max_path_length + 2,
    vocab_size = 16, # {0:pad, 1-12:actions, 13:start, 14:end}, 15 rounded up to 16
    n_layer = 2,
    n_head = 4,
    n_embd = 128,
    dropout = 0.0,
    bias = True, # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    causal = True
)

# Optimizer (AdamW)
learning_rate = 6e-4
weight_decay = 1e-1
beta_1 = 0.9
beta_2 = 0.95
grad_clip = 1.0

# Learning rate
decay_lr = True
warmup_iters = 2000
lr_decay_iters = total_training_iters
min_lr = 6e-5

# System
device = 'cuda' if torch.cuda.is_available() else 'mps'
device_type = 'cuda' if 'cuda' in device else 'mps'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False

# Previous global variables for logging
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging

# -----------------------------------------------------------------------------
# DDP
# -----------------------------------------------------------------------------
backend = 'nccl'
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    ddp_world_size = 1
    ddp_rank = 0
next_token_predictions_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * decoder_args['block_size']
print(f"(padded) tokens per iteration will be: {next_token_predictions_per_iter:,}")

# -----------------------------------------------------------------------------
# Initializations
# -----------------------------------------------------------------------------
if master_process: os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
iter_num = 0
best_val_loss = 1e9

# -----------------------------------------------------------------------------
# Data Loader
# -----------------------------------------------------------------------------

class DataLoader:
    def __init__(self):
        self.metadata_train = pickle.load(open(os.path.join(data_dir, 'metadata_train.pkl'), 'rb'))
        self.metadata_val = pickle.load(open(os.path.join(data_dir, 'metadata_val.pkl'), 'rb'))
        data_train = np.fromfile(os.path.join(data_dir, 'data_train.bin'), dtype=self.metadata_train['dtype']).reshape(self.metadata_train['shape'])
        data_val = np.fromfile(os.path.join(data_dir, 'data_val.bin'), dtype=self.metadata_val['dtype']).reshape(self.metadata_val['shape'])
        train_permutation = np.random.permutation(self.metadata_train['shape'][0])
        val_permutation = np.random.permutation(self.metadata_val['shape'][0])
        self.data_train = data_train[train_permutation]
        self.data_val = data_val[val_permutation]
        self.train_index = ddp_rank
        self.val_index = ddp_rank
        self.train_epoch = 0 # number of times the training data has been iterated over
        self.val_epoch = 0 # number of times the validation data has been iterated over

    def get_batch(self, split, eval=False):
        if split == 'train':
            if eval:
                batch = self.data_train[np.random.choice(self.metadata_train['shape'][0], batch_size)]
            else:  
                if self.train_index + batch_size*ddp_world_size > self.metadata_train['shape'][0]:
                    train_permutation = np.random.permutation(self.metadata_train['shape'][0])
                    self.data_train = self.data_train[train_permutation]
                    self.train_index = ddp_rank
                    self.train_epoch += 1
                batch = self.data_train[self.train_index : self.train_index + batch_size * ddp_world_size : ddp_world_size]
                self.train_index += batch_size * ddp_world_size
        else:
            if eval:
                batch = self.data_val[np.random.choice(self.metadata_val['shape'][0], batch_size)]
            else:
                if self.val_index + batch_size*ddp_world_size > self.metadata_val['shape'][0]:
                    self.val_permutation = np.random.permutation(self.metadata_val['shape'][0])
                    self.val_index = ddp_rank
                    self.val_epoch += 1
                batch = self.data_val[self.val_index : self.val_index + batch_size * ddp_world_size : ddp_world_size]
                self.val_index += batch_size * ddp_world_size

        encoder_input = torch.from_numpy(batch[:, :4*max_relator_length].astype(np.int64)).to(device)
        decoder_input = torch.from_numpy(batch[:, 4*max_relator_length:-1].astype(np.int64)).to(device)
        targets = torch.from_numpy(batch[:, 4*max_relator_length+1:].astype(np.int64)).to(device)
        return encoder_input, decoder_input, targets

data_loader = DataLoader()

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
if init_from == 'scratch':
    print("Initializing a new model from scratch")
    encoder_config = EncoderConfig(**encoder_args)
    decoder_config = DecoderConfig(**decoder_args)
    model_config = GPTConfig(encoder=encoder_config, decoder=decoder_config)
    model = GPT(model_config)
else:
    print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder_args = checkpoint['encoder_args']
    decoder_args = checkpoint['decoder_args']
    encoder_config = EncoderConfig(**encoder_args)
    decoder_config = DecoderConfig(**decoder_args)
    model_config = GPTConfig(encoder=encoder_config, decoder=decoder_config)
    model = GPT(model_config)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary (delete weird prefixes)
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
model.to(device)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta_1, beta_2), device_type)

# Compile
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# DDP wrapper
if ddp: model = DDP(model, device_ids=[ddp_local_rank])

# Gradient scaler
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))

# Loss Estimator
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X_e, X_d, Y = data_loader.get_batch(split, eval=True)
            with ctx:
                logits, loss = model(X_e, X_d, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

X_e, X_d, Y = data_loader.get_batch('train')
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if wandb_log:
        wandb.log({
            "iter": iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr": lr,
            "mfu": running_mfu*100, # convert to percentage
        })
    if losses['val'] < best_val_loss or always_save_checkpoint:
        best_val_loss = losses['val']
        if iter_num > 0:
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'encoder_args': encoder_args,
                'decoder_args': decoder_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        with ctx:
            logits, loss = model(X_e, X_d, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X_e, X_d, Y = data_loader.get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        # if local_iter_num >= 5: # let the training loop settle a bit
        #     mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        #     running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")#, mfu {running_mfu*100:.2f}%")
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > total_training_iters:
        break

if ddp:
    destroy_process_group()