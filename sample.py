"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import numpy as np
from model import GPTConfig, GPT, EncoderConfig, DecoderConfig

from acmdp_jax import ACMDP 

max_relator_length = 256
mdp = ACMDP(max_relator_length)
# -----------------------------------------------------------------------------
out_dir = 'out' # ignored if init_from is not 'resume'
data_dir = 'data'
start_token = 13 # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 10 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = None # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cpu' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# init from a model saved in a specific directory
ckpt_path = os.path.join(out_dir, 'ckpt(1).pt')
checkpoint = torch.load(ckpt_path, map_location=device)
encoder_config = EncoderConfig(**checkpoint['encoder_args'])
decoder_config = DecoderConfig(**checkpoint['decoder_args'])
gptconf = GPTConfig(encoder=encoder_config, decoder=decoder_config)
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

x_decoder = torch.tensor([start_token], device=device).reshape(1, 1)
print(f"x_decoder.shape: {x_decoder.shape}")

metadata_val = pickle.load(open(os.path.join(data_dir, 'metadata_val.pkl'), 'rb'))
data_val = np.fromfile(os.path.join(data_dir, 'data_val.bin'), dtype=metadata_val['dtype']).reshape(metadata_val['shape'])
data = torch.tensor(data_val[np.random.choice(data_val.shape[0], 1)], device=device).reshape(1, -1)
x_encoder = data[:, :4*max_relator_length].long()

x_encoder_np = x_encoder.cpu().numpy().reshape(-1)
start = x_encoder_np[:2*max_relator_length] - 2
goal = x_encoder_np[2*max_relator_length:] - 2
start_tuple = tuple(tuple(int(x) for x in r) for r in mdp.array_to_tuple(start))
goal_tuple = tuple(tuple(int(x) for x in r) for r in mdp.array_to_tuple(goal))
print(f"start: {start_tuple}")
print(f"goal: {goal_tuple}")

def print_trajectory(start, goal, action_path):
    print(f"start: {tuple(tuple(int(x) for x in r) for r in mdp.array_to_tuple(start))}")
    state = start
    for action in action_path:
        print(f"action: {action}")
        state = mdp.transition(state, action)
        print(f"state: {tuple(tuple(int(x) for x in r) for r in mdp.array_to_tuple(state))}")
    print(f"goal reached: {np.all(state == goal)}")

def check_trajectory(start, goal, action_path):
    state = start
    for action in action_path:
        state = mdp.transition(state, action)
    return np.all(state == goal)

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x_encoder, x_decoder, max_new_tokens, temperature=temperature, top_k=top_k)
            y = y[y != 0]
            y = y[1:-1] - 1
            y = y.tolist()
            print(f"actions: {y},goal reached: {check_trajectory(start, goal, y)}")
            print('---------------')
