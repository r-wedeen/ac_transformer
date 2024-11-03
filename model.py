"""Forked from nanogpt"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.causal = config.causal
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal: att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection: TODO: useless?
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class CrossAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.decoder.n_embd % config.decoder.n_head == 0
        assert config.encoder.n_embd % config.encoder.n_head == 0
        self.y_kv = nn.Linear(config.encoder.n_embd, 2 * config.decoder.n_embd, bias=config.decoder.bias)
        self.x_q = nn.Linear(config.decoder.n_embd, config.decoder.n_embd, bias=config.decoder.bias)
        self.c_proj = nn.Linear(config.decoder.n_embd, config.decoder.n_embd, bias=config.decoder.bias)
        self.attn_dropout = nn.Dropout(config.decoder.dropout)
        self.resid_dropout = nn.Dropout(config.decoder.dropout)
        self.n_head = config.decoder.n_head
        self.n_embd_decoder = config.decoder.n_embd
        self.n_embd_encoder = config.encoder.n_embd
        self.dropout = config.decoder.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self, x, y):
        B, T, C = x.size() # batch size, decoder input length, n_embd_decoder
        B, S, D = y.size() # batch size, encoder input length, n_embd_encoder
        q = self.x_q(x) # (B, T, C) -> (B, T, C)
        k, v = self.y_kv(y).split(self.n_embd_decoder, dim=2) # (B, S, D) -> (B, S, C), (B, S, C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, S, hs)

        if self.flash:
            z = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, S)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            z = att @ v # (B, nh, T, S) x (B, nh, S, hs) -> (B, nh, T, hs)
        z = z.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        z = self.resid_dropout(self.c_proj(z))
        return z
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x_e):
        x_e = x_e + self.attn(self.ln_1(x_e))
        x_e = x_e + self.mlp(self.ln_2(x_e))
        return x_e
    
class DecoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.decoder.n_embd, bias=config.decoder.bias)
        self.attn = SelfAttention(config.decoder)
        self.ln_2 = LayerNorm(config.decoder.n_embd, bias=config.decoder.bias)
        self.cross_attn = CrossAttention(config)
        self.ln_3 = LayerNorm(config.decoder.n_embd, bias=config.decoder.bias)
        self.mlp = MLP(config.decoder)

    def forward(self, x_d, y_e):
        x_d = x_d + self.attn(self.ln_1(x_d))
        x_d = x_d + self.cross_attn(self.ln_2(x_d), y_e) #y_e already has layer norm applied
        x_d = x_d + self.mlp(self.ln_3(x_d))
        return x_d
    
@dataclass
class EncoderConfig:
    block_size: int = 4 * 256 # max_relator_length = 256
    vocab_size: int = 5 # {-2, -1, 0, 1, 2} + 2 ==> pad=2
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_segments: int = 4
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    causal: bool = False

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.encoder.vocab_size is not None
        assert config.encoder.block_size is not None
        self.config = config.encoder    

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(self.config.vocab_size, self.config.n_embd, padding_idx=2),
            wpe = nn.Embedding(self.config.block_size, self.config.n_embd),
            wse = nn.Embedding(self.config.n_segments, self.config.n_embd),
            drop = nn.Dropout(self.config.dropout),
            h = nn.ModuleList([EncoderBlock(self.config) for _ in range(self.config.n_layer)]),
            ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_encoder):
        device = idx_encoder.device
        b, t = idx_encoder.shape
        assert t == self.config.block_size, f"Cannot forward sequence of length {t}, block size is {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t//4)
        seg0 = torch.zeros(t//4, dtype=torch.long, device=device)
        seg1 = torch.ones(t//4, dtype=torch.long, device=device)
        seg2 = 2 * torch.ones(t//4, dtype=torch.long, device=device)
        seg3 = 3 * torch.ones(t//4, dtype=torch.long, device=device)
        seg = torch.cat([seg0, seg1, seg2, seg3], dim=0)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx_encoder) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        seg_emb = self.transformer.wse(seg) # segment embeddings of shape (t, n_embd)
        x_e = self.transformer.drop(tok_emb + pos_emb + seg_emb)
        for block in self.transformer.h:
            x_e = block(x_e)
        y_e = self.transformer.ln_f(x_e)
        return y_e
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= (self.transformer.wpe.weight.numel() 
                         + self.transformer.wse.weight.numel() 
                         + self.transformer.wte.weight.numel())
        return n_params

    
@dataclass
class DecoderConfig:
    block_size: int = 128 # max length of action path
    vocab_size: int = 15 # {0:pad, 1-12:actions, 13:start, 14:end}
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    causal: bool = True

class Decoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.decoder.vocab_size is not None
        assert config.decoder.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.decoder.vocab_size, config.decoder.n_embd, padding_idx=0),
            wpe = nn.Embedding(config.decoder.block_size, config.decoder.n_embd),
            drop = nn.Dropout(config.decoder.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder.n_layer)]),
            ln_f = LayerNorm(config.decoder.n_embd, bias=config.decoder.bias),
        ))
        self.lm_head = nn.Linear(config.decoder.n_embd, config.decoder.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # TODO: IS THIS NECESSARY?
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.decoder.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx_decoder, y_e, targets=None):
        device = idx_decoder.device
        b, t = idx_decoder.shape
        assert t <= self.config.decoder.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx_decoder) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x_d = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x_d = block(x_d, y_e)
        x_d = self.transformer.ln_f(x_d)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x_d)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x_d[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

@dataclass
class GPTConfig:
    encoder: EncoderConfig
    decoder: DecoderConfig

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        n_params_encoder = self.encoder.get_num_params()
        n_params_decoder = self.decoder.get_num_params()
        print("number of encoder parameters: %.2fM" % (n_params_encoder/1e6,))
        print("number of decoder parameters: %.2fM" % (n_params_decoder/1e6,))
        print("total number of parameters: %.2fM" % ((n_params_encoder + n_params_decoder)/1e6,))

    def get_num_params(self, non_embedding=True):
        return self.encoder.get_num_params(non_embedding) + self.decoder.get_num_params(non_embedding)

    def forward(self, idx_encoder, idx_decoder, targets=None):
        y_e = self.encoder(idx_encoder)
        logits, loss = self.decoder(idx_decoder, y_e, targets)
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx_decoder, idx_encoder, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx_decoder if idx_decoder.size(1) <= self.decoder_config.block_size else idx_decoder[:, -self.decoder_config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond, idx_encoder)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx_decoder = torch.cat((idx_decoder, idx_next), dim=1)

        return idx_decoder
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu
    