import torch
import torch.nn as nn
from dataclasses import dataclass


class Llama2(nn.Module):
    def __init__(self, cfg:dataclass):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, dtype=cfg.dtype)

        # Transformer block
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.n_layers)])

        # RMSNorm
        self.final_norm = RMSNorm(cfg.emb_dim)

        # Final output layer
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False, dtype=cfg.dtype)
    
    def forward(self,  in_idx:torch.Tensor):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds 
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

### RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, embd_dim:int, eps=1e-5):
        super().__init__()
        self.eps = eps # epsilon
        self.embd_dim = embd_dim
        self.weight = nn.Parameter(torch.ones(embd_dim)).float()
    
    def forward(self, x:torch.Tensor):
        means = x.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(means + self.eps)
        return (x_norm * self.weight).to(dtype=x.dtype)

### TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, cfg:dataclass):
        super().__init__()
        self.att = MultiHeadedAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dtype=cfg.dtype
        )

        self.ff = FeedForward(cfg)

        # RMSNorms
        self.norm1 = RMSNorm(cfg.emb_dim)
        self.norm2 = RMSNorm(cfg.emb_dim)
    
    def forward(self, x:torch.Tensor):
        # save x for the residual connections 
        residual = x
        x = self.norm1(x)
        x = self.att(x)
        x = x + residual

        # reset residual connections
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual
        
        return x

## RoPE
def precompute_rope_params(head_dim, theta_base=10_000, context_length=4096):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / ( theta_base ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin

def compute_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    x_rotated = (x * cos) + (rotated * sin)
    return x_rotated.to(dtype=x.dtype)

### Multi-headed Attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads, dtype=None):
        super().__init__()
        assert d_out % num_heads == 0, "The output dimension (d_out) mus be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads 

        # Set q, k, v, vectors and output projection
        self.W_query = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, d_out, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, bias=False, dtype=dtype)

        # Create token masking 
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        # Create RoPE parameters
        cos, sin = precompute_rope_params(head_dim=self.head_dim, context_length=context_length)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
    
    def forward(self, x:torch.Tensor):
        b, num_tokens, d_in = x.shape

        # Calculate the q, k, v vectors
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Remember the output dim is all of the q, k, v vectors concatonated. Need to split
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Now we need to swap the head dimension and the context dim
        queries:torch.Tensor = queries.transpose(1,2)
        keys:torch.Tensor = keys.transpose(1,2)
        values:torch.Tensor = values.transpose(1,2)

        # compute the positional encodings for the key and the values
        keys = compute_rope(keys, self.cos, self.sin)
        queries = compute_rope(queries, self.cos, self.sin)

        # Attention scores via scaled dot-product
        attn_scores = queries @ keys.transpose(2,3)

        # Masking
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use mask to fill upper triangle of attention to -inf for norm
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # normalizing the weights
        attn_weights = torch.softmax(attn_scores / keys.size(-1)**0.5, dim=-1)

        context_vec = (attn_weights @ values).transpose(1,2)

        # combine the heads again self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # projection
        return context_vec