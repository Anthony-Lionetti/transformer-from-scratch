import torch
from dataclasses import dataclass

@dataclass
class Llama2Config7B:
    vocab_size = 32000      # Vocabulary size
    context_length = 4096   # Context length
    emb_dim = 4096          # Embedding dimension
    n_heads = 32            # Number of attention heads
    ff_bias = False         # Feed forward bias 
    n_layers = 32           # Number of layers
    hidden_dim = 11008      # NEW = Size of the intermediate dimension in FeedForward
    dtype = torch.bfloat16  # NEW: Lower-precision dtype to reduce memory usage
