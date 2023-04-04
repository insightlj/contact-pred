import torch

from config import BATCH_SIZE
from torch import nn

L = 150
multihead_attn = nn.MultiheadAttention(256, 8, batch_first=True)
query = key = value = torch.ones((BATCH_SIZE, L, 256))

ls = []
for j in range(64):
    for i in range(6):
        attn_output, attn_output_weights = multihead_attn(query, key, value)
        query = key = value = attn_output
    ls.append(attn_output_weights)

attn_output_weights = torch.stack(ls, dim=1)

