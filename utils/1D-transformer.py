import torch

from config import BATCH_SIZE
from torch import nn

L = 158

encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(BATCH_SIZE, L, 256)
out = transformer_encoder(src)
