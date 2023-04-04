import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 应该放在main.py/main_transformer.py中

BATCH_SIZE = 1
EPOCH_NUM = 30
avg_L = 158
k_value = int(avg_L)
run_name = "transformer_minor"
error_file_name = 'transformer_minor.txt'

loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:0")
# device = torch.device("cpu")
