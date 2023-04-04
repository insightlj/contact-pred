import torch

BATCH_SIZE = 1
EPOCH_NUM = 30
avg_L = 158
k_value = int(avg_L/2)
run_name = "transformer_minor"
error_file_name = 'errorlog_transformer.txt'

loss_fn = torch.nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")