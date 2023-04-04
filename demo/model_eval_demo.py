import torch
from main import test_dataset
from torch.utils.data import DataLoader


from config import BATCH_SIZE, device
from utils.vis_contact_label_8_classes import vis_contact
from utils.eval_acc_staged_8_classes import eval_acc_stage

DEMO_SIZE = 10
net_pt_name = "model/checkpoint/8_classes_I/epoch9.pt"

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
net_pt = torch.load(net_pt_name, map_location=device)

i = 0
total_topL_long = 0
for data in test_dataloader:
    i += 1
    if i > DEMO_SIZE:
        break
    embed, atten, contact_label, L = data
    embed = embed.to(device)
    atten = atten.to(device)
    contact_label = contact_label.reshape(-1)
    contact_label = contact_label.to(device)
    L = L.to(device)

    pred = net_pt(embed, atten)

    topL_long, topL_medium, topL_short = eval_acc_stage(L, pred, contact_label, BATCH_SIZE)
    total_topL_long += topL_long

    vis_contact(pred, str(i)+"_pred", True)
    vis_contact(contact_label, str(i)+"_contact_label", False)

print("avg_topL_long:{}".format(total_topL_long/DEMO_SIZE))