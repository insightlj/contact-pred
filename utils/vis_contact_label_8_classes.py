import pandas as pd
import seaborn as sns
import torch

import matplotlib.pyplot as plt
from torch.nn.functional import softmax

def vis_contact(label, filename, is_pred=False):
    # batch = 1
    # label [batch*L*L, dim_out]
    L_pow_2 = label.shape[0]
    L = int(L_pow_2 ** 0.5)

    if is_pred:
        label = softmax(label, dim=1)
        _, label = torch.sort(label, dim=1, descending=True)
        label = label[:,0]
        label = label.reshape(L, L)

    else:
        label = label.reshape(L,L)

    label = (label + label.t()) / 2
    label = label.cpu()
    df = pd.DataFrame(label.numpy())
    sns.heatmap(df, cmap="YlGnBu", cbar=False)
    plt.savefig("plot/"+filename)

if __name__== "__main__":
    pred = torch.load("pred.pt")
    contact_label = torch.load("contact_label.pt")

    vis_contact(pred, "pred.png", is_pred=True)
    vis_contact(contact_label, "contact_label.png")

