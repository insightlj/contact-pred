import torch
from torch.nn.functional import softmax
from config import k_value, device


def topK(k, pred, contact_label):
    # pred: [batchl*l*l, dim_out]
    k = k.to(device)

    pred = softmax(pred, dim=1)
    pred_class_1 = pred[:,:2].sum(dim=1)   # 选取contact在8以内的部分

    pred_descending, index = torch.sort(pred_class_1, descending=True)
    pred_topK = pred_descending[:k]
    index_topK = index[:k]

    pred_topK[pred_topK >= 0.5] = 0
    pred_topK[pred_topK < 0.5] = 1

    contact_label[contact_label<2] = 1
    contact_label[contact_label>=2] = 0
    contact_label_topK = contact_label[index_topK]
    accuracy = 1 - (abs(pred_topK - contact_label_topK).sum() / k)

    return accuracy


def eval_acc_stage(k_value, pred, contact_label, batch_size):
    """分别评价long，medium，short的准确率；并且只评价其中一个三角的准确率"""
    # pred: [batch*L*L, dim_out]
    # batch_size = 1

    len = pred.shape[0]
    dim_out = pred.shape[1]
    L = int((len / batch_size) ** 0.5)

    pred = pred.reshape(L, L, dim_out)
    pred = pred.permute(2,0,1)
    contact_label = contact_label.reshape(L, L)


    if L >= 24:
        pred_long = torch.triu(pred, 24)
        pred_medium = torch.triu(pred, 12) - torch.triu(pred, 24)
        pred_short = torch.triu(pred, 6) - torch.triu(pred, 12)

        pred_long = pred_long.permute(1,2,0)
        pred_medium = pred_medium.permute(1,2,0)
        pred_short = pred_short.permute(1,2,0)

        contact_label_long = torch.triu(contact_label, 24)  # x>=24
        contact_label_medium = torch.triu(contact_label, 12) - torch.triu(contact_label, 24)  # 12<=x<24
        contact_label_short = torch.triu(contact_label, 6) - torch.triu(contact_label, 12)  # 6<=x<12

    else:
        raise ValueError("蛋白质的长度小于24")

    topK_long = topK(k_value, pred_long.reshape(-1, 8), contact_label_long.reshape(-1))
    topK_medium = topK(k_value, pred_medium.reshape(-1, 8), contact_label_medium.reshape(-1))
    topK_short = topK(k_value, pred_short.reshape(-1, 8), contact_label_short.reshape(-1))

    return topK_long.item(), topK_medium.item(), topK_short.item()


if __name__ == "__main__":
    pred = torch.load("pred.pt")
    contact_label = torch.load("contact_label.pt")

    t_long, t_medium, t_short =  eval_acc_stage(k_value, pred=pred, contact_label=contact_label, batch_size=1)
    print("topK_long:{}\ntopK_medium:{}\ntopK_short:{}".format(t_long, t_medium, t_short))
