import torch
from torch.nn.functional import softmax
from config import k


def eval_acc(pred, contact_label, class_num, error_tolerant, sum):
    pred = pred.reshape((-1))
    contact_label = contact_label.reshape((-1))

    if class_num == 2:
        accuracy = 1 - (abs(pred - contact_label).sum() / sum)

    else:
        diff = abs(pred - contact_label)
        diff[diff <= error_tolerant] = 0
        diff[diff > error_tolerant] = 1
        accuracy = 1 - (diff.sum() / sum)

    return accuracy


def topK(k, pred_class_1, contact_label):
    # pred: [batchl*l*l, dim_out]
    # pred_class_1: [batch*l*l]
    # only for binary classification

    pred_descending, index = torch.sort(pred_class_1, descending=True)
    pred_topK = pred_descending[:k]
    index_topK = index[:k]

    pred_topK[pred_topK >= 0.5] = 1
    pred_topK[pred_topK < 0.5] = 0

    contact_label_topK = contact_label[index_topK]
    accuracy = 1 - (abs(pred_topK - contact_label_topK).sum() / k)

    return accuracy


def eval_acc_stage(k, pred, contact_label, batch_size):
    """分别评价long，medium，short的准确率；并且只评价其中一个三角的准确率"""
    # pred: [batch*L*L, dim_out]

    pred = softmax(pred, dim=1)
    pred = pred[:, 1]  # pred_class_1

    # class_num = pred.shape[1]  # setting for multi-classes
    len = pred.shape[0]
    L = int((len / batch_size) ** 0.5)

    pred = pred.reshape(batch_size, L, L)
    contact_label = contact_label.reshape(batch_size, L, L)

    if L >= 24:
        pred_long = torch.triu(pred, 24)
        pred_medium = torch.triu(pred, 12) - torch.triu(pred, 24)
        pred_short = torch.triu(pred, 6) - torch.triu(pred, 12)

        contact_label_long = torch.triu(contact_label, 24)  # x>=24
        contact_label_medium = torch.triu(contact_label, 12) - torch.triu(contact_label, 24)  # 12<=x<24
        contact_label_short = torch.triu(contact_label, 6) - torch.triu(contact_label, 12)  # 6<=x<12

    else:
        raise ValueError("蛋白质的长度小于24")

    topK_long = topK(k, pred_long.reshape(-1), contact_label_long.reshape(-1))
    topK_medium = topK(k, pred_medium.reshape(-1), contact_label_medium.reshape(-1))
    topK_short = topK(k, pred_short.reshape(-1), contact_label_short.reshape(-1))

    return topK_long.item(), topK_medium.item(), topK_short.item()


if __name__ == "__main__":
    pred = torch.load("pred.pt")
    contact_label = torch.load("contact_label.pt")

    t_long, t_medium, t_short =  eval_acc_stage(k, pred=pred, contact_label=contact_label, batch_size=1)
    print("topK_long:{}\ntopK_medium:{}\ntopK_short:{}".format(t_long, t_medium, t_short))
