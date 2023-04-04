import torch
from torch.nn.functional import softmax

# 直接取值最大的k个位点
def eval_topK(k, pred, contact_label):
    # pred: [batchl*l*l, dim_out]
    # pred_class_1: [batch*l*l]
    # only for binary classification
    pred_softmax = softmax(pred, dim=1)
    pred_class_1 = pred[:, 1]

    pred_descending, index = torch.sort(pred_class_1, descending=True)
    # pred_topK = pred_descending[:k]
    index_topK = index[:k]

    pred_softmax = pred_softmax[:, 1][index_topK]
    pred_softmax[pred_softmax>=0.5] = 1
    pred_softmax[pred_softmax<0.5] = 0

    contact_label_topK = contact_label[index_topK]
    topK = abs(pred_softmax - contact_label_topK).sum() / len(contact_label_topK)

    return topK


# def eval_topK(k, pred, contact_label):
#     # pred: [batchl*l*l, dim_out]
#     # pred_class_1: [batch*l*l]
#     # only for binary classification
#     # pred_softmax = softmax(pred, dim=1)
#     pred_class_1 = pred[:, 1]
#
#     pred_descending, index = torch.sort(pred_class_1, descending=True)
#     # pred_topK = pred_descending[:k]
#     index_topK = index[:k]
#
#     contact_label_topK = contact_label[index_topK]
#     topK = contact_label_topK.sum() /  k
#
#     return topK


# 先softmax，再取值最大的k个位点。只不过这样的话，排在前面的几乎都是1（我觉得这种方式更合理一些。
# 因为我在预测的过程中并没有对预测结果的值进行限定，而尽限定了两个值之间的关系[与contact_label之间的交叉熵]）
def eval_topK(k, pred, contact_label):
    # pred: [batchl*l*l, dim_out]
    # pred_class_1: [batch*l*l]
    # only for binary classification
    pred = softmax(pred, dim=1)
    pred_class_1 = pred[:, 1]

    pred_descending, index = torch.sort(pred_class_1, descending=True)
    # pred_topK = pred_descending[:k]
    index_topK = index[:k]

    contact_label_topK = contact_label[index_topK]
    topK = contact_label_topK.sum() /  k

    return topK


if __name__ == "__main__":
    # pred = [0.98,0.3, 0.88,0.6,0.67,0.5,0.61, 0.52,0.4,0.2,-0.9,0.1,0.7,0.72,0.63,0.49,0.3,0.4,0.5,0.6,0.7,0.2,0.66,
    #         0.1,0.2,0.3,0.5,0.8,-0.2,0.7,-0.05,0.7]
    # pred = torch.tensor(pred).reshape(16, 2)
    #
    # contact = [0,1,0,0,0,0,1,0,1,1,0,0,0,0,0,1]
    # contact = torch.tensor(contact)

    pred = torch.load("pred.pt")
    contact = torch.load("contact_label.pt")
    topk = eval_topK(100, pred, contact)