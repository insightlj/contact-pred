import torch

def dist(K):
    # 输入为coor_CB (L,3)

    T = K.t()

    K_2 = torch.pow(K, 2).sum(dim=1).unsqueeze(1)
    T_2 = K_2.t()

    dist_2 = K_2 + T_2 - 2 * K @ T
    dist_2[dist_2 < 0] = 0
    dist = pow(dist_2, 0.5)
    return dist


def label_from_dist(P):
    # 输入为距离矩阵(L,L), 输出为label(L,L)

    P = P // 4
    P[P > 7] = 7

    return P


def caculator_label(coor, a, train_mode):
    # 输入为原始坐标数据(L,4,3), 输出为基于CB计算的label(L,L)

    coor_CB = coor[:, 3, :]
    label = label_from_dist(dist(coor_CB))

    L = label.shape[0]

    if train_mode and L > 192:
        label = label[a:a + 192, a:a + 192]

    label = torch.LongTensor(label.numpy())
    label = label.reshape(-1)
    return label

