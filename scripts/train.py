import torch
from torch import optim

from config import device, k_value, BATCH_SIZE
from utils.eval_acc_staged_8_classes import eval_acc_stage


def train(train_dataloader, model, loss_fn, writer, epoch_ID, learning_rate=5e-4, use_cuda=True):

    total_train_step = 1
    total_loss = 0
    total_topK = 0

    if epoch_ID < 5:
        learning_rate = 2e-4

    else:
        learning_rate = 1e-4

    model.to(device)
    model.train()

    local_step = 0
    for data in train_dataloader:
        embed, atten, contact_label, L = data

        if use_cuda:
            contact_label = contact_label.to(device)
            embed = embed.to(device)
            atten = atten.to(device)

        pred = model(embed, atten)
        contact_label = contact_label.reshape(-1)

        loss = loss_fn(pred, contact_label)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if torch.isinf(loss) or torch.isnan(loss):
            loss = torch.tensor(1, dtype=torch.float32, device=pred.device, requires_grad=True)
            print("Attentinon: loss!!!!")

        loss.backward()

        local_step += 1
        if local_step == 5:
            optimizer.step()
            optimizer.zero_grad()
            local_step = 0

        total_train_step = total_train_step + 1

        l = loss.item()
        total_loss = total_loss + l

        topK, _, _ = eval_acc_stage(L, pred, contact_label,BATCH_SIZE)
        total_topK = total_topK + topK

        avg_train_loss = total_loss / total_train_step
        avg_topK = total_topK / total_train_step


        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("avg_train_loss", avg_train_loss, total_train_step)
            writer.add_scalar("avg_topK", avg_topK, total_train_step)


    return avg_train_loss, avg_topK