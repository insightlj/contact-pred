import torch
from utils.eval_topK import eval_topK
from utils.eval_acc_staged_8_classes import eval_acc_stage
from config import k_value,device,BATCH_SIZE

def test(test_dataloader, model, k_value, writer=None):
    total_test_step = 1
    total_test_loss = 0
    total_topK = 0

    test_data_size = len(test_dataloader)

    model.eval()
    l = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for data in test_dataloader:
            embed, atten, contact_label, L = data
            embed = embed.to(device)
            atten = atten.to(device)
            contact_label = contact_label.to(device)

            contact_label = contact_label.reshape(-1)  # 将contact_label的batch融合
            pred = model(embed, atten)
            loss = l(pred, contact_label)

            ##########TEMP_CODE############
            # torch.save(pred, "pred.pt")
            # torch.save(contact_label, "contact_label.pt")
            ##############################

            topK, _, _ = eval_acc_stage(L, pred, contact_label, batch_size=BATCH_SIZE)
            # print(topK)

            total_test_loss = total_test_loss + loss.item()
            total_topK = total_topK + topK

            if writer:
                writer.add_scalar("avg_loss", total_test_loss / total_test_step, total_test_step)
                writer.add_scalar("avg_topK_long", total_topK / total_test_step, total_test_step)

            total_test_step += 1

    avg_test_loss = total_test_loss / test_data_size
    avg_topK_long = total_topK / test_data_size

    return avg_test_loss, avg_topK_long


if __name__ == "__main__":

    from torch.utils.tensorboard import SummaryWriter
    from config import device, BATCH_SIZE
    from main import test_dataloader
    from utils.eval_acc_staged_8_classes import eval_acc_stage

    import os

    pt_dir = "model/checkpoint/0214_8_classes_1158/"
    logs_folder_name = "0214_8_classes_1158"
    pt_list = os.listdir(pt_dir)
    num_pt = len(pt_list)

    logs_name_sum = "logs/test_" + logs_folder_name + "/" + "summary"
    writer_sum = SummaryWriter(logs_name_sum)

    for ID in range(num_pt):
        print("开始验证模型{}……".format(ID))

        pt_name = "epoch" + str(ID) + ".pt"
        net = torch.load(pt_dir + pt_name)
        net.to(device)

        logs_dir = "logs/test_" + logs_folder_name + "/" + "epoch" + str(ID)
        writer = SummaryWriter(logs_dir)

        # test
        avg_test_loss, avg_topK_long = test(test_dataloader, net, k_value, writer=writer)

        logs_name = "logs/test_" + logs_folder_name + "/" + "epoch" + str(ID)
        writer_test = SummaryWriter(logs_name)
        writer_sum.add_scalar("avg_test_loss", avg_test_loss, int(ID))
        writer_sum.add_scalar("avg_topK_long", avg_topK_long, int(ID))
