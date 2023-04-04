from main import train_dataset
total_L = 0
for _, __, ___, l in train_dataset:
    total_L += l

avg_L = total_L / len(train_dataset)
print(avg_L)



# 测试集的平均蛋白质长度为158
from main import test_dataset

total_L = 0
for a, b, c, l in test_dataset:
    total_L += l

avg_L = total_L / len(test_dataset)
print(avg_L)
