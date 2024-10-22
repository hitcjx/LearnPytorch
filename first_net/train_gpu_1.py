''''''
'''只需在网络模型（zz），数据（输入，标注），损失函数加上.cuda()即可'''
import torch
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
import torchvision
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root='./dataset', train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root='./dataset', train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
#"{}".format()  格式化输出
print('训练数据集的长度：{}'.format(train_data_size))
print('测试数据集的长度：{}'.format(test_data_size))

# 利用 Dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
zz = Zhuzhu()
if torch.cuda.is_available():
    zz = zz.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
#1e-2 = 0.01
optimizer = torch.optim.SGD(zz.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter('./logs')

for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))

    # 训练开始
    zz.train()   # 好习惯，加上就行
    for data in train_dataloader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = zz(imgs)
        loss = loss_fn(outputs, targets)

        #优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_test_step)

    # 测试步骤开始
    zz.eval()  # 好习惯，加上就行
    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            outputs = zz(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()

    print("整体数据集上的Loss：{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(zz, "zz_{}.pth".format(i))
    # torch.save(zz.state_dict(), "zz_{}.pth".format(i))
    print("模型已保存")

writer.close()