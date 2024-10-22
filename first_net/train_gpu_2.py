import torch
from torch.utils.tensorboard import SummaryWriter

from model import *

# 定义训练的设备
device = torch.device("cuda:0")
'''还是神经网络，数据，损失函数，加上.to(device)'''
'''当电脑进入挂起状态再打开后用gpu跑代码可能会报错，这时重启电脑即可'''
'''环境配置:conda下想要用gpu运行代码需要conda安装cudatoolkit+cudnn'''

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
zz.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

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
        imgs = imgs.to(device)
        targets = targets.to(device)
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
            imgs = imgs.to(device)
            targets = targets.to(device)
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