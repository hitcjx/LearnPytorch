''''''
'''调试技巧:zz->受保护的的特性->modules->model->受保护的特性->modules'''
'''环境配置:conda下想要用gpu运行代码需要conda安装cudatoolkit+cudnn'''
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear, Sequential
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

dataset = torchvision.datasets.CIFAR10('./data', train=False,
                                       transform=ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)

class Zhuzhu(nn.Module):
    def __init__(self):
        super(Zhuzhu, self).__init__()

        '''根据设计的模型得知下一步的out_channels,再计算stride和padding'''
        '''Linear的1024不会算？先跑前面的代码得到结果再继续'''
        self.model = Sequential(
            Conv2d(3, 32, 5,
                   1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5,
               1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5,
               1, padding=2),
            MaxPool2d(2),
            nn.Flatten(),
            Linear(1024,64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x

loss = nn.CrossEntropyLoss()
zz = Zhuzhu()

'''将模型移到GPU'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
zz.to(device)

'''函数的括号内按ctrl+p可以看需要什么参数'''
'''学习率为0.01'''
'''zz.parameters（）表示获取zz中所有可训练参数'''
optim = torch.optim.SGD(zz.parameters(), lr=0.002)

for epoch in range(20):
    running_loss = 0.0
    '''以下的for循环也只是把所有data过了一边，即一轮学习，我们应该有多轮学习'''
    for data in dataloader:
        '''下面的过程：首先通过loss函数算出损失，然后将grad中储存的梯度值清零(注)
        用backward计算每个参数的梯度值，使用优化器对参数进行优化
        注：pytorch为了提高效率将每次计算的梯度累加起来，故需要先清零'''
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)  # 将数据移动到 GPU
        outputs = zz(imgs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(running_loss)