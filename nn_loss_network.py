'''手搓cifar10 model'''
import torch
import torchvision.datasets
from torch import nn
from torch.nn import Linear, Sequential
from torch.nn import Conv2d, MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

from nn_loss import targets

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
for data in dataloader:
    imgs, targets = data
    outputs = zz(imgs)
    result_loss = loss(outputs, targets)
    result_loss.backward()
    print('ok')