import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10('./dataset',train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class Zhuzhu(nn.Module):
    def __init__(self):
        super(Zhuzhu, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self, input):
        output = self.linear1(input)
        return output

zz = Zhuzhu()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    '''data是NCHW,reshape之后就变成批次为1,通道数为1,高度为1,宽度为像素数的一维向量
       线性层只能处理一维向量'''
    '''下面一行等效于output = torch.reshape(imgs, (1,1,1,-1))'''
    output = torch.flatten(imgs)
    print(output.shape)
    output = zz(output)
    print(output.shape)