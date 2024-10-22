import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader

from dataloader import targets

dataset = torchvision.datasets.CIFAR10('./dataset',train=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset,batch_size=64)

class Zhuzhu(nn.Module):
    def __init__(self):
        super(Zhuzhu,self).__init__()
        '''一般来说，输入的图像是rgb，则通道数为3'''
        self.conv1 = Conv2d(in_channels=3,out_channels=6,kernel_size=3,
                            stride=1,padding=0)
    # 注：forward为魔法函数
    def forward(self,x):
        x = self.conv1(x)
        return x

zz = Zhuzhu()

for data in dataloader:
    imgs, targets = data
    output = zz(imgs)
    print(output.shape)
