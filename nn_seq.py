'''手搓cifar10 model'''
import torch
from torch import nn
from torch.nn import Linear, Sequential
from torch.nn import Conv2d, MaxPool2d
from torch.utils.tensorboard import SummaryWriter


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

zz = Zhuzhu()
print(zz)
'''可以用已知输入的形状来查看模型是否正确'''
input = torch.ones(64, 3, 32, 32)
output = zz(input)
print(output.shape)

'''查看网络结构'''
writer = SummaryWriter('logs')
writer.add_graph(zz, input)
writer.close()