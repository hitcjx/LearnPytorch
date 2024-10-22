import torch
import torchvision.datasets
from torch import nn
from torch.ao.nn.quantized import Sigmoid
from torch.nn import ReLU,Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

'''本代码不能同时实现两个函数，需要修改forward函数'''

'''ReLU:线性修正单元，f(x) = max(0, x)'''
'''Sigmoid:乙状结肠函数'''
'''非线性激活的作用：
引入非线性: 非线性激活函数将线性模型的输出转换为非线性输出，从而使网络能够学习更复杂的函数关系。
提高模型表达能力: 通过引入非线性，神经网络能够学习更复杂的函数，从而提高模型的表达能力，更好地拟合数据。
模拟生物神经元: 非线性激活函数类似于生物神经元的激活机制，它们只在输入达到一定阈值时才会激活。'''

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))

dataset = torchvision.datasets.CIFAR10('./dataset',train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Zhuzhu(nn.Module):
    def __init__(self):
        super(Zhuzhu, self).__init__()
        self.relu1 = ReLU()
        self.sigmoid1 = Sigmoid()

    def forward(self,input):
        output = self.sigmoid1(input)
        return output

zz = Zhuzhu()

output = zz(input)
print(output)

writer = SummaryWriter('logs')
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, global_step=step)
    output1 = zz(imgs)
    writer.add_images('output', output1, step)
    step += 1

writer.close()