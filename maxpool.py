import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('./dataset',train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset=dataset,batch_size=64)

class Zhuzhu(nn.Module):
    def __init__(self):
        super(Zhuzhu, self).__init__()
        '''最大池化即选取kernel内最大值,ceil_mode即是否增加边缘'''
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

zz = Zhuzhu()

writer = SummaryWriter('logs')
step = 0

'''千万要注意是 in dataloader 而不是 in dataset'''
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    output = zz(imgs)
    writer.add_images('output', output, step)
    step += 1

writer.close()