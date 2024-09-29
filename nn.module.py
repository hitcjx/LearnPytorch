import torch
from torch import nn

# 试着搭建一个神经网络
class Zhuzhu(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input+1
        return output

zz = Zhuzhu()
x = torch.tensor(1.0)
output = zz(x)
print(output)
