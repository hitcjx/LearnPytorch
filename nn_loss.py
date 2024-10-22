import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs,(1, 1, 1, 3))
targets = torch.reshape(targets,(1, 1, 1, 3))

'''可以用reduction='sum'设置计算方式，和（sum），平均值（mean）'''
'''L1Loss的inputs和targets中各数字直接相减，再累加或求平均,再,只接受浮点型'''
loss_l1 = L1Loss(reduction='sum')
result = loss_l1(inputs, targets)

'''MSELoss求的是方差的和或者平均数'''
loss_mse = nn.MSELoss()
result_mse = loss_mse(inputs, targets)
print(result,result_mse)

'''CrossEntropyLoss公式：-X[target]+ln(exp(X[0])+exp(X[1])+...)
   X为输入的数组，target为数组中的一个序号对应的东西
   如：分类，人0,狗1,猫2,input为三者的概率组成的数组，则有了一个有序号的数组'''
x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = nn.CrossEntropyLoss()
result_cross = loss_cross(x, y)
print(result_cross)
