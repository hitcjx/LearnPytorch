'''tip:在pytorch官方文档torch.nn中的convolution部分'''
# 注：此处使用的是functionnal，实际上我们并不使用，只是用以学习
import torch
import torch.nn.functional as F

input = torch.tensor([[1,2,0,3,1],
                      [0,1,2,3,1],
                      [1,2,1,0,0],
                      [5,2,3,1,1],
                      [2,1,0,1,1]])

kernel = torch.tensor([[1,2,1],
                       [0,1,0],
                       [2,1,0]])

# 查阅文档之后发现尺寸不符合输入要求，则使用reshape
# batch_size为1,通道数为1
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))

# 卷积,stride为卷积核移动步长
output = F.conv2d(input,kernel,stride=1)
print(output)
output2 = F.conv2d(input,kernel,stride=2)
print(output2)
# padding即填充宽度，填充部分的值默认为0
# 填充使得边缘特征得到更好的利用
output3 = F.conv2d(input,kernel,stride=1,padding=1)
print(output3)