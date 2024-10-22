import torch
import torchvision

#方式1->保存方式1,加载模型
model = torch.load("vgg16_method1.pth")
print(model)

#方式2->保存方式2,加载模型
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)

#陷阱
model = torch.load("zz_method1.pth")
#这样会报错，需要把zz = ZZ()注释掉或