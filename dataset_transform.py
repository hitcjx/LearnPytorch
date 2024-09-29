'''tip:torch官网有数据集的集合(Docs),将版本换成1.0.0
   包括数据集信息和调用方法，参数介绍'''
import torchvision.datasets
from examples.dataset.write_dataset_encrypted import dataset

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 训练数据集
train_set = torchvision.datasets.CIFAR10(root='./dataset',train=True,
                                         transform=dataset_transform,download=True)
# 测试数据集
test_set = torchvision.datasets.CIFAR10(root='./dataset',train=False,
                                        transform=dataset_transform,download=True)

print(test_set[0])
# 得到输出结果之后有未知成分可以用debug来看一下是什么成分