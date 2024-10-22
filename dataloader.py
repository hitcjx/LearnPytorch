'''在pytorch官网的Docs里面版本换成1.0.0搜索dataloader获得相关信息'''
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset",train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
# batch_size是每次抽取样本数，shuffe是每次洗牌，默认false（不洗牌）,
test_loader = DataLoader(dataset=test_data,batch_size=4,shuffle=True,num_workers=0)
# batch_size=4,则Dataloader每次取的都是4个img和4个target的集合(随机抽取)

writer = SummaryWriter('logs')
step = 0
for data in test_loader:
    imgs,targets = data
    #print(imgs,imgs.shape,type(imgs))
    #注意此处为add_images!!!有s！！！
    print(imgs.shape)
    writer.add_images('dataloader',imgs,step)
    step += 1
writer.close()