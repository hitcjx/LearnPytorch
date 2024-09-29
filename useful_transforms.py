'''复习python知识：
   class C:
   def __f__(self,a):
        operation

   类似上面这样的函数，在我们
   c = C(a)时就会自动调用这个函数了
'''

from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

'''可以ctrl+单击transforms，再点击，层层点击查看其函数，以下介绍其中几个
   请重点关注输入输出及格式，然后关注需要的参数，一般写的很详细，比网上查好'''

img_path = '/home/xuan/PycharmProjects/LearnPytorch/dataset/train/ants/6240329_72c01e663e.jpg'
img = Image.open(img_path)
writer = SummaryWriter('logs')

# ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
'''此时不用额外设置dataformats了'''
writer.add_image('t',img_tensor,1)

# Normalize
'''两个参数为平均值mean和标准差std
   公式：output = (input-mean)/std
   这是将值化到（-1,1）之间'''
trans_norm = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm = trans_norm(img_tensor)
writer.add_image('n',img_norm,2)

# Resize
'''值得注意的是Resize的输入，输出均为PIL类型'''
'''我发现这个函数修改尺寸并不是裁剪，而是等比例压缩了'''
'''仅输入一个数也可以，则会将图片的短边匹配到那个数'''
trans_resize = transforms.Resize((256,256))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
writer.add_image('r',img_resize,3)

# Compose-Resize
trans_resize2 = transforms.Resize((256,256))
trans_compose = transforms.Compose([trans_resize2,trans_tensor])
img_compose = trans_compose(img)
writer.add_image('c',img_compose,4)

# RandomCrop
'''输入输出仍然是PIL'''
'''输入一个数则裁剪出正方形'''
trans_random = transforms.RandomCrop(30)
trans_compose2 = transforms.Compose([trans_random,trans_tensor])
for i in range(10):
    img_random = trans_compose2(img)
    writer.add_image('randomcrop',img_random,i)

writer.close()