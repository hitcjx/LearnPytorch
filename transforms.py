from PIL import Image
from torchvision import transforms

img_path = '/home/xuan/PycharmProjects/LearnPytorch/dataset/train/ants/6240329_72c01e663e.jpg'
img = Image.open(img_path)

'将图片转化成张量，第一行代码是一个实例化（？）的过程'
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)
print(tensor_img)