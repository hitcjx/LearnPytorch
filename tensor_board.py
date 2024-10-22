'''tip:对类，函数右键，再点击“查找用法”可以查看其参数等信息
   或者将鼠标放在上面不点也行'''
'''tip:查看文件类型：print(type(img))'''
'''tip:有时候函数对输入的图像格式及shape有要求，请查看其信息进行调整
   print(img.shape)查看shape'''

from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('logs')
'''注：如果有时候有奇怪情况，可以考虑把logs里面的文件清除再重新搞'''
'''
for i in range(100):
    writer.add_scalar('y=x',i,i)
''''''接下来用 tensorboard --logdir='logs' 
   注：logs就是跑代码生产的文件夹名'''

img_path = '/home/xuan/PycharmProjects/LearnPytorch/dataset/train/ants/6240329_72c01e663e.jpg'
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
'''我们可以从'1'开始，不断add_iamge'''
writer.add_image('test',img_array,3,dataformats='HWC')

writer.close()