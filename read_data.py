'''tip:在python控制台里面跑代码可以看到变量'''
'''本代码只是标签为文件名的数据集的方法，还有其他类型的数据集要灵活变通'''
from torch.utils.data import Dataset
import os
from PIL import Image

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        '''拼接'''
        self.path = os.path.join(self.root_dir,self.label_dir)
        '''把文件夹里面的文件排成列表'''
        self.img_path = os.listdir(self.path)

    def __getitem__(self,idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.img_path)

root_dir = 'dataset/train'
ants_label_dir = 'ants'
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)
'''将二者合并'''
train_dataset = ants_dataset + bees_dataset

img,label = ants_dataset[0]
img.show()
print(len(ants_dataset),len(bees_dataset),len(train_dataset))

