import gzip
import os
import torch
from torch import nn
from torch.utils import data
import numpy as np
import pandas as pd

class MyDataset(data.Dataset):
    def __init__(self,path,nodes,center=False,one_hot=False):
        super().__init__()
        self.nodes = nodes
        self.center = center
        self.one_hot = one_hot
        self.data = pd.read_csv(path)
        self.inputs = self.data.drop(columns=['target']).values
        self.targets = self.data[['target']].values
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.name = path.split('/')[-1].split('.')[0]
        if self.one_hot:
        # 将target转换为one-hot编码
            self.targets = torch.tensor(pd.get_dummies(self.targets.squeeze()).values, dtype=torch.float32)
        else:
            self.targets.reshape(-1,1)
        self.data = None
        self.c = self.targets.shape[1] # 输出的维度
        self.d = self.inputs.shape[1] 

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        return x, y

    def get_nodes_dataset_one_column(self,index):
        # 数据集的标签用一列表示，名称为target可以使用这个函数进行获取
        num = len(self.inputs) // self.nodes
        node_input = self.inputs[num * index:num * (index + 1)]
        node_target = self.targets[num * index:num * (index + 1)]
        if not self.one_hot:
            node_target = node_target.reshape(-1,1)

        return NodeDataSet(node_input,node_target,self.name,self.one_hot)
    
    def __len__(self):
        return len(self.inputs)
    
class NodeDataSet(data.Dataset):
    def __init__(self,inputs,targets,name,one_hot):
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.name = name
        self.one_hot = one_hot
        if len(self.targets.shape) <= 1:
            self.c = 1
        else:
            self.c = self.targets.shape[1] # 输出的维度
        if len(self.inputs.shape) <= 1:
            self.d = 1
        else:
            self.d = self.inputs.shape[1]
    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        return x, y
    def __len__(self):
        return len(self.inputs)
    

class NodeDataSet_image(NodeDataSet):
    def __init__(self,inputs,targets,image,name,one_hot):
        super().__init__(inputs,targets,name,one_hot)
        self.image = image



class A9A(MyDataset):
    def __init__(self,nodes, center=False,one_hot=False):
        self.root = r'./data/A9A/a9a.csv'
        super().__init__(self.root,nodes, center,one_hot)



class BreastCancer(MyDataset):
    def __init__(self,nodes, center=False,one_hot=False):
        self.root = r'./data/breast_cancer/breast_cancer.csv'
        super().__init__(self.root,nodes, center,one_hot)


    
class fashion_mnist(MyDataset):
    def __init__(self,nodes, center=False,one_hot=False):
        self.root = r'./data/fashion_mnist/fashion_mnist.csv'
        super().__init__(self.root,nodes, center,one_hot)
    
class fashion_mnist_blurring(MyDataset):
    def __init__(self,nodes,image_index,center=False,one_hot=False):
        self.root = r'./data/fashion_mnist/fashion_mnist_10.csv'
        super().__init__(self.root,nodes,center,one_hot)
        # 产生nodes个模糊核和模糊图像
        self.H = []
        self.X = []
        self.Y = []
        self.noise_list = []
        self.HX_list = []
        # 对数据进行模糊处理
        self.image = self.inputs[image_index].reshape(28,28)
        
        for i in range(nodes):
            # length = np.random.randint(1, 2)
            # angle = np.random.randint(0,90)
            noise = np.random.normal(0, 1/255.0, (28, 28))
            self.H.append(torch.tensor(self._motion_blur_kernel(1, 5),dtype=torch.float32).reshape(1,-1))
            # Y = HX + noise,其中HX是卷积操作,X是获取到的图像,并且结果与原图像形状一致
            HX = torch.nn.functional.conv2d(self.image.reshape(1,1,28,28),self.H[i].reshape(1,1,1,-1),padding='same')
            y = (HX).reshape(1,-1)
            self.Y.append(y+noise.reshape(1,-1))
            # self.Y.append(y)
            self.noise_list.append(noise)
            self.HX_list.append(HX)
        self.inputs = torch.cat(self.H)
        self.targets = torch.cat(self.Y)

    def get_nodes_dataset_one_column(self, index):
        return NodeDataSet_image(self.inputs[index].reshape(1,-1),self.targets[index].reshape(1,-1),self.image,self.name,self.one_hot)

    def _motion_blur_kernel(self,length, angle):
        """生成运动模糊核矩阵。
        
        length: 模糊的长度。
        angle: 模糊的角度，以度为单位。
        """
        kernel_size = int(length * 2) + 1
        kernel = np.zeros((kernel_size, kernel_size))
        kernel_center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                # 计算相对于中心点的位置
                x = i - kernel_center
                y = j - kernel_center
                # 计算沿运动方向的距离
                distance = abs(x * np.cos(np.radians(angle)) + y * np.sin(np.radians(angle)))
                if distance < length:
                    kernel[i, j] = 1
        kernel /= kernel.sum()  # 归一化
        return kernel
    
class covtype(MyDataset):
    def __init__(self,nodes, center=False,one_hot=False):
        self.root = r'./data/covtype/covtype.csv'
        super().__init__(self.root,nodes, center,one_hot)
    
 
