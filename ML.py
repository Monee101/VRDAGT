import sys
import time
import numpy as np
import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import Subset
import os
from data import A9A,BreastCancer,covtype
import torch.nn as nn
from models.MyModel import ModelEnum
from algorithm.alg import *
from utils.tools import *
import scipy.io as scio
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

arg = arg_praser()

nodes = arg.nodes # 节点数 

one_hot = arg.onehot # 是否one hot编码

dataset_name = arg.dataset # 数据集名称
Model_name = arg.model # 模型名称
if dataset_name == 'covtype':
    covtype_data = covtype(nodes,center=False,one_hot=one_hot)
    datasets = [covtype_data.get_nodes_dataset_one_column(i) for i in range(nodes)]
elif dataset_name == 'A9A':
    a9a_data = A9A(nodes,center=False,one_hot=one_hot)
    datasets = [a9a_data.get_nodes_dataset_one_column(i) for i in range(nodes)]

if Model_name == 'simple_mlp':
    eum_model = ModelEnum.simple_mlp
elif Model_name == 'log':
    eum_model = ModelEnum.logistic_regression

train_dataset_nodes = []
test_dataset_nodes = []
train_size = int(0.8 * len(datasets[0]))
test_size = len(datasets[0]) - train_size

batch_size = arg.batchsize
it_num = train_size//batch_size
print('it_num:',it_num)

for dataset in datasets:
    # 不随机划分训练集和测试集
    train_dataset, test_dataset = Subset(dataset, list(range(train_size))),Subset(dataset, list(range(train_size, train_size+test_size)))
    train_dataset_nodes.append(train_dataset)
    test_dataset_nodes.append(test_dataset)

# 创建log文件夹
dir_path = r'./log/对比试验'+time.strftime(" %Y年%m月%d日%H时%M分%S秒", time.localtime())
file_path = dir_path + r'/log.txt'
os.makedirs(dir_path)


A,B = generate_adjacency_matrix(nodes, kind='rc')
A = torch.tensor(A,dtype=torch.float32)
B = torch.tensor(B,dtype=torch.float32) 
print('A:',A)
print('B:',B)
# C_path =  r'D:\my_project\GT_VR_Simulation\a9a\data\new_C.mat'
# C =  scio.loadmat(C_path)['C_store']
# C = torch.tensor(C,dtype=torch.float32)
GT_DAG_config = {'nodes':nodes,'Phi':None,'theta':None,'opt_x':None,'beta1':0.9,'beta2':0.999,'train_dataset_nodes':train_dataset_nodes,
                'train_dataset_nodes':train_dataset_nodes,'test_dataset_nodes':test_dataset_nodes,'alpha':arg.alpha,'v_min':10**-8,'v_max':100,
                'T':arg.T,'batch_size':batch_size,'model':eum_model,'A':A,'B':B,'dir_path':dir_path,'file_path':file_path,'C':None}

GT_DAG_config['P'] = 0.1

DAGT_Instance = DAGT(GT_DAG_config)
VRDAGT_Instance = VRDAGT(GT_DAG_config)
GT_DAG_config['alpha'] = 0.1
GT_VR_Instance = GT_VR(GT_DAG_config)
AB_SAGA_Instance = AB_SAGA(GT_DAG_config)

VRDAGT_Instance.run()
GT_VR_Instance.run()
DAGT_Instance.run()
AB_SAGA_Instance.run()


Model_GT_DAG = DAGT_Instance.model_avg
Model_AB_SAGA = AB_SAGA_Instance.model_avg
Model_GT_DAG_VR = VRDAGT_Instance.model_avg
Model_GT_VR = GT_VR_Instance.model_avg
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rc('text', usetex=True)
# 绘制机器学习任务
marksize = 3.5
T = arg.T
step = 1
index=range(0,T+T//100,T//100)

fig,ax = plt.subplots(2,1,figsize=(10,10))
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel(r'f($\hat{x}$)')
ax[0].plot(index[::step],DAGT_Instance.loss_list[::step],'go-',label='DAGT',markersize=marksize)
ax[0].plot(index[::step],AB_SAGA_Instance.loss_list[::step],'rv-',label='AB-SAGA',markersize=marksize)
ax[0].plot(index[::step],GT_VR_Instance.loss_list[::step],'y*-',label='GT-VR',markersize=marksize)
ax[0].plot(index[::step],VRDAGT_Instance.loss_list[::step],'b^-',label='VRDAGT',markersize=marksize)
ax[0].legend()


ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Optimality Gap')
ax[1].plot(index[::step],DAGT_Instance.opt_gap_list[::step],'go-',label='DAGT',markersize=marksize)
ax[1].plot(index[::step],AB_SAGA_Instance.opt_gap_list[::step],'rv-',label='AB-SAGA',markersize=marksize)
ax[1].plot(index[::step],GT_VR_Instance.opt_gap_list[::step],'y*-',label='GT-VR',markersize=marksize)
ax[1].plot(index[::step],VRDAGT_Instance.opt_gap_list[::step],'b^-',label='VRDAGT',markersize=marksize)


ax[1].legend()
ax[1].legend()
plt.show()