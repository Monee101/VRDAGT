import time
import torch
from torch.utils.data import Subset
import os
from data import fashion_mnist_blurring
import torch.nn as nn
from models.MyModel import ModelEnum
from algorithm.alg import *
from utils.tools import *
import scipy.io as scio

for i in range(0,3):
    nodes = 10 # 节点数
    image_num = i
    fashion_mnist_bur = fashion_mnist_blurring(nodes,image_num,center=False,one_hot=False)
    datasets = [Subset(fashion_mnist_bur.get_nodes_dataset_one_column(i),[0]) for i in range(nodes)]

    # 创建通信网络,所有算法都使用同一个网络
    # 创建log文件夹
    # dir_path = r'./log/对比试验'+time.strftime(" %Y年%m月%d日%H时%M分%S秒", time.localtime())
    # file_path = dir_path + r'/log.txt'
    # os.makedirs(dir_path)

    # 方便一次跑多个数据集用于绘图
    dir_path = r'./draw/image_deb_'+str(i)
    file_path = dir_path + r'/log.txt'
    os.makedirs(dir_path)

    # 存储本次的模糊数据
    np.save(dir_path+'/blur_data.npy',fashion_mnist_bur.Y[0].numpy())

    eum_model = ModelEnum.image_deblurring

    A,B = generate_adjacency_matrix(nodes, kind='rc')
    A = torch.tensor(A,dtype=torch.float32)
    B = torch.tensor(B,dtype=torch.float32)
    print('A:',A)
    print('B:',B)
    # C_path =  r'D:\my_project\GT_VR_Simulation\a9a\data\new_C.mat'
    # C =  scio.loadmat(C_path)['C_store']
    # C = torch.tensor(C,dtype=torch.float32)
    GT_DAG_config = {'nodes':nodes,'Phi':None,'theta':None,'opt_x':None,'beta1':0.9,'beta2':0.999,
                    'train_dataset_nodes':datasets,'test_dataset_nodes':None,'alpha':0.005,'v_min':10**-8,'v_max':100,
                    'T':10000,'batch_size':1,'model':eum_model,'A':A,'B':B,'dir_path':dir_path,'file_path':file_path,'C':None}

    GT_DAG_config['P'] = 0.1

    # 学习率0.001是最优的

    DAGT = DAGT(GT_DAG_config)
    ABSAGA_Instance = AB_SAGA(GT_DAG_config)
    VRDAGT_Instance = VRDAGT(GT_DAG_config)
    GTVR_Instance = GT_VR(GT_DAG_config)
    GTVR_Instance.run()
    DAGT.run()
    ABSAGA_Instance.run()
    VRDAGT_Instance.run()
    Model_GT_DAG = DAGT.model_avg
    Model_AB_SAGA = ABSAGA_Instance.model_avg
    Model_GT_DAG_VR = VRDAGT_Instance.model_avg
    Model_GT_VR = GTVR_Instance.model_avg

    # 绘制loss和optgap的对比图，同样只绘制points个点
    points = 100

    #绘制loss和optgap的对比图，每个模型的线例不同
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    marksize = 3.5
    T = DAGT.T
    step = T//points
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('loss')
    ax[0].plot(range(0,T,step),DAGT.loss_list[::step],'go-',label='GT_DAG',markersize=marksize)
    ax[0].plot(range(0,T,step),ABSAGA_Instance.loss_list[::step],'rv-',label='AB_SAGA',markersize=marksize)
    ax[0].plot(range(0,T,step),GTVR_Instance.loss_list[::step],'y*-',label='GT_VR',markersize=marksize)
    ax[0].plot(range(0,T,step),VRDAGT_Instance.loss_list[::step],'b^-',label='GT_DAG_VR',markersize=marksize)
    ax[0].legend()
    ax[0].set_title('loss on train dataset')

    #局部放大图，放大后100个点
    axins_loss = ax[0].inset_axes([0.6, 0.6, 0.2, 0.2])
    axins_loss.plot(range(T-100,T),ABSAGA_Instance.loss_list[T-100:],'rv-',label='AB_SAGA',markersize=marksize,markevery=5)
    axins_loss.plot(range(T-100,T),GTVR_Instance.loss_list[T-100:],'y*-',label='GT_VR',markersize=marksize,markevery=5)
    axins_loss.plot(range(T-100,T),DAGT.loss_list[T-100:],'go-',label='GT_DAG',markersize=marksize,markevery=5)
    axins_loss.plot(range(T-100,T),VRDAGT_Instance.loss_list[T-100:],'b^-',label='GT_DAG_VR',markersize=marksize,markevery=5)

    # 设定放大区域
    x1, x2, y1, y2 = T-100, T, min(VRDAGT_Instance.loss_list[T-100:]), max(DAGT.loss_list[T-100:])
    axins_loss.set_xlim(x1, x2)
    axins_loss.set_ylim(y1, y2)
    # # 添加连接线
    mark_inset(ax[0], axins_loss, loc1=3, loc2=1, fc="none", ec='0.5')

    ############################################################################################################


    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('optgap')
    ax[1].plot(range(0,T,step),DAGT.opt_gap_list[::step],'go-',label='GT_DAG',markersize=marksize)
    ax[1].plot(range(0,T,step),ABSAGA_Instance.opt_gap_list[::step],'rv-',label='AB_SAGA',markersize=marksize)
    ax[1].plot(range(0,T,step),GTVR_Instance.opt_gap_list[::step],'y*-',label='GT_VR',markersize=marksize)
    ax[1].plot(range(0,T,step),VRDAGT_Instance.opt_gap_list[::step],'b^-',label='GT_DAG_VR',markersize=marksize)
    #局部放大图，放大后100个点
    axins_optgap = ax[1].inset_axes([0.6, 0.3, 0.3, 0.3])
    axins_optgap.plot(range(T-100,T),DAGT.opt_gap_list[T-100:],'go-',label='GT_DAG',markersize=marksize,markevery=2)
    axins_optgap.plot(range(T-100,T),ABSAGA_Instance.opt_gap_list[T-100:],'rv-',label='AB_SAGA',markersize=marksize,markevery=2)
    axins_optgap.plot(range(T-100,T),GTVR_Instance.opt_gap_list[T-100:],'y*-',label='GT_VR',markersize=marksize,markevery=2)
    axins_optgap.plot(range(T-100,T),VRDAGT_Instance.opt_gap_list[T-100:],'b^-',label='GT_DAG_VR',markersize=marksize,markevery=2)
    # 设定放大区域
    x1, x2, y1, y2 = T-100, T, 0, max(DAGT.opt_gap_list[T-100:])
    axins_optgap.set_xlim(x1, x2)
    axins_optgap.set_ylim(y1, y2)
    # # 添加连接线
    mark_inset(ax[1], axins_optgap, loc1=3, loc2=1, fc="none", ec='0.5')
    ax[1].legend()
    ax[1].set_title('optgap on train dataset')
    ax[1].legend()
    plt.savefig(dir_path+'/loss_optgap.png')


    # 绘制模糊图像和去模糊后的图像
    orgin_iamge = fashion_mnist_bur.image
    blur_image = fashion_mnist_bur.Y[0].reshape(28,28).numpy()
    deblur_image_GT_DAG = Model_GT_DAG.state_dict()['x'].reshape(28,28).numpy()
    deblur_image_AB_SAGA = Model_AB_SAGA.state_dict()['x'].reshape(28,28).numpy()
    deblur_image_GT_DAG_VR = Model_GT_DAG_VR.state_dict()['x'].reshape(28,28).numpy()
    deblur_image_GT_VR = Model_GT_VR.state_dict()['x'].reshape(28,28).numpy()
    fig,ax = plt.subplots(2,3,figsize=(10,7))
    ax[0][0].imshow(orgin_iamge,cmap='gray')
    ax[0][0].set_title('orgin image')
    ax[0][1].imshow(blur_image,cmap='gray')
    ax[0][1].set_title('blur image')
    ax[0][2].imshow(deblur_image_GT_DAG,cmap='gray')
    ax[0][2].set_title('deblur image GT_DAG')
    ax[1][0].imshow(deblur_image_AB_SAGA,cmap='gray')
    ax[1][0].set_title('deblur image AB_SAGA')
    ax[1][1].imshow(deblur_image_GT_DAG_VR,cmap='gray')
    ax[1][1].set_title('deblur image GT_DAG_VR')
    ax[1][2].imshow(deblur_image_GT_VR,cmap='gray')
    ax[1][2].set_title('deblur image GT_VR')
    plt.savefig(dir_path+'/image.png')
    plt.close()


    # 绘制psnr和ssim
    fig,ax = plt.subplots(2,1,figsize=(10,10))
    ax[0].set_xlabel('iteration')
    ax[0].set_ylabel('psnr')
    ax[0].plot(range(0,T,step),DAGT.psnr_list[::step],'go-',label='GT_DAG',markersize=marksize)
    ax[0].plot(range(0,T,step),ABSAGA_Instance.psnr_list[::step],'rv-',label='AB_SAGA',markersize=marksize)
    ax[0].plot(range(0,T,step),GTVR_Instance.psnr_list[::step],'y*-',label='GT_VR',markersize=marksize)
    ax[0].plot(range(0,T,step),VRDAGT_Instance.psnr_list[::step],'b^-',label='GT_DAG_VR',markersize=marksize)
    ax[0].legend()

    ax[1].set_xlabel('iteration')
    ax[1].set_ylabel('ssim')
    ax[1].plot(range(0,T,step),DAGT.ssim_list[::step],'go-',label='GT_DAG',markersize=marksize)
    ax[1].plot(range(0,T,step),ABSAGA_Instance.ssim_list[::step],'rv-',label='AB_SAGA',markersize=marksize)
    ax[1].plot(range(0,T,step),GTVR_Instance.ssim_list[::step],'y*-',label='GT_VR',markersize=marksize)
    ax[1].plot(range(0,T,step),VRDAGT_Instance.ssim_list[::step],'b^-',label='GT_DAG_VR',markersize=marksize)
    ax[1].legend()
    plt.savefig(dir_path+'/psnr_ssim.png')
    plt.close()
