import torch
import numpy as np
from tqdm import tqdm
from utils.tools import *


# GT-based Distributed Adaptive Gradient Algorithm实现
def GT_DAG(x,config):

    nodes = config['nodes']
    x_div = config['x_div']
    Phi = config['Phi']
    theta = config['theta']



    loss_list = []
    opt_gap_list = []

    m = torch.zeros(nodes, x_div, 1)# 所有节点的动量梯度

    beta1 = config['beta1']  # 动量梯度的指数衰减速率，0<beta1<1

    beta2 = config['beta2']  # 自适应步长缩放向量的指数衰减速率，0<beta2<1

    alpha = config['alpha'] # 学习步长

    v_min = config['v_min'] # 自适应步长缩放向量的最小值
    v_max = config['v_max'] # 自适应步长缩放向量的最大值

    # 迭代次数
    T = config['T']
    # 初始话s为所有节点的梯度初始值
    s,_,_ = get_gradient(x,config)

    # 初始化v为s与s的Hardmard积
    v = s * s # 所有节点的自适应步长缩放向量

        # 产生邻接矩阵A
    A,B = generate_adjacency_matrix(nodes, 0.5,'rc')
    A = torch.tensor(A)
    B = torch.tensor(B)
    old_gradient = s
    C = generate_adjacency_matrix(nodes, 0.5,'d')
    C = torch.tensor(C)
    from scipy.linalg import eig
    w1,vl1,vr1 = eig(B,left=True,right=True)
    vr1 = abs(vr1[:,np.where(np.abs(w1-1)<1e-8)])
    vr1 = vr1/np.sum(vr1)
    vr1 = torch.tensor(vr1,dtype=torch.float64).reshape(nodes,1)
    print('B',B)
    print("vr1:",vr1)

    loop = tqdm(range(T), desc="GT_DAG Iteration")
    for t in loop:
        # 更新动量m
        m = beta1 * m + (1 - beta1) * s.reshape(nodes,x_div,1)
        # 更新辅助变量v_hat
        v_hat = beta2 * v + (1 - beta2) * s.reshape(nodes,x_div,1) * s.reshape(nodes,x_div,1) # 所有节点的自适应步长缩放向量的辅助向量
        # 更新自适应步长缩放向量v
        v = np.clip(v_hat, v_min, v_max)
        # 更新辅助矩阵V,构造对角矩阵
        # V = torch.linalg.inv(torch.diag_embed(v.reshape(nodes,x_div), dim1=1, dim2=2)) ** 0.5


        # 更新自变量x,网络拓扑是全连接的
        x = compose_x(x,config)
        x = A@x.reshape(nodes,x_div)- (alpha * v**-0.5 * m).reshape(nodes,x_div)

        # 更新梯度s
        x = separate_x(x,config)
        grad,loss,noise = get_gradient(x,config)

        loss_list.append(loss.item())

        s = B@s.reshape(nodes,x_div) + grad.reshape(nodes,x_div)- old_gradient.reshape(nodes,x_div)
        old_gradient = grad
        p3 = vr1@torch.ones(nodes,dtype=torch.float64).reshape(1,nodes)@s.reshape(nodes,x_div)
        p4 = 1/nodes * torch.ones((nodes,nodes),dtype=torch.float64)@s.reshape(nodes,x_div)

        s_norm= torch.norm(s-p3)**2
        s_norm_2 = torch.norm(s-p4)**2

        v3 = vr1@torch.ones(nodes,dtype=torch.float64).reshape(1,nodes)@v.reshape(nodes,x_div)
        v4 = 1/nodes * torch.ones((nodes,nodes),dtype=torch.float64)@v.reshape(nodes,x_div)

        v_norm = torch.norm(v.reshape(nodes,x_div)-v3)**2
        v_norm_2 = torch.norm(v.reshape(nodes,x_div)-v4)**2
        opt_gap = opt_gap_fun(x,config)
        opt_gap_list.append(opt_gap.item())
        loop.set_postfix(loss=loss.item(),grad=torch.norm(grad).item(),opt_gap=opt_gap.item(),noise=torch.norm(noise).item())

    print('v:\n',v)
    print('v_hat:\n',v_hat)
    print('m:\n',m)
    print('s:\n',s)
    print('v_norm_vr:',v_norm)
    print('v_norm_avg:',v_norm_2)
    print('s_norm_vr:',s_norm)
    print('s_norm_avg:',s_norm_2)

    # 画出每次迭代的loss,并进行美化
    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.xlim(0,T)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration')
    plt.savefig('loss.png')
    plt.show()
    plt.close()

    # 画出每次迭代的opt_gap,并进行美化
    plt.plot(opt_gap_list)
    plt.xlim(0,T)
    plt.xlabel('Iteration')
    plt.ylabel('Opt_gap')
    plt.title('Opt_gap vs. Iteration')
    plt.savefig('opt_gap.png')
    plt.show()
    plt.close()
    return x

# GT-based Distributed Adaptive Gradient Algorithm实现
def GT_DAG_VR(x,config):

    nodes = config['nodes']
    x_div = config['x_div']
    Phi = config['Phi']
    theta = config['theta']



    loss_list = []
    opt_gap_list = []

    m = torch.zeros(nodes, x_div, 1)# 所有节点的动量梯度

    beta1 = config['beta1']  # 动量梯度的指数衰减速率，0<beta1<1

    beta2 = config['beta2']  # 自适应步长缩放向量的指数衰减速率，0<beta2<1

    alpha = config['alpha'] # 学习步长

    v_min = config['v_min'] # 自适应步长缩放向量的最小值
    v_max = config['v_max'] # 自适应步长缩放向量的最大值

    # 迭代次数
    T = config['T']
    # 初始话s为所有节点的梯度初始值
    s,_,_ = get_gradient(x,config)

    # 初始化v为s与s的Hardmard积
    v = s * s # 所有节点的自适应步长缩放向量

        # 产生邻接矩阵A
    A,B = generate_adjacency_matrix(nodes, 0.5,'rc')
    A = torch.tensor(A)
    B = torch.tensor(B)
    old_gradient = s
    C = generate_adjacency_matrix(nodes, 0.5,'d')
    C = torch.tensor(C)
    from scipy.linalg import eig
    w1,vl1,vr1 = eig(B,left=True,right=True)
    vr1 = abs(vr1[:,np.where(np.abs(w1-1)<1e-8)])
    vr1 = vr1/np.sum(vr1)
    vr1 = torch.tensor(vr1,dtype=torch.float64).reshape(nodes,1)
    print('B',B)
    print("vr1:",vr1)

    loop = tqdm(range(T), desc="GT_DAG Iteration")
    for t in loop:
        # 更新动量m
        m = beta1 * m + (1 - beta1) * s.reshape(nodes,x_div,1)
        # 更新辅助变量v_hat
        v_hat = beta2 * v + (1 - beta2) * s.reshape(nodes,x_div,1) * s.reshape(nodes,x_div,1) # 所有节点的自适应步长缩放向量的辅助向量
        # 更新自适应步长缩放向量v
        v = np.clip(v_hat, v_min, v_max)
        # 更新辅助矩阵V,构造对角矩阵
        # V = torch.linalg.inv(torch.diag_embed(v.reshape(nodes,x_div), dim1=1, dim2=2)) ** 0.5


        # 更新自变量x,网络拓扑是全连接的
        x = compose_x(x,config)
        x = A@x.reshape(nodes,x_div)- (alpha * v**-0.5 * m).reshape(nodes,x_div)

        # 更新梯度s
        x = separate_x(x,config)
        grad,loss,noise = get_gradient(x,config)

        loss_list.append(loss.item())

        s = B@s.reshape(nodes,x_div) + grad.reshape(nodes,x_div)- old_gradient.reshape(nodes,x_div)
        old_gradient = grad
        p3 = vr1@torch.ones(nodes,dtype=torch.float64).reshape(1,nodes)@s.reshape(nodes,x_div)
        p4 = 1/nodes * torch.ones((nodes,nodes),dtype=torch.float64)@s.reshape(nodes,x_div)

        s_norm= torch.norm(s-p3)**2
        s_norm_2 = torch.norm(s-p4)**2

        v3 = vr1@torch.ones(nodes,dtype=torch.float64).reshape(1,nodes)@v.reshape(nodes,x_div)
        v4 = 1/nodes * torch.ones((nodes,nodes),dtype=torch.float64)@v.reshape(nodes,x_div)

        v_norm = torch.norm(v.reshape(nodes,x_div)-v3)**2
        v_norm_2 = torch.norm(v.reshape(nodes,x_div)-v4)**2
        opt_gap = opt_gap_fun(x,config)
        opt_gap_list.append(opt_gap.item())
        loop.set_postfix(loss=loss.item(),grad=torch.norm(grad).item(),opt_gap=opt_gap.item(),noise=torch.norm(noise).item())

    print('v:\n',v)
    print('v_hat:\n',v_hat)
    print('m:\n',m)
    print('s:\n',s)
    print('v_norm_vr:',v_norm)
    print('v_norm_avg:',v_norm_2)
    print('s_norm_vr:',s_norm)
    print('s_norm_avg:',s_norm_2)

    # 画出每次迭代的loss,并进行美化
    import matplotlib.pyplot as plt
    plt.plot(loss_list)
    plt.xlim(0,T)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss vs. Iteration')
    plt.savefig('loss.png')
    plt.show()
    plt.close()

    # 画出每次迭代的opt_gap,并进行美化
    plt.plot(opt_gap_list)
    plt.xlim(0,T)
    plt.xlabel('Iteration')
    plt.ylabel('Opt_gap')
    plt.title('Opt_gap vs. Iteration')
    plt.savefig('opt_gap.png')
    plt.show()
    plt.close()
    return x