import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import argparse

seed = 20

import networkx as nx
import random

def is_strongly_connected(G):
    return nx.is_strongly_connected(G)

def generate_strongly_connected_graph(num_nodes):
    # 创建一个空的有向图
    G = nx.DiGraph()

    # 添加节点
    G.add_nodes_from(range(num_nodes))

    # 确保图是强连通的
    while not is_strongly_connected(G):
        # 随机选择两个节点
        node1 = random.randint(0, num_nodes - 1)
        node2 = random.randint(0, num_nodes - 1)

        # 确保不是自环
        if node1 != node2:
            # 随机决定边的方向
            direction = random.choice(['to', 'from'])
            if direction == 'to':
                G.add_edge(node1, node2)
            else:
                G.add_edge(node2, node1)

    return G

# 使用Erd ̋os–Rényi方法生成图的邻接矩阵，输入为图的节点数n和边数m，每个节点以p的概率连接到其他节点
def generate_adjacency_matrix(n: int, p: float=1,kind:'str'='d'):
    """
    生成图的邻接矩阵
    :param n: 图的节点数
    :param p: 每个节点以p的概率连接到其他节点
    :return: 图的邻接矩阵
    """
    if kind == 'd':
        A = np.zeros((n, n))

        # 变量degree用于记录每个节点的度数

        degree = np.zeros(n)

        # A是无向图，所以A是对称矩阵，只需要生成上三角矩阵再赋值给下三角矩阵即可
        # 同时，需要判断生成的矩阵A是否连通，如果不连通则重新生成
        while not is_connected_dfs(A):
            for i in range(n):
                for j in range(i+1, n):
                    if np.random.rand() < p:
                        A[i][j] = 1
                        A[j][i] = 1
                        degree[i] += 1
                        degree[j] += 1

        DouStoMatrix = A.copy()

        # 使用Metropolis weight protocol调整为双随机矩阵
        for i in range(n):
            for j in range(i, n):
                if i != j and A[i][j] == 1:
                    DouStoMatrix[i][j] = 1 / (max(degree[i], degree[j])+1)
                    DouStoMatrix[j][i] = DouStoMatrix[i][j]
                elif i == j:
                    # 求出节点i的邻居的度
                    max_degree = []
                    for k in range(n):
                        if A[i][k] == 1:
                            max_degree.append(degree[k] if degree[k] > degree[i] else degree[i])
                    DouStoMatrix[i][j] = 1 - sum([1 / (max_degree[k] + 1) for k in range(len(max_degree))])
                else:
                    DouStoMatrix[i][j] = 0
                    DouStoMatrix[j][i] = 0

        # 检查谱半径是否小于1
        m_2 = np.ones((n,n))*1/n
        a,b=np.linalg.eig(DouStoMatrix-m_2) #a为特征值集合，b为特征值向量
        if(np.max(np.abs(a))<1):
            print("谱半径小于1符合要求")
        else:
            print("谱半径大于1不符合要求,谱半径为：",np.max(np.abs(a)))

        return DouStoMatrix
    elif kind == 'rc':
        # 邻接矩阵是有向图
        g = generate_strongly_connected_graph(n)
        # # 使用NetworkX绘制图形
        # print('绘图')
        # nx.draw(g, with_labels=True, node_color='lightblue', edge_color='gray')
        # plt.show()
        # 将图转换为邻接矩阵
        A = nx.to_numpy_array(g).T
        B = nx.to_numpy_array(g).T
        print('图G是否强连通:',is_strongly_connected(g))
        for i in range(n):
            A[i] = A[i] / np.sum(A[i])
            B[:,i] = B[:,i] / np.sum(B[:,i])
        return A,B

def DFS(visited, graph, vertex):
    visited[vertex] = True
    for neighbor in range(len(graph)):
        if graph[vertex][neighbor] != 0 and not visited[neighbor]:
            DFS(visited, graph, neighbor)


def is_connected_dfs(graph):
    V = len(graph)
    visited = [False] * V
    DFS(visited, graph, 0)
    for visit in visited:
        if not visit:
            return False
    return True

def generate_noise(nodes,x_div):
    global seed
    #  固定随机种子
    np.random.seed(50)
    torch.manual_seed(50)
    noise_list = np.zeros((nodes,x_div))
    # 定义截断正态分布的参数
    mu = 0
    sigma = np.sqrt(0.04)
    a = -0.1  # 截断下界
    b = 0.1  # 截断上界
    # 计算正确的截断参数
    a, b = (a - mu) / sigma, (b - mu) / sigma
    # 创建截断正态分布对象
    trunc_norm = truncnorm(a, b, loc=mu, scale=sigma)

    for i in range(nodes):
        samples = trunc_norm.rvs(x_div)
        noise_list[i] = samples
    return torch.tensor(noise_list).reshape(nodes,x_div,1)


def generate_matrix_with_eigenvalues(eigenvalues, nodes,size):
    M = np.zeros((nodes,size, size))
    for i in range(len(eigenvalues)):

        # 生成一个随机正交矩阵
        A = np.random.randn(size, size)
        Q, _ = np.linalg.qr(A)  # Q is an orthogonal matrix

        # 创建一个对角矩阵，对角线上是特征值
        Lambda = np.diag(eigenvalues[i])

        # 生成具有特定特征值的矩阵
        M[i] = Q @ Lambda @ Q.T
        # 判断M的特征值是否符合要求，特征值范围在[0.05,1]
        if np.max(np.abs(np.linalg.eigvals(M[i])))>1 or np.min(np.abs(np.linalg.eigvals(M[i])))<0.05:
            print("第",i,"个矩阵的特征值不符合要求")
            print("特征值为：",np.linalg.eigvals(M[i]))
            print("最大特征值为：",np.max(np.abs(np.linalg.eigvals(M[i]))))
            print("最小特征值为：",np.min(np.abs(np.linalg.eigvals(M[i]))))
        else:
            print("第",i,"个矩阵的特征值符合要求")
    return M


def compose_x(x,config):
    """
    合并所有节点的x
    """
    x_div = config['x_div']
    nodes = config['nodes']
    # 关闭梯度
    for i in range(len(x)):
        x[i].requires_grad = False
    x = torch.stack(x,0)
    return x

def separate_x(x,config):
    """
    将所有节点的x分开
    """
    c = config['c']
    d = config['d']
    x_list = []
    if c <= 1:
        for i in range(len(x)):
            x_list.append(x[i].reshape(len(x[i]),1).clone().detach().requires_grad_(True))
    else:
        for i in range(len(x)):
            x_list.append(x[i].reshape(c,d).clone().detach().requires_grad_(True))
    return x_list

def arg_praser():
    """
    --dataset: 数据集名称
    --model: 模型名称
    --nodes: 节点数
    --T: 迭代次数
    --batchsize: 批量大小
    --onehot: 是否使用独热编码
    --alpha: 学习率
    """

    parser = argparse.ArgumentParser(description='please input the parameters')
    parser.add_argument('--dataset', type=str, default='covtype', help='dataset name')
    parser.add_argument('--model', type=str, default='simple_mlp', help='model name')
    parser.add_argument('--nodes', type=int, default=10, help='number of nodes')
    parser.add_argument('--T', type=int, default=6400, help='number of iterations')
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument('--onehot', action="store_true", help='one hot encoding')
    parser.add_argument('--alpha', type=float, default=0.01, help='learning rate')
    
    return parser.parse_args()
