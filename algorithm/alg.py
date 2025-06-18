from abc import abstractmethod
from copy import deepcopy
import os
import torch
import numpy as np
from tqdm import tqdm
from utils.tools import *
from models.MyModel import *
import logging
# import scipy.io as scio
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy.linalg import eig
class Algorithm:
    def __init__(self, config):
        self.config = config
        self.loss_list = []
        self.opt_gap_list = []
        self.alpha = config['alpha'] # 学习步长
        self.T = config['T']
        self.nodes = config['nodes']
        self.model = config['model']
        self.train_dataset_nodes = config['train_dataset_nodes']
        self.test_dataset_nodes = config['test_dataset_nodes']
        self.c = self.train_dataset_nodes[0].dataset.c
        self.d = self.train_dataset_nodes[0].dataset.d
        self.name = None
        self.dir_path = config['dir_path']
        self.file_path = config['file_path']
        self.opt_gap_list = []
        self.model_list = [None for _ in range(self.nodes)]
        self._init_model()
        self.model_avg = deepcopy(self.model_list[0])
        self.A = config['A']
        self.B = config['B']
        # 计算A的特征值为1的左特征向量
        w, vl, vr = eig(self.A, left=True)
        index = np.where(np.abs(w - 1) < 1e-6)[0][0]
        # 取特征向量的实部
        self.pi_r = torch.tensor(np.abs(vl[:,index]), dtype=torch.float32)
        self.pi_r = self.pi_r / torch.sum(self.pi_r) # 归一化
        # temp = self.pi_r.reshape(1,self.nodes)@self.A
        self.psnr_list = list()
        self.ssim_list = list()
        self.acc_test_list = list()
        self.acc_train_list = list()
        if self.model != ModelEnum.image_deblurring:
            self.psnr_list.append(None)
            self.ssim_list.append(None)
        self.one_hot = self.train_dataset_nodes[0].dataset.one_hot
    @abstractmethod
    def run(self):
        pass

    def _init_model(self):
        if self.model == ModelEnum.logistic_regression:
            for i in range(self.nodes):
                self.model_list[i] = logistic_regression(self.d,self.c)
        elif self.model == ModelEnum.simple_mlp:
            for i in range(self.nodes):
                self.model_list[i] = SimpleMLP(self.d,self.c)
        elif self.model == ModelEnum.image_deblurring:
            for i in range(self.nodes):
                self.model_list[i] = ImageDeblurring(self.d,self.c)
        elif self.model == ModelEnum.alexnet:
            for i in range(self.nodes):
                self.model_list[i] = AlexNet(self.d,self.c)
        elif self.model == ModelEnum.lenet:
            for i in range(self.nodes):
                self.model_list[i] = LeNet(self.d,self.c)
        
    def _compose_model(self,name):
        # 将所有模型的name层参数拼接在一起,维度为(nodes,div),div为参数维度展开后的长度
        return torch.cat([model.state_dict()[name].reshape(1,-1) for model in self.model_list],0)
    
    def _separate_model(self,x,name):
        # 将拼接的参数分离为各个模型的name层参数,在原地修改
        for i in range(self.nodes):
            for name_it, param in self.model_list[i].named_parameters():
                if name == name_it:
                    param.data = x[i].reshape(param.data.shape)

    def getGradient(self,avg_grad=False):
        loss_list = []
        gradients = {}
        if avg_grad:
            for i in range(self.nodes):
                self.model_list[i].train()
                dataset = self.train_dataset_nodes[i]
                input,target = dataset[:]
                self.model_list[i].get_gradient(input,target)
        else:
            # 随机梯度下降
            for i in range(self.nodes):
                self.model_list[i].train()
                dataset = self.train_dataset_nodes[i]
                batch_num = np.random.randint(0,len(dataset))
                input,target = dataset[batch_num:(batch_num+1)]
                self.model_list[i].get_gradient(input,target)
        
        # 将所有模型的梯度拼接在一起,维度为(nodes,div),div为参数维度展开后的长度
        for i in range(self.nodes):
            for name, param in self.model_list[i].named_parameters():
                if name not in gradients:
                    gradients[name] = param.grad.clone().detach().reshape(1,-1)
                else:
                    gradients[name] = torch.cat((gradients[name],param.grad.clone().detach().reshape(1,-1)),0)

        return gradients
    
    def log(self):
        # 计算log下的文件夹数量，如果超过5个，则删除最早的文件夹和文件夹下的文件
        log_path = r'./log'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_list = os.listdir(log_path)
        while len(log_list) >= 10:
            log_list = sorted(log_list)
            # 删除最早的文件夹和文件夹下的文件
            path = os.path.join(log_path,log_list[0])
            for file in os.listdir(path):
                os.remove(os.path.join(path,file))
            os.rmdir(path)
            # 更新log_list
            log_list = os.listdir(log_path)

        # 保存日志，记录所有输入的参数、输出的参数、loss、opt_gap等
        logging.basicConfig(level=logging.INFO, filename=self.file_path, filemode='w', format='%(asctime)s - %(levelname)s: %(message)s')
        logging.info('nodes:'+str(self.nodes))
        logging.info('alpha:'+str(self.alpha))
        logging.info('algorithm:'+str(self.name))
        logging.info('T:'+str(self.T))
        logging.info('c:'+str(self.c))
        logging.info('d:'+str(self.d))
        logging.info('model:'+str(self.model))
        logging.info('dataset:'+str(self.train_dataset_nodes[0].dataset.name))
        logging.info('loss:'+str(self.loss_list[-1]))
        logging.info('opt_gap:'+str(self.opt_gap_list[-1]))
        # 将loss和opt_gap导出为np的数据文件
        np.save(self.dir_path+r'/loss'+self.name+'.npy',np.array(self.loss_list))
        np.save(self.dir_path+r'/opt_gap'+self.name+'.npy',np.array(self.opt_gap_list))
        if self.model == ModelEnum.image_deblurring:
            logging.info('psnr:'+str(self.psnr_list[-1]))
            logging.info('ssim:'+str(self.ssim_list[-1]))
            np.save(self.dir_path+r'/psnr'+self.name+'.npy',np.array(self.psnr_list))
            np.save(self.dir_path+r'/ssim'+self.name+'.npy',np.array(self.ssim_list))
        else:
            logging.info('acc_test:'+str(self.acc_test_list[-1]))
            logging.info('acc_train:'+str(self.acc_train_list[-1]))
            np.save(self.dir_path+r'/acc_test'+self.name+'.npy',np.array(self.acc_test_list))
            np.save(self.dir_path+r'/acc_train'+self.name+'.npy',np.array(self.acc_train_list))
        # 保存avg_model的参数
        torch.save(self.model_avg.state_dict(),self.dir_path+r'/model_avg'+self.name+'.pt')
        logging.info('end\n\n')

    def _opt_gap_fun(self):
        
        self.models_average(self.model_list,self.model_avg)
        # 计算psnr和ssim
        if self.model == ModelEnum.image_deblurring:
            orgin_image = self.train_dataset_nodes[0].dataset.image.numpy()
            deblur_image = self.model_avg.state_dict()['x'].reshape(28,28).detach().numpy()
            p = psnr(deblur_image,orgin_image,data_range=np.max(orgin_image))
            ss = ssim(deblur_image,orgin_image,data_range=np.max(orgin_image))
            self.psnr_list.append(p)
            self.ssim_list.append(ss)
        else:
            with torch.no_grad():
                acc_test = 0
                acc_train = 0
                for i in range(self.nodes):
                    dataset_test = self.test_dataset_nodes[i]
                    dataset_train = self.train_dataset_nodes[i]
                    input,target = dataset_test[:]
                    acc_test += self.model_avg.accuracy(input,target,self.one_hot)
                    input,target = dataset_train[:]
                    acc_train += self.model_avg.accuracy(input,target,self.one_hot)
                acc_train /= self.nodes
                acc_test /= self.nodes
                self.acc_test_list.append(acc_test)
                self.acc_train_list.append(acc_train)
        # 计算梯度
        self.model_avg.train()
        self.model_avg.zero_grad()
        loss = torch.tensor(0.0).reshape(1)
        for i in range(self.nodes):
            dataset = self.train_dataset_nodes[i]
            input,target = dataset[:]
            loss+=self.model_avg.loss(self.model_avg(input),target)
        loss = loss/self.nodes
        loss.backward()
        self.loss_list.append(loss.item())
        first_term = 0
        second_term = 0
        with torch.no_grad():
            for name,param in self.model_avg.named_parameters():
                for i in range(self.nodes):
                    second_term += torch.norm(self.model_list[i].state_dict()[name] - param)**2
                first_term += torch.norm(param.grad)**2
        # print('first_term:',first_term.item())
        # print('second_term:',second_term.item())
        return first_term+second_term/self.nodes
    
    def models_average(self,models, fl_model):
        # 计算模型的加权平均，权重由pi_r决定
        with torch.no_grad():
            for name, param in fl_model.named_parameters():
                param.data = torch.zeros_like(param)
                for i in range(self.nodes):
                    param.data += self.pi_r[i]*models[i].state_dict()[name]

# GT-based Distributed Adaptive Gradient Algorithm实现
class DAGT(Algorithm):
    def __init__(self, config):
        super(DAGT, self).__init__(config)
        self.beta1 = config['beta1']
        self.beta2 = config['beta2']
        self.v_min = config['v_min']
        self.v_max = config['v_max']
        self.v_dict = {}
        self.m_dict = {}
        self.s_dict = {}
        self.name = 'DAGT'

    def run(self):
        self._init_vsm()
        old_gradient = self.s_dict
        opt_gap = torch.tensor(float('inf'))
        
        loop = tqdm(range(self.T), desc=self.name+" Iteration")
        for t in loop:
            for name,_  in self.model_list[0].named_parameters():
                self.m_dict[name] = self.beta1 * self.m_dict[name] + (1-self.beta1) * self.s_dict[name]
                v_hat = self.beta2 * self.v_dict[name] + (1-self.beta2) * self.s_dict[name] * self.s_dict[name]
                self.v_dict[name] = torch.clip(v_hat,self.v_min,self.v_max)
                x = self._compose_model(name)
                x = self.A@x - (self.alpha*self.v_dict[name]**-0.5 * self.m_dict[name])
                self._separate_model(x,name)
            # 更新完所有模型的参数后，计算梯度
            grad = self.getGradient()
            #更新s
            for name,_ in self.model_list[0].named_parameters():
                self.s_dict[name] = self.B@self.s_dict[name] + grad[name] - old_gradient[name]
            old_gradient = grad
            # 每隔self.T//100次计算一次opt_gap
            if (t+1) % (self.T//100) == 0 or t+1==1:
                opt_gap = self._opt_gap_fun()
                self.opt_gap_list.append(opt_gap.item())
            loop.set_postfix(loss=self.loss_list[-1],opt_gap=self.opt_gap_list[-1],psnr=self.psnr_list[-1],ssim=self.ssim_list[-1])
        self.log()
            
    def _init_vsm(self):
        grad = self.getGradient(True) # 每个模型都有一个梯度
        self.s_dict = grad
        for key in self.s_dict.keys():
            self.m_dict[key] = torch.zeros_like(self.s_dict[key])
            self.v_dict[key] = self.s_dict[key]**2

class VRDAGT(DAGT):
    def __init__(self, config):
        super(VRDAGT, self).__init__(config)
        self.P = config['P']
        self.tau_list = deepcopy(self.model_list)
        self.tau_grad_avg = {}
        self.name = 'VRDAGT'
             
    def getGradient(self, avg_grad=False):
        gradients = {}
        if avg_grad:
            for i in range(self.nodes):
                self.tau_list[i].load_state_dict(self.model_list[i].state_dict())
                self.model_list[i].train()
                dataset = self.train_dataset_nodes[i]
                input,target = dataset[:]
                self.model_list[i].get_gradient(input,target)
                for name, param in self.model_list[i].named_parameters():
                    if name not in gradients:
                        gradients[name] = param.grad.clone().detach().reshape(1,-1)
                    else:
                        gradients[name] = torch.cat((gradients[name],param.grad.clone().detach().reshape(1,-1)),0)
            self.tau_grad_avg = deepcopy(gradients)
        else:
            for i in range(self.nodes):
                if np.random.rand() > self.P:
                    self.tau_list[i].train()
                    self.model_list[i].train()
                    dataset = self.train_dataset_nodes[i]
                    batch_num = np.random.randint(0,len(dataset))
                    # batch_num = self.s_i_k[self.T][i]
                    input,target = dataset[batch_num:(batch_num+1)]
                    self.model_list[i].get_gradient(input,target)
                    self.tau_list[i].get_gradient(input,target)
                    for m, t in zip(self.model_list[i].named_parameters(),self.tau_list[i].named_parameters()):
                        if m[0] not in gradients:
                            gradients[m[0]] = m[1].grad.clone().detach().reshape(1,-1) - t[1].grad.clone().detach().reshape(1,-1) + self.tau_grad_avg[m[0]][i,:].reshape(1,-1)
                        else:
                            gradients[m[0]] = torch.cat((gradients[m[0]],m[1].grad.clone().detach().reshape(1,-1) - t[1].grad.clone().detach().reshape(1,-1) + self.tau_grad_avg[m[0]][i,:].reshape(1,-1)),0)
                else:
                    # 节点i采用全局梯度
                    self.tau_list[i].load_state_dict(self.model_list[i].state_dict())
                    self.model_list[i].train()
                    dataset = self.train_dataset_nodes[i]
                    input,target = dataset[:]
                    self.model_list[i].get_gradient(input,target)
                    for name, param in self.model_list[i].named_parameters():
                        if name not in gradients:
                            gradients[name] = param.grad.clone().detach().reshape(1,-1)
                        else:
                            gradients[name] = torch.cat((gradients[name],param.grad.clone().detach().reshape(1,-1)),0)
                        self.tau_grad_avg[name][i,:] = gradients[name][i,:]
                # 将所有模型的梯度拼接在一起,维度为(nodes,div),div为参数维度展开后的长度
        return gradients
    
class AB_SAGA(Algorithm):
    def __init__(self,config):
        super(AB_SAGA,self).__init__(config)
        self.name = 'AB_SAGA'
        self.grad_avg = [{} for _ in range(self.nodes)]
    def run(self):
        self.grad_tabel = self._init_grad_tabel()
        self.s_dict,_ = self.getGradient(True)
        self.grad_avg = deepcopy(self.s_dict)
        self.min_loss = torch.tensor(float('inf'))
        old_gradient = self.s_dict
        opt_gap = torch.tensor(float('inf'))
        loop = tqdm(range(self.T), desc=self.name+" Iteration")
        for t in loop:
            for name,_  in self.model_list[0].named_parameters():
                
                x = self._compose_model(name)
                x = self.A@(x - self.alpha*self.s_dict[name])
                self._separate_model(x,name)
            # 更新完所有模型的参数后，计算梯度
            grad,loss = self.getGradient()
            #更新s
            for name,_ in self.model_list[0].named_parameters():
                self.s_dict[name] = self.B@(self.s_dict[name] + grad[name] - old_gradient[name])
            old_gradient = grad
            if (t+1) % (self.T//100) == 0 or t+1==1:
                opt_gap = self._opt_gap_fun()
                self.opt_gap_list.append(opt_gap.item())
            loop.set_postfix(loss=self.loss_list[-1],opt_gap=self.opt_gap_list[-1],psnr=self.psnr_list[-1],ssim=self.ssim_list[-1])
        self.log()

    def _init_grad_tabel(self):
        # 对每一个样本初始化梯度表
        grad_table = [{} for _ in range(self.nodes)]
        for i in range(self.nodes):
            dataset = self.train_dataset_nodes[i]
            self.model_list[i].train()
            for j in range(len(dataset)):
                input,target = dataset[j:j+1]
                self.model_list[i].get_gradient(input,target)
                for name,param in self.model_list[i].named_parameters():
                    if name not in grad_table[i]:
                        grad_table[i][name] = [param.grad.clone().detach().reshape(1,-1)]
                    else:
                        grad_table[i][name].append(param.grad.clone().detach().reshape(1,-1))
        return grad_table

    def getGradient(self, avg_grad=False):
        loss_list = []
        gradients = {}
        if avg_grad:
            for i in range(self.nodes):
                self.model_list[i].train()
                dataset = self.train_dataset_nodes[i]
                input,target = dataset[:]
                loss_list.append(self.model_list[i].get_gradient(input,target))
                for name,param in self.model_list[i].named_parameters():
                    if name not in gradients:
                        gradients[name] = param.grad.clone().detach().reshape(1,-1)
                    else:
                        gradients[name] = torch.cat((gradients[name],param.grad.clone().detach().reshape(1,-1)),0)
        else:
            for i in range(self.nodes):
                self.model_list[i].train()
                dataset = self.train_dataset_nodes[i]
                batch_num = np.random.randint(0,len(dataset))
                input,target = dataset[batch_num:(batch_num+1)]
                loss_list.append(self.model_list[i].get_gradient(input,target))
                for name,param in self.model_list[i].named_parameters():
                    if name not in gradients:
                        gradients[name] = param.grad.clone().detach().reshape(1,-1) - self.grad_tabel[i][name][batch_num] + self.grad_avg[name][i,:]
                    else:
                        gradients[name] = torch.cat((gradients[name],param.grad.clone().detach().reshape(1,-1) - self.grad_tabel[i][name][batch_num] + self.grad_avg[name][i,:]),0)
                    self.grad_avg[name][i,:] = (self.grad_avg[name][i,:]*len(self.train_dataset_nodes[i]) - self.grad_tabel[i][name][batch_num] + param.grad.clone().detach().reshape(1,-1))/len(self.train_dataset_nodes[i])
                    self.grad_tabel[i][name][batch_num] = param.grad.clone().detach().reshape(1,-1)

        return gradients,torch.mean(torch.tensor(loss_list))

class GT_VR(VRDAGT):
    def __init__(self, config):
        super(GT_VR, self).__init__(config)
        self.name = 'GT_VR'
        self.C = config['C']
    def run(self):
        self.s_dict = self.getGradient(True)
        self.min_loss = torch.tensor(float('inf'))
        old_gradient = self.s_dict
        opt_gap = torch.tensor(float('inf'))
        loop = tqdm(range(self.T), desc=self.name+" Iteration")

        for t in loop:
            for name,_  in self.model_list[0].named_parameters():
                x = self._compose_model(name)
                x = self.A@(x - self.alpha*self.s_dict[name])
                self._separate_model(x,name)
            #更新完所有模型的参数后，计算梯度
            grad = self.getGradient()
            #更新s
            for name,_ in self.model_list[0].named_parameters():
                self.s_dict[name] = self.B@(self.s_dict[name] + grad[name] - old_gradient[name])
            old_gradient = grad
            if (t+1) % (self.T//100) == 0 or t+1==1:
                opt_gap = self._opt_gap_fun()
                self.opt_gap_list.append(opt_gap.item())
            loop.set_postfix(loss=self.loss_list[-1],opt_gap=self.opt_gap_list[-1],psnr=self.psnr_list[-1],ssim=self.ssim_list[-1])

        self.log()
    




        
        