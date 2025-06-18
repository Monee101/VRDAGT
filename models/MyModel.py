from matplotlib import pyplot as plt
from torch import nn
import torch
import time
class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = str(type(self))

    def load(self,path):
        self.load_state_dict(torch.load(path))

    def save(self,name):
        """
        模型保存至checkpoints目录中，默认命名方式为模型名称+时间
        """
        if name is not None:
            prefix = 'checkpoints/' + name
        else:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(),name)
        return name

    def forward(self, input):
        pass

    def get_gradient(self, input, target):
        pass
    
    def loss(self, output, target):
        pass

    def predict(self, input, one_hot=True):
        # 独热编码预测函数
        if one_hot:
            output = self(input)
            _, predicted = torch.max(output, 1)
            return predicted
        else:
            # 非独热编码预测函数
            pred = self(input)
            pred[torch.where(pred>=0)]=1
            pred[torch.where(pred<0)]=-1
            return pred

    def accuracy(self,input,target,one_hot=True):
        if one_hot:
            pred = self.predict(input,one_hot)
            return (pred == torch.argmax(target, 1)).float().mean().item()
        else:
            pred = self.predict(input,one_hot)
            return (pred == target).float().mean().item()

class SimpleMLP(Model):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 50)
        self.fc2 = nn.Linear(50, output_dim)
    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = self.fc2(x)
        # softmax
        x = torch.nn.functional.softmax(x, dim=1)
        return x
    
    def get_gradient(self, input, target):
        """
        计算梯度
        """
        # 计算损失
        output = self(input)
        loss = self.loss(output, target)
        # 梯度清零
        self.zero_grad()
        # 反向传播
        loss.backward()
        return loss
    
    def loss(self, output, target):
        """
        计算损失
        """
        return nn.CrossEntropyLoss()(output, target)
    
class AlexNet(Model):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 96, 11, 4, 2)
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)
        self.fc1 = nn.Linear(256*6*6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, output_dim)

    def forward(self, input):
        x = torch.relu(self.conv1(input))
        x = torch.max_pool2d(x, 3, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 3, 2)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.max_pool2d(x, 3, 2)
        x = x.view(-1, 256*6*6)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_gradient(self, input, target):
        """
        计算梯度
        """
        # 计算损失
        output = self(input)
        loss = self.loss(output, target)
        # 梯度清零
        self.zero_grad()
        # 反向传播
        loss.backward()
        return loss
    
    def loss(self, output, target):
        """
        计算损失
        """
        return nn.CrossEntropyLoss()(output, target)

class LeNet(Model):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        # 开根号input_dim
        self.input_dim = int(input_dim**0.5)
        self.conv1 = nn.Conv2d(1, 1, 5)
        self.fc1 = nn.Linear(1*12*12, 10)  # Reduce the number of neurons in the first fully connected layer
        self.fc2 = nn.Linear(10, output_dim)

    def forward(self, input):
        input = input.view(-1, 1, self.input_dim, self.input_dim)
        x = torch.relu(self.conv1(input))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 1*12*12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_gradient(self, input, target):
        """
        计算梯度
        """
        # 计算损失
        output = self(input)
        loss = self.loss(output, target)
        # 梯度清零
        self.zero_grad()
        # 反向传播
        loss.backward()
        return loss
    
    def loss(self, output, target):
        """
        计算损失
        """
        return nn.CrossEntropyLoss()(output, target)

class logistic_regression(Model):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        # 计算逻辑回归损失
        self.c = output_dim
        self.d = input_dim
        self.x = nn.Parameter(torch.zeros(self.c,self.d,requires_grad=True,dtype=torch.float32))
        # 初始化参数
        # nn.init.xavier_normal_(self.x)

    def forward(self, input):
        mid = input @ self.x.T
        return mid
    
    def get_gradient(self, input, target):
        """
        计算梯度
        """
        # 计算损失
        loss = self.loss(self(input), target)
        # 梯度清零
        self.zero_grad()
        # 反向传播
        loss.backward()
        return loss
    
    def loss(self, input, target):
        """
        计算损失
        """
        # lambda1 = 0.5*10**(-5) # 正则项系数
        lambda1 = 0.001
        logits = target *input 
        # f = torch.sum(1/(1+torch.exp(logits))/len(input),dim=0) + lambda1 * torch.norm(self.x)**2
        geman_mcclure = torch.sum(0.01 * self.x**2 / (1 + self.x**2))
        f = torch.sum(1/(1+torch.exp(logits))/len(input),dim=0) + lambda1 * geman_mcclure
        return f

class ImageDeblurring(Model):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.c = output_dim
        self.d = input_dim
        self.x = nn.Parameter(torch.zeros(self.c,1,requires_grad=True,dtype=torch.float32))
        # 初始化参数
        # nn.init.xavier_normal_(self.x)

    def forward(self, input):
        return torch.nn.functional.conv2d(self.x.reshape(1,1,28,28),input.reshape(1,1,1,-1),padding='same')
    
    def loss(self,input,target):
        """
        定义损失函数 
        """
        lambda_ = 0.01
        target = target.reshape(28,28)
        geman_mcclure = torch.sum(0.01 * self.x**2 / (1 + self.x**2))
        return torch.norm(input - target.reshape(int(self.c**0.5),int(self.c**0.5)))**2 + lambda_* geman_mcclure
    
    def get_gradient(self, input, target):
        """
        计算梯度
        """
        # 计算损失
        loss = self.loss(self(input), target)
        # 梯度清零
        self.zero_grad()
        # 反向传播
        loss.backward()
        return loss
        


# 模型枚举类
class ModelEnum:
    logistic_regression = 'logistic_regression'
    simple_mlp = 'simple_mlp'
    image_deblurring = 'image_deblurring'
    alexnet = 'alexnet'
    lenet = 'lenet'
