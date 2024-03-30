import numpy as np
import torch 
from torch import nn
from torchvision import datasets, transforms,utils
from PIL import Image
import matplotlib.pyplot as plt

# 定义超参数
batch_size = 128 # 每个批次（batch）的样本数

# 对输入的数据进行标准化处理
# transforms.ToTensor() 将图像数据转换为 PyTorch 中的张量（tensor）格式，并将像素值缩放到 0-1 的范围内。
# 这是因为神经网络需要的输入数据必须是张量格式，并且需要进行归一化处理，以提高模型的训练效果。
# transforms.Normalize(mean=[0.5],std=[0.5]) 将图像像素值进行标准化处理，使其均值为 0，标准差为 1。
# 输入数据进行标准化处理可以提高模型的鲁棒性和稳定性，减少模型训练过程中的梯度爆炸和消失问题。
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.5],std=[0.5])])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=transform, 
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', 
                                          train=False, 
                                          transform=transform, 
                                          download=True)
                                          
# 创建数据加载器（用于将数据分次放进模型进行训练）
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True, # 装载过程中随机乱序
                                           num_workers=2) # 表示2个子进程加载数据
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=2) 
print(len(train_dataset))
print(len(test_dataset))
# batch=128
# train_loader=60000/128 = 469 个batch
# test_loader=10000/128=79 个batch
print(len(train_loader))
print(len(test_loader))
for i in range(0,5):
    oneimg,label = train_dataset[i]
    grid = utils.make_grid(oneimg)
    grid = grid.numpy().transpose(1,2,0) 
    std = [0.5]
    mean = [0.5]
    grid = grid * std + mean
    # 可视化图像
    plt.subplot(1, 5, i+1)
    plt.imshow(grid)
    plt.axis('off')

plt.show()
class CNN(nn.Module):
    # 定义网络结构
    def __init__(self):
        super(CNN, self).__init__()
        # 图片是灰度图片，只有一个通道
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, 
                               kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, 
                               kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=7*7*32, out_features=256)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=256, out_features=10)
	
    # 定义前向传播过程的计算函数
    def forward(self, x):
        # 第一层卷积、激活函数和池化
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # 第二层卷积、激活函数和池化
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # 将数据平展成一维
        x = x.view(-1, 7*7*32)
        # 第一层全连接层
        x = self.fc1(x)
        x = self.relu3(x)
        # 第二层全连接层
        x = self.fc2(x)
        return x
