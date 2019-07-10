#1.导入相关包
import os
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import torchvision#torchvision包的主要功能是实现数据的处理、导入和预览
import matplotlib.pyplot as plt

#2.设置超参数
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001              # learning rate
DOWNLOAD_MNIST = False  #是否下载

#3.下载数据集
#如果./mnist目录不存在或者./mnist目录下无文件,置DOWNLOAD_MNIST=True
if not os.path.exists('./mnist') or not os.listdir('./mnist'):
    DOWNLOAD_MNIST=True

#4.得到训练集
train_data=torchvision.datasets.MNIST(
    root='./mnist',         #保存路径
    train=True,             #True：训练集，False：测试集
    transform=torchvision.transforms.ToTensor(),#数据类型变为Tensor，且[0,255]==>[0,1]
    download=DOWNLOAD_MNIST   #是否下载
)
# torchvision.datasets中包含了以下数据集
#
# MNIST
# COCO（用于图像标注和目标检测）(Captioning and Detection)
# LSUN Classification
# ImageFolder
# Imagenet-12
# CIFAR10 and CIFAR100
# STL10
# SVHN
# PhotoTour
#=============================================================
# 对Conversion进行变换
# class torchvision.transforms.ToTensor
# 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor
#
# call(pic)
#
# 参数: pic (PIL.Image or numpy.ndarray) – 图片转换为张量.
# 返回结果: 转换后的图像。
# 返回样式: Tensor张量
#=====================================================================
#多行注释快捷键：CTRL+/
# print(train_data.train_data.size())
# print(train_data.train_labels.size())
# plt.imshow(train_data.train_data[0],cmap='gray')
# plt.title('%i' % train_data.train_labels[0])
# plt.show()
#
# print(train_data.train_data[0:5,:,:])
# print('\n',torch.unsqueeze(train_data.train_data[0:5,:,:],dim=1))
print(type(train_data))
#5.加载训练集
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

#6.得到测试集，取前2000个样本,增加1维，转换为torch.FloatTensor类型，并归一化
test_data=torchvision.datasets.MNIST(root='./mnist',train=False)
#测试集数据
test_x=torch.unsqueeze(test_data.test_data[:2000],dim=1).type(torch.FloatTensor)/255
#测试集标签
test_y=test_data.test_labels[:2000]

#7.定义CNN网络
#输入shape:[1,28,28]
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        #conv1: 16 * 5 * 5保持宽高的卷积(stride=1, padding=2), relu，2 * 2的maxpool
        self.conv1=nn.Sequential(
            #卷积函数
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=5,stride=1,padding=2),#===>[16,28,28]
            nn.ReLU(),                                                                  #===>[16,28,28]
            #池化函数
            nn.MaxPool2d(kernel_size=2)                                                 #===>[16,14,14]
        )
        #conv2:32*5*5保持宽高的卷积(stride=1, padding=2), relu，2 * 2的maxpool
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),                                                     #===>[32,14,14]
            nn.ReLU(),                                                                  #===>[32,14,14]
            nn.MaxPool2d(kernel_size=2)                                                 #===>[32,7,7]
        )
        #全连接层：[32*7*7,10]
        self.out=nn.Linear(32*7*7,10)

    #重写forward()
    def forward(self, x):
        x=self.conv1(x)
        print(x.shape)
        x=self.conv2(x)
        #全连接之前，展平,保留batch的维度[batch,32,7,7]==>[batch,32*7*7]
        x=x.view(x.size(0),-1)#x.size(0)指batchsize的值,而-1指在不告诉函数有多少列的情况下，根据原tensor数据和batchsize自动分配列数
        output=self.out(x)
        return output,x

#8.搭建CNN
cnn=CNN()
print(cnn)

#9.定义loss函数，优化器
loss_func=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)

#10.批量训练
for epoch in range(EPOCH):
    for step,(batch_x,batch_y) in enumerate(train_loader):
        #enumerate:产生带索引值(step,从0开始)的枚举
        #batch_x:[50,1,28,28]
        x_test=torch.ones((50,1,28,28))
        output = cnn(x_test)[0]
        #output=cnn(batch_x)[0]
        loss=loss_func(output,batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1)%50==0:
            test_output=cnn(test_x)[0]
            # xx=torch.max(test_output, 1)
            # print(xx)
            # print(xx[1])

            pred_y = torch.max(test_output, 1)[1].data.numpy()
            # torch.max(test_output,1)返回值是一个元组(tensor1,tensor2)
            # tensor1是test_output在dim=1上最大值数组的tensor,tensor2是test_output在dim=1上最大值下表(0-9)数组的tensor

            #与test标签数组Tensor比较，相同为1，不同为0(将bool转换为int，再转换为float),数组Tensor求和/float(test_y.size(0))=accuracy
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

test_output,_ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print('predict number:',pred_y)
print('real number',test_y[:10].numpy())