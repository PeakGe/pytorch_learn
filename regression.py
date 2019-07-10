# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:59:58 2019

@author: gefeng
"""

import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# [1000]==>[1000,1]
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)#增加一维为了与batch对应
# x.size()返回值为shape
# torch.zeros()生成与x同样shape的全0的tensor

#torch.normal(means, std, out=None) → → Tensor
#返回一个张量，包含了从指定均值means和标准差std的离散正态分布中抽取的一组随机数。
#标准差std是一个张量，包含每个输出元素相关的正态分布标准差。参数:
#means (float, optional) - 均值
#std (Tensor) - 标准差
#out (Tensor) - 输出张量
y = x.pow(2) + 0.1*torch.normal(torch.zeros(*x.size()))

# plot dataset
plt.scatter(x.numpy(), y.numpy())
plt.show()

# 将数据集放入torch的数据集中
torch_dataset = Data.TensorDataset(x, y)
#torch.utils.data实现数据装载,dataset:数据集名称，batch_size,shuffle:装载时是否将数据随机打乱，num_workers：线程数
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,)


#构建一个继承了torch.nn.Module的新类，来完成对前向传播函数和后向传播函数的重写。
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()#父类初始化，第一个参数是该类名
        self.hidden = torch.nn.Linear(1, 20)   # hidden layer
        self.predict = torch.nn.Linear(20, 1)   # output layer

    #使用init函数中定义的网络层来构造前向传播的过程
    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

if __name__ == '__main__':
    #搭建了4个神经网络
    net_SGD         = Net()
    net_Momentum    = Net()
    net_RMSprop     = Net()
    net_Adam        = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    #定义4个不同的优化器
    opt_SGD         = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum    = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop     = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam        = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    #定义loss函数，不需要传入参数：均方误差损失函数
    loss_func = torch.nn.MSELoss()
    #每一个网络训练得到的loss都是1个列表类型
    losses_his = [[], [], [], []]   # record loss

    # training
    for epoch in range(EPOCH):
        print('Epoch: ', epoch)
        #每次取1个batch的数据(b_x,b_y)
        for step, (b_x, b_y) in enumerate(loader):          # for each training step
            #对4个不同的网络使用不同的优化器，保存4个不同的loss
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)              # 为每一个网络计算输出结果
                loss = loss_func(output, b_y)  # 为每一个网络计算loss
                opt.zero_grad()                # 为下一个batch训练清除参数
                loss.backward()                # 后向传播，计算梯度
                opt.step()                     # 梯度更新
                l_his.append(loss.data.numpy())     # loss recoder

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim((0, 0.2))
    plt.show()