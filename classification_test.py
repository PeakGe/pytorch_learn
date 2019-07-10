#2分类问题
#1.导入相关的包
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

#2.构造数据集
n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)      # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)               # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2*n_data, 1)     # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)                # class1 y data (tensor), shape=(100, 1)

x=torch.cat((x0,x1),0).type(torch.FloatTensor)
y=torch.cat((y0,y1),0).type(torch.LongTensor)
# torch.cat(seq, dim=0, out=None) → Tensor
# 在给定维度上对输入的张量序列seq 进行连接操作。
# torch.cat()可以看做 torch.split() 和 torch.chunk()的逆运算。
# 参数:
# seq（Tensors的序列） - 可以是相同类型的Tensor的任何python序列。
# dim（int，可选） - 张量连接的尺寸
# out（Tensor，可选） - 输出参数

#3.定义一个继承torch.nn.Module的类
class Net(torch.nn.Module):
    def __init__(self,num_features,num_hidden,num_output):
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(num_features,num_hidden)
        self.out=torch.nn.Linear(num_hidden,num_output)

    def forward(self, x):
        x=F.relu(self.hidden(x))
        x=self.out(x)
        return x

#4.定义网络，loss，优化器
net=Net(2,20,2)
print(net)

#定义交叉熵损失函数
loss_func=torch.nn.CrossEntropyLoss()
#定义SGD优化器
optimizer=torch.optim.SGD(net.parameters(),lr=0.002)

#5.迭代训练
for epoch in range(200):
    out=net(x)

    loss=loss_func(out,y)
    optimizer.zero_grad()#应该在反向传播之前
    loss.backward()
    optimizer.step()

    if epoch % 2 == 0:
        # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
