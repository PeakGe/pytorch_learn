#对二次函数y=x^2的回归

#1.导入相关包
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import matplotlib.pyplot as plt

#2.设置LR，BATCH_SIZE,EPOCH
LR=0.01
BATCH_SIZE=32
EPOCH=200

#3.生成数据集
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)

#torch.unsqueeze(input, dim, out=None)
# 返回一个新的张量，对输入的制定位置插入维度 1
# 注意： 返回张量与输入张量共享内存，所以改变其中一个的内容会改变另一个。
# 如果dim为负，则将会被转化\( dim+input.dim()+1 \)
# 参数:
# tensor (Tensor) – 输入张量
# dim (int) – 插入维度的索引
# out (Tensor, 可选的) – 结果张量

y=x.pow(2)+0.1*torch.randn(x.size())#x.size():[100,1]
#torch.randn(*sizes, out=None) → Tensor
# 返回一个张量，包含了从正态分布(均值为0，方差为 1，即高斯白噪声)中抽取一组随机数。 Tensor的形状由变量sizes定义
#参数:
#sizes (int...) – 整数序列，定义了输出形状
#out (Tensor, 可选) - 结果张量

#4.散点图
plt.scatter(x.numpy(),y.numpy())
plt.show()

#5.加载数据集
# 先转换成 torch 能识别的 Dataset
torch_dataset=Data.TensorDataset(x,y)
# class torch.utils.data.TensorDataset(data_tensor, target_tensor)
# 包装数据和目标张量的数据集。
# 通过沿着第一个维度索引两个张量来恢复每个样本。
# 参数：
# data_tensor (Tensor) －　包含样本数据
# target_tensor (Tensor) －　包含样本目标（标签）

#torch.utils.data.DataLoader实现数据装载,dataset:数据集名称，batch_size,shuffle:装载时是否将数据随机打乱，num_workers：线程数
loader=Data.DataLoader(dataset=torch_dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
# class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, num_workers=0,
# collate_fn=<function default_collate>, pin_memory=False, drop_last=False)
# 数据加载器。组合数据集和采样器，并在数据集上提供单进程或多进程迭代器。
#
# 参数：
#
# dataset (Dataset) – 从中​​加载数据的数据集。
# batch_size (int, optional) – 批训练的数据个数(默认: 1)。
# shuffle (bool, optional) – 设置为True在每个epoch重新排列数据（默认值：False,一般打乱比较好）。
# sampler (Sampler, optional) – 定义从数据集中提取样本的策略。如果指定，则忽略shuffle参数。
# batch_sampler（sampler，可选） - 和sampler一样，但一次返回一批索引。与batch_size，shuffle，sampler和drop_last相互排斥。
# num_workers (int, 可选) – 用于数据加载的子进程数。0表示数据将在主进程中加载​​（默认值：0）
# collate_fn (callable, optional) – 合并样本列表以形成小批量。
# pin_memory (bool, optional) – 如果为True，数据加载器在返回前将张量复制到CUDA固定内存中。
# drop_last (bool, optional) – 如果数据集大小不能被batch_size整除，设置为True可删除最后一个不完整的批处理。
#   如果设为False并且数据集的大小不能被batch_size整除，则最后一个batch将更小。(默认: False)

#6构建一个继承了torch.nn.Module的新类，重写前向传播过程
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(1,20)
        self.predict=torch.nn.Linear(20,1)

    def forward(self, x):
        x=F.relu(self.hidden(x))
        x = self.predict(x)
        return x

# 如果一个.py文件（模块）被直接运行时，则其没有包结构，其__name__值为__main__，即模块名为__main__。
#
# 所以，if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
# 当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
if __name__ == '__main__':

#7.定义1个网络，loss，优化器

    net=Net()
    #定义均方差损失函数
    loss_func=torch.nn.MSELoss()
    #定义Adam优化器，指定lr和betas
    optimizer=torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))

    plt.ion()
#迭代训练
    for step in range(EPOCH):
        #网络输出
        predict=net(x)
        loss=loss_func(predict,y)## 一定要是输出在前，标签在后 (1. nn output, 2. target)

        #在本次反向传播之前,清除上一次训练得到的梯度
        optimizer.zero_grad()
        #反向传播,计算梯度
        loss.backward()
        #更新梯度,单词优化
        optimizer.step()

        if (step+1)%5==0:
            plt.cla()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)
            plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.pause(0.1)

plt.ioff()
plt.show()
