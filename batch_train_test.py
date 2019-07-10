#输出每次批量训练的数据

#1.导包
import torch
import torch.utils.data as Data

#2.随机数种子
torch.manual_seed(1)

#3.BATCH_SIZE
BATCH_SIZE=10

#4.生成数据集
x=torch.linspace(-1,1,50)
y=torch.linspace(1,100,50)

#5.加载数据集1==>torch.utils.data.TensorDataSet(x,y)
#           2==>torch.utils.data.DataLoader(......)
torch_dataset=Data.TensorDataset(x,y)
loader=Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
#6.批量训练
def show_batch():
    for epoch in range(10):
        for step,(batch_x,batch_y) in enumerate(loader):
            print('\nepoch :',epoch,'| step:',step,'| batch_x:',batch_x.numpy(),'| batch_y:',batch_y.numpy())

if __name__=='__main__':
    show_batch()