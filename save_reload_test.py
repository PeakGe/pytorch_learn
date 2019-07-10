#1--保存整个网络==torch.save(net,'文件名')
# 加载时，得到整个网络==torch.load('文件名')
#2--只保存网络的参数==torch.save(net.state_dict(),'文件名')
# 加载前要先定义好和保存网络同样架构的网络，然后只需加载参数==net'.load_state_dict(torch.load_state_dict('文件名'))

import torch
import matplotlib.pyplot as plt

x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+0.1*torch.normal(torch.zeros(x.size()))

def save():
    net1=torch.nn.Sequential(
        torch.nn.Linear(1,20),
        torch.nn.ReLU(),
        torch.nn.Linear(20,1)
    )
    loss_func=torch.nn.MSELoss()
    optimizer=torch.optim.Adam(net1.parameters(),lr=0.02)

    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

    #训练结束后，保存模型
    torch.save(net1,'net1.pkl')
    #只保存参数
    torch.save(net1.state_dict(),'net1_params.pkl')

def reload():
    net2=torch.load('net1.pkl')
    predict=net2(x)
    # plot result
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)

def reload_param():
    net3=torch.nn.Sequential(
        torch.nn.Linear(1, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1)
    )
    net3.load_state_dict(torch.load('net1_params.pkl'))
    predict = net3(x)

    # plot result
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)
    plt.show()

if __name__=='__main__':
    save()

    reload()

    reload_param()