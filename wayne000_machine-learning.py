import torch

from torch.autograd import Variable

import torch.nn.functional as F

import matplotlib.pyplot as plt



tensor = torch.FloatTensor([[1,2],[3,4]])

variable = Variable(tensor,requires_grad=True)



print(tensor)

print(variable)



t_out = torch.mean(tensor*tensor)#平均值

v_out = torch.mean(variable*variable)

print(t_out)#=1/4*sum(var*var)

print(v_out)



v_out.backward()

print(variable.grad)#=d(v_out)/d(variable)=1/4*2*variable = variable/2
import torch

from torch.autograd import Variable

import torch.nn.functional as F

import matplotlib.pyplot as plt



x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)

y = x.pow(2)+20*torch.rand(x.size())



x,y = Variable(x) ,Variable(y)



class Net(torch.nn.Module):  # 继承 torch 的 Module

    def __init__(self, n_feature, n_hidden, n_output):

        super(Net, self).__init__()     # 继承 __init__ 功能

        # 定义每层用什么样的形式

        self.hidden = torch.nn.Linear(n_feature, n_hidden)   

        # 隐藏层线性输出

        self.predict = torch.nn.Linear(n_hidden, n_output)   

        # 输出层线性输出



    def forward(self, x):   # 这同时也是 Module 中的 forward 功能

        # 正向传播输入值, 神经网络分析出输出值

        x = F.sigmoid(self.hidden(x))      # 激励函数(隐藏层的线性值)

        x = self.predict(x)             # 输出值

        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)

#print(net)  # net 的结构

plt.ion()   # 画图

plt.show()



optimizer = torch.optim.SGD(net.parameters(), lr=0.2)  # 传入 net 的所有参数, 学习率

loss_func = torch.nn.MSELoss()      # 预测值和真实值的误差计算公式 (均方差)



for t in range(100):

    prediction = net(x)     # 喂给 net 训练数据 x, 输出预测值

    loss = loss_func(prediction, y)     # 计算两者的误差

    optimizer.zero_grad()   # 清空上一步的残余更新参数值

    loss.backward()         # 误差反向传播, 计算参数更新值

    optimizer.step()    # 将参数更新值施加到 net 的 parameters 上



#畫圖

for t in range(200):

    if t % 10 == 0:

        # plot and show learning process

        plt.cla()

        plt.scatter(x.data.numpy(), y.data.numpy())

        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)

        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color':  'red'})

        plt.pause(0.1)