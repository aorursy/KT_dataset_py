# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import torch

from torch import nn



import matplotlib.pyplot as plt

import pandas as pd



# real price come from 

# Spot Prices for Crude Oil and Petroleum Product. (May.28.2020). U.S. Energy Information Administration. Retrieved from https://www.eia.gov/dnav/pet/pet_pri_spt_s1_d.htm

df1 = pd.read_csv(r'../input/real-price/RealPrice.csv')

print(df1)
# real price come from 

df1 = pd.read_csv(r'../input/real-price/RealPrice.csv', usecols=[1])



table_val = df1.values

#print(table_val)

seq_number = table_val[0:165, :]

print(seq_number[0])

print(seq_number.shape)



seq_number = seq_number.flatten()

seq_number = seq_number[:, np.newaxis]

print(seq_number.shape)



# # print(repr(seq))

# # 1949~1960, 12 years, 12*12==144 month

seq_week = np.arange(33)

seq_day = np.arange(5)

seq_week_day = np.transpose(

    [np.repeat(seq_week, len(seq_day)),

     np.tile(seq_day, len(seq_week))],

)  # Cartesian Product



seq = np.concatenate((seq_number, seq_week_day), axis=1)

print(seq.shape)

print(seq)



# normalization

print("seq.mean(axis=0)")

train_mean = seq[:125].mean(axis=0)

train_std = seq[:125].std(axis=0)

print(train_mean)

print(train_std)

seq = (seq - train_mean) / train_std
class RegLSTM(nn.Module):

    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):

        super(RegLSTM, self).__init__()



        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn

        self.reg = nn.Sequential(

            nn.Linear(mid_dim, mid_dim),

            nn.Tanh(),

            nn.Linear(mid_dim, out_dim),

        )  # regression



    def forward(self, x):

        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)



        seq_len, batch_size, hid_dim = y.shape

        y = y.view(-1, hid_dim)

        y = self.reg(y)

        y = y.view(seq_len, batch_size, -1)

        return y



    """

    PyCharm Crtl+click nn.LSTM() jump to code of PyTorch:

    Examples::

        >>> rnn = nn.LSTM(10, 20, 2)

        >>> input = torch.randn(5, 3, 10)

        >>> h0 = torch.randn(2, 3, 20)

        >>> c0 = torch.randn(2, 3, 20)

        >>> output, (hn, cn) = rnn(input, (h0, c0))

    """



    def output_y_hc(self, x, hc):

        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)



        seq_len, batch_size, hid_dim = y.size()

        y = y.view(-1, hid_dim)

        y = self.reg(y)

        y = y.view(seq_len, batch_size, -1)

        return y, hc
inp_dim = 3

out_dim = 1

mid_dim = 8

mid_layers = 1

batch_size = 12 * 4

mod_dir = '.'



'''load data'''

# data, train_mean, train_std = load_data()

data = seq



data_x = data[:-1, :]

data_y = data[+1:, 0]

assert data_x.shape[1] == inp_dim



train_size = int(len(data_x) * 0.78)

print(train_size)



train_x = data_x[:train_size]

train_y = data_y[:train_size]

train_x = train_x.reshape((train_size, inp_dim))

train_y = train_y.reshape((train_size, out_dim))



'''build model'''

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)



'''train'''

var_x = torch.tensor(train_x, dtype=torch.float32, device=device)

var_y = torch.tensor(train_y, dtype=torch.float32, device=device)



batch_var_x = list()

batch_var_y = list()



for i in range(batch_size):

    j = train_size - i

    batch_var_x.append(var_x[j:])

    batch_var_y.append(var_y[j:])



from torch.nn.utils.rnn import pad_sequence

batch_var_x = pad_sequence(batch_var_x)

batch_var_y = pad_sequence(batch_var_y)



with torch.no_grad():

    weights = np.tanh(np.arange(len(train_y)) * (np.e / len(train_y)))

    weights = torch.tensor(weights, dtype=torch.float32, device=device)



print("Training Start")

for e in range(384):

    out = net(batch_var_x)



    # loss = criterion(out, batch_var_y)

    loss = (out - batch_var_y) ** 2 * weights

    loss = loss.mean()



    optimizer.zero_grad()

    loss.backward()

    optimizer.step()



    if e % 64 == 0:

        print('Epoch: {:4}, Loss: {:.5f}'.format(e, loss.item()))

# torch.save(net.state_dict(), '{}/net.pth'.format(mod_dir))

# print("Save in:", '{}/net.pth'.format(mod_dir))



'''eval'''

# net.load_state_dict(torch.load('{}/net.pth'.format(mod_dir), map_location=lambda storage, loc: storage))

net = net.eval()



test_x = data_x.copy()



# why ? zhangkuan

# test_x[train_size:, 0] = 0



test_x = test_x[:, np.newaxis, :]

print(test_x.shape)

# print(test_x)



test_x = torch.tensor(test_x, dtype=torch.float32, device=device)



'''simple way but no elegant'''

# for i in range(train_size, len(data) - 2):

#     test_y = net(test_x[:i])

#     test_x[i, 0, 0] = test_y[-1]



'''elegant way but slightly complicated'''

eval_size = 1

zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)

test_y, hc = net.output_y_hc(test_x[:train_size], (zero_ten, zero_ten))

test_x[train_size + 1, 0, 0] = test_y[-1]

for i in range(train_size + 1, len(data) - 2):

    test_y, hc = net.output_y_hc(test_x[i:i + 1], hc)

    test_x[i + 1, 0, 0] = test_y[-1]

pred_y = test_x[1:, 0, 0]

pred_y = pred_y.cpu().data.numpy()



print("pred_y")

# print(pred_y)



diff_y = pred_y[train_size:] - data_y[train_size:-1]

l1_loss = np.mean(np.abs(diff_y))

l2_loss = np.mean(diff_y ** 2)

print("L1: {:.3f}    L2: {:.3f}".format(l1_loss, l2_loss))



plt.plot(pred_y, 'r', label='pred')

plt.plot(data_y[0:127], 'b', label='real', alpha=0.3)

plt.plot([train_size, train_size], [-1, 2], color='k', label='train | pred')

plt.legend(loc='best')

#plt.savefig('lstm_reg.png')



#print(pred_y*train_std[0] + train_mean[0])

plt.pause(0.5)