import os



input = '../input'

output = './'

file_train =  os.path.join(input, 'train.csv')

file_test = os.path.join(input, 'test.csv')



file_result = os.path.join(output, 'submission.csv')



batch_size = 512

lr = 0.01

n_epochs = 20



train_coverage = 0.8
import logging



logging.basicConfig( level=logging.DEBUG,)



# console_logging_handler = logging.StreamHandler()

# console_logging_handler.setLevel(logging.DEBUG)

# logging.getLogger('').addHandler(console)





import numpy as np

import torch



np.random.seed(137)

torch.manual_seed(137)
import pandas as pd



pd_train = pd.read_csv(file_train, encoding = "UTF-8")

pd_test = pd.read_csv(file_test, encoding = "UTF-8")
pd_train.head()
pd_test.head()
from torch.utils.data import Dataset

import torch



class drds(Dataset):

    def __init__(self, data, train=True):

        self.data = data

        self.train = train

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        cdata = self.data.iloc[index]

        if self.train:

            inputs = torch.FloatTensor(cdata[1:]).view(1,28,28)

            target = torch.LongTensor([cdata[0]])[0]

        else:

            inputs = torch.FloatTensor(cdata).view(1,28,28)

            target = torch.LongTensor([0])[0]

        return inputs, target



ds_train = drds(pd_train, train=True)

ds_test = drds(pd_test, train=False)
from torch.autograd import Variable

from torch.utils.data import DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

import math



def get_sampler(start, stop):

    return SubsetRandomSampler(np.arange(start, stop, dtype=np.int64))



cnt_train = math.ceil(train_coverage * ds_train.__len__())



loader_train = DataLoader( ds_train, batch_size=batch_size, shuffle=False,

            sampler=get_sampler(0, cnt_train), num_workers=0, drop_last=True, )



loader_val = DataLoader( ds_train, batch_size=batch_size, shuffle=False,

            sampler=get_sampler(0, cnt_train), num_workers=0, drop_last=True, )



loader_test = DataLoader( ds_test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False, )
device = torch.device("cpu")



if torch.cuda.is_available():

    device = torch.device('cuda')



print("using device: {}".format(device))
import math

import torch

import torch.nn as nn

import torch.nn.functional as F



import logging





p_dropout = 0.001



class drnet(nn.Module):

    def __init__(self):

        super(drnet, self).__init__()

        

        self.seq = nn.Sequential(

            nn.Conv2d(

                in_channels=1,

                out_channels=8,

                kernel_size=2,

                stride=1,

                padding=0,

            ),

            nn.BatchNorm2d(8),

            nn.RReLU(inplace=True),

            nn.Conv2d(

                in_channels=8,

                out_channels=16,

                kernel_size=3,

                stride=1,

                padding=0,

            ),

            nn.MaxPool2d(kernel_size=2),

            # More layers

            nn.BatchNorm2d(16),

            nn.ReLU(inplace=True),

            # nn.Dropout(p=p_dropout),

            nn.Conv2d(

                in_channels=16,

                out_channels=16,

                kernel_size=3,

                stride=1,

                padding=0,

            ),

            nn.MaxPool2d(kernel_size=2),

            # 

            nn.BatchNorm2d(16), # BN

            nn.ReLU(inplace=True), # ReLU

        )



        in_features = int(self.conv(torch.zeros(1, 1, 28,28)).size(1))

        out_features = 10

        

        logging.info("initialized cnn.conv, num feature dimension: {}".format(

            in_features))

        

        self.fc = nn.Sequential(

            nn.Linear(in_features, out_features),

            # nn.LogSoftmax(dim=1),

            nn.Softmax(dim=1),

        )

        

        self.models = {}



    def conv(self, x):

        x = self.seq(x)

        # x = torch.mean(x, dim=2, keepdim=True)

        x = x.view(x.size(0), -1)

        # print('#### view:shape: ', x.shape)

        # view:shape:  torch.Size([2880, 14976])

        return x



    def forward(self, x):

        x = self.conv(x)

        return self.fc(x)



    def save(self, key):

        model = self.state_dict()

        self.models[key] = model



    def load(self, key):

        if key in self.models:

            self.load_state_dict(self.models[key], strict=True)

        else:

            logging.error("key {} not found".format(key))



net = drnet().to(device)
best_acc = 0.0
import torch.optim as optim

import torch.nn as nn



loss = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=lr)





def validation():

    loss_all = 0.0

    loss_cur = 0.0



    n_batches = len(loader_val)

    print_every = n_batches // 5



    correct = 0

    total = 0

    

    for i, data in enumerate(loader_val):

        inputs, targets = data

        inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))

        outputs = net(inputs)

        loss_size = loss(outputs, targets)

        loss_all += float(loss_size.data)

        loss_cur += float(loss_size.data)

        

        _, vpredicted = torch.max(outputs, 1)

        for pi, predicted in enumerate(vpredicted):

            expected = int(targets[pi])

            if expected == predicted:

                correct += 1

            total += 1

                

        if (i + 1 ) % print_every == 0:

            avg_loss_all = loss_all / (i + 1)

            avg_loss_cur = loss_cur / print_every

            loss_cur = 0.0

            acc = correct / total

            print("validation: {}, progress: {:.02f}% loss: {:.04f}/{:.04f}, acc: {:.04f}".format(

                epoch, 

                (100 * (i+1)/n_batches),

                avg_loss_cur,

                avg_loss_all,

                acc,

            ))

    if total == 0:

        return 0.0

    else:

        return correct / total

        

n_batches = len(loader_train)

print_every = n_batches // 5



for epoch in range(n_epochs):

    loss_all = 0.0

    loss_cur = 0.0



    print("epoch: {}".format(epoch))

    for i, data in enumerate(loader_train):

        inputs, targets = data

        inputs, targets = Variable(inputs.to(device)), Variable(targets.to(device))

        

        optimizer.zero_grad()

        outputs = net(inputs)

        loss_size = loss(outputs, targets)

        loss_size.backward()

        optimizer.step()

        

        loss_all += float(loss_size.data)

        loss_cur += float(loss_size.data)

        

        if (i + 1 ) % print_every == 0:

            avg_loss_all = loss_all / (i + 1)

            avg_loss_cur = loss_cur / print_every

            loss_cur = 0.0

            print("epoch: {}, progress: {:.02f}% loss: {:.04f}/{:.04f}".format(

                epoch, 

                (100 * (i+1)/n_batches),

                avg_loss_cur,

                avg_loss_all,

            ))



    train_loss = loss_all / n_batches

    val_acc = validation()

    print("epoch: {}, train loss: {:.04f}, validation accuracy: {:.04f}".format(

                epoch, 

                train_loss,

                val_acc,

            ))

    if val_acc > best_acc:

        print("val_acc ({}) > best_acc ({}), updating".format(val_acc, best_acc))

        net.save("best")

        best_acc = val_acc



net.load("best")

last_val = validation()

print("current: train loss: {:.04f}, validation loss: {:.04f}".format(

                train_loss,

                last_val,

            ))
with open(file_result, "w") as f:

    il = 1

    f.write('ImageId,Label\n')

    n_batches = len(loader_test)

    print_every = n_batches // 5

    for i, data in enumerate(loader_test):

        inputs, _ = data

        inputs = Variable(inputs.to(device))

        outputs = net(inputs)

        _, vpredicted = torch.max(outputs, 1)

        for pi, predicted in enumerate(vpredicted):

            f.write("{},{}\n".format(il, predicted))

            il += 1



        if (i + 1 ) % print_every == 0:

            print("save progress: {:.02f}%".format(

                (100 * (i+1)/n_batches),

            ))

    print("done")