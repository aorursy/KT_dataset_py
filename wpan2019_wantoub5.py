# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import torch

import time

from tqdm import tqdm

import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler

from torch.utils.data import DataLoader, Dataset

from torchvision import transforms, models

from torch.utils.data import DataLoader, Dataset
train_csv_path = '../input/wantoub7/train.csv'

train_df = pd.read_csv(train_csv_path)

test_csv_path = '../input/wantoub7/test.csv'

test_df = pd.read_csv(test_csv_path)
class MNISTResNet(nn.Module):

    def __init__(self):

        super(MNISTResNet, self).__init__()

        self.conv0_0 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn0_0 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu0_0 = nn.ReLU()

        

        self.conv1_0 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1_0 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu1_0 = nn.ReLU()

        self.conv1_1 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1_1 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        

        self.conv1_2 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1_2 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu1_2 = nn.ReLU()

        self.conv1_3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1_3 = nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        

        

        self.conv2_0 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn2_0 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu2_0 = nn.ReLU()

        self.conv2_1 = nn.Conv1d(128,128, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2_1 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        

        self.conv2_2 = nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2_2 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu2_2 = nn.ReLU()

        self.conv2_3 = nn.Conv1d(128,128, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn2_3 = nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        

        

        self.conv3_0 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn3_0 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu3_0 = nn.ReLU()

        self.conv3_1 = nn.Conv1d(256,256, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3_1 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        

        self.conv3_2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3_2 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu3_2 = nn.ReLU()

        self.conv3_3 = nn.Conv1d(256,256, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn3_3 = nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        

        self.conv4_0 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)

        self.bn4_0 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu4_0 = nn.ReLU()

        self.conv4_1 = nn.Conv1d(512,512, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn4_1 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)        

        self.conv4_2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn4_2 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.relu4_2 = nn.ReLU()

        self.conv4_3 = nn.Conv1d(512,512, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn4_3 = nn.BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        

        self.dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(in_features=1536, out_features=2, bias=True)

#         self.fc = nn.Linear(in_features=1536, out_features=1, bias=True)

    def forward(self, x):

        x = self.conv0_0(x)

        x = self.bn0_0(x)

        x = self.relu0_0(x)



        x = self.conv1_0(x)

        x = self.bn1_0(x)

        x = self.relu1_0(x)

        x = self.conv1_1(x)

        x = self.bn1_1(x)

        x = self.conv1_2(x)

        x = self.bn1_2(x)

        x = self.relu1_2(x)

        x = self.conv1_3(x)

        x = self.bn1_3(x)



        x = self.conv2_0(x)

        x = self.bn2_0(x)

        x = self.relu2_0(x)

        x = self.conv2_1(x)

        x = self.bn2_1(x)

        x = self.conv2_2(x)

        x = self.bn2_2(x)

        x = self.relu2_2(x)

        x = self.conv2_3(x)

        x = self.bn2_3(x)



        x = self.conv3_0(x)

        x = self.bn3_0(x)

        x = self.relu3_0(x)

        x = self.conv3_1(x)

        x = self.bn3_1(x)

        x = self.conv3_2(x)

        x = self.bn3_2(x)

        x = self.relu3_2(x)

        x = self.conv3_3(x)

        x = self.bn3_3(x)



        x = self.conv4_0(x)

        x = self.bn4_0(x)

        x = self.relu4_0(x)

        x = self.conv4_1(x)

        x = self.bn4_1(x)

        x = self.conv4_2(x)

        x = self.bn4_2(x)

        x = self.relu4_2(x)

        x = self.conv4_3(x)

        x = self.bn4_3(x)

        

        x = x.view(x.size(0), -1)

        

        x = self.dropout(x)

        

        x = self.fc(x)

        return x
def dataset(train_df,bs):

    class_sample_count = np.array(

        [len(np.where(train_df["Number-of-complaints"] == t)[0]) for t in np.unique(train_df["Number-of-complaints"])])

    weight = 1. / class_sample_count

    samples_weight = np.array([weight[t] for t in train_df["Number-of-complaints"].values])



    samples_weight = torch.from_numpy(samples_weight)

    samples_weight = samples_weight.double()

    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))



    target = torch.from_numpy(train_df["Number-of-complaints"].values).long()

    data = (torch.from_numpy(train_df.values[:,:-1])).type(torch.FloatTensor)

    train_dataset = torch.utils.data.TensorDataset(data, target)



    data_loader = DataLoader(train_dataset, batch_size=bs, num_workers=1, sampler=sampler)

    return data_loader



def dataset_v(train_df):

    target = torch.from_numpy(train_df["Number-of-complaints"].values).long()

    data = (torch.from_numpy(train_df.values[:,:-1])).type(torch.FloatTensor)

    train_dataset = torch.utils.data.TensorDataset(data, target)



    data_loader = DataLoader(train_dataset,np.shape(train_df.values)[0])

    return data_loader



def dataset_t(train_df,bs):

    target = torch.from_numpy(train_df["Number-of-complaints"].values).long()

    data = (torch.from_numpy(train_df.values[:,:-1])).type(torch.FloatTensor)

    train_dataset = torch.utils.data.TensorDataset(data, target)



    data_loader = DataLoader(train_dataset,batch_size=bs)

    return data_loader
from sklearn.metrics import f1_score

bs = 700

rand_seed = 3

fraction=0.9

n_epochs = 30

best_score = 0.



model = MNISTResNet()

if torch.cuda.is_available():

    model.cuda()



loss_fn = torch.nn.CrossEntropyLoss().cuda()

start_time = time.time()



for epoch in range(n_epochs):

    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    df_1 = train_df.sample(frac = fraction , random_state=rand_seed)

    df_2 = train_df.drop(df_1.index)



    train_loader = dataset(df_1,bs)

    val_loader = dataset_v(df_2)

    test_loader = dataset_v(test_df)



    avg_loss = 0.

    train_error = 0.

    for  i, (x_batch, y_batch) in enumerate(train_loader):

        x_batch = x_batch[:,None,:].type(torch.FloatTensor)

        if torch.cuda.is_available():

            x_batch = x_batch.cuda()

            y_batch = y_batch.cuda()



        y_pred = model(x_batch)

        train_preds = torch.max(y_pred.cpu(), 1)[1]

        train_error += np.square(np.subtract(train_preds.cpu().numpy(),y_batch.cpu().numpy())).sum()/(2*len(y_batch))

        loss = loss_fn(y_pred, y_batch)



        optimizer.zero_grad()



        loss.backward()



        optimizer.step()

        avg_loss += loss.item() / len(train_loader)

    

    model.eval()



    valid_preds = np.zeros((x_batch.size(0)))#x_batch.size(0)



    avg_val_loss = 0.

    avg_val_correct = 0.

    err = 0.

    recall = 0.

    for i, (x_batch, y_batch) in enumerate(val_loader):

        x_batch = x_batch[:,None,:].type(torch.FloatTensor)

        if torch.cuda.is_available():

            x_batch = x_batch.cuda()

            y_batch = y_batch.cuda()

        y_pred = model(x_batch).detach()



        avg_val_loss += loss_fn(y_pred, y_batch).item() / len(val_loader)

        valid_preds = torch.max(y_pred.cpu(), 1)[1]

        recall += (np.multiply(valid_preds.cpu().numpy(),y_batch.cpu().numpy())).sum()/ (y_batch.cpu().numpy().sum()+1)

        err += np.square(np.subtract(valid_preds.cpu().numpy(),y_batch.cpu().numpy())).sum()/(2*len(y_batch))

        score = f1_score(y_true = y_batch.cpu().numpy(), y_pred=valid_preds.cpu().numpy())

    avg_val_error = err/len(val_loader)  

    avg_val_recall = recall/len(val_loader)  

    elapsed_time = time.time() - start_time

    fdrecall = score/avg_val_recall

    PATH = 'net'+ str(epoch) +'.pkl'

    torch.save(model.state_dict(), PATH)

    print('Epoch {}/{} \t Target {}/{}/{} \t loss={:.4f} \t val_loss={:.4f} \t val_err={:.4f} \t val_recall={:.4f} \t f1_score={:.4f} \t f1/recall={:.4f} \t model={} \t time={:.2f}s'.format(

            epoch + 1, n_epochs, y_batch.cpu().numpy().sum() , valid_preds.cpu().numpy().sum() , len(y_batch) , avg_loss, avg_val_loss, avg_val_error, avg_val_recall, score , fdrecall , PATH ,elapsed_time))

    if score > best_score:

        best_score = score

        best_path = PATH

print(best_score,best_path)
model.load_state_dict(torch.load(best_path))

# model.load_state_dict(torch.load('net12.pkl'))

bss = 40000



test_loader = dataset_t(test_df,bss)



test_error = 0.

model.eval()

output = np.zeros((np.shape(test_df.values)[0]))

for i, (x_batch, y_batch) in enumerate(test_loader):

    x_batch = x_batch[:,None,:].type(torch.FloatTensor)

    if torch.cuda.is_available():

        x_batch = x_batch.cuda()

        y_batch = y_batch.cuda()

    y_pred = model(x_batch).detach()



    output[i*bss:i*bss+np.shape(y_pred)[0]] = torch.max(y_pred.cpu(), 1)[1]

    test_preds = torch.max(y_pred.cpu(), 1)[1]

    recall = (np.multiply(test_preds.cpu().numpy(),y_batch.cpu().numpy())).sum()/ (y_batch.cpu().numpy().sum()+1)

    err += np.square(np.subtract(test_preds.cpu().numpy(),y_batch.cpu().numpy())).sum()/(2*len(y_batch))

    score = f1_score(y_true = y_batch.cpu().numpy(), y_pred=test_preds.cpu().numpy())

    fdrecall = score/recall

    print('Target {}/{}/{} \t test_recall={:.4f} \t f1_score={:.4f} \t f1/recall={:.4f} \t'.format(

            y_batch.cpu().numpy().sum() , test_preds.cpu().numpy().sum() , len(y_batch) , recall, score , fdrecall))

# output.to_csv("output.csv", index=False)

data = pd.DataFrame(output)



writer = pd.ExcelWriter('Prediction.xlsx')		# 写入Excel文件

data.to_excel(writer, 'Prediction', float_format='%.5f')		# ‘page_1’是写入excel的sheet名

writer.save()



writer.close()