import pandas as pd
import numpy as np

import torch
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(777)
torch.manual_seed(777)
if device == 'cuda' :
    torch.cuda.manual_seed_all(777)

# 학습 파라미터 설정
learning_rate = 0.0001
training_epochs = 470
batch_size = 15

# Data load
train_data = pd.read_csv('train_data.csv', header=None, skiprows=1, usecols=range(0, 13))
test_data = pd.read_csv('test_data.csv', header=None, skiprows=1, usecols=range(0, 12))

# Data 파싱
x_train_data = train_data.loc[:, 1:13]
y_train_data = train_data.loc[:, 0]

# 파싱한 Data를 numpy의 array로 변환
x_train_data = np.array(x_train_data)
y_train_data = np.array(y_train_data)

test_data = np.array(test_data)

# 변환한 numpy의 array를 Tensor로 변환
x_train_data = torch.FloatTensor(x_train_data)
y_train_data = torch.LongTensor(y_train_data)

test_data = torch.FloatTensor(test_data)

# data_loader에 이용할 하나의 train Dataset으로 변환
train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)

# data_loader 설정
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

# 모델 설계
linear1 = torch.nn.Linear(12, 1024, bias=True)
linear2 = torch.nn.Linear(1024, 2048, bias=True)
linear3 = torch.nn.Linear(2048, 1024, bias=True)
linear4 = torch.nn.Linear(1024, 1024, bias=True)
linear5 = torch.nn.Linear(1024, 7, bias=True)
leakyrelu = torch.nn.LeakyReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_normal_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)

model = torch.nn.Sequential(linear1, leakyrelu,
                            linear2, leakyrelu,
                            linear3, leakyrelu,
                            linear4, leakyrelu,
                            linear5).to(device)

loss = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

# 모델 학습
total_batch = len(data_loader)

for epoch in range(training_epochs) :
    avg_cost = 0

    for X, Y in data_loader :

        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)
        cost = loss(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print('Epoch : {:4d}'.format(epoch+1), 'Cost : {:.9f}'.format(avg_cost))

print('Learning Finishied')

# 모델 평가
with torch.no_grad() :
    test_data = test_data.to(device)

    prediction = model(test_data)
    prediction = torch.argmax(prediction, 1)
    prediction = prediction.cpu().numpy().reshape(-1, 1)

submit = pd.read_csv('submission_format.csv')

for i in range(len(prediction)) :
    submit['Lable'][i] = prediction[i].item()

submit.to_csv('result.csv', index=False, header=True)
