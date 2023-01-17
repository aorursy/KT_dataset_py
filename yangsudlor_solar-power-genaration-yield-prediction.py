import numpy as np 
import pandas as pd 
sensor_data = '../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv'
gen_data = '../input/solar-power-generation-data/Plant_1_Generation_Data.csv'

sensor_df = pd.read_csv(sensor_data)
gen_df = pd.read_csv(gen_data)

sensor_df = sensor_df.drop(['PLANT_ID','SOURCE_KEY'],axis='columns')
gen_df = gen_df.drop(['PLANT_ID'], axis='columns')
gen_df["DATE_TIME"] = pd.to_datetime(gen_df["DATE_TIME"])
sensor_df["DATE_TIME"] = pd.to_datetime(sensor_df["DATE_TIME"])
df = pd.merge(sensor_df,gen_df,on="DATE_TIME",how="inner")
df.head()
x_train = df.groupby(['DATE_TIME']).sum().iloc[:-1,-2:-1]
for feature_name in x_train.columns:
    x_train[feature_name] = (x_train[feature_name] - x_train[feature_name].min())/(x_train[feature_name].max()-x_train[feature_name].min())
y_train = df.groupby(['DATE_TIME']).sum().iloc[1:,-2:-1]
for feature_name in y_train.columns:
    y_train[feature_name] = (y_train[feature_name] - y_train[feature_name].min())/(y_train[feature_name].max()-y_train[feature_name].min())
x_train.iloc[:277,:]
y_train.iloc[24:30,:]
x_train = x_train.values
y_train = y_train.values
import torch
from sklearn.model_selection import KFold
k = 10
kf = KFold(n_splits=k)
kf_data = {"train" : [],"valid" : [], "test_list" : []}
for train_index, valid_index in kf.split(x_train):
    kf_data['train'].append(torch.utils.data.TensorDataset(torch.from_numpy(x_train[train_index]),torch.from_numpy(y_train[train_index])))
    kf_data['valid'].append(torch.utils.data.TensorDataset(torch.from_numpy(x_train[valid_index]),torch.from_numpy(y_train[valid_index])))
    kf_data['test_list'].append((x_train[valid_index],y_train[valid_index]))
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_1 = 200
        hidden_2 = 200
        hidden_3 = 100
        # linear layer (1 -> hidden_1)
        self.fc1 = nn.Linear(1, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden2 -> hidden_3)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        # linear layer (n_hidden3 -> 1)
        self.fc4 = nn.Linear(hidden_3, 1)
        # dropout layer (p=0.5)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 1)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc3(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc4(x)
        return x
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for data, target in loaders['train']:
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            #clear gradient
            optimizer.zero_grad()
            ## find the loss and update the model parameters accordingly
            output = model(data.float())
            loss = torch.sqrt(criterion(output, target.float()))
            loss.backward()
            optimizer.step()
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss += loss.item()*data.size(0)

        ######################    
        # validate the model #
        ######################
        model.eval()
        for data, target in loaders['valid']:
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## update the average validation loss
            output = model(data.float())
            loss = torch.sqrt(criterion(output, target.float()))
            valid_loss += loss.item()*data.size(0)

        # calculate average losses
        train_loss = train_loss/len(loaders['train'].dataset)
        valid_loss = valid_loss/len(loaders['valid'].dataset)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss))

        ## TODO: save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Valid loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    # return trained model
    return model
batch_size = 32
for i in range(k):
    model = Net()
    if use_cuda:
        model.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)
    train_loader = torch.utils.data.DataLoader(kf_data['train'][i], batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(kf_data['valid'][i], batch_size=batch_size, shuffle=True)
    loaders = {'train' : train_loader, 'valid' : valid_loader}
    print()
    print(f'Fold {i + 1}')
    model = train(500, loaders, model, optimizer,criterion, use_cuda, 'model_fold_'+str(i+1)+'.pth')
import matplotlib.pyplot as plt
def test(loaders, model, criterion, use_cuda,print_every=15):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval()
    for batch_idx, (data, target) in enumerate(loaders['test']):
        # move to GPU
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data.float())
        # calculate the loss
        loss = torch.sqrt(criterion(output, target.float()))
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[0]
        

    print('Test RMSELoss: {:.6f}\n'.format(test_loss))
    
testdata = []
for i in range(k):
    testdata.append(torch.utils.data.TensorDataset(torch.from_numpy(kf_data['test_list'][i][0]),torch.from_numpy(kf_data['test_list'][i][1])))
    test_loader = torch.utils.data.DataLoader(testdata[i], batch_size=batch_size)
    loaders['test'] = test_loader
    model.load_state_dict(torch.load('model_fold_'+str(i+1)+'.pth'))
    print()
    print(f'Fold {i + 1}')
    test(loaders, model, criterion, use_cuda)
def prediction(input_data,model_path):
    model.load_state_dict(torch.load(model_path))
    out = []
    if use_cuda:
        model.cuda()
    model.eval()
    for batch_idx, (data, target) in enumerate(input_data):
        if use_cuda:
                data, target = data.cuda(), target.cuda()
        output = model(data.float())
        out.append(list(np.squeeze(output.data.max(1, keepdim=True)[0]).cpu().numpy()))
    result = []
    for i in out:
        for j in i:
            result.append(j)
    return result
for test_index in range(k):
    test_loader = torch.utils.data.DataLoader(testdata[test_index], batch_size=batch_size)
    predict = prediction(test_loader,'model_fold_'+str(test_index+1)+'.pth')
    
    x_plot = range(len(kf_data['test_list'][test_index][1]))

    plt.plot(x_plot, kf_data['test_list'][test_index][1], label = "DATA")
    plt.plot(x_plot, predict, label = "PREDICT")

    plt.xlabel('Time')
    plt.ylabel('Normalized Daily_yield')
    plt.title('Compared Results of Fold ' + str(test_index+1))

    plt.legend()
    plt.show()
def n_day_prediction(data,model_path,n):
    result = []
    out = []
    model.load_state_dict(torch.load(model_path))
    if use_cuda:
        model.cuda() 
    model.eval()
    for i in range(n):
        output = model(data.float())
        out.append(list(np.squeeze(output.data.max(1, keepdim=True)[0]).cpu().numpy()))
        result = []
        for i in out:
            for j in i:
                result.append(j)
        data = result[:]
        data = torch.from_numpy(np.array(data))
    return result
n = 2
first_point = torch.from_numpy(x_train[:277*n]) #3days data

for test_index in range(k):
    npred = n_day_prediction(first_point,'model_fold_'+str(test_index+1)+'.pth',n)
    x_plot = range(len(npred))

    plt.plot(x_plot, npred, label = "PREDICT")
    plt.xlabel('Time')
    
    plt.ylabel('Normalized Predicted Daily_yield')
    plt.title('Predicted in next ' + str(3*n) + ' days for Fold ' + str(test_index+1))

    plt.legend()
    plt.show()
    '''
    Total_yield = dialy_yield(prediction) + Total_yield(before)
    '''