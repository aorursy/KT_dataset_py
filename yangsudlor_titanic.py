import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import torch

from sklearn.model_selection import KFold

import torch.nn as nn

import torch.nn.functional as F
traindata = "../input/normalizedtitanic/train.csv"

testdata = "../input/normalizedtitanic/test.csv"



train = pd.read_csv(traindata)

test = pd.read_csv(testdata)

submit = pd.DataFrame(test.PassengerId)
x_train = train.iloc[:,2:]

for feature_name in x_train.columns:

    x_train[feature_name] = (x_train[feature_name] - x_train[feature_name].min())/(x_train[feature_name].max()-x_train[feature_name].min())

x_train = x_train.values

y_train = train['Survived'].values



x_test = test.iloc[:,1:]

for feature_name in x_test.columns:

    x_test[feature_name] = (x_test[feature_name] - x_test[feature_name].min())/(x_test[feature_name].max()-x_test[feature_name].min())
k = 5

kf = KFold(n_splits=k)

kf_data = {"train" : [],"valid" : []}

data_scikit = {"train" : [],"valid" : []}

for train_index, valid_index in kf.split(x_train):

    kf_data['train'].append(torch.utils.data.TensorDataset(torch.from_numpy(x_train[train_index]),torch.from_numpy(y_train[train_index])))

    kf_data['valid'].append(torch.utils.data.TensorDataset(torch.from_numpy(x_train[valid_index]),torch.from_numpy(y_train[valid_index])))

    data_scikit['train'].append([x_train[train_index],y_train[train_index]])

    data_scikit['valid'].append([x_train[valid_index],y_train[valid_index]])
use_cuda = torch.cuda.is_available()



class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        hidden_1 = 200

        hidden_2 = 100

        hidden_3 = 50

        # linear layer (7 -> hidden_1)

        self.fc1 = nn.Linear(7, hidden_1)

        # linear layer (n_hidden -> hidden_2)

        self.fc2 = nn.Linear(hidden_1, hidden_2)

        # linear layer (n_hidden -> 10)

        self.fc3 = nn.Linear(hidden_2, hidden_3)

        self.fc4 = nn.Linear(hidden_3, 2)

        # dropout layer (p=0.5)

        # dropout prevents overfitting of data

        self.dropout = nn.Dropout(0.5)



    def forward(self, x):

        # flatten image input

        x = x.view(-1, 7)

        # add hidden layer, with relu activation function

        x = F.relu(self.fc1(x))

        # add dropout layer

        x = self.dropout(x)

        # add hidden layer, with relu activation function

        x = F.relu(self.fc2(x))

        # add dropout layer

        x = self.dropout(x)

        # add output layer

        x = F.relu(self.fc3(x))

        x = self.dropout(x)

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

            loss = criterion(output, target)

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

            loss = criterion(output, target)

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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    train_loader = torch.utils.data.DataLoader(kf_data['train'][i], batch_size=batch_size, shuffle=True)

    valid_loader = torch.utils.data.DataLoader(kf_data['valid'][i], batch_size=batch_size, shuffle=True)

    loaders = {'train' : train_loader, 'valid' : valid_loader}

    print()

    print(f'Fold {i + 1}')

    model = train(100, loaders, model, optimizer,criterion, use_cuda, 'model_fold_'+str(i+1)+'.pth')
import random

random.seed(30)

for i in range(k):

    c = list(zip(data_scikit['train'][i][0], data_scikit['train'][i][1]))

    random.shuffle(c)

    data_scikit['train'][i][0], data_scikit['train'][i][1] = zip(*c)
from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_recall_fscore_support as score

from sklearn.metrics import f1_score



model_scikit = GaussianNB()

print(f"Results for model {type(model_scikit).__name__}")

for i in range(k):

    model_scikit.fit(data_scikit['train'][i][0],data_scikit['train'][i][1])

    

    predictions = model_scikit.predict(data_scikit['valid'][i][0])

    

    correct = 0

    incorrect = 0

    total = 0

    for actual, predicted in zip(data_scikit['valid'][i][1], predictions):

        total += 1

        if actual == predicted:

            correct += 1

        else:

            incorrect += 1

    precision, recall, fscore, _ = score(data_scikit['valid'][i][1], predictions)

    f1 = f1_score(data_scikit['valid'][i][1], predictions)

    print(f"Fold {i+1}")

    print(f"Correct: {correct}")

    print(f"Incorrect: {incorrect}")

    print(f"Accuracy: {100 * correct / total:.2f}%")

    print(f"F1: {fscore}")

    print(f"Precision: {precision}")

    print(f"Recall: {recall}")

    print(f"F1 whole dataset: {f1}")

    print()

model_scikit = DecisionTreeClassifier()

print(f"Results for model {type(model_scikit).__name__}")

for i in range(k):

    model_scikit.fit(data_scikit['train'][i][0],data_scikit['train'][i][1])

    

    predictions = model_scikit.predict(data_scikit['valid'][i][0])

    

    correct = 0

    incorrect = 0

    total = 0

    for actual, predicted in zip(data_scikit['valid'][i][1], predictions):

        total += 1

        if actual == predicted:

            correct += 1

        else:

            incorrect += 1

    precision, recall, fscore, _ = score(data_scikit['valid'][i][1], predictions)

    f1 = f1_score(data_scikit['valid'][i][1], predictions)

    print(f"Fold {i+1}")

    print(f"Correct: {correct}")

    print(f"Incorrect: {incorrect}")

    print(f"Accuracy: {100 * correct / total:.2f}%")

    print(f"F1: {fscore}")

    print(f"Precision: {precision}")

    print(f"Recall: {recall}")

    print(f"F1 whole dataset: {f1}")

    print()
for i in range(k):

    Survived = []

    valid_loader = torch.utils.data.DataLoader(kf_data['valid'][i], batch_size=batch_size)

    model.load_state_dict(torch.load('model_fold_'+str(i+1)+'.pth'))

    model.eval()

    for batch_idx, (data, _) in enumerate(valid_loader):

        if use_cuda:

                data, _ = data.cuda(), _.cuda()

        output = model(data.float())

        Survived.append(list(np.squeeze(output.data.max(1, keepdim=True)[1]).cpu().numpy()))

    result = []

    for j in Survived:

        for l in j:

            result.append(l)

    correct = 0

    incorrect = 0

    total = 0

    for actual, predicted in zip(data_scikit['valid'][i][1], np.array(result)):

        total += 1

        if actual == predicted:

            correct += 1

        else:

            incorrect += 1

    precision, recall, fscore, _ = score(data_scikit['valid'][i][1], np.array(result))

    f1 = f1_score(data_scikit['valid'][i][1], np.array(result))

    print(f"Fold {i+1}")

    print(f"Correct: {correct}")

    print(f"Incorrect: {incorrect}")

    print(f"Accuracy: {100 * correct / total:.2f}%")

    print(f"F1: {fscore}")

    print(f"Precision: {precision}")

    print(f"Recall: {recall}")

    print(f"F1 whole dataset: {f1}")

    print()