import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
traindata = '../input/thai-mnist-classification/mnist.train.map.csv'
train = pd.read_csv(traindata)

augment = transforms.Compose([transforms.RandomRotation(30),
                              transforms.Resize(256),
                              transforms.CenterCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])

transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
imgs = []
labels = []
x_test = []
y_test = []
for i in range(len(train)):
    image = Image.open('../input/thai-mnist-classification/train/' + train['id'][i]).convert('RGB')
    image_t = augment(image)
    imgs.append(image_t)
    labels.append(train['category'][i])            
labels = np.array(labels)
imgs = torch.stack(imgs)
holdout = int(0.2 * len(imgs))
x_valid = imgs[:holdout]
y_valid = labels[:holdout]

x_train = imgs[holdout:]
y_train = torch.from_numpy(labels[holdout:])

train_data = torch.utils.data.TensorDataset(x_train,y_train)
valid_data = torch.utils.data.TensorDataset(x_valid[:len(x_valid)//2],torch.from_numpy(y_valid[:len(x_valid)//2]))
test_data = torch.utils.data.TensorDataset(x_valid[len(x_valid)//2:],torch.from_numpy(y_valid[len(x_valid)//2:]))
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)
loaders = {'train' : trainloader, 'valid' : validloader, 'test': testloader}
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    global valid_loss_save
    global train_loss_save
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
        train_loss_save.append(train_loss)
        valid_loss_save.append(valid_loss)
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss))
        
        #save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Valid loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), save_path)
            valid_loss_min = valid_loss
    # return trained model
    return model
import torchvision.models as models
import torch.nn as nn

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
    
fc = nn.Sequential(nn.Linear(25088,4096),
                           nn.ReLU(),
                           nn.Dropout(p=0.25),
                           nn.Linear(4096,4096),
                           nn.ReLU(),
                           nn.Dropout(p=0.25),
                           nn.Linear(4096,10))
                   

model.classifier = fc

use_cuda = torch.cuda.is_available() 

if use_cuda:
    model = model.cuda()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.000001)
valid_loss_save = []
train_loss_save = []
model = train(80, loaders, model, optimizer, 
                      criterion, use_cuda, 'model.pth')
x_plot = range(len(valid_loss_save))

plt.plot(x_plot, valid_loss_save, label = "Valid")
plt.plot(x_plot[:], train_loss_save[:], label = "Train")

plt.xlabel('Epoch')
plt.ylabel('loss')
plt.title('Compared Loss')

plt.legend()
plt.show()
def test(loaders, model, criterion, use_cuda):

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
        loss = criterion(output, target)
        # update average test loss 
        test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
        # convert output probabilities to predicted class
        pred = output.data.max(1, keepdim=True)[1]
        # compare predictions to true label
        correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
        total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))
model.load_state_dict(torch.load('model.pth'))
test(loaders, model, criterion, use_cuda)
train_rulesdata = '../input/thai-mnist-classification/train.rules.csv'
train_rules = pd.read_csv(train_rulesdata)
# image = []
# image.append(transform(Image.open('../input/thai-mnist-classification/train/' + train_rules['feature2'][0]).convert('RGB')))
# image = torch.stack(image)
# model.eval()
# if use_cuda:
#     image = image.cuda()
# output = model(image)
# _, predicted = torch.max(output, 1)
# int(predicted[0].cpu().numpy())
train_rules.groupby(['feature1']).max()
model.eval()
for i in range(len(train_rules)):
    try:
        image = []
        image.append(transform(Image.open('../input/thai-mnist-classification/train/' + train_rules['feature1'][i]).convert('RGB')))
        image = torch.stack(image)
        if use_cuda:
            image = image.cuda()
        output = model(image)
        _, predicted = torch.max(output, 1)
        train_rules['feature1'][i] = int(predicted[0].cpu().numpy())
    except:
        train_rules['feature1'][i] = -1
        
    image = []
    image.append(transform(Image.open('../input/thai-mnist-classification/train/' + train_rules['feature2'][i]).convert('RGB')))
    image = torch.stack(image)
    if use_cuda:
        image = image.cuda()
    output = model(image)
    _, predicted = torch.max(output, 1)
    train_rules['feature2'][i] = int(predicted[0].cpu().numpy())
        
    image = []
    image.append(transform(Image.open('../input/thai-mnist-classification/train/' + train_rules['feature3'][i]).convert('RGB')))
    image = torch.stack(image)
    if use_cuda:
        image = image.cuda()
    output = model(image)
    _, predicted = torch.max(output, 1)
    train_rules['feature3'][i] = int(predicted[0].cpu().numpy())
x_train_rules = train_rules.iloc[:,1:-1]
x_train_rules = x_train_rules.values
y_train_rules = train_rules['predict'].values
x_train_rules = x_train_rules.astype(float)
x_train_rules
from sklearn.model_selection import KFold
k = 5
kf = KFold(n_splits=k)
kf_data = {"train" : [],"valid" : []}
data_scikit = {"train" : [],"valid" : []}
for train_index, valid_index in kf.split(x_train_rules):
    kf_data['train'].append(torch.utils.data.TensorDataset(torch.from_numpy(x_train_rules[train_index]),torch.from_numpy(y_train_rules[train_index])))
    kf_data['valid'].append(torch.utils.data.TensorDataset(torch.from_numpy(x_train_rules[valid_index]),torch.from_numpy(y_train_rules[valid_index])))
    data_scikit['train'].append([x_train_rules[train_index],y_train_rules[train_index]])
    data_scikit['valid'].append([x_train_rules[valid_index],y_train_rules[valid_index]])
import random
random.seed(30)
for i in range(k):
    c = list(zip(data_scikit['train'][i][0], data_scikit['train'][i][1]))
    random.shuffle(c)
    data_scikit['train'][i][0], data_scikit['train'][i][1] = zip(*c)
from sklearn import svm
model_scikit =  svm.SVC()
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
    print(f"Fold {i+1}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Accuracy: {100 * correct / total:.2f}%")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        hidden_1 = 500
        hidden_2 = 500
        hidden_3 = 500
        hidden_4 = 500
        # linear layer (3 -> hidden_1)
        self.fc1 = nn.Linear(3, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, hidden_3)
        self.fc4 = nn.Linear(hidden_3, hidden_4)
        self.fc5 = nn.Linear(hidden_4, 99)
        # dropout layer
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # flatten image input
        x = x.view(-1, 3)
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
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x
batch_size = 32
for i in range(k):
    model = Net()
    if use_cuda:
        model.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train_loader = torch.utils.data.DataLoader(kf_data['train'][i], batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(kf_data['valid'][i], batch_size=batch_size, shuffle=True)
    loaders = {'train' : train_loader, 'valid' : valid_loader, 'test' : valid_loader}
    print()
    print(f'Fold {i + 1}')
    valid_loss_save = []
    train_loss_save = []
    model = train(100, loaders, model, optimizer,criterion, use_cuda, 'model_fold_'+str(i+1)+'.pth')
    x_plot = range(len(valid_loss_save))

    plt.plot(x_plot, valid_loss_save, label = "Valid")
    plt.plot(x_plot[:], train_loss_save[:], label = "Train")

    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title('Compared Loss')

    plt.legend()
    plt.show()
for i in range(k):
    model.load_state_dict(torch.load('model_fold_'+str(i+1)+'.pth'))
    if use_cuda:
        model.cuda()
    test(loaders, model, criterion, use_cuda)
import math
def solve(f1,f2,f3):
    if(f1 == 0):
        return f2*f3
    elif(f1 == 1):
        return abs(f2-f3)
    elif(f1 == 2):
        return (f2 + f3)*abs(f2 - f3)
    elif(f1 == 3):
        return ((f2**2)+1)*(f2) +(f3)*(f3+1)
    elif(f1 == 4):
        return 50 + (f2 - f3)
    elif(f1 == 5):
        return min([f2 , f3])
    elif(f1 == 6):
        return max([f2 , f3])
    elif(f1 == 7):
        return math.floor(((f2*f3)/9)*11)
    elif(f1 == 8):
        return math.floor((f3*(f3 +1) - f2*(f2-1))/2)
    elif(f1 == 9):
        return 50 + f2
    else:
        return f2 + f3
test_rulesdata = '../input/thai-mnist-classification/test.rules.csv'
test_rules = pd.read_csv(test_rulesdata)
model.eval()
for i in range(len(test_rules)):
    try:
        image = []
        image.append(transform(Image.open('../input/thai-mnist-classification/test/' + test_rules['feature1'][i]).convert('RGB')))
        image = torch.stack(image)
        if use_cuda:
            image = image.cuda()
        output = model(image)
        _, predicted = torch.max(output, 1)
        test_rules['feature1'][i] = int(predicted[0].cpu().numpy())
    except:
        test_rules['feature1'][i] = -1
        
    image = []
    image.append(transform(Image.open('../input/thai-mnist-classification/test/' + test_rules['feature2'][i]).convert('RGB')))
    image = torch.stack(image)
    if use_cuda:
        image = image.cuda()
    output = model(image)
    _, predicted = torch.max(output, 1)
    test_rules['feature2'][i] = int(predicted[0].cpu().numpy())
        
    image = []
    image.append(transform(Image.open('../input/thai-mnist-classification/test/' + test_rules['feature3'][i]).convert('RGB')))
    image = torch.stack(image)
    if use_cuda:
        image = image.cuda()
    output = model(image)
    _, predicted = torch.max(output, 1)
    test_rules['feature3'][i] = int(predicted[0].cpu().numpy())
x_test_rules = test_rules.iloc[:,1:-1]
x_test_rules = x_test_rules.values
submit = pd.DataFrame()
submit['id'] = test_rules.id.values
result = []
for i in range(len(submit)):
    result.append(solve(x_test_rules[i][0],x_test_rules[i][1],x_test_rules[i][2]))
submit['predict']= np.array(result)
submit.to_csv('submit.csv',index=False)