import numpy as np
import pandas as pd
import torch
from torchvision import datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


start_time = time.time()        
train_data = pd.read_csv("../input/digit-recognizer/train.csv",dtype= np.float32)
test_data = pd.read_csv("../input/digit-recognizer/test.csv",dtype= np.float32)
Y = train_data.label.values
X = train_data.loc[:, train_data.columns != "label"].values/255
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state = 42)
Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)
X_train = torch.from_numpy(X_train)
Y_test = torch.from_numpy(Y_test).type(torch.LongTensor)
X_test = torch.from_numpy(X_test)
train_set = torch.utils.data.TensorDataset(X_train,Y_train)
test_set = torch.utils.data.TensorDataset(X_test,Y_test)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
testdata_loader = torch.utils.data.DataLoader(torch.tensor(test_data.values))
%matplotlib inline
    

dataiter = iter(train_loader)
images, labels = dataiter.next()


img = images[0]
img = img.numpy()
img = img.reshape(28,28)


fig = plt.figure(figsize = (5,5)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
        self.conv1 = nn.Conv2d(1 ,12, 3, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.conv2 = nn.Conv2d(12 ,24, 3, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=24)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(600, 120)
        self.bn3 = nn.BatchNorm1d(num_features=120)
        self.fc2 = nn.Linear(120, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias,0)
                
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn3(self.fc1(x)))
        x = self.relu(self.fc2(x))
        x = self.softmax(x)
        return x
    

torch.manual_seed(14)
model = Network()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
n_epochs = 30
from torch.autograd import Variable
steps = 0
print_every = 5
A_testLoss = []
A_trainLoss = []
A_testAcu = []
A_trainAcu = []

model.train()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    train_accuracy = 0.0
    
    for images, labels in train_loader:
        steps += 1
        optimizer.zero_grad()
        images = Variable(images.view(images.shape[0], 1, 28, 28)).to(device)
        labels = Variable(labels.type(torch.LongTensor)).to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        top_p, prediction = outputs.topk(1, dim=1)
        equals = prediction == labels.view(*prediction.shape)
        train_accuracy += torch.mean(equals.type(torch.FloatTensor))
        if steps % print_every == 0:
            test_loss = 0
            test_accuracy = 0
            model.eval()
            with torch.no_grad():
                for images, labels in test_loader:
                    images = Variable(images.view(images.shape[0], 1, 28, 28)).to(device)
                    labels = Variable(labels).to(device)
                    outputs = model(images)
                    batch_loss = criterion(outputs, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(outputs)
                    top_p, prediction = ps.topk(1, dim = 1)
                    equals = prediction == labels.view(*prediction.shape)
                    test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * test_accuracy / len(test_loader)
    train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * train_accuracy/len(train_loader)
    A_testLoss.append(test_loss)
    A_trainLoss.append(train_loss)
    A_testAcu.append(test_accuracy)
    A_trainAcu.append(train_accuracy)
    

    print('Epoch: {} \tTraining Loss: {:.6f} \tTraining Accuracy: {:.6f} %'.format(epoch, train_loss, train_accuracy))
    print('\t\tTest Loss: {:.6f} \t\tTest Accuracy: {:.6f} %'.format(test_loss,test_accuracy))

fig, ax = plt.subplots()
ax.plot(A_trainLoss, label='Training Loss')
ax.plot(A_testLoss, label='Test Loss')
plt.title("Losses")
plt.legend()
fig, ax = plt.subplots()
ax.plot(A_trainAcu, label='Training Accuracy')
ax.plot(A_testAcu, label='Test Accuracy')
plt.title("Accuracy")
plt.legend()
pds = []

with torch.no_grad():
    model.eval()
    
    for images in testdata_loader:
        images = Variable(images.view(images.shape[0], 1, 28, 28)).to(device)
        output = model(images)
        ps = torch.exp(output)
        top_p, prediction = ps.topk(1, dim=1)
        pds += prediction.cpu().numpy().tolist()

final = np.array(pds).flatten()
print(len(final))
pdss = []
submission_data = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
for i in range(0,len(pds)):
    pdss.append(str(pds[i])[1:-1])
submission_data["Label"] = pdss
submission_data.head()
submission_data.to_csv("submission_data.csv", index = False)
print("Tiempo de ejecucion:",(time.time() - start_time))