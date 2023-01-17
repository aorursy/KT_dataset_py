import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os
traindata = '../input/super-ai-image-classification/train/train/train.csv'
testdata = '../input/super-ai-image-classification/val/val/val.csv'
train = pd.read_csv(traindata)

first_class = ['930195df-7e46-4ca6-91a0-2764f44e13c6.jpg',
'5270bca6-2716-4f72-9601-004fba1c536e.jpg',
'c0894ee2-a50a-4eea-aa01-6ab833017f1d.jpg',
'39c1d8dd-5ee0-46f5-9422-45444db462c6.jpg',
'd20bcabd-ca32-4e0d-929e-4f1e8f942e27.jpg',
'7eb4e7cf-e503-4b85-9ede-7f01993d2f84.jpg',
'dec754a3-04d2-44bb-9c1e-d2306be703fd.jpg',
'e88bda06-4b3c-409c-90fc-56cbe973441e.jpg',
'2494d994-b1f7-4c0e-ba4e-b6b213076e4e.jpg',
'73565495-0aeb-42a3-9e29-17841fd75a36.jpg',
'86181240-1ff5-456d-9ef2-0e79be13d63b.jpg',
'd17010da-c97b-4e5b-9230-240f570ceb9c.jpg',
'd16da93b-558d-4264-8115-f06f37d41338.jpg',
'6e8eb027-d722-4a93-8d01-5003b1575bb4.jpg',
'516f4d0c-8487-48d8-9ad8-acb1d13e71a9.jpg',
'ddd8b119-43c0-4488-983e-f8d28835e453.jpg',
'3f6cad70-f68e-434a-9eaf-7328d1b422bd.jpg']

augment = transforms.Compose([transforms.RandomRotation(30),
                              transforms.RandomResizedCrop(224),
                              transforms.RandomHorizontalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

transform = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

imgs = []
labels = []
for i in range(len(train)):
    image = Image.open('../input/super-ai-image-classification/train/train/images/' + train['id'][i])
    if train['id'][i] in first_class:
        print(i)
    image_t = augment(image)
    imgs.append(image_t)
    labels.append(train['category'][i])
labels = np.array(labels)
imgs = torch.stack(imgs)
holdout = int(0 * len(imgs))
#x_valid = imgs[381:781]
#y_valid = torch.from_numpy(labels[381:781])

#x_train = []
#y_train = []
#for i in range(len(imgs)):
#    if i >= 381 and i < 781:
#        pass
#    else:
#        x_train.append(imgs[i])
#        y_train.append(labels[i])
#y_train = torch.from_numpy(np.array(y_train))
#x_train = torch.stack(x_train)
x_train = imgs[holdout:]
y_train = torch.from_numpy(labels[holdout:])

train_data = torch.utils.data.TensorDataset(x_train,y_train)
#valid_data = torch.utils.data.TensorDataset(x_valid,y_valid)
y_train
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
loaders = {'train' : trainloader}
trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=32, shuffle=True)
loaders = {'train' : trainloader, 'valid' : validloader}
def train(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    """returns trained model"""
    global valid_loss_save
    global train_loss_save
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
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            train_loss += loss.item()*data.size(0)
            

        ######################    
        # validate the model #
        ######################
#         model.eval()
#         for data, target in loaders['valid']:
#             # move to GPU
#             if use_cuda:
#                 data, target = data.cuda(), target.cuda()
#             ## update the average validation loss
#             output = model(data)
#             loss = criterion(output, target)
#             valid_loss += loss.item()*data.size(0)
            
        # calculate average losses
        train_loss = train_loss/len(loaders['train'].dataset)
#         valid_loss = valid_loss/len(loaders['valid'].dataset)
        
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch,
            train_loss,
            valid_loss))
        train_loss_save.append(train_loss)
        valid_loss_save.append(valid_loss)
        torch.save(model.state_dict(), save_path)
        
        ## TODO: save the model if validation loss has decreased
#         if valid_loss <= valid_loss_min:
#             print('Valid loss decreased ({:.6f} --> {:.6f}). Saving model ...'.format(
#             valid_loss_min,
#             valid_loss))
#             torch.save(model.state_dict(), save_path)
#             valid_loss_min = valid_loss
    # return trained model

    return model
import torchvision.models as models
import torch.nn as nn

model_transfer = models.vgg16(pretrained=True)

for param in model_transfer.parameters():
    param.requires_grad = False
    
classifier = nn.Sequential(nn.Linear(25088,4096),
                           nn.ReLU(),
                           nn.Dropout(p=0.25),
                           nn.Linear(4096,4096),
                           nn.ReLU(),
                           nn.Dropout(p=0.25),
                           nn.Linear(4096,2))

    
model_transfer.classifier = classifier

use_cuda = torch.cuda.is_available() 

if use_cuda:
   model_transfer = model_transfer.cuda()
import torch.optim as optim
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.Adam(model_transfer.classifier.parameters(), lr=0.0001)
valid_loss_save = []
train_loss_save = []
model_transfer = train(10, loaders, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
model_transfer
import torch.nn as nn
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    def __init__(self):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv8 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 2)
        #dropout
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = self.pool(F.relu(self.conv8(x)))
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv8(x))
        x = self.pool(F.relu(self.conv8(x)))
        #flatten
        x = x.view(-1, 25088)
        #Ly
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

model = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model.cuda()
model
import torch.optim as optim

### TODO: select loss function
criterion = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

valid_loss_save = []
train_loss_save = []
model = train(50, loaders, model, optimizer, 
                      criterion, use_cuda, 'model6.pt')
model.load_state_dict(torch.load('model6.pt'))
x_plot = range(len(valid_loss_save))

plt.plot(x_plot, valid_loss_save, label = "Valid")
plt.plot(x_plot[5:], train_loss_save[5:], label = "Train")

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
    for batch_idx, (data, target) in enumerate(loaders['train']):
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
test(loaders, model, criterion, use_cuda)
test_dir = os.listdir('../input/super-ai-image-classification/val/val/images/')
test = pd.read_csv(testdata)
test_imgs = []
test_imgs_id = []
test_labels = []
for images in test_dir:
    image = Image.open('../input/super-ai-image-classification/val/val/images/' + images)
    test_imgs_id.append(images)
    image_t = transform(image)
    test_imgs.append(image_t)
test_imgs = torch.stack(test_imgs)
test_data = torch.utils.data.TensorDataset(test_imgs,test_imgs)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
model.load_state_dict(torch.load('model6.pt'))
model.eval()
for data ,_ in test_loader:
    if use_cuda:
        data = data.cuda()
    output = model(data)
    test_labels.append(list(np.squeeze(output.data.max(1, keepdim=True)[1]).cpu().numpy()))
result = []
for i in test_labels:
    for j in i:
        result.append(j)
submit = pd.DataFrame(test_imgs_id,columns =['id'])
submit['category'] = result
predict = []
for i in range(len(submit)):
    if submit['category'][i] == 1:
        predict.append(submit['id'][i])
len(predict)
submit.to_csv('superAI_Submit_vgg16_architec30e.csv',index=False)
from matplotlib.pyplot import imshow
image = Image.open('../input/super-ai-image-classification/val/val/images/' + predict[20])
imshow(np.asarray(image))