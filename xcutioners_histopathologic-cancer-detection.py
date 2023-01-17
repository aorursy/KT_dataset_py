# Libraries

import os

import numpy as np

import pandas as pd

import cv2

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split



import torch 

import torch.nn as nn

import torch.nn.functional as F

import torchvision

import torchvision.transforms as transforms

from torch.utils.data import TensorDataset, DataLoader, Dataset
## Parameters for model



# Hyper parameters

num_epochs = 24

num_classes = 2

batch_size = 128

learning_rate = 0.001



# Device configuration

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
labels = pd.read_csv('../input/train_labels.csv')

sub = pd.read_csv('../input/sample_submission.csv')

train_path = '../input/train/'

test_path = '../input/test/'
#Splitting data into train and val

train, val = train_test_split(labels, stratify=labels.label, test_size=0.1)

len(train), len(val)
class MyDataset(Dataset):

    def __init__(self, df_data, data_dir = './', transform=None):

        super().__init__()

        self.df = df_data.values

        self.data_dir = data_dir

        self.transform = transform



    def __len__(self):

        return len(self.df)

    

    def __getitem__(self, index):

        img_name,label = self.df[index]

        img_path = os.path.join(self.data_dir, img_name+'.tif')

        image = cv2.imread(img_path)

        if self.transform is not None:

            image = self.transform(image)

        return image, label
trans_train = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(64, padding_mode='reflect'),

                                  transforms.RandomHorizontalFlip(), 

                                  transforms.RandomVerticalFlip(),

                                  #transforms.RandomCrop(),

                                  transforms.RandomRotation(20),

                                  

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



trans_valid = transforms.Compose([transforms.ToPILImage(),

                                  transforms.Pad(64, padding_mode='reflect'),

                                  transforms.ToTensor(),

                                  transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])])



dataset_train = MyDataset(df_data=train, data_dir=train_path, transform=trans_train)

dataset_valid = MyDataset(df_data=val, data_dir=train_path, transform=trans_valid)



loader_train = DataLoader(dataset = dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)

loader_valid = DataLoader(dataset = dataset_valid, batch_size=batch_size//2, shuffle=False, num_workers=0)
len(loader_valid)

class SimpleCNN(nn.Module):

    def __init__(self):

        # ancestor constructor call

        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=2)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=2)

        self.bn1 = nn.BatchNorm2d(32)

        self.bn2 = nn.BatchNorm2d(64)

        self.bn3 = nn.BatchNorm2d(128)

        self.bn4 = nn.BatchNorm2d(256)

        self.bn5 = nn.BatchNorm2d(512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avg = nn.AvgPool2d(8)

        self.fc = nn.Linear(512 * 1 * 1, 2) # !!!

    def forward(self, x):

        x = self.pool(F.leaky_relu(self.bn1(self.conv1(x)))) # first convolutional layer then batchnorm, then activation then pooling layer.

        x = self.pool(F.leaky_relu(self.bn2(self.conv2(x))))

        x = self.pool(F.leaky_relu(self.bn3(self.conv3(x))))

        x = self.pool(F.leaky_relu(self.bn4(self.conv4(x))))

        x = self.pool(F.leaky_relu(self.bn5(self.conv5(x))))

        x = self.avg(x)

        #print(x.shape) # lifehack to find out the correct dimension for the Linear Layer

        x = x.view(-1, 512 * 1 * 1) # !!!

        x = self.fc(x)

        return x
model = SimpleCNN().to(device)
# Loss and optimizer

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
%%time

train_losses, test_losses = [],[]

for epoch in range(num_epochs):

    running_loss = 0

    for images, labels in loader_train:

        images = images.to(device)

        labels = labels.to(device)

        

        optimizer.zero_grad()

        

        log_ps = model(images)

        loss = criterion(log_ps, labels)

        loss.backward()

        optimizer.step()

        

        running_loss += loss.item()

        

    else:

        ## TODO: Implement the validation pass and print out the validation accuracy

        test_loss = 0

        accuracy = 0

        # Test the model

        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

        with torch.no_grad():

            for images, labels in loader_valid:

                images = images.to(device)

                labels = labels.to(device)

                

                log_ps = model(images)

                test_loss += criterion(log_ps, labels)

                

                ps = torch.exp(log_ps)

                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor))

                

                                

        train_losses.append(running_loss/len(loader_train))

        test_losses.append(test_loss/len(loader_valid))

        

        print("Epoch: {}/{}..".format(epoch+1, num_epochs),

             "Training Loss: {:.3f}..".format(running_loss/len(loader_train)),

             "Test Loss: {:3f}.. ".format(test_loss/len(loader_valid)),

             "Test Accuracy: {:.3f}".format(accuracy/len(loader_valid)))

        

        torch.save(model.state_dict(), 'model1.ckpt')
%matplotlib inline

%config InlineBackend.figure_format = 'retina'



import matplotlib.pyplot as plt
plt.plot(train_losses, label='Training loss')

plt.plot(test_losses, label="validation loss")

plt.legend(frameon=False)
torch.save(model.state_dict(), 'model1.ckpt')
dataset_valid = MyDataset(df_data=sub, data_dir=test_path, transform=trans_valid)

loader_test = DataLoader(dataset = dataset_valid, batch_size=32, shuffle=False, num_workers=0)
model.eval()



preds = []

for batch_i, (data, target) in enumerate(loader_test):

    data, target = data.cuda(), target.cuda()

    output = model(data)



    pr = output[:,1].detach().cpu().numpy()

    for i in pr:

        preds.append(i)

sub.shape, len(preds)

sub['label'] = preds

sub.to_csv('s.csv', index=False)
sub.head()
# Train the model

#total_step = len(loader_train)

#for epoch in range(num_epochs):

#    for i, (images, labels) in enumerate(loader_train):

##        images = images.to(device)

 #       labels = labels.to(device)

 #       

#        # Forward pass

#        outputs = model(images)

#        loss = criterion(outputs, labels)

#        

#        # Backward and optimize

#        optimizer.zero_grad()

#        loss.backward()

#        optimizer.step()

#        

#        if (i+1) % 100 == 0:

#            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 

#                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
# Test the model

#model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)

#with torch.no_grad():

#    correct = 0

#    total = 0

#    for images, labels in loader_valid:

#        images = images.to(device)

#        labels = labels.to(device)

#        outputs = model(images)

#        _, predicted = torch.max(outputs.data, 1)

#        total += labels.size(0)

#        correct += (predicted == labels).sum().item()

#          

#    print('Test Accuracy of the model on the 22003 test images: {} %'.format(100 * correct / total))



# Save the model checkpoint

#torch.save(model.state_dict(), 'model.ckpt')
# import the modules we'll need

from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "s.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

#df = pd.DataFrame(np.random.randn(50, 4), columns=list('ABCD'))



# create a link to download the dataframe

create_download_link(sub)
my_submission = pd.DataFrame(sub)

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)