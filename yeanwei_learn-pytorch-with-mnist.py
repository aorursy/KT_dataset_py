import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import torch 

import torch.nn as nn

from torchvision import transforms



from torch.utils.data import Dataset, DataLoader

from torch.utils.data.sampler import SubsetRandomSampler

import torchvision.transforms as transforms
import pandas as pd

import numpy as np



from matplotlib.pyplot import imshow

import matplotlib.pylab as plt

from PIL import Image
train_on_gpu = torch.cuda.is_available()



if not train_on_gpu:

    print('Training on CPU...')

else:

    print('Training on GPU...')
dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

dataset.head(5)
plt.imshow(dataset.iloc[0][1:].values.astype(np.uint8).reshape((28, 28)), cmap='binary')

plt.title(dataset.iloc[0][0])

plt.show()
class DatasetMNIST(Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        data = self.data.iloc[index]

        

        image = data[1:].values.astype(np.uint8).reshape((28, 28))

        label = data[0]

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image, label
BATCH_SIZE = 16

VALID_SIZE = 0.2



transform_train = transforms.Compose([

    transforms.ToPILImage(),

    transforms.RandomRotation(20),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])



transform_valid = transforms.Compose([

    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(mean=(0.5,), std=(0.5,))

])
train_data = DatasetMNIST(dataset, transform=transform_train)

valid_data = DatasetMNIST(dataset, transform=transform_valid)
num_train = len(train_data)

indices = list(range(num_train)) #list 0 to num_train

np.random.shuffle(indices) #shuffle them

split = int(np.floor(VALID_SIZE * num_train)) #number of valid images

train_idx, valid_idx = indices[split:], indices[:split] #take their own parts of indices



print("Size of Train Set: ", len(train_idx))

print("Size of Valid Set: ", len(valid_idx))



train_sampler = SubsetRandomSampler(train_idx) #samples elements randomly from a given list of indices

valid_sampler = SubsetRandomSampler(valid_idx)



train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)

valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, sampler=valid_sampler)
fig, axis = plt.subplots(2, 5, figsize=(10, 8))

images, labels = next(iter(train_loader))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        image, label = images[i], labels[i]

        # image shape now is (1,28,28) for tensor

        # use .view(28,28) to shape it to (28,28) in order to show it

        ax.imshow(transforms.ToPILImage()(image), cmap='binary')

        ax.set(title = f"{label}")
class CNN_model(nn.Module):

    def __init__(self):

        super(CNN_model, self).__init__()

        

        self.conv1 = nn.Sequential(

            nn.Conv2d(1, 16, 3, padding=1),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.Conv2d(16, 16, 3, padding=1),

            nn.BatchNorm2d(16),

            nn.ReLU(),

            nn.MaxPool2d(2,2)

        )

        

        self.conv2 = nn.Sequential(

            nn.Conv2d(16, 32, 3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),

            nn.BatchNorm2d(32),

            nn.ReLU(),

            nn.MaxPool2d(2,2)

        )

        

        self.conv3 = nn.Sequential(

            nn.Conv2d(32, 64, 3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),

            nn.BatchNorm2d(64),

            nn.ReLU()

        )

        

        self.fc = nn.Linear(64*7*7, 10)

        self.bn_fc = nn.BatchNorm1d(10)



    def forward(self, x):

        out = self.conv1(x)

        out = self.conv2(out)

        out = self.conv3(out)

        out = out.view(out.size(0), -1)

        out = self.fc(out)

        out = self.bn_fc(out)

        

        return out

    

model = CNN_model()

model
if train_on_gpu:

    model = model.cuda()

    print("Train on GPU")

else:

    print("Train on CPU")
criterion = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
n_epochs=20



valid_loss_min = np.Inf #min loss for saving best model

train_losses, valid_losses = [], []

accuracy_list=[]



N_valid = len(valid_idx)
torch.cuda.empty_cache()
for epoch in range(n_epochs):

    train_loss = 0

    

    # set to train whenever needs to train

    model.train()



    for x, y in train_loader:

        if train_on_gpu:

            x, y = x.cuda(), y.cuda()

        

        optimizer.zero_grad()

        z = model(x)

        loss = criterion(z, y)

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        

    valid_loss = 0

    correct = 0

    accuracy = 0

    

    with torch.no_grad():

    # set to eval to evaluate the performance of current model

        model.eval()



        for x_test, y_test in valid_loader:

            if train_on_gpu:

                x_test, y_test = x_test.cuda(), y_test.cuda()



            z = model(x_test)

            valid_loss += criterion(z, y_test).item()

            # get the class of predicted value

            _, yhat = torch.max(z.data, 1)

            # count the results that have been predicted correctly

            correct += (yhat == y_test).sum().item() 

        

    accuracy = correct / N_valid

    train_loss = train_loss / len(train_loader)

    valid_loss = valid_loss / len(valid_loader)

    accuracy_list.append(accuracy)

    train_losses.append(train_loss)

    valid_losses.append(valid_loss)

    

    print(f"Epoch: {epoch}/{n_epochs} ",

          f"Training Loss: {train_loss:.4f} ",

          f"Validation Loss: {valid_loss:.4f} ",

          f"Test Accuracy: {accuracy:.4f}")

    

    #check is this model perform better in valid loss

    network_learned = valid_loss < valid_loss_min

    

    if network_learned:

        valid_loss_min = valid_loss

        

        checkpoint = {'model': CNN_model(),

                      'state_dict': model.state_dict(),

                      'optimizer' : optimizer.state_dict()}



        torch.save(checkpoint, 'model_mnist.pth')

        print("Saved this model!")

        

    torch.cuda.empty_cache()
plt.plot(train_losses, label='Training Loss')

plt.plot(valid_losses, label='Validation Loss')

plt.legend(frameon=False)

plt.show()
plt.plot(accuracy_list, label='Validation Accuracy')

plt.legend(frameon=False)

plt.show()
checkpoint = torch.load('model_mnist.pth')

model.load_state_dict(checkpoint['state_dict'])

optimizer.load_state_dict(checkpoint['optimizer'])
print(model)
test_dataset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

test_dataset.head(5)
class TestDatasetMNIST(Dataset):

    def __init__(self, data, transform=None):

        self.data = data

        self.transform = transform

    

    def __len__(self):

        return len(self.data)

    

    def __getitem__(self, index):

        image = self.data.iloc[index].values.astype(np.uint8).reshape((28, 28))

        

        if self.transform is not None:

            image = self.transform(image)

            

        return image
test_data = TestDatasetMNIST(test_dataset, transform=transform_valid)

test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
fig, axis = plt.subplots(2, 5, figsize=(10, 8))

images =  next(iter(test_loader))



for i, ax in enumerate(axis.flat):

    with torch.no_grad():

        model.eval()

        img = images[i]

        img_test = img.unsqueeze(0)



        if train_on_gpu:

            img_test = img_test.cuda()

            

        z = model(img_test)

        _, yhat = torch.max(z.data, 1)

        ax.imshow(img.squeeze(0), cmap='binary')

        ax.set(title = f"{yhat.item()}")
submissions = [['ImageId', 'Label']]

img_id = 1



with torch.no_grad():

    model.eval()



    for x in test_loader:

        if train_on_gpu:

            x = x.cuda()

        

        z = model(x)

        _, yhat = torch.max(z.data, 1)

        

        for pred in yhat:

            submissions.append([img_id, pred.item()])

            img_id += 1

            

print(f"Submission size: {len(submissions)-1}")

print(f"Test data size: {len(test_dataset)}")
import csv



with open('submission.csv', 'w') as submissionFile:

    writer = csv.writer(submissionFile)

    writer.writerows(submissions)

    

print('Submission Complete!')