# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import torchvision

import PIL

from imgaug import augmenters as iaa

import imgaug as ia

import os

print(os.listdir("../input"))

import torch.utils.data



# Any results you write to the current directory are saved as output.
## without loading from datasets.mnset, it looks like we may have to define a custom dataset in order to do actual image augmentation

from torchvision import transforms as TR; #ToPILImage

from sklearn.preprocessing import OneHotEncoder



class MnistDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, num_samples, transform=None):

        """

        Args:

            csv_file (string): Path to the csv file with annotations.

            root_dir (string): Directory with all the images.

            transform (callable, optional): Optional transform to be applied

                on a sample.

        """

        self.images = pd.read_csv(csv_file)[0:num_samples]

        self.labels = self.images['label'][0:num_samples];

#         onehot_encoder = OneHotEncoder(sparse = False)

#         yhot = onehot_encoder.fit_transform(self.labels.values.reshape(-1,1))

#        self.labels = yhot;

        

        self.images.drop('label', axis = 1, inplace = True)

        self.transform = transform

    def __len__(self):

        return self.images.shape[0]

    

    def __getitem__(self, idx):

        '''

        seems highly inefficient

        '''

        sample =self.images.loc[idx, :].values.reshape(28,28,1).astype('int32');

        label = self.labels.loc[idx];

        #convert to pil image

        #print(sample.shape, type(sample))

        if self.transform:

            sample = self.transform(sample)

        #convert back to torch.tensor.

        label = torch.from_numpy(np.array([label])).view(-1);

        return sample.type(torch.FloatTensor), label.type(torch.LongTensor)
## test the Mnist

## augmentation transforms #takes some times to load as csv

transforms = torchvision.transforms.Compose([

    torchvision.transforms.ToPILImage(),

    torchvision.transforms.RandomHorizontalFlip(),

    torchvision.transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),

    torchvision.transforms.ToTensor(),

])

# why does it have to be in f***** PIL image format... to do the transforms...

num_samples = 10000

mnist = MnistDataset("../input/train.csv", num_samples, transform = transforms)

## test get item...

import matplotlib.pyplot as plt

# s = train.loc[0,:].values.reshape(28,28,1);

im, label = mnist.__getitem__(0)

print(type(im))

imnum = im.numpy();

print(imnum.shape)

plt.imshow(imnum.squeeze())

plt.show()



custom_data_loader = torch.utils.data.DataLoader(mnist, batch_size=1000,

                        shuffle=True)



print(mnist.labels.shape)

c = 0;

for x, y in custom_data_loader:

    print(len(x))

    print(x[1].shape)

    print(y.shape)

    break;

## what can I do with a dataloader?
train = pd.read_csv("../input/train.csv")

print(train.columns)

y = train['label'];

train.drop('label', axis = 1, inplace = True)



from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse = False)

yhot = onehot_encoder.fit_transform(y.values.reshape(-1,1))

print(yhot.shape)

#image shape is 28 by 28



nptrain = train.values;

nptrain = nptrain/255

nptrain = np.reshape(nptrain, (42000, 1,28,28))

print(nptrain.shape)





num_examples = 10000;

BATCH_SIZE = 1000;

train_data = torch.from_numpy(nptrain[0:num_examples,:,:])

train_data = train_data.type('torch.DoubleTensor')

ytrain = torch.from_numpy(y.values[0:num_examples]);

ytrain = ytrain.type('torch.DoubleTensor')

from sklearn.model_selection import train_test_split

Xtrain, Xval, Ytrain, Yval = train_test_split(train_data.numpy(), ytrain.numpy(), test_size=0.2, random_state=42);

print(Xtrain.shape)

# X_train = torch.from_numpy(Xtrain).type('torch.DoubleTensor').cuda();

# Xval = torch.from_numpy(Xval).type('torch.DoubleTensor').cuda();

# Y_train = torch.from_numpy(Ytrain).type('torch.DoubleTensor').cuda();

# Yval = torch.from_numpy(Yval).type('torch.DoubleTensor').cuda();



torch_X_train = torch.from_numpy(Xtrain).type(torch.FloatTensor)

torch_y_train = torch.from_numpy(Ytrain).type(torch.LongTensor) # data type is long



# create feature and targets tensor for test set.

torch_X_test = torch.from_numpy(Xval).type(torch.FloatTensor)

torch_y_test = torch.from_numpy(Yval).type(torch.LongTensor) # data type is long



# Pytorch train and test sets

Train = torch.utils.data.TensorDataset(torch_X_train,torch_y_train)

Test = torch.utils.data.TensorDataset(torch_X_test,torch_y_test)



# data loader

train_loader = torch.utils.data.DataLoader(Train, batch_size = BATCH_SIZE, shuffle = False)

test_loader = torch.utils.data.DataLoader(Test, batch_size = BATCH_SIZE, shuffle = False)

print(train_loader)



c = 0;

for x, y in train_loader:

    print(x.shape)

    print(y[0:4])

import torch

from torch.autograd import Variable

import torch.nn.functional as F



class MnistCNN(torch.nn.Module):

    def __init__(self):

        super(MnistCNN, self).__init__()

        #Input channels = 3, output channels = 18

        # we only specify layers, but not for example activations

        

        #we have to be responsible for all sizing operations

        #first two arguments are in_channels (input filter size), out_channels (size of filter)

        # kernel size is number of filter stacks

        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=1)

        ## output is 26x26x3

        torch.nn.init.xavier_uniform_(self.conv1.weight)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)



        #self.d1 = torch.nn.Dropout(p = 0.1)

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size = 2, stride = 1, padding = 1);

        torch.nn.init.xavier_uniform_(self.conv2.weight)

        self.conv_bn2 = torch.nn.BatchNorm2d(32)



        self.conv3 = torch.nn.Conv2d(32,8, kernel_size = 1, stride = 1, padding = 1)

        torch.nn.init.xavier_uniform_(self.conv3.weight)



        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.d3 = torch.nn.Dropout(p = 0.5)

        ## reduces size  to 13x13x 3

        # in_features, out_features

        self.fc1 = torch.nn.Linear(np.prod(8*8*8), 64)

        self.bn1 = torch.nn.BatchNorm1d(64) #parameter here is...



        #64 input features, 10 output features for our 10 defined classes, 0-9

        self.fc2 = torch.nn.Linear(64, 10)



    def forward(self, x):

        #batch has shape (num_samples, 1, 28, 28)

        x = self.pool1(self.conv1(x));

        x = self.conv_bn2(self.conv2(x));

        x = F.relu(self.conv3(x));

        x = self.d3(self.pool(x))

        #print(x.shape)

        #Recall that the -1 infers this dimension from the other given dimension

        #print(x.shape)

        x = x.view(-1, np.prod(8*8*8))

        #print(x.shape)

        #Computes the activation of the first fully connected layer

        x = self.bn1(F.relu(self.fc1(x)))

        #Computes the second fully connected layer (activation applied later)

        x = self.fc2(x)

        

        #get softmax

        

        return(x)

    

    def evaluate(self, image_batch, batch_labels):

        image_tensor = image_batch.float()

        input = Variable(image_tensor)

        output = self.forward(input)

        index = output.data.cpu().numpy().argmax(axis = 1)

        batch_labels = batch_labels.data.cpu().numpy();

        return 1-np.count_nonzero(index-batch_labels)/len(batch_labels);        

        
# cnn = MnistCNN();

# test_data = torch.from_numpy(nptrain[0:10,:,:])

# test_data = test_data.type('torch.DoubleTensor')

# cnn.forward(test_data.float())

## establish training workflow

epochs = 200;

criterion = torch.nn.CrossEntropyLoss()
#convert to one hot

#optimizer = torch.optim.SGD(cnn.parameters(), lr = 0.1, momentum=0.4)

cnn = MnistCNN();

cnn.cuda()

cnn.zero_grad()

optimizer = torch.optim.Adam(cnn.parameters(), lr = 0.001,betas=(0.9, 0.999));

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98, last_epoch=-1)

device = torch.device('cuda:0')



for i in range(epochs):

    #for local_batch, local_labels in custom_data_loader:

    for local_batch, local_labels in custom_data_loader:

        # Transfer to GPU

        local_batch, local_labels = local_batch.to(device), local_labels.to(device)

        optimizer.zero_grad();

        yhat = cnn.forward(local_batch);

#         print(yhat.shape, local_labels.shape)

        loss = criterion(yhat, local_labels.squeeze());

        loss.backward();

        optimizer.step();

    if(i%20 == 0):

#         print('val correct: '+str(dev_correct))

        print('loss: '+str(loss))

        scheduler.step();    

# for i in range(epochs):

#     for j in range(num_mini_batches):

#         optimizer.zero_grad()

#         ybatch = Ytrain[j*minibatch_size:(j+1)*minibatch_size]

#         yhat = cnn.forward(Xtrain[j*minibatch_size:(j+1)*minibatch_size,:,:,:].cuda().float());

#         loss = criterion(yhat, ybatch.cuda().long())

#         loss.backward();

#         optimizer.step();

#     dev_correct = cnn.evaluate(Xval, Yval)

        
##evaluate model after training

def predict_image(image, model):

    image_tensor = image.float()

    image_tensor = image_tensor.unsqueeze_(0)

    input = Variable(image_tensor)

    output = model(input)

    print(output.shape)

    index = output.data.cpu().numpy().argmax()

    return index



def predict_batch(image_batch, model):

    image_tensor = image_batch.float()

    input = Variable(image_tensor)

    output = model(input)

    index = output.data.cpu().numpy().argmax(axis = 1)

    return index
index = 7

test_image = train_data[index,:,:].cuda().float()

print(predict_image(test_image, cnn))

print(ytrain[index])



test_batch = train_data[0:10, :,:].cuda().float();

preds = predict_batch(test_batch, cnn)

print(type(preds))
## generate predictions for everything in test

test = pd.read_csv("../input/test.csv")

print(test.columns)

nptest = test.values;

nptest = nptest/255.0;

print(nptest.shape)

nptest = np.reshape(nptest, (28000, 1,28,28))
#run every test image through the model

print(nptest.shape)

test_data = torch.from_numpy(nptest)

test_data = test_data.type('torch.DoubleTensor')

print(test_data.shape)
batch_size = 1000;

num_test_batches = 28000//batch_size;

predicted_labels = [];

for i in range(num_test_batches):

    #print((i*batch_size,(i+1)*batch_size))

    test_batch = test_data[i*batch_size:(i+1)*batch_size, :,:,:].cuda().float();

    #print(test_batch.shape)

    preds = predict_batch(test_batch, cnn)

    predicted_labels +=  list(preds);

    #print(len(predicted_labels))

print(len(predicted_labels))
submission_ex = pd.read_csv("../input/sample_submission.csv")

#print(submission_ex)

#predictions = pd.DataFrame(list(range(1,28001)), columns = ['ImageId'])

predictions = pd.DataFrame(predicted_labels, columns = ['Label']);

#predictions['Label'] = predicted_labels;

#print(predictions)

predictions.to_csv('predictions.csv', index_label = 'ImageId')