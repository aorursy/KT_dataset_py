# importing the libraries
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
import os
import scipy.io
import math

from sklearn.utils import shuffle

from PIL import Image
import requests
from io import BytesIO

import pandas as pd
!ls '/kaggle/input/fashion-product-images-dataset/fashion-dataset'
styles = pd.read_csv('/kaggle/input/fashion-product-images-dataset/fashion-dataset/styles.csv', error_bad_lines=False)

shirts = styles[styles['articleType'].isin(['Shirts'])]
tshirts = styles[styles['articleType'].isin(['Tshirts'])]
pants =  styles[styles['articleType'].isin(['Track Pants','Shorts', 'Trunk', 'Trousers', 'Track Pants', 'Tights', 'Lounge Pants', 'Lounge Shorts', 'Leggings', 'Jeans', 'Jeggings'])]
# np.unique(styles['articleType'])
shirts, tshirts, pants = shirts['id'].to_numpy(), tshirts['id'].to_numpy(), pants['id'].to_numpy()
shirts.shape, tshirts.shape, pants.shape
image_path = '/kaggle/input/fashion-product-images-dataset/fashion-dataset/images/'
IMG_SIZE = 128
LIMIT_IMAGES = 2400
NUM_OUTPUTS = 3
shirt_images = []
for shirt in shirts[:LIMIT_IMAGES]:
    img = cv2.imread(f'{image_path}{shirt}.jpg')
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32')
    img = img / 255.0
    shirt_images.append(img)
shirt_images = np.array(shirt_images)
shirt_images.shape
tshirt_images = []
for tshirt in tshirts[:LIMIT_IMAGES]:
    img = cv2.imread(f'{image_path}{tshirt}.jpg')
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32')
    img = img / 255.0
    tshirt_images.append(img)
tshirt_images = np.array(tshirt_images)
tshirt_images.shape
pant_images = []
for pant in pants[:LIMIT_IMAGES]:
    img = cv2.imread(f'{image_path}{pant}.jpg')
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype('float32')
    img = img / 255.0
    pant_images.append(img)
pant_images = np.array(pant_images)
pant_images.shape
X = np.concatenate((shirt_images, tshirt_images, pant_images), axis = 0)
# Y = np.concatenate((np.repeat(np.array((1,0,0)), shirt_images.shape[0]), np.repeat(np.array((1,0,0)), tshirt_images.shape[0]), np.repeat(np.array((1,0,0)), pant_images.shape[0])))
Y = np.repeat([0, 1, 2], [shirt_images.shape[0], tshirt_images.shape[0], pant_images.shape[0]], axis=0)
# X = np.concatenate((shirt_images, pant_images), axis = 0)
# Y = np.concatenate((np.repeat(0, shirt_images.shape[0]), np.repeat(2, pant_images.shape[0])))
X.shape, Y.shape
X, Y = shuffle(X, Y)
X.shape, Y.shape
trainX, test_x, trainY, test_y = train_test_split(X, Y, test_size = 0.1)
train_x, val_x, train_y, val_y = train_test_split(trainX, trainY, test_size = 0.2)
(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

# train_x = train_x.reshape(train_x.shape[0], train_x.shape[3], IMG_SIZE, IMG_SIZE)
train_x  = torch.from_numpy(train_x)
train_x = train_x.permute(0,3,1,2)
train_y = train_y.astype(int);
train_y = torch.from_numpy(train_y)

# val_x = val_x.reshape(val_x.shape[0], val_x.shape[3], IMG_SIZE, IMG_SIZE)
val_x  = torch.from_numpy(val_x)
val_x = val_x.permute(0,3,1,2)
val_y = val_y.astype(int);
val_y = torch.from_numpy(val_y)
train_x.shape, train_y.shape, val_x.shape, val_y.shape

# test_x = test_x.reshape(test_x.shape[0], test_x.shape[3], IMG_SIZE, IMG_SIZE)
test_x  = torch.from_numpy(test_x)
test_x = test_x.permute(0,3,1,2)
test_y = test_y.astype(int);
test_y = torch.from_numpy(test_y)
train_x.shape, train_y.shape, val_x.shape, val_y.shape, test_x.shape, test_y.shape
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = Sequential(
            Linear(int(IMG_SIZE * IMG_SIZE / 4), NUM_OUTPUTS)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        x = self.linear_layers(x)
        return x
model = Net()
optimizer = Adam(model.parameters(), lr=0.07)
criterion = CrossEntropyLoss()
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()
    
print(model)
def train(epoch):
    model.train()
    tr_loss = 0
    # getting the training set
    X_train, Y_train = Variable(train_x), Variable(train_y)
    # getting the validation set
    X_val, Y_val = Variable(val_x), Variable(val_y)
    # converting the data into GPU format
    if torch.cuda.is_available():
        X_train = X_train.cuda()
        Y_train = Y_train.cuda()
        X_val = X_val.cuda()
        Y_val = Y_val.cuda()

    # clearing the Gradients of the model parameters
    optimizer.zero_grad()
    
    # prediction for training and validation set
    output_train = model(X_train)
    output_val = model(X_val)

    # computing the training and validation loss
    loss_train = criterion(output_train, Y_train)
    loss_val = criterion(output_val, Y_val)
    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())

    # computing the updated weights of all the model parameters
    loss_train.backward()
    optimizer.step()
    tr_loss = loss_train.item()
    if epoch%2 == 0:
        # printing the validation loss
        print('Epoch : ',epoch+1, '\t', 'loss :', loss_val)
n_epochs = 40
generation_num = 0
train_losses = []
val_losses = []
best_loss = 100000000000
best_model = None
generation_num += 1
for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    train(epoch)
    if best_loss > val_losses[-1]:
        best_loss = val_losses[-1]
        best_model = model.state_dict()

plt.plot(train_losses, label='Training loss')
plt.plot(val_losses, label='Validation loss')
plt.legend()
plt.show()
# prediction for training set
with torch.no_grad():
  if(torch.cuda.is_available()):
    output = model(train_x.cuda())
  else:
    output = model(train_x)
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on training set
train_sample_accuracy = accuracy_score(train_y, predictions)
print("Accuracy on In-Sample Training Set: %s"% train_sample_accuracy)
# prediction for validation set
with torch.no_grad():
  if(torch.cuda.is_available()):
    output = model(val_x.cuda())
  else:
    output = model(val_x)
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on validation set
val_sample_accuracy = accuracy_score(val_y, predictions)
print("Accuracy on In-Sample Validation Set: %s"% val_sample_accuracy)

# prediction for out-of-sample set
with torch.no_grad():
  if(torch.cuda.is_available()):
    output = model(test_x.cuda())
  else:
    output = model(test_x)
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# accuracy on out-of-sample set
out_sample_accuracy = accuracy_score(test_y, predictions)
print("Accuracy on Out-Of-Sample Validation (Test) Set: %s"% out_sample_accuracy)
# torch.save(model.state_dict(), f'model_{generation_num}.h5')
torch.save(model.state_dict(), f'/kaggle/working/model_3_bigset_{generation_num}.h5')
image_url = "https://www.lordsindia.com/image/cache/1/TROUSER/NEW%20TROUSER%202019/a480b0c2e22dcccf2276c4116ad6ff10-500x500.jpg"
response = requests.get(image_url)
pil_img = Image.open(BytesIO(response.content))

raw_img = np.array(pil_img)
raw_img = raw_img [:,:,::-1].copy()
raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
raw_img = cv2.resize(raw_img, (IMG_SIZE, IMG_SIZE))
plt.imshow(raw_img)
nparr = np.array([raw_img])
nparr = nparr.astype('float32')
nparr /= 255

pred_x  = torch.from_numpy(nparr)
pred_x = pred_x.permute(0,3,1,2)

# prediction for out-of-sample set
with torch.no_grad():
  if(torch.cuda.is_available()):
    output = model(pred_x.cuda())
  else:
    output = model(pred_x)
    
softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

print(predictions)
print("Shirt" if predictions[0] == 0 else "Tshirt" if predictions[0] == 1 else "Pant")
