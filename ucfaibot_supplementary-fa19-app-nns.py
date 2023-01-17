# This is a bit of code to make things work on Kaggle

import os

from pathlib import Path



if os.path.exists("/kaggle/input/ucfai-supplementary-fa19-app-nns/data"):

    DATA_DIR = Path("/kaggle/input/ucfai-supplementary-fa19-app-nns/data")

else:

    DATA_DIR = Path("data/")



!pip install torchsummary
# general imports

import numpy as np

import time

import os

import glob

import matplotlib.pyplot as plt

from PIL import Image

import cv2

import pandas as pd

from sklearn.model_selection import train_test_split



# torch imports

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

import torch.backends.cudnn as cudnn



# torchvision

from torchvision import transforms

from torchvision.datasets import ImageFolder

from torch.utils.data import TensorDataset, DataLoader, Dataset

from torchvision import models as pretrained_models
try:

    import torchsummary

except:

    torchsummary = None



from tabulate import tabulate



BATCH_TEMPLATE = "Epoch [{} / {}], Batch [{} / {}]:"

EPOCH_TEMPLATE = "Epoch [{} / {}]:"

TEST_TEMPLATE = "Epoch [{}] Test:"



def print_iter(curr_epoch=None, epochs=None, batch_i=None, num_batches=None, writer=None, msg=False, **kwargs):

    """

    Formats an iteration. kwargs should be a variable amount of metrics=vals

    Optional Arguments:

        curr_epoch(int): current epoch number (should be in range [0, epochs - 1])

        epochs(int): total number of epochs

        batch_i(int): current batch iteration

        num_batches(int): total number of batches

        writer(SummaryWriter): tensorboardX summary writer object

        msg(bool): if true, doesn't print but returns the message string



    if curr_epoch and epochs is defined, will format end of epoch iteration

    if batch_i and num_batches is also defined, will define a batch iteration

    if curr_epoch is only defined, defines a validation (testing) iteration

    if none of these are defined, defines a single testing iteration

    if writer is not defined, metrics are not saved to tensorboard

    """

    if curr_epoch is not None:

        if batch_i is not None and num_batches is not None and epochs is not None:

            out = BATCH_TEMPLATE.format(curr_epoch + 1, epochs, batch_i, num_batches)

        elif epochs is not None:

            out = EPOCH_TEMPLATE.format(curr_epoch + 1, epochs)

        else:

            out = TEST_TEMPLATE.format(curr_epoch + 1)

    else:

        out = "Testing Results:"



    floatfmt = []

    for metric, val in kwargs.items():

        if "loss" in metric or "recall" in metric or "alarm" in metric or "prec" in metric:

            floatfmt.append(".4f")

        elif "accuracy" in metric or "acc" in metric:

            floatfmt.append(".2f")

        else:

            floatfmt.append(".6f")



        if writer and curr_epoch:

            writer.add_scalar(metric, val, curr_epoch)

        elif writer and batch_i:

            writer.add_scalar(metric, val, batch_i * (curr_epoch + 1))



    out += "\n" + tabulate(kwargs.items(), headers=["Metric", "Value"], tablefmt='github', floatfmt=floatfmt)



    if msg:

        return out

    print(out)



def summary(model, input_dim):

    if torchsummary is None:

        raise(ModuleNotFoundError, "TorchSummary was not found!")

    torchsummary.summary(model, input_dim)
folders = {"train": DATA_DIR / 'chest_xray' / 'train', "test": DATA_DIR / 'chest_xray' / 'test'}



num_normal = 0

num_pne = 0

for f in folders.values():

    num_normal += len(glob.glob(str(f / 'NORMAL' / '*')))

    num_pne += len(glob.glob(str(f / 'PNEUMONIA' / '*')))



# plot number of cases

plt.figure(figsize=(10,8))

plt.bar([0, 1], [num_normal, num_pne], color=[(1, 0.5, 0), (0, 0.5, 1)])

plt.title('Number of cases', fontsize=14)

plt.xlabel('Case type', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.xticks([0, 1], ['Normal', 'Pneumonia'])

plt.show()
sample_imgs = glob.glob(str(folders["train"] / 'NORMAL' / '*'))[:5] + glob.glob(str(folders["train"] / 'PNEUMONIA' / '*'))[:5]



# plot some normal and pneumonia images

f, ax = plt.subplots(2,5, figsize=(30,10))

for i in range(10):

    img = cv2.imread(sample_imgs[i])

    print(img.shape)

    ax[i//5, i%5].imshow(img, cmap='gray')

    if i<5:

        ax[i//5, i%5].set_title("Pneumonia")

    else:

        ax[i//5, i%5].set_title("Normal")

    ax[i//5, i%5].axis('off')

    ax[i//5, i%5].set_aspect('auto')

plt.show()
class CustomImageFolder(Dataset):

    

    def __init__(self, root, transforms=None):

        # YOUR CODE HERE

        raise NotImplementedError()

    

    def __getitem__(self, index):

        # YOUR CODE HERE

        raise NotImplementedError()

    

    def __len__(self):

        # YOUR CODE HERE

        raise NotImplementedError()
input_size = (224, 224)

num_workers = 4

batch_size = 16



transform = transforms.Compose([transforms.Resize(input_size),

                                        transforms.ColorJitter(brightness=15, contrast=15, saturation=15),

                                        transforms.RandomHorizontalFlip(p=0.5),

                                        transforms.RandomRotation(90),

                                        transforms.ToTensor(),

                                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

                                    

train_dataset = ImageFolder(folders["train"], transform=transform)

test_dataset = ImageFolder(folders["test"], transform=transform)



train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)

test_dataloader = DataLoader(test_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size)



print(f"Number of: Train Images: {len(train_dataset)}, Test Images: {len(test_dataset)}")

print(f"Dataloader Sizes: Train: {len(train_dataloader)}, Test: {len(test_dataloader)}")
def get_padding(output_dim, input_dim, kernel_size, stride):

    """

    Calculates padding given in output and input dim, and parameters of the convolutional layer



    Arguments should all be integers. Use this function to calculate padding for 1 dimesion at a time.

    Output dimensions should be the same or bigger than input dimensions



    Returns 0 if invalid arguments were passed, otherwise returns an int or tuple that represents the padding.

    """



    padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2



    if padding < 0:

        return 0

    else:

        return padding



# can use this to help with padding calculations, or use in model directly!

print(get_padding(224, 224, 4, 1))
def cnn_block(input_channels, output_channels, kernel_size, stride, padding):

    layers = [nn.Conv2d(input_channels, output_channels, kernel_size, stride=stride, padding=padding)]

    layers += [nn.BatchNorm2d(output_channels)]

    layers += [nn.ReLU(inplace=True)]

    

    return layers
class CNN_Model(nn.Module):

    def __init__(self, use_pretrained=False, input_dim=224):

        super(CNN_Model, self).__init__()

        

        self.dim = input_dim

        

        if use_pretrained:

            self.features = pretrained_models.resnet18(pretrained=True)

            #for layer in self.features.parameters()

            #    layer.requires_grad = False

            

            self.classifier = nn.Sequential(nn.Linear(1000, 1), nn.Sigmoid())

        else:

            self.features, out_c = self.build_layers()

            self.classifier = self.build_classifier(out_c)

    

    

    def forward(self, x):

        # squeeze out those features! It removes extra dimensions of 1 from shape.

        x = self.features(x)

        return self.classifier(x.squeeze()).squeeze()

    

    

    def build_classifier(self, in_c):

        layers = []

        # YOUR CODE HERE

        raise NotImplementedError()

        

    

    def build_layers(self):

        layers = []

        in_c = 3

        out_c = 64

        dim = self.dim

        

        layers += cnn_block(in_c, out_c, 3, 1, get_padding(dim, dim, 4, 1))

        in_c = out_c

        out_c = 64

        

        layers += [nn.MaxPool2d(2, stride=2)]

        dim = dim // 2

        # YOUR CODE HERE

        raise NotImplementedError()

        layers += [nn.AdaptiveAvgPool2d(1)]

        

        return nn.Sequential(*layers), out_c

        
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device: {}".format(device))

learn_rate = 0.0001

epochs = 20



model = CNN_Model(use_pretrained=True).to(device)



opt = optim.Adam(model.parameters(), lr=learn_rate)

criterion = nn.BCELoss()



summary(model, (3, 224, 224))
# defines a test run through data

# epoch of -1 is just a test run

def test(epoch):

    model.eval()

    test_loss = 0

    correct = 0

    total = 0

    

    with torch.no_grad():

        for inputs, targets in test_dataloader:

            inputs, targets = inputs.to(device), targets.float().to(device)

            outputs = model(inputs)



            loss = criterion(outputs, loss_const)



            test_loss += loss.item()            

            # round off decimal predictions to either 0 or 1

            preds = torch.round(outputs)

            total += targets.size(0)

            # sum correct predictions

            correct += torch.sum(preds == targets.data)



        if epoch == -1:

            print_iter(test_loss=test_loss/len(test_dataloader), val_acc=(correct / total) * 100.0)

        else:

            print_iter(curr_epoch=epoch, writer=writer, val_loss=test_loss/len(test_dataloader), val_acc=(correct / total) * 100.0)



            return test_loss/len(test_dataloader), (correct / total) * 100.0

print_step = 5

best_loss = 0

best_acc = 0



print("Training Starting...")

for e in range(epochs):

    model.train()

    train_loss = 0

    correct = 0

    total = 0

    epoch_start_time = time.time()



    for inputs, targets in train_dataloader:

        inputs, targets = inputs.to(device), targets.float().to(device)

        outputs = model(inputs)



        opt.zero_grad()

        

        # backward

        loss = criterion(outputs, targets)

        loss.backward()

        opt.step()

        

        train_loss += loss.item() # .item() extracts the raw loss value from the tensor object

        

        # round off decimal predictions to either 0 or 1

        preds = torch.round(outputs)

        total += targets.size(0)

        # sum correct predictions

        correct += torch.sum(preds == targets.data)



        if i % print_step == 0:

            print_iter(curr_epoch=e, epochs=epochs, batch_i=i, num_batches=len(train_dataloader), loss=train_loss/(i+1), acc=(correct / total) * 100.0)



    print_iter(curr_epoch=e, epochs=epochs, writer=writer, loss=train_loss/len(train_dataloader), acc=(correct / total) * 100.0, time=(time.time() - epoch_start_time) / 60)



    val_loss, val_acc = test(e)

    

    if best_acc < val_acc:

        print('Saving Checkpoint..')

        checkpoint_path = "best.weights.pt"

        state = {'net': model.state_dict(), 'acc': val_acc}

        torch.save(state, checkpoint_path)

        best_acc = val_acc
data = pd.read_csv(DATA_DIR / 'cervical.csv')

data.head()
data.info()
# Replace '?' with nans

data = data.replace('?', np.nan)

data.head()
data.isnull().sum()
data = data.convert_objects(convert_numeric=True)

data.info()
data['Number of sexual partners'] = data['Number of sexual partners'].fillna(data['Number of sexual partners'].median())

data['Smokes'] = data['Smokes'].fillna(1)

# Fill the rest here!
data.isnull().sum()