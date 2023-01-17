from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-sp20-ml-apps").exists():

    DATA_DIR /= "ucfai-core-sp20-ml-apps"

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-ml-apps/data

    DATA_DIR = Path("data")
from pathlib import Path

import os



if os.path.exists("/kaggle/input/ucfai-core-sp20-ml-apps"):

    DATA_DIR = Path("/kaggle/input/ucfai-core-sp20-ml-apps")

else:

    DATA_DIR = Path("data/")
# general stuff

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as img

import pandas as pd



# sklearn models and metrics

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, matthews_corrcoef



from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

from sklearn.neighbors import KNeighborsClassifier



# pytorch imports

import torch 

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch import optim



!pip install torchsummary

from torchsummary import summary
df = pd.read_csv(DATA_DIR / 'cumulative.csv', delimiter=',')

df.head()
# make a new column that is a binary classificaiton of whether or not a planet is a canidate

disposition = [0] * len(df['koi_pdisposition'])

for i in range(len(df['koi_pdisposition'])):

    disposition[i] = 1 if (df['koi_pdisposition'][i] == 'CANDIDATE') else 0 



df.insert(1, "disposition", disposition)



columns = ["disposition", "koi_period", "koi_impact", "koi_srad", "koi_slogg", "ra", "dec"]

df = df[columns].dropna()

df.head()
# visualize data



# custom color map for our dataset

color = np.where(df['disposition'] == 1, 'green', 'red')

fig, axs = plt.subplots(3, figsize=(15,10))



# make sure to play around with these to better understand the dataset



axs[0].scatter(df['koi_slogg'], df['koi_impact'], c=color)

axs[0].set_xlabel("impact")

axs[0].set_ylabel("slogg")



axs[1].scatter(df['koi_srad'], df['koi_impact'], c=color)

axs[1].set_xlabel("impact")

axs[1].set_ylabel("srad")



axs[2].scatter(df['ra'], df['dec'], c=color)

axs[2].set_xlabel("ra")

axs[2].set_ylabel("dec")
#X_train, X_test, Y_train, Y_test = train_test_split(df)

X = pd.DataFrame(columns=['koi_period', 'koi_slogg', 'koi_srad', 'koi_impact', 'ra', 'dec'], data=df).values

Y = pd.DataFrame(columns=['disposition'], data=df).values.ravel()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# make a model

# YOUR CODE HERE

raise NotImplementedError()



# make predictions and test accuracy

# YOUR CODE HERE

raise NotImplementedError()
from torchvision.datasets import ImageFolder

from torchvision.models import resnet18

from torchvision import transforms



from torch.utils.data import random_split



# define transform for image data:

input_size = (224, 224)



# this will resize the image, convert it to a tensor, convert pixel values to be in range of [0, 1]

# and normalize the data

data_transform = transforms.Compose([transforms.Resize(input_size),

                                     transforms.ToTensor(),

                                     transforms.Normalize(mean=[0.5, 0.5, 0.5],

                                     std=[0.5, 0.5, 0.5])

                                    ])



# load in data

dataset = ImageFolder(DATA_DIR / "dataset", transform=data_transform)

types = pd.read_csv(DATA_DIR / "pokemon_fixed.csv")



# split data

test_split = 0.2



# get number of samples that should be in training set

train_size = int(len(dataset) * (1 - test_split))



# split the dataset into training and testing Subsets

train, test = random_split(dataset, (train_size, len(dataset) - train_size))



print(f"Number of training samples: {len(train)}, testing: {len(test)}")

print("\n".join(dataset.classes))
def show_random_imgs(num_imgs):

    for i in range(num_imgs):

        # Choose a random image

        rand = np.random.randint(0, len(dataset) + 1)

        

        # Read in the image

        ex = img.imread(dataset.imgs[rand][0])

        

        # Get the image's label

        pokemon = dataset.classes[dataset.imgs[rand][1]]

        

        # Show the image and print out the image's size (really the shape of it's array of pixels)

        plt.imshow(ex)

        print('Image Shape: ' + str(ex.shape))

        plt.axis('off')

        plt.title(pokemon)

        plt.show()



show_random_imgs(3)
types.head()
batch_size = 16



# Define train and test dataloaders

# Name them train_dataloader and test_dataloader

# YOUR CODE HERE

raise NotImplementedError()



types = types.drop("Type2", axis=1)



# need to convert the pokemon name to their class number

classes = {name.lower(): i for i, name in enumerate(dataset.classes)}

types = types.replace(to_replace=classes)



# now we need to turn the types to class indicies as well

unique_types = sorted(types.Type1.unique())

int_to_type = {i: t for i, t in enumerate(unique_types)}

type_to_int = {t: i for i, t in enumerate(unique_types)}



types = types.replace(to_replace=type_to_int)



# turn dataframe into a dictionary

# keys are the class number of the pokemon and it gives the type for the pokemon

types = {t[0]: t[1] for t in types.values}



# finally, lets make a function to get a tensor of target types given input pokemon names

# input should be a torch tensor

def get_types(pokemon_classes):

    return torch.tensor([types[c.item()] for c in pokemon_classes])



print(get_types([torch.tensor(5)]))
# example two output model

class TwoOutputModel(nn.Module):

    def __init__(self):

        self.fc1 = nn.Linear(500, 250)

        self.out1 = nn.Linear(250, 10)

        self.out2 = nn.Linear(250, 1)

    

    def forward(x):

        x = self.fc1(x)

        

        out1 = self.out1(x)

        out2 = self.out2(x)

        

        return out1, out2



# two output loss example:

# c = nn.CrossEntropyLoss()

# target = 1

# output_1, output_2, = model(input)

# loss_1 = c(output_1, target)

# loss_2 = c(output_2, target)

# loss = loss_1 + loss_2

# loss.backward()

# optimizer.step()
# It is good practice to maintain input dimensions as the image is passed through convolution layers

# With a default stride of 1, and no padding, a convolution will reduce image dimenions to:

            # out = in - m + 1, where m is the size of the kernel and in is a dimension of the input



# Use this function to calculate the padding size neccessary to create an output of desired dimensions



def get_padding(input_dim, output_dim, kernel_size, stride):

  # Calculates padding necessary to create a certain output size,

  # given a input size, kernel size and stride

  

  padding = (((output_dim - 1) * stride) - input_dim + kernel_size) // 2

  

  if padding < 0:

    return 0

  else:

    return padding



get_padding(224, 224, 3, 1)
class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        

        # put your model layers here

        # YOUR CODE HERE

        raise NotImplementedError()

        

        # define our two output layers

        self.name = nn.Linear(512, len(dataset.classes))

        self.type_ = nn.Linear(512, len(unique_types))

    

    # Write the forward method for this network (it's quite simple since we've defined the network in blocks already)

    def forward(self, x):

        # YOUR CODE HERE

        raise NotImplementedError()

        

        # x should be the output from your last layer before the output layers

        name = self.name(x)

        type_ = self.type_(x)

        return name, type_
model = CNN()



# define the criterion and optimizer below, with those names

# YOUR CODE HERE

raise NotImplementedError()



# Note: may take a bit to train on kaggle!

epochs = 10

checkpoint_path = "best.model.pt"



model.to(device)

summary(model, (3, *input_size))
# Defines a test run through our testing data

def test(epoch):

    model.eval()

    test_loss = 0

    correct_names = 0

    correct_types = 0

    total = 0

    for i, data in enumerate(test_dataloader):

        with torch.no_grad(): # doesn't calculate gradients since we are testing

            inputs, targets = data

            

            type_targets = get_types(targets).to(device)

            inputs, targets = inputs.to(device), targets.to(device)

            

            # get model outputs, run criterion on each output, then sum losses

            # write out the first 2 parts here, name them loss_name and loss_type for each loss

            # YOUR CODE HERE

            raise NotImplementedError()

            

            loss = loss_name + loss_type



        test_loss += loss.item()

        _, predicted_names = name_out.max(dim=1)

        _, predicted_type = type_out.max(dim=1)

        

        total += targets.size(0)

        correct_names += predicted_names.eq(targets).sum().item()

        correct_types += predicted_type.eq(type_targets).sum().item()

    

    # defines loss, name accuracy, type accuracy

    results = (test_loss/len(test_dataloader), (correct_names / total) * 100.0, (correct_types / total) * 100.0)

    

    # epoch less than 0 means we are just testing outside training

    if epoch < 0: 

        print("Test Results: loss: {:.4f}, name_acc: {:.2f}%, type_acc: {:.2f}%".format(

            results[0], results[1], results[2]))

    else:

        print("Epoch [{}] Test: loss: {:.4f}, name_acc: {:.2f}%, type_acc: {:.2f}%".format(

            epoch + 1, results[0], results[1], results[2]))



        return results
# Training phase

print_step = len(train_dataloader) // 50

best_loss = 0



for e in range(epochs):

    model.train()

    

    # define inital metric values

    train_loss = 0

    correct_names = 0

    correct_types = 0

    total = 0

    

    for i, data in enumerate(train_dataloader):

        inputs, targets = data

        

        # get our type targets using our helper function

        type_targets = get_types(targets).to(device)

        inputs, targets = inputs.to(device), targets.to(device)

        

        # zero out previous gradients

        optimizer.zero_grad()



        # forward

        name_out, type_out = model(inputs)

        

        # backward

        # get model outputs, run criterion on each output, then sum losses

        # write out the first 2 parts here, name them loss_name and loss_type for each loss

        # YOUR CODE HERE

        raise NotImplementedError()

        

        # sum the losses for backprop

        loss = loss_name + loss_type

        

        # calculate gradients and update weights

        loss.backward()

        optimizer.step()

        

        # calculate our accuracy metrics and loss

        train_loss += loss.item() # .item() extracts the raw loss value from the tensor object

        _, predicted_names = name_out.max(dim=1)

        _, predicted_type = type_out.max(dim=1)

        

        total += targets.size(0)

        correct_names += predicted_names.eq(targets).sum().item()

        correct_types += predicted_type.eq(type_targets).sum().item()



        if i % print_step == 0:

            print("Epoch [{} / {}], Batch [{} / {}]: loss: {:.4f}, name_acc: {:.2f}%, type_acc: {:.2f}%".format(

                e+1, epochs, i+1, len(train_dataloader), train_loss/(i+1), (correct_names / total) * 100.0, (correct_types / total) * 100.0))



    print("Epoch [{} / {}]: loss: {:.4f}, name_acc: {:.2f}%, type_acc: {:.2f}%".format(

        e+1, epochs, train_loss/(len(train_dataloader)), (correct_names / total) * 100.0, (correct_types / total) * 100.0))



    val_loss, val_name_acc, val_type_acc = test(e)

    

    if val_loss < best_loss or e == 0: # model improved

        print('---Loss improved! Saving Checkpoint---')

        state = {'net': model.state_dict(), 'loss': val_loss, 'epoch': e}

        torch.save(state, checkpoint_path)

        best_loss = val_loss

best_cp = torch.load(checkpoint_path)

model.load_state_dict(best_cp["net"])



# Lets see the final results

test(-1)
def display_results(num_images):

    was_training = model.training

    model.eval()

    images_so_far = 0

    fig = plt.figure(num_images, (15,20))



    with torch.no_grad():

        for i, (images, targets) in enumerate(test_dataloader): 

            images = images.to(device)

            

            type_targets = get_types(targets)

            

            name_out, type_out = model(images)

            _, predicted_names = name_out.max(dim=1)

            _, predicted_type = type_out.max(dim=1)



            for j in range(images.size()[0]):

                # plot images for display

                images_so_far += 1

                ax = plt.subplot(num_images//2, 2, images_so_far)

                ax.axis('off')

                # title is the actual and predicted values

                ax.set_title('Actual Name: {}\nPrediction: {}\nActual Type: {}\n Prediction: {}'.format(

                    dataset.classes[targets[j]], dataset.classes[predicted_names[j]], int_to_type[type_targets[j].item()],

                    int_to_type[predicted_type[j].item()]))

                

                image = images.cpu().data[j].numpy().transpose((1, 2, 0))

                

                # undo our pytorch transform for display

                mean = np.array([0.5, 0.5, 0.5])

                std = np.array([0.5, 0.5, 0.5])

                image = std * image + mean

                image = np.clip(image, 0, 1)

                

                plt.imshow(image)

                if images_so_far == num_images:

                    model.train(mode=was_training)

                    return

            

        model.train(mode=was_training)
display_results(6)