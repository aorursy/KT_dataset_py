from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-sp20-nns").exists():

    DATA_DIR /= "ucfai-core-sp20-nns"

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-sp20-nns/data

    DATA_DIR = Path("data")
# change this if running locally

DATA_DIR = "/kaggle/input/ucfai-core-sp20-nns"

# DATA_DIR = "."
import numpy as np

import pandas as pd

import torch 

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch import optim

import time



!pip install torchsummary

from torchsummary import summary
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
# create a tensor

new_tensor = torch.Tensor([[1, 2], [3, 4]])



# create a 2 x 3 tensor with random values

empty_tensor = torch.Tensor(2, 3)



# create a 2 x 3 tensor with random values between -1and 1

uniform_tensor = torch.Tensor(2, 3).uniform_(-1, 1)



# create a 2 x 3 tensor with random values from a uniform distribution on the interval [0, 1)

rand_tensor = torch.rand(2, 3)



# create a 2 x 3 tensor of zeros

zero_tensor = torch.zeros(2, 3)
new_tensor
new_tensor[0][0] = 5

new_tensor
new_tensor = new_tensor.to(device)

new_tensor
# data type of a tensor, notice that running on a GPU will give a type of cuda.<datatype>Tensor, 

# in this case a torch.cuda.FloatTensor

print(new_tensor.type())  



# shape of a tensor, both give the same thing

print(new_tensor.shape)    

print(new_tensor.size())   



# dimension of a tensor, how many dimensions is has (2D, 3D, etc.)

print(new_tensor.dim())
np_ndarray = np.random.randn(2,2)

np_ndarray
# NumPy ndarray to PyTorch tensor

to_tensor = torch.from_numpy(np_ndarray)



to_tensor
n_in, n_h, n_out, batch_size = 10, 5, 1, 10
x = torch.randn(batch_size, n_in)

y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
# a linear function is defined as nn.Linear(num_input_nodes, num_output_nodes)

model = nn.Sequential(nn.Linear(n_in, n_h),

                     nn.Sigmoid(),

                     nn.Linear(n_h, n_out),

                     nn.Sigmoid())
criterion = nn.MSELoss()
# pass the parameters of our model to our optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# put our model and data onto our device for increase speed

model = model.to(device)

x, y = x.to(device), y.to(device)

for epoch in range(50):

    # Forward Pass

    y_pred = model(x)

    # Compute and print loss

    loss = criterion(y_pred, y)

    print('epoch: ', epoch,' loss: ', loss.item())

    

    # Zero the gradients, this is needed so we don't keep gradients from the last interation

    optimizer.zero_grad()

    

    # perform a backward pass (backpropagation)

    loss.backward()

    

    # Update the parameters

    optimizer.step()
dataset = pd.read_csv(f"{DATA_DIR}/train.csv", header=None)



dataset.head()
dataset.info()
dataset = dataset.values

dataset = np.delete(dataset, 0, 0).astype(np.float32)



dataset = pd.DataFrame(dataset) # convert back to a dataframe

dataset.head()
dataset.info()
dataset.isna().sum()
# get numpy array from our dataframe

dataset = dataset.values



# split into x and y sets



X = dataset[:,:-1].astype(np.float32)



Y = dataset[:,-1].astype(np.float32)



# Our Y shape is missing the second axis, so add that now since pytorch won't accept the data otherwise

Y = np.expand_dims(Y, axis = 1)



# Test-Train split

from sklearn.model_selection import train_test_split



xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.1)
class PyTorch_Dataset(Dataset):

  

  def __init__(self, data, outputs):

        self.data = data

        self.outputs = outputs



  def __len__(self):

        'Returns the total number of samples in this dataset'

        return len(self.data)



  def __getitem__(self, index):

        'Returns a row of data and its output'

        # for more advanced dataset, more preprocessing would go here

        x = self.data[index]

        y = self.outputs[index]



        return x, y
train_dataset = PyTorch_Dataset(xTrain, yTrain)

val_dataset = PyTorch_Dataset(xTest, yTest)



datasets = {'Train': train_dataset, 'Validation': val_dataset}
dataloaders = {x: DataLoader(datasets[x], batch_size=16, shuffle=True, num_workers = 4)

              for x in ['Train', 'Validation']}
class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        # call the constructor of the super class

        super(NeuralNet, self).__init__()

        # define our input->hidden layer

        self.fc1 = nn.Linear(input_size, hidden_size)

        # define our hidden->output layer

        self.fc2 = nn.Linear(hidden_size, num_classes)  

    

    def forward(self, x):

        # pass through our first linear function

        x = self.fc1(x)

        # apply relu activation function

        x = F.relu(x)

        # pass through second layer, which outputs our final raw value

        x = self.fc2(x)

        # apply sigmoid activation to get a prediction value between 0 and 1

        x = torch.sigmoid(x)

        

        return x
inputSize =  8         # how many pieces of data for input

hiddenSize = 15        # Number of units in the middle hidden layer

numClasses = 1         # Only has two classes, so 1 output node

numEpochs = 20         # How many training cycles

learningRate = .01     # Learning rate



# Creating our model

model = NeuralNet(inputSize, hiddenSize, numClasses)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr = learningRate)

model.to(device) # remember to put your model onto the device!



summary(model, X.shape)
def run_epoch(model, dataloaders, device, phase):

  

  # holds values for our total loss and accuracy

  running_loss = 0.0

  running_corrects = 0

  

  # put model into the proper mode.

  if phase == 'Train':

    model.train()

  else:

    model.eval()

  

  # Looping through batches

  for i, (inputs, labels) in enumerate(dataloaders[phase]):

    

    # ensures we're doing this calculation on our GPU if possible

    inputs = inputs.to(device)

    labels = labels.to(device)

    

    # Zero parameter gradients

    optimizer.zero_grad()

    

    # Calculate gradients only if we're in the training phase

    with torch.set_grad_enabled(phase == 'Train'):

      

      # This calls the forward() function on a batch of inputs

      outputs = model(inputs)



      # Calculate the loss of the batch

      loss = criterion(outputs, labels)



      # Adjust weights through backpropagation if we're in training phase

      if phase == 'Train':

        loss.backward()

        optimizer.step()

      

    # Get binary predictions

    preds = torch.round(outputs)



    # Document statistics for the batch

    running_loss += loss.item() * inputs.size(0) # .item() gets the value of the loss as a raw value

    running_corrects += torch.sum(preds == labels) # sums all the times where the prediction equals our label (correct)

    

  # Calculate epoch statistics

  epoch_loss = running_loss / len(datasets[phase])

  epoch_acc = running_corrects.double() / len(datasets[phase])

  

  return epoch_loss, epoch_acc
def train(model, criterion, optimizer, num_epochs, dataloaders, device):

    start = time.time()



    best_model_wts = model.state_dict()

    best_acc = 0.0

    

    print('| Epoch\t | Train Loss\t| Train Acc\t| Valid Loss\t| Valid Acc\t|')

    print('-' * 73)

    

    # Iterate through epochs

    for epoch in range(num_epochs):

        

        # Training phase

        train_loss, train_acc = run_epoch(model, dataloaders, device, 'Train')

        

        # Validation phase

        val_loss, val_acc = run_epoch(model, dataloaders, device, 'Validation')

           

        # Print statistics after the validation phase

        print("| {}\t | {:.4f}\t| {:.4f}\t| {:.4f}\t| {:.4f}\t|".format(epoch + 1, train_loss, train_acc, val_loss, val_acc))



        # Copy and save the model's weights if it has the best accuracy thus far

        if val_acc > best_acc:

            best_acc = val_acc

            best_model_wts = model.state_dict()



    total_time = time.time() - start

    

    print('-' * 74)

    print('Training complete in {:.0f}m {:.0f}s'.format(total_time // 60, total_time % 60))

    print('Best validation accuracy: {:.4f}'.format(best_acc))



    # load best model weights and return them

    model.load_state_dict(best_model_wts)

    return model
model = train(model, criterion, optimizer, numEpochs, dataloaders, device)
# Function which generates predictions, given a set of inputs

def test(model, inputs, device):

  model.eval()

  inputs = torch.tensor(inputs).to(device)

  

  outputs = model(inputs).cpu().detach().numpy()

  

  preds = np.where(outputs > 0.5, 1, 0)

  

  return preds
preds = test(model, xTest, device)
# import functions for matthews and confusion matrix

from sklearn.metrics import matthews_corrcoef, confusion_matrix

import matplotlib.pyplot as plt



matthews_corrcoef(preds, yTest)
confusion = confusion_matrix(preds, yTest)



def plot_confusion_matrix(confusion):

  categories = ["Not Diabetic", "Diabetic"]

  fig, ax = plt.subplots()

  im = ax.imshow(confusion)

  ax.set_yticks(np.arange(len(categories)))

  ax.set_yticklabels(categories)



  for i in range(len(categories)):

    for j in range(len(confusion)):

      ax.text(i, j, confusion[i, j], ha="center", va="center", color="white")



plot_confusion_matrix(confusion)
#TODO, make a better model!



# YOUR CODE HERE

raise NotImplementedError()
# Run this to generate the submission file for the competition!

### Make sure to name your model variable "model" ###



# load in test data:

test_data = pd.read_csv(f"{DATA_DIR}/test.csv", header=None).values

# remove row with column labels:

test_data = np.delete(test_data, 0, 0)



# convert to float32 values

X = test_data.astype(np.float32)

# get indicies for each entry in test data

indicies = [i for i in range(len(X))]



# generate predictions

preds = test(model, X, device)



# create our pandas dataframe for our submission file. Squeeze removes dimensions of 1 in a numpy matrix Ex: (161, 1) -> (161,)

preds = pd.DataFrame({'Id': indicies, 'Class': np.squeeze(preds)})



# save submission csv

preds.to_csv('submission.csv', header=['Id', 'Class'], index=False)