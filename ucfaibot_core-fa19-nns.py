from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-fa19-nns").exists():

    DATA_DIR /= "ucfai-core-fa19-nns"

elif DATA_DIR.exists():

    # no-op to keep the proper data path for Kaggle

    pass

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-nns/data

    DATA_DIR = Path("data")
import numpy as np

import pandas as pd

import torch 

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch import optim

import time
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
new_tensor[0, 0] = 5

new_tensor
# type of a tensor

print(new_tensor.type())  



# shape of a tensor

print(new_tensor.shape)    

print(new_tensor.size())   



# dimension of a tensor

print(new_tensor.dim())
np_ndarray = np.random.randn(2,2)

np_ndarray
# NumPy ndarray to PyTorch tensor

to_tensor = torch.from_numpy(np_ndarray)



to_tensor
torch.cuda.is_available()
n_in, n_h, n_out, batch_size = 10, 5, 1, 10
x = torch.randn(batch_size, n_in)

y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])
model = nn.Sequential(

    nn.Linear(n_in, n_h),

    nn.Sigmoid(),

    nn.Linear(n_h, n_out),

    nn.Sigmoid()

)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(50):

    # Forward Propagation

    y_pred = model(x)

    

    # Compute and print loss

    loss = criterion(y_pred, y)

    print('epoch: ', epoch,' loss: ', loss.item())

    

    # Zero the gradients

    optimizer.zero_grad()

    

    # perform a backward pass (backpropagation)

    loss.backward()

    

    # Update the parameters

    optimizer.step()
dataset = pd.read_csv(DATA_DIR / "train.csv", header=None).values



pd.DataFrame(dataset).head()
dataset = np.delete(dataset, 0, 0)



pd.DataFrame(dataset).head()
# split into x and y sets

X = dataset[:,:-1].astype(np.float32)



Y = dataset[:,-1].astype(np.float32)



# Needed to make PyTorch happy

Y = np.expand_dims(Y, axis = 1)



# Test-Train split

from sklearn.model_selection import train_test_split



split = train_test_split(X, Y, test_size=0.1)

xTrain, xTest, yTrain, yTest = split



# Here we're defining what component we'll use to train this model

# We want to use the GPU if available, if not we use the CPU

# If your device is not cuda, check the GPU option in the Kaggle Kernel



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

device
class PyTorch_Dataset(Dataset):

    def __init__(self, data, outputs):

        self.data = data

        self.outputs = outputs



    def __len__(self):

        'Returns the total number of samples in this dataset'

        return len(self.data)



    def __getitem__(self, index):

        'Returns a row of data and its output'

      

        x = self.data[index]

        y = self.outputs[index]



        return x, y
train_dataset = PyTorch_Dataset(xTrain, yTrain)

val_dataset = PyTorch_Dataset(xTest, yTest)



datasets = {'Train': train_dataset, 'Validation': val_dataset}
dataloaders = {

    x: DataLoader(datasets[x], batch_size=16, shuffle=True, num_workers=4)

    for x in ['Train', 'Validation']

}
inputSize    =  8    # how many classes of input

hiddenSize   = 15    # Number of units in the middle

numClasses   =  1    # Only has two classes

numEpochs    = 20    # How many training cycles

learningRate = 0.01  # Learning rate



class NeuralNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):

        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size) 

        self.fc2 = nn.Linear(hidden_size, num_classes)  

    

    def forward(self, x):

        x = F.relu(self.fc1(x))

        return torch.sigmoid(self.fc2(x))
# Creating our model

model = NeuralNet(inputSize, hiddenSize, numClasses)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr = learningRate)

model.to(device)

print(model)
def run_epoch(model, dataloaders, device, phase):

  

    running_loss = 0.0

    running_corrects = 0

    

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

        running_loss += loss.item() * inputs.size(0)

        running_corrects += torch.sum(preds == labels)

    

    # Calculate epoch statistics

    epoch_loss = running_loss / datasets[phase].__len__()

    epoch_acc = running_corrects.double() / datasets[phase].__len__()

  

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

from sklearn.metrics import confusion_matrix, matthews_corrcoef



matthews_corrcoef(preds, yTest)
confusion_matrix(preds, yTest)
# TODO, make a better model!



# YOUR CODE HERE

raise NotImplementedError()
# Run this to generate the submission file for the competition!

### Make sure to name your model variable "model" ###



# load in test data:

test_data = pd.read_csv(DATA_DIR / "test.csv", header=None).values

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