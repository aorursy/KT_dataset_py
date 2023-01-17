import torch

import torchvision

import torch.nn as nn

import pandas as pd

import matplotlib.pyplot as plt

import torch.nn.functional as F

from torchvision.datasets.utils import download_url

from torch.utils.data import DataLoader, TensorDataset, random_split
DATASET_URL = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"

DATA_FILENAME = "insurance.csv"

download_url(DATASET_URL, '.')
dataframe_raw = pd.read_csv(DATA_FILENAME)

dataframe_raw.head()
your_name = "Vighnesh" # at least 5 characters
def customize_dataset(dataframe_raw, rand_str):

    dataframe = dataframe_raw.copy(deep=True)

    # drop some rows

    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))

    # scale input

    dataframe.bmi = dataframe.bmi * ord(rand_str[1])/100.

    # scale target

    dataframe.charges = dataframe.charges * ord(rand_str[2])/100.

    # drop column

    if ord(rand_str[3]) % 2 == 1:

        dataframe = dataframe.drop(['region'], axis=1)

    return dataframe
dataframe = customize_dataset(dataframe_raw, your_name)

dataframe.head()
num_rows = len(dataframe)

print(num_rows)
num_cols = len(dataframe.columns)

print(num_cols)
input_cols = list(dataframe.drop('charges',axis=1).columns)

input_cols
categorical_cols = list(dataframe.select_dtypes(include='object').columns)

categorical_cols
output_cols = [dataframe.columns[-1]]

output_cols
# Write your answer here

import numpy as np

# min_charge = np.min(dataframe.charges)

min_charge = dataframe.charges.min()

print("Minimum charge = ",min_charge)

# max_charge = np.max(dataframe.charges)

max_charge = dataframe.charges.max()

print("Maximum charge = ",max_charge)

# avg_charge = np.mean(dataframe.charges)

avg_charge = dataframe.charges.mean()

print("Average charge = ",avg_charge)
# Plotting the distribution of 'charges' column

import seaborn as sns

fig, axs = plt.subplots(ncols=2)

sns.set_style("darkgrid")

plt.rcParams['font.size'] = 14

plt.rcParams['figure.figsize'] = (9, 5)

#plt.title("Distribution of charges")

sns.distplot(dataframe.charges, ax=axs[0]) # Skewed data

sns.distplot(np.log(dataframe.charges),ax=axs[1]) # Trying to make data normal using log transformation
def dataframe_to_arrays(dataframe):

    # Make a copy of the original dataframe

    dataframe1 = dataframe.copy(deep=True)

    # Convert non-numeric categorical columns to numbers

    for col in categorical_cols:

        dataframe1[col] = dataframe1[col].astype('category').cat.codes

    # Extract input & outupts as numpy arrays

    #inputs_array = np.array(dataframe1[input_cols])

    inputs_array = dataframe1.drop('charges',axis=1).values

    #targets_array = np.array(dataframe1[output_cols])

    targets_array = dataframe1[['charges']].values

    return inputs_array, targets_array
inputs_array, targets_array = dataframe_to_arrays(dataframe)

print(inputs_array.shape, targets_array.shape)

inputs_array, targets_array
inputs = torch.from_numpy(inputs_array).to(torch.float32)

targets = torch.from_numpy(targets_array).to(torch.float32)
inputs.dtype, targets.dtype
print(inputs,targets)
dataset = TensorDataset(inputs, targets)
val_percent = 0.1 # between 0.1 and 0.2

val_size = int(num_rows * val_percent)

print(val_size)

train_size = num_rows - val_size

print(train_size)



train_ds, val_ds = random_split(dataset,[train_size, val_size]) # Use the random_split function to split dataset into 2 parts of the desired length
print(len(train_ds))

print(len(val_ds))
batch_size = 64 # Try to experiment with different batch sizes
train_loader = DataLoader(train_ds, batch_size, shuffle=True)

val_loader = DataLoader(val_ds, batch_size)
for xb, yb in train_loader:

    print("inputs:", xb)

    print("targets:", yb)

    break
input_size = len(input_cols)

print(input_size)

output_size = len(output_cols)

print(output_size)
class InsuranceModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.linear = nn.Linear(input_size,output_size) 

        

    def forward(self, xb):

        out = self.linear(xb)                          

        return out

    

    def training_step(self, batch):

        inputs, targets = batch 

        # Generate predictions

        out = self(inputs)          

        # Calcuate loss

        loss = F.l1_loss(out, targets)                

        return loss

    

    def validation_step(self, batch):

        inputs, targets = batch

        # Generate predictions

        out = self(inputs)

        # Calculate loss

        loss = F.l1_loss(out, targets)                    

        return {'val_loss': loss.detach()}

        

    def validation_epoch_end(self, outputs):

        batch_losses = [x['val_loss'] for x in outputs]

        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses

        return {'val_loss': epoch_loss.item()}

    

    def epoch_end(self, epoch, result, num_epochs):

        # Print result every 20th epoch

        if (epoch+1) % 500 == 0 or epoch == num_epochs-1:

            print("Epoch [{}], val_loss: {:.4f}".format(epoch+1, result['val_loss']))
model = InsuranceModel()
list(model.parameters())
def evaluate(model, val_loader):

    outputs = [model.validation_step(batch) for batch in val_loader]

    return model.validation_epoch_end(outputs)



def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):

    history = []

    optimizer = opt_func(model.parameters(), lr)

    for epoch in range(epochs):

        # Training Phase 

        for batch in train_loader:

            loss = model.training_step(batch)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

        # Validation phase

        result = evaluate(model, val_loader)

        model.epoch_end(epoch, result, epochs)

        history.append(result)

    return history
result = evaluate(model, val_loader) # Use the the evaluate function

print(result)
# model = InsuranceModel() # In case of re-initialization
epochs = 1000

lr = 0.001

history1 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 1500

lr = 0.05

history2 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 2000

lr = 0.1

history3 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 2500

lr = 0.4

history4 = fit(epochs, lr, model, train_loader, val_loader)
epochs = 3000

lr = 0.8

history5 = fit(epochs, lr, model, train_loader, val_loader)
val_loss = history5[-1]

val_loss
def predict_single(input, target, model):

    inputs = input.unsqueeze(0)

    predictions = model(inputs)               

    prediction = predictions[0].detach()

    print("Input:", input)

    print("Target:", target)

    print("Prediction:", prediction)
input, target = val_ds[0]

predict_single(input, target, model)
input, target = val_ds[10]

predict_single(input, target, model)
input, target = val_ds[13]

predict_single(input, target, model)
input, target = val_ds[54]

predict_single(input, target, model)
input, target = val_ds[87]

predict_single(input, target, model)