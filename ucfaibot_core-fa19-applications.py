from pathlib import Path



DATA_DIR = Path("/kaggle/input")

if (DATA_DIR / "ucfai-core-fa19-applications").exists():

    DATA_DIR /= "ucfai-core-fa19-applications"

elif DATA_DIR.exists():

    # no-op to keep the proper data path for Kaggle

    pass

else:

    # You'll need to download the data from Kaggle and place it in the `data/`

    #   directory beside this notebook.

    # The data should be here: https://kaggle.com/c/ucfai-core-fa19-applications/data

    DATA_DIR = Path("data")
# import all the libraries you need



# torch for NNs

import torch 

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from torch import optim



# general imports

from sklearn.model_selection import train_test_split

import time

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv(DATA_DIR / "master.csv")
dataset.head()
print("Total entries: {}, null entries: {}".format(len(dataset["HDI for year"]), dataset["HDI for year"].isnull().sum()))
dataset = dataset.drop("HDI for year", axis=1).drop("country-year", axis=1)

dataset.head()
dataset.describe()
dataset.info()
country_set = sorted(set(dataset["country"]))

country_map = {country : i for i, country in enumerate(country_set)}



sex_map = {'male': 0, 'female': 1}



age_set = sorted(set(dataset["age"]))

age_map = {age: i for i, age in enumerate(age_set)}



gen_set = sorted(set(dataset["generation"]))

gen_map = {gen: i for i, gen in enumerate(gen_set)}



def gdp_fix(x):

    x = int(x.replace(",", ""))

    return x



dataset = dataset.replace({"country": country_map, "sex": sex_map, "generation": gen_map, "age": age_map})

dataset[" gdp_for_year ($) "] = dataset.apply(lambda row: gdp_fix(row[" gdp_for_year ($) "]), axis=1)
dataset.head()
dataset.info()
dataset.describe()
print((dataset["year"] - 1985) / 31)
X, Y = dataset.drop("suicides/100k pop", axis=1).values, dataset["suicides/100k pop"].values
# Split data here using train_test_split

# YOUR CODE HERE

raise NotImplementedError()
print("X shape: {}, Y shape: {}".format(X.shape, Y.shape))
# run this if you are using torch and a NN

class Torch_Dataset(Dataset):

    def __init__(self, data, outputs):

        self.data = data

        self.outputs = outputs



    def __len__(self):

        #'Returns the total number of samples in this dataset'

        return len(self.data)



    def __getitem__(self, index):

        #'Returns a row of data and its output'

      

        x = self.data[index]

        y = self.outputs[index]



        return x, y



# use the above class to create pytorch datasets and dataloader below

# REMEMBER: use torch.from_numpy before creating the dataset! Refer to the NN lecture before for examples
# Lets get this model!

# for your output, it will be one node, that outputs the predicted value. What would the output activation function be?

# YOUR CODE HERE

raise NotImplementedError()