import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np
train_data = pd.read_csv("../input/train.csv")

test_data = pd.read_csv("../input/test.csv")
print("Train data dimensions: ", train_data.shape)

print("Test data dimensions: ", test_data.shape)
train_data.head(5)
print("Number of missing values",train_data.isnull().sum().sum())
train_data.describe()
contFeatureslist = []

for colName,x in train_data.iloc[1,:].iteritems():

    #print(x)

    if(not str(x).isalpha()):

        contFeatureslist.append(colName)

print(contFeatureslist)