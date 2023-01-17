# Pull dependencies

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import matplotlib.pyplot as pyplot

from collections import defaultdict

import kagglegym

import numpy as np 

import pandas as pd 



# Create environment

env = kagglegym.make()

# Get first observation

observation = env.reset()

# Get the train dataframe

train = observation.train

train.shape
train_test = train[train['timestamp']==0] # Make a df with just the first timestamp

print (len(train_test['timestamp'])) # See how many values are in the first timestamp
train[0:750].fillna(0, inplace=True) # fill NaN in the first time stamp

train.groupby('id').ffill() # forward fill from there

train.head(1) 
# Filling na values 

mean_values = train.mean(axis=0)

train.fillna(mean_values, inplace=True)

train.head()
# Store all Pearson R values between features in dict

### Warning - this cell takes a while to run

feature_corr = defaultdict(dict)



x_cols = [col for col in train.columns if col not in ['id','timestamp','y']]

y_cols = [col for col in train.columns if col not in ['id','timestamp','y']]



for col in x_cols: #

    for colum in y_cols:

        correlation = np.corrcoef(train[col].values, train[colum].values)[0,1]

        feature_corr[col][colum] = correlation



feature_corr = pd.DataFrame(feature_corr)

feature_corr.index.name = "Feature Y"

feature_corr.columns.name = "Feature X"
# Plot the heat map

import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    plt.figure(figsize=(3,4)) # Probably need to fix this

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_ylabel(df.index.name)

    ax.set_title("Pearson R between Features")

    pyplot.colorbar(axim)

    

heat_map(feature_corr)
# Working on Pearson R between asset returns below

Asset_Returns = defaultdict(dict) # Dict to hold everything



# Need a list of each asset

Asset_List = []

Asset_List_2 = []

for x in train.id:

    if x not in Asset_List:

        Asset_List.append(x)

    if x not in Asset_List_2:

        Asset_List_2.append(x)

print (len(Asset_List))

print (len(Asset_List_2))
print ("Starting Length of Asset List 2:", len(Asset_List_2))



New_Asset_List_2 = []



for asset in Asset_List_2:

    New_Frame = train[train['id']==asset]

    if len(New_Frame['y']) == 906:

        New_Asset_List_2.append(asset)



print ("New Length of Asset List 2:", len(New_Asset_List_2))



for asset in New_Asset_List_2:

    New_Frame = train[train['id']==asset]

    #print (len(New_Frame['y']))  

    

    # Ok, both asset lists have 652 assets, each with 906 values. We can build the heat map
print ("Starting Length of Asset List:", len(Asset_List))



New_Asset_List = []



for asset in Asset_List:

    New_Frame = train[train['id']==asset]

    if len(New_Frame['y']) == 906:

        New_Asset_List.append(asset)



print ("New Length of Asset List:", len(New_Asset_List))



for asset in New_Asset_List:

    New_Frame = train[train['id']==asset]

    #print (len(New_Frame['y']))  

    

# This seems to have worked, need to replicate for Asset_List_2 (done above)    
# Store all Pearson R values between asset returns in dict

# First run of this cell taking a long time, will likely time out: Edit it did time out

# Split the Asset Lists in smaller batches?...but then we don't get the whole picture

# Solution????????????





### Warning - this cell takes a while to run



Return_Pearson = defaultdict(dict)



for asset_1 in New_Asset_List:

    for asset_2 in New_Asset_List_2:

        train_1 = train[train['id']==asset_1]

        train_2 = train[train['id']==asset_2]

        correlation = np.corrcoef(train_1['y'].values, train_2['y'].values)[0,1]

        Return_Pearson[asset_1][asset_2] = correlation



Return_Pearson = pd.DataFrame(Return_Pearson)

Return_Pearson.index.name = "Asset ID"

Return_Pearson.columns.name = "Asset ID"
# Plot the heat map

import matplotlib.pyplot as pyplot



def heat_map(df):

    fig = pyplot.figure()

    ax = fig.add_subplot(111)

    axim = ax.imshow(df.values,cmap = pyplot.get_cmap('RdYlGn'), interpolation = 'nearest')

    ax.set_xlabel(df.columns.name)

    ax.set_ylabel(df.index.name)

    ax.set_title("Pearson R between Asset Returns")

    pyplot.colorbar(axim)

    

heat_map(Return_Pearson)



#### To Do - Add ticks