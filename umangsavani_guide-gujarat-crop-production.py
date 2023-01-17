import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/gujarat-crop-production/gujarat_crops_prediction.csv") 
dataset.head()
#count number of missing value

dataset.isnull().sum()
#fill missing value in Production column with mean value.

from sklearn.impute import SimpleImputer

missing_value = SimpleImputer(missing_values = np.nan , strategy = 'mean', verbose=0)

missing_value = missing_value.fit(np.array(dataset['Production']).reshape(len(dataset['Production']),1))

dataset["Production"] = missing_value.transform(np.array(dataset['Production']).reshape(len(dataset['Production']),1))
#count the number of missing value after handling of missing value.

dataset.isnull().sum()
#i remove state name field because it contain only one value.

dataset = dataset.drop(['State_Name'],axis=1)

dataset.head()
#now we find production of crop per area.

#this step is require because of we didn't take any standered scale for measurement of area.

dataset["Production/Area"] = dataset["Production"]/dataset["Area"]
dataset.head()
dataset = dataset.drop(['Area','Production'],axis=1)
dataset.head()
#Now i want to plot bar chart of crops vs production of crops per hectare.

#i choose minimum ratio of production of crops per hectare is 2.5.

#i want to plot crops on graph those have ratio atleast 2.5 Kg/hectare. 

District_Name = dataset['District_Name'].unique()

for district in District_Name:

    plt.figure(figsize=(16,9))

    plt.title("Production of Crops per Hectare in "+district +" District")

    sns.barplot(dataset.loc[(dataset.District_Name == district) & (dataset["Production/Area"] >= 2.5) , "Crop"],dataset.loc[(dataset.District_Name == district) & (dataset['Production/Area'] >= 2.5) , "Production/Area"])

    plt.xlabel("Crop")

    plt.ylabel("Production per Hectare (Kg/Hectare)")

    plt.show()
#Now i want to plot bar chart of production of crops per season.

District_Name = dataset['District_Name'].unique()

for district in District_Name:

    plt.figure(figsize=(16,9))

    plt.title("Production per Season in "+district +" District")

    sns.barplot(dataset.loc[dataset.District_Name == district , "Season"],dataset.loc[dataset.District_Name == district , "Production/Area"])

    plt.xlabel("Season")

    plt.ylabel("Production per Hectare (Kg/Hectare)")

    plt.show()
#Now i want to plot bar chart of production of perticular crop per season in every district. 

#Also minimum 2.5 Kg/hectare production is require to show in our graph.

District_Name = dataset['District_Name'].unique()

for district in District_Name:

    Season = dataset.loc[(dataset.District_Name == district),"Season"].unique()

    for Season in Season:   

        plt.figure(figsize=(16,9))

        plt.title("Production of Crops per Hectare in "+district +" District in "+Season+" Season")

        sns.barplot(dataset.loc[(dataset.Season == Season) & (dataset["Production/Area"] >= 2.5)  , "Crop"],dataset.loc[(dataset.Season == Season) & (dataset["Production/Area"] >= 2.5) , "Production/Area"])

        plt.xlabel("Crop")

        plt.ylabel("Production per Hectare (Kg/Hectare)")

        plt.show()