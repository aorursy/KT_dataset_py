import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
dataset = pd.read_csv("../input/crop-production-in-india-statevise/crop_production.csv")
dataset.isnull().sum()
from sklearn.impute import SimpleImputer

missingvalue = SimpleImputer(missing_values = np.nan , strategy = 'mean', verbose=0)

missingvalue = missingvalue.fit(np.array(dataset['Production']).reshape(len(dataset['Production']),1))

dataset['Production'] = missingvalue.transform(np.array(dataset['Production']).reshape(len(dataset['Production']),1)) 
dataset.isnull().sum()
#number of sample as per state

dataset.State_Name.value_counts().plot(kind='bar', title="No. of sample as per state")
#District wise Graphical representation of production per season 

District_Name = dataset.District_Name.unique()

for dist in District_Name:

    plt.bar(dataset.loc[dataset.District_Name == dist , "Season"],dataset.loc[dataset.District_Name == dist , "Production"])

    plt.title("Season vs Production Graph of "+ dist + " District")

    plt.xlabel("Season")

    plt.ylabel("Production")

    plt.show()
#production per state

State_Name = dataset.State_Name.unique()

for state in State_Name:

    plt.bar(dataset.loc[dataset.State_Name == state , "Season"],dataset.loc[dataset.State_Name == state , "Production"])

    plt.title("Season vs Production Graph of "+ state + " State")

    plt.xlabel("Season")

    plt.ylabel("Production")

    plt.show()
#crops in our dataset

crop_name = dataset.Crop.unique()

crop_name
#number of crops throughout entier india  

crop_name.shape
#Graphical representation of area of planting of perticuler crops

#by using this graph we can find that which crop have highest planting in india

plt.barh(dataset["Crop"],dataset["Area"])

plt.title("Crop vs Area of planting")

plt.xlabel("Area")

plt.ylabel("Crop")

plt.show()
#Graphical representation of area of planting of perticuler crops per state

#by using this graph we can find that which crop have highest planting in perticuler state

for state in State_Name:

    plt.barh(dataset.loc[dataset.State_Name == state,"Crop"],dataset.loc[dataset.State_Name == state ,"Area"])

    plt.title("Crop vs Area of planting in "+ state)

    plt.xlabel("Area")

    plt.ylabel("Crop")

    plt.show()