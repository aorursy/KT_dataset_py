#import panda class

import pandas as pd
#load Iris data set

data = pd.read_csv('../input/Iris.csv')

#Drop Id columns beacause it is not necessary since index will be populated automatically

data.drop(columns='Id', inplace=True)

#Display first 5 row to check the data set

data.head()
#Use get_dummies method from panda class to create dummy variable and store those in dummies

dummies = pd.get_dummies(data.Species)

#Display last 5 dummies

dummies.tail()
#Concate the newly created dummy variables with the loaded data and store in merged_data

merged_data = pd.concat([data,dummies], axis=1)

#Display merged_data

merged_data.head()
#Drop the Species column since we've done one hot encoding, hence Species column is not necessary and store in final_data

final_data = merged_data.drop(columns='Species')

#Display first 5 rows of the final_data

final_data.head()