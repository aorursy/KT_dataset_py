import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns
data=pd.read_csv("../input/Iris.csv")
data.head()
del data["Id"]
data.info()
species=data['Species'].unique()
listofcolumns=data.columns

print(listofcolumns)
listofcolumns=data.columns

listofNumericalcolumns=[]



for i in listofcolumns:

    if data[i].dtype == 'float64':

        listofNumericalcolumns.append(i)

print('listofNumericalcolumns :',listofNumericalcolumns)

print('Species:',species)





for i in range(len(listofNumericalcolumns)):

    for j in range(len(species)):  

        print(listofNumericalcolumns[i]," : ",species[j])
data.describe()
data.groupby("Species").size()
data.plot(kind='box')
data.hist(figsize=(10,5))

plt.show()
print("HIST PLOT OF INDIVIDUAL Species")

print(species)



for spice in species:

        data[data['Species']==spice].hist(figsize=(10,5))
sns.violinplot(data=data,x='Species',y='PetalLengthCm')
sns.violinplot(data=data,x='Species',y='PetalWidthCm')
sns.violinplot(data=data,x='Species',y='SepalLengthCm')
sns.violinplot(data=data,x='Species',y='SepalWidthCm')
import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier 

from sklearn.preprocessing import LabelEncoder 

from sklearn.metrics import confusion_matrix 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 

  

#Importing the dataset 

d = data.iloc[:, :] 

  

#checking for null values 

print("Sum of NULL values in each column. ") 

print(d.isnull().sum()) 

  

#seperating the predicting column from the whole dataset 

X = d.iloc[:, :-1].values 

y = data.iloc[:, 4].values 

  

#Encoding the predicting variable 

labelencoder_y = LabelEncoder() 

y = labelencoder_y.fit_transform(y) 

  

#Spliting the data into test and train dataset 

X_train, X_test, y_train, y_test = train_test_split( 

              X, y, test_size = 0.3, random_state = 0) 

  

#Using the random forest classifier for the prediction 

classifier=RandomForestClassifier() 

classifier=classifier.fit(X_train,y_train) 

predicted=classifier.predict(X_test) 

  

#printing the results 

print ('Confusion Matrix :') 

print(confusion_matrix(y_test, predicted)) 

print ('Accuracy Score :',accuracy_score(y_test, predicted)) 

print ('Report : ') 

print (classification_report(y_test, predicted)) 