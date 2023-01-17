# import all package



import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import sys

import os
# Read csv data from local disk

df = pd.read_csv("../input/Admission_Predict.csv")



# This line change the name of column i.e. "Chance of Admit " to "Chance of Admit" ,in last we remove just space

df=df.rename(columns = {'Chance of Admit ':'Chance of Admit'})



# it may be needed in the future.

serialNo = df["Serial No."].values



# Delete serial number from dataframe

df.drop(["Serial No."],axis=1,inplace = True)



df.head()
# It gives statistical overview of given data



df.describe()
# Check if there is null or not and if there is null then i will give total number of null values i.e. count



df.isnull().sum()
# Correlation between different varaibles



fig,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")

plt.show()
# Values of correlation w.r.t Chance of Admit 



cor=df.corr()['Chance of Admit'].sort_values(ascending=False)

cor
print("Not Having Research:",len(df[df.Research == 0]))

print("Having Research:",len(df[df.Research == 1]))

y=[len(df[df.Research==0]),len(df[df.Research==1])]

x = ["Not Having Research","Having Research"]

plt.bar(x,y)

plt.title("Research Experience")

plt.xlabel("Canditates")

plt.ylabel("Frequency")

plt.show()
# Scatter plot between CGPA vs GRE score

# df.GRE Score did not work because there is space between GRE and Score ,so i use df["GRE Score"]



plt.scatter(df["GRE Score"],df.CGPA)

plt.title("CGPA for GRE Scores")

plt.xlabel("GRE Score")

plt.ylabel("CGPA")

plt.show()
# Number of candidates w.r.t university and their 75% acceptance chance

# value_counts() mean return a Series containing counts of unique values



s = df[df["Chance of Admit"] >= 0.75]["University Rating"].value_counts().head(5)

plt.title("University Ratings of Candidates with an 75% acceptance chance")

s.plot(kind='bar',figsize=(20, 10))

plt.xlabel("University Rating")

plt.ylabel("Candidates")

plt.show()
# Average score needed for candidate to admit 

# If they have average score given below then there is chance of 90% to get admission



df[(df['Chance of Admit']>0.90)].mean()
# Modify column data in dataframe 

# Replace value>=0.75 into 1 

# Replace value<0.75 into 0

df.loc[df['Chance of Admit']>=0.75,['Chance of Admit']]=1

df.loc[df['Chance of Admit']<0.75,['Chance of Admit']]=0
import keras



# X is input for our model

X=df.iloc[:,:-1].values



# y is output for our model

y=df.iloc[:,-1].values
# Split dataset in train and test



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.7,random_state=1)
from keras.models import Sequential

from keras.layers.core import Dense, Activation





# Initialising the ANN

classifier = Sequential()



# Adding the input layer and the first hidden layer

classifier.add(Dense(output_dim = 128, init = 'uniform', activation = 'relu', input_dim = 7))



# Adding the second hidden layer

classifier.add(Dense(output_dim = 32, init = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.75)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred)

cm