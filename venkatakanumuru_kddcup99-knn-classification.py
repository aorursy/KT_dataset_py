# dataset is taken from "https://datahub.io/machine-learning/kddcup99#resource-kddcup99_zip"



# knn clasification using k values from 1 to 40 to see which k value is good



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix



#csv to dataframe conversion 

df = pd.read_csv("../input/kddcup99.csv")



fin = []



for i in range(1,4):

    v = set(df.iloc[:,i].values)

    fin += v     

    

d = dict(enumerate(fin, start=1))

mymap = dict((v,k) for k,v in d.items())



#making string to numbers

dataset = df.applymap(lambda s: mymap.get(s) if s in mymap else s)



#selecting the columns excluding lables

X = dataset.iloc[:, :-1].values



#selecting the lables column 

y = dataset.iloc[:, -1].values



#Train Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)



#scaling by doing normalisation

scaler = StandardScaler()

scaler.fit(X_train)



X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)





#Training and Predictions using library 

classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)



#testing 

y_pred = classifier.predict(X_test)



#elavuating the results 

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))



#now k values variations

error = []



# Calculating error for K values between 1 and 40

for i in range(5, 20):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))

    

    

#ploting 

plt.figure(figsize=(12, 6))

plt.plot(range(5, 20), error, color='red', linestyle='dashed', marker='o',

         markerfacecolor='blue', markersize=10)

plt.title('Error Rate K Value')

plt.xlabel('K Value')

plt.ylabel('Mean Error')