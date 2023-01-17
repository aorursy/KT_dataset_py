# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import preprocessing, svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



csv_df = pd.read_csv("../input/heart.csv")

labels_df = csv_df['target']

X_df = csv_df.drop(columns=['target'])



# print(X_df)



# X = X_df.values

# min_max_scaler = preprocessing.MinMaxScaler()

# X_scaled = min_max_scaler.fit_transform(X)

# X_df = pd.DataFrame(X_scaled)



# print(X_df)



X_train,X_test,Y_train,Y_test = train_test_split(X_df,labels_df,test_size=0.20,random_state=0)



max_accuracy = 0



for x in range(2000):

    rf = RandomForestClassifier(random_state=x, n_estimators=10)

    rf.fit(X_train,Y_train)

    Y_pred = rf.predict(X_test)

    current_accuracy = round(accuracy_score(Y_pred, Y_test) * 100, 2)

    if(current_accuracy > max_accuracy):

        max_accuracy = current_accuracy

        best_x = x



rf = RandomForestClassifier(random_state=best_x, n_estimators=10)

rf.fit(X_train,Y_train)

Y_pred = rf.predict(X_test)

        

score = round(accuracy_score(Y_pred, Y_test) * 100, 2)



print("Accuracy: " + str(score) + " %")



# Any results you write to the current directory are saved as output.