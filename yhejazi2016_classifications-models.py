# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import metrics 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel(os.path.join(dirname, filename))
print(df.head())
X =df.drop('group_id',axis = 1 )

X =df.drop('YEAR',axis = 1 )

y= df['group_id']

print(y)
from sklearn.model_selection import train_test_split

 



X_train ,X_test ,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 50,shuffle=False)

print(X_train)

print(y_train)
from sklearn.naive_bayes import GaussianNB

nb =  GaussianNB() 



nb.fit(X_train,y_train)

# making predictions on the testing set 

y_pred = nb.predict(X_test) 




class_pred_accuracy = metrics.accuracy_score(y_test,y_pred,True)*100 

print(class_pred_accuracy)

#averagestring, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]

print("Precision Score : ",metrics.precision_score(y_test, y_pred, pos_label=1,average='micro'))

print("Precall Score : ",metrics.recall_score(y_test, y_pred,   pos_label=1, average='micro'))
# desction tree classifier  

from sklearn.metrics import confusion_matrix 

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report

    # Decision tree with entropy 

clf_entropy = DecisionTreeClassifier() #max_depth=7

  

    # Performing training 

clf_entropy.fit(X_train, y_train)    



 # Predicton on test with giniIndex 

y_pred = clf_entropy.predict(X_test) 

print("Predicted values:") 

print(y_pred) 



print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))
print ("Accuracy : ",accuracy_score(y_test,y_pred)*100) 
#KNN classifier 

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=11)

knn.fit(X_train,y_train)



# get y predication 

y_pred = knn.predict(X_test)

y_pred.shape

print(y_pred)




model_accurecy = metrics.accuracy_score(y_test,y_pred,True)*100 

print(model_accurecy)

#averagestring, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘samples’, ‘weighted’]

print("Precision Score : ",metrics.precision_score(y_test, y_pred, pos_label=1,average='weighted'))

print("recall Score : ",metrics.recall_score(y_test, y_pred,   pos_label=1, average='weighted'))

print("Report : ", metrics.classification_report(y_test, y_pred)) 