import numpy as np 

import pandas as pd 

import matplotlib.pylab as plt

import seaborn as sns

from sklearn.metrics import f1_score



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Reading in the datasets

train = pd.read_csv("../input/dont-overfit-ii/train.csv")

test = pd.read_csv("../input/dont-overfit-ii/test.csv")
# Taking a look at the structure of the training dataset and testing dataset

display(train.shape)

display(test.shape)
train.head()
test.head()
# Plotting the target distribution

train.groupby('target').count()['id'].plot(kind='bar',color=['lightpink','lightblue'])

plt.ylabel('Count')

plt.title('Target Distribution')

plt.show()
# Preparing the data

X=train.drop(['id','target'],axis=1)

y=train['target']
# Splitting the dataset in training and validation

from sklearn.model_selection import train_test_split

random_seed = 4

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state=random_seed)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



rf = RandomForestClassifier(n_estimators = 100)

rf.fit(X_train, y_train)



pred = rf.predict(X_val)

display(pred)

validation = y_val.values

display(validation)
counter = 0

for i in range(len(pred)):

    if pred[i]==validation[i]:

        counter+=1

print("Accuracy is:","{0:.0%}".format(counter/len(pred)))

f1_score(validation,pred)
X_test=test.drop(['id'],axis=1)

testPred = rf.predict(X_test)

testPred
submissions=pd.DataFrame({"id": test["id"], "target": testPred})

submissions.to_csv("submission.csv", index=False, header=True)