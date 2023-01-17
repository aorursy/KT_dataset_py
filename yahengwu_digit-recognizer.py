import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
display(train.shape)

display(train.head())

display(test.shape)

display(test.head())
data = train.iloc[:,1:].values

target = train.iloc[:,0].values
display(data.shape)

display(target.shape)
for index, (image, label) in enumerate(zip(data[0:5],target[0:5])):

    plt.subplot(1, 5, index + 1)

    plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)

    plt.title('Training: %i' % label)
random_seed = 4

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state=random_seed)
display(X_train.shape)

display(y_train.shape)

display(X_test.shape)

display(y_test.shape)
rf = RandomForestClassifier(n_estimators = 100)

rf.fit(X_train, y_train)
pred = rf.predict(X_test)

display(pred)
score = rf.score(X_test, y_test)

display(score)
testPred = rf.predict(test)

testPred
cm = metrics.confusion_matrix(y_test,pred)

print(cm)
plt.figure(figsize=(10,10))

sns.heatmap(cm, annot=True,fmt=".2f",linewidths=.5,square = True, cmap = 'Blues_r')

plt.xlabel('Predicted Label')

plt.ylabel('Actual Label')
submissions=pd.DataFrame({"ImageId": list(range(1,len(testPred)+1)), "Label": testPred})

submissions.to_csv("submission.csv", index=False, header=True)