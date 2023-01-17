#taken from
#https://www.youtube.com/watch?v=aZsZrkIgan0&feature=youtu.be
#import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as pt
%matplotlib inline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as knn
#import files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#read files
#Reading train file:
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
#Reading test file:
test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
#Reading sample sub file:
sample_sub = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
train
train.info()
test
sample_sub
#model
clf=DecisionTreeClassifier()
#set training file lengths
dt = train.values
dt = train.astype('int64')
train_size = int(len(dt) * 0.5)
#training dataset lengths
xtrain=train.iloc[0:train_size,1:]
train_label=train.iloc[0:train_size,0]
#fit model
clf.fit(xtrain,train_label)
#testing data lengths
xtest=train.iloc[train_size:,1:]
actual_label=train.iloc[train_size,0]
#plot image
#d=xtest.iloc[8]
#d=d.values.reshape(28,28)
#pt.imshow(255-d,cmap='gray')
#print(clf.predict([xtest[8]]))
#pt.show()
p=clf.predict(xtest)
p
p.shape
count=0
for i in range(0,train_size):
    count+=1 if p[i]==actual_label else 0
print("Accuracy= ", (count/train_size)*100)
#set test file lengths
tt = test.values
tt = test.astype('int64')
test_size = int(len(tt))
#testing data lengths
xtest1=test.iloc[test_size:,1:]
#predict on test set
p1=clf.predict(test)
p1
p1.shape
submission = pd.DataFrame({
        "ImageId": test.index+1,
        "Label": p1
    })
submission.to_csv('submission.csv', index=False)
submission