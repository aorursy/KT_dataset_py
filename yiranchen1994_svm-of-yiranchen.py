import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
%matplotlib inline
data_train=pd.read_csv("/Users/yiranchen/kaggle/1_Digit_Recog/DATA/train.csv")
train_limage,test_limage=train_test_split(data_train,train_size=0.8,random_state=0)
train_image=train_limage.iloc[:,1:]

train_label=train_limage.iloc[:,0:1]

test_image=test_limage.iloc[:,1:]

test_label=test_limage.iloc[:,0:1]


p=np.array(train_image.values)



p=p.reshape(p.shape[0],28,28)

plt.imshow(p[0],cmap="gray")

classifier=svm.SVC()
train_label.values.ravel().shape
classifier.fit(train_image,train_label.values.ravel())
classifier.score(test_image.iloc[:500,:],test_label.iloc[:500,:].values.ravel())
train_image=train_image.where(train_image==0,1)

test_image=test_image.where(test_image==0,1)
classifier.fit(train_image.iloc[0:5000,:],train_label.iloc[0:5000,:].values.ravel())
classifier.score(test_image.iloc[:500,:],test_label.iloc[:500,:].values.ravel())
DataTest=pd.read_csv("/Users/yiranchen/kaggle/1_Digit_Recog/DATA/test.csv")

DataTest=DataTest.where(DataTest==0,1)
Results=classifier.predict(DataTest)
p2=np.array(DataTest.values)

p2=p2.reshape(p2.shape[0],28,28)

plt.imshow(p2[3],cmap="gray")
sub={"ImageID":DataTest.index.values+1,"Label":Results}

sub=pd.DataFrame(sub)
sub.to_csv("/Users/yiranchen/kaggle/1_Digit_Recog/DATA/results.csv",header=True,index=False)