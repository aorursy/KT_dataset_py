

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import zipfile

with zipfile.ZipFile('/kaggle/input/leaf-classification/sample_submission.csv.zip') as z_samp:
    z_samp.extractall()
import zipfile

with zipfile.ZipFile('/kaggle/input/leaf-classification/train.csv.zip') as z:
    z.extractall()
with zipfile.ZipFile('/kaggle/input/leaf-classification/images.zip') as z_img:
    z_img.extractall()
    
with zipfile.ZipFile('/kaggle/input/leaf-classification/test.csv.zip') as z_test:
    z_test.extractall()
os.listdir()
len(os.listdir('images'))
import matplotlib.pyplot as plt
plt.figure(figsize=(20,15))
import cv2 as cv
from keras.preprocessing.image import load_img
for i in range(25):
    j=np.random.choice((os.listdir('images')))
    plt.subplot(5,5,i+1)
    img=load_img(os.path.join('/kaggle/working/images',j))
    plt.imshow(img)
data=pd.read_csv('train.csv',index_col=False)
test_data=pd.read_csv('test.csv',index_col=False)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


encoder=LabelEncoder()
le=encoder.fit(data.species)
labels=le.transform(data.species)
classes=list(le.classes_)
classes
test_data.columns
data=data.drop(['id','species'],axis=1)
test_id=test_data.id
test_data=test_data.drop(['id'],axis=1)
len(labels)
unique_lables=np.unique(labels)
unique_lables
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(data,labels,test_size=.2,shuffle=True,stratify=labels)
from sklearn.svm import SVC


svc_model=SVC()

svc_model.fit(x_train,y_train)
svc_model.score(x_train,y_train)
svc_model.score(x_test,y_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda=LinearDiscriminantAnalysis()

lda.fit(x_train,y_train)
lda.score(x_train,y_train)
lda.score(x_test,y_test)
predicted=lda.predict_proba(test_data)
predicted
sample_df=pd.read_csv('sample_submission.csv',index_col=False)
sample_df
df_sub=pd.DataFrame(predicted,columns=sample_df.columns[1:])
df_sub1=pd.DataFrame(test_id)
final_sub=pd.concat([df_sub1,df_sub],axis=1)

final_sub
final_sub.to_csv('leaf_sub.csv')