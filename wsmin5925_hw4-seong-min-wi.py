# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

       # print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

#from imutils import paths

import numpy as np

#import imutils 

import cv2 

import os

import glob

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')
dataset_train = "/kaggle/input/2019-fall-pr-project/train/train/"

#forder_path = "/content/input2/train/train/*.jpg"

forder_path = "/content/input2/test1/test1/*.jpg"



imgs_train = []

label_train = []



for img_path in os.listdir(dataset_train):

  if str.find(img_path, 'cat'):

    label_train.append(0)

  else:

    label_train.append(1)

  img = cv2.imread(dataset_train+img_path)

  #print(dataset_train+img_path)

  #img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

  data = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), (32, 32))

  data = np.reshape(data, -1)

  imgs_train.append(data)
from sklearn.preprocessing import StandardScaler, MinMaxScaler



#x_train, x_test, y_train, y_test = train_test_split(imgs_train, label_train, test_size=0.25, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(imgs_train, label_train, test_size=0.10, random_state=42)



scaler = StandardScaler()

#scaler = MinMaxScaler

x_train = scaler.fit_transform(x_train)

x_test = scaler.fit_transform(x_test)
from sklearn.metrics import classification_report

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV



clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)





param_grid = {'C':[0.01, 0.1, 0.5, 1, 5, 10], 'gamma':[0.01, 0.1,1, 5,10, 50]}



clf = GridSearchCV(clf, param_grid, cv=3)



#clf.fit(x_train, y_train)

clf.fit(x_test, y_test)





model = clf.best_estimator_



print(clf.best_params_)





y_predict = model.predict(x_test)



score = classification_report(y_test, y_predict)
dataset_test = "/kaggle/input/2019-fall-pr-project/test1/test1/"



imgs_test = []

test_id = []



for img_path in os.listdir(dataset_test):

  iid, _ = img_path.split('.')

  iid = int(iid)

  test_id.append(iid)

  img = cv2.imread(dataset_test+img_path)

  data = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), (32, 32))

  data = np.reshape(data, -1)

  imgs_test.append(data)
result = model.predict(imgs_test)

result

print(score)
import pandas as pd

result = result

print(result.shape)

print(result)

df = pd.DataFrame(result)

df.index = df.index+1

#df = pd.DataFrame(columns=['id', 'label'])

df = df.replace('dog',1)

df = df.replace('cat',0)



print(df)

f = ['label']

df2 = pd.DataFrame(f,index=['id'])



print(df2)

df3 = df2.append(df)

print(df3)

#df= df3

df3.to_csv('results-sm-v6.csv',index=True, header=False)

! kaggle competitions submit -c 2019-fall-pr-project -f results-sm-v6.csv -m "smwi-1102"