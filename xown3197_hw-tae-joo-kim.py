# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        



# Any results you write to the current directory are saved as output.
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import numpy as np

import cv2 

import os

dataset_train ="/kaggle/input/2019-fall-pr-project/train/train/"



imgs = []

labels = []



for img_dir in os.listdir(dataset_train):

    print(img_dir)

    if str.find(img_dir, 'cat'):

        labels.append(0)

    else:

        labels.append(1)

    img = cv2.imread(dataset_train+img_dir)

    data = cv2.resize(img, (32, 32))

    data = np.reshape(data, -1)

    imgs.append(data)
print(len(labels[:-1]))

print(len(imgs))
x_train, x_test, y_train, y_test = train_test_split(imgs, labels, test_size=0.25, random_state=42)



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)
clf = SVC(kernel='rbf', class_weight='balanced', random_state=42)





param_grid = {'C':[0.1, 1], 'gamma':[0.0001, 0.001, 0.01]}



clf = GridSearchCV(clf, param_grid, cv=3)



#clf.fit(x_train, y_train)

clf.fit(x_test, y_test)





model = clf.best_estimator_



print(clf.best_params_)
dataset_test = "/kaggle/input/2019-fall-pr-project/test1/test1/"



imgs_test = []

test_id = []



for img_dir in os.listdir(dataset_test):

    iid, _ = img_dir.split('.')

    test_id.append(iid)

    img = cv2.imread(dataset_test+img_dir)

    data = cv2.resize(img, (32, 32))

    data = np.reshape(data, -1)

    imgs_test.append(data)
imgs_test = scaler.transform(imgs_test)
y_predict = model.predict(imgs_test)



#score = classification_report(y_test, y_predict)
print(y_predict)
df = pd.DataFrame([y_predict])

df = df.T

df
df.index = np.arange(1,len(df)+1)

df.index.name = 'id'

df