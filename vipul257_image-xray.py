# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
os.listdir('../input/chest-xray-pneumonia/chest_xray/train/NORMAL')
import os
labels = []
for i in os.listdir('../input/chest-xray-pneumonia/chest_xray/train/NORMAL'):
    labels.append(0)
for i in os.listdir('../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'):
    labels.append(1)
labels
import cv2
loc1 = '../input/chest-xray-pneumonia/chest_xray/train/NORMAL'
loc2 = '../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA'
features = []
from tqdm import tqdm
for i in tqdm(os.listdir(loc1)):
    f1 = cv2.imread(os.path.join(loc1,i),0)
    f1 = cv2.resize(f1,(100,100))
    features.append(f1)
    
for i in tqdm(os.listdir(loc2)):
    f2 = cv2.imread(os.path.join(loc2,i),0)
    f2 = cv2.resize(f2,(100,100))
    features.append(f2)
import numpy as np
X = np.array(features)
Y = np.array(labels)
Y.shape = (5216,1)
X.shape = (5216,10000)
chest_data = np.hstack((Y,X))

import pandas as pd
chest_data = pd.DataFrame(chest_data)
chest_data.to_csv('chest_data.csv')
pd.read_csv('./chest_data.csv',index_col=0)
from keras.utils import np_utils
Xt = (X - X.mean())/X.std()
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xt,Y)
from sklearn.ensemble import RandomForestClassifier
rmodel = RandomForestClassifier()
rmodel.fit(xtrain,ytrain)
print(rmodel.score(xtrain,ytrain))
print(rmodel.score(xtest,ytest))
Result = np.array(['Normal','Pneumonia'])
print(Result[ytest[1]])
print('Prediction',Result[rmodel.predict([xtest[1]])])
plt.imshow(xtest[1].reshape(100,100))
plt.show()
import pickle
filename = './finalized_model.sav'
pickle.dump(rmodel, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(xtest,ytest)
print(result)
import cv2
i = cv2.imread('../input/chest-xray-pneumonia/chest_xray/train/NORMAL/NORMAL2-IM-1006-0001.jpeg')
import matplotlib.pyplot as plt
plt.imshow(i)
plt.show()
for i in os.listdir(loc1):
    print(os.path.join(loc1,i))
print(xtest[0:1,:])
