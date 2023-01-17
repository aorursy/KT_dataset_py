

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df=pd.read_csv('/kaggle/input/zoo-animal-classification/zoo.csv')
df_1=pd.read_csv('/kaggle/input/zoo-animal-classification/class.csv')
df.animal_name.value_counts().count()
# data has 100 different animals 
df.class_type.value_counts().count()
# the target feature has 7 different values


from keras.utils import to_categorical
x=df.iloc[:,1:-1]
x
y=df.iloc[:,-1]
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.3,random_state=2,stratify=y)
model=SVC()
model.fit(x_train,y_train)
print(f' Training Accuracy {model.score(x_train,y_train)}')
f'Test Accuracy {model.score(x_test,y_test)}'
from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier(n_jobs=15,n_neighbors=3,p=1,weights='distance')
knn_model.fit(x_train,y_train)
print(f' Training Accuracy {knn_model.score(x_train,y_train)}')
f' Testing Accuracy {knn_model.score(x_train,y_train)}'