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
import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

filepath= '/kaggle/input/pima-indians-diabetes-database/diabetes.csv'

dataset = pd.read_csv(filepath)

dataset
##Remove NaN values



zeronotacc=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']

for column in zeronotacc:

    dataset[column]= dataset[column].replace(0,np.NaN)

    mean= int (dataset[column].mean(skipna=True))

    dataset[column]= dataset[column].replace(np.NaN,mean)

    

    

    
##define the indepandant and depandant variables

x=dataset.iloc[:,0:8]

y=dataset.iloc[:,8]


x_train,x_test,y_train,y_test= train_test_split(x,y,random_state=0,test_size=0.2)
sc_x= StandardScaler()

x_train= sc_x.fit_transform(x_train)

x_test= sc_x.fit_transform(x_test)



import math

math.sqrt(len(y_test))

### Therefore we use 11 as the nearest neighbor
classify= KNeighborsClassifier (n_neighbors=11, p =2, metric= 'euclidean')

classify.fit(x_train,y_train)
ypred=classify.predict(x_test)

cm= confusion_matrix(y_test,ypred)

cm
print('f1 score:')

print(f1_score(y_test,ypred))
print('accuracy score:')

print(accuracy_score(y_test,ypred))