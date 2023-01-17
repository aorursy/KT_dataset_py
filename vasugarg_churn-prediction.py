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
import pandas as pd

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
data=pd.read_csv('../input/dataset/Churn (1).csv')
data.describe(include='O')                   #describe catagorical data

data.describe()                              #describe numeric data

#here total charges column is look like in numeric but described in catagorical so convert this into float
data["TotalCharges"]=pd.to_numeric(data["TotalCharges"],errors='coerce')  
data.describe() #now this total charges have missing values
data["TotalCharges"].fillna(1397.4750,inplace=True)     #filling missing values with median
data.describe()                                  #now the data have no missing values
data=data.drop("customerID",axis=1)             #drop the customerID column because its useless

newdata=pd.get_dummies(data,drop_first=True)    #this function will encoded all the catgorical data
column=list(newdata.columns)

features=list(set(column)-set(["Churn_Yes"]))

X=newdata[features].values

Y=newdata["Churn_Yes"].values

ss=StandardScaler()

newdata=ss.fit_transform(X)

Xt,Xts,Yt,Yts=train_test_split(X,Y,test_size=0.30,random_state=0) 
for i in range(1,30):

    classifier=KNeighborsClassifier(n_neighbors=i)              #initialization of classifier with k value

    classifier.fit(Xt,Yt)                                       #training model 

    p=classifier.predict(Xts)                                   #finding predictions on test data

    acc=accuracy_score(Yts,p)

    print(acc,"at k=",i)   
    classifier=LogisticRegression(random_state=0)              #initialization of logistic classifier 

    classifier.fit(Xt,Yt)                                       #training model 

    p=classifier.predict(Xts)                                   #finding predictions on test data

    acc=accuracy_score(Yts,p)

    print(acc)