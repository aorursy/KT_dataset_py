# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
dataset='../input/train.csv'
data='../input/test.csv'
dataset=pd.read_csv(dataset)
data=pd.read_csv(data)
X=dataset[['Pclass','Sex','Age','SibSp','Parch','Embarked']]
XT=data[['Pclass','Sex','Age','SibSp','Parch','Embarked']]
y=dataset.Survived




# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_encoder_x1=LabelEncoder()
X['Sex']=label_encoder_x1.fit_transform(X['Sex'])
X=np.array(X)

for i in X:
    if(i[-1]=='S'):
        i[-1]=1
    elif(i[-1]=='Q'):
       i[-1]=2
    elif(i[-1]=='C'):
       i[-1]=3
    elif(i[-1]==''):
        i[-1]=1
        
        
label_encoder_x2=LabelEncoder()
XT['Sex']=label_encoder_x2.fit_transform(XT['Sex'])
XT=np.array(XT)

for i in XT:
    if(i[-1]=='S'):
        i[-1]=1
        
    elif(i[-1]=='Q'):
       i[-1]=2
    elif(i[-1]=='C'):
       i[-1]=3
    elif(i[-1]==''):
        i[-1]=1



from sklearn.preprocessing import Imputer
my_imputer=Imputer()
X=my_imputer.fit_transform(X)
XT=my_imputer.fit_transform(XT)

  
ohe=OneHotEncoder(categorical_features=[5])
X=ohe.fit_transform(X).toarray()
X=pd.DataFrame(X)

ohe=OneHotEncoder(categorical_features=[5])
XT=ohe.fit_transform(XT).toarray()
XT=pd.DataFrame(XT)


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)
XT=sc.fit_transform(XT)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(output_dim=4,init='uniform',activation='relu',input_dim=8))
classifier.add(Dense(output_dim=4,init='uniform',activation='relu'))
classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(X,y,batch_size=5,nb_epoch=15000)
x=classifier.predict(XT)
s=[]
for i in x:
    for j in i:
        if(j>=0.5):
            s.append(1)
        else:
            s.append(0)
A=data[['PassengerId']]
A['Survived']=s
my_submission=pd.DataFrame({'PassengerId':A.PassengerId,'Survived':A.Survived})
my_submission.to_csv('submission.csv',index=False)
            
