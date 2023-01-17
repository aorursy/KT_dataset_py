

import numpy as np # linear algebra
import pandas as pd 
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
datadict = datasets.load_breast_cancer()
datadict.keys()
x=datadict['data']
y=datadict['target']
pd.DataFrame(x,columns=datadict['feature_names']).head()
x_train,x_test,y_train,y_test=train_test_split(x,y,
                                              test_size=0.2,
                                              random_state=5)
print("Training data :",x_train.shape,y_train.shape)
print("Testing data :",x_test.shape,y_test.shape)
model=linear_model.LogisticRegression()
model.fit(x_train,y_train)
predictions=model.predict(x_test)
accuracy=np.mean(y_test==predictions)
print('The accuracy is :',accuracy)
