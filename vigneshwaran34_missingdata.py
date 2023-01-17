import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("../input/diabetes.csv")
df.describe()
columns = df.columns
columns
d=df[columns]==0
print(d.sum())
df[columns[1:6]] = df[columns[1:6]].replace('NaN',np.NaN)
df.isnull().sum()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

values=df.values
X= values[:,:8]
y=values[:,8]
model = LinearDiscriminantAnalysis()
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=True,test_size=0.3)
kfold = KFold(random_state=7,n_splits=3)
result = cross_val_score(model, X, y, scoring='accuracy')
result = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(result.mean())
df1=df.copy()
df1.fillna(df.mean(),inplace=True)
df1.isnull().sum()
df2 = df.copy()
from sklearn.preprocessing import Imputer
imputer=Imputer()
values2 = df2.values
transformed_values_values=imputer.fit_transform(values2)
print(np.isnan(transformed_values_values).sum())
model = LinearDiscriminantAnalysis()
kfold = KFold(n_splits=3, random_state=7)
result = cross_val_score(model, transformed_values_values, y, cv=kfold, scoring='accuracy')

print(result.mean())

