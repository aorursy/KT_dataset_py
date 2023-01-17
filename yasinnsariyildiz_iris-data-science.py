import pandas as pd

import numpy as np

import seaborn as sns

df = pd.read_csv("../input/iris-flower-dataset/IRIS.csv")

df.head()
df.info()
df.species = pd.Categorical(df.species)
df.describe().T
df.nunique()
df.isnull().values.any()
df.isnull().sum()
df.species.value_counts()
df_int =df.drop("species",axis = 1)
from sklearn.neighbors import LocalOutlierFactor

LOF = LocalOutlierFactor(n_neighbors = 20, contamination = 0.1)

LOF.fit_predict(df_int)
df_score = LOF.negative_outlier_factor_
esik_deger = np.sort(df_score)[15]

clear_tf =df_score > esik_deger
outliers = df[~clear_tf]

outliers
clears = df_int[clear_tf]
df_int = clears

df_int
df.info()
df.groupby("species")["sepal_length","sepal_width","petal_length","petal_width"].mean()
df.corr()
sns.pairplot(df,hue= "species");
sns.jointplot("petal_length","sepal_length",data = df,kind= "reg");
x = df.drop("species",axis=1)

y = df["species"]
from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.2)
from sklearn.linear_model import LogisticRegression
loj= LogisticRegression(solver = "liblinear")

loj_model = loj.fit(x,y)
loj_model.intercept_
loj_model.coef_
y_pred= loj_model.predict(x) 
from sklearn.metrics import accuracy_score

accuracy_score(y,y_pred)