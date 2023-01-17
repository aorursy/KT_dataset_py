import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
startups = pd.read_csv("../input/50-startupscsv/50_Startups.csv",sep=",") 
df = startups.copy()

df.head()
df.info()
df.shape
df.isna().sum()
corr=df.corr()
corr

sns.heatmap (corr,color="yellow");
sns.scatterplot(x="R&D Spend" , y="Profit" , color="yellow",data=df);
df.hist(figsize = (13,13),color="yellow")

plt.show()
df.describe()

df.State.unique()
dummydf=pd.get_dummies(df.State)
df_State.head()
df.drop('State', axis=1 , inplace =True) #drop ile silme işlemlerini yapıyoruz.
df=pd.concat([df,df_State],axis=1)
df.head()
XBagimsiz = df[["Marketing Spend","R&D Spend"]]
XBagimli = df[["Profit"]]
XBagimsiz

XBagimli
from sklearn.model_selection import train_test_split

XBagimsiz_train, XBagimsiz_test, XBagimli_train, XBagimli_test = train_test_split(XBagimsiz,XBagimli, test_size = 0.25, random_state = 2, shuffle=1)
XBagimsiz_train
XBagimsiz_test
XBagimli_train
XBagimli_test
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(XBagimsiz_train ,XBagimli_train)
y_pred=model.predict(XBagimsiz_test)
df_tah_ger = pd.DataFrame({"Gerçek" : XBagimli_test, "Tahmin Edilen" : y_pred})

df_tah_ger

import sklearn.metrics as metrics
print("(MAE):", metrics.mean_absolute_error(XBagimli_test,y_pred))
print("(MSE):", metrics.mean_squared_error(XBagimli_test ,y_pred))
print("(RMSE):", np.sqrt(metrics.mean_squared_error(XBagimli_test, y_pred)))
model.score(XBagimsiz_train, XBagimli_train)