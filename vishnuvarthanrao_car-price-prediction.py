import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
plt.figure(figsize=(10,10))        

df = pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.shape
print(df['Seller_Type'].unique())
print(df['Transmission'].unique())
df.isnull().sum()

# No Null Values 
df.columns
final_df = df.drop(columns="Car_Name")
final_df
final_df["Current_Year"] =2020
final_df
final_df["No_of_Years_Old"] = final_df["Current_Year"] - final_df["Year"]
final_df.head()
final_df.drop(["Year","Current_Year"],axis =1,inplace=True)
final_df
final_df = pd.get_dummies(final_df,drop_first=True)
final_df
final_df.corr()
import seaborn as sns

sns.pairplot(final_df)
corr=final_df.corr()
top_cor = corr.index
plt.figure(figsize=(20,20))
heat_map = sns.heatmap(final_df[top_cor].corr(),annot=True,cmap="RdYlGn")
X =final_df.drop("Selling_Price",axis=1)
Y=final_df["Selling_Price"]
X.head()
Y.head()
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,Y)
print(model.feature_importances_)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred=reg.predict(X_test)
reg.score(X_test,Y_test)