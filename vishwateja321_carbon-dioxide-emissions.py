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
import matplotlib.pyplot as plt
import seaborn as sns
file = "/kaggle/input/co2-emission-by-vehicles/CO2 Emissions_Canada.csv"
df = pd.read_csv(file)
df.head()
df.info()
df.describe(include='all')
df.hist(figsize=(20,10),bins=50)
df.isna().sum()
df['Transmission'].unique()
df['Transmission'] = np.where(df['Transmission'].isin(['A4','A5','A6','A7','A8','A9','A10']),"Automatic",df['Transmission'])
df['Transmission'] = np.where(df['Transmission'].isin(["AM5", "AM6", "AM7", "AM8", "AM9"]),"Automated Manual",df['Transmission'])
df['Transmission'] = np.where(df['Transmission'].isin(["AS4", "AS5", "AS6", "AS7", "AS8", "AS9", "AS10"]),"Automatic with Select Shift",df['Transmission'])
df['Transmission'] = np.where(df['Transmission'].isin(["AV", "AV6", "AV7", "AV8", "AV10"]),"Continuously Variable",df['Transmission'])
df['Transmission'] = np.where(df['Transmission'].isin(["M5", "M6", "M7"]),"Manual",df['Transmission'])
df['Transmission'].unique()
sns.distplot(df['CO2 Emissions(g/km)'])
sns.violinplot(df['CO2 Emissions(g/km)'])
plt.figure(figsize=(20,5))
df.groupby(['Make'])['Make'].count().sort_values(ascending=False).plot(kind='bar')
df.groupby(['Model'])['Model'].count().sort_values(ascending=False).head(10)
plt.figure(figsize=(15,5))
df.groupby(['Vehicle Class'])['Vehicle Class'].count().sort_values().plot(kind='bar')
plt.tight_layout()
plt.xticks(fontsize=8,rotation=45)
plt.figure(figsize=(15,5))
df.groupby(['Cylinders'])['Cylinders'].count().sort_values(ascending=False).plot(kind='bar')
df.groupby(['Transmission'])['Transmission'].count().sort_values(ascending=False).plot(kind='bar')
df.groupby(['Fuel Type'])['Fuel Type'].count().sort_values(ascending=False).plot(kind='bar')
plt.figure(figsize=(20,5))
df.groupby(['Make'])['CO2 Emissions(g/km)'].mean().sort_values(ascending=False).plot(kind='bar')
plt.ylabel('CO2 Emissions(g/km)')
plt.figure(figsize=(10,3))
df.groupby(['Vehicle Class'])['CO2 Emissions(g/km)'].mean().sort_values(ascending=False).plot(kind='bar')
plt.xticks(fontsize=8)
plt.ylabel("CO2 Emissions(g/km)")
plt.figure(figsize=(15,3))
df.groupby(['Engine Size(L)'])['CO2 Emissions(g/km)'].median().sort_values(ascending=True).plot(kind='bar')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emissions(g/km)')
df.groupby(['Cylinders'])['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.ylabel('CO2 Emissions(g/km)')
df.groupby(['Transmission'])['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.ylabel('CO2 Emissions(g/km)')
df.groupby(['Fuel Type'])['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.ylabel('CO2 Emissions(g/km)')
plt.figure(figsize=(25,5))
df.groupby(['Fuel Consumption City (L/100 km)'])['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')
plt.ylabel("CO2 Emissions(g/km)")
plt.figure(figsize=(20,5))
df.groupby(['Fuel Consumption Hwy (L/100 km)'])['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')
plt.ylabel("CO2 Emissions(g/km)")
plt.figure(figsize=(20,5))
df.groupby(['Fuel Consumption Comb (mpg)'])['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')
plt.ylabel("CO2 Emissions(g/km)")
sns.scatterplot(df['Fuel Consumption City (L/100 km)'],df['CO2 Emissions(g/km)'],hue=df['Fuel Type'])
sns.scatterplot(df['Fuel Consumption Hwy (L/100 km)'],df['CO2 Emissions(g/km)'],hue=df['Fuel Type'])
sns.scatterplot(df['Fuel Consumption Comb (L/100 km)'],df['CO2 Emissions(g/km)'],hue=df['Engine Size(L)'])
sns.scatterplot(df['Fuel Consumption Comb (mpg)'],df['CO2 Emissions(g/km)'],hue=df['Engine Size(L)'])
plt.figure(figsize=(8,6))
sns.boxplot(df['Fuel Type'],df['CO2 Emissions(g/km)'])
sns.pointplot(df['Cylinders'],df['CO2 Emissions(g/km)'])
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True)
Ft = pd.get_dummies(df['Fuel Type'],drop_first=True,prefix='Fuel')
df = df.drop(['Fuel Type'],axis=1)
df = pd.concat([df,Ft],axis=1)
Tr = pd.get_dummies(df['Transmission'],drop_first=True)
df = df.drop(['Transmission'],axis=1)
df = pd.concat([df,Tr],axis=1)
df.head()
X = df.drop(['CO2 Emissions(g/km)','Fuel Consumption Comb (L/100 km)'],axis=1)
y = df['CO2 Emissions(g/km)']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
x_train.head()
cat_cols = ['Make','Model','Vehicle Class']
import category_encoders as ce
target_enc = ce.CatBoostEncoder(cols = cat_cols)
target_enc.fit(x_train[cat_cols],y_train)
train_enc = target_enc.transform(x_train[cat_cols])
test_enc = target_enc.transform(x_test[cat_cols])
train_enc.head()
x_train = x_train.drop(['Make','Model','Vehicle Class'],axis=1)
x_test = x_test.drop(['Make','Model','Vehicle Class'],axis=1)
x_train = pd.concat([x_train,train_enc],axis=1)
x_test = pd.concat([x_test,test_enc],axis=1)
x_train.head()
x_test.head()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
best_features = SelectKBest(score_func=chi2)
fit = best_features.fit(x_train,y_train)
best = pd.DataFrame(fit.scores_,columns=['scores'])
best['var'] = x_train.columns
best.sort_values(by='scores' ,ascending=False)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xs_train = scaler.fit_transform(x_train)
xs_test = scaler.fit_transform(x_test)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model1 = lr.fit(x_train,y_train)
y_pred1 = model1.predict(x_test) 
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
r2_score(y_test,y_pred1)
mse = mean_squared_error(y_test,y_pred1)
mse
rmse = np.sqrt(mse)
rmse
mae = mean_absolute_error(y_test,y_pred1)
mae
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=3)
model2 = knn.fit(xs_train,y_train)
y_pred2 = model2.predict(xs_test)
r2_score(y_test,y_pred2)
mean_squared_error(y_test,y_pred2)
from sklearn.svm import LinearSVR
svr = LinearSVR()
model3 = svr.fit(xs_train,y_train)
y_pred3 = model3.predict(xs_test)
r2_score(y_test,y_pred3)
mean_squared_error(y_test,y_pred3)
from sklearn.tree import DecisionTreeRegressor
dtree = DecisionTreeRegressor()
model4 = dtree.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
r2_score(y_test,y_pred4)
mean_squared_error(y_test,y_pred4)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100)
model5 = rf.fit(x_train,y_train)
y_pred5 = model5.predict(x_test)
r2_score(y_test,y_pred5)
mean_squared_error(y_test,y_pred5)
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
model6 = gb.fit(x_train,y_train)
y_pred6 = model6.predict(x_test)
r2_score(y_test,y_pred6)
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor()
model7 = ada.fit(x_train,y_train)
y_pred7 = model7.predict(x_test)
r2_score(y_test,y_pred7)
mean_squared_error(y_test,y_pred7)

