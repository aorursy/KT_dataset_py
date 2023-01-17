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
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,mean_squared_error


df=pd.read_csv('../input/car-price-prediction/CarPrice_Assignment.csv')
df=df.drop(['car_ID'],axis=1)
df.head()
df['CarName'] = df['CarName'].str.split(' ',expand=True)
df['CarName'] = df['CarName'].replace({'maxda':'mazda','nissan':'Nissan','porcshce':'porsche','toyouta':'toyota','vokswagen':'volkswagen','vw':'volkswagen'})
df['symboling']=df['symboling'].astype(str)
#checking for duplicates
df.loc[df.duplicated()]
#segregation of numerical and categorical variable
num=df.select_dtypes(exclude=['object']).columns
cat=df.select_dtypes(include=['object']).columns
df_num=df[num]
df_cat=df[cat]
plt.figure(figsize=(15,15))
df_cat['CarName'].value_counts().plot(kind='bar')
plt.xlabel('CarName',fontweight='bold',fontsize=18)
plt.ylabel('Count',fontweight='bold',fontsize=18)
#visualisation of distribution of car price
plt.figure(figsize=(15,15))
sns.distplot(df['price'])
plt.xlabel('Price',fontweight='bold')
sns.pairplot(df_num)
df_cat.columns
plt.figure(figsize=(15,15))
plt.subplot(3,3,1)
plt.bar(df['fuelsystem'],df['price'])
plt.xlabel('fuelsystem')
plt.subplot(3,3,2)
plt.bar(df['fueltype'],df['price'])
plt.xlabel('fueltype')
plt.subplot(3,3,3)
plt.bar(df['aspiration'],df['price'])
plt.xlabel('aspiration')
plt.subplot(3,3,4)
plt.bar(df['doornumber'],df['price'])
plt.xlabel('doornumber')
plt.subplot(3,3,5)
plt.bar(df['carbody'],df['price'])
plt.xlabel('carbody')
plt.subplot(3,3,6)
plt.bar(df['drivewheel'],df['price'])
plt.xlabel('drivewheel')
plt.subplot(3,3,7)
plt.bar(df['enginelocation'],df['price'])
plt.xlabel('enginelocation')
plt.subplot(3,3,8)
plt.bar(df['enginetype'],df['price'])
plt.xlabel('enginetype')
plt.subplot(3,3,9)
plt.bar(df['cylindernumber'],df['price'])
plt.xlabel('cylinder number')







df_cat.columns
x=df_num['price']
y=df_cat['fueltype']
plt.figure(figsize=(15,15))
plt.subplot(4,3,1)
df_cat['CarName'].value_counts().plot(kind='bar')
plt.subplot(4,3,2)
df_cat['fueltype'].value_counts().plot(kind='bar')
plt.subplot(4,3,3)
df_cat['aspiration'].value_counts().plot(kind='bar')
plt.subplot(4,3,4)
df_cat['doornumber'].value_counts().plot(kind='bar')
plt.subplot(4,3,5)
df_cat['carbody'].value_counts().plot(kind='bar')
plt.subplot(4,3,6)
df_cat['drivewheel'].value_counts().plot(kind='bar')
plt.subplot(4,3,7)
df_cat['enginelocation'].value_counts().plot(kind='bar')
plt.subplot(4,3,8)
df_cat['enginetype'].value_counts().plot(kind='bar')
plt.subplot(4,3,9)
df_cat['cylindernumber'].value_counts().plot(kind='bar')
plt.subplot(4,3,10)
df_cat['fuelsystem'].value_counts().plot(kind='bar')


sns.countplot('fuelsystem',hue='fueltype',data=df_cat)
plt.figure(figsize=(20,15))
sns.countplot('CarName',hue='fueltype',data=df)
plt.figure(figsize=(20,15))
sns.countplot('CarName',hue='carbody',data=df_cat)
plt.figure(figsize=(18,10))
sns.countplot('CarName',hue='enginelocation',data=df_cat)
plt.figure(figsize=(20,10))
sns.countplot('CarName',hue='enginetype',data=df_cat)
plt.figure(figsize=(20,10))
sns.countplot('CarName',hue='cylindernumber',data=df_cat)
plt.xlabel('No of cylinders',fontweight='bold')
plt.ylabel('Count',fontweight='bold')
plt.figure(figsize=(20,10))
sns.countplot('CarName',hue='fuelsystem',data=df_cat)
plt.figure(figsize=(20,10))
sns.countplot('carbody',hue='doornumber',data=df_cat)
plt.figure(figsize=(20,10))
sns.countplot('carbody',hue='enginelocation',data=df_cat)
plt.figure(figsize=(20,10))
sns.countplot('carbody',hue='cylindernumber',data=df_cat)

plt.figure(figsize=(20,10))
sns.countplot('enginelocation',hue='doornumber',data=df_cat)
df_cat.columns
df_num.columns

for i in range(0,len(df.price)):
    if df['price'][i] <= 15000:
        df.loc[i,'car'] = 'cheap'
    elif (df['price'][i] > 15000) & (df['price'][i] <= 30000):
        df.loc[i,'car'] = 'affordable'
    elif (df['price'][i] > 30000) & (df['price'][i] <= 45000):
        df.loc[i,'car'] = 'expensive'
    elif df['price'][i] > 45000:
        df.loc[i,'car'] = 'luxury'
col= ['wheelbase', 'carlength','carwidth','curbweight',
      'horsepower','citympg','boreratio','highwaympg',
      'enginetype','cylindernumber','enginesize',
      'fuelsystem', 'fueltype','aspiration','doornumber',
      'carbody', 'drivewheel', 'enginelocation','car','price']
df=df[col]
df=pd.get_dummies(df,drop_first=True)
df.columns
col=['wheelbase', 'carlength', 'carwidth', 'curbweight', 'horsepower',
       'citympg', 'boreratio', 'highwaympg', 'enginesize',
       'enginetype_dohcv', 'enginetype_l', 'enginetype_ohc', 'enginetype_ohcf',
       'enginetype_ohcv', 'enginetype_rotor', 'cylindernumber_five',
       'cylindernumber_four', 'cylindernumber_six', 'cylindernumber_three',
       'cylindernumber_twelve', 'cylindernumber_two', 'fuelsystem_2bbl',
       'fuelsystem_4bbl', 'fuelsystem_idi', 'fuelsystem_mfi',
       'fuelsystem_mpfi', 'fuelsystem_spdi', 'fuelsystem_spfi', 'fueltype_gas',
       'aspiration_turbo', 'doornumber_two', 'carbody_hardtop',
       'carbody_hatchback', 'carbody_sedan', 'carbody_wagon', 'drivewheel_fwd',
       'drivewheel_rwd', 'enginelocation_rear', 'car_cheap', 'car_expensive',
       'car_luxury','price']
df=df[col]
df.info()
y=df.iloc[:,41].values
x=df.iloc[:,0:41].values
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.20,random_state=0)
sc=StandardScaler()
train_x=sc.fit_transform(train_x)
test_x=sc.transform(test_x)

model=LinearRegression()
a=model.fit(train_x,train_y)
y_pred=model.predict(test_x)
y_pred
test_y
r2_score(y_pred,test_y)
model.intercept_
model.coef_
