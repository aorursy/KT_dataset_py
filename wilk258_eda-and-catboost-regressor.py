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
import random
df = pd.read_csv('/kaggle/input/insurance/insurance.csv')
df.head(3)
df.shape
df.dtypes
fig, ax = plt.subplots(2,2,figsize=(15,20))

sns.countplot(x='sex',data=df, ax=ax[0,0])#, color="orangered");
ax[0,0].set_title("Gender Count");

sns.countplot(x='children',data=df, ax=ax[0,1]) #color="lightseagreen";
ax[0,1].set_title("Children Number Count");

sns.countplot(x='smoker',data=df, ax=ax[1,0]) #color="orangered");
ax[1,0].set_title("Smoker Number Count");

sns.countplot(x='region',data=df, ax=ax[1,1])# color="lightseagreen");
ax[1,1].set_title("Region Number Count");
df_age = df.groupby('age').size().reset_index(name='counts')
n = df_age['age'].unique().__len__()+1
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(100)
c = random.choices(all_colors, k=n)

plt.figure(figsize=(18,15), dpi= 80)
plt.bar(df_age['age'], df_age['counts'], color=c, width=.5)

plt.gca().set_xticklabels(df_age['age'], rotation=60, horizontalalignment= 'right')
plt.title("Number of Age", fontsize=22)
plt.ylabel('# Age')
plt.ylim(0, 80)
plt.show()
fig, ax = plt.subplots(2,2,figsize=(15,20))

sns.barplot(x="sex", y="charges", data=df, ax=ax[0,0])
ax[0,0].set_title("Gender medical charge");

sns.barplot(x="children", y="charges", data=df, ax=ax[0,1])
ax[0,1].set_title("Children Number And Medical Charge");

sns.barplot(x="smoker", y="charges", data=df, ax=ax[1,0])
ax[1,0].set_title("Smoker And Non-Smoker Medical Charge");

sns.barplot(x="region", y="charges", data=df, ax=ax[1,1])
ax[1,1].set_title("Region And Medical Charge");
fig, ax = plt.subplots(2,1,figsize=(15,20))

sns.distplot(df.charges, ax=ax[0])
ax[0].set_title("Charge Distribution");

sns.distplot(np.log(df.charges),ax=ax[1])
ax[1].set_title("Charge Log Distribution");
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
sns.stripplot(df.age, df.charges, jitter=0.25, size=8, ax=ax, linewidth=.5)

plt.title('Age Medical Charge distribution', fontsize=22)
plt.show()
fig, ax = plt.subplots(2,2,figsize=(15,20))

sns.barplot(x="sex", y="bmi", data=df, ax=ax[0,0])
ax[0,0].set_title("Gender Bmi");

sns.barplot(x="children", y="bmi", data=df, ax=ax[0,1])
ax[0,1].set_title("Children Number And Bmi");

sns.barplot(x="smoker", y="bmi", data=df, ax=ax[1,0])
ax[1,0].set_title("Smoker And Bmi");

sns.barplot(x="region", y="bmi", data=df, ax=ax[1,1])
ax[1,1].set_title("Region And Bmi");
sns.distplot(df.bmi)
plt.title("Bmi Distribution");
fig, ax = plt.subplots(figsize=(16,10), dpi= 80)    
sns.stripplot(df.age, df.bmi, jitter=0.25, size=8, ax=ax, linewidth=.5)

plt.title('Age Medical Bmi', fontsize=22)
plt.show()
fig, ax = plt.subplots(3,1,figsize=(15,20))

sns.countplot(x="sex", hue="smoker", data=df,ax=ax[0])
ax[0].set_title("Gender And Smoker");


sns.countplot(x="children", hue="smoker", data=df,ax=ax[1])
ax[1].set_title("Children Number And Smoker");

sns.countplot(x="region", hue="smoker", data=df,ax=ax[2])
ax[1].set_title("Region And Smoker");
fig, ax = plt.subplots(2,1,figsize=(15,20))

sns.countplot(x="sex", hue="region", data=df,ax=ax[0])
ax[0].set_title("Gender And region");

sns.countplot(x="children", hue="region", data=df,ax=ax[1])
ax[1].set_title("Children Number And region");

plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)

plt.title('Correlogram of df', fontsize=22)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
value = df.loc[:,['bmi', 'charges']]
train_ml=value.iloc[:int(value.shape[0]*0.95)]
valid_ml=value.iloc[int(value.shape[0]*0.95):]
from sklearn.metrics import mean_squared_error,r2_score
from catboost import CatBoostRegressor
X = np.array(train_ml["bmi"]).reshape(-1,1)
y = np.array(train_ml["charges"]).reshape(-1,1)
cb_model = CatBoostRegressor(iterations=500,
                             learning_rate=0.05,
                             depth=10,
                             random_seed = 42,
                             bagging_temperature = 0.2,
                             od_type='Iter',
                             metric_period = 50,
                             od_wait=20)
cb_model.fit(X, y)
r2_score(cb_model.predict(X), y)
plt.figure(figsize=(11,6))
prediction_cat=cb_model.predict(np.array(train_ml["bmi"]).reshape(-1,1))
plt.plot(train_ml["charges"],label="Actual charges")
plt.plot(train_ml.index,prediction_cat, linestyle='solid',label="Predicted charges using Catboost",color='black')
plt.xlabel('bmi')
plt.ylabel('charges')
plt.title("charges Linear Regression Prediction")
plt.xticks(rotation=90)
plt.legend()
from sklearn import preprocessing
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV
X = np.array(train_ml["bmi"]).reshape(-1,1)
y = np.array(train_ml["charges"]).reshape(-1,1)
kfold = KFold(n_splits=5, random_state = 2020, shuffle = True)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(X,y)
prediction_valid_bgm=model_lgb.predict(np.array(valid_ml["bmi"]).reshape(-1,1))
print("Validation LightBGM prediction:",prediction_valid_bgm)
plt.figure(figsize=(11,6))
prediction_bgm=model_lgb.predict(np.array(train_ml["bmi"]).reshape(-1,1))
plt.plot(train_ml["charges"],label="Actual Charges")
plt.plot(train_ml.index,prediction_bgm, linestyle='--',label="Predicted Charges using LightBGM",color='black')
plt.xlabel('bmi')
plt.ylabel('charges')
plt.title("Charges Linear Regression Prediction")
plt.xticks(rotation=90)
plt.legend()