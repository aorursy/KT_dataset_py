# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
g_df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)
w_df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)
print('Plant_1_Generation_Data : {} records'.format(len(g_df)))
g_df.info()
g_df.head(5)
print('Plant_1_Weather_Sensor_Data : '.format(len(w_df)))
w_df.info()
w_df.head(5)
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

fig, ax = plt.subplots(2,1, figsize=(15,6))
ax[0].set_title("POWER PLANT 1 DAILY CAPACITY")  
sns.lineplot(data=[g_df["DC_POWER"],g_df["AC_POWER"]], ax=ax[0], palette="tab20", linewidth=1)
ax[1].set_title("WEATHER ON POWER PLANT 1")  
sns.lineplot(data=[w_df["AMBIENT_TEMPERATURE"],w_df["MODULE_TEMPERATURE"],w_df["IRRADIATION"]], ax=ax[1], palette="tab20", linewidth=2.5)
gen_df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)
plan_g = gen_df.groupby('DATE_TIME').agg({'DC_POWER': 'mean','AC_POWER': 'mean','DAILY_YIELD': 'mean','TOTAL_YIELD': 'mean',}).reset_index()
plan_g.head()
weather_df = pd.read_csv('/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv', parse_dates=['DATE_TIME'], infer_datetime_format=True)
plan_w = weather_df[['DATE_TIME', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
plan_w.head()
#join Power data + Weather data
def extract_fea(dataframe):
    m_df = dataframe.copy()
    window_day = 3
    for i in range(1, window_day+1):
        m_df[f'DATE_TIME_P{i}D'] = m_df['DATE_TIME'] + datetime.timedelta(days=-i)    
    m_df['DATE_TIME_N3D'] = m_df['DATE_TIME'] + datetime.timedelta(days=3)
    m_df['DATE_TIME_N7D'] = m_df['DATE_TIME'] + datetime.timedelta(days=7)

    
    for j in range(1, window_day+1):
        m_df = m_df.join(dataframe.set_index('DATE_TIME'), how='inner', on=f'DATE_TIME_P{j}D', rsuffix=f'_P{j}D')  
    
    m_df = m_df.join(dataframe.set_index('DATE_TIME')[['TOTAL_YIELD']], how='inner', on='DATE_TIME_N3D', rsuffix='_N3D')
    m_df = m_df.join(dataframe.set_index('DATE_TIME')[['TOTAL_YIELD']], how='inner', on='DATE_TIME_N7D', rsuffix='_N7D')
    fea_col = []
    label_col = ['TOTAL_YIELD_N3D', 'DATE_TIME_N7D']
    
    for c in m_df.columns:
    
        if c.startswith('DATE_TIME'):
            continue
        
        if c in label_col:
            continue
        fea_col.append(c)
        
    X = m_df[fea_col].values
    y3day = m_df['TOTAL_YIELD_N3D'].values
    y7day = m_df['TOTAL_YIELD_N7D'].values
    return X, y3day, y7day
merge_plan1   = pd.merge(plan_g, plan_w, how='inner', left_on='DATE_TIME', right_on='DATE_TIME')
fea, pw_3day, pw_7day = extract_fea(merge_plan1)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,mean_absolute_error
# from sklearn.metrics import mean_absolute_error
def K_Fold(model, X, Y, cv=10):
    kf = KFold(n_splits=cv,random_state=None,shuffle=True)
    kf.get_n_splits(X,Y)
    
    RMSE = []
    score=[]
    MAE=[]
    
    for train_data, test_data in kf.split(X):
        X_train = X[train_data]
        X_test = X[test_data]

        Y_train = Y[train_data]
        Y_test = Y[test_data]
        
        m=model.fit(X_train, Y_train)
        Y_pred = np.round(model.predict(X_test))
        
        accur = np.sqrt(mean_squared_error(Y_test, Y_pred))
        RMSE.append(accur)

        MAE.append(mean_absolute_error(Y_test, Y_pred))
        score.append(m.score(X,Y))
    rmse = np.mean(RMSE)
    Score = np.mean(score)
    mae = np.mean(MAE)
    return rmse,Score,mae
Random_Forest = RandomForestRegressor(random_state=1)
Decision_Tree = DecisionTreeRegressor(random_state=1)
Linear = LinearRegression()
# Train Model 3 day
Random_Forest_3DAY = K_Fold(Random_Forest, fea, pw_3day, cv=10)
Decision_Tree_3DAY = K_Fold(Decision_Tree, fea, pw_3day, cv=10)
Linear_3DAY = K_Fold(Linear, fea, pw_3day, cv=10)
# Evaluate
print('Regression Result 3 Day **************')
print('\n','Random Forest : (RMSE,Score,MAE) :', Random_Forest_3DAY,'\n','Decision Tree : (RMSE,Score,MAE) :', Decision_Tree_3DAY,'\n','Linear Regression : (RMSE,Score,MAE) :', Linear_3DAY)
# Train Model 7 day
Random_Forest_7DAY = K_Fold(Random_Forest, fea, pw_7day, cv=10)
Decision_Tree_7DAY = K_Fold(Decision_Tree, fea, pw_7day, cv=10)
Linear_7DAY = K_Fold(Linear, fea, pw_7day, cv=10)
# Evaluate
print('Regression Result in 7 Day')
print('\n','Random Forest : (RMSE,Score,MAE) :', Random_Forest_7DAY,'\n','Decision Tree : (RMSE,Score,MAE) :', Decision_Tree_7DAY,'\n','Linear Regression : (RMSE,Score,MAE) :', Linear_7DAY)