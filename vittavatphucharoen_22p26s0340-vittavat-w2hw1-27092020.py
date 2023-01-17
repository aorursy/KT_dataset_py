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
## Import Dataset

gen_p1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Generation_Data.csv")

sensor_p1 = pd.read_csv("../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
## แปลง columns type เป็น datetime

gen_p1['DATE_TIME']= pd.to_datetime(gen_p1['DATE_TIME'],format='%d-%m-%Y %H:%M')
## Explore

gen_p1.info()

gen_p1.head(6)
sensor_p1.info()

sensor_p1.head()
## Looking for Missing SOURCE_KEY

gen_p1.groupby(by = ['DATE_TIME']).size()[gen_p1.groupby(by = ['DATE_TIME']).size() != 22]
## เพื่อที่จะสามารถนำโมเดลไปใช้กับ inventor ทั้ง 22 ตัวได้ เพราะฉะนั้นในช่วงวันไหนที่ไม่
gen_p1['DATE_TIME'][1]
gen_p1[gen_p1['SOURCE_KEY'] == gen_p1['SOURCE_KEY'][1]][gen_p1[gen_p1['SOURCE_KEY'] == gen_p1['SOURCE_KEY'][1]]['DAILY_YIELD'] != 0].head()

## EDA to understand DAILY_YIELD and TOTAL_YIELD columns

## Look at just one inveter

import datetime as dt



gen_p1[gen_p1['SOURCE_KEY'] == gen_p1['SOURCE_KEY'][1]][gen_p1[gen_p1['SOURCE_KEY'] == gen_p1['SOURCE_KEY'][1]]['DAILY_YIELD'] != 0]

invt_sk_1 = gen_p1[gen_p1['SOURCE_KEY'] == gen_p1['SOURCE_KEY'][1]][gen_p1[gen_p1['SOURCE_KEY'] == gen_p1['SOURCE_KEY'][1]]['DAILY_YIELD'] != 0]



## Split DATE_TIME column

invt_sk_1['DATE'] = pd.to_datetime(invt_sk_1['DATE_TIME']).dt.date

invt_sk_1['TIME'] = pd.to_datetime(invt_sk_1['DATE_TIME']).dt.time



## group_by DATE

invt_sk_1.groupby(by = 'DATE').max()
invt_sk_1.index
## ทั้ง DAILY_YIELD และ TOTAL_YIELD เป็นการรวมแบบ accumulate

## ดู DC_POWER และ AC_POWER ใน 1 วัน

invt_sk_1[invt_sk_1['DATE'] == invt_sk_1['DATE'][532]].head()
gen_p1['DATE'] = pd.to_datetime(gen_p1['DATE_TIME']).dt.date

gen_p1.head()
## Create Line Plot เพื่อดู Yield ของ inventor แต่ละตัวว่าใกล้เคียงกันไหม

from ggplot import *

ggplot(gen_p1, aes(x = 'DATE_TIME', y = 'DAILY_YIELD', group = 'SOURCE_KEY', color = 'SOURCE_KEY')) + geom_line()

## Data Wrangling สำหรับเตรียมสร้าง Regression Model

gen_p1d = gen_p1.groupby(['DATE']).agg({'DC_POWER': 'mean',

                                           'AC_POWER': 'mean',

                                        'DAILY_YIELD': 'max'})

gen_p1d.head()
sensor_p1.head()
## แปลง columns type เป็น datetime

sensor_p1['DATE_TIME']= pd.to_datetime(sensor_p1['DATE_TIME'],format='%Y-%m-%d %H:%M')

sensor_p1['DATE'] = pd.to_datetime(sensor_p1['DATE_TIME']).dt.date

sensor_p1.head()
sensor_p1d = sensor_p1.groupby(['DATE']).agg({'AMBIENT_TEMPERATURE': 'mean',

                                               'MODULE_TEMPERATURE': 'mean',

                                                'IRRADIATION': 'mean'})

sensor_p1d.head()
gen_p1d.join(sensor_p1d, on = ['DATE']).columns
## Join Generation Dataset and Sensor Dataset together.

daily_fres = gen_p1d.join(sensor_p1d, on = ['DATE'])



## เรียง column ใหม่

daily_fres = daily_fres[['DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE',

       'MODULE_TEMPERATURE', 'IRRADIATION', 'DAILY_YIELD']]

daily_fres = daily_fres.reset_index()

daily_fres

ggplot(daily_fres, aes(x = 'DATE', y = 'DAILY_YIELD')) + geom_point()
## transform data for create model

rows, cols = daily_fres.shape

daily_fres['lag_1'] = [np.nan,np.nan,np.nan,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]

daily_fres['lag_2'] = [np.nan,np.nan,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]

daily_fres['lag_3'] = [np.nan,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]

daily_fres['lag_4'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]

daily_fres



df_l1 = daily_fres.drop(['lag_2','lag_3','lag_4'], axis = 1)

df_l2 = daily_fres.drop(['lag_1','lag_3','lag_4'], axis = 1)

df_l3 = daily_fres.drop(['lag_1','lag_2','lag_4'], axis = 1)

df_l4 = daily_fres.drop(['lag_1','lag_2','lag_3'], axis = 1)

daily_fres
## join 3 days before

df_l1 = pd.merge(df_l1, df_l2, how='left', left_on='lag_1', right_on='lag_2', suffixes=('', '_1before'))

df_l1 = pd.merge(df_l1, df_l3, how='left', left_on='lag_1', right_on='lag_3', suffixes=('', '_2before'))

df_l1 = pd.merge(df_l1, df_l4, how='left', left_on='lag_1', right_on='lag_4', suffixes=('', '_3before'))

df_l1
## delete some rows and columns

data_3d_before = df_l1.drop(['lag_1',

                             'lag_2',

                             'lag_3',

                             'lag_4',

                             'DATE_1before',

                             'DATE_2before',

                             'DATE_3before',

                             'DC_POWER',

                             'AC_POWER',

                             'AMBIENT_TEMPERATURE',

                             'MODULE_TEMPERATURE',

                             'IRRADIATION'], axis = 1)[6:]
## look at new table

data_3d_before.info()

data_3d_before.head()
## reset index

data_3d_before = data_3d_before.reset_index().drop('index', axis = 1)
data_3d_before.tail()
## ทดลอง

data_3d_before.index,data_3d_before.index%10
## ทดลอง

data_3d_before[data_3d_before.index%10 == 0]
data_3d_before
## Create fn เพื่อแยก folds และ คำนวณค่า RMSE ของ Model จากแต่ละ folds

def kfolds_model_ind(df, inds, model_type, Y, drop):

    model_list = []

    rmse_list = []

    y_hat = []

    from sklearn.linear_model import LinearRegression

    from sklearn.ensemble import RandomForestRegressor



    for ind in inds:

        test_set = df[df.index%10 == ind]

        train_set = df[df.index%10 != ind]

        if model_type == 'LinearRegression':

            model = LinearRegression()

            model_fitted = model.fit(train_set.drop([drop,Y], axis = 1), train_set[Y])

            y_hat = model_fitted.predict(test_set.drop([drop,Y], axis = 1))

            y_hat = np.array(y_hat)

            actual = np.array(test_set[Y])

            rmse = (((y_hat - actual)**2).sum()/y_hat.size)**(-2)



        elif model_type == 'RF':

            model = RandomForestRegressor(n_estimators=150, min_samples_split=2)

            model_fitted = model.fit(train_set.drop([drop,Y], axis = 1), train_set[Y])

            y_hat = model_fitted.predict(test_set.drop([drop,Y], axis = 1))

            y_hat = np.array(y_hat)

            actual = np.array(test_set[Y])

            rmse = (((y_hat - actual)**2).sum()/y_hat.size)**(-2)

            

        model_list.append(model_fitted)

        rmse_list.append(rmse)

    return rmse_list

        
## คำนวณ RMSE ของ Linear Regression และ Random Forest Model สำหรับการใช้ข้อมูลก่อนหน้า 3 วันทำนาย Yield ในวันนั้น (ใช้ข้อมูลเมื่อวานซืน เมื่อวาน วันนี้ ทำนาย Yield ของวันพรุ่งนี้)

res_lr = kfolds_model_ind(data_3d_before, [0,1,2,3,4,5,6,7,8,9], 'LinearRegression','DAILY_YIELD', 'DATE')

res_rf = kfolds_model_ind(data_3d_before, [0,1,2,3,4,5,6,7,8,9], 'RF','DAILY_YIELD', 'DATE')



rmse_lr_rf = pd.DataFrame({'RMSE_LR': res_lr,

                          'RMSE_RF' : res_rf})
rmse_lr_rf