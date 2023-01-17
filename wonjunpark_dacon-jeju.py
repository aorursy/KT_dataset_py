import pandas as pd

import numpy as np

import sklearn

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns   

from tqdm import tqdm, trange
print('Pandas : %s'%(pd.__version__))

print('Numpy : %s'%(np.__version__))

print('Scikit-Learn : %s'%(sklearn.__version__))

!python --version
def grap_year(data):

    data = str(data)

    return int(data[:4])



def grap_month(data):

    data = str(data)

    return int(data[4:])
jeju = pd.read_csv('../input/dacon-jeju-creditcard-competition/201901-202003.csv')
jeju2 = pd.read_csv('../input/daconjejuaprildataset/202004.csv')
jeju3 = pd.concat([jeju,jeju2], ignore_index=True)
data = jeju3.fillna('')

data['year'] = data['REG_YYMM'].apply(lambda x: grap_year(x))

data['month'] = data['REG_YYMM'].apply(lambda x: grap_month(x))

data = data.drop(['REG_YYMM'], axis=1)
data.head()
print(data['CNT'].describe())
print(data['CSTMR_CNT'].describe())
test = True



if test:

    patition = int(len(data)*0.001) #24697

    data2 = data[:patition]

else:

    data2 = data.copy()
data['CNT'][1]
def feature_extra(data):

    for i in trange(len(data)):

        if data['CNT'][i] <= 5:

            data['CNT'][i] = 0

        elif data['CNT'][i] > 5 & data['CNT'][i] <= 12:

            data['CNT'][i] = 1

        elif data['CNT'] > 12 & data['CNT'][i] <= 40:

            data['CNT'][i] = 2

        else:

            data['CNT'][i] = 3

            

        if data['CSTMR_CNT'][i] <= 4:

            data['CSTMR_CNT'][i] = 0

        elif data['CSTMR_CNT'][i] > 4 & data['CSTMR_CNT'][i] <= 8:

            data['CSTMR_CNT'][i] = 1

        elif data['CSTMR_CNT'] > 8 & dtdata1['CSTMR_CNT'][i] <= 24:

            data['CSTMR_CNT'][i] = 2

        else:

            data['CSTMR_CNT'][i] = 3

            

    return data
data3 = feature_extra(data2)
# 데이터 정제

df = data3.copy()

df = df.drop(['CARD_CCG_NM', 'HOM_CCG_NM'], axis=1)



columns = ['CARD_SIDO_NM', 'STD_CLSS_NM', 'HOM_SIDO_NM', 'AGE', 'SEX_CTGO_CD', 'FLC', 'year', 'month','CNT','CSTMR_CNT']

df = df.groupby(columns).sum().reset_index(drop=False)



# 인코딩

dtypes = df.dtypes

encoders = {}

for column in df.columns:

    if str(dtypes[column]) == 'object':

        encoder = LabelEncoder()

        encoder.fit(df[column])

        encoders[column] = encoder

        

df_num = df.copy()        

for column in encoders.keys():

    encoder = encoders[column]

    df_num[column] = encoder.transform(df[column])
df_num.head()
# 상관관계

plt.figure(figsize=(10,10))

sns.heatmap(data = df_num.corr(), annot=True, 

fmt = '.2f', linewidths=.5, cmap='Blues')
def dataset_cv(data, cv):

    train_num = data.sample(frac=1, random_state=0)

    

    x = train_num.drop(['AMT'], axis=1)

    y = np.log1p(train_num['AMT'])

    

    k = int(len(x)*0.2)

    

    if (cv == 1):

        x_train = x[k:]

        y_train = y[k:]

        x_val = x[:k]

        y_val = y[:k]

        

    elif (cv == 2):

        x_train = x[k*2:]

        x_train = x_train.append(x[:k])

        y_train = y[k*2:]

        y_train = y_train.append(y[:k])



        x_val = x[k:k*2]

        y_val = y[k:k*2]

        

    elif (cv == 3):

        x_train = x[k*3:]

        x_train = x_train.append(x[:k*2])

        y_train = y[k*3:]

        y_train = y_train.append(y[:k*2])



        x_val = x[k*2:k*3]

        y_val = y[k*2:k*3]

        

    elif (cv == 4):

        x_train = x[k*4:]

        x_train = x_train.append(x[:k*3])

        y_train = y[k*4:]

        y_train = y_train.append(y[:k*3])



        x_val = x[k*3:k*4]

        y_val = y[k*3:k*4]

        

    elif (cv == 5):

        x_train = x[:k*4]

        y_train = y[:k*4]

        x_val = x[k*4:]

        y_val = y[k*4:]

        

    return x_train, y_train, x_val, y_val
import lightgbm as lgb



params = {

            'boosting_type': 'gbdt',

            'objective': 'tweedie',

            'tweedie_variance_power': 1.1,

            'metric': 'rmse',

            'subsample': 0.5,

            'subsample_freq': 1,

            'learning_rate': 0.03,

            'num_leaves': 2**11-1,

            'min_data_in_leaf': 2**12-1,

            'feature_fraction': 0.5,

            'max_bin': 1000,

            'n_estimators': 1000,

            'boost_from_average': False,

            'verbose': -1

        }

def run_lgbm():

    submission_list = []

    

    for i in range(5):

        print(i+1,'loop..')

        x_train, y_train, x_val, y_val = dataset_cv(df_num,i+1)

        

        train_ds = lgb.Dataset(x_train, label=y_train)

        val_ds = lgb.Dataset(x_val, label=y_val)

        

        model = lgb.train(params,

                  train_ds,

                  10,

                  val_ds,

                  verbose_eval = 1000,

                  early_stopping_rounds = 100

                 )

        

        CARD_SIDO_NMs = df_num['CARD_SIDO_NM'].unique()

        STD_CLSS_NMs  = df_num['STD_CLSS_NM'].unique()

        HOM_SIDO_NMs  = df_num['HOM_SIDO_NM'].unique()

        AGEs          = df_num['AGE'].unique()

        SEX_CTGO_CDs  = df_num['SEX_CTGO_CD'].unique()

        FLCs          = df_num['FLC'].unique()

        years         = [2020]

        months        = [7]

        CSTMR_CNTs    = df_num['CSTMR_CNT'].unique()

        CNTs          = df_num['CNT'].unique()



        temp = []

        for CARD_SIDO_NM in CARD_SIDO_NMs:

            for STD_CLSS_NM in STD_CLSS_NMs:

                for HOM_SIDO_NM in HOM_SIDO_NMs:

                    for AGE in AGEs:

                        for SEX_CTGO_CD in SEX_CTGO_CDs:

                            for FLC in FLCs:

                                for year in years:

                                    for month in months:

                                        for CSTMR_CNT in CSTMR_CNTs:

                                            for CNT in CNTs:

                                                temp.append([CARD_SIDO_NM, STD_CLSS_NM, HOM_SIDO_NM, AGE, SEX_CTGO_CD, FLC, year, month, CSTMR_CNT, CNT])



        

        temp = np.array(temp)

        temp = pd.DataFrame(data=temp, columns=x_train.columns)

        

        pred = model.predict(temp)

            

        pred = np.expm1(pred)



        temp['AMT'] = np.round(pred, 0)

        temp['REG_YYMM'] = temp['year']*100 + temp['month']

        temp = temp[['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM', 'AMT']]

        temp = temp.groupby(['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM']).sum().reset_index(drop=False)

        

       

        temp['CARD_SIDO_NM'] = encoders['CARD_SIDO_NM'].inverse_transform(temp['CARD_SIDO_NM'])

        temp['STD_CLSS_NM'] = encoders['STD_CLSS_NM'].inverse_transform(temp['STD_CLSS_NM'])



        submission = pd.read_csv('../input/dacon-jeju-creditcard-competition/submission.csv', index_col=0)

        submission = submission.drop(['AMT'], axis=1)

        submission = submission.merge(temp, left_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], right_on=['REG_YYMM', 'CARD_SIDO_NM', 'STD_CLSS_NM'], how='left')

        submission.index.name = 'id'

        

        submission_list.append(submission)

        

    return submission_list
submission_list = run_lgbm()
for i in range(1,5):

    submission_list[0]['AMT'] = submission_list[0]['AMT'] + submission_list[i]['AMT']
submission_list[0]['AMT'] = submission_list[0]['AMT']/5
submission_list[0].to_csv('lgbm_5cv_submission.csv', encoding='utf-8-sig')

submission_list[0].head()