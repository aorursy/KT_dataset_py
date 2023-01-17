# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train =  pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")



train['Date'] = pd.to_datetime(train['Date'])



train['day'] = train['Date'].apply(lambda x: x.dayofyear-21).astype(int)   #day of 1/21

print(train.head())
train_wc=train[train['Country_Region']!='China'].reset_index(drop=True)

add_data=pd.read_csv('../input/datadata/covid19countryinfo.csv')

add_data.rename(columns={'country':'Country_Region'},inplace = True)

add_data.rename(columns={'region':'Province_State'},inplace = True)

train_wc=pd.merge(train_wc,add_data, on=['Country_Region','Province_State'], how='left')

train_wc
print(len(train_wc[pd.isnull(train_wc['hospibed'])]))

print(len(train_wc[pd.isnull(train_wc['pop'])]))

print(len(train_wc[pd.isnull(train_wc['medianage'])]))
for column in ['hospibed','medianage','density','gdp2019','pop']:             #'hospibed','medianage','density'

    mean_val = train_wc[column].mean()

    train_wc[column].fillna(mean_val, inplace=True)



train_wc.isnull().sum() > 0
for number in range(len(train_wc['day'])):

	if train_wc['day'][number]>6:

		train_wc.loc[number,'ConfirmedCases4']=train_wc['ConfirmedCases'][number-4].astype(int)

		train_wc.loc[number,'ConfirmedCases5']=train_wc['ConfirmedCases'][number-5].astype(int)

		train_wc.loc[number,'ConfirmedCases1']=train_wc['ConfirmedCases'][number-1].astype(int)

		train_wc.loc[number,'ConfirmedCases2']=train_wc['ConfirmedCases'][number-2].astype(int)

		train_wc.loc[number,'ConfirmedCases3']=train_wc['ConfirmedCases'][number-3].astype(int)

		train_wc.loc[number,'ConfirmedCases6']=train_wc['ConfirmedCases'][number-6].astype(int)

		train_wc.loc[number,'ConfirmedCases1-5']=(train_wc['ConfirmedCases'][number-1].astype(int)+

                                                train_wc['ConfirmedCases'][number-2].astype(int)

                                                +train_wc['ConfirmedCases'][number-3].astype(int)

                                                +train_wc['ConfirmedCases'][number-4].astype(int)

                                                +train_wc['ConfirmedCases'][number-5].astype(int) )                       

		train_wc.loc[number,'Fatalities1']=train_wc['Fatalities'][number-1].astype(int)

		train_wc.loc[number,'Fatalities2']=train_wc['Fatalities'][number-2].astype(int)

		train_wc.loc[number,'Fatalities3']=train_wc['Fatalities'][number-3].astype(int)

		train_wc.loc[number,'Fatalities1-5']=(train_wc['Fatalities'][number-1].astype(int)

                                            +train_wc['Fatalities'][number-2].astype(int)

                                            +train_wc['Fatalities'][number-3].astype(int)

                                            +train_wc['Fatalities'][number-4].astype(int)

                                            +train_wc['Fatalities'][number-5].astype(int))

	else:

		train_wc.loc[number,'ConfirmedCases1']=0

		train_wc.loc[number,'ConfirmedCases2']=0

		train_wc.loc[number,'ConfirmedCases3']=0

		train_wc.loc[number,'ConfirmedCases4']=0

		train_wc.loc[number,'ConfirmedCases5']=0

		train_wc.loc[number,'ConfirmedCases6']=0

		train_wc.loc[number,'ConfirmedCases1-5']=0

		train_wc.loc[number,'Fatalities1']=0

		train_wc.loc[number,'Fatalities2']=0

		train_wc.loc[number,'Fatalities3']=0

		train_wc.loc[number,'Fatalities1-5']=0
train_wc
def func(x):

    try:

        name = x['Country_Region'] + "/" + x['Province_State']

    except:

        name = x['Country_Region']

    return name        

train_wc['name'] = train_wc.apply(lambda x: func(x), axis=1)
train_wc.to_csv('train_wc1.csv',index=False)

train_wc
train_wc = pd.read_csv('train_wc1.csv')

temp = []

names = train_wc['name'].unique()

for name in names[:]:

    df = train_wc[train_wc['name']==name].reset_index(drop=True)    

    df['dayc1'] = (df['ConfirmedCases']<1).sum()

    a=(df['ConfirmedCases']>=1).sum()-(df['ConfirmedCases']>=100).sum()

    if (df['ConfirmedCases']<100).sum()==77:

        df['dayc1_100'] = 50

    else :

        df['dayc1_100'] = a    

    b=(df['ConfirmedCases']>=100).sum()-(df['ConfirmedCases']>=1000).sum()

    if (df['ConfirmedCases']<1000).sum()==77:

        df['dayc100_1000'] = 50

    else :

        df['dayc100_1000'] = b  

        

    df['dayf1']=(df['Fatalities']<1).sum()  

    a=(df['Fatalities']>=1).sum()-(df['Fatalities']>=100).sum()

    if (df['Fatalities']<100).sum()==77:

        df['dayf1_100'] = 50

    else:

        df['dayf1_100'] = a    

    b=df['dayf100_200'] = (df['Fatalities']>=100).sum()-(df['Fatalities']>=200).sum()

    if (df['Fatalities']<200).sum()==77:

        df['dayf100_200'] = 50

    else:

        df['dayc100_1000'] = b        

    temp.append(df)     

temp=pd.concat(temp).reset_index(drop=True)

train_wc=temp
test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")

test['Date'] = pd.to_datetime(test['Date'])

test['day'] = test['Date'].apply(lambda x: x.dayofyear-21).astype(int) 

test['Date']=test['Date'].dt.date

test['name'] = test.apply(lambda x: func(x), axis=1)

tt=pd.merge(train_wc,test, on=['Country_Region','Province_State','day'],how='left')

tt.rename(columns={'Date_x':'Date'},inplace = True)

test=test[test['day']>=78]

tt=tt.append(test).reset_index(drop=True)

tt.to_csv("tt.csv",index=False)

tt.columns  

tt = tt[tt['Country_Region']!='China'].reset_index(drop=True)
# params

import lightgbm as lgb

SEED = 42

params = {'num_leaves': 8,

          'min_data_in_leaf': 5,  

          'objective': 'regression',

          'max_depth': 4,     # #最大的树深，设为-1时表示不限制树的深度

          'learning_rate': 0.01,

          'boosting': 'gbdt',

          'bagging_freq': 5,  # 5

          'bagging_fraction': 0.8,  # 0.5,

          'feature_fraction': 0.8201,

          'bagging_seed': SEED,

          'reg_alpha': 1,  # 1.728910519108444,

          'reg_lambda': 4.9847051755586085,

          'random_state': SEED,

          'metric': 'mse',   #mse

          'verbosity': 100,

          'min_gain_to_split': 0.02,  # 0.01077313523861969,

          'min_child_weight': 5,  # 19.428902804238373,

          'num_threads': 6,

          }
col_target='ConfirmedCases'

col_var = [

           'ConfirmedCases1',

           'ConfirmedCases1-5',

     #  'ConfirmedCases2',

      #     'ConfirmedCases3',

     #      'ConfirmedCases4',

     #  'ConfirmedCases5',

     #      'ConfirmedCases6', 

     #      'Fatalities1',

     #      'Fatalities1-5', 

     #      'Fatalities2',

     #  'Fatalities3', 

       'day', 

           'dayc1',

     #      'dayc100_1000', 

           'dayc1_100', 

           'dayf1',

     #      'dayf100_200',

     #  'dayf1_100',

     #      'density', 

           'gdp2019',

     #      'hospibed', 

     #      'medianage',

    #   'pop'

          ]



df_train = tt[tt['day']<=77]

df_valid = tt[(70<tt['day']) & (tt['day']<=77)]

df_test = tt[pd.isna(tt['ForecastId'])==False]

X_train = df_train[col_var]

X_valid = df_valid[col_var]

y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)



train_data = lgb.Dataset(X_train,label=y_train)

valid_data = lgb.Dataset(X_valid,label=y_valid)



num_round = 100

model = lgb.train(params,train_data,num_round,valid_sets=valid_data)
col_target='Fatalities'

col_var2 = [

           'ConfirmedCases',

        #   'ConfirmedCases1',

        #   'ConfirmedCases1-5',

       #'ConfirmedCases2',

       #    'ConfirmedCases3',

       #    'ConfirmedCases4',

       #'ConfirmedCases5',

       #    'ConfirmedCases6', 

           'Fatalities1',

       #    'Fatalities1-5', 

     #      'Fatalities2',

      # 'Fatalities3', 

       'day', 

     #      'dayc100_1000', 

      #     'dayc1_100', 

           'dayf1',

      #     'dayf100_200',

  #     'dayf1_100',

      #     'density', 

      #     'gdp2019',

      #     'hospibed', 

      #     'medianage',

   #'pop'

          ]



df_train = tt[(tt['day']<=77) & (tt['day']>=10)]

df_valid = tt[(67<tt['day']) & (tt['day']<=77)]

print(df_train)

X_train = df_train[col_var2]

X_valid = df_valid[col_var2]

y_train = df_train[col_target].values

y_valid = df_valid[col_target].values



y_train = np.log(df_train[col_target].values.clip(0, 1e10)+1)

y_valid = np.log(df_valid[col_target].values.clip(0, 1e10)+1)

train_data = lgb.Dataset(X_train,label=y_train)

valid_data = lgb.Dataset(X_valid,label=y_valid)



num_round = 500

model2 = lgb.train(params,train_data,num_round,valid_sets=valid_data)
# display feature importance

tmp = pd.DataFrame()

tmp["feature"] = col_var2

tmp["importance"] = model2.feature_importance()

tmp = tmp.sort_values('importance', ascending=False)

tmp
tt1 = pd.read_csv('../input/datadata/tt_1.csv')

tt1 = tt1[tt1['Country_Region']!='China'].reset_index(drop=True)

tt1
for number in range(len(tt1['Date'])):     #改108

    if tt1.loc[number,'day']>=78:

        tt1.loc[number,'ConfirmedCases4']=tt1['ConfirmedCases'][number-4].astype(int)

        tt1.loc[number,'ConfirmedCases5']=tt1['ConfirmedCases'][number-5].astype(int)

        tt1.loc[number,'ConfirmedCases1']=tt1['ConfirmedCases'][number-1].astype(int)

        tt1.loc[number,'ConfirmedCases2']=tt1['ConfirmedCases'][number-2].astype(int)

        tt1.loc[number,'ConfirmedCases3']=tt1['ConfirmedCases'][number-3].astype(int)

        tt1.loc[number,'ConfirmedCases6']=tt1['ConfirmedCases'][number-6].astype(int)

        tt1.loc[number,'ConfirmedCases1-5']=(tt1['ConfirmedCases'][number-1].astype(int)+

                                                tt1['ConfirmedCases'][number-2].astype(int)

                                                +tt1['ConfirmedCases'][number-3].astype(int)

                                                +tt1['ConfirmedCases'][number-4].astype(int)

                                                +tt1['ConfirmedCases'][number-5].astype(int))                       

        tt1.loc[number,'Fatalities1']=tt1['Fatalities'][number-1].astype(int)

        tt1.loc[number,'Fatalities2']=tt1['Fatalities'][number-2].astype(int)

        tt1.loc[number,'Fatalities3']=tt1['Fatalities'][number-3].astype(int)

        tt1.loc[number,'Fatalities1-5']=(tt1['Fatalities'][number-1].astype(int)

                                            +tt1['Fatalities'][number-2].astype(int)

                                            +tt1['Fatalities'][number-3].astype(int)

                                            +tt1['Fatalities'][number-4].astype(int)

                                            +tt1['Fatalities'][number-5].astype(int))

        tt1.loc[number,'dayc1']=tt1['dayc1'][number-1].astype(int)

        tt1.loc[number,'dayc100_1000']=tt1['dayc100_1000'][number-1].astype(int)

        tt1.loc[number,'dayc1_100']=tt1['dayc1_100'][number-1].astype(int)

        tt1.loc[number, 'dayf1']=tt1['dayf1'][number-1].astype(int)

        tt1.loc[number,'dayf100_200']=tt1['dayf100_200'][number-1].astype(int)

        tt1.loc[number,'dayf1_100']=tt1['dayf1_100'][number-1].astype(int)

        tt1.loc[number,'density']=tt1['density'][number-1].astype(int)

        tt1.loc[number, 'gdp2019'  ]=tt1['gdp2019'][number-1].astype(int)

        tt1.loc[number, 'hospibed'  ]=tt1['hospibed'][number-1].astype(int)

        tt1.loc[number,'medianage' ]=tt1['medianage'][number-1].astype(int)

        tt1.loc[number,'pop']=tt1['pop'][number-1].astype(int)

        #train

        X_valid = tt1[col_var].iloc[number]

        X_valid2 = tt1[col_var2].iloc[number]

        pred_f = model.predict(X_valid)

        pred_c = model2.predict(X_valid2)

        a=(np.exp(pred_f)-1).clip(0, 1e10)

        if a<=tt1.loc[number,'ConfirmedCases1']+1:

            tt1.loc[number,'ConfirmedCases']= tt1.loc[number,'ConfirmedCases1']+1

        else:

            tt1.loc[number,'ConfirmedCases']= a

        b=(np.exp(pred_c)-1).clip(0, 1e10)

        

        tt1.loc[number,'Fatalities']=3*tt1.loc[number,'Fatalities1']-tt1.loc[number-1,'Fatalities1']-b

    print(number)
tt1.to_csv('hhhh.csv')
submission=pd.read_csv("../input/subsub/submission (1).csv")

submission.to_csv("submission.csv",index=False)