import numpy as np

import pandas as pd

import xgboost as xgb
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")



df_train.head()
#mask = (df_train['num_room'] == 2.0) & (df_train['build_year'] > 2000.0)

#plt.title("City center distance VS Price (2 rooms, new building)")

#sns.jointplot(x='kremlin_km', y='price_doc', data=df_train[mask])

#plt.title("Distance to Sadovoe VS Price (2 rooms, new building)")

#sns.jointplot(x='sadovoe_km', y='price_doc', data=df_train[mask])

#plt.title("Distance to Bulvarnoe VS Price (2 rooms, new building)")

#sns.jointplot(x='bulvar_ring_km', y='price_doc', data=df_train[mask])

#plt.title("Distance to MKAD VS Price (2 rooms, new building)")

#sns.jointplot(x='mkad_km', y='price_doc', data=df_train[mask])
#plt.figure(figsize=(12,8))

#sns.stripplot(y='sub_area', x='kremlin_km', data=df_train)

#plt.xlabel('Kremlin km', fontsize=12)

#plt.ylabel('Name of the district', fontsize=12)

#_ = plt.yticks(fontsize=8)
#plt.figure(figsize=(12,8))

#sns.boxplot(x='sub_area', y='price_doc', data=df_train[mask])

#plt.ylabel('Price', fontsize=12)

#plt.xlabel('Name of the district', fontsize=12)

#_ = plt.xticks(rotation='vertical', fontsize=8)
df_train[df_train['floor'].isnull()][['full_sq', 'num_room', 'state', 'build_year', 'sub_area', 'kremlin_km']].head()
#cols = [c for c in df_train.columns if 'area' in c]

#cols
#cols = [c for c in df_train.columns if 'km' in c]

#cols
train_df['floor'].fillna(-99, inplace=True)
build_year_cols = [

    'build_count_before_1920',

    'build_count_1921-1945',

    'build_count_1946-1970',

    'build_count_1971-1995',

    'build_count_after_1995'

]



mean_build_year_map = {

    'build_count_before_1920' : 1920,

    'build_count_1921-1945' : 1933,

    'build_count_1946-1970' : 1958,

    'build_count_1971-1995' : 1983,    

    'build_count_after_1995' : 1995

}



def compute_build_year(row):     

    count_build_year_list = np.array([(mean_build_year_map[i], c) for i, c in zip(row.index, row)])

    s = np.sum(count_build_year_list[:,1])

    year = np.sum(count_build_year_list[:,0] * count_build_year_list[:,1]) / s

    return int(year)



def fix_build_year(df):

    mask = df[build_year_cols].isnull()

    mask = mask.any(axis=1)

    mask = df['build_year'].isnull() & ~mask

    df.loc[mask, 'build_year'] = df[mask][build_year_cols].apply(compute_build_year, axis=1)

    df.loc[df_train['build_year'].isnull(), 'build_year'] = 1970

    

fix_build_year(df_train)

fix_build_year(df_test)
def to_yearmonth(yearmonthdate_str):

    """

    Convert '2016-01-23' -> 201601

    """    

    yearmonth = int(yearmonthdate_str[:4] + yearmonthdate_str[5:7])

    return yearmonth



df_train['yearmonth'] = df_train['timestamp'].apply(to_yearmonth)

df_train['age'] = df_train['yearmonth'] - df_train['build_year'] * 100



df_test['yearmonth'] = df_test['timestamp'].apply(to_yearmonth)

df_test['age'] = df_test['yearmonth'] - df_test['build_year'] * 100
df_train['price_doc'].hist(bins=50)
df_train['price_doc'].apply(np.log).hist(bins=50)
df_train['price_doc'] = df_train['price_doc'].apply(np.log)
df_train = df_train[df_train['yearmonth'] > 201500]
y_train = df_train['price_doc'].values

id_test = df_test['id']



df_train.drop(['id', 'timestamp', 'price_doc'], axis=1, inplace=True)  # throw away timestamp just for test

df_test.drop(['id', 'timestamp'], axis=1, inplace=True)



num_train = len(df_train)

df_all = pd.concat([df_train, df_test])

print(df_all.shape)



X_all = np.c_[

    df_all.select_dtypes(exclude=['object']).values,

    pd.get_dummies(df_all.select_dtypes(include=['object'])).values

]

print(X_all.shape)



X_train = X_all[:num_train]

X_test = X_all[num_train:]
xgb_params = {

    'eta': 0.02,

    'max_depth': 6,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}



dtrain = xgb.DMatrix(X_train, y_train)

dtest = xgb.DMatrix(X_test)
#cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=800, early_stopping_rounds=100,

#                   verbose_eval=True, show_stdv=False)
#cv_result[['train-rmse-mean', 'test-rmse-mean']].plot()
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=800)
y_pred = model.predict(dtest)



df_sub = pd.DataFrame({'id': id_test, 'price_doc': np.exp(y_pred)})



df_sub.to_csv('sub.csv', index=False)