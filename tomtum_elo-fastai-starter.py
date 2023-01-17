from fastai.tabular import *

from fastai import *
import datetime
path = Path('../input')

path.ls()
cur = Path('')

cur.ls()
df_train = pd.read_csv(path/'train.csv')

df_test = pd.read_csv(path/'test.csv')

df_hist_trans = pd.read_csv(path/'historical_transactions.csv')

df_new_merchant_trans = pd.read_csv(path/'new_merchant_transactions.csv')
df_test.head().T
# Preprocess and merge datasets

# Preprocessing from: https://www.kaggle.com/chauhuynh/my-first-kernel-3-699
for df in [df_hist_trans,df_new_merchant_trans]:

    df['category_2'].fillna(1.0,inplace=True)

    df['category_3'].fillna('A',inplace=True)

    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
def get_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
# Fastai version of data preprocessing

# add_datepart(df_train, 'first_active_month')

# add_datepart(df_test, 'first_active_month')
for df in [df_hist_trans,df_new_merchant_trans]:

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    df['year'] = df['purchase_date'].dt.year

    df['weekofyear'] = df['purchase_date'].dt.weekofyear

    df['month'] = df['purchase_date'].dt.month

    df['dayofweek'] = df['purchase_date'].dt.dayofweek

    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)

    df['hour'] = df['purchase_date'].dt.hour

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})

    df['category_1'] = df['category_1'].map({'Y':1, 'N':0})

    #https://www.kaggle.com/c/elo-merchant-category-recommendation/discussion/73244

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']
aggs = {}

for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:

    aggs[col] = ['nunique']



aggs['purchase_amount'] = ['sum','max','min','mean','var']

aggs['installments'] = ['sum','max','min','mean','var']

aggs['purchase_date'] = ['max','min']

aggs['month_lag'] = ['max','min','mean','var']

aggs['month_diff'] = ['mean']

# aggs['authorized_flag'] = ['sum']#, 'mean'] # df_train['hist_authorized_flag_mean'] is all NaN; sum is 0

aggs['weekend'] = ['sum', 'mean']

# aggs['category_1'] = ['sum', 'mean'] # all zeros

aggs['card_id'] = ['size']



for col in ['category_2','category_3']:

    df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')

    aggs[col+'_mean'] = ['mean']    



new_columns = get_new_columns('hist',aggs)

df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)

df_hist_trans_group.columns = new_columns

df_hist_trans_group.reset_index(drop=False,inplace=True)

df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days

df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']

df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;gc.collect()
aggs = {}

for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:

    aggs[col] = ['nunique']

aggs['purchase_amount'] = ['sum','max','min','mean','var']

aggs['installments'] = ['sum','max','min','mean','var']

aggs['purchase_date'] = ['max','min']

aggs['month_lag'] = ['max','min','mean','var']

aggs['month_diff'] = ['mean']

aggs['weekend'] = ['sum', 'mean']

aggs['category_1'] = ['sum', 'mean']

aggs['card_id'] = ['size']



for col in ['category_2','category_3']:

    df_new_merchant_trans[col+'_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')

    aggs[col+'_mean'] = ['mean']

    

new_columns = get_new_columns('new_hist',aggs)

df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)

df_hist_trans_group.columns = new_columns

df_hist_trans_group.reset_index(drop=False,inplace=True)

df_hist_trans_group['new_hist_purchase_date_diff'] = (df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group['new_hist_purchase_date_min']).dt.days

df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff']/df_hist_trans_group['new_hist_card_id_size']

df_hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

df_test = df_test.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;gc.collect()
del df_hist_trans;gc.collect()

del df_new_merchant_trans;gc.collect()
df_train.head(5)
df_train['outliers'] = 0

df_train.loc[df_train['target'] < -30, 'outliers'] = 1

df_train['outliers'].value_counts()
for df in [df_train,df_test]:

    df['first_active_month'] = pd.to_datetime(df['first_active_month']) # remove active month afterwards?

    df['dayofweek'] = df['first_active_month'].dt.dayofweek

    df['weekofyear'] = df['first_active_month'].dt.weekofyear

    df['month'] = df['first_active_month'].dt.month

    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days

    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\

                     'new_hist_purchase_date_min']:

        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']

    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']



for f in ['feature_1','feature_2','feature_3']:

    order_label = df_train.groupby([f])['outliers'].mean()

    df_train[f] = df_train[f].map(order_label)

    df_test[f] = df_test[f].map(order_label)
df_train_columns = [c for c in df_train.columns if c not in ['card_id', 'first_active_month','target','outliers']]

target = df_train['target']

# del df_train['target']
# remove dates

del df_train['first_active_month']
df_train.head()
# Save data

df_train.to_csv('proc-train-data.csv')

df_test.to_csv('proc-test-data.csv')
# Load Data

df_train = pd.read_csv('proc-train-data.csv', index_col=0, header=0)

df_test = pd.read_csv('proc-test-data.csv', index_col=0, header=0)
df_train.head()
# Create Databunch
cat_names = ['card_id', 'feature_1', 'feature_2', 'feature_3', 'hist_month_nunique', 'hist_hour_nunique', 'hist_weekofyear_nunique', 

             'hist_dayofweek_nunique', 'hist_year_nunique', 'hist_subsector_id_nunique', 'hist_merchant_id_nunique', 'hist_merchant_category_id_nunique',

             'hist_category_2_mean_mean', 'hist_category_3_mean_mean', 'hist_purchase_date_uptonow', 'new_hist_month_nunique', 'new_hist_hour_nunique', 

             'new_hist_weekofyear_nunique', 'new_hist_dayofweek_nunique', 'new_hist_year_nunique', 'new_hist_subsector_id_nunique', 'new_hist_merchant_id_nunique',

             'new_hist_merchant_category_id_nunique', 'dayofweek', 'weekofyear', 'month' # 'outliers',

            ]

cont_names = ['hist_purchase_amount_sum', 'hist_purchase_amount_max', 'hist_purchase_amount_min', 'hist_purchase_amount_mean', 

              'hist_purchase_amount_var', 'hist_installments_sum', 'hist_installments_max', 'hist_installments_min', 'hist_installments_mean',

              'hist_installments_var', 'hist_purchase_date_max', 'hist_purchase_date_min', 'hist_month_lag_max', 'hist_month_lag_min', 'hist_month_lag_mean', 

              'hist_month_lag_var', 'hist_month_diff_mean', 'hist_weekend_sum', 'hist_weekend_mean', 'hist_card_id_size', 'hist_purchase_date_diff', 

              'hist_purchase_date_average', 'new_hist_purchase_amount_sum', 'new_hist_purchase_amount_max', 'new_hist_purchase_amount_min', 

              'new_hist_purchase_amount_mean', 'new_hist_purchase_amount_var', 'new_hist_installments_sum', 'new_hist_installments_max', 'new_hist_installments_min',

              'new_hist_installments_mean', 'new_hist_installments_var', 'new_hist_purchase_date_max', 'new_hist_month_lag_max', 'new_hist_month_lag_min', 

              'new_hist_month_lag_mean', 'new_hist_month_lag_var', 'new_hist_month_diff_mean', 'new_hist_weekend_sum', 'new_hist_weekend_mean', 

              'new_hist_category_1_sum', 'new_hist_category_1_mean', 'new_hist_card_id_size', 'new_hist_category_2_mean_mean', 'new_hist_category_3_mean_mean',

              'new_hist_purchase_date_diff', 'new_hist_purchase_date_average', 'new_hist_purchase_date_uptonow', 'elapsed_time', 'hist_first_buy', 'new_hist_first_buy',

              'card_id_total', 'purchase_amount_total', 'new_hist_purchase_date_min'

             ]
for cat in df_train.columns:

    if cat not in cat_names and cat not in cont_names:

        print(cat)

        

for cat in cat_names:

    if cat not in df_train.columns:

        print(cat)

        

for cat in cont_names:

    if cat not in df_train.columns:

        print(cat)
len(df_train)
dep_var = 'target'

procs = [FillMissing, Categorify, Normalize]
df_test[cont_names] = df_test[cont_names].fillna(df_test[cont_names].median(axis=0))

del df_train['outliers']
test_data = TabularList.from_df(df_test, path=".", cat_names=cat_names, cont_names=cont_names, procs=procs)
data = (TabularList.from_df(df_train, path=".", cat_names=cat_names, cont_names=cont_names, procs=procs)

                           .random_split_by_pct(0.20)

                           .label_from_df(cols=dep_var)

                           .add_test(test_data)

                           .databunch(bs=4096))
data.show_batch(rows=10)
# Create a tabular learner Model
y_range = [df_train['target'].min()*1.3, df_train['target'].max()*1.3]

y_range
learn = tabular_learner(data, layers=[1000,500], metrics=rmse, callback_fns=[callbacks.SaveModelCallback],

                        wd=1, emb_drop=0.1, ps=[5e-3, 5e-2], y_range=y_range)
# Find learning rate

learn.lr_find()
learn.recorder.plot()
# Train!

learn.fit_one_cycle(20,5e-2)
# Grab some predictions for ensembling

preds_1,tgt_1 = learn.get_preds(DatasetType.Test)
# Sometimes you can add this section after initial training

# learn.callback_fns.append(callbacks.SaveModelCallback)`

# learn.callback_fns
learn.fit_one_cycle(10,1e-3)
rmse(*learn.get_preds())
learn.fit_one_cycle(15,1e-4)
rmse(*learn.get_preds())
learn.fit_one_cycle(15,1e-5)
learn.fit_one_cycle(20,5e-7)
rmse(*learn.get_preds())
learn.recorder.plot_losses()
# Check sample submission

samp = pd.read_csv(path/'sample_submission.csv')

samp.head()
# Predict on test set

preds,tgt = learn.get_preds(DatasetType.Test)
preds.median(), preds.mean(), preds.max(), preds.min()
test_reload_subm = pd.read_csv(path/'test.csv')
subm = pd.DataFrame({'card_id': test_reload_subm['card_id'], 'target': preds.squeeze()})

subm.head()
subm2 = pd.DataFrame({'card_id': test_reload_subm['card_id'], 'target': preds_1.squeeze()})

subm2.head()
# Save multiple predictions so that you can ensemble at the end
s_n = 1

# s_n += 1

# s_n
subm.to_csv(f'subm{s_n}.csv', index=False)

subm2.to_csv(f'subm2.csv', index=False) # Comment this out
# Confirm it saved correctly

c = pd.read_csv(f'subm{s_n}.csv')

c.head()
# Download individual results from kaggle kernel

# from IPython.display import FileLink
# FileLink(f'subm{s_n}.csv')
# Ensemble some predictions
!ls
s1 = pd.read_csv('subm1.csv')

s2 = pd.read_csv('subm2.csv')

# s3 = pd.read_csv('subm3.csv')

# s4 = pd.read_csv('subm4.csv')

# s5 = pd.read_csv('subm5.csv')
s_all = pd.concat([s1,s2['target']],axis=1) #,s3['target'],s4['target'],s5['target']],axis=1)

s_all.head()
s_all.describe().T
s_all.describe().T.describe()
s_all_mean = s_all.mean(axis=1)

s_all_mean.head()
subm_ens = pd.DataFrame({'card_id': test_reload_subm['card_id'], 'target': s_all_mean})

subm_ens.to_csv(f'subm_ens.csv', index=False)