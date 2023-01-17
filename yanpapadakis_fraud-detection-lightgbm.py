import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, roc_auc_score

import lightgbm as lgb
df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')

df.info()
# Correct Column Name

df.rename(columns={"oldbalanceOrg":"oldbalanceOrig"},inplace=True)
df.head(10)
df.describe().T
# Data Ordered by Step

tmp = pd.Index(df.step)

for s in range(df.step.max()):

    if s < 10 or s > 735:

        print("step {:3d}: ".format(s+1),tmp.get_loc(s+1))
cnt = df.step.value_counts().sort_index()

ax = cnt.plot(logy=True,title="Transaction Count per Hour")
%%time

df.groupby('isFraud')[['nameOrig','nameDest']].describe()
df.groupby(['type','isFraud']).step.count().unstack()
df.groupby('type')['amount'].describe().style.background_gradient(high=0.4,axis=0).format("{:,.2f}")
pd.crosstab(df.isFraud,df.isFlaggedFraud)
df.groupby('isFlaggedFraud')['amount'].describe().style.background_gradient(high=1).format("{:,.2f}")
df['origAmountError'] = (df.oldbalanceOrig - df.newbalanceOrig - df.amount) / (df.oldbalanceOrig+df.newbalanceOrig+df.amount+1) * 2

df.groupby(['isFraud','type'])['origAmountError'].describe()
df['nameOrig'].str[:1].value_counts()
df['nameDest'].str[:1].value_counts()
frd = df[df.isFraud==1]

frd
df['destType'] = df['nameDest'].str[:1]
df['day'] = df.step // 24

df.day.value_counts().sort_index().plot(title='Transactions by Day')

plt.show()

df['hour'] = df.step % 24

df.hour.value_counts().sort_index().plot(title='Transactions by Hour')

plt.show()
df.info()
for col in ['type','destType','hour']:

    print('One Way Analysis for ',col)

    display(df.groupby(col)['isFraud'].mean().to_frame().style.bar().format("{:.2%}"))
for col in ['day','step']:

    print('Fraud Rate by ',col)

    grp = df.groupby([col,'isFraud']).size()

    #display(grp.unstack().style.format('{:.2f}').bar())

    ax = df.groupby(col).isFraud.mean().plot(logy=True)

    plt.show()
# drop list 

droplist=['isFlaggedFraud','nameDest','nameOrig','step','hour','day','type']
MLData = pd.get_dummies(df.loc[df.type.apply(lambda x: x in ['CASH_OUT','TRANSFER'])].drop(labels=droplist,axis=1))

X=MLData.drop('isFraud',axis=1)

Y=MLData.isFraud
X_train, X_test, y_train, y_test = train_test_split(

    MLData.drop('isFraud',axis=1), 

    MLData.isFraud, 

    test_size = 0.3, 

    random_state=2019, 

    shuffle=False

)
%%time



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval  = lgb.Dataset(X_test, y_test, reference=lgb_train)



# specify configurations as dict

params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': 'binary_logloss',

    'num_leaves': 4,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 1

}



gbm = lgb.train(params,

            lgb_train,

            num_boost_round=15,

            valid_sets=[lgb_train,lgb_eval],

            valid_names=['train', 'eval'])
ax = lgb.plot_importance(gbm, max_num_features=10)
lgb.create_tree_digraph(gbm,tree_index=0,precision=1,show_info=['leaf_count'])
lgb.create_tree_digraph(gbm,tree_index=1,precision=1,show_info=['leaf_count'])
y_test_pred = gbm.predict(X_test)
print('Validation AUC Score: {:.3%}'.format(roc_auc_score(y_test,y_test_pred)))
print('Validation F1 Score: {:.3%}'.format(f1_score(y_test,y_test_pred>0.5)))
pd.crosstab(pd.cut(y_test_pred,np.arange(0,1.01,0.1)),y_test)