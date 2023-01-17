import pandas as pd

loding = pd.read_csv('../input/loding.csv')

loding.head()

combine = [loding]
# 0 too low loding, 1 nice, 2 too much

for dataset in combine:

    dataset.loc[dataset['withoutSalt']<150,'withoutSalt']=0

#     dataset.loc[(dataset['withoutSalt']<=200) & (dataset['withoutSalt']>100),'withoutSalt']=1

    dataset.loc[dataset['withoutSalt']>150,'withoutSalt']=1

loding.loc[loding['withoutSalt']==1].count()

# loding.head()
loding[['Gender', 'withoutSalt']].groupby(['Gender'], as_index=False).mean()
# pd.crosstab(loding['Education_Level'],loding['withoutSalt'])

loding[['Education_Level', 'withoutSalt']].groupby(['Education_Level'], as_index=False).mean()

# loding[['Education_Level', 'withoutSalt']].groupby(['withoutSalt'], as_index=False).mean()
loding['AgeBand'] = pd.cut(loding['Age'], 4)

# loding[['AgeBand', 'withoutSalt']].groupby(['AgeBand'], as_index=False).count()

# pd.crosstab(loding['AgeBand'],loding['withoutSalt'])

loding[['AgeBand', 'withoutSalt']].groupby(['AgeBand'], as_index=False).mean()

for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 31, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 31) & (dataset['Age'] <= 43), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 43) & (dataset['Age'] <= 55), 'Age'] = 2

#     dataset.loc[(dataset['Age'] > 55) & (dataset['Age'] <= 67), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 55, 'Age'] = 3
# pd.crosstab(loding['Occupation'],loding['withoutSalt'])

loding[['Occupation', 'withoutSalt']].groupby(['Occupation'], as_index=False).mean()
loding[['City', 'withoutSalt']].groupby(['City'], as_index=False).mean()
loding[['Nutrition', 'withoutSalt']].groupby(['Nutrition'], as_index=False).mean()
loding[['Smoking', 'withoutSalt']].groupby(['Smoking'], as_index=False).mean()
loding[['lodized', 'withoutSalt']].groupby(['lodized'], as_index=False).mean()
loding[['awarenness', 'withoutSalt']].groupby(['awarenness'], as_index=False).mean()
loding['knowledgeBand'] = pd.cut(loding['knowledge'], 4)

# loding[['AgeBand', 'withoutSalt']].groupby(['AgeBand'], as_index=False).count()

# pd.crosstab(loding['AgeBand'],loding['withoutSalt'])

loding[['knowledgeBand', 'withoutSalt']].groupby(['knowledgeBand'], as_index=False).mean()
for dataset in combine:    

    dataset.loc[ dataset['knowledge'] <= 6.5, 'knowledge'] = 0

    dataset.loc[(dataset['knowledge'] > 6.5) & (dataset['knowledge'] <= 12), 'knowledge'] = 1

    dataset.loc[(dataset['knowledge'] > 12) & (dataset['knowledge'] <= 17.5), 'knowledge'] = 2

#     dataset.loc[(dataset['Age'] > 55) & (dataset['Age'] <= 67), 'Age'] = 3

    dataset.loc[ dataset['knowledge'] > 17.5, 'knowledge'] = 3
feature_cols = ['Gender','Education_Level','Age','Smoking','lodized','awarenness','knowledge']

valid_fraction = 0.15

valid_rows = int(len(loding)*valid_fraction)

loding = loding.sort_values('Id')

train = loding[:-valid_rows*2]

valid = loding[-valid_rows*2:-valid_rows]

test = loding[-valid_rows:]
import lightgbm as lgb

# train_data = lgb.Dataset(train[feature_cols],train['withoutSalt'])

# val_data = lgb.Dataset(valid[feature_cols],valid['withoutSalt'])

# dtest = lgb.Dataset(test[feature_cols],label=test['withoutSalt'])

dtrain = lgb.Dataset(train[feature_cols], label=train['withoutSalt'])

dvalid = lgb.Dataset(valid[feature_cols], label=valid['withoutSalt'])

dtest = lgb.Dataset(test[feature_cols], label=test['withoutSalt'])

print(dtrain)

param = {'num_leaves':64,'objective':'binary'}

param['metric'] = 'auc'

num_round = 1000

bst = lgb.train(param, dtrain, num_round,valid_sets = [dvalid],early_stopping_rounds=10)



from sklearn import metrics



ypred = bst.predict(test[feature_cols])

score = metrics.roc_auc_score(test['withoutSalt'], ypred)

print(f"Test score: {score}")
feature_cols = ['Gender','Education_Level','Age','Smoking','lodized','awarenness','knowledge']

valid_fraction = 0.20

valid_rows = int(len(loding)*valid_fraction)

loding = loding.sort_values('Id')

train = loding[:-valid_rows]

test = loding[-valid_rows:]
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()

decision_tree.fit(train[feature_cols], train['withoutSalt'])

pred = decision_tree.predict(test[feature_cols])

acc_decision_tree = round(decision_tree.score(train[feature_cols], train['withoutSalt']) * 100, 2)

print(acc_decision_tree)

ans = test['withoutSalt'].as_matrix()

sumNum = len(pred)

correct_num = 0

for i in range(0,sumNum):

    if(pred[i]==ans[i]):

        correct_num = correct_num+1

print('test_auc='+str(correct_num/sumNum)) 