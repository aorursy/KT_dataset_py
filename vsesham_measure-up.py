# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.io.json import json_normalize

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LinearRegression

from sklearn.metrics import cohen_kappa_score

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import MinMaxScaler

from sklearn import linear_model

from sklearn.cluster import KMeans



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sns.set()
train_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

test_df = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
train_df.info()
cols_list = train_df.columns.tolist()

#for col in cols_list:

#    print(train_df[col].value_counts().sort_values(ascending=False))



cols_list
train_df.isna().sum()
test_df.info()
specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')



specs.head()
labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')



labels.head()
#train_df = train_df.loc[(train_df['event_code'] == 4100) | (train_df['event_code'] == 4110)]

#train_df = train_df.loc[train_df['type'] == 'Assessment']



#train_df = pd.merge(train_df, specs, on='event_id', how='inner')



#train_df.head()
train_df = pd.merge(train_df,labels,on=['game_session','installation_id'],how='inner')



train_df.info()
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], infer_datetime_format=True)

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], infer_datetime_format=True)



train_df['event_hour'] = train_df['timestamp'].dt.hour

train_df['event_day'] = train_df['timestamp'].dt.dayofweek



test_df['event_hour'] = test_df['timestamp'].dt.hour

test_df['event_day'] = test_df['timestamp'].dt.dayofweek
train_df[['title_x','world','type']] = train_df[['title_x','world','type']].astype('category')



df=pd.get_dummies(train_df,columns=['title_x','world','type'], prefix=['title','world','type'])



df.info()
df=df.rename(columns={"title_Bird Measurer (Assessment)":"BirdMeasurer","title_Cart Balancer (Assessment)":"CartBalancer",

                               "title_Cauldron Filler (Assessment)":"CauldronFiller","title_Chest Sorter (Assessment)":"ChestSorter",

                               "title_Mushroom Sorter (Assessment)":"MushroomSorter","world_CRYSTALCAVES":"CRYSTALCAVES",

                               "world_MAGMAPEAK":"MAGMAPEAK","world_TREETOPCITY":"TREETOPCITY","type_Assessment":"type"})
train_df_gp = df.groupby('installation_id')['timestamp'].agg('max').reset_index()

df = pd.merge(df,train_df_gp,on=['installation_id','timestamp'],how='inner')



df.info()
df.head()
cols = ['game_session',

 'installation_id',

 'event_day',

 'event_hour',

 'BirdMeasurer',

 'CartBalancer',

 'CauldronFiller',

 'ChestSorter',

 'MushroomSorter',

 'CRYSTALCAVES',

 'MAGMAPEAK',

 'TREETOPCITY',

 'accuracy',

 'accuracy_group']



df = df[cols]

df.head()
#test_df = test_df.loc[(test_df['event_code'] == 4100) | (test_df['event_code'] == 4110)]

#test_df = test_df.loc[test_df['type'] == 'Assessment']
test_df[['title','world','type']] = test_df[['title','world','type']].astype('category')



df_test=pd.get_dummies(test_df,columns=['title','world','type'], prefix=['title','world','type'])
df_test=df_test.rename(columns={"title_Bird Measurer (Assessment)":"BirdMeasurer","title_Cart Balancer (Assessment)":"CartBalancer",

                               "title_Cauldron Filler (Assessment)":"CauldronFiller","title_Chest Sorter (Assessment)":"ChestSorter",

                               "title_Mushroom Sorter (Assessment)":"MushroomSorter","world_CRYSTALCAVES":"CRYSTALCAVES",

                               "world_MAGMAPEAK":"MAGMAPEAK","world_TREETOPCITY":"TREETOPCITY","type_Assessment":"type"})
test_df_gp = df_test.groupby('installation_id')['timestamp'].agg('max').reset_index()

test_df_merge = pd.merge(df_test,test_df_gp,on=['installation_id','timestamp'],how='inner')
test_df_merge.info()
cols_test = ['game_session',

 'installation_id',

 'event_day',

 'event_hour',

 'BirdMeasurer',

 'CartBalancer',

 'CauldronFiller',

 'ChestSorter',

 'MushroomSorter',

 'CRYSTALCAVES',

 'MAGMAPEAK',

 'TREETOPCITY',

]



test_df_subset = test_df_merge[cols_test]



test_df_subset.head()

cat1 = sns.catplot(x="BirdMeasurer", hue="accuracy_group", data=df,

                height=6, aspect=1.5, kind="count", palette="colorblind")



cat1


cat2 = sns.catplot(x="CartBalancer", hue="accuracy_group", data=df,

                height=6, aspect=1.5, kind="count", palette="colorblind")



cat2
cat3 = sns.catplot(x="CauldronFiller", hue="accuracy_group", data=df,

                height=6, aspect=1.5, kind="count", palette="colorblind")



cat3
cat4 = sns.catplot(x="ChestSorter", hue="accuracy_group", data=df,

                height=6, aspect=1.5, kind="count", palette="colorblind")



cat4
cat5 = sns.catplot(x="MushroomSorter", hue="accuracy_group", data=df,

                height=6, aspect=1.5, kind="count", palette="colorblind")



cat5
cat6 = sns.catplot(x="CRYSTALCAVES", hue="accuracy_group", data=df,

                height=6, aspect=1.5, kind="count", palette="colorblind")



cat6
cat7 = sns.catplot(x="MAGMAPEAK", hue="accuracy_group", data=df,

                height=6, aspect=1.5, kind="count", palette="colorblind")



cat7
cat8 = sns.catplot(x="TREETOPCITY", hue="accuracy_group", data=df,

                height=6, aspect=1.5, kind="count", palette="colorblind")



cat8
def perform_logistic_regression(df_X, df_Y, test_df_X):

    logistic_regression = LogisticRegression()

    logistic_regression.fit(df_X, df_Y)

    pred_Y = logistic_regression.predict(test_df_X)

    accuracy = round(logistic_regression.score(df_X, df_Y) * 100,2)

    returnval = {'model':'Logistic Regression','accuracy':accuracy}

    return returnval
def perform_svc(df_X, df_Y, test_df_X):

    svc_clf = SVC()

    svc_clf.fit(df_X, df_Y)

    pred_Y = svc_clf.predict(test_df_X)

    accuracy = round(svc_clf.score(df_X, df_Y) * 100, 2)

    returnval = {'model':'SVC', 'accuracy':accuracy}

    return returnval
def perform_linear_svc(df_X, df_Y, test_df_X):

    svc_linear_clf = LinearSVC()

    svc_linear_clf.fit(df_X, df_Y)

    pred_Y = svc_linear_clf.predict(test_df_X)

    accuracy = round(svc_linear_clf.score(df_X, df_Y) * 100, 2)

    returnval = {'model':'LinearSVC', 'accuracy':accuracy}

    return returnval
def perform_rfc(df_X, df_Y, test_df_X):

    rfc_clf = RandomForestClassifier(n_estimators = 100 ,oob_score=True, max_features=None)

    rfc_clf.fit(df_X, df_Y)

    pred_Y = rfc_clf.predict(test_df_X)

    accuracy = round(rfc_clf.score(df_X, df_Y) * 100, 2)

    returnval = {'model':'RandomForestClassifier','accuracy':accuracy}

    return returnval
def perform_knn(df_X, df_Y, test_df_X):

    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(df_X, df_Y)

    pred_Y = knn.predict(test_df_X)

    accuracy = round(knn.score(df_X, df_Y) *100,2)

    returnval = {'model':'KNeighborsClassifier','accuracy':accuracy}

    return returnval
def perform_gnb(df_X, df_Y, test_df_X):

    gnb = GaussianNB()

    gnb.fit(df_X, df_Y)

    pred_Y = gnb.predict(test_df_X)

    accuracy = round(gnb.score(df_X, df_Y)*100,2)

    returnval = {'model':'GaussianNB','accuracy':accuracy}

    return returnval
def perform_dtree(df_X, df_Y, test_df_X):

    dtree = DecisionTreeClassifier()

    dtree.fit(df_X, df_Y)

    pred_Y = dtree.predict(test_df_X)

    accuracy = round(dtree.score(df_X, df_Y)*100,2)

    returnval = {'model':'DecisionTreeClassifier','accuracy':accuracy}

    return returnval
def perform_linear_regression(df_X, df_Y, test_df_X):

    linear_regression = LinearRegression()

    linear_regression.fit(df_X, df_Y)

    pred_Y = linear_regression.predict(test_df_X)

    # size_y = pred_Y.size

    # cks = cohen_kappa_score(df_Y[:size_y], pred_Y, weights="quadratic")

    accuracy = round(linear_regression.score(df_X, df_Y)*100,2)

    returnval = {'model':'LinearRegression','accuracy':accuracy}

    return returnval
X = df.drop(['game_session','installation_id','accuracy','accuracy_group'],axis=1)

y = df['accuracy_group']



test_X = test_df_subset.drop(['game_session','installation_id'],axis=1)
linreg_val = perform_linear_regression(X, y, test_X)

lr_val = perform_logistic_regression(X, y, test_X)

svc_val = perform_svc(X, y, test_X)

svc_lin_val = perform_linear_svc(X, y, test_X)

rfc_val = perform_rfc(X, y, test_X)

knn_val = perform_knn(X, y, test_X)

gnb_val = perform_gnb(X, y, test_X)

dtree_val = perform_dtree(X, y, test_X)

    

model_accuracies = pd.DataFrame()

model_accuracies = model_accuracies.append([linreg_val, lr_val,svc_val,svc_lin_val, rfc_val, knn_val, gnb_val, dtree_val])

# [linreg_val, lr_val,svc_val,svc_lin_val, rfc_val, knn_val, gnb_val, dtree_val]

cols = list(model_accuracies.columns.values)

cols = cols[-1:] + cols[:-1]

model_accuracies = model_accuracies[cols]

model_accuracies = model_accuracies.sort_values(by='accuracy')

print(model_accuracies)

plt.figure()

plt.xticks(rotation=90)

sns.barplot(x='model', y='accuracy', data=model_accuracies)
lg = RandomForestClassifier(n_estimators = 100 ,oob_score=True, max_features=None).fit(X, y)



y_pred = lg.predict(test_X)



y_pred = lg.predict(test_X)



test_X['accuracy_group'] = y_pred



test_X['accuracy_group'] = test_X['accuracy_group'].astype('int')



test_X['installation_id'] = test_df_subset['installation_id']



final_df = test_X[['installation_id','accuracy_group']]

final_df.to_csv('submission.csv',sep=',',index=False)



#final_df['accuracy_group'].value_counts()
