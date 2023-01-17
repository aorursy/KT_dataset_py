import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score

from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import precision_score, recall_score, confusion_matrix,  roc_curve, precision_recall_curve, accuracy_score, roc_auc_score

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import PCA

from sklearn.ensemble import BaggingClassifier

from lightgbm import LGBMClassifier

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

import plotly.offline as py

import os



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 



sns.set()



dataset = pd.read_csv('../input/ks-projects-201801.csv').set_index('ID')

dataset.head()
# Changfe state to boolean for predictions.(successful and failed)

print(dataset['state'].unique())



state_mapper = {"successful": 1} 



indexState = dataset[(dataset['state'] == 'live') | (dataset['state'] == 'undefined') | (dataset['state'] == 'suspended')].index

dataset['state'].drop(indexState , inplace=True)

dataset['state'] = dataset['state'].map(state_mapper)

dataset['state'].fillna(0, inplace = True)



print(dataset.groupby('state').size())
print("data nan Count:")

print('Nan in data:\n',dataset.isnull().sum())
plt.figure(figsize=(9,5))

sns.heatmap(dataset.corr(),linewidths=.5,cmap="YlGnBu")

plt.show()
# Goal



dataset['goal'].apply(lambda x: np.round(x))

dataset.groupby('goal').size().plot(title = "Goal Distribution",logx = True, figsize=(14,5))

plt.show()



dataset['goalQ'] = pd.qcut(dataset['goal'], 4,labels=False)



quarter_mapper = {}

for column in dataset['goalQ']:

    quarter_mapper[column] = "quar_"+str(column)



dataset['goalQ'] = dataset['goalQ'].map(quarter_mapper)



state_successful = dataset[dataset['state'] == True]

state_failed = dataset[dataset['state'] == False]

state_successful.groupby('goalQ').count()['state'].plot(title = "Goal Distribution",marker='o', figsize=(14,5), color = 'g')

state_failed.groupby('goalQ').count()['state'].plot(color = 'r',marker='o')

state_successful.groupby('goalQ').size().plot.bar(color = 'b', alpha = 0.5)

state_failed.groupby('goalQ').size().plot.bar(color = 'y',alpha = 0.5)

plt.show()



print("By spliting to 4 quarters \nwe can see that when the goad is bigger the more chance to failed colocation the money")
# main category feature



main_category_mean = dataset.groupby('main_category').mean()['state'].sort_values(ascending = False)

print(main_category_mean)

main_category_mean.plot.bar(figsize=(14,5), title = "Main category state mean")

plt.axhline(0.25,color = 'b',linestyle='--')

plt.axhline(0.41,color = 'b',linestyle='--')

plt.show()
bins = np.array([0.0, 0.3, 0.41, 1])

inds = np.digitize(main_category_mean, bins)

category_scores = pd.DataFrame(inds, index = main_category_mean.index)

print(category_scores)

category_scores_dict = category_scores.to_dict()[0]

category_mapper = {1 : 'low',2 : 'mid',3 : 'high'} 



dataset['main_category_score'] = dataset['main_category'].map(category_scores_dict)

dataset['main_category_score'] = dataset['main_category_score'].map(category_mapper)

dataset.groupby('main_category_score').mean()['state'].plot.bar(title = "Main category score mean")

plt.show()
category_groupby = dataset.groupby('category').agg([np.size,np.mean])['state']

mean_size = category_groupby['size'].mean()

category_groupby = category_groupby[category_groupby['size'] > mean_size] # drop low size

category_groupby.sort_values(by = 'mean' ,ascending = False, inplace = True)



category_groupby['round_mean'] = round(category_groupby['mean'],1)

# high_mean_category = category_mean[category_mean['mean'] > 0.5]

# low_mean_category = category_mean[category_mean['mean'] < 0.21]



category_groupby['mean'].plot.bar(figsize=(14,5), title = "category mean")

plt.show()

category_groupby['round_mean'].plot.bar(figsize=(14,5), title = "category round mean")

plt.show()



bins = np.array([0.1,0.2, 0.3, 0.4, 0.5])

inds = np.digitize(category_groupby['round_mean'], bins)

category_scores = pd.DataFrame(inds, index = category_groupby['round_mean'].index)

category_scores_dict = category_scores.to_dict()

category_mapper = {1 : 'low',2 : 'low_mid',3 : 'mid',4 : 'mid_high',5 : 'high'} 





dataset['category_score'] = dataset['category'].map(category_scores_dict[0])

dataset['category_score'].fillna(round(dataset['category_score'].median()), inplace = True)

dataset['category_score'] = dataset['category_score'].map(category_mapper)

dataset.groupby('category_score').mean()['state'].sort_values().plot.bar(title = "sub category score mean")

plt.show()
# Backers

plt.figure(figsize=(10,5))

sns.kdeplot(state_failed['backers'],shade=True,color='Red', label='failed').set_xlim(0,20000)

sns.kdeplot(state_successful['backers'],shade=True,color='Green', label='successful').set_xlim(0,20000)



plt.title('Backers Vs State')

plt.axvline(2000,color = 'b',linestyle='--')

plt.show()



print("we can see distribution between failed (under 2000), and successful (bigger the 2000)")
# amount pledged by "crowd"



plt.figure(figsize=(10,5))

sns.kdeplot(state_failed['pledged'],shade=True,color='Red', label='failed').set_xlim(0.5)

sns.kdeplot(state_successful['pledged'],shade=True,color='Green', label='successful').set_xlim(0.5)

plt.title('pledged Vs State')

plt.axvline(200000,color = 'b',linestyle='--')

plt.axvline(3700000,color = 'b',linestyle='--')

plt.show()



#We can see distribution between failed (under 200000), and successful (bigger the 200000)
# deadline

dataset['launched_year'] = pd.DatetimeIndex(dataset['launched']).year

dataset['launched_month'] = pd.DatetimeIndex(dataset['launched']).month



plt.figure(figsize=(20,5))

dataset.groupby(['launched_year','launched_month']).mean()['state'].plot.bar(title = "launched year and month Vs state")

plt.axvline(63.5,color = 'b',linestyle='--')

plt.axvline(83.5,color = 'b',linestyle='--')

plt.show()



print(dataset.groupby('launched_year').size())

print()

print("between 7.2014 to 3.2016 there was growth of use in kickstart, but this doesn’t translate in more projects getting funded.")

print()

print("from - http://icopartners.com/2016/02/2015-in-review/")

print("'The growth of total number of projects is significant though, meaning more creators are coming to Kickstarter to finance their projects, but this doesn’t translate in more projects getting funded.'")



# between 7.2014 to 3.2016 there was less successful



def weak_year(year):

    if year >= 2014 & year <= 2016:

        if year == 2014 & year > 6:

            return True

        if year == 2016 & year < 3:

            return True

        if year == 2015:

            return True

    return False



dataset['is_launched_weak_year'] = dataset['launched_year'].apply(weak_year)



dataset['launched_day'] = pd.DatetimeIndex(dataset['launched']).day

dataset.groupby('launched_day').mean()['state'].plot.bar(title = "launched day Vs state")

plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))



dataset['launched_hour'] = pd.DatetimeIndex(dataset['launched']).hour

dataset['dayfloat']=dataset.launched_day+dataset.launched_hour/24.0

dataset['monthfloat']=dataset.launched_month+dataset.launched_day/28.



dataset['x_launched']=np.sin(2.*np.pi*dataset.monthfloat/12.)

dataset['y_launched']=np.cos(2.*np.pi*dataset.monthfloat/12.)



ax = sns.scatterplot(x="x_launched", y="y_launched", hue="state",style="state",alpha = 0.4,palette = 'Set1_r',ax = axes[1], data=dataset)

ax.set_title("launched time")



dataset['deadline_hour'] = pd.DatetimeIndex(dataset['deadline']).hour

dataset['deadline_year'] = pd.DatetimeIndex(dataset['deadline']).year

dataset['deadline_month'] = pd.DatetimeIndex(dataset['deadline']).month



dataset['dayfloat']=dataset.launched_day+dataset.launched_hour/24.0

dataset['monthfloat']=dataset.launched_month+dataset.launched_day/28.



dataset['x_deadline']=np.sin(2.*np.pi*dataset.monthfloat/12.)

dataset['y_deadline']=np.cos(2.*np.pi*dataset.monthfloat/12.)



ax = sns.scatterplot(x="x_deadline", y="y_deadline", hue="state",style="state",alpha = 0.2,palette = 'Set1_r',ax = axes[0], data=dataset)

ax.set_title("deadline time")

plt.show()
#  currncy

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13,5))



dataset.groupby(['currency']).mean()['state'].plot.bar(title = "mean currency vs state", ax = axes[0])

dataset.groupby(['currency']).sum()['state'].plot.bar(title = "count currency vs state", ax = axes[1])

plt.show()



# it better use EasyPeasy ,but I dont what to enable internet with tbhe karnel

def covenrtToUSD(goal,currency):

    switcher = {

        'USD': goal,

        'EUR': goal*1.13109,

        'MXN': goal*0.05176,

        'AUD': goal*0.70616,

        'GBP': goal*1.32556,

        'CAD': goal*0.75065,

        'SEK': goal*0.10741,

        'NOK': goal*0.11655,

        'CHF': goal*0.99581,

        'HKD': goal*0.12739,

        'DKK': goal*0.00834,

        'SGD': goal*0.73782,

        'NZD': goal*0.68293,

    }

    return switcher.get(currency,goal)



dataset["goal"] = dataset.apply(lambda x: covenrtToUSD(x.goal, x.currency), axis=1)
dataset["project_time"] = (pd.DatetimeIndex(dataset['deadline']) - pd.DatetimeIndex(dataset['launched'])).days

dataset.plot.scatter("project_time","state", title= "project_time VS state")

plt.axvline(6000,color = 'b',linestyle='--')

plt.show()



print("We will remove the project that bigger then 6000 days (16 year)")

print("kickstart was founded in April 28, 2009 - 10 year ago")

print("bigger then 6000:",len(dataset[dataset["project_time"] > 6000]))

print("less then 6000:",len(dataset[dataset["project_time"] < 6000]))



# drop outliar 

dataset['goal_per_day'] = dataset["goal"] / (dataset["project_time"]+0.0001)

sns.swarmplot(x="state", y="goal_per_day", data=dataset.sample(300,random_state = 444))

plt.title("goal per day")

plt.show()


fig, axes = plt.subplots(nrows=2, ncols=1)

fig.tight_layout()

sns.kdeplot(state_failed['usd_pledged_real'],shade=True,color='Red', label='failed', ax = axes[0])

ax = sns.kdeplot(state_successful['usd_pledged_real'],shade=True,color='Green', label='successful', ax = axes[0])

ax.set_title("usd pledged Vs state")

sns.kdeplot(state_failed['usd_goal_real'],shade=True,color='Red', label='failed', ax = axes[1])

ax = sns.kdeplot(state_successful['usd_goal_real'],shade=True,color='Green', label='successful', ax = axes[1])

ax.set_title("usd goal Vs state")



plt.show()





dataset['diff_pledged_goal'] = round(np.log(dataset['usd_pledged_real']+1) - np.log(dataset['usd_goal_real']+1))



plt.title('diff_pledged_goal Vs state')



state_successful = dataset[dataset['state'] == True]

state_failed = dataset[dataset['state'] == False]

sns.kdeplot(state_failed['diff_pledged_goal'],shade=True,color='Red', label='failed')

sns.kdeplot(state_successful['diff_pledged_goal'],shade=True,color='Green', label='successful')

plt.show()
dataset['diff_pledged_desirable_real'] = round(np.log(dataset['usd_pledged_real']+1) - np.log(dataset['usd pledged']+1))



state_successful = dataset[dataset['state'] == True]

state_failed = dataset[dataset['state'] == False]



sns.kdeplot(state_failed['diff_pledged_desirable_real'],shade=True,color='Red', label='failed').set_xlim(-0.5,0.5)

sns.kdeplot(state_successful['diff_pledged_desirable_real'],shade=True,color='Green', label='successful').set_xlim(-0.5,0.5)

plt.title("pledged_real - usd pledged Vs state")

plt.show()



dataset['diff_pledged_desirable_real'].fillna(dataset['diff_pledged_desirable_real'].mean(), inplace = True)
dataset['word_count'] = dataset['name'].str.split().apply(np.size)

word_count_size_mean = dataset.groupby('word_count').agg([np.mean,np.size])['state']

word_count_size_mean = word_count_size_mean[word_count_size_mean['size'] > 500]

word_count_size_mean['mean'].plot.area(title = "Number of word")

plt.show()



dataset['name_len'] = dataset['name'].str.len()

dataset.groupby('name_len').mean()['state'].plot(title = "Name character number")

plt.show()



dataset.dropna(subset = ['name'] ,inplace = True)


# help func for model_performance

# please use:

# y_pred = clf.predict(X_valid)

# y_score = clf.predict_proba(X_valid)[:,1]

# X_train, X_test, y_train, y_test = train_test_split(X, y)

def model_performance(model) : 

    #Conf matrix

    conf_matrix = confusion_matrix(y_test, y_pred)

    #Show metrics

    tp = conf_matrix[1,1]

    fn = conf_matrix[1,0]

    fp = conf_matrix[0,1]

    tn = conf_matrix[0,0]

    Accuracy  =  ((tp+tn)/(tp+tn+fp+fn))

    Precision =  (tp/(tp+fp))

    Recall    =  (tp/(tp+fn))

    F1_score  =  (2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn)))))



    model_roc_auc = round(roc_auc_score(y_test, y_score) , 3)



    print(conf_matrix)

    ax= plt.subplot()

    sns.heatmap(conf_matrix, annot=True, ax = ax); 



    # labels, title and ticks

    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

    ax.set_title('Confusion Matrix'); 

    ax.xaxis.set_ticklabels(['successful', 'failed']); ax.yaxis.set_ticklabels(['successful', 'failed'])

    plt.show()

    

    

    y_pred_proba = y_score

    fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)

    auc = roc_auc_score(y_test, y_pred_proba)

    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

    plt.title("roc_curve")

    plt.legend(loc=4)

    plt.show()

  

    print(classification_report(y_test,y_pred))

    

    print('model_roc_auc',model_roc_auc)
def train_test_split_balance_min(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)



    X_train['state'] = y_train

    positiveData = X_train[X_train['state'] == True]

    nagitiveData = X_train[X_train['state'] == False]

    if len(positiveData) > len(nagitiveData):

        X_train = pd.concat([nagitiveData,positiveData.sample(len(nagitiveData))])

    else:

        X_train = pd.concat([positiveData,nagitiveData.sample(len(positiveData))])

    

    y_train = X_train['state']

    X_train = X_train.drop('state',axis = 1)

    return X_train, X_test, y_train, y_test



def train_test_split_balance_max(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y)



    X_train['state'] = y_train

    positiveData = X_train[X_train['state'] == True]

    nagitiveData = X_train[X_train['state'] == False]

    if len(positiveData) > len(nagitiveData):

        X_train = pd.concat([X_train,nagitiveData.sample(len(positiveData) - len(nagitiveData))])

    else:

        X_train = pd.concat([X_train,positiveData.sample(len(nagitiveData) - len(positiveData))])

    

    y_train = X_train['state']

    X_train = X_train.drop('state',axis = 1)

    return X_train, X_test, y_train, y_test
to_prdic = dataset.drop('state',axis = 1)

y = dataset['state']



prdict_feature = ['diff_pledged_goal']

prdict_df = to_prdic[prdict_feature]

mms = MinMaxScaler()

mms.fit(prdict_df)

X = pd.DataFrame(mms.transform(prdict_df))
X_train, X_test, y_train, y_test = train_test_split(X, y)



clf = DecisionTreeClassifier(max_depth = 2,class_weight = "balanced")

clf.fit(X_train,y_train)



clf.fit(X_train, y_train)



scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_macro')

print("Cross validated:",scores)

print("Cross validated mean:",scores.mean())
# Perper Transformer

from sklearn.preprocessing import QuantileTransformer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer  

from sklearn.pipeline import Pipeline

from sklearn.pipeline import FeatureUnion

from sklearn.preprocessing import FunctionTransformer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.compose import ColumnTransformer

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LassoCV

from sklearn.preprocessing import KBinsDiscretizer
class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key



    def fit(self, x, y=None):

        return self



    def transform(self, data_dict):

        return data_dict[self.key]
X = dataset.drop('state',axis = 1)

y = dataset['state']



# select features

prdict_feature = ['goal','name','main_category_score','category_score','project_time','goal_per_day','word_count']

X = X[prdict_feature]



# split

X_train, X_test, y_train, y_test = train_test_split(X, y)



# name

name_pipeline = Pipeline([('name_column', ItemSelector(key = 'name')),

                          ('vectorizer', CountVectorizer(max_features=100))])



# dummeys

one_hot_pipeline = Pipeline([('hot_columns', ItemSelector(key = ['main_category_score','category_score'])),

                             ('oneHowEncoder', OneHotEncoder(handle_unknown='ignore',sparse=True))])



# min max scaler

min_max_pipeline = Pipeline([('min_max_columns', ItemSelector(key = ['project_time','word_count','goal_per_day'])),

                             ('minMaxScaler', MinMaxScaler())])



# min max scaler

k_bins_pipeline = Pipeline([('goal', ItemSelector(key = ['goal'])),

                             ('k_bins', KBinsDiscretizer(n_bins = 4,encode = 'onehot',strategy = 'quantile'))])



# FeatureUnion 

feature_pipeline = FeatureUnion([('one_hot',one_hot_pipeline),

                                 ('name',name_pipeline),

                                 ('min_max',min_max_pipeline),

                                 ('k_bins',k_bins_pipeline)])





feature_pipeline.fit(X_train)



X_train = feature_pipeline.transform(X_train)

X_test = feature_pipeline.transform(X_test)

%%time

scoreTest_DT = []

scoreTrain_DT = []

for number in range(1,30):

    clf = DecisionTreeClassifier(max_depth = number, class_weight = "balanced")

    clf.fit(X_train,y_train)

    y_score_train = clf.predict_proba(X_train)[:,1]

    y_score_test = clf.predict_proba(X_test)[:,1]



    scoreTrain_DT.append(round(roc_auc_score(y_train, y_score_train) , 3))

    scoreTest_DT.append(round(roc_auc_score(y_test, y_score_test) , 3))

    

pd.DataFrame({'test roc score':scoreTest_DT,'train roc score':scoreTrain_DT}).plot(grid = True)

plt.xlabel('Max depth')

plt.ylabel('Score')

plt.title("DecisionTreeClassifier")

plt.show()



clf_DT = DecisionTreeClassifier(max_depth = np.array(scoreTest_DT).argmax(), class_weight = "balanced")

clf_DT.fit(X_train,y_train)



print("DT roc_train:",round(roc_auc_score(y_train, clf_DT.predict_proba(X_train)[:,1]) , 3))

print("DT roc_test:",round(roc_auc_score(y_test, clf_DT.predict_proba(X_test)[:,1]) , 3))



# DT_cross_score = cross_val_score(clf_DT, X_train, y_train, cv=5, scoring='roc_auc').mean()

DT_roc = round(roc_auc_score(y_test, clf_DT.predict_proba(X_test)[:,1]) , 3)
%%time

scoreTest_RF = []

scoreTrain_RF = []

for number in range(1,30):

    clf = RandomForestClassifier(max_depth = number,n_estimators = 100, class_weight = "balanced")

    clf.fit(X_train,y_train)

    y_score_train = clf.predict_proba(X_train)[:,1]

    y_score_test = clf.predict_proba(X_test)[:,1]



    scoreTrain_RF.append(round(roc_auc_score(y_train, y_score_train) , 3))

    scoreTest_RF.append(round(roc_auc_score(y_test, y_score_test) , 3))

    

pd.DataFrame({'test roc score':scoreTest_RF,'train roc score':scoreTrain_RF}).plot(grid = True)

plt.xlabel('Max depth')

plt.ylabel('Score')

plt.title("RandomForestClassifier")

plt.show()







cls_RF = RandomForestClassifier(max_depth = np.array(scoreTest_RF).argmax(),n_estimators = 100, class_weight = "balanced")

cls_RF.fit(X_train,y_train)



print("RF roc_train:",round(roc_auc_score(y_train, cls_RF.predict_proba(X_train)[:,1]) , 3))

print("RF roc_test:",round(roc_auc_score(y_test, cls_RF.predict_proba(X_test)[:,1]) , 3))



# RF_cross_score = cross_val_score(cls_RF, X_train, y_train, cv=5, scoring='roc_auc').mean()

RF_roc = round(roc_auc_score(y_test, cls_RF.predict_proba(X_test)[:,1]) , 3)
%%time

# scoreTest = []

# scoreTrain = []

# knnRange = [50,100,200,300,350]

# for number in knnRange:

#     clf = KNeighborsClassifier(n_neighbors = number)

#     clf.fit(X_train,y_train)

#     y_score_train = clf.predict_proba(X_train)[:,1]

#     y_score_test = clf.predict_proba(X_test)[:,1]



#     scoreTrain.append(round(roc_auc_score(y_train, y_score_train) , 3))

#     scoreTest.append(round(roc_auc_score(y_test, y_score_test) , 3))

    

# pd.DataFrame({'test roc score':scoreTest,'train roc score':scoreTrain}).plot(grid = True)

# plt.xlabel('Max depth')

# plt.ylabel('Score')

# plt.title("KNeighborsClassifier")

# plt.show()



# clf_KNN = KNeighborsClassifier(n_neighbors = knnRange[np.array(scoreTest).argmax()])

clf_KNN = KNeighborsClassifier(n_neighbors = 50)

clf_KNN.fit(X_train,y_train)



print("KNN roc_train:",round(roc_auc_score(y_train, clf_KNN.predict_proba(X_train)[:,1]) , 3))

print("KNN roc_test:",round(roc_auc_score(y_test, clf_KNN.predict_proba(X_test)[:,1]) , 3))



# KNN_cross_score = cross_val_score(clf_KNN, X_train, y_train, cv=5, scoring='roc_auc').mean()

KNN_roc = round(roc_auc_score(y_test, clf_KNN.predict_proba(X_test)[:,1]) , 3)
%%time

clf_NB = GaussianNB()

clf_NB.fit(X_train.todense(),y_train)

y_score_train = clf_NB.predict_proba(X_train.todense())[:,1]

y_score_test = clf_NB.predict_proba(X_test.todense())[:,1]



print("NB roc_train:",round(roc_auc_score(y_train, y_score_train) , 3))

print("NB roc_test:",round(roc_auc_score(y_test, y_score_test) , 3))



# NB_cross_score = cross_val_score(clf_NB, X_train.todense(), y_train, cv=5, scoring='roc_auc').mean()

NB_roc = round(roc_auc_score(y_test, y_score_test) , 3)
%%time



scoreTest_LR = []

scoreTrain_LR = []

C_range = [1,2,3,4,5,6,7,8]

for number in C_range:

    clf = LogisticRegression(C = number, class_weight = "balanced", penalty = 'l2')

    clf.fit(X_train,y_train)

    y_score_train = clf.predict_proba(X_train)[:,1]

    y_score_test = clf.predict_proba(X_test)[:,1]

    scoreTrain_LR.append(round(roc_auc_score(y_train, y_score_train) , 3))

    scoreTest_LR.append(round(roc_auc_score(y_test, y_score_test) , 3))

     

pd.DataFrame({'test roc score':scoreTest_LR,'train roc score':scoreTrain_LR}).plot(grid = True)

plt.xlabel('C')

plt.ylabel('Score')

plt.title("LogisticRegression")

plt.show()



print("LR roc_train:",round(roc_auc_score(y_train, clf.predict_proba(X_train)[:,1]) , 3))

print("LR roc_test:",round(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]) , 3))



clf_LR = LogisticRegression(C = C_range[np.array(scoreTest_LR).argmax()], class_weight = "balanced")

clf_LR.fit(X_train,y_train)



# LR_cross_score = cross_val_score(clf_LR, X_train, y_train, cv=5, scoring='roc_auc').mean()

LR_roc = round(roc_auc_score(y_test, clf_LR.predict_proba(X_test)[:,1]) , 3)
%%time

from sklearn.neural_network import MLPClassifier



clf_MLP = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=100,tol=0.0001, alpha=0.0001,

                     solver='sgd', verbose= False)

clf_MLP.fit(X_train, y_train)



print("MLP roc_train:",round(roc_auc_score(y_train, clf_MLP.predict_proba(X_train)[:,1]) , 3))

print("MLP roc_test:",round(roc_auc_score(y_test, clf_MLP.predict_proba(X_test)[:,1]) , 3))



# MLP_cross_score = cross_val_score(clf_MLP, X_train, y_train, cv=5, scoring='roc_auc').mean()

MLP_roc = round(roc_auc_score(y_test, clf_MLP.predict_proba(X_test)[:,1]) , 3)
%%time

scoreTrain_LGBMC = []

scoreTest_LGBMC = []

n_estimators = [100,400,800,1000]

for number in n_estimators:

    clf = LGBMClassifier(

        n_estimators= number,

        num_leaves=15,

        colsample_bytree=.8,

        subsample=.8,

        max_depth=7,

        reg_alpha=.1,

        reg_lambda=.1,

        min_split_gain=.01

    )

    clf.fit(X_train,y_train)

    y_score_train = clf.predict_proba(X_train)[:,1]

    y_score_test = clf.predict_proba(X_test)[:,1]

    scoreTrain_LGBMC.append(round(roc_auc_score(y_train, y_score_train) , 3))

    scoreTest_LGBMC.append(round(roc_auc_score(y_test, y_score_test) , 3))

     

pd.DataFrame({'test roc score':scoreTest_LGBMC,'train roc score':scoreTrain_LGBMC}).plot(grid = True)

plt.xlabel('n_estimators')

plt.ylabel('Score')

plt.title("LGBMClassifier")

plt.show()





clf_lgbm = LGBMClassifier(

        n_estimators= n_estimators[np.array(scoreTest_LGBMC).argmax()],

        num_leaves=15,

        colsample_bytree=.8,

        subsample=.8,

        max_depth=7,

        reg_alpha=.1,

        reg_lambda=.1,

        min_split_gain=.01

    )

clf_lgbm.fit(X_train, y_train)



print("LGBM roc_train:",round(roc_auc_score(y_train, clf_lgbm.predict_proba(X_train)[:,1]) , 3))

print("LGBM roc_test:",round(roc_auc_score(y_test, clf_lgbm.predict_proba(X_test)[:,1]) , 3))



LGBM_roc = round(roc_auc_score(y_test, clf_lgbm.predict_proba(X_test)[:,1]) , 3)
%%time

clf_MLP_vot = MLPClassifier(hidden_layer_sizes=(50,50), max_iter=100,tol=0.0001, alpha=0.0001,

                     solver='sgd', verbose= False)

clf_LR_vot = LogisticRegression(C = C_range[np.array(scoreTest_LR).argmax()], class_weight = "balanced")

clf_DT_vot = DecisionTreeClassifier(max_depth = np.array(scoreTest_DT).argmax(), class_weight = "balanced")

clf_lgbm_vot = LGBMClassifier(

        n_estimators= n_estimators[np.array(scoreTest_LGBMC).argmax()],

        num_leaves=15,

        colsample_bytree=.8,

        subsample=.8,

        max_depth=7,

        reg_alpha=.1,

        reg_lambda=.1,

        min_split_gain=.01

    )



clf_vot = VotingClassifier(estimators=[('MLP', clf_MLP_vot), ('RL', clf_LR_vot),('LGBM',clf_lgbm_vot)],voting='soft', weights=[1,1,2])

clf_vot.fit(X_train, y_train)



print("voting roc_train:",round(roc_auc_score(y_train, clf_vot.predict_proba(X_train)[:,1]) , 3))

print("voting roc_test:",round(roc_auc_score(y_test, clf_vot.predict_proba(X_test)[:,1]) , 3))



VOT_roc = round(roc_auc_score(y_test, clf_vot.predict_proba(X_test)[:,1]) , 3)
%%time

clf_bag = LogisticRegression(C = C_range[np.array(scoreTest_LR).argmax()], class_weight = "balanced")

clf_LR_Bag = BaggingClassifier(base_estimator = clf_bag,n_estimators = 10)

clf_LR_Bag.fit(X_train, y_train)



print("Bagging roc_train:",round(roc_auc_score(y_train, clf_LR_Bag.predict_proba(X_train)[:,1]) , 3))

print("Bagging roc_test:",round(roc_auc_score(y_test, clf_LR_Bag.predict_proba(X_test)[:,1]) , 3))



LR_BAG_roc = round(roc_auc_score(y_test, clf_LR_Bag.predict_proba(X_test)[:,1]) , 3)


indexs = ["DecisionTreeClassifier","RandomForestClassifier","KNeighborsClassifier","GaussianNB","LogisticRegression","MLPClassifier","VotingClassifier","BaggingClassifierLR","LGBMClassifier"]

models = pd.DataFrame([], index = indexs)

models["Roc"] = [DT_roc,RF_roc, KNN_roc,NB_roc,LR_roc,MLP_roc,VOT_roc,LR_BAG_roc,LGBM_roc]

models.plot.barh(figsize=(10,10 ), xlim = (0.6,0.8),colormap='YlOrRd_r')

plt.axvline(models["Roc"].max(),color = 'g',linestyle='--')

plt.show()



models
y_pred = clf_lgbm.predict(X_test)

y_score = clf_lgbm.predict_proba(X_test)[:,1]



scores = cross_val_score(clf_lgbm, X_train, y_train, cv=5, scoring='roc_auc')

print("Cross validated:",scores)

print("Cross validated mean:",scores.mean())

print()

model_performance('clf_lgbm')
clf_lgbm
print("successful count:",len(y_train[y_train == True]))

print("failed count:",len(y_train[y_train == False]))
X_train, X_test, y_train, y_test = train_test_split_balance_min(X, y)

feature_pipeline.fit(X_train)

X_train = feature_pipeline.transform(X_train)

X_test = feature_pipeline.transform(X_test)



clf_lgbm.fit(X_train, y_train)

print(classification_report(y_test,clf_lgbm.predict(X_test)))

print("roc:",round(roc_auc_score(y_test, clf_lgbm.predict_proba(X_test)[:,1]) , 3))
X_train, X_test, y_train, y_test = train_test_split_balance_max(X, y)

feature_pipeline.fit(X_train)

X_train = feature_pipeline.transform(X_train)

X_test = feature_pipeline.transform(X_test)



clf_lgbm.fit(X_train, y_train)

print(classification_report(y_test,clf_lgbm.predict(X_test)))

print("roc:",round(roc_auc_score(y_test, clf_lgbm.predict_proba(X_test)[:,1]) , 3))