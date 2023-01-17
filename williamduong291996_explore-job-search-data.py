# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns 

import os





from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import lightgbm as lgb



from imblearn.under_sampling import RandomUnderSampler



print(os.listdir("../input"))



# 1) title proximity tfidf: Measures the closeness of query and job title.

# 2) description proximity tfidf: Measures the closeness of query and job description.

# 3) main query tfidf: A score related to user query closeness to job title and job description. 4) query jl score: Measures the popularity of query and job listing pair.

# 5) query title score: Measures the popularity of query and job title pair.

# 6) city match: Indicates if the job listing matches to user (or, user-specified) location.

# 7) job age days: Indicates the age of job listing posted.

# 8) apply: Indicates if the user has applied for this job listing.

# 9) search date pacific: Date of the activity.

# 10) class id: Class ID of the job title clicked.



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/Apply_Rate_2019.csv")
dataset.info()
dataset.head()
split_date = pd.datetime(2018,1,26)

dataset['search_date_pacific'] = pd.to_datetime(dataset['search_date_pacific'])



train_data = dataset[dataset['search_date_pacific']<=split_date]

valid_data  = dataset[dataset['search_date_pacific']>split_date]
train_data.info()
valid_data.info()
train_data.drop_duplicates()

train_data.info()
features = list(dataset.columns[:8])

features


sns.pairplot(train_data[features].dropna())
# create a test set to check the accuracy beside the final prediction

train_set, test_set= train_test_split(train_data[features] , test_size=0.1, random_state=30, shuffle=True)
train_set.info()

test_set.info()
# generate data to fill out missing value 

# build regression model 

import lightgbm as lgb

params = {'objective':'regression',

          'num_leaves' : 40,

          'min_data_in_leaf' : 20,

          'max_depth' : 4,

          'learning_rate': 0.01,

          "metric": 'rmse',

          "random_state" : 42,

          "verbosity": -1}

features = ['main_query_tfidf',

 'query_jl_score',

 'query_title_score',

'job_age_days']



# train_set = train_set[:50000]



X1 = train_set[train_set['title_proximity_tfidf'].isnull()== False]

X1_test = train_set[train_set['title_proximity_tfidf'].isnull()== True]



# X1.info()

# X1_test.info()

# sns.pairplot(X1[features].dropna())

X1_train, X1_valid, y1_train, y1_valid = train_test_split(X1[features], X1['title_proximity_tfidf'], test_size=0.1, random_state=30, shuffle=True)



X2 = train_set[train_set['description_proximity_tfidf'].isnull()== False]

X2_test = train_set[train_set['description_proximity_tfidf'].isnull()== True]

X2_train, X2_valid, y2_train, y2_valid = train_test_split(X2[features], X2['description_proximity_tfidf'], test_size=0.1, random_state=30, shuffle=True)



X3 = train_set[train_set['city_match'].isnull()== False]

X3_test = train_set[train_set['city_match'].isnull()== True]

X3_train, X3_valid, y3_train, y3_valid = train_test_split(X3[features], X3['city_match'], test_size=0.1, random_state=30, shuffle=True)







X_non_zeros = X1[X1['title_proximity_tfidf'] > 0.0]

X_zeros = X1[X1['title_proximity_tfidf'] == 0.0]

ax = sns.countplot(X_zeros["apply"],label="Count")       # M = 212, B = 357

N, Y = X_zeros["apply"].value_counts()

print('Number of apply: ',Y)

print('Number of not apply : ',N)



bx = sns.countplot(X_non_zeros["apply"],label="Count")       # M = 212, B = 357

N, Y = X_non_zeros["apply"].value_counts()

print('Number of apply: ',Y)

print('Number of not apply : ',N)
day_post_below = train_set[train_set['job_age_days']< 200]

day_post_above = train_set[train_set['job_age_days']>= 200]





N, Y = day_post_below["apply"].value_counts()

print('Number of apply: ',Y)

print('Number of not apply : ',N)



# bx = sns.countplot(X_non_zeros["apply"],label="Count")       # M = 212, B = 357

N, Y = day_post_above["apply"].value_counts()

print('Number of apply: ',Y)

print('Number of not apply : ',N)
# Investigate city match



city_match = train_set[train_set['city_match'].isnull()== False]

city_match.info()

match_apply = city_match[(city_match['city_match'] == 1) & (city_match['apply'] == 1)]

match_not_apply = city_match[(city_match['city_match'] == 1) & (city_match['apply'] == 0)]

not_match_apply = city_match[(city_match['city_match'] == 0) & (city_match['apply'] == 1)]

not_match_not_apply = city_match[(city_match['city_match'] == 0) & (city_match['apply'] == 0)]



print ("match apply : ",match_apply.shape)

print ("match not apply : ",match_not_apply.shape)

print(" not match apply : ",not_match_apply.shape)

print(" not match not apply : ",not_match_not_apply.shape)

sns.pairplot(X1_test[features])
# create dataset for lightgbm

lgb1_train = lgb.Dataset(X1_train, y1_train)

lgb1_eval = lgb.Dataset(X1_valid, y1_valid, reference=lgb1_train)

gbm1 = lgb.train(params,

                lgb1_train,

                num_boost_round=5000,

                valid_sets=lgb1_eval,

                early_stopping_rounds=5)
lgb2_train = lgb.Dataset(X2_train, y2_train)

lgb2_eval = lgb.Dataset(X2_valid, y2_valid, reference=lgb2_train)



gbm2 = lgb.train(params,

                lgb2_train,

                num_boost_round=5000,

                valid_sets=lgb2_eval,

                early_stopping_rounds=5)






city_match_classifier = KNeighborsClassifier(3)

print (X3_train.values.shape)

city_match_classifier.fit(X3_train.values,y3_train)

y3_valid_predict = city_match_classifier.predict(X3_valid[features].values)

print (roc_auc_score(y3_valid, y3_valid_predict))
# def fill_miss_value(row,model1,model2,model3,features):

#     if (np.isnan(row['title_proximity_tfidf'])):

#         r =  model1.predict(row[features])

#         print (r.shape)

#         row['title_proximity_tfidf'] =  model1.predict(row[features])[0]

#     if (np.isnan(row['description_proximity_tfidf'])):

#         b = model2.predict(row[features])

#         print (b.shape)

#         row['description_proximity_tfidf'] = model2.predict(row[features])[0]

#     if (np.isnan(row['city_match'])):

#         print (row[features].shape)

#         c = model3.predict(row[features])

#         print (c.shape)

#     return row

# 





title_proximity_predict_result = gbm1.predict(X1_test[features].values)

print (title_proximity_predict_result[:5])

X1_test.loc[:,'title_proximity_tfidf']= title_proximity_predict_result

X1 = X1.append(X1_test)

print (X1.info())



X2 = X1[X1['description_proximity_tfidf'].isnull()== False]

X2_test = X1[X1['description_proximity_tfidf'].isnull()== True]

description_proximity_predict_result = gbm2.predict(X2_test[features].values)

print (description_proximity_predict_result[:5])

X2_test.loc[:,'description_proximity_tfidf']= description_proximity_predict_result

X2 = X2.append(X2_test)

print(X2.info())



X3 = X2[X2['city_match'].isnull()== False]

X3_test = X2[X2['city_match'].isnull()== True]

city_predict_result = city_match_classifier.predict(X3_test[features].values)

X3_test.loc[:,'city_match']= city_predict_result

print (X3_test.info())

X3 = X3.append(X3_test)

print (X3.info())



train_set = X3

X1 = test_set[test_set['title_proximity_tfidf'].isnull()== False]

X1_test = test_set[test_set['title_proximity_tfidf'].isnull()== True]



title_proximity_predict_result = gbm1.predict(X1_test[features].values)

X1_test.loc[:,'title_proximity_tfidf']= title_proximity_predict_result

X1 = X1.append(X1_test)

print (X1.info())



X2 = X1[X1['description_proximity_tfidf'].isnull()== False]

X2_test = X1[X1['description_proximity_tfidf'].isnull()== True]

description_proximity_predict_result = gbm2.predict(X2_test[features].values)

X2_test.loc[:,'description_proximity_tfidf']= description_proximity_predict_result

X2 = X2.append(X2_test)

print(X2.info())



X3 = X2[X2['city_match'].isnull()== False]

X3_test = X2[X2['city_match'].isnull()== True]

city_predict_result = city_match_classifier.predict(X3_test[features].values)

X3_test.loc[:,'city_match']= city_predict_result

print (X3_test.info())

X3 = X3.append(X3_test)

print (X3.info())



test_set = X3
# ax = sns.countplot(train_set["apply"],label="Count")   

# N, Y = train_set["apply"].value_counts()

# print('Number of apply: ',Y)

# print('Number of not apply : ',N)

# investigate features with missing values 

features_missing = ["title_proximity_tfidf","description_proximity_tfidf","city_match"]

for f in features_missing:

    miss = train_set[train_set[f].isnull()]

    N, Y = miss["apply"].value_counts()

    print('Number of apply for missing %s: %d' % (f,Y))

    print('Number of not apply for missing %s : %d' %(f,N))

    notmiss = train_set[(train_set[f].isnull()!=True)]

    print (notmiss[f].describe())

valid_data.info()
# use 

X1 = valid_data[valid_data['title_proximity_tfidf'].isnull()== False]

X1_test = valid_data[valid_data['title_proximity_tfidf'].isnull()== True]



title_proximity_predict_result = gbm1.predict(X1_test[features].values)

X1_test.loc[:,'title_proximity_tfidf']= title_proximity_predict_result

X1 = X1.append(X1_test)



X2 = X1[X1['description_proximity_tfidf'].isnull()== False]

X2_test = X1[X1['description_proximity_tfidf'].isnull()== True]

description_proximity_predict_result = gbm2.predict(X2_test[features].values)

X2_test.loc[:,'description_proximity_tfidf']= description_proximity_predict_result

X2 = X2.append(X2_test)



X3 = X2[X2['city_match'].isnull()== False]

X3_test = X2[X2['city_match'].isnull()== True]

city_predict_result = city_match_classifier.predict(X3_test[features].values)

X3_test.loc[:,'city_match']= city_predict_result

X3 = X3.append(X3_test)



valid_data = X3
train_set_list = []

sampling_technique = ["undersampling","oversampling","original data"]
# perform undersampling on the dataset

rus = RandomUnderSampler(random_state=0)

X_resampled, y_resampled = rus.fit_resample(train_set[features].values, train_set["apply"])

y_resampled.shape

train_set_list.append([X_resampled,y_resampled])
# perform oversampling on the dataset

from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=0)

X_over_resampled, y_over_resampled = ros.fit_resample(train_set[features].values, train_set["apply"])


train_set_list.append([X_over_resampled,y_over_resampled])
X_train = train_set[features].values

y_train = train_set["apply"]



train_set_list.append([X_train,y_train])

X_test = test_set[features].values

y_test = test_set["apply"]
# Classifier try out



classifiers = [

    LogisticRegression(),

     KNeighborsClassifier(3),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    LinearDiscriminantAnalysis()

]



# Logging for Visual Comparison

log_cols=["Classifier", "Accuracy","AUC"]

log = pd.DataFrame(columns=log_cols)



index = 0

for X_train_set,y_train_set in train_set_list:

#     print (X_train_set.shape)''

    print('****Sampling technique ****')

    print (sampling_technique[index])

    index+=1

    for clf in classifiers:

        clf.fit(X_train_set,y_train_set)

#         clf.fit(X_train, y_train)

        name = clf.__class__.__name__



        print("="*30)

        print(name)

        print('****Results****')

        train_predictions = clf.predict(X_test)

        acc = accuracy_score(y_test, train_predictions)

        print("Accuracy: {:.4%}".format(acc))

        auc = roc_auc_score(y_test, train_predictions)

        print("auc: {:.4%}".format(auc))



        valid_predictions=clf.predict(valid_data[features])

        acc = accuracy_score(valid_data['apply'],valid_predictions)

        print("valid Accuracy: {:.4%}".format(acc))

        auc = roc_auc_score(valid_data['apply'],valid_predictions)

        print("valid auc: {:.4%}".format(auc))

    

#     train_predictions = clf.predict_proba(X_test)

#     ll = log_loss(y_test, train_predictions)

#     print("Log Loss: {}".format(ll))

    

        log_entry = pd.DataFrame([[name, acc, auc]], columns=log_cols)

        log = log.append(log_entry)

    sns.set_color_codes("muted")

    sns.barplot(x='AUC', y='Classifier', data=log, color="b")



    plt.xlabel('AUC %')

    plt.title('Classifier AUC')

    plt.show()



    sns.set_color_codes("muted")

    sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")



    plt.xlabel('Accuracy %')

    plt.title('Classifier Accuracy')

    plt.show()

    

print("="*30)