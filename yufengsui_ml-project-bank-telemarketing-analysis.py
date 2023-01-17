import pandas as pd

import numpy as np

from pandas import read_csv

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

import os



linkname = '../input/bank-full.csv'

dataset1 = pd.read_csv(linkname, sep = ';')
# View the first 5 rows in the dataset

dataset1.head()
# Step 1: Delete the rows which column 'poutcome' contains 'other'

condition = dataset1.poutcome == 'other'

dataset2 = dataset1.drop(dataset1[condition].index, axis = 0, inplace = False)
# Step 2: Replace 'unknown' in job and education with 'other'

dataset2[['job','education']] = dataset2[['job','education']].replace(['unknown'],'other')
from scipy.stats import zscore



dataset2[['balance']].mean()

dataset2[['balance']].mean()



dataset2['balance_outliers'] = dataset2['balance']

dataset2['balance_outliers']= zscore(dataset2['balance_outliers'])



condition1 = (dataset2['balance_outliers']>3) | (dataset2['balance_outliers']<-3 )

dataset3 = dataset2.drop(dataset2[condition1].index, axis = 0, inplace = False)
dataset4 = dataset3.drop('balance_outliers', axis=1)
# Step 1: Change column name: 'y' to 'response'

dataset4.rename(index=str, columns={'y': 'response'}, inplace = True)



def convert(dataset4, new_column, old_column):

    dataset4[new_column] = dataset4[old_column].apply(lambda x: 0 if x == 'no' else 1)

    return dataset4[new_column].value_counts()



convert(dataset4, "response_binary", "response")
# Step 2: Drop column "contact" which is useless

dataset5 = dataset4.drop('contact', axis=1)
# Step 3: Change the unit of 'duration' from seconds to minutes

dataset5['duration'] = dataset5['duration'].apply(lambda n:n/60).round(2)
# Step 4: Change 'month' from words to numbers for easier analysis

lst = [dataset5]

for column in lst:

    column.loc[column["month"] == "jan", "month_int"] = 1

    column.loc[column["month"] == "feb", "month_int"] = 2

    column.loc[column["month"] == "mar", "month_int"] = 3

    column.loc[column["month"] == "apr", "month_int"] = 4

    column.loc[column["month"] == "may", "month_int"] = 5

    column.loc[column["month"] == "jun", "month_int"] = 6

    column.loc[column["month"] == "jul", "month_int"] = 7

    column.loc[column["month"] == "aug", "month_int"] = 8

    column.loc[column["month"] == "sep", "month_int"] = 9

    column.loc[column["month"] == "oct", "month_int"] = 10

    column.loc[column["month"] == "nov", "month_int"] = 11

    column.loc[column["month"] == "dec", "month_int"] = 12
# Step 1: Drop rows that 'duration' < 5s

condition2 = (dataset5['duration']<5/60)

dataset6 = dataset5.drop(dataset5[condition2].index, axis = 0, inplace = False)
# Step 2: Drop customer values with 'other' education

condition3 = (dataset6['education'] == 'other')

dataset7 = dataset6.drop(dataset6[condition3].index, axis = 0, inplace = False)
dist_age_balance = plt.figure(figsize = (10,6))



ra1 = dist_age_balance.add_subplot(1,2,1) 

ra2 = dist_age_balance.add_subplot(1,2,2)



ra1.hist(dataset7['age'])

ra1.set_title('The Distribution of Age')



ra2.hist(dataset7['balance'], color = 'skyblue')

ra2.set_title('The Distribution of Balance')



plt.tight_layout() 

plt.show()
scatter_age_balance = dataset7.plot.scatter('age','balance',figsize = (7,5))



plt.title('The Relationship between Age and Balance ')

plt.show()
dist_dur_cam = dataset7[['duration','campaign']].plot(kind = 'box', 

                                                      figsize = (8,8),

                                                      subplots = True, layout = (1,2),

                                                      sharex = False, sharey = False,

                                                      title='The Distribution of Duration and Campaign')

plt.show()
import seaborn as sns

dur_cam = sns.lmplot(x='duration', y='campaign',data = dataset7,

                     hue = 'response',

                     fit_reg = False,

                     scatter_kws={'alpha':0.6}, height =7)



plt.axis([0,65,0,65])

plt.ylabel('Number of Calls')

plt.xlabel('Duration of Calls (Minutes)')

plt.title('The Relationship between the Number and Duration of Calls (with Response Result)')



# Annotation

plt.axhline(y=5, linewidth=2, color="k", linestyle='--')

plt.annotate('Higher subscription rate when calls <5',xytext = (35,13),

             arrowprops=dict(color = 'k', width=1),xy=(30,6))

plt.show()
from pandas.plotting import scatter_matrix

matrix = scatter_matrix(dataset7[['age','balance','duration','campaign']],figsize=(10,8))



plt.suptitle('The Scatter Matrix of Age, Balance, Duration and Campaign')

plt.show()
corr_data = dataset7[['age','balance','duration','campaign','month_int','previous','response_binary']]

corr = corr_data.corr()



cor_plot = sns.heatmap(corr,annot=True,cmap='RdYlGn',linewidths=0.2,annot_kws={'size':10})

fig=plt.gcf()

fig.set_size_inches(6,5)

plt.xticks(fontsize=10,rotation=-30)

plt.yticks(fontsize=10)

plt.title('Correlation Matrix')

plt.show()
lst = [dataset7]

for column in lst:

    column.loc[column["age"] < 30,  'age_group'] = 20

    column.loc[(column["age"] >= 30) & (column["age"] <= 39), 'age_group'] = 30

    column.loc[(column["age"] >= 40) & (column["age"] <= 49), 'age_group'] = 40

    column.loc[(column["age"] >= 50) & (column["age"] <= 59), 'age_group'] = 50

    column.loc[column["age"] >= 60, 'age_group'] = 60
count_age_response_pct = pd.crosstab(dataset7['response'],dataset7['age_group']).apply(lambda x: x/x.sum() * 100)

count_age_response_pct = count_age_response_pct.transpose() 
age = pd.DataFrame(dataset7['age_group'].value_counts())

age['% Contacted'] = age['age_group']*100/age['age_group'].sum()

age['% Subscription'] = count_age_response_pct['yes']

age.drop('age_group',axis = 1,inplace = True)



age['age'] = [30,40,50,20,60]

age = age.sort_values('age',ascending = True)
plot_age = age[['% Subscription','% Contacted']].plot(kind = 'bar',

                                              figsize=(8,6), color = ('green','red'))

plt.xlabel('Age Group')

plt.ylabel('Subscription Rate')

plt.xticks(np.arange(5), ('<30', '30-39', '40-49', '50-59', '60+'),rotation = 'horizontal')

plt.title('Subscription vs. Contact Rate by Age')

plt.show()
lst = [dataset7]

for column in lst:

    column.loc[column["balance"] <= 0,  'balance_group'] = 'no balance'

    column.loc[(column["balance"] > 0) & (column["balance"] <= 1000), 'balance_group'] = 'low balance'

    column.loc[(column["balance"] > 1000) & (column["balance"] <= 5000), 'balance_group'] = 'average balance'

    column.loc[(column["balance"] > 5000), 'balance_group'] = 'high balance'
count_balance_response_pct = pd.crosstab(dataset7['response'],dataset7['balance_group']).apply(lambda x: x/x.sum() * 100)

count_balance_response_pct = count_balance_response_pct.transpose()
bal = pd.DataFrame(dataset7['balance_group'].value_counts())

bal['% Contacted'] = bal['balance_group']*100/bal['balance_group'].sum()

bal['% Subscription'] = count_balance_response_pct['yes']

bal.drop('balance_group',axis = 1,inplace = True)



bal['bal'] = [1,2,0,3]

bal = bal.sort_values('bal',ascending = True)
plot_balance = bal[['% Subscription','% Contacted']].plot(kind = 'bar',

                                               color = ('royalblue','skyblue'),

                                               figsize = (8,6))



plt.title('Subscription vs Contact Rate by Balance Level')

plt.ylabel('Subscription Rate')

plt.xlabel('Balance Category')

plt.xticks(rotation = 'horizontal')



# label the bar

for rec, label in zip(plot_balance.patches,

                      bal['% Subscription'].round(1).astype(str)):

    plot_balance.text(rec.get_x() + rec.get_width()/2, 

                      rec.get_height() + 1, 

                      label+'%',  

                      ha = 'center', 

                      color = 'black')
age_balance1 = pd.DataFrame(dataset7.groupby(['age_group','balance_group'])['response_binary'].sum())

age_balance2 = pd.DataFrame(dataset7.groupby(['age_group','balance_group'])['response'].count())



age_balance1['response'] = age_balance2['response']

age_balance1['response_rate'] = age_balance1['response_binary']/ (age_balance1['response'])

age_balance1 = age_balance1.drop(['response_binary','response'],axis =1)



age_balance1 = age_balance1.unstack()
age_bal = age_balance1.plot(kind='bar',figsize = (10,6))



# Set x ticks

plt.xticks(np.arange(5),('<30', '30-39', '40-49', '50-59', '60+'),rotation = 'horizontal')



# Set legend

plt.legend(['Average Balance','High Balance','Low Balance','No Balance'],loc = 'best',ncol = 1)



plt.ylabel('Subscription Rate')

plt.xlabel('Age Group')

plt.title('The Subscription Rate of Different Balance Levels in Each Age Group')

plt.show()
count_job_response_pct = pd.crosstab(dataset7['response'],dataset7['job']).apply(lambda x: x/x.sum() * 100)

count_job_response_pct = count_job_response_pct.transpose()
plot_job = count_job_response_pct['yes'].sort_values(ascending = True).plot(kind ='barh',

                                                                           figsize = (12,6))

                                                                               

plt.title('Subscription Rate by Job')

plt.xlabel('Subscription Rate')

plt.ylabel('Job Category')



# Label each bar

for rec, label in zip(plot_job.patches,

                      count_job_response_pct['yes'].sort_values(ascending = True).round(1).astype(str)):

    plot_job.text(rec.get_width()+0.8, 

                  rec.get_y()+ rec.get_height()-0.5, 

                  label+'%', 

                  ha = 'center', 

                  va='bottom')
count_month_response_pct = pd.crosstab(dataset7['response'],dataset7['month_int']).apply(lambda x: x/x.sum() * 100)

count_month_response_pct = count_month_response_pct.transpose()
month = pd.DataFrame(dataset7['month_int'].value_counts())

month['% Contacted'] = month['month_int']*100/month['month_int'].sum()

month['% Subscription'] = count_month_response_pct['yes']

month.drop('month_int',axis = 1,inplace = True)



month['Month'] = [5,7,8,6,11,4,2,1,10,9,3,12]

month = month.sort_values('Month',ascending = True)
plot_month = month[['% Subscription','% Contacted']].plot(kind ='line',

                                                          figsize = (10,6),

                                                          marker = 'o')



plt.title('Subscription vs. Contact Rate by Month')

plt.ylabel('Subscription and Contact Rate')

plt.xlabel('Month')



ticks = np.arange(1,13,1)

plt.xticks(ticks)



# Annotation: peak of contact

y = month['% Contacted'].max()

x = month['% Contacted'].idxmax()

plt.annotate('May: Peak of contact', xy=(x+0.1, y+0.1), xytext=(x+1,y+4), arrowprops=dict(facecolor='black', headwidth=6, width=1, headlength=4), horizontalalignment='left', verticalalignment='top')



# Annotation: peak of subscription rate

y = month['% Subscription'].max()

x = month['% Subscription'].idxmax()

plt.annotate('March: Peak Subscription rate', xy=(x+0.1, y+0.1), xytext=(x+1,y+1), arrowprops=dict(facecolor='black', headwidth=6, width=1, headlength=4), horizontalalignment='left', verticalalignment='top')



plt.show()
linkname = '../input/bank_cleaned.csv'

dataset = read_csv(linkname)

dataset = dataset.drop(['Unnamed: 0'], axis=1)
# View the first 5 rows of cleaned data

dataset.head()
dataset.drop(['marital'],axis=1, inplace=True)

dataset1 = dataset.iloc[:, 0:7]
dataset2 = pd.get_dummies(dataset1, columns = ['job'])

dataset2 = pd.get_dummies(dataset2, columns = ['education'])

dataset2['housing'] = dataset2['housing'].map({'yes': 1, 'no': 0})

dataset2['default'] = dataset2['default'].map({'yes': 1, 'no': 0})

dataset2['loan'] = dataset2['loan'].map({'yes': 1, 'no': 0})

dataset_response = pd.DataFrame(dataset['response_binary'])

dataset2 = pd.merge(dataset2, dataset_response, left_index = True, right_index = True)
array = dataset2.values



# Features: first 20 columns

X = array[:,0:-1]



# Target variable: 'response_binary'

Y = array[:,-1]
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
# 20% of the data will be used for testing

test_size= 0.20

seed = 7

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=test_size, random_state=seed)
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB
models = []

models.append(('LR', LogisticRegression()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))
results_c = []

names_c = []



for name, model in models:

    # define how to split off validation data ('kfold' how many folds)

    kfold = KFold(n_splits=10, random_state=seed)    

    # train the model

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')    

    results_c.append(cv_results)

    names_c.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
fig = plt.figure()

fig.suptitle('Perfomance of Classification Algorithms')

ax = fig.add_subplot(111)

plt.boxplot(results_c)

ax.set_xticklabels(names_c)

plt.show()
LR = LogisticRegression()

LR.fit(X_train, Y_train)
predictions = LR.predict(X_test)
# Accuracy Score 

print(accuracy_score(Y_test, predictions))
# Confusion Matrix 

from sklearn.metrics import confusion_matrix

import pylab as pl



print(confusion_matrix(Y_test, predictions))



cm = confusion_matrix(Y_test, predictions)

pl.matshow(cm)

pl.title('Confusion matrix of the classifier')

pl.colorbar()

pl.show()
# Classification Report

from sklearn.metrics import classification_report

print(classification_report(Y_test, predictions))
dataset4 = dataset2.drop(['response_binary'],axis = 1)

dataset4['duration'] = dataset['duration']
array = dataset4.values

X = array[:,0:20]

Y = array[:,20]
test_size= 0.20

seed = 10

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=test_size, random_state=seed)
from sklearn.linear_model import LinearRegression

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
models = []

models.append(('LR', LinearRegression()))

models.append(('LASSO', Lasso()))

models.append(('RIDGE', Ridge()))

models.append(('EN', ElasticNet()))

models.append(('KNN', KNeighborsRegressor()))

models.append(('CART', DecisionTreeRegressor()))
results_e2 = []

names_e2 = []



for name, model in models:

    # define how to split off validation data

    kfold = KFold(n_splits=10, random_state=seed)

    

    # train the model

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')

    results_e2.append(cv_results)

    names_e2.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Plot results

fig = plt.figure(figsize=(9,6))

ax = fig.add_subplot(111)

plt.boxplot(results_e2)



fig.suptitle('Algorithm Comparison')

ax.set_xticklabels(names_e2)

plt.show()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



pipelines = []

pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LinearRegression())])))

pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO', Lasso())])))

pipelines.append(('ScaledRIDGE', Pipeline([('Scaler', StandardScaler()),('RIDGE', Ridge())])))

pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN', ElasticNet())])))

pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))

pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeRegressor())])))
results_e2_normalized = []

names_e2_normalized = []



for name, model in pipelines:

    # define how to split off validation data

    kfold = KFold(n_splits=10, random_state=seed)

    

    # train the model

    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_squared_error')

    results_e2_normalized.append(cv_results)

    names_e2_normalized.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Plot results

fig = plt.figure(figsize=(9,6))

ax = fig.add_subplot(111)

plt.boxplot(results_e2_normalized)



fig.suptitle('Algorithm Comparison')

ax.set_xticklabels(names_e2_normalized)

plt.show()
ridge = Ridge()

ridge.fit(X_train, Y_train)

predicted_y = ridge.predict(X_test)
from sklearn.metrics import mean_squared_error

print('The MSE is', mean_squared_error(Y_test, predicted_y))