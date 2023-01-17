import string
import re, math, os, sys
import sklearn
import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd
pd.options.display.max_rows=100000
pd.options.display.max_columns=100000
pd.options.display.float_format = '{:.2f}'.format
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
plt.gcf().subplots_adjust(top=0.5, bottom=0.4)
from IPython.display import display
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
%matplotlib inline
ksData=pd.read_csv("../input/ks-projects-201801.csv", sep=",")
ksData.describe()
ksData.head()
categoryCount = pd.DataFrame(ksData.groupby('category').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='category', y="counts", data=categoryCount.head(15))
fig.axis(ymin=0, ymax=25000)
plt.xticks(rotation=75)
categoryCount = pd.DataFrame(ksData.groupby('main_category').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='main_category', y="counts", data=categoryCount.head(15))
fig.axis(ymin=0, ymax=65000)
plt.xticks(rotation=75);
FilmsProjects=ksData.loc[ksData['main_category']=='Film & Video',:]
categoryCount = pd.DataFrame(FilmsProjects.groupby('category').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='category', y="counts", data=categoryCount.head(15))
fig.axis(ymin=0, ymax=19000)
plt.xticks(rotation=75);
categoryCount = pd.DataFrame(ksData.groupby('country').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='country', y="counts", data=categoryCount.head(15))
fig.axis(ymin=0, ymax=300000)
plt.xticks(rotation=75);
ksData['state'].unique()
ksData.groupby('state').size().sort_values(ascending=False).rename('counts').reset_index()
percentage=(ksData.groupby('state').size()/(ksData.shape[0]))
relativePart=percentage.values*100
t=np.char.mod('%.2f', relativePart)
labels=percentage.index+" - "+t+"%"
explode=(0,0,0,0.5,0,0)
matplotlib.pyplot.axis("equal")
patches, texts =plt.pie(percentage, explode=explode, shadow=True, startangle=90, radius=2)
plt.legend(patches, labels, bbox_to_anchor=(-0.1, 1.),
           fontsize=8)
categoryCount = pd.DataFrame(ksData.groupby('usd_goal_real').size().sort_values(ascending=False).rename('counts').reset_index())
f, ax = plt.subplots(figsize=(10, 5))
fig = sns.barplot(x='usd_goal_real', y="counts", data=categoryCount.head(50))
fig.axis(ymin=0, ymax=35000)
plt.xticks(rotation=75);
success=ksData.loc[ksData['state']=='successful']
failed=ksData.loc[ksData['state']=='failed']
boxPlotData=[success['usd_goal_real'], failed['usd_goal_real']]
plt.boxplot(boxPlotData, labels=['success', 'failed'])
ksData['percentage']=(ksData['usd_pledged_real']/ksData['usd_goal_real']);
s=ksData.loc[:,'percentage']
plt.hist(s, bins=100, range=(0,1))
ksData['deadline_new']=pd.to_datetime(ksData['deadline'], dayfirst=True)
ksData['launched_new']=pd.to_datetime(ksData['launched'], dayfirst=True)
ksData['duration']=ksData['deadline_new']-ksData['launched_new']
ksData['duration']=ksData['duration'].dt.days # only days
ksData['duration'].describe()
ksData['duration'].median()
plt.boxplot(ksData['duration'], labels=['duration'])
ksData[ksData['duration'] > 100]
ksData.drop(ksData[ksData['duration'] > 100].index, inplace=True)
plt.boxplot(ksData['duration'], labels=['duration'])
ksDataSF=ksData.loc[(ksData['state']=='successful') | (ksData['state']=='failed'), :]
keep=ksDataSF.columns.drop(['ID', 'name', 'deadline', 'goal', 'launched','pledged','usd pledged','deadline_new', 'launched_new'])
ksDataSF=ksDataSF[keep]
target='state'
ksDataSF=pd.get_dummies(ksDataSF, drop_first=True, columns=['category', 'main_category', 'currency', 'country'])
# ksDataSF.head()
le=sklearn.preprocessing.LabelEncoder()
ksDataSF['state']=le.fit_transform(ksDataSF['state'])
features=ksDataSF.columns.drop('state')
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
le_name_mapping
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(ksDataSF[features], ksDataSF[target], train_size=0.7)
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_train,y_train)
pred = logistic.predict(X_test)
print("***********")
print("accuracy_score:", sklearn.metrics.accuracy_score(y_test, pred, normalize=True))
print("classification_report:")
print(sklearn.metrics.classification_report(y_test, pred))
print("confusion_matrix:")
print(sklearn.metrics.confusion_matrix(y_test, pred))
Coef=(logistic.coef_).tolist()[0]
featuresList=features.tolist()
zipped=zip(Coef,featuresList)
list(zipped)
features=features.drop(['backers','usd_pledged_real', 'percentage'])
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(ksDataSF[features], ksDataSF[target], train_size=0.7)
logistic = linear_model.LogisticRegression(C=1e5)
logistic.fit(X_train,y_train)
pred = logistic.predict(X_test)
print("***********")
print("accuracy_score:", sklearn.metrics.accuracy_score(y_test, pred, normalize=True))
print("classification_report:")
print(sklearn.metrics.classification_report(y_test, pred))
print("confusion_matrix:")
print(sklearn.metrics.confusion_matrix(y_test, pred))

Coef=(logistic.coef_).tolist()[0]
featuresList=features.tolist()
zipped=zip(Coef,featuresList)
list(zipped)