import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
plt.style.use('seaborn')
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data.head()
data.info()
salary = data.salary.dropna(inplace=True)
data.salary.head()
sns.boxplot(x=data.gender, y=data.salary, data=data);
salary_record = data.salary.groupby(data.gender)
print(salary_record.mean())
print(salary_record.std())
gender_salary_record = data.status.groupby(data.gender)
gender_salary_record.value_counts()
sns.countplot(data.status, hue=data.gender);
dept_salary_record = data.status.groupby([data.degree_t, data.gender])
dept_salary_record.value_counts()
sns.countplot(data.status, hue=data.degree_t);
salary_records = data.status.groupby([data.degree_t, data.gender, data.hsc_s])
salary_records.value_counts()
sns.boxplot(data=data, x='salary', y='ssc_p');
sns.swarmplot(data=data, x='status', y='degree_p', hue='gender');
sns.swarmplot(data=data, x='status', y='hsc_p', hue='gender');
sns.swarmplot(data=data, x='status', y='ssc_p', hue='gender');
sns.regplot(x='ssc_p', y='hsc_p', data=data);
import statsmodels.api
import statsmodels.formula.api as smf
data_copy = data.copy()
data_copy.ssc_p = data_copy.ssc_p.subtract(data_copy.ssc_p.mean())
data_copy.hsc_p = data_copy.hsc_p.subtract(data_copy.hsc_p.mean())
data_copy.degree_p = data_copy.degree_p.subtract(data_copy.degree_p.mean())
data_copy.etest_p = data_copy.etest_p.subtract(data_copy.etest_p.mean())
data_copy.mba_p = data_copy.mba_p.subtract(data_copy.mba_p.mean())
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_copy.status = le.fit_transform(data.status)
data_copy.status.value_counts()
sns.regplot(data=data_copy, x='ssc_p', y='status', label='SSC')
sns.regplot(data=data_copy, x='hsc_p', y='status', label='HSC')
sns.regplot(data=data_copy, x='degree_p', y='status', label='DEGREE')
sns.regplot(data=data_copy, x='mba_p', y='status', label='MBA')
plt.legend();
reg1 = smf.ols('status ~ ssc_p', data=data_copy).fit()
reg2 = smf.ols('status ~ hsc_p', data=data_copy).fit()
print(reg1.summary())
print(reg2.summary())
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.linear_model import LassoLarsCV
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
le = LabelEncoder()
data_copy.status = le.fit_transform(data.status)
data_copy.status.value_counts()
features = data_copy[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p']]
targets = data_copy.status

ftrain, ftest, ttrain, ttest = train_test_split(features, targets, train_size=0.4)
print(ftrain.shape)
print(ftest.shape)
print(ttrain.shape)
print(ttest.shape)
classifier = DecisionTreeClassifier()
classifier = classifier.fit(features, targets)
prediction = classifier.predict(ftest)

sklearn.metrics.accuracy_score(ttest, prediction)
ax = sns.distplot(ttest, kde=False, color='r', hist=True)
sns.distplot(prediction, kde=False, ax=ax, color='g', hist=True);
classifier = RandomForestClassifier(n_estimators=25)
classifier = classifier.fit(features, targets)
prediction = classifier.predict(ftest)

sklearn.metrics.accuracy_score(ttest, prediction)
extra = ExtraTreesClassifier()
extra.fit(ftrain, ttrain)

extra.feature_importances_
trees = range(50)
accuracy = np.zeros(50)

for item in trees:
    classifier = RandomForestClassifier(n_estimators = item+1)
    classifier = classifier.fit(features, targets)
    prediction = classifier.predict(ftest)
    accuracy[item] = sklearn.metrics.accuracy_score(ttest, prediction)

accuracy
plt.plot(trees, accuracy)
plt.ylabel('Accuracy Score')
plt.xlabel('Number of Trees');
features_data = features.copy()

features_data.ssc_p = preprocessing.scale(features_data.ssc_p.astype('float64'))
features_data.hsc_p = preprocessing.scale(features_data.hsc_p.astype('float64'))
features_data.degree_p = preprocessing.scale(features_data.degree_p.astype('float64'))
features_data.etest_p = preprocessing.scale(features_data.etest_p.astype('float64'))
features_data.mba_p = preprocessing.scale(features_data.mba_p.astype('float64'))
ftrain, ftest, ttrain, ttest = train_test_split(features_data, targets, train_size=0.4, random_state=123)
print(ftrain.shape)
print(ftest.shape)
print(ttrain.shape)
print(ttest.shape)
model = LassoLarsCV(cv=10, precompute=False).fit(ftrain, ttrain)
dict(zip(features_data.columns, model.coef_))
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca();
plt.plot(m_log_alphas, model.coef_path_.T);
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k', label = 'alpha CV');
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths');