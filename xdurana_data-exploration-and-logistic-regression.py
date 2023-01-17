import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.metrics import confusion_matrix
data = pd.read_csv('../input/data.csv')
data.columns
data[data['id'].duplicated()].id
data.drop('id', axis=1, inplace=True)

data.drop('Unnamed: 32', axis=1, inplace=True)
sns.countplot(data['diagnosis'])
data['diagnosis'] = data['diagnosis'].map({'M' : 1, 'B' : 0})
train, test = train_test_split(data, test_size = 0.3)
k = 20

corrmat = train.corr()

cols = corrmat.nlargest(k, 'diagnosis')['diagnosis'].index

cm = np.corrcoef(train[cols].values.T)

sns.set(font_scale=1.5)

hm = sns.clustermap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.figure(figsize=(30,30))

plt.setp(hm.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)

plt.show()
features = [

    'diagnosis',

    'perimeter_se',

    'perimeter_worst',

    'concavity_worst',

    'concave points_worst',

    'texture_worst',

    'smoothness_worst',

    'symmetry_worst'

]
sns.set()

sns.pairplot(train[features], size = 2.5)

plt.show();
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
sns.boxplot(train['perimeter_se'])
train = train[train['perimeter_se'] < 15]
sns.distplot(train['perimeter_se'])

fig = plt.figure()

res = stats.probplot(train['perimeter_se'], plot=plt)
predictors = features[2:]



train_X = train[predictors]

train_y = train.diagnosis



test_X = test[predictors]

test_y = test.diagnosis
model = LogisticRegression()

model.fit(train_X, train_y)
predictions = model.predict(test[predictors])

accuracy = metrics.accuracy_score(predictions, test['diagnosis'])

print("Accuracy : %s" % "{0:.3%}".format(accuracy))
metrics.roc_auc_score(predictions, test['diagnosis'])
confusion_matrix(test['diagnosis'], predictions)