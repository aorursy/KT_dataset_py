# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
import seaborn as sns
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
%matplotlib inline
rd= pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
rd.head(10)
rd.columns
from IPython.core.display import HTML # permet d'afficher du code html dans jupyter
display(HTML(rd.head(10).to_html()))
rd.shape
rd.count()
rd.describe()
rd.quality.value_counts()
rd['quality'].value_counts().head(10).plot.bar()
d = rd[rd.quality.isin(rd.quality.value_counts().head(5).index)]

sns.boxplot(
    x='quality',
    y='pH',
    data=d
)
d = rd[rd.quality.isin(rd.quality.value_counts().head(5).index)]

sns.boxplot(
    x='quality',
    y='alcohol',
    data=d
)
rd['pH'].value_counts().head(15).sort_index().plot.bar()
sns.kdeplot(rd.query('pH < 4').pH)
fig = sns.FacetGrid(rd, hue="quality", aspect=3, palette="Set2") # aspect=3 permet d'allonger le graphique
fig.map(sns.kdeplot, "pH", shade=True)
fig.add_legend()
rd.columns
sns.distplot(rd.pH, color='blue')
rd['log_pH'] = np.log(rd.pH+1)
rd.describe()
sns.kdeplot(rd.log_pH, color='blue')
bins = (2, 6, 8)
group_names = ['bad', 'good']
rd['quality'] = pd.cut(rd['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
rd['quality'] = label_quality.fit_transform(rd['quality'])
rd['quality'].value_counts()
rd.head(10)
sns.countplot(rd['quality'])
X = rd.drop('quality', axis = 1)
y = rd['quality']
from sklearn import preprocessing
minmax = preprocessing.MinMaxScaler(feature_range=(0, 1))
rd[['alcohol', 'log_pH']] = minmax.fit_transform(rd[['alcohol', 'log_pH']])
sns.distplot(rd.log_pH, color='blue')
sns.distplot(rd.alcohol, color='red')
scaler = preprocessing.StandardScaler()
rd[['alcohol', 'log_pH']] = scaler.fit_transform(rd[['alcohol', 'log_pH']])
sns.distplot(rd.log_pH, color='blue')
sns.distplot(rd.alcohol, color='red')
rd.info()
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape)
print(X_test.shape)
ps = preprocessing.StandardScaler()
X_train = ps.fit_transform(X_train)
X_test = ps.fit_transform(X_test)
# Importation des méthodes de mesure de performances
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_lr = lr.predict(X_test)
print(confusion_matrix(y_test,y_lr))
print(accuracy_score(y_test,y_lr))
print(classification_report(y_test, y_lr))
probas = lr.predict_proba(X_test)
print(probas)
dfprobas = pd.DataFrame(probas,columns=['proba_0','proba_1'])
dfprobas['y'] = np.array(y_test)
dfprobas
plt.figure(figsize=(10,10))
sns.distplot(1-dfprobas.proba_0[dfprobas.y==0], bins=50)
sns.distplot(dfprobas.proba_1[dfprobas.y==1], bins=50)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])
roc_auc = auc(false_positive_rate, true_positive_rate)
print (roc_auc)
plt.figure(figsize=(12,12))
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.plot([0,0,1],[0,1,1],'g:') 
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
from sklearn import ensemble
rf = ensemble.RandomForestClassifier()
rf.fit(X_train, y_train)
y_rf = rf.predict(X_test)
print(classification_report(y_test, y_rf))
cm = confusion_matrix(y_test, y_rf)
print(cm)
rf1 = ensemble.RandomForestClassifier(n_estimators=10, min_samples_leaf=10, max_features=3)
rf1.fit(X_train, y_train)
y_rf1 = rf.predict(X_test)
print(classification_report(y_test, y_rf1))
from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)
print(classification_report(y_test, pred_sgd))
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
pred_svc = svc.predict(X_test)
print(classification_report(y_test, pred_svc))
#Finding best parameters for our SVC model
param = {
    'C': [0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear', 'rbf'],
    'gamma' :[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4]
}
grid_svc = GridSearchCV(svc, param_grid=param, scoring='accuracy', cv=10)
grid_svc.fit(X_train, y_train)
{'C': 1.2, 'gamma': 0.9, 'kernel': 'rbf'}
svc2 = SVC(C = 1.2, gamma =  0.9, kernel= 'rbf')
svc2.fit(X_train, y_train)
pred_svc2 = svc2.predict(X_test)
print(classification_report(y_test, pred_svc2))
from sklearn import model_selection
param_grid = {
              'n_estimators': [10, 100, 500],
              'min_samples_leaf': [1, 20, 50]
             }
estimator = ensemble.RandomForestClassifier()
rf_gs = model_selection.GridSearchCV(estimator, param_grid)
rf_gs.fit(X_train, y_train)
print(rf_gs.best_params_)
rf2 = rf_gs.best_estimator_
y_rf2 = rf2.predict(X_test)
print(classification_report(y_test, y_rf2))
!pip install xgboost
import xgboost as XGB
xgb  = XGB.XGBClassifier()
xgb.fit(X_train, y_train)
y_xgb = xgb.predict(X_test)
cm = confusion_matrix(y_test, y_xgb)
print(cm)
print(classification_report(y_test, y_xgb))