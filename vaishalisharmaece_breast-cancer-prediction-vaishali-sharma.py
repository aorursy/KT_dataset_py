

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head(5)
df.drop('Unnamed: 32',axis=1,inplace=True)
df.info()
## Target variable

df.diagnosis.value_counts()
#a. boxplot

for i in df.drop('diagnosis',axis=1).columns:

    sns.boxplot(df.diagnosis,df[i])

    plt.show()
#b. histogram: not possible as target categorical dtype

df.diagnosis.value_counts().plot.bar()
#e Unique Values accross all columns :

#   all values are numberic.
# f Duplicate values

df[df.duplicated()]
sns.heatmap(df.corr())
for i in df.drop('diagnosis',axis=1).columns:

    sns.barplot(df.diagnosis,df[i])

    plt.show()
### j Pairplot : execute it didn't executed due to pc overload

sns.pairplot(df)
# 2. drop all duplicated

df[df.duplicated()]

# no duplicated rows to drop
# 3. Drop all non-essential features

useless_cols = []

for i in df.drop('diagnosis',axis=1).columns:

    j = len(df[i].value_counts())

    if j == 1:

        print(i,'has value count',j)

        useless_cols.append(i)
df.drop(useless_cols,axis=1,inplace=True)

df.head(1)
df.head()
# 6: Clean data : Not required all data already cleaned

df.info()
sns.pairplot(df.drop('id',axis=1),hue='diagnosis')
#8 Classification ML

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score, KFold

from sklearn import metrics

from sklearn.metrics import roc_auc_score
df.diagnosis = df.diagnosis.map({'M':0,'B':1})
from sklearn.ensemble import RandomForestClassifier



X = df.drop('diagnosis',axis=1)

y = df[['diagnosis']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=42)

model = RandomForestClassifier(n_estimators=100,

                               random_state=101,

                               class_weight='balanced')
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

print('AUC-ROC Score :',roc_auc_score(y_test, y_pred))

print('F1-Score :',f1_score(y_test,y_pred))

print('Report:\n',classification_report(y_test, y_pred))

print('confusion Matrix:\n',confusion_matrix(y_pred,y_test))
print('cross validation:',cross_val_score(model, X, y, cv=5).mean())
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import GradientBoostingClassifier
# 9

models = {'RandomForestClassifier':RandomForestClassifier(),

          'LogisitcRegression':LogisticRegression(),

          'GradientBoosingClassifier':GradientBoostingClassifier(),

          'XGBoost':XGBClassifier()}

mn = []

auc = []

f1 = []

acc = []

for model_name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print('\nF1 Score For', model_name,'>>',f1_score(y_test,pred))

    print('cross validation for',model_name,'::>',cross_val_score(model, X, y, cv=5).mean())
# we take XGBoost model as it has best score
model
### we pick XGboost as it has best f1score
plt.figure(figsize=(25,25))

sns.heatmap(df.corr(),annot=True)
independent_feature = ['concavity_worst','texture_se'] # from above heatmap