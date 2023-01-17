# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/bank-marketing-dataset"))

# Any results you write to the current directory are saved as output.
bank_data = pd.read_csv("../input/bank-marketing-dataset/bank.csv",delimiter=";")
bank_data.head(5)
bank_data.dtypes
bank_data.columns
bank_data.shape
bank_data.info()
dummy_bank = pd.get_dummies(bank_data, columns=['job','marital',
                                         'education','default',
                                         'housing','loan',
                                         'contact','month',
                                         'poutcome'])
dummy_bank.shape
dummy_bank_train = dummy_bank[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous','y',
       'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
       'job_management', 'job_retired', 'job_self-employed', 'job_services',
       'job_student', 'job_technician', 'job_unemployed', 'job_unknown',
       'marital_divorced', 'marital_married', 'marital_single',
       'education_primary', 'education_secondary', 'education_tertiary',
       'education_unknown', 'default_no', 'default_yes', 'housing_no',
       'housing_yes', 'loan_no', 'loan_yes', 'contact_unknown', 'month_may',
       'poutcome_unknown']]
dummy_bank_train.shape
dummy_bank_train.y.unique()
c = pd.value_counts(dummy_bank_train.y, sort = True).sort_index()
c
import matplotlib.pyplot as plt
c1 = c[0]/(c[0]+c[1] )#521/4000
c2  = c[1]/(c[0]+c[1])#1 - .1305
sizes = [c1,c2]
sizes
plot = plt.pie(sizes, labels = ['no','yes'],autopct='%1.1f%%',
        shadow=True, startangle=45 )
plt.axis('equal') 
plt.title("Class Imbalance Problem")
plt.show()
data_y = pd.DataFrame(dummy_bank_train['y'])
data_X = dummy_bank_train.drop(['y'], axis=1)
print(data_X.count())
print(data_y.count())
from sklearn.utils import shuffle
train_data= pd.concat([data_X,data_y], axis=1)
train_data.y.replace(('yes','no'),(1,0), inplace=True)
X_1 =train_data[ train_data["y"]==1 ]
X_0=train_data[train_data["y"]==0]
X_0=shuffle(X_0,random_state=42).reset_index(drop=True)
X_1=shuffle(X_1,random_state=42).reset_index(drop=True)

ALPHA=1.5

X_0=X_0.iloc[:round(len(X_1)*ALPHA),:]
data_final=pd.concat([X_1, X_0])
d = pd.value_counts(data_final['y'])
d
c1 = d[0]/(d[0]+d[1] )
c2  = d[1]/(d[0]+d[1])
sizes = [c1,c2]
plot = plt.pie(sizes, labels = ['no','yes'],autopct='%1.1f%%',
        shadow=True, startangle=45 )
plt.axis('equal') 
plt.title("Class Imbalance Problem")
plt.show()
data_final.head()
#dummy_bank.y.factorize()
data_y = pd.DataFrame(data_final['y'])
data_X = data_final.drop(['y'], axis=1)
print(data_X.columns)
print(data_y.columns)
test = pd.read_csv("../input/test-bank/test.xls")
test = test.drop(['Id'], axis=1)
test.head()
test.shape
test_dummy = pd.get_dummies(test, columns=['job','marital',
                                         'education','default',
                                         'housing','loan',
                                         'contact','month',
                                         'poutcome'])
test_dummy.columns
data_X.head()
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
lreg = logreg.fit(data_X,data_y)
lreg.score(data_X,data_y)
y_pred = logreg.predict(test_dummy)
y_pred.shape
from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save(outfile, y_pred)
df = pd.DataFrame(y_pred)
df.to_csv('pred.csv')
from sklearn import svm
from xgboost import XGBClassifier
clf = XGBClassifier()
clf
boost = clf.fit(data_X,data_y)
y_pred2 = clf.predict(test_dummy)
boost.score(data_X,data_y)
df = pd.DataFrame(y_pred2)
df.to_csv('pred2.csv')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 200)#criterion = entopy,gini
rfc.fit(data_X,data_y)
rfcpred = rfc.predict(test_dummy)
rfc.score(data_X,data_y)
df = pd.DataFrame(rfcpred)
df.to_csv('pred3.csv')
from sklearn.ensemble import GradientBoostingClassifier
gbk = GradientBoostingClassifier()
gbk.fit(data_X,data_y)
gbkpred = gbk.predict(test_dummy)
gbk.score(data_X,data_y)
df = pd.DataFrame(rfcpred)
df.to_csv('pred4.csv')
