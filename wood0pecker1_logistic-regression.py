import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
data=pd.read_csv('../input/bank-full.csv')
data.head()
data.dtypes
data.education.unique()
data.y.value_counts()
data.y.value_counts().plot(kind='bar')
data.groupby('y').mean()
data.shape
data.columns
data.groupby('job').mean()
data.groupby('education').mean()
%matplotlib inline
pd.crosstab(data.job,data.y).plot(kind='bar')
pd.crosstab(data.education,data.y).plot(kind='bar',stacked=True)
pd.crosstab(data.month,data.y).plot(kind='bar')
pd.crosstab(data.poutcome,data.y).plot(kind='bar')
cols=['job','marital','education','default','housing','loan','contact','month','poutcome','day']
for var in cols:
    dat=pd.get_dummies(data[var],prefix=var+'_')
    data=data.join(dat)
data=dates.copy()
data.columns.size
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
dataf=data[to_keep]
dataf.columns.size
datafcols=dataf.columns.values.tolist()
y=['y']
X=[i for i in datafcols if i not in y]
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.feature_selection import RFE
logreg=LogisticRegression()
rfe=RFE(logreg)
rfe=rfe.fit(dataf[X],data[y])
rfe.support_
rfe.ranking_
import statsmodels.api as sm
cols=dataf.columns.values.tolist()
colf=[i for i in cols if '_' in i]
colf.append('y')
dataff=dataf[cols]
X=dataff[colf[0:-1]]
y=data[colf[-1]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
pred=logreg.predict(X_test)
(pred==y_test).sum()
logreg.score(X_test,y_test)
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold=model_selection.KFold(n_splits=10,random_state=7)
lrg=LogisticRegression()
res=cross_val_score(lrg,X_train,y_train,cv=kfold,scoring='accuracy')
res.mean()