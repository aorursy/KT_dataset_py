# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')
df
X=df.drop('price_range',axis=1)
y=df['price_range']
df.isna().sum()
df['blue'].value_counts().sort_index().plot(kind='bar',rot=0)
plt.xlabel('Bluetooth')
plt.ylabel('Count')
df.columns

len(df['ram'].unique())
df['fc']=np.where(df['fc']==0,df['fc'].median(),df['fc'])
df['fc'].unique()
fig,ax=plt.subplots(figsize=(10,8))
sns.countplot(df['fc'],hue=df['price_range'])
plt.xlabel('Front Camera')
plt.ylabel('Count')
df['mobile_wt'].value_counts().sort_index().plot(kind='barh',figsize=(15,25))
sns.distplot(df['ram'])
cor=df.loc[:,['four_g','three_g']].corr()
cor
count=df['price_range'].value_counts()
count.plot(kind='bar',rot=0)
plt.xlabel('Price_range')
plt.ylabel('Count')
len(df['battery_power'].unique())
scorefun=SelectKBest(score_func=chi2)
fit=scorefun.fit(X,y)
dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(X.columns)
featureselect=pd.concat([dfscores,dfcolumns],axis=1)
featureselect.columns=["score","features"]
featureselect
print(featureselect.nlargest(8,'score'))
X=df.loc[:,['ram','px_height','battery_power','px_width','mobile_wt','int_memory']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
model=RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
from sklearn.model_selection import cross_val_score
scores=cross_val_score(model,X_train,y_train,cv=10)
scores
scores.mean()
