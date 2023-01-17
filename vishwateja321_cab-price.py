# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True,style='darkgrid')
file = '/kaggle/input/sigmacabprediction/SigmaCab-Train.csv'
df = pd.read_csv(file)
df.head()
df = df.drop(['Trip_ID'],axis=1)
df.describe(include='all').transpose()
df.shape
df.info()
df.isnull().sum()
df.hist(figsize=(15,10))
sns.countplot(df['Surge_Pricing_Type'],hue=df['Gender'])
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
sns.countplot(df['Type_of_Cab'])

plt.subplot(1,2,2)
sns.countplot(df['Type_of_Cab'],hue=df['Surge_Pricing_Type'])
sns.countplot(df['Destination_Type'])
sns.boxplot(df['Life_Style_Index'])
sns.pointplot(df['Surge_Pricing_Type'],df['Cancellation_Last_1Month'])
sns.boxplot(df['Surge_Pricing_Type'],df['Customer_Rating'])
sns.boxplot(df['Surge_Pricing_Type'],df['Var3'])
df.groupby(['Destination_Type'])['Surge_Pricing_Type'].value_counts()
df.groupby(['Destination_Type'])['Surge_Pricing_Type'].value_counts().plot()
df['Destination_Type'].value_counts()
sns.boxplot(df['Surge_Pricing_Type'],df['Life_Style_Index'])
sns.countplot(df['Customer_Since_Months'],hue=df['Surge_Pricing_Type'])
sns.boxplot(df['Surge_Pricing_Type'],df['Trip_Distance'])
sns.pointplot(df['Surge_Pricing_Type'],df['Trip_Distance'])
df.groupby(['Destination_Type'])['Trip_Distance'].mean().sort_values(ascending=False).plot()
plt.figure(figsize=(15,8))
sns.boxplot(df['Destination_Type'],df['Trip_Distance'])
df.groupby(['Type_of_Cab','Surge_Pricing_Type'])['Surge_Pricing_Type'].count()
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputed_Toc = imputer.fit_transform(np.array(df['Type_of_Cab']).reshape(-1,1))
Toc = pd.DataFrame(imputed_Toc,columns=['Type_of_Cab'])
Toc.head()
Toc.isna().any()
df['imputed_Toc'] = df['Type_of_Cab'].isna()
df['imputed_Toc'] = df['imputed_Toc'].replace([True,False],[1,0])
df.head()
df = df.drop(['Type_of_Cab'],axis=1)
df = pd.concat([df,Toc],axis=1)
df.head()
df['Customer_Since_Months'].value_counts()
df['imputed_Csm'] = df['Customer_Since_Months'].isna()
df['imputed_Csm'] = df['imputed_Csm'].replace([True,False],[1,0])
df['Customer_Since_Months'] = df['Customer_Since_Months'].fillna(0)
df['Customer_Since_Months'].value_counts()
df.isna().sum()
df.head()
df.info()
df['Destination_Type'].value_counts()
DT = {'A':14,'B':13,'C':12,'D':11,'E':10,'F':9,'G':8,'H':7,'I':6,'J':5,'K':4,'L':3,'N':2,'M':1}
df['Destination_Type'] = df['Destination_Type'].map(DT)
df.head()
df.info()
T = pd.get_dummies(df['Type_of_Cab'],drop_first=True,prefix='Toc')
T
df = pd.concat([df,T],axis=1)
G = pd.get_dummies(df['Gender'],drop_first=True,prefix='G')
df = pd.concat([df,G],axis=1)
df = df.drop(['Type_of_Cab','Gender'],axis=1)
df['Confidence_Life_Style_Index'] = df['Confidence_Life_Style_Index'].replace(np.nan,'mis')
C = pd.get_dummies(df['Confidence_Life_Style_Index'],drop_first=True,prefix='CLSI')
df = pd.concat([df,C],axis=1)
df = df.drop(['Confidence_Life_Style_Index'],axis=1)
df.head()
cols = df.columns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
imp = IterativeImputer(max_iter=10,random_state=0)
df = imp.fit_transform(df)
df = pd.DataFrame(df,columns=cols)
df.head()
df.isnull().sum()
corr = df.corr()
plt.figure(figsize=(20,10))
sns.heatmap(corr,annot=True)
df.describe().transpose()
X = df.drop(['Surge_Pricing_Type'],axis=1)
y = df['Surge_Pricing_Type']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
x_train.shape
y_test.shape
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xs_train = scaler.fit_transform(x_train)
xs_test = scaler.fit_transform(x_test)
from sklearn.linear_model import LogisticRegression
log = LogisticRegression(random_state=1)
model1 = log.fit(xs_train,y_train)
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
y_pred1 = model1.predict(xs_test)
accuracy_score(y_test,y_pred1)
f1_score(y_test,y_pred1,average='macro')
from sklearn.svm import LinearSVC,svc
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(max_depth=15,ccp_alpha=0.0)
model3 = dtc.fit(x_train,y_train)
y_pred3 = model3.predict(x_test)
accuracy_score(y_test,y_pred3)
f1_score(y_test,y_pred3,average='macro')
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
model4 = rfc.fit(x_train,y_train)
y_pred4 = model4.predict(x_test)
accuracy_score(y_test,y_pred4)
f1_score(y_test,y_pred4,average='macro')
from sklearn.ensemble import AdaBoostClassifier

abcl = AdaBoostClassifier(n_estimators=300,learning_rate=0.3)
model5 = abcl.fit(x_train,y_train)
y_pred5 = model5.predict(x_test)
accuracy_score(y_test,y_pred5)
f1_score(y_test,y_pred5,average='macro')
from sklearn.ensemble import GradientBoostingClassifier
gbcl  = GradientBoostingClassifier()
model6 = gbcl.fit(x_train,y_train)
y_pred6 = model6.predict(x_test)
accuracy_score(y_test,y_pred6)
