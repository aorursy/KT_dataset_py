
import os

import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

import numpy as np
import seaborn as sns
import re
%pylab inline

df=pd.read_csv('../input/flavors_of_cacao.csv')
dfalt=df # make one copy of daaframe 

print(df.isnull().any())
print(df.dtypes)
print(df.describe())
print(df.info())
print(df.head())
print(df.columns)
df[['Bean\nType', 'Broad Bean\nOrigin']].replace(r'\s+', np.nan, regex=True, inplace = True)
#Preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder 
le=LabelEncoder()
ohe=OneHotEncoder()
df['Companymake']=le.fit_transform(df['Company\xa0\n(Maker-if known)'])
df['Barname']=le.fit_transform(df['Specific Bean Origin\nor Bar Name'])
df['Companylocationencoded']=le.fit_transform(df['Company\nLocation'])

print(df.columns)
df=df[[ 'Companymake', 'Barname','Companylocationencoded','REF', 'Review\nDate', 'Cocoa\nPercent', 'Rating']]
print(df.info())

a=list(df['Cocoa\nPercent'])
b=[]
for i in a:
    i=i[:2]
    b.append(i)
print(b)
b1=Series(b)
df['Cocoa_in_Percent']=b1
df['Cocoa_in_Percent'].astype(int)
df['Country']=dfalt['Company\nLocation']
df=df[['Companymake', 'Barname', 'Companylocationencoded', 'REF',
       'Review\nDate','Cocoa_in_Percent' , 'Rating' ,'Country']]
print(df.corr())
sns.heatmap(df.corr())
plt.figure(figsize=(14,12))
df[['Country','Rating']].groupby('Country').mean().round(1).plot(kind='line')
df[['Country','Rating']].groupby('Country').mean().round(1)[:10]
g = sns.jointplot(df['Barname'][:10],df['Rating'][:10] , kind="kde", size=7, space=0)

plt.figure(figsize=(12,10))
y=df['Cocoa_in_Percent']
x=df['Rating']
df['Cocoa_in_Percent']=y.astype(str).astype(int)
df['Cocoa_in_Percent'].values
y=df['Cocoa_in_Percent']
sns.barplot(x, y, palette="BuGn_d")

sns.barplot(x, df['REF'], palette="BuGn_d")
sns.regplot(y=df['Rating'],x=df['Barname'], data=df);
sns.regplot(y=df['Rating'],x=df['REF'], data=df);
sns.regplot(y=df['Rating'],x=df['Companymake'], data=df);
import seaborn as sns
plt.figure(figsize=(12,10))
sns.set(style="darkgrid", color_codes=True)
l=df['Cocoa_in_Percent']
o=df['REF']
p=df['Barname']
q=df['Review\nDate']
r=df['Rating']

g = sns.jointplot(l, r, data=df, kind="reg",
                  xlim=(0, 60), ylim=(0, 12), color="r", size=7)
sns.regplot(y=df['Rating'],x=df['Cocoa_in_Percent'], data=df);
df = df[df.loc[:,'Cocoa_in_Percent'] < 90]
df = df[df.loc[:,'Cocoa_in_Percent'] > 50]
sns.regplot(x=df['Rating'],y=df['Cocoa_in_Percent'], data=df);
sns.regplot(y=df['Rating'],x=df['Cocoa_in_Percent'], data=df,
           lowess=True);
rating=DataFrame(df['Rating'])
ratingalt=rating.replace([1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0,
              5.0],['A','B','C','D','E','F','G','H','I','J','K','L','M'],inplace=True)
ratingalt=Series(ratingalt)
df['Ratingcategory']=rating
rating

X=df[['Companymake', 'Barname', 'Companylocationencoded', 'REF',
       'Review\nDate', 'Cocoa_in_Percent']]
y=df[['Ratingcategory']]
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X_std, y, test_size=0.33, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# loading library
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=1)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print (accuracy_score(y_test, pred))

from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
 max_depth=4, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

y_pred = clf_entropy.predict(X_test)
y_pred

print ("Accuracy is ", accuracy_score(y_test,y_pred)*100)
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
clr_logist=lr.fit(X_train, y_train)
y_pred = clr_logist.predict(X_test)
y_pred
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


