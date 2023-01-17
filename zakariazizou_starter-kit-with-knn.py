#importing the essential librarys 
%matplotlib inline



import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

#importing the dataset 
df = pd.read_csv(r'../input/adult-income/adult.csv')

df.head()
df['income'] = df['income'].map({'<=50K':'0' , '>50K':'1'})
df.head()
df = df.replace('?',np.nan)
for i   in ( df.columns) :

    

    if (df[i].dtypes == object):

     print(i , '\n')

     print (df[i].value_counts(), '\n')

   
null_val = df.isna().sum()

null_val = pd.DataFrame(null_val , columns=['null'])

sum_total = len(df)

null_val['percent %'] = null_val['null']/sum_total

null_val['percent %'] = null_val['percent %']*100

null_val.sort_values('percent %', ascending=False)



null_val.drop(['age', 'fnlwgt', 'education', 'educational-num',

       'marital-status', 'relationship', 'race', 'gender',

       'capital-gain', 'capital-loss', 'hours-per-week', 

       'income' ] , axis=0)
for i in df[['workclass','occupation','native-country']] : 

    print ('for the attribute ****', i ,'\n')

    print ( '\n', df[i].unique() , '\n')
for i in df[['workclass','occupation','native-country']] : 

    df[i].fillna(df[i].mode()[0] , inplace = True)
df[['workclass','occupation','native-country']].isna().sum()
null_val_coret = df[['workclass','occupation','native-country']].isna().sum()

null_val_coret = pd.DataFrame(null_val_coret , columns=['null'])

null_val_coret['percent'] = (null_val_coret['null']/sum_total)*100

null_val_coret
df.describe(include='all').T
df.info()

fig = plt.figure(figsize=(10,10))

sns.boxplot(x='income' , y='age' , data=df)

plt.show()
fig = plt.figure(figsize=(15,8))

ax = sns.countplot(x='workclass' , hue='income' , palette="bright", edgecolor="0" , data=df).set_title('count')

plt.xticks(rotation=45)
fig = plt.figure(figsize=(15,8))

ax = sns.countplot(x='education' , hue='income' , palette='bright' , edgecolor='0' , data=df)

plt.xticks(rotation=45)
fig = plt.figure(figsize=(15,8))

g = sns.countplot(x='marital-status' , hue='income' , data=df , palette='bright' , edgecolor='0') 
fig = plt.figure(figsize=(15,8))

ax = sns.countplot(x='gender' , hue='income' , palette='bright' , edgecolor='0' , data=df)

plt.xticks(rotation=45)
fig = plt.figure(figsize=(15,8))

ax = sns.countplot(x='occupation' , hue='income' , palette='bright' , edgecolor='0' , data=df)

plt.xticks(rotation=45)
fig = plt.figure(figsize=(15,8))

ax = sns.countplot(x='relationship' , hue='income' , palette='bright' , edgecolor='0' , data=df)

plt.xticks(rotation=45)
plt.figure(figsize=(20,7))

ax = sns.catplot(y='race' , hue='income' , kind='count' , col='gender' , data=df , palette='bright' , edgecolor='0') 

df['hours-per-week'].hist(figsize=(8,8))

plt.show()
df['income'] = df['income'].astype(int)
plt.figure(figsize=(10,10))

sns.heatmap(df.corr())
df.describe().T
df = df.drop(['educational-num'] , axis=1)

df = df.drop(['fnlwgt'] , axis=1)
df.head()
df = df[(df['hours-per-week'] < 80)]

df = df[(df['hours-per-week'] > 20)]
print((df['hours-per-week']).min())

print(df['hours-per-week'].max())

df['hours-per-week'].mean()

print ('> 60 hour/week :' , df.income[df['hours-per-week'] > 60].count())

print ('> 40 hour/week :' , df.income[df['hours-per-week'] > 40].count())

print ('> 50 hour/week :' , df.income[df['hours-per-week'] > 50].count())
df['age'].describe()
print ('75% outlier :' , df.age[df['age'] > 60].count())

print ('25% outlier :' , df.age[df['age'] < 20].count())
df = df[df ['age'] > 20]

df = df[df ['age'] < 60]
print ('75% outlier :' , df.age[df['age'] > 60].count())

print ('25% outlier :' , df.age[df['age'] < 20].count())
df.columns
df.head()
dfen = df
dfen.head()
from sklearn import preprocessing 

le = preprocessing.LabelEncoder()
dfen [[ 'workclass', 'education', 'marital-status', 'occupation',

       'relationship', 'race', 'gender',  'native-country', 'income']]   =   dfen [[ 'workclass', 'education', 'marital-status', 'occupation',

       'relationship', 'race', 'gender', 'native-country', 'income']].apply(le.fit_transform)
dfen.head()
X = dfen.drop(['income'] , axis=1)

Y = dfen['income']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train , y_train)
yhat = knn.predict(X_test)
yhat
y_test.values
print(knn.score(X_test , y_test))
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,yhat))
print(classification_report(y_test,yhat))
from sklearn.model_selection import cross_val_score
accuracy_rate = []



for i in range (1,10):

    knn = KNeighborsClassifier(n_neighbors=i)

    score = cross_val_score(knn , X , Y , cv=10) 

    accuracy_rate.append(score.mean())
print(accuracy_rate)
plt.figure(figsize=(15,7))

plt.plot(range(1,10) , accuracy_rate , marker='o' , color='blue' , markerfacecolor='red')

plt.title('accuracy_rate vs. K Value')

plt.xlabel('K')

plt.ylabel('accuracy_rate')
max(accuracy_rate)
knn_final = KNeighborsClassifier(n_neighbors=6)



knn_final.fit(X_train , y_train)
pred = knn_final.predict(X_test)
print(confusion_matrix(y_test , pred))
print(classification_report(y_test , pred))
print(knn_final.score(X_test , y_test))