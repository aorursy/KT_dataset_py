%matplotlib inline

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



%config InlineBackend.figure_format = 'retina'

sns.set()
df = pd.read_csv('../input/UCI_Credit_Card.csv')

df.info()

df.sample(10)
print('SEX ' + str(sorted(df['SEX'].unique())))

print('EDUCATION ' + str(sorted(df['EDUCATION'].unique())))

print('MARRIAGE ' + str(sorted(df['MARRIAGE'].unique())))

print('PAY_0 ' + str(sorted(df['PAY_0'].unique())))

print('default.payment.next.month ' + str(sorted(df['default.payment.next.month'].unique())))
fill = (df.EDUCATION == 0) | (df.EDUCATION == 5) | (df.EDUCATION == 6)

df.loc[fill, 'EDUCATION'] = 4



print('EDUCATION ' + str(sorted(df['EDUCATION'].unique())))
df = df.rename(columns={'default.payment.next.month': 'DEFAULT', 

                        'PAY_0': 'PAY_1'})

df.head()
df.describe().transpose()
sns.set(rc={'figure.figsize':(27,10)})

sns.set_context("talk", font_scale=0.7)

    

sns.heatmap(df.iloc[:,1:].corr(), cmap='Reds', annot=True);
sns.set(rc={'figure.figsize':(9,7)})

sns.set_context("talk", font_scale=0.8)



edu = sns.countplot(x='EDUCATION', hue='DEFAULT', data=df)

edu.set_xticklabels(['Graduate School','University','High School','Other'])

plt.show()
default0 = df.groupby(df['EDUCATION'][df['DEFAULT'] == 0]).size().reset_index(name='NOT_DEFAULT')

default1 = df.groupby(df['EDUCATION'][df['DEFAULT'] == 1]).size().reset_index(name='DEFAULT')

total = df.groupby('EDUCATION').size().reset_index(name='TOTAL')



eduTable = default0.join(default1['DEFAULT']).join(total['TOTAL'])

eduTable['EDUCATION'] = ['Graduate School','University','High School','Other']



eduTable
eduTable['NOT_DEFAULT'] = round((default0['NOT_DEFAULT']/total['TOTAL'])*100,2)

eduTable['DEFAULT'] = round((default1['DEFAULT']/total['TOTAL'])*100,2)



eduPct = eduTable.iloc[:,0:3]

eduPct = eduPct.rename(columns={'NOT_DEFAULT': 'NOT_DEFAULT(%)', 'DEFAULT': 'DEFAULT(%)'})



eduPct
sns.set(rc={'figure.figsize':(9,4)})

sns.set_context("talk", font_scale=0.8)



ax = eduPct.plot(x='EDUCATION', kind='barh', stacked=True, title='Education Level vs. Default')

ax.set_xlabel('PERCENT')

ax.get_legend().set_bbox_to_anchor((1, 0.9))

plt.show()
sns.set(rc={'figure.figsize':(9,7)})

sns.set_context("talk", font_scale=0.8)



marri = sns.countplot(x="MARRIAGE", hue='DEFAULT', data=df )

marri.set_xticklabels(['Others','Married','Single','Divorce'])

plt.show()

default0 = df.groupby(df['MARRIAGE'][df['DEFAULT'] == 0]).size().reset_index(name='NOT_DEFAULT')

default1 = df.groupby(df['MARRIAGE'][df['DEFAULT'] == 1]).size().reset_index(name='DEFAULT')

total = df.groupby('MARRIAGE').size().reset_index(name='TOTAL')



marriTable = default0.join(default1['DEFAULT']).join(total['TOTAL'])

marriTable['MARRIAGE'] = ['Others','Married','Single','Divorce']



marriTable
marriTable['NOT_DEFAULT'] = round((default0['NOT_DEFAULT']/total['TOTAL'])*100,2)

marriTable['DEFAULT'] = round((default1['DEFAULT']/total['TOTAL'])*100,2)



marriPct = marriTable.iloc[:,0:3]

marriPct = marriPct.rename(columns={'NOT_DEFAULT': 'NOT_DEFAULT(%)', 'DEFAULT': 'DEFAULT(%)'})



marriPct
sns.set(rc={'figure.figsize':(9,4)})

sns.set_context("talk", font_scale=0.8)



ax = marriPct.plot(x='MARRIAGE', kind='barh', stacked=True, title='Marital Status vs. Default')

ax.set_xlabel('PERCENT')

ax.get_legend().set_bbox_to_anchor((1, 0.9))

plt.show()
sns.set(rc={'figure.figsize':(15,7)})

sns.set_context("talk", font_scale=0.8)



pay1 = sns.countplot(y="PAY_1", hue='DEFAULT', data=df)

pay1.set_yticklabels(['No Consumption','Paid in Full','Use Revolving Credit','Delay 1 mth','Delay 2 mths'

                     ,'Delay 3 mths','Delay 4 mths','Delay 5 mths','Delay 6 mths','Delay 7 mths','Delay 8 mths'])

pay1.set_title('Credit Behaviour (most recent month)')



plt.show()
default0 = df.groupby(df['PAY_1'][df['DEFAULT'] == 0]).size().reset_index(name='NOT_DEFAULT')

default1 = df.groupby(df['PAY_1'][df['DEFAULT'] == 1]).size().reset_index(name='DEFAULT')

total = df.groupby('PAY_1').size().reset_index(name='TOTAL')



pay1Table = default0.join(default1['DEFAULT']).join(total['TOTAL'])

pay1Table['PAY_1'] = ['No Consumption','Paid in Full','Use Revolving Credit','Delay 1 mth','Delay 2 mths'

                     ,'Delay 3 mths','Delay 4 mths','Delay 5 mths','Delay 6 mths','Delay 7 mths','Delay 8 mths']



pay1Table
pay1Table['NOT_DEFAULT'] = round((default0['NOT_DEFAULT']/total['TOTAL'])*100,2)

pay1Table['DEFAULT'] = round((default1['DEFAULT']/total['TOTAL'])*100,2)



pay1Pct = pay1Table.iloc[:,0:3]

pay1Pct = pay1Pct.rename(columns={'NOT_DEFAULT': 'NOT_DEFAULT(%)', 'DEFAULT': 'DEFAULT(%)'})



pay1Pct
sns.set(rc={'figure.figsize':(9,5)})

sns.set_context("talk", font_scale=0.8)



ax = pay1Pct.sort_index(ascending=False).plot(x='PAY_1', kind='barh', stacked=True, title='Credit Behaviour vs. Default')

ax.set_xlabel('PERCENT')

ax.get_legend().set_bbox_to_anchor((1, 0.9))

plt.show()
error1 = df.query('BILL_AMT1 < 0 and DEFAULT == 1').loc[:,('ID','BILL_AMT1','DEFAULT')]

error1.sample(5)
error2 = df.query('BILL_AMT1 > LIMIT_BAL').loc[:,('ID','LIMIT_BAL','BILL_AMT1')]

error2.sample(5)
df['AGE'].describe()
sns.distplot(df['AGE'], norm_hist=False, kde=False);
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(14,5))



ax1.set_title('All Client', fontsize=14)

ax2.set_title('Defaulted Client', fontsize=14)



sns.distplot(df['AGE'], norm_hist=False, kde=False, ax=ax1);

sns.distplot(df['AGE'][df['DEFAULT'] == 1], norm_hist=True, kde=False, ax=ax2);


default0 = df.groupby(df['AGE'][df['DEFAULT'] == 0]).size().reset_index(name='NOT_DEFAULT')

default0 = default0.fillna(0)

default1 = df.groupby(df['AGE'][df['DEFAULT'] == 1]).size().reset_index(name='DEFAULT')

default1 = default1.fillna(0)

total = df.groupby('AGE').size().reset_index(name='TOTAL')



ageTable = total.join(default0.set_index('AGE'),on='AGE').join(default1.set_index('AGE'),on='AGE')

ageTable = ageTable[['AGE', 'NOT_DEFAULT', 'DEFAULT', 'TOTAL']]

ageTable = ageTable.fillna(0)

ageTable
ageTable['NOT_DEFAULT'] = round((ageTable['NOT_DEFAULT']/ageTable['TOTAL'])*100,2)

ageTable['DEFAULT'] = round((ageTable['DEFAULT']/ageTable['TOTAL'])*100,2)



agePct = ageTable.iloc[:,0:3]

agePct = agePct.rename(columns={'NOT_DEFAULT': 'NOT_DEFAULT(%)', 'DEFAULT': 'DEFAULT(%)'})



agePct
sns.set(rc={'figure.figsize':(9,10)})

sns.set_context("talk", font_scale=0.5)



ax = agePct.sort_index(ascending=False).plot(x='AGE', kind='barh', stacked=True, title='Age vs. Default')

ax.set_xlabel('PERCENT')

ax.get_legend().set_bbox_to_anchor((1, 0.98))

plt.show()
X = df[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE'

        ,'PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'

        ,'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'

        ,'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]

y = df['DEFAULT'] 
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.metrics import accuracy_score



logmodel = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



accuracy_score(y_test,predictions)
X = df[['SEX','EDUCATION','MARRIAGE','AGE']]

y = df['DEFAULT']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



accuracy_score(y_test,predictions)
X = df[['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']]

y = df['DEFAULT']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



accuracy_score(y_test,predictions)
X = df[['PAY_1']]

y = df['DEFAULT']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

logmodel = LogisticRegression(solver='lbfgs', max_iter=500, random_state=0)

logmodel.fit(X_train,y_train)



predictions = logmodel.predict(X_test)



accuracy_score(y_test,predictions)