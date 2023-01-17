import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report,accuracy_score,roc_curve,confusion_matrix

from sklearn.preprocessing import binarize
columns=['age','workclass','fnlwgt','education','education-num','marital_status','occupation','relationship','race','sex','capital_gain'

        ,'capital-loss','hours-per-week','native-country','income']
test_dir = '/kaggle/input/us-census-data/adult-test.csv'

training_dir = '/kaggle/input/us-census-data/adult-training.csv'
data=pd.read_csv(training_dir,names=columns)
data.head()
data.describe()
data.info()
data.isnull().sum()
age_categ=[]

for age in data.age:

    if age<13:

        age_categ.append('kid')

    else:

        if age<19:

            age_categ.append('teen')

        else:

            if age<35:

                age_categ.append('young')

            else:

                if age<50:

                    age_categ.append('adult')

                else:

                    age_categ.append('old')

data.insert(1,'age_categ',age_categ)
sns.countplot(data.age_categ)

data.drop(['age'],axis=1,inplace =True)
data.workclass.unique()

(data.workclass==' ?').sum()/len(data)*100

# 5 percent of workclass is filled with ?
data.workclass.replace(' ?',data.workclass.mode()[0],inplace=True)

data.workclass.replace(' Never-worked',' Without-pay',inplace=True)
plt.xticks(rotation=90)

sns.countplot(data.workclass)
data.fnlwgt.plot(kind='box')
data=data[data.fnlwgt<600000]
data.fnlwgt.plot(kind='box')

plt.figure(figsize=(10,10))

plt.xticks(rotation=90)

sns.countplot(data.education)
sns.countplot(data['education-num'])

#education number is alternative way of representating education column so we can drop one of them
data.drop(['education'],axis=1,inplace=True)
data['marital_status'].unique()
plt.xticks(rotation=90)

sns.countplot(data['marital_status'])
data['occupation'].unique()
data.occupation.replace(' ?',data.occupation.mode()[0],inplace=True)


plt.xticks(rotation=90)

sns.countplot(data.occupation)
data.relationship.unique()

plt.xticks(rotation=90)

sns.countplot(data.relationship)
plt.xticks(rotation=90)

sns.countplot(data.race)
data.race.unique()

replace=data.race.unique()[2:]

for to_replace in replace:

    print(to_replace)

    data['race'].replace(to_replace,' Other',inplace=True)
sns.countplot(data.race)
sns.countplot(data.sex)

data['is_capital']=[0 if capital==0 else 1 for capital in data['capital_gain']]

sns.countplot(data['is_capital'])

data['is_loss']=[0 if capital==0 else 1 for capital in data['capital-loss']]

sns.countplot(data['is_loss'])

#dropping capital_gain and capital_loss

data.drop(['capital_gain','capital-loss'],axis=1,inplace=True)
data['hours-per-week'].hist(bins=15)



diff_hours_categ=['>=60','>40&<60','<=40&>30','<=30']

hours_categ=[]

for hours in data['hours-per-week']:

    if hours>=60:

        hours_categ.append(diff_hours_categ[0])

    else:

        if hours>40:

            hours_categ.append(diff_hours_categ[1])

        else:

            if hours>30:

                hours_categ.append(diff_hours_categ[2])

            else:

                hours_categ.append(diff_hours_categ[3])

data['hours_categ_week']=hours_categ
sns.countplot(data['hours_categ_week'])

#dropping hours per week

data.drop(['hours-per-week'],axis=1,inplace=True)
data['native-country'].value_counts()

# we can make only two native country United-States and other

data['native-country']=[' United-States' if country==' United-States' else ' Other' for country in data['native-country']]

sns.countplot(data['native-country'])

sns.countplot(data.income)

data.head()



diff_categ_count=data['age_categ'].value_counts()

group_table=data.groupby(['age_categ','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
#since adult and old distrbution is similar in income so we can make them one 

data.age_categ.replace('old','adult',inplace=True)
diff_categ_count=data['workclass'].value_counts()

group_table=data.groupby(['workclass','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
sns.violinplot(data['income'],data['fnlwgt'],inner='quart')

data.drop(['fnlwgt'],axis=1,inplace=True)

diff_categ_count=data['education-num'].value_counts()

group_table=data.groupby(['education-num','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
# 15 and 16  , 11 and 12 , 2 and 3 ,4 to 7 can be combined

replace_dict={

    15:16,11:12,3:2,5:4,6:4,7:4

}

for num in replace_dict:

    data.replace(num,replace_dict[num],inplace=True)
diff_categ_count=data['education-num'].value_counts()

group_table=data.groupby(['education-num','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
diff_categ_count=data['marital_status'].value_counts()

group_table=data.groupby(['marital_status','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
#reducing some categories

data.replace(' Married-civ-spouse',' Married-AF-spouse',inplace=True)

data.replace(' Married-spouse-absent',' Widowed',inplace=True)


diff_categ_count=data['occupation'].value_counts()

group_table=data.groupby(['occupation','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True,figsize=(10,10))

plt.ylabel('percentage of income categ')


diff_categ_count=data['relationship'].value_counts()

group_table=data.groupby(['relationship','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
diff_categ_count=data['race'].value_counts()

group_table=data.groupby(['race','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
diff_categ_count=data['sex'].value_counts()

group_table=data.groupby(['sex','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')


diff_categ_count=data['native-country'].value_counts()

group_table=data.groupby(['native-country','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')


diff_categ_count=data['is_capital'].value_counts()

group_table=data.groupby(['is_capital','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
diff_categ_count=data['is_loss'].value_counts()

group_table=data.groupby(['is_loss','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
diff_categ_count=data['hours_categ_week'].value_counts()

group_table=data.groupby(['hours_categ_week','income']).size().astype(float)

for categ in group_table.index.levels[0]:

    for income in group_table[categ].index:

        group_table[categ][income]=group_table[categ][income]/diff_categ_count[categ]*100

group_table.unstack().plot(kind='bar',stacked=True)

plt.ylabel('percentage of income categ')
data.replace('>=60','>40',inplace=True)

data.replace('>40&<60','>40',inplace=True)
data.head()

features=list(data.columns)

print(features)



features.remove('income')

X=data[features].copy()

Y=data['income']
X.head()

le=LabelEncoder()

for feature in features:

    X[feature]=le.fit_transform(X[feature])

Y=[0 if val == ' <=50K' else 1 for val in Y]
X=pd.get_dummies(X,columns=features)

X.head()
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.20,random_state=9)

lr=LogisticRegression()

lr.fit(train_x,train_y)

print('accuracy on training data:',lr.score(train_x,train_y))
predicted_y=lr.predict(test_x)

print(classification_report(test_y,predicted_y))

print('accuracy_score is on test data: ',accuracy_score(test_y,predicted_y))
plt.figure(figsize=(5,5))

sns.heatmap(confusion_matrix(test_y,predicted_y),annot=True,fmt='.5g')

plt.ylabel('actual class')

plt.xlabel('predicted class')