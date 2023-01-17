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
import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt
df=pd.read_csv('../input/bank-marketing-dataset/bank.csv')

df.head()
df.info()

df.shape
df.describe()
df.isnull().sum()
plt.figure(figsize=(20,12))

plt.subplot(211)

sns.countplot(x = 'job', data = df, order = df['job'].value_counts().index)

plt.subplot(212)

sns.boxplot(x='job',y='age',data=df)
plt.figure(figsize=(15,8))

sns.boxplot(x='default',y='balance',hue='deposit',data=df)
plt.figure(figsize=(15,8))

sns.boxplot(x='job',y='balance',hue='deposit',data=df)
plt.figure(figsize=(15,8))

sns.boxplot(x='education',y='balance',hue='deposit',data=df)
plt.figure(figsize=(15,8))

sns.countplot('marital', data = df)
plt.figure(figsize=(20,6))

plt.subplot(311)

plt.title('Price Distrubution by Martial Status',fontsize=20)

sns.distplot(df[df.marital=='married'].balance)

plt.ylabel('married')

plt.subplot(312)

sns.distplot(df[df.marital=='divorced'].balance)

plt.ylabel('divorced')

plt.subplot(313)

sns.distplot(df[df.marital=='single'].balance)

plt.ylabel('single')
plt.figure(figsize=(8,16))

group = df.groupby('education')

med_balance = group.aggregate({'balance':np.median}).sort_values('balance', ascending = False)

print(med_balance)

sns.boxplot(x = 'education', y = 'balance', data = df,order = med_balance.index, color = 'steelblue')
plt.rcParams['figure.figsize']=(20,10)

plt.subplot(121)

sns.stripplot(x='housing',y='balance',data=df)

plt.subplot(122)

sns.stripplot(x='loan',y='balance',data=df)
dp=df.deposit.value_counts()

plt.figure(figsize=(16,8))

plt.subplot(1,2,1)

labels = ['Have depoist','No deposit']

explode=[0.05,0.05]

plt.pie(dp,

labels=labels,

explode=explode,

autopct='%.2f%%',

pctdistance=0.5,

shadow=True,

wedgeprops= {'linewidth':1,'edgecolor':'green'},

labeldistance = 1.1,

textprops=dict(color='k',  #  字体颜色                    

               fontsize=15),

startangle=30),

plt.legend(bbox_to_anchor=(1.0, 1.0), loc=1, borderaxespad=0,fontsize=12)

plt.title('Deposit',fontsize=20)

plt.subplot(1,2,2)

sns.barplot(x="education", y="balance", hue="deposit", data=df, estimator=lambda x: len(x) / len(df)*100)

plt.ylabel('(%)')

plt.show()
#画有无定期存款的各年龄段分布状况

plt.figure(figsize=(16,8))

plt.subplot(211)

sns.distplot(df[df.deposit=='yes'].age)

#distplot和barplot的区别：distplot是会自动按照区间比例画柱状图及其趋势线，而barplot则是通常要取确定数值或数值的均值绘图，而不能反映数据分布。

plt.ylabel('deposit=yes')

plt.subplot(212)

sns.distplot(df[df.deposit=='no'].age)

plt.ylabel('deposit=no')
#同样对不同的年龄层，是否有定期存款的分类所占比例画图，所以要按年龄划分年龄层

data=df

data['age_status']=data['age']

def agerank(age):

    if age<20:

        age_status='teen'

    elif age>=20 and age<30:

        age_status='young'

    elif age>=30 and age<40:

        age_status='mid'

    elif age>=40 and age<60:

        age_status='mid_old'

    else:age_status='old'

    return age_status

data.age_status=data.age_status.transform(lambda x:agerank(x))

data1=(data.groupby(['age_status','deposit']).age.count()/data.groupby(['age_status']).age.count()).to_frame().reset_index() 

#reset_index()将会将原来的索引index作为新的一列，否则标识数字的属性部分不会成为列

sns.barplot(x='age_status',y='age',data=data1,hue='deposit')

#age在这一部分实际上已经变为了百分比，而不是原来的数值，上一个语句因为是把age分类了，并没有换属性名，所以现实的是百分比

plt.ylabel('(%)')

data1
#原理同上，同样对不同的年龄层，上一次营销的结果的分类所占比例画图

data2=(data[data.poutcome!='unknown'].groupby(['age_status','poutcome']).age.count()/data.groupby(['age_status']).age.count()).to_frame().reset_index() 

sns.barplot(x='age_status',y='age',data=data2,hue='poutcome')

plt.ylabel('(%)')

data2
data['percent']=1 #添加新的一列全为1的值，

data3=(data.groupby(['job','deposit']).percent.count()/data.groupby(['job']).percent.count()).to_frame().reset_index()

data4=(data[data.poutcome!='unknown'].groupby(['job','poutcome']).percent.count()/data.groupby(['job']).percent.count()).to_frame().reset_index()



plt.subplot(211)

sns.barplot(x='job',y='percent',data=data3,hue='deposit')

plt.subplot(212)

sns.barplot(x='job',y='percent',data=data4,hue='poutcome')
import datetime

# date=bank.pdays

now=datetime.datetime.today()

bank_date=df

bank_date['compain_date']=bank_date.pdays.transform(lambda x:now-datetime.timedelta(days=x))

bank_date['month']=bank_date['compain_date'].transform(lambda x:x.strftime('%m'))

plt.bar(bank_date['month'].value_counts().index,bank_date['month'].value_counts())

plt.xlabel('month')
data=bank_date.groupby(['month','poutcome']).count().reset_index()

sns.barplot(x='month',y='age',data=data,hue='poutcome')
sns.barplot(x='month',y='age',data=data[data['poutcome']!='unknown'],hue='poutcome')
from sklearn.preprocessing import LabelEncoder

data5=df

data5['deposit']=LabelEncoder().fit_transform(data5['deposit'])

#把deposit转化为数值变量

cancel_corr = df.corr()["deposit"]

cancel_corr.abs().sort_values(ascending = False)
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

# preprocessing

data5 = data5[data5['deposit'].notna()]

features = ['duration', 'pdays', 'previous', 'campaign','balance']

X = data5[features]

Y = data5["deposit"]

# missing value with median

num_transformer = SimpleImputer(strategy="median")

num_transformer.fit_transform(X)

# extract training data (80%) and test data (20%)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)
Y_pred=logreg.predict(X_test)

# get confusion matrix

cnf_matrix = metrics.confusion_matrix(Y_test, Y_pred)

# visualize confusion matrix

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

# accuracy, percision, recall

print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))

print("Precision:",metrics.precision_score(Y_test, Y_pred))

print("Recall:",metrics.recall_score(Y_test, Y_pred))
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

num_features = ['duration', 'pdays', 'previous', 'campaign','balance','day','age']

cat_features = ['job','marital','education', 'default','housing','loan','contact','poutcome','month']

features = num_features + cat_features

X = df.drop(["deposit"], axis=1)[features]

y = df["deposit"]

num_transformer = SimpleImputer(strategy="constant", fill_value=0)

# deal with categorical data

cat_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy="constant", fill_value="unkown")), 

                                   ("onehot", OneHotEncoder(handle_unknown="ignore"))])

preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),

                                               ("cat", cat_transformer, cat_features)])
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.8, random_state=0)

model = Pipeline(steps=[('preprocessor', preprocessor),('rf', RandomForestClassifier(random_state=42,n_jobs=-1))])

model.fit(X_train, y_train)

pred = model.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, pred)

# visualize confusion matrix

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')

# accuracy, percision, recall

print("Accuracy:",metrics.accuracy_score(y_test, pred))

print("Precision:",metrics.precision_score(y_test, pred))

print("Recall:",metrics.recall_score(y_test, pred))