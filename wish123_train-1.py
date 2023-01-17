import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train=pd.read_csv("../input/titanic/train.csv")
train.head()
train.columns= map(str.lower,train.columns)
train= train.rename(columns={'passengerid':'id'})
train.head()
#fare amount spent by male and female
train.groupby('sex')['fare'].sum()
train['sex'].value_counts().plot(title='sex ratio', kind='bar',x='sex',y='count',color='g')
plt.show()
train['pclass'].value_counts().sort_index().plot(kind='bar')
sns.countplot(train['pclass'])
sns.barplot(train['survived'], train['fare'], train['pclass'])
train[train['sex']=='male']['survived'].value_counts().plot(kind='bar')


sns.barplot(train['sex'], train['survived'], train['pclass'])
#class one, male more than 25 years age who survived (1) or died(0)
train[(train.pclass==1)&(train.sex=='male')&(train.age>=25)]['survived'].value_counts().sort_index().plot(kind='bar')

train['name title']= train['name'].apply(lambda x: x.split (",")[1].split(" ")[1])
train.head()

train['name title'].value_counts()
def surname(x):
    if x in[ "Mr." ,"Miss.", "Mrs."]:
        return 'main'
    else: 'newee'

train['nt']=train['name title'].apply(surname)
train['nt'].value_counts()
