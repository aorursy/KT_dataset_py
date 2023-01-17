import pandas as pd
import numpy as np
df = pd.read_csv('../input/titanic-dataset/train.csv')
df.head()
df.shape
df.columns
df.info()
df.describe()
df.describe(include='object')
df['Sex'].value_counts()
df['Embarked'].value_counts()
df['Sex'].value_counts(normalize=True)
df['Name'].head()
df['Age'].mean()
df['Age'].min()
df['Sex'].value_counts()
df.loc[0:15 , 'Name':'Embarked']
df.iloc[0:15 ,0:4]
df[df['Age']==df['Age'].max()]['Name']
df.sort_values(by='Age' , ascending=True).head()
df.sort_values(by=['Name' , 'Age'] , ascending=[True , False]).head()
d={'male':0 , 'female':1}
df['Sex']=df['Sex'].map(d)
df.head()
o={0:'male' , 1:'female'}
df=df.replace({'Sex':o})
df.head()
df.groupby(by='Survived')['Age'].describe()
df.groupby(by='Embarked')['Age'].agg(np.mean)
pd.crosstab(df['Sex'],df['Survived'], normalize=True)
df.pivot_table(['Age' , 'Fare'],['Survived'], aggfunc='mean')
import matplotlib.pyplot as plt
import seaborn as sns
%config InlineBackend.figure_format='retina'
sns.countplot(x='Survived', data=df)
sns.countplot(x='Survived', hue='Sex', data=df)
sns.lineplot(x='Age' , y='Fare' , data=df)
df.plot.scatter('Age' , 'Fare')
df.plot.scatter('Age' , 'Survived')
df.plot.scatter('Age','Name')
df['Age'].hist(bins=70)
import matplotlib.pyplot as plt
sizes=df['Survived'].value_counts()
fig1,ax1=plt.subplots()
ax1.pie(sizes,labels=['Not Survived' , 'Survived'] , autopct='%1.1f%%' , shadow=True)
plt.show()