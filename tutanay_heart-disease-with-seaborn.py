# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.linear_model import LinearRegression

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/heart.csv')
data.head()
data.info()
data.columns
data.target.value_counts()
data.corr()
f,ax=plt.subplots(figsize=(15,15))

sns.heatmap(data.corr(),annot=True,fmt='.1f',ax=ax,linewidth=.5)

plt.show()

data.sex=data.sex/max(data.sex)

sns.jointplot(data.age,data.oldpeak,color="red",kind="kde",height=7)

plt.savefig('graph.png')

plt.show()
#sns.swarmplot(x="age",y="oldpeak",hue="target",color="orange",data=data)

plt.figure(figsize=(12,8))

ax = sns.swarmplot(x="age", y="oldpeak",hue="target",data=data)

#yorum satırındaki ve altındaki aynı şeyler



plt.show()
plt.figure(figsize=(12,8))

sns.boxplot(data.age,data.oldpeak,color="blue")

plt.show()
plt.figure(figsize=(15,10))

sns.violinplot(x="age",y="oldpeak",color="yellow",data=data)

plt.show()
plt.figure(figsize=(12,8))

sns.pointplot(x="age",y="oldpeak",hue="target",data=data,markers=["x","o"],linestyles=["-","--"],palette="Set2")

plt.show()
def sinplot(flip=1):

    x = np.linspace(0, 14, 100)

    for i in range(1, 7):

        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

        sns.set_palette("husl")

sinplot()
plt.figure(figsize=(12,8))

sns.countplot(x="age",hue="target",data=data,color="blue",alpha=0.5)

plt.show()
plt.figure(figsize=(12,8))

sns.lmplot(x="age",y="oldpeak",hue="target",data=data,palette="RdBu")

plt.show()


data.fbs.dropna(inplace=True)



labels=data.fbs.value_counts().index

colors=["purple","turquoise"]

explode=[0,0]

sizes=data.fbs.value_counts().values

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=explode,colors=colors,labels=labels,autopct='%1.1f%%')

plt.title('Fasting blood sugar chart',color="blue",fontsize=15)

sns.kdeplot(data.oldpeak,data.age,shade=True,cut=7)

plt.show()
#BAR PLOT

data.columns
#BAR PLOT

#trestbps - oldpeak dinlenme kan basıncı (hastaneye girişte mm Hg cinsinden)-Dinlenmeye göre egzersizle indüklenen ST depresyonu

plt.figure(figsize=(15,10))

sns.barplot(x=data['trestbps'],y=data['oldpeak'])

plt.xticks(rotation=45)

plt.xlabel('trestbps')

plt.ylabel('Oldpeak')

plt.title('Trestbps-Oldpeak Bar Plot')