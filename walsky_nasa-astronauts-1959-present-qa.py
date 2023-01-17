import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

from matplotlib.pyplot import pie, axis, show

from matplotlib import cm

%matplotlib inline



data = pd.read_csv('../input/astronauts.csv')

data.head()
data = data.sort('Space Flight (hr)', ascending=[0])
plt.figure(figsize=(8,8))

data['Space Flight (hr)'].head(25).plot.bar()

plt.xlabel('Index')

plt.ylabel('Time in Space')
countCollege=data["Alma Mater"].value_counts()

plt.figure(figsize=(10,10))

UniversitiesGraph = sns.countplot(y="Alma Mater", data=data,

                   order=countCollege.nlargest(50).index,

                   palette='GnBu_d')

plt.show()
CollegeCount = data['Undergraduate Major'].value_counts()

plt.figure(figsize=(12,15))

CollegeGraph = sns.countplot(y="Undergraduate Major", data=data,

                   order=CollegeCount.index,

                   palette='GnBu_d')

plt.show()
print("Total out of 356: ", data['Graduate Major'].count(), "has a Graduate Degree")

graduateCount = data['Graduate Major'].count()

result = graduateCount/356 * 100

result = format(result, '.2f')

print(result, '%')

#print(data['Graduate Major'].value_counts())
data['Was Military?'] = data['Military Rank'].apply(lambda x: 0 if type(x) == float else 1)

data['Was Military?'].replace([0, 1], ["Wasn't Military", "Was Military"], inplace=True)
militaryGraph2 = data['Was Military?'].value_counts()

print(militaryGraph2)



plt.figure(figsize=(8,8))

pie(militaryGraph2, labels=militaryGraph2.index, autopct='%1.1f%%');

plt.show()
plt.figure(figsize=(10,5))

BranchGraph = sns.countplot(y="Military Branch", data=data,

                   order=data['Military Branch'].value_counts().index,

                   palette='GnBu_d')

plt.show()
plt.figure(figsize=(10,5))

RankGraph = sns.countplot(y="Military Rank", data=data,

                   order=data['Military Rank'].value_counts().index,

                   palette='GnBu_d')

plt.show()
Gender = data['Gender'].value_counts()

print(Gender)



plt.figure(figsize=(8,8))

pie(Gender, labels=Gender.index, autopct='%1.1f%%', startangle=180)



plt.show()