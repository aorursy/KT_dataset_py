import seaborn as sns

import pandas as pd

import numpy as np



import matplotlib.pyplot as plt  

import seaborn as sns
data=pd.read_csv('../input/titanic/train.csv')
print("Train: rows:{} columns:{}".format(data.shape[0], data.shape[1]))
data.describe()
data.describe(include='object')
data.info()
sns.barplot(data["Sex"], data["Survived"])



print("Percentage of females survived:", data["Survived"][data["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males survived:", data["Survived"][data["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
sns.barplot(data["Pclass"], data["Survived"])

plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 10)

plt.xlabel("Socio-Economic class")

plt.ylabel("% of Passenger Survived")

labels = ['1st', '2nd', '3rd']

val = [0,1,2] 

plt.xticks(val, labels);



print("Percentage of 1st class who survived:", data["Survived"][data["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of 2nd class who survived:", data["Survived"][data["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of 3rd class who survived:", data["Survived"][data["Pclass"] == 3].value_counts(normalize = True)[1]*100)
plt.subplot(1,2,1)

sns.barplot(data['Sex'],data['Survived'])

plt.title('Survived vs Sex')

plt.subplot(1,2,2)

sns.countplot('Sex',hue='Survived',data=data)

plt.title('Sex:Survived vs Dead')

plt.tight_layout()
plt.subplot(1,2,1)

data['Pclass'].value_counts().plot.bar()

plt.title('Number Of Passengers By Pclass')

plt.ylabel('Count')

plt.subplot(1,2,2)

sns.countplot('Pclass',hue='Survived',data=data)

plt.title('Pclass:Survived vs Dead')

plt.tight_layout()
plt.figure(figsize=(13,5))

plt.subplot(1,2,1)

sns.violinplot("Pclass","Age", hue="Survived", data=data,split=True)

plt.title('Pclass and Age vs Survived')

plt.yticks(range(0,110,10))

plt.subplot(1,2,2)

sns.violinplot("Sex","Age", hue="Survived", data=data,split=True)

plt.title('Sex and Age vs Survived')

plt.yticks(range(0,110,10))

plt.tight_layout()
plt.figure(figsize=(20,8))



plt.subplot(2,2,1)

sns.countplot('Embarked',data=data)

plt.title('No. Of Passengers Boarded')



plt.subplot(2,2,2)

sns.countplot('Embarked',hue='Sex',data=data)

plt.title('Male-Female Split for Embarked')

plt.tight_layout()



plt.subplot(2,2,3)

sns.countplot('Embarked',hue='Survived',data=data)

plt.title('Embarked vs Survived')

plt.tight_layout()



plt.subplot(2,2,4)

sns.countplot('Embarked',hue='Pclass',data=data)

plt.title('Embarked vs Pclass')

plt.show()



plt.tight_layout()
ax= sns.boxplot(x="Pclass", y="Age", data=data)

ax= sns.stripplot(x="Pclass", y="Age", data=data, jitter=True, edgecolor="gray")

plt.show()
tab = pd.crosstab(data['Sex'], data['Survived'])

print(tab)



tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)

plt.xlabel('Gender')

plt.ylabel('Survival Percentage')
sns.lmplot(x='Age', y='Fare', hue='Survived', 

           data=data.loc[data['Survived'].isin([1,0])], 

           fit_reg=False)
data.corr()
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(),cmap="ocean",annot=True)
sns.distributions._has_statsmodels=False

sns.pairplot(data, kind="scatter", hue="Survived", palette="Set2",plot_kws=dict(s=80, edgecolor="black", linewidth=0.4))