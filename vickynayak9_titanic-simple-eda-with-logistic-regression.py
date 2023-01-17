# Import the libararies

# Ignore warnings

import warnings

warnings.filterwarnings('ignore')



# Handle Table-Like data and Matrics

import numpy as np

import pandas as pd



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



# Configure visualization

%matplotlib inline
# Read the file

titanic_data = pd.read_csv('../input/titanic-dataset/titanic_data.csv')
titanic_data.head(10)
print('The Dataset Shape of Titanic is :', titanic_data.shape)
titanic_data.describe()
# Total num of missing values in complete Dataset

titanic_data.isnull().sum().sum()
# Total num of missing values in each variable

titanic_data.isnull().sum()
# Percentage of Missing Values

titanic_data.isnull().sum()/len(titanic_data) * 100
titanic_data.isnull().mean() * 100
#Vishualize missing values in Dataset

sns.heatmap(titanic_data.isnull(), cbar = True).set_title("Missing values heatmap")
# Total num of unique values in each variable

titanic_data.nunique()
# Since ‘Embarked’ only had two missing values and the largest number of commuters embarked from Southampton, 

# the probability of boarding from Southampton is higher. So, we fill the missing values with Southampton.

titanic_data.Embarked.fillna(titanic_data.Embarked.mode()[0], inplace = True)
# Check Embarked missing value after "Data Imputaion" method applied.

titanic_data.isnull().sum()
#Categorized the people on the basis of their salutations.

#Extract Salutation from Name field  

titanic_data['Salutation'] = titanic_data.Name.apply(lambda name: name.split(',')[1].split('.')[0].strip())
# Check extra coulmn "Salutation" added to the Dataset

titanic_data.head(10)
# Check unique values in new column Salutation

print (titanic_data['Salutation'].unique())

print ("\n")

print ("Total number of unique values in ['Salutation']:", titanic_data['Salutation'].nunique())
# Check counts of Salutation by Sex with Totals

pd.crosstab(titanic_data.Salutation,titanic_data.Sex, margins=True)
# See the above table graphically

pd.crosstab(titanic_data.Salutation,titanic_data.Sex, margins=True).T.style.background_gradient(cmap='summer_r')
titanic_data['Salutation'].replace(['Mlle','Mme','Ms','Lady','the Countess','Dr','Major','Capt','Sir','Don','Jonkheer','Col','Rev'],

                                   ['Miss','Miss','Miss','Mrs','Mrs','Mr','Mr','Mr','Mr','Mr','Other','Other','Other'],inplace=True)
# Check counts of Salutation by Sex with Totals

pd.crosstab(titanic_data.Salutation,titanic_data.Sex, margins=True)
# Re-check the above table again to see if changes are made correctly

pd.crosstab(titanic_data.Salutation,titanic_data.Sex, margins=True).T.style.background_gradient(cmap='summer_r')
# Get only that row which meets the condition above

titanic_data.loc[(titanic_data["Salutation"] == "Mr") & (titanic_data["Sex"] =='female')]
# Change the value in 'Salutation' with index value = 796

titanic_data.at[796,'Salutation']= 'Mrs'
# Re-check the row changed with above method

titanic_data.loc[[796]]
# Re-check the above table again to see if changes are made correctly

pd.crosstab(titanic_data.Salutation,titanic_data.Sex, margins=True).T.style.background_gradient(cmap='summer_r')
# For Example look at the NaN value in Age of a sample row before applying the method to fill

titanic_data.loc[[5]]
#lets check the average age by Salutation first

titanic_data.groupby('Salutation')['Age'].mean()
## Assigning the NaN Values with the Cell values of the mean ages

titanic_data.loc[(titanic_data.Age.isnull())&(titanic_data.Salutation=='Master'),'Age'] = 5

titanic_data.loc[(titanic_data.Age.isnull())&(titanic_data.Salutation=='Miss'),'Age'] = 22

titanic_data.loc[(titanic_data.Age.isnull())&(titanic_data.Salutation=='Mr'),'Age'] = 33

titanic_data.loc[(titanic_data.Age.isnull())&(titanic_data.Salutation=='Mrs'),'Age'] = 36

titanic_data.loc[(titanic_data.Age.isnull())&(titanic_data.Salutation=='Other'),'Age'] = 46
# Re-check the example and look at the NaN value in Age of a sample row as seen above 

titanic_data.loc[[5]]
# Check the dataset

titanic_data.head()
#So no null values left finally in 'Age' column

titanic_data.Age.isnull().any() 
# Check Embarked missing value after "Data Imputaion" method.

titanic_data.isnull().sum()
fig1,ax1 = plt.subplots(figsize=(8,8))

values= titanic_data['Survived'].value_counts().values.tolist()

labels= titanic_data['Survived'].value_counts().keys().tolist()

ax1.pie(values, labels= labels, autopct='%1.1f%%',shadow=True, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.legend(labels, loc="lower right")

plt.title('Distribution by Survived')

plt.show()
#fig1, ax1 = plt.subplots()

#sns.barplot(x="Survived", y="Survived", data=titanic_data, estimator=lambda x: len(x) / len(titanic_data) * 100, orient='h')

#sns.set_style("whitegrid")

#sns.set_context(font_scale=2,rc={"font.size":15, "axes.labelsize":10})

#ax1.set_xlabel=("Percent")

#ax1.set_ylabel=("Suvived")

#plt.show()
f,ax=plt.subplots(1,2,figsize=(16,8))

titanic_data['Survived'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True, startangle=90)

ax[0].set_title('Survived Vs Died')

ax[0].set_ylabel('')



sns.countplot('Survived',data=titanic_data,ax=ax[1])

ax[1].set_title('Survived Vs Died')

plt.show()
pd.crosstab(titanic_data.Pclass, titanic_data.Survived, margins=True).style.background_gradient(cmap='autumn_r')
print('Total Passengers in (PClass 1) are :', titanic_data['Pclass'].value_counts()[1])

print('Total Passengers in (PClass 2) are :', titanic_data['Pclass'].value_counts()[2])

print('Total Passengers in (PClass 3) are :', titanic_data['Pclass'].value_counts()[3])
# Pie Chart of Passengers by PClass

labels = ['PClass 1', 'PClass 2', 'PClass 3']

sizes = [titanic_data['Pclass'].value_counts()[1],

        titanic_data['Pclass'].value_counts()[2],

        titanic_data['Pclass'].value_counts()[3]

        ]

# print(sizes) # adds up to 891, which is the total number of passengers

plt.figure(figsize=(10,7))

plt.pie(sizes, labels=labels, autopct='%1.0f%%', shadow=True)

plt.axis('equal')

plt.show()
sns.countplot(x="Pclass", hue="Survived", data=titanic_data)
print('% of Survival by PClass')



print('Total PClass 1 survived :', titanic_data.Survived[titanic_data.Pclass == 1].sum())



print('% of PClass 1 survived :', titanic_data.Survived[titanic_data.Pclass == 1].sum()/titanic_data[titanic_data.Pclass == 1].Survived.count() * 100)



print('Total PClass 2 survived :', titanic_data.Survived[titanic_data.Pclass == 2].sum())



print('% of PClass 2 survived :', titanic_data.Survived[titanic_data.Pclass == 2].sum()/titanic_data[titanic_data.Pclass == 2].Survived.count() * 100)



print('Total PClass 3 survived :', titanic_data.Survived[titanic_data.Pclass == 3].sum())



print('Total PClass 3 survived :', titanic_data.Survived[titanic_data.Pclass == 3].sum()/titanic_data[titanic_data.Pclass == 3].Survived.count() * 100)
sns.catplot('Pclass','Survived', kind='point', data=titanic_data)
sns.factorplot("Pclass", "Survived", "Sex", data=titanic_data, kind="bar", palette="muted", legend=True)
sns.factorplot("Pclass", "Survived", "Salutation", data=titanic_data, kind='bar', palette="muted", legend=True)
sns.catplot('Pclass','Survived',hue='Sex',kind='point',data=titanic_data)

plt.show()
titanic_data.groupby(['Sex'])['Survived'].value_counts()
pd.crosstab(titanic_data.Sex, titanic_data.Survived, margins=True).style.background_gradient(cmap='autumn_r')
titanic_data.groupby(['Sex','Pclass'])['Survived'].value_counts()
sns.catplot(x='Sex', y='Survived', kind='bar', data=titanic_data);
print (titanic_data['Sex'].value_counts())

print (titanic_data['Sex'].value_counts()/len(titanic_data) * 100)
sns.catplot(x='Sex', col='Survived', kind='count', data=titanic_data);
print('Total women survived :', titanic_data[titanic_data.Sex == 'female'].Survived.sum())



print('Total men survived :', titanic_data[titanic_data.Sex == 'male'].Survived.sum())



print('% of women survived :', titanic_data[titanic_data.Sex == 'female'].Survived.sum()/titanic_data[titanic_data.Sex == 'female'].Survived.count() * 100)



print('% of men survived :', titanic_data[titanic_data.Sex == 'male'].Survived.sum()/titanic_data[titanic_data.Sex == 'male'].Survived.count() * 100)
f,ax=plt.subplots(1,2,figsize=(16,8))

titanic_data['Sex'].value_counts().plot.pie(explode=[0,0.05],autopct='%1.1f%%',ax=ax[0],shadow=True, startangle=90)

ax[0].set_title('Male Vs Female')

ax[0].set_ylabel('')



sns.countplot('Sex', hue='Survived', data=titanic_data,ax=ax[1])

ax[1].set_title('Sex Vs Survived')

plt.xlabel('Sex',fontsize =15)

#plt.ylabel('Total',fontsize =15)

plt.xticks(fontsize =13)

plt.yticks(fontsize =13)



plt.show()
print('The youngest passenger on board is:', round(titanic_data['Age'].min(),ndigits=2), 'months')

print('The oldest passenger on board is:', round(titanic_data['Age'].max(),ndigits=2), 'years')

print('The average age of passenger on board is:', round(titanic_data['Age'].mean(),ndigits=2), 'years')
f, ax = plt.subplots(1,2,figsize=(16,8))



titanic_data[titanic_data['Survived']==0].Age.plot.hist(ax=ax[0], bins = 20, edgecolor='black', color = 'red')

ax[0].set_title('Survived = 0')

x1=list(range(0,85,5))

ax[0].set_xticks(x1)



titanic_data[titanic_data['Survived']==1].Age.plot.hist(ax=ax[1], bins = 20, edgecolor='black', color = 'green')

ax[1].set_title('Survived = 1')

x2=list(range(0,85,5))

ax[1].set_xticks(x2)



plt.show()
# Check Dataset before converting to bins for Age

titanic_data.head(5)
#Converting to bins for Age

titanic_data['Age_bins'] = pd.cut(x=titanic_data['Age'], bins=[0,10,20,30,40,50,60,70,80,90])
#Check Age_bins column first 5 rows

titanic_data['Age_bins'].head(10)
# Check Dataset after converting to bins for Age (Notice new column added as 'Age_bins')

titanic_data.head(10)
#Let's Plot the Age bins into bargraph

plt.figure(figsize=(15,5))

sns.countplot(x="Age_bins", hue="Survived", data=titanic_data)

#plt.legend(loc='upper right')

plt.show()
titanic_data.loc[titanic_data['Age'] <= 16, 'Age_band'] = 0

titanic_data.loc[(titanic_data['Age'] > 16) & (titanic_data['Age'] <=32), 'Age_band'] = 1

titanic_data.loc[(titanic_data['Age'] > 32) & (titanic_data['Age'] <=48), 'Age_band'] = 2

titanic_data.loc[(titanic_data['Age'] > 48) & (titanic_data['Age'] <=64), 'Age_band'] = 3

titanic_data.loc[titanic_data['Age'] > 64, 'Age_band'] = 4
titanic_data.head()
titanic_data['Age_band'].value_counts().to_frame().style.background_gradient(cmap='summer')
sns.factorplot('Age_band','Survived', data = titanic_data, col = 'Pclass')
titanic_data.head()
# Check Dataset before converting to bins for Sibsp

titanic_data.head(10)
titanic_data['SibSp'].unique()
#Converting to bins for Sibsp

titanic_data['Sibsp_bins'] = pd.cut(x=titanic_data['SibSp'], bins=[-0.1,0,1,2,3,4,5,6,7,8])
# Check the Dataset after applying Sibsp bins method

titanic_data.head(10)
pd.crosstab(titanic_data.SibSp, titanic_data.Survived, margins=True).style.background_gradient(cmap='autumn_r')
#Let's Plot the Age bins into bargraph

plt.figure(figsize=(15,5))

sns.countplot(x="Sibsp_bins", hue="Survived", data=titanic_data)

#plt.legend(loc='upper right')

plt.show()
f,ax = plt.subplots(1,2,figsize=(20,8))

sns.barplot('SibSp','Survived', data= titanic_data,ax=ax[0])

ax[0].set_title('SibSp vs Survived')



sns.factorplot('SibSp','Survived', data= titanic_data,ax=ax[1])

ax[1].set_title('SibSp vs Survived')

plt.close(2)

plt.show()
pd.crosstab(titanic_data.Parch, titanic_data.Survived, margins=True).style.background_gradient(cmap='autumn_r')
titanic_data['Parch'].unique()
#Converting to bins for Parch

titanic_data['Parch_bins'] = pd.cut(x=titanic_data['Parch'], bins=[-0.1,0,1,2,3,4,5,6])
titanic_data.head()
#Let's Plot the Age bins into bargraph

plt.figure(figsize=(15,5))

sns.countplot(x="Parch_bins", hue="Survived", data=titanic_data)

plt.legend(loc='upper right')

plt.show()
# Check the Dataset before adding a new column

titanic_data.head()
titanic_data['Family'] = titanic_data.Parch + titanic_data.SibSp
# Check Dataset after adding a new column 'Family' 

titanic_data.head()
titanic_data['Is_Alone'] = titanic_data.Family == 0
# Check Dataset after adding a new column 'Is_Alone' 

titanic_data.head()
sns.catplot(x='Is_Alone', col='Survived', kind='count', data=titanic_data);
sns.catplot('Is_Alone','Survived', kind='point', data=titanic_data)
sns.distplot(titanic_data['Fare'])
print ('Passenger maximum fare paid :', titanic_data['Fare'].max())

print ('Passenger minimum fare paid :', titanic_data['Fare'].min())

print ('Passenger mid range fare paid :', titanic_data['Fare'].median())

print ('Passenger average fare paid :', titanic_data['Fare'].mean())
f,ax=plt.subplots(1,3,figsize=(20,8))

sns.distplot(titanic_data[titanic_data['Pclass']==1].Fare,ax=ax[0])

ax[0].set_title('Fares in Pclass 1')



sns.distplot(titanic_data[titanic_data['Pclass']==2].Fare,ax=ax[1])

ax[1].set_title('Fares in Pclass 2')



sns.distplot(titanic_data[titanic_data['Pclass']==3].Fare,ax=ax[2])

ax[2].set_title('Fares in Pclass 3')

plt.show()
plt.figure(figsize=(15,5))

plt.annotate(s="Rich Passengers",xy=(512.3292,0),xytext=(480,-0.05))

sns.boxplot(x=titanic_data['Fare'])

plt.show()
titanic_data['Fare'].quantile([0.1, 0.25, 0.5, 0.75])
Q1 = titanic_data['Fare'].quantile(0.25)

Q3 = titanic_data['Fare'].quantile(0.75)

IQR = Q3 - Q1

print('Q1 is :',Q1)

print('Q3 is :',Q3)

print('IQR is :',IQR)
# Printing 'Fare' amount in lower limit with Outliers

print([titanic_data['Fare']] < (Q1 - 1.5 * IQR))
# Printing 'Fare' amount in upper limit with Outliers

print([titanic_data['Fare']] > (Q3 + 1.5 * IQR))
titanic_data['Fare_Category'] = pd.cut(titanic_data['Fare'], bins=[-0.5,0,8,15,32,120,515], labels=['Free_Pass','Low','Mid','High_Mid','High','Rich_Pass'])
titanic_data.head()
#Let's Plot the Age bins into bargraph

plt.figure(figsize=(15,5))

sns.countplot(x="Fare_Category", hue="Survived", data=titanic_data)

#plt.legend(loc='upper right')

plt.show()
sns.factorplot('Fare_Category','Survived', kind='point', data=titanic_data)
sns.catplot('Fare_Category','Survived', hue='Sex', kind='bar', data=titanic_data)
titanic_data['Embarked'].replace(['S','C','Q'],

                                   ['Southampton','Cherbourg','Queenstown'],inplace=True)
titanic_data.head()
pd.crosstab(titanic_data.Embarked, titanic_data.Survived, margins=True).style.background_gradient(cmap='autumn_r')
#Let's Plot the Age bins into bargraph

plt.figure(figsize=(5,5))

sns.countplot(x="Embarked", hue = 'Pclass', data=titanic_data)

plt.show()
sns.catplot('Embarked','Survived', kind='point', data=titanic_data)
#titanic_data.to_csv('titanic_data_with_extra_columns.csv', index=False)
# Check the Dataset before removing unnecessary columns

titanic_data.head()
#Let's drop not needed columns

titanic_data.drop(['PassengerId','Name','Age','Ticket','Fare','Cabin','Age_bins','Sibsp_bins','Parch_bins'], axis=1, inplace=True)
# Check the Dataset after removing unnecessary columns

titanic_data.head()
# Check the Dataset for their datatypes

titanic_data.dtypes
#Check Dataset before changing categorical values into Numeric 

titanic_data.head()
# Import Libarary

from sklearn import preprocessing
LabelEncoder = preprocessing.LabelEncoder()
# Change numeric values for each columns

titanic_data['Sex'] = LabelEncoder.fit_transform(titanic_data['Sex'])

titanic_data['Embarked'] = LabelEncoder.fit_transform(titanic_data['Embarked'])

titanic_data['Salutation'] = LabelEncoder.fit_transform(titanic_data['Salutation'])

titanic_data['Is_Alone'] = LabelEncoder.fit_transform(titanic_data['Is_Alone'])

titanic_data['Fare_Category'] = LabelEncoder.fit_transform(titanic_data['Fare_Category'])
#Check Dataset after changing categorical values into Numeric 

titanic_data.head()
# Check the Dataset for their datatypes

titanic_data.dtypes
#data.corr()-->correlation matrix

sns.heatmap(titanic_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
# Importing all necessary ML Packages

from sklearn.linear_model import LogisticRegression # logistic Regression

from sklearn.model_selection import train_test_split # Training and Testing Data Split

from sklearn import metrics # accuracy measure

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # All report Metrix
# Split the Datase

train, test = train_test_split(titanic_data, test_size=0.25, random_state=0,stratify=titanic_data['Survived'])

train_X = train[train.columns[1:]]

test_X = train[train.columns[1:]]

train_Y = train[train.columns[:1]]

test_Y = train[train.columns[:1]]

X = titanic_data[titanic_data.columns[1:]]

Y = titanic_data['Survived']
X.head()
Y.head()
train_X.head()
train_Y.head()
test_X.head()
test_Y.head()
log_model = LogisticRegression()

log_model.fit(train_X,train_Y)

log_prediction = log_model.predict(test_X)

print('The accuracy of the Logistic Regression is :', metrics.accuracy_score(log_prediction, test_Y))
print(confusion_matrix(test_Y,log_prediction))
# See the confusion Merics graphicailly

sns.heatmap(confusion_matrix(test_Y,log_prediction), cmap='summer',annot=True,fmt='2.0f')

plt.title('Confusion matrix')

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
#The precision is the ratio tp / (tp + fp) [175/(175+48)] 

metrics.precision_score(test_Y,log_prediction)
#The recall is the ratio tp / (tp + fn) 175/(175+81))

metrics.recall_score(test_Y,log_prediction)
#let's Check the accuracy 

metrics.accuracy_score(test_Y,log_prediction)
# Let'see them in Summary Table

print(classification_report(test_Y,log_prediction))
log_prediction_proba = log_model.predict_proba(test_X)[::,1]

fpr, tpr, _ = metrics.roc_curve(test_Y, log_prediction_proba)

auc = metrics.roc_auc_score(test_Y, log_prediction_proba)

plt.plot(fpr,tpr,label="Titanic data, auc="+str(auc))

plt.legend(loc=4)

plt.show()