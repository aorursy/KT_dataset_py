# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import matplotlib.pyplot as plt
import seaborn as sns
#machine learning models 
from sklearn.ensemble import RandomForestClassifier

# obtain data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
compare = pd.read_csv("../input/gender_submission.csv")
#to display all columns
pd.set_option('display.max_columns', None)
train.head()
#summarize data 
train.info()
print("-----------")
test.info()
print("-----------")
train.describe()
#PassengerId, Name, Ticket are not likely affect the survival rate, therefore, drop
# Cabin has too many missing values (204 out of 891) has value, drop 
train = train.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)
test_id = test['PassengerId']
test = test.drop(["PassengerId","Name","Ticket","Cabin"], axis=1)

train.head()

#Embarked Analysis 
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(16,5)) #set grid for sns plot
sns.barplot('Embarked','Survived', data=train, ax=axis1).set_title('Embarked Distribute') #number of passengers from different Embarkment
sns.countplot(x='Survived', hue="Embarked", data=train, order=[1,0], ax=axis2).set_title('Survival Count-Embarked') #survival count by Embarkment

survival_rate_embarked = train[["Embarked", "Survived"]].groupby(["Embarked"],as_index=False).mean() #get survival rate
sns.barplot(x='Embarked', y='Survived', data=survival_rate_embarked, order=['S','C','Q'], ax=axis3).set_title('Survival Rate-Embarked')
#Here we can clearly see Embarked is a factor that affects the survival rate and passengers from Cherbourg has higher survival rate

# further dev: create two dummy variables for each port since there are 3 levels
#Embarked - Adjust for ML
#The port in which a passenger has embarked. C - Cherbourg, S - Southampton, Q = Queenstown
#revalue embarked into 1 - Cherbourg, 2 - Southampton, 3 = Queenstown
train["Embarked"] = train["Embarked"].replace(["C","S","Q"], [1,2,3])
train["Embarked"].head()

#make sdjustment to test data too 
test["Embarked"] = test["Embarked"].replace(["C","S","Q"], [1,2,3])

# Display the box plots on 3 separate rows and 1 column
fig, axes = plt.subplots(nrows=1, ncols=4)

# Generate a box plot of the fare prices for the First passenger class
train.loc[train['Pclass'] == 1].plot(ax=axes[0], y='Fare', kind='box', title = "Fare in Pclass1")
# Generate a box plot of the fare prices for the Second passenger class
train.loc[train['Pclass'] == 2].plot(ax=axes[1], y='Fare', kind='box', title = "Fare in Pclass2")

# Generate a box plot of the fare prices for the Third passenger class
train.loc[train['Pclass'] == 3].plot(ax=axes[2], y='Fare', kind='box', title = "Fare in Pclass3")

#overall Fare
train["Fare"].plot(ax=axes[3], y='Fare', kind='box', title = "Fare in Pclass3")

# Display the plot
plt.show()

# There are anormaly, remove them Upper+3std
#Remove Anormaly data to be continued

#Fare 
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(16,10)) #set grid for sns plot
sns.distplot(train["Fare"], ax=axis1) #distribution of fares
sns.boxplot(x=train["Survived"], y=train["Fare"], data=train, ax=axis2)
#from the plots we can infer the fare price has influence in survival rate, survivers are who paid higher fare in general
#Age
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
sns.distplot(train["Age"].dropna(), ax=axis1).set_title("Age Distribution")

#seperate kids from adults (Age<18) and elder (Age>60)
df_age = pd.DataFrame(index=range(len(train)), columns=["Survived","Age","Adulthood"])
df_age[["Survived","Age"]] = train[["Survived","Age"]]
df_age.loc[df_age['Age'] <18, 'Adulthood'] = 'Child'
df_age.loc[ (df_age['Age'] >= 18) & (df_age['Age'] < 60), 'Adulthood'] = 'Adult'
df_age.loc[ df_age['Age'] >= 60, 'Adulthood'] = 'Elder'
#aggregate
Adulthood = df_age[["Adulthood", "Survived"]].groupby(["Adulthood"],as_index=False).mean() #get survival rate
sns.barplot(x=Adulthood["Adulthood"], y=Adulthood["Survived"], data=Adulthood, ax=axis2).set_title("Survival Rate By Adulthood")
#Cleary, Children has higher survival rate then adult than elders
#Sex Analysis
fig, (axis1,axis2) = plt.subplots(1,2,figsize=(15,5))
sns.countplot(x='Survived', hue="Sex", data=train, order=[1,0], ax=axis1).set_title('Survival Count-Sex')
sex = train[["Survived","Sex"]].groupby(train["Sex"], as_index=False).mean()
sns.barplot(x=["Female", "Male"], y=sex["Survived"], data= sex).set_title('Survival Rate-Sex')

#Female has higher survival rate
# Sex adjust for Machine Learning
#convert male to 0 and female to 1
train["Sex"] = train["Sex"].replace(["male", "female"], [0,1])
test["Sex"] = test["Sex"].replace(["male", "female"], [0,1])
#Parch and SibSp Analysis
#My hypothesis is passengers with parents or child will have low survival rate because they tend to save the child(s) and parents
#While male passengers with spouse or siblings might have lower survival rate. Lets see

#convert data. Has parent(s) or child(s) = 1, has no parent(s) or child(s) = 0
fig, (axis1,axis2,axis3) = plt.subplots(1,3,figsize=(15,5))
train['Parch'].loc[(train['Parch']>0)] = 1
parch = train[["Survived","Parch"]].groupby(train["Parch"], as_index=False).mean()
sns.barplot(x=parch["Parch"], y=parch["Survived"], data=parch, ax=axis1).set_title("Survival Rate By Parch")

#convert data. Has spouse or siblings = 1, has no spouse or siblings = 0
train['SibSp'].loc[(train['SibSp']>0)] = 1
sibsp = train[["Survived","SibSp"]].groupby(train["SibSp"], as_index=False).mean()
sns.barplot(x=sibsp["SibSp"], y=sibsp["Survived"], data=sibsp, ax=axis2).set_title("Survival Rate By SibSp")

# the SibSo and Parch behaved very similarly
# now lets take a look at male vs Female in SibSp 
sibspsex = pd.DataFrame(train.groupby(["Sex", "SibSp"], as_index=False)["Survived"].agg("mean"))
#convert int to string to concat
sibspsex["Sex"] = sibspsex["Sex"].astype(str)
sibspsex["SibSp"] = sibspsex["SibSp"].astype(str)
sibspsex["Sibspsex"] = sibspsex[["Sex", "SibSp"]].apply(lambda x: "".join(x), axis=1)
sns.barplot(x=sibspsex["Sibspsex"], y=sibspsex["Survived"], data=sibspsex, ax=axis3).set_title("Survival Rate - Sex and SibSp")
#Female's survival rate is the still higher than male's 
#For males: those who has sibling(s) or spouse have higher survival rate
#For Female: those who has sibling(s) or spouse have lowe survival rate
# with above conclusion, I want to infer females had sacrificed their life to save spouse's life given the sociaety at that time is male dominant
sibspsex = pd.DataFrame(train.groupby(["Sex", "SibSp"], as_index=False)["Survived"].agg("mean"))
#convert int to string to concat
sibspsex["Sex"] = sibspsex["Sex"].astype(str)
sibspsex["SibSp"] = sibspsex["SibSp"].astype(str)
sibspsex["sibspsex"] = sibspsex[["Sex", "SibSp"]].apply(lambda x: "".join(x), axis=1)
sibspsex
train.head()
test.head()
#Fill NaNs in Embarked column with random port
ports = list(train["Embarked"].drop_duplicates().dropna())
print(ports)
train["Embarked"] = train["Embarked"].fillna(ports[random.randint(0,2)])
#Fill NaNs in Age column with random number from 1 to 80
train["Age"] = train["Age"].fillna(random.randint(1,80))
train["Age"].isnull().values.any()

#verify there is no na in the train df
print(f"Is there any missing value in train data frame? {train.isnull().values.any()}")

#fill na in test data as well
test["Embarked"] = test["Embarked"].fillna(ports[random.randint(0,2)])
test["Age"] = test["Age"].fillna(test["Age"].fillna(random.randint(1,80)))
test["Fare"] = test["Fare"].fillna(test["Fare"].median())
print(f"Is there any missing value in test data frame? {test.isnull().values.any()}")

# define training and testing data sets
Y_train = train["Survived"]
X_train = train.drop("Survived",axis=1)
X_test  = test
# Random Forests

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)
Y_pred
submission = pd.DataFrame({
        "PassengerId": test_id,
        "Survived": Y_pred
    })
submission.to_csv('titanic_prediction_submission.csv', index=False)
from sklearn.metrics import accuracy_score
accuracy_score(compare["Survived"], Y_pred)
submission.head()
#combine predicted result to actual result for benchmarking 
submission = submission.rename(columns={"Survived": "Survived_pred"})
result = pd.concat([compare, submission["Survived_pred"]], axis=1)
result.head()
result.loc[(result["Survived"] == result["Survived_pred"]), "Benchmark"] = 1
result.loc[(result["Survived"] != result["Survived_pred"]), "Benchmark"] = 0
accuracy = (result['Benchmark'].sum())/len(result)
accuracy





