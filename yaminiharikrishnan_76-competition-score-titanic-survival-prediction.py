import pandas as pd

filename = "/kaggle/input/titanic/train.csv"

titanic_data = pd.read_csv(filename)

titanic_data.head()

import seaborn as sns 

import matplotlib as plt

titanic_data.head(10)
#Let's find how many rows and columns are present in the dataset

titanic_data.shape
sns.countplot(x="Survived",data=titanic_data)
titanic_data["Survived"].value_counts()
sns.countplot(x="Survived",hue = "Sex", data = titanic_data)
women = titanic_data.loc[titanic_data.Sex == 'female']["Survived"]

rate_women = sum(women)/len(women)



print("% of women who survived:", rate_women)
men = titanic_data.loc[titanic_data.Sex == 'male']["Survived"]

rate_men = sum(men)/len(men)



print("% of men who survived:", rate_men)
sns.countplot(x="Survived",hue = "Pclass", data = titanic_data)
titanic_data["Age"].plot.hist()
titanic_data["Fare"].plot.hist(bins=20, figsize=(10,5))
titanic_data.info()
sns.countplot(x="SibSp",data=titanic_data)
sns.countplot(x="Parch",data=titanic_data)
#Checking for null values in the dataset

titanic_data.isnull()

#False means data has a value, True means data does NOT have any value 
titanic_data.isnull().sum()
sns.boxplot(x="Pclass",y="Age",data=titanic_data)
titanic_data.head(5)
# We will drop Cabin column as it has 77 % data is missing! 

titanic_data.drop("Cabin",axis=1, inplace=True)
titanic_data.head(5)
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
titanic_data.isnull().sum()
sex=pd.get_dummies(titanic_data["Sex"])
sex=pd.get_dummies(titanic_data["Sex"],drop_first=True)
sex.head(5)
pd.get_dummies(titanic_data["Embarked"])
embark=pd.get_dummies(titanic_data["Embarked"],drop_first=True)
embark.head(5)

#drop first value because if Q and S is 0 that means automatically C is 1
Pcl=pd.get_dummies(titanic_data["Pclass"])
Pcl=pd.get_dummies(titanic_data["Pclass"],drop_first=True)
Pcl.head(5)
titanic_data=pd.concat([titanic_data,sex,embark,Pcl],axis=1)
titanic_data.head(5)
titanic_data.drop(['Sex','Embarked','Pclass','Name','Ticket','Fare'],axis=1, inplace=True)


titanic_data.head(5)
titanic_data.isnull().sum()
## Train Data 

X = titanic_data.drop("Survived",axis=1)

y = titanic_data["Survived"]

print(X) 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=42) 

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_train) 
from sklearn.metrics import classification_report

classification_report(y_train,predictions)

from sklearn.metrics import confusion_matrix 

confusion_matrix(y_train,predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_train,predictions)
# Beginning to import the test.csv file ....



import pandas as pd

filename1 = "/kaggle/input/titanic/test.csv"

test_data = pd.read_csv(filename1)

test_data.shape
test_data.head()
test_data.info()
test_data["Age"].plot.hist()
test_data["Fare"].plot.hist()
test_data["Fare"].plot.hist(bins=20, figsize=(10,5))
sns.countplot(x="SibSp",data=test_data)
sns.countplot(x="Parch",data=test_data)
test_data.isnull()

#False means data has a value, True means data does NOT have any value
test_data.isnull().sum()
sns.boxplot(x="Pclass",y="Age",data=test_data)
test_data['Age'].fillna(test_data['Age'].mean(),inplace=True)
test_data.drop("Cabin",axis=1, inplace=True)
test_data.isnull().sum()
sex=pd.get_dummies(test_data["Sex"],drop_first=True)
sex.head(5)
embark=pd.get_dummies(test_data["Embarked"],drop_first=True)
embark.head(5)
Pcl=pd.get_dummies(test_data["Pclass"],drop_first=True)
Pcl.head(5)
test_data=pd.concat([test_data,sex,embark,Pcl],axis=1)
test_data.head(5)

test_data.drop(['Sex','Embarked','Pclass','Name','Ticket','Fare'],axis=1, inplace=True)
test_data.head(5)
# Make sure after all the pre-processing you do on the test data the number of row enteries is 418 

# otherwise it will throw an error in the final submission file   

test_data.shape
test_data.isnull().sum()
# Now we are using the trained model over the test.csv file to predict the Survived passengers

test_predictions = logmodel.predict(test_data)

print(test_predictions)
#I am creating a DataFrame with the passengers ids and our prediction regarding whether they survived or not

submission = pd.DataFrame({'PassengerId':test_data['PassengerId'],'Survived':test_predictions})



#Lets take a quick look at the final data submission - first 5 rows

submission.head()
submission.shape
# Now I need to Convert DataFrame to a csv file as required by the Competition site to be uploaded



filename = 'Titanic Predictions 1.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)