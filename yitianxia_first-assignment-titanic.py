#Load data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Drop features we are not going to use
train_data = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin'],axis=1)
test_data = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin'],axis=1)

#Look at the first 3 rows of our training data
train_data.head(10)

#Look at the detail of variables in our training data
train_data.describe()

#Plot a pie chart to see the survival rate
train_data['Survived'].value_counts().plot.pie(autopct = '%1.2f%%')
#Make a correlation matrix heat map to see the correlation between variables
train_corr = train_data.corr()
a = plt.subplots(figsize=(15,9))
a = sns.heatmap(train_corr, vmin=-1, vmax=1 , annot=True , square=True)
# Survival rate vs. Pclass
train_data[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar()

# Survival rate vs. Age
facet = sns.FacetGrid(train_data, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train_data['Age'].max()))
facet.add_legend()
# Survival rate vs. sex
train_data[['Sex','Survived']].groupby(['Sex']).mean().plot.bar()
# Survival rate vs. embarked
sns.countplot('Embarked',hue='Survived',data=train_data)
#Convert ['male','female'] to [1,0] so that our decision tree can be built
for df in [train_data,test_data]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})

#Use mode to fill in missing values of Embarked and Convert ['S','C','Q'] to [-1,0,1] so that our decision tree can be built
train_data['Embarked'].fillna(train_data['Embarked'].mode().iloc[0], inplace=True)
test_data['Embarked'].fillna(test_data['Embarked'].mode().iloc[0], inplace=True)
for df in [train_data,test_data]:
    df['Embarked_numeric']=df['Embarked'].map({'S':-1,'C':0,'Q':1})

#Use 30 to fill in missing values of Age since the average age of both Train and Test data is around 30
train_data.Age[train_data.Age.isnull()] = 30
test_data.Age[test_data.Age.isnull()] = 30

#Check the data
train_data.info()
test_data.info()
#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary', 'Embarked_numeric']
target = 'Survived'
from sklearn.tree import DecisionTreeClassifier

#Create classifier object with default hyperparameters
clf = DecisionTreeClassifier()  

#Fit our classifier using the training features and the training target values
clf.fit(train_data[features],train_data[target]) 
#Make predictions using the features from the test data set
predictions = clf.predict(test_data[features])

#Display our predictions - they are either 0 or 1 for each training instance 
#depending on whether our algorithm believes the person survived or not.
predictions
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()
#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)