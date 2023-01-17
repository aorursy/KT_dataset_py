#Load data
import pandas as pd
import seaborn as sns

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Drop features we are not going to use
train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

#Look at the first 10 rows of our training data
train.head(10)
#Convert ['male','female'] to [1,0] so that our decision tree can be built
for df in [train,test]:
    df['Sex_binary'] = df['Sex'].map({'male':1,'female':0})
    
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Look at the first 10 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train[features].head(10)
#Display first 10 target variables
train[target].head(10).values
sns.factorplot('Pclass','Survived', order = [1,2,3], data = train, size = 5)
# logistic regression 

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

#Fit our model using the training features and the training target values

logreg.fit(train[features], train[target])

logreg.score(train[features], train[target])
predictions = logreg.predict(test[features])

#Display our predictions - they are either 0 or 1 for each training instance 
#depending on whether our algorithm believes the person survived or not.

predictions
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 10 rows
submission.head(10)
#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)