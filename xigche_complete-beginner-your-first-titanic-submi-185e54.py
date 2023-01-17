#Load data
import pandas as pd

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

#Drop features we are not going to use
train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)

#Look at the first 3 rows of our training data
train.head(5)
#Convert ['male','female'] to [1,0] so that our decision tree can be built
for df in [train,test]:
    df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    
#Fill in missing age values with 0 (presuming they are a baby if they do not have a listed age)
train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)

#Select feature column names and target variable we are going to use for training
features = ['Pclass','Age','Sex_binary']
target = 'Survived'

#Look at the first 3 rows (we have over 800 total rows) of our training data.; 
#This is input which our classifier will use as an input.
train[features].head(5)
#Display first 3 target variables
train[target].head(3).values
from sklearn.tree import DecisionTreeClassifier

#Create classifier object with default hyperparameters
clf = DecisionTreeClassifier(max_depth=3)  

#Fit our classifier using the training features and the training target values
clf.fit(train[features],train[target]) 
#Create decision tree ".dot" file

#Remove each '#' below to uncomment the two lines and export the file.
#from sklearn.tr#ee import export_graphviz
#export_graphviz(#clf,out_file='titanic_tree.dot',feature_names=features,rounded=True,filled=True,class_names=['Survived','Did not Survive'])
#Display decision tree

#Blue on a node or leaf means the tree thinks the person did not survive
#Orange on a node or leaf means that tree thinks that the person did survive

#In Chrome, to zoom in press control +. To zoom out, press control -. If you are on a Mac, use Command.

#Remove each '#' below to run the two lines below.
#from IPython.core.display import Image, display
#display(Image('titanic_tree.png', width=1900, unconfined=True))
#Make predictions using the features from the test data set
predictions = clf.predict(test[features])

#Display our predictions - they are either 0 or 1 for each training instance 
#depending on whether our algorithm believes the person survived or not.
predictions
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()
import seaborn as sns

sns.lmplot('PassengerId','Survived',data=submission)
#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'Titanic Predictions 1.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)