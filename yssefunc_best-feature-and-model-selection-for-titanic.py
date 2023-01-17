#importing library
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd
# read train datasets
train = pd.read_csv('../input/train.csv')
#check the train dataset
train.head()
# check the tail of dataset
train.tail()
# Display all informations
train.info()
# to get information three quartiles, mean, count, minimum and maximum values and the standard deviation.
train.describe()
#now, we are checking start with a pairplot, and check for missing values
sns.heatmap(train.isnull(),cbar=False)
#Data Cleaning and Data Drop Process
train['Fare'] = train['Fare'].fillna(train['Fare'].dropna().median())
train['Age'] = train['Age'].fillna(train['Age'].dropna().median())
# Change to categoric column to numeric
train.loc[train['Sex']=='male','Sex']=0
train.loc[train['Sex']=='female','Sex']=1
# instead of nan values
train['Embarked']=train['Embarked'].fillna('S') 
# Change to categoric column to numeric
train.loc[train['Embarked']=='S','Embarked']=0
train.loc[train['Embarked']=='C','Embarked']=1
train.loc[train['Embarked']=='Q','Embarked']=2
#Drop unnecessary columns
drop_elements = ['Name','Cabin','Ticket']
train = train.drop(drop_elements, axis=1)
train.head()
#heatmap for train dataset
import matplotlib.pyplot as plt
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
#Look at the our train data again
train.head()
# Now, data is clean and read to a analyze
sns.heatmap(train.isnull(),cbar=False)
# how many people survived or not... %60 percent died %40 percent survived
fig = plt.figure(figsize=(18,6))
train.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
#Age with survived
plt.scatter(train.Survived, train.Age, alpha=0.1)
plt.title("Age with Survived")
plt.show()
#Count the pessenger class
fig = plt.figure(figsize=(18,6))
train.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
#Women Men together graph
female_color='pink'
train.Sex[train.Survived==1].value_counts(normalize=True).plot(kind='bar', alpha=0.5,color=[female_color,'b'])
plt.title("Sex of survived")
plt.show()
# which columns we have
train.columns

from sklearn.feature_selection import VarianceThreshold

mdlsel = VarianceThreshold(threshold=0.5)
mdlsel.fit(train)
ix = mdlsel.get_support()
#data1 = mdlsel.transform(train) 
data1 = pd.DataFrame(mdlsel.transform(train), columns = train.columns.values[ix])
data1.head()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

X = train.drop("Survived",axis=1)
y = train["Survived"]

mdlsel = SelectKBest(chi2, k=5) 
mdlsel.fit(X,y)
ix = mdlsel.get_support() 
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix]) # en iyi leri aldi... 7 tane...
data2.head(n=5)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression

X = train.drop("Survived",axis=1)
y = train["Survived"]

# Linear Model
linmdl = LogisticRegression()
linmdl.fit(X,y)
mdl = SelectFromModel(linmdl,prefit=True)
ix = mdl.get_support() 
data3 = pd.DataFrame(mdl.transform(X), columns = X.columns.values[ix]) 
data3.head(n=5)
#last feature selection
from sklearn.feature_selection import RFE

mdl = RFE(linmdl,n_features_to_select=5)
mdl.fit(X,y)
ix = mdl.get_support() 

data4 = pd.DataFrame(mdl.transform(X), columns = X.columns.values[ix]) 
data4.head(n=5)

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#input and output
target = train['Survived']
features = train[['Pclass','Sex','SibSp','Parch','Age']]

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)


classifier = linear_model.LogisticRegression()
classifier_ = classifier.fit(X_train,y_train)
target_predict=classifier_.predict(X_test)


print("Logistic Regression Score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)
poly = preprocessing.PolynomialFeatures(degree=2,include_bias=False)
poly_features = poly.fit_transform(features)

classifier_ = classifier.fit(X_train,y_train)
print("Polynomial Features: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = train['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = train[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

my_forest = RandomForestClassifier(max_depth=5, min_samples_split=10, n_estimators=500, random_state=5,criterion = 'entropy')


my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)

print("Random forest score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

target = train['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = train[data_features_names].values

#Build test and training test
X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.3,random_state=42)

decision_tree = tree.DecisionTreeClassifier(random_state=1,criterion = 'entropy',min_samples_split = 100)


decision_tree_ = decision_tree.fit(X_train,y_train)
target_predict=decision_tree_.predict(X_test)

print("Decision tree score: ",accuracy_score(y_test,target_predict))

from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))

import graphviz
generalized_tree = tree.DecisionTreeClassifier(
        random_state = 1,
        max_depth = 5,
        min_samples_split=2
)

generalized_tree_ = generalized_tree.fit(features,target)

print("Generalized tree score: ", generalized_tree_.score(features,target))




dot_data=tree.export_graphviz(generalized_tree_,feature_names=data_features_names,out_file=None)
graph = graphviz.Source(dot_data)
graph