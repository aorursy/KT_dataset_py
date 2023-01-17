import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")



# Feature Selection

#Univariate Selection

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



#Feature Importance

from sklearn.ensemble import ExtraTreesClassifier



#Importing alll the necessary packages to use the various classification algorithms

from sklearn.linear_model import LogisticRegression  # for Logistic Regression algorithm

from sklearn.model_selection import train_test_split #to split the dataset for training and testing

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.tree import DecisionTreeClassifier #for using Decision Tree Algoithm

from sklearn.metrics import confusion_matrix #Summarises Count values of Predictions

from sklearn.metrics import accuracy_score 

from sklearn.metrics import classification_report 
#Let's import the required Iris.csv dataset

df=pd.read_csv("../input/iris/Iris.csv")

df.head()
#Let's check for Missing Values

#Let's see how many categorical and numerical variables we have in our Dataset.

df.info()
#let's see the stats and understand the average of all the features ad the distribution of data in percentiles.

#If its a big data set you can use BokPlots to see the density of data located in percentile and also check for Outliers.

#Since Iris is a clean and a normalized Dataset there is very little that we can do with Exploratory Data Analysis.

df.describe()
# Since the 'Id' column is irrelevant to our Analysis we drop the column.

df=df.drop('Id',axis=1) 

df
#Let's try and see how each feature is correlated with one another.

correlation=df.corr()

print(correlation)
# Heat maps are great for making trends in this kind of data more readily apparent. 

# Particularly when the data is ordered and there is clustering.

plt.figure(figsize=(5,5))

sns.heatmap(correlation, annot=True,cmap="YlGnBu")

plt.show()
#A pairplot plot a pairwise relationships in a dataset.

sns.pairplot(df, size=2.5, hue="Species")

plt.show()
X = df.iloc[:,0:4]  #independent columns

Y = df.iloc[:,-1]    #target column i.e Species

print("Feature Variable X:","\n",X,"\n"*2,"Target Variable Y:","\n",Y, )
bestfeatures = SelectKBest(score_func=chi2, k=3)

fit = bestfeatures.fit(X,Y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']

print(featureScores)
featureScores.plot(kind='barh')

plt.show()
model = ExtraTreesClassifier()

model.fit(X,Y)
print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(5).plot(kind='barh')

plt.show()
train, test = train_test_split(df, test_size = 0.3)

print(train.shape)

print(test.shape)
train_X=train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_Y=train['Species']

test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_Y =test['Species']
SVM = svm.SVC()

SVM.fit(train_X,train_Y)

prediction=SVM.predict(test_X)

print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_Y))
Y_pred=SVM.predict(test_X)

Y_true=test_Y

cm=confusion_matrix(Y_true,Y_pred)

f, ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")

plt.ylabel("Y_true")

plt.show()
LR = LogisticRegression()

LR.fit(train_X,train_Y)

prediction=LR.predict(test_X)

print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction,test_Y))
Y_pred=LR.predict(test_X)

Y_true=test_Y

cm=confusion_matrix(Y_true,Y_pred)

f, ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")

plt.ylabel("Y_true")

plt.show()
DTC=DecisionTreeClassifier()

DTC.fit(train_X,train_Y)

prediction=DTC.predict(test_X)

print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,test_Y))
Y_pred=DTC.predict(test_X)

Y_true=test_Y

cm=confusion_matrix(Y_true,Y_pred)

f, ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")

plt.ylabel("Y_true")

plt.show()
KNN=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours for putting the new data into a class

KNN.fit(train_X,train_Y)

prediction=KNN.predict(test_X)

print('The accuracy of the KNN is',metrics.accuracy_score(prediction,test_Y))
Y_pred=KNN.predict(test_X)

Y_true=test_Y

cm=confusion_matrix(Y_true,Y_pred)

f, ax=plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("Y_pred")

plt.ylabel("Y_true")

plt.show()