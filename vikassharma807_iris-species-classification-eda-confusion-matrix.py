import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/Iris.csv')
df.head(5)
# checking for NaN values :
df.isnull().sum()
plt.figure(figsize=(15,6))
plt.xlabel('SepalLength in Cm' ,fontsize = 12)
plt.ylabel('PetalLength in Cm' ,fontsize = 12)
sns.stripplot(x = 'SepalLengthCm', y = 'PetalLengthCm', data = df,size = 7,jitter = False,palette='cool')
plt.figure(figsize=(15,10))
plt.subplots_adjust(hspace = .25)
plt.subplot(2,2,1)
sns.boxplot(x="Species", y="PetalLengthCm", data=df,palette='winter')
plt.subplot(2,2,2)
sns.violinplot(x="Species", y="PetalLengthCm", data=df, size=6,palette='spring')
plt.subplot(2,2,3)
sns.boxplot(x="Species", y="PetalWidthCm", data=df,palette='winter')
plt.subplot(2,2,4)
sns.violinplot(x="Species", y="PetalWidthCm", data=df, size=6,palette='spring')
plt.figure(figsize=(15,6))
plt.xlabel('SepalLength in Cm' ,fontsize = 12)
plt.ylabel('PetalLength in Cm' ,fontsize = 12)
sns.stripplot(x = 'SepalWidthCm', y = 'PetalWidthCm', data = df,size = 7,jitter = False,palette='spring')
plt.figure(figsize=(15,10))
plt.subplots_adjust(hspace = .25)
plt.subplot(2,2,1)
sns.boxplot(x="Species", y="SepalLengthCm", data=df,palette='winter')
plt.subplot(2,2,2)
sns.violinplot(x="Species", y="SepalLengthCm", data=df, size=6,palette='spring')
plt.subplot(2,2,3)
sns.boxplot(x="Species", y="SepalWidthCm", data=df,palette='winter')
plt.subplot(2,2,4)
sns.violinplot(x="Species", y="SepalWidthCm", data=df, size=6,palette='spring')
sns.pairplot(df.drop("Id", axis=1), hue="Species", size=3,palette='cool')
sns.heatmap(cbar=False,annot=True,data=df.corr(),cmap='spring')
from sklearn.model_selection import train_test_split
x = df.iloc[:,0:4].values
y = df.iloc[:,5].values
# Encoding the Categorical Data :
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
# spliting our Data Set :
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.4, random_state = 0)
print('xtrain : ')
print(xtrain)
print('ytrain : ')
print(ytrain)
print('xtest : ')
print(xtest)
print('ytest : ')
print(ytest)
# Manage data at same scale :
from sklearn.preprocessing import StandardScaler
scx=StandardScaler()
xtrain=scx.fit_transform(xtrain)
xtest=scx.transform(xtest)
from sklearn.linear_model import LogisticRegression
logistic_regressor = LogisticRegression(random_state=0)
logistic_regressor.fit(xtrain,ytrain)
log_predictions = logistic_regressor.predict(xtest)
log_predictions
logistic_accuracy = logistic_regressor.score(xtest,ytest)
logistic_accuracy
from sklearn.svm import SVC
svc = SVC()
svc.fit(xtrain,ytrain)
svc_predictions = svc.predict(xtest)
svc_predictions
svc_accuracy = svc.score(xtest,ytest)
svc_accuracy
from sklearn.naive_bayes import GaussianNB
NB = GaussianNB()
NB.fit(xtrain, ytrain)
NB_predictions = NB.predict(xtest)
NB_predictions
NB_accuracy = NB.score(xtest,ytest)
NB_accuracy
from sklearn.tree import DecisionTreeClassifier
dec_tree_classifier = DecisionTreeClassifier()
dec_tree_classifier.fit(xtrain, ytrain)
dec_tree_predictions = dec_tree_classifier.predict(xtest)
dec_tree_predictions
dec_tree_accuracy = dec_tree_classifier.score(xtest,ytest)
dec_tree_accuracy
from sklearn.ensemble import RandomForestClassifier
ran_forest_classifier = RandomForestClassifier()
ran_forest_classifier.fit(xtrain, ytrain)
rn_predictions = ran_forest_classifier.predict(xtest)
rn_predictions
ran_forest_accuracy = ran_forest_classifier.score(xtest,ytest)
ran_forest_accuracy
Models = ['Logistic Regression','Support Vector Machines','Naive Bayes','Decision Tree', 'Random Forest']
Accuracy = []

score = [logistic_accuracy,svc_accuracy, NB_accuracy, dec_tree_accuracy, ran_forest_accuracy]

for i in score :
    Accuracy.append(round(i*100))
Performance_of_Models = pd.DataFrame({'Model' : Models , 'Score' : Accuracy}).sort_values(by='Score', ascending=False)
Performance_of_Models
from sklearn.metrics import accuracy_score, confusion_matrix
matrix_1 = confusion_matrix(ytest, log_predictions) 
matrix_2 = confusion_matrix(ytest, svc_predictions) 
matrix_3 = confusion_matrix(ytest, rn_predictions) 
df_1 = pd.DataFrame(matrix_1,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

df_2 = pd.DataFrame(matrix_2,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

df_3 = pd.DataFrame(matrix_3,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])
plt.figure(figsize=(20,5))
plt.subplots_adjust(hspace = .25)
plt.subplot(1,3,1)
plt.title('confusion_matrix(logistic regression)')
sns.heatmap(df_1, annot=True,cmap='Blues')
plt.subplot(1,3,2)
plt.title('confusion_matrix(Support vector machines)')
sns.heatmap(df_2, annot=True,cmap='Greens')
plt.subplot(1,3,3)
plt.title('confusion_matrix(Random forest)')
sns.heatmap(df_3, annot=True,cmap='Reds')
plt.show()

