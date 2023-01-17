import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')
train.head()
train.info()
train.describe()
plt.figure(figsize=(15,3))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.set_style('whitegrid')
sns.set_palette('bright')

sns.countplot(x='Survived',data=train)
sns.countplot(x='Survived',hue='Sex',data=train)
sns.countplot(x='Survived',hue='Pclass',data=train)
plt.figure(figsize=(13,5))
sns.distplot(train['Age'].dropna(),kde=False,bins=30)
g = sns.FacetGrid(train, hue="Sex", height=6, aspect=2, palette='coolwarm')
g = g.map(plt.hist, "Age", bins=20, alpha=0.8)
g.add_legend()
plt.figure(figsize=(13,5))
sns.countplot(x='SibSp',data=train)
plt.figure(figsize=(13,5))
sns.distplot(train['Fare'].dropna(),kde=False,bins=40)
# ploting bar plot for Embarked vs Survived
sns.barplot(x="Embarked",y="Survived",data= train)
plt.figure(figsize=(13, 5))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')
train.groupby('Pclass')['Age'].mean()
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 38

        elif Pclass == 2:
            return 30

        else:
            return 25

    else:
        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
plt.figure(figsize=(15,3))
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)
train.head()
train.dropna(inplace=True)
train.info()
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)
# sex
# embark
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
train = pd.concat([train,sex,embark],axis=1)
train.drop('PassengerId',axis=1,inplace=True)
train.head()
X=train.drop('Survived',axis=1)
y=train['Survived']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

def print_validation_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    f1_sc=f1_score(y_true,y_pred)
    print("Accuracy Score : "+ str(acc_sc))
    print("F1 Score : "+ str(f1_sc))

def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  
                cmap="Blues", cbar=False)
    plt.ylabel('true label')
    plt.xlabel('predicted label')
from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression(max_iter=1000)
clf_log.fit(X_train,y_train)
pred_log = clf_log.predict(X_test)
print_validation_report(y_test,pred_log)
acc_log=accuracy_score(y_test,pred_log)
plot_confusion_matrix(y_test,pred_log)
# FEATURE IMPORTANCE

importance = abs(clf_log.coef_[0])
coeffecients = pd.DataFrame(importance, X_train.columns)
coeffecients.columns = ['Coeffecient']
plt.figure(figsize=(15,4))
plt.bar(X_train.columns,importance)
plt.show()
from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski', metric_params=None, n_jobs=None,n_neighbors=7, p=2, weights='uniform')
clf_knn.fit(X_train,y_train)
pred_knn = clf_knn.predict(X_test)
print_validation_report(y_test,pred_knn)
acc_knn=accuracy_score(y_test,pred_knn)
plot_confusion_matrix(y_test,pred_knn)
error_rate = []
accuracy_scs =[]

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    accuracy_sc=accuracy_score(y_test,pred_i)
    accuracy_scs.append(accuracy_sc)
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(10,6))
plt.plot(range(1,40),accuracy_scs,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Accuracy Score vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy Score')
from sklearn.svm import SVC

clf_svc = SVC(kernel="linear")
clf_svc.fit(X_train,y_train)
pred_svc = clf_svc.predict(X_test)
print_validation_report(y_test,pred_svc)
acc_svc=accuracy_score(y_test,pred_svc)
plot_confusion_matrix(y_test,pred_svc)
from sklearn.naive_bayes import GaussianNB
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
pred_nb = clf_nb.predict(X_test)
print_validation_report(y_test,pred_nb)
acc_nb=accuracy_score(y_test,pred_nb)
plot_confusion_matrix(y_test,pred_nb)
from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier()
clf_rf.fit(X_train, y_train)
pred_rf = clf_rf.predict(X_test)
print_validation_report(y_test,pred_rf)
acc_rf=accuracy_score(y_test,pred_rf)
plot_confusion_matrix(y_test,pred_nb)
models = pd.DataFrame({
    'Model': ["LOGISTIC REGRESSION","K NEAREST NEIGHBORS","SUPPORT VECTOR MACHINE","NAIVE BAYES","RANDOM FOREST"],
    'Score': [acc_log,acc_knn,acc_svc,acc_nb,acc_rf
              ]})
models.sort_values(by='Score', ascending=False)
test= pd.read_csv("../input/titanic/test.csv")
test.info()
test['Fare'].fillna(test['Fare'].mean(),inplace=True)
test.info()
test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
test.drop('Cabin',axis=1,inplace=True)
test.dropna(inplace=True)
sex = pd.get_dummies(test['Sex'],drop_first=True)
embark = pd.get_dummies(test['Embarked'],drop_first=True)
test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)
test = pd.concat([test,sex,embark],axis=1)
passengerID = test['PassengerId']
test.drop('PassengerId',axis=1,inplace=True)
test.head()
test_copy=test.copy()
test= pd.read_csv("../input/titanic/test.csv")

prediction = clf_log.predict(test_copy)

output = pd.DataFrame({ 'PassengerId' : passengerID, 'Survived': prediction })
output.to_csv('submission.csv', index=False)
print(output)
