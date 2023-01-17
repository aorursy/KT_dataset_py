import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler,StandardScaler

from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV,KFold,StratifiedKFold,cross_val_score

from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,recall_score,f1_score,roc_auc_score,auc,roc_curve

import re as re



from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import Perceptron

from sklearn.svm import SVC, LinearSVC

import warnings 

warnings.filterwarnings('ignore')



from sklearn.pipeline import Pipeline
train_data = pd.read_csv('../input/titanic-survival-prediction/train.csv')

test_data = pd.read_csv('../input/titanic-survival-prediction/test.csv',)

combined = [train_data,test_data]

train_data.head()
train_data.info()

print('-' * 50)

test_data.info()
sns.set(style="whitegrid")

sns.countplot(train_data['Survived'],data = train_data)

plt.show()
print('Percentage of null values in each column of train data:\n')

(train_data.isnull().sum() / train_data.shape[0]) * 100
print('Percentage of null values in each column of test data:\n')

(test_data.isnull().sum() / test_data.shape[0]) * 100
train_data.describe(include = 'all')
plt.figure(figsize = (10,8))

sns.heatmap(train_data.corr(),annot=True,cbar = True)

plt.show()
train_data.drop(['PassengerId','Cabin'],inplace=True,axis = 1)

test_data.drop(['PassengerId','Cabin'],inplace=True,axis = 1)
train_data.columns
train_data.loc[0,train_data.dtypes == object]
print(train_data.groupby(['Sex','Survived'])['Survived'].count())



sns.set(style="whitegrid")

sns.countplot(train_data['Survived'],hue = 'Sex',data = train_data)

plt.show()
train_data.drop(['Ticket'],axis = 1,inplace = True)

test_data.drop(['Ticket'],axis = 1,inplace = True)
print(train_data.groupby(['Embarked','Survived'])['Survived'].count())

sns.set(style="whitegrid")

sns.countplot(train_data['Survived'],hue = 'Embarked',data = train_data)

plt.show()
for dataset in combined:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_data['Title'], train_data['Sex']))

print('-' * 50)

print(pd.crosstab(test_data['Title'], test_data['Sex']))
for dataset in combined:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



print(train_data[['Title', 'Survived']].groupby(['Title'], as_index = False).mean())
for dataset in combined:

    age_avg 	   = dataset['Age'].mean()

    age_std 	   = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

    

train_data['CategoricalAge'] = pd.cut(train_data['Age'], 5)



print (train_data[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean())
for dataset in combined:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_data[['FamilySize','Survived']].groupby('FamilySize',as_index = False).mean()
for dataset in combined:

    dataset['FamilySizeCategory'] = 0

    for i in range(len(dataset)) : 

        if(dataset.loc[i,'FamilySize'] <= 4):

            dataset.loc[i,'FamilySizeCategory'] = 2

        elif((dataset.loc[i,'FamilySize'] > 4) & (dataset.loc[i,'FamilySize'] < 8)):

            dataset.loc[i,'FamilySizeCategory'] = 1

        else:

            dataset.loc[i,'FamilySizeCategory'] = 0           

    

print (train_data[['FamilySizeCategory', 'Survived']].groupby(['FamilySizeCategory'], as_index=False).mean())

train_data['FamilySizeCategory'].nunique()
for dataset in combined:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print (train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean())
for dataset in combined:

    dataset['Fare'] = dataset['Fare'].fillna(train_data['Fare'].median())

train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], 4)

print (train_data[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean())
for dataset in combined:

    dataset['Embarked'] = dataset['Embarked'].fillna(train_data['Embarked'].mode()[0])

train_data[['Embarked','Survived']].groupby(['Embarked'],as_index = False).mean()
train_data.info()
for dataset in combined:

    dataset['Sex'] = dataset['Sex'].map( {'female' : 0,'male' : 1} )

    

    dataset.loc[dataset['Age'] <= 16,'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32),'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48),'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64),'Age'] = 3

    dataset.loc[(dataset['Age'] > 64) & (dataset['Age'] <= 80),'Age'] = 4



    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)



    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )



    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    drop_columns = ['Name','SibSp','Parch','FamilySize']

    dataset.drop(drop_columns,axis = 1,inplace = True)



drop_col = ['CategoricalAge','CategoricalFare']

train_data.drop(drop_col,axis = 1,inplace = True)
train_data.FamilySizeCategory.nunique()
train_data = pd.get_dummies(train_data, prefix = ['Title'], columns = ['Title'])

test_data = pd.get_dummies(test_data, prefix = ['Title'], columns = ['Title'])
train_data.FamilySizeCategory.nunique()
plt.figure(figsize = (12,10))

sns.heatmap(train_data.corr(),annot=True,cbar = True)

plt.show()
x_train = train_data.iloc[:,1:]

y_train = train_data.iloc[:,0]

x_test = test_data

y_train
def best_classifier_using_kfold(clf,k = 10):

    best_accuracy = 0



    kf = StratifiedKFold(n_splits = k,random_state = 42,shuffle = False)

    best_pred,best_pred_y,best_train_x = [],[],[]

    

    accuracies, precisions, f1scores, recall_scores, roc_auc_scores = [],[],[],[],[]

    for train_index,test_index in kf.split(x_train,y_train):

        

        train_x,test_x = x_train.values[train_index],x_train.values[test_index]

        train_y,test_y = y_train.values[train_index],y_train.values[test_index]

        clf.fit(train_x,train_y)

        y_pred = clf.predict(test_x)

        if(accuracy_score(y_pred,test_y) >= best_accuracy):

            best_pred = y_pred

            best_pred_y = test_y

            best_train_x = test_x

        accuracies.append(accuracy_score(y_pred,test_y))

        precisions.append(precision_score(y_pred,test_y))

        f1scores.append(f1_score(y_pred,test_y))

        recall_scores.append(recall_score(y_pred,test_y))

        roc_auc_scores.append(roc_auc_score(y_pred,test_y))

    indices = ['Mean Accuracy','Mean of Precision Scores','Mean of F1 scores','Mean of Recall score','Mean of Roc-Auc score']



    eval = pd.DataFrame([np.mean(accuracies) * 100,np.mean(precisions) * 100,np.mean(f1scores)     * 100,np.mean(recall_scores) * 100,np.mean(roc_auc_scores) * 100],columns=['Value'],index=indices)



    cm = pd.DataFrame(confusion_matrix(best_pred,best_pred_y),index = ['Survived','Not Survived'],columns = ['Survived','Not Survived'])

    eval.index.name = 'Metrics'

    

    sns.set(font_scale=1.4) # for label size

    sns.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size

    plt.show()

    

    print(eval)

    print('cross-val-score',np.mean(cross_val_score(clf,x_train,y_train,cv = 10)) * 100)

    

    y_pred = clf.predict_proba(x_train)

    fpr,tpr,_ = roc_curve(y_train,y_pred[:,1])

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)        

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([-0.5, 1.05])

    plt.ylim([0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend()

    plt.title('ROC curve')

    plt.show()

model1 = LogisticRegression()

best_classifier_using_kfold(model1)
model2 = GaussianNB()

best_classifier_using_kfold(model2)
model3 = DecisionTreeClassifier()

best_classifier_using_kfold(model3)

model4 = KNeighborsClassifier()

best_classifier_using_kfold(model4)
model5 = MLPClassifier()

best_classifier_using_kfold(model5)
model6 = SVC(kernel = 'linear',probability=True)

best_classifier_using_kfold(model6)
model3_pg = {

    'criterion':['gini','entropy'],

    'splitter':['best','random'],

    'min_impurity_decrease': [0,0.25,0.5]

}
model5_pg = {

    'hidden_layer_sizes': [(30,40,30), (40,40,40,40,40)],

    'activation': ['tanh', 'relu','logistic'],

    'solver': ['sgd', 'adam'],

    'alpha': [0.0001, 0.0002, 0.0004,0.001,0.002,0.01],

    'learning_rate': ['constant','adaptive'],

}
leaf_size = list(range(1,50))

n_neighbors = list(range(1,30))

p=[1,2]

model4_pg = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)
model6_pg = {

     'C': [0.1, 1, 10, 100],  

     'gamma': [0.1, 0.01, 0.001, 0.0001], 

     'kernel': ['rbf', 'poly', 'sigmoid']  ,

     'probability': [True,False]

}
rscv = RandomizedSearchCV(model6, param_distributions = model6_pg, n_iter = 100, cv = 10, verbose=False, random_state=42, n_jobs = -1)

rscv.fit(x_train,y_train)
y = rscv.best_estimator_.predict(x_test)

y_pred = rscv.best_estimator_.predict_proba(x_test)

y[y == 1].size
count = 0

for i in range(0,y_pred.shape[0]):

    if((y_pred[i][1] >= 0.4) & (y_pred[i][1] <= 0.6)):

        count += 1

        y[i] = 0

count
rscv_submit = pd.DataFrame(y,columns = ['Survived'],index = [i + 892 for i in range(0,418)])

rscv_submit.index.name = 'PassengerId'

rscv_submit.to_csv('rscv_submission.csv')