%matplotlib inline



import re

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



sns.set()



from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12,8





from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import LinearSVC

from sklearn.ensemble import VotingClassifier



from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score



from sklearn.metrics import r2_score

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')



train_df.head()
test_df.head()
print(train_df.info())

print('_'*40 + '\n')

print(test_df.info())
print('Train dataset has only {} unique tickets'.format(len(train_df['Ticket'].unique())))

print('-'*40)

print('Test dataset has only {} unique tickets'.format(len(test_df['Ticket'].unique())))
new_train_df = train_df.copy()

new_test_df = test_df.copy()
def total_famil_members(df):

    return df['SibSp']+df['Parch']+1
def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    else:

        return ""
for data in [new_train_df, new_test_df]:

    

    data['Members'] = total_famil_members(data)

    data['Adjusted_Fare'] = data['Fare']/data['Members']



    data['Title'] = data['Name'].apply(get_title)
for data in [new_train_df, new_test_df]:

    

    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    data['Title'] = data['Title'].replace('Mlle', 'Miss')

    data['Title'] = data['Title'].replace('Ms', 'Miss')

    data['Title'] = data['Title'].replace('Mme', 'Mrs')

sns.countplot('Embarked', data=new_train_df)
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for data in [new_train_df, new_test_df]:   

    

    data['Title'] = data['Title'].map(title_mapping)

    data['Title'] = data['Title'].fillna(0)

    

    data['Embarked'] = data['Embarked'].fillna('S')    

    data['Sex'] = data['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

for data in [new_train_df, new_test_df]:

    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
new_train_df.head()
new_test_df.head()
new_train_df.info()
new_test_df.info()
fig,ax = plt.subplots(2,2)

sns.boxplot(x='Title', y='Age', data=new_train_df[new_train_df['Sex']==0], ax=ax[0][0])

sns.boxplot(x='Title', y='Age', data=new_train_df[new_train_df['Sex']==1], ax=ax[0][1])

sns.boxplot(x='Title', y='Age', data=new_test_df[new_test_df['Sex']==0], ax=ax[1][0])

sns.boxplot(x='Title', y='Age', data=new_test_df[new_test_df['Sex']==1], ax=ax[1][1])

ax[0][0].set_title(label='Female Age distribution(Train)')

ax[0][1].set_title(label='Male Age distribution(Train)')

ax[1][0].set_title(label='Female Age distribution(Test)')

ax[1][1].set_title(label='Male Age distribution(Test)')

plt.tight_layout()

plt.show()
data_train = new_train_df[~new_train_df['Age'].isnull()]

data_test = new_test_df[~new_test_df['Age'].isnull()]
data_train.head()
X_train = data_train[['Pclass','Sex','SibSp','Parch','Title']]

y_train = data_train['Age']



X_test = data_test[['Pclass','Sex','SibSp','Parch','Title']]

y_test = data_test['Age']



X_train.head()
model_age_prediction = RandomForestRegressor(n_estimators=900, max_depth=6, min_samples_leaf=0.001, random_state=100)

model_age_prediction.fit(X_train, y_train)
y_predict = model_age_prediction.predict(X_test)
fig1,ax1 = plt.subplots(1,2)

ax1[0].scatter(y_train, model_age_prediction.predict(X_train))

ax1[1].scatter(y_test, y_predict)



ax1[0].set_title('Train data vs predictions train data')

ax1[1].set_title('Test data vs predictions test data')
print('test score is: {}'.format(r2_score(y_test, y_predict)))

print('training score is: {}'.format(r2_score(y_train, model_age_prediction.predict(X_train))))
train_missing_predicted = model_age_prediction.predict(new_train_df[new_train_df['Age'].isnull()][['Pclass','Sex','SibSp','Parch','Title']])

test_missing_predicted = model_age_prediction.predict(new_test_df[new_test_df['Age'].isnull()][['Pclass','Sex','SibSp','Parch','Title']])
new_train_df['Age'][np.isnan(new_train_df['Age'])] = train_missing_predicted

new_test_df['Age'][np.isnan(new_test_df['Age'])] = test_missing_predicted
print(new_train_df.info())

print('_'*40)

print(new_test_df.info())
new_test_df[new_test_df['Adjusted_Fare'].isna()]
new_test_df['Adjusted_Fare'] = new_test_df.groupby('Pclass')['Adjusted_Fare'].transform(lambda x: x.fillna(x.median()))

new_test_df.info()
new_test_df['Fare'] = new_test_df['Adjusted_Fare']*new_test_df['Members']

new_test_df.info()
new_train_df.columns
data_model = new_train_df[['Survived', 'Pclass', 'Sex', 'Age', 'Embarked', 'Members', 'Adjusted_Fare', 'Title']]

data_model.head()
data_model.info()
sns.heatmap(data_model.corr(), annot=True, cmap='viridis')
train_X, test_X, train_y, test_y = train_test_split(data_model.drop(['Survived'], axis=1), data_model['Survived'],\

                                                    stratify=data_model['Survived'], random_state=123, test_size=0.25)
lr = LogisticRegression()

params_lr = {'C':[0.001, 0.01, 0.1, 1, 10, 100]}
lr_cv = GridSearchCV(lr, params_lr, n_jobs=-1, cv=5)
lr_cv.fit(train_X, train_y)
lr_cv_results = pd.DataFrame(lr_cv.cv_results_)

lr_cv_results.head()
lr_cv_best_model = pd.DataFrame({'Params':lr_cv.best_params_.values(), 'best_score':lr_cv.best_score_,\

                                'Train Score':accuracy_score(train_y, lr_cv.best_estimator_.predict(train_X)),\

                                'Test Score': accuracy_score(test_y, lr_cv.best_estimator_.predict(test_X))})
lr_cv_best_model.head()
svc = LinearSVC()

params_svc = params_lr
svc_cv = GridSearchCV(svc, params_svc, n_jobs=-1)
svc_cv.fit(train_X, train_y)
svc_cv_results = pd.DataFrame(svc_cv.cv_results_)

svc_cv_results.head()
svc_cv_best_model = pd.DataFrame({'Params':svc_cv.best_params_.values(), 'best_score':svc_cv.best_score_,\

                                'Train Score':accuracy_score(train_y, svc_cv.best_estimator_.predict(train_X)),\

                                'Test Score': accuracy_score(test_y, svc_cv.best_estimator_.predict(test_X))})
svc_cv_best_model.head()
max_depth =[]

min_samples_leaf = []

cv_rf_scores = []

test_roc_scores = []

train_roc_scores = []

test_acc_scores = []

train_acc_scores = []

for depth in [5,6,7,8,9]:

    for samples_leaf in [0.009, 0.01]:

        rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, max_features='log2', max_depth=depth, min_samples_leaf=samples_leaf, random_state=100)

        rf.fit(train_X, train_y)

        cv_rf_scores.append(cross_val_score(rf, train_X, train_y, cv=10).mean())

        max_depth.append(depth)

        min_samples_leaf.append(samples_leaf)

        test_roc_scores.append(roc_auc_score(test_y, rf.predict(test_X)))

        test_acc_scores.append(accuracy_score(test_y, rf.predict(test_X)))

        train_roc_scores.append(roc_auc_score(train_y, rf.predict(train_X)))

        train_acc_scores.append(accuracy_score(train_y, rf.predict(train_X)))
rf_cv_scores = pd.DataFrame({'Max_depth':max_depth, 'Min_samples_leaf':min_samples_leaf, 'CV_Scores':cv_rf_scores,\

                             'Test_roc_score':test_roc_scores, 'Train_roc_scores':train_roc_scores,\

                             'Test_acc_score':test_acc_scores, 'Train_acc_scores':train_acc_scores})
rf_cv_scores_sorted = rf_cv_scores.sort_values(by='CV_Scores').reset_index()

rf_cv_scores_sorted.head(18)
plt.plot(rf_cv_scores_sorted.sort_values(by='CV_Scores')['Test_acc_score'], label='Test accuracy score')

plt.plot(rf_cv_scores_sorted.sort_values(by='CV_Scores')['Train_acc_scores'], label='Train accuracy score')



plt.axvline(x=7, ls='--', c='k', label='best-model')

plt.legend()

plt.show()
classifiers = [('Logistic Regression',LogisticRegression(C=0.1)),\

               ('SVC', LinearSVC(C=0.01)),\

               ('Random Forest', RandomForestClassifier(n_estimators=400, n_jobs=-1, max_features='log2', max_depth=7, min_samples_leaf=0.01, random_state=100))]
vc = VotingClassifier(estimators=classifiers, n_jobs=-1)
vc.fit(train_X, train_y)
vc_scores = pd.DataFrame({'Best_score':cross_val_score(vc, train_X, train_y, cv=5).mean(),\

                          'Train Score':accuracy_score(train_y, vc.predict(train_X)),\

                          'Test Score': accuracy_score(test_y, vc.predict(test_X))}, index=[0])
vc_scores
sns.heatmap(confusion_matrix(test_y, vc.predict(test_X)), annot=True, fmt='.0f')
print('ROC_AUC score for VC is {}'.format(roc_auc_score(test_y, vc.predict(test_X))))

print('Accuracy score for VC is {}'.format(accuracy_score(test_y, vc.predict(test_X))))
rf1 = RandomForestClassifier(n_estimators=400, n_jobs=-1, max_features='log2', max_depth=7, min_samples_leaf=0.01, random_state=100)

rf1.fit(train_X, train_y)
sns.heatmap(confusion_matrix(test_y, rf1.predict(test_X)), annot=True, fmt='.0f')
print('ROC_AUC score for RF is {}'.format(roc_auc_score(test_y, rf1.predict(test_X))))

print('Accuracy score for RF is {}'.format(accuracy_score(test_y, rf1.predict(test_X))))
feature_importances = pd.Series(index=train_X.columns, data=rf1.feature_importances_)

sorted_feature_importances = feature_importances.sort_values()
sns.set()

sorted_feature_importances.plot(kind='barh')
results = rf1.predict(new_test_df[['Pclass', 'Sex', 'Age', 'Embarked', 'Members', 'Adjusted_Fare', 'Title']])
submission = pd.DataFrame({'PassengerId':new_test_df.PassengerId, 'Survived':results})
submission.to_csv('submission.csv', index=False)