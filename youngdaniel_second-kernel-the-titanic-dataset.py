import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.Survived.dtype
test['Survived'] = np.nan
dataset = pd.concat([train, test], axis = 0,sort = False)
dataset
# Because we used np.nan for missing values in test, concat changes the dtype of Survived to float.
# Should this be int8 or int64? Seems like int8 is more appropriate. Previous run says int64.
dataset.isnull().sum()
dataset['Cabin-missing'] = dataset.Cabin.isnull()
g = sns.catplot(data = dataset, col = 'Cabin-missing', x = 'Pclass', kind = 'count' )
g.set_axis_labels('Passenger Class', 'Count')
dataset[dataset['Cabin'].notnull()].Pclass.value_counts()
dataset['Age-missing'] = dataset.Age.isnull()
sns.countplot(data = dataset[dataset.Age.isnull()], x = 'Survived')
dataset[dataset.Embarked.isnull()]
dataset.loc[[61,829], 'Embarked'] = 'S'
dataset[dataset.Fare.isnull()]
dataset.Sex = dataset.Sex.map(lambda x: 1 if x == 'male' else 0)

dataset['Southampton'] = dataset.Embarked.map(lambda x: 1 if x == 'S' else 0)
dataset['Cherbourg'] = dataset.Embarked.map(lambda x: 1 if x == 'C' else 0)
dataset['FirstClass'] = dataset.Pclass.map(lambda x: 1 if x == 1 else 0)
dataset['SecondClass'] = dataset.Pclass.map(lambda x: 1 if x == 2 else 0)
dataset[dataset.Name.map(lambda x: 'Andersson' in x)]
dataset[dataset.Ticket == '3701']
sns.catplot(data = dataset[dataset.Survived.notnull()], col = 'Survived', x = 'Sex', kind = 'count') 
sns.heatmap(dataset[['Pclass', 'Fare']].corr(), annot = True)
from sklearn.preprocessing import Imputer

# Fill in missing Age data
imputer = Imputer(missing_values = 'NaN', strategy = 'median')
imputer = imputer.fit(dataset[['PassengerId', 'Age']])
dataset[['PassengerId', 'Age']] = imputer.transform(dataset[['PassengerId', 'Age']])

# Fill in missing Fare data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean')
impute = imputer.fit(dataset.loc[dataset.Pclass == 3, ['PassengerId', 'Fare']])
dataset[['PassengerId', 'Fare']] = imputer.transform(dataset[['PassengerId', 'Fare']])
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import StandardScaler

scorer = make_scorer(accuracy_score, greater_is_better = True)

# we split into two versions: set0 does not have passenger class encoded, set1 does.
train_set0 = dataset[dataset.Survived.notnull()].drop(columns = ['Ticket', 'FirstClass', 'SecondClass',
                                                                 'Cabin','PassengerId', 'Survived', 
                                                                 'Name', 'Embarked'])
train_set1 = dataset[dataset.Survived.notnull()].drop(columns = ['Ticket', 'PassengerId', 'Pclass', 'Cabin', 
                                                                 'Survived', 'Embarked', 'Name'])
train_labels = dataset[dataset.Survived.notnull()].Survived
test_set0 = dataset[dataset.Survived.isnull()].drop(columns = ['Ticket', 'FirstClass', 'SecondClass', 
                                                               'Cabin', 'PassengerId', 'Survived', 
                                                               'Name', 'Embarked'])
test_set1 = dataset[dataset.Survived.isnull()].drop(columns = ['Ticket', 'PassengerId', 'Pclass', 'Cabin', 
                                                                'Survived', 'Embarked', 'Name'])
test_ids = dataset[dataset.Survived.isnull()].PassengerId
features0 = list(train_set0.columns)
features1 = list(train_set1.columns)



# Some models require feature scaling, so we will apply feature scaling 
scaler = StandardScaler()
train_set0 = scaler.fit_transform(train_set0)
test_set0 = scaler.transform(test_set0)
train_set1 = scaler.fit_transform(train_set1)
test_set1 = scaler.transform(test_set1)
# The labels are binary so scaling is unnecessary
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

model = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
cv_score = cross_val_score(model, train_set1, train_labels, cv = 10, scoring = scorer)
cv_score.mean(), cv_score.std()
model.fit(train_set1, train_labels)
predicted_labels = model.predict(test_set1)
submission = pd.DataFrame(data = {'PassengerId': test_ids, 'Survived': predicted_labels})
submission = submission.astype('int64')
submission.to_csv('RF_initial_submission.csv', index = False)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# we've already imported the RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
def cv_model(train_set, train_labels, model, name, model_results):
    cv_scores = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)
    model_results = model_results.append(pd.DataFrame({'model': name,
                                                       'cv_mean': cv_scores.mean(), 
                                                       'cv_std': cv_scores.std()}, 
                                                      index = [0]), 
                                         ignore_index = True)
    return model_results
import warnings
warnings.filterwarnings('ignore', category = FutureWarning)

model_results = pd.DataFrame(columns = ['model', 'cv_mean', 'cv_std'])

model_results = cv_model(train_set1, train_labels, 
                         LogisticRegression(), 'LR', model_results)
model_results = cv_model(train_set1, train_labels, 
                         GaussianNB(),  'GNB', model_results)
model_results = cv_model(train_set1, train_labels, 
                         SVC(), 'SVC', model_results)
model_results = cv_model(train_set1, train_labels, 
                         LinearSVC(), 'LSVC', model_results)
model_results = cv_model(train_set1, train_labels, 
                         KNeighborsClassifier(5), 'kNN-5', model_results)
model_results = cv_model(train_set1, train_labels,
                         KNeighborsClassifier(10), 'kNN-10', model_results)
model_results = cv_model(train_set1, train_labels,
                         KNeighborsClassifier(15), 'kNN-15', model_results)
model_results = cv_model(train_set1, train_labels,
                         KNeighborsClassifier(20), 'kNN-20', model_results)
model_results = cv_model(train_set1, train_labels,
                         DecisionTreeClassifier(), 'Tree', model_results)
model_results = cv_model(train_set1, train_labels,
                         RandomForestClassifier(n_estimators = 100), 'RF', model_results)
model_results = cv_model(train_set1, train_labels,
                         LinearDiscriminantAnalysis(), 'LDA', model_results)
model_results.set_index('model', inplace = True)
model_results = model_results.sort_values(by = 'cv_mean')
model_results['cv_mean'].plot.bar(color = 'orange', yerr = list(model_results['cv_std']),
                                  edgecolor = 'k', linewidth = 1.5)
model_results.reset_index(inplace = True)
plt.title('Model Score (Accuracy) Results')
svc_model = SVC()
svc_model.fit(train_set1, train_labels)
predicted_labels = svc_model.predict(test_set1)
submission = pd.DataFrame(data = {'PassengerId': test_ids, 'Survived': predicted_labels})
submission = submission.astype('int64')
submission.to_csv('SVC_submission.csv', index = False)

lsvc_model = LinearSVC(max_iter = 10000)
lsvc_model.fit(train_set1, train_labels)
predicted_labels = svc_model.predict(test_set1)
submission = pd.DataFrame(data = {'PassengerId': test_ids, 'Survived': predicted_labels})
submission = submission.astype('int64')
submission.to_csv('LSVC_submission.csv', index = False)

knn15_model = KNeighborsClassifier(15)
knn15_model.fit(train_set1, train_labels)
predicted_labels = knn15_model.predict(test_set1)
submission = pd.DataFrame(data = {'PassengerId': test_ids, 'Survived': predicted_labels})
submission = submission.astype('int64')
submission.to_csv('kNN15_submission.csv', index = False)

feature_importances = pd.DataFrame({'features': features1, 'importance': model.feature_importances_})
feature_importances = feature_importances.set_index('features').sort_values(by = 'importance')
feature_importances.plot.barh(color = 'blue', edgecolor = 'k', linewidth = 2, legend = False)
plt.title('Feature Importances')
feature_importances.reset_index(inplace = True)
# model without fare data
train_set_nofare = dataset[dataset.Survived.notnull()].drop(columns = ['Ticket', 'Cabin', 'Embarked', 
                                                                'PassengerId', 'Survived', 'Name', 'Fare'])
train_labels_nofare = dataset[dataset.Survived.notnull()].Survived
test_set_nofare = dataset[dataset.Survived.isnull()].drop(columns = ['Ticket', 'Cabin', 'Embarked',
                                                              'PassengerId', 'Survived', 'Name', 'Fare'])
test_ids_nofare = dataset[dataset.Survived.isnull()].PassengerId
features_nofare = list(train_set_nofare.columns)

model = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
cv_score = cross_val_score(model, train_set, train_labels, cv = 10, scoring = scorer)
cv_score.mean(), cv_score.std()
model.fit(train_set_nofare, train_labels_nofare)
predicted_labels_nofare = model.predict(test_set_nofare)
feature_importances_nofare = pd.DataFrame({'features': features_nofare, 
                                           'importance': model.feature_importances_})
feature_importances_nofare
submission = pd.DataFrame({'PassengerId': test_ids_nofare, 'Survived': predicted_labels_nofare})
submission = submission.astype('int64')
submission.to_csv('RF_nofare_submission.csv', index = False)

# no fares scores 0.72727















