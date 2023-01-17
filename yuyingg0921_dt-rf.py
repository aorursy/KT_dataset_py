import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn import tree
df = pd.read_csv('../input/train.csv')

df.head()
df = df.drop('PassengerId', axis = 1)

df = df.drop('Name', axis = 1)

df = df.drop('Ticket', axis = 1)
df['Survived'].value_counts()
df.isnull().sum()
age = df['Age']

avgage = age.sum(axis = 0, skipna = True) /len(age)

df['Age'].fillna(avgage, inplace = True)
df = df.drop('Cabin', axis = 1)

df = df.drop('Embarked', axis = 1)
df.isnull().sum()
sex_map = {'male' : 0, 'female' :1}

df['Sex'] = df['Sex'].map(sex_map)

df.head()
df_train = df[:700]

df_train.head()
df_test = df[700:]

df_test.head()
y = df_train['Survived'].values

df_train = df_train.drop('Survived', 1)



y
import graphviz 

dtree=tree.DecisionTreeClassifier(max_depth=4)

dtree=dtree.fit(df_train,y)

dot_data = tree.export_graphviz(dtree, 

                filled=True, 

                feature_names=list(df_train),

                class_names=['die','survive'],

                special_characters=True)

graph = graphviz.Source(dot_data)  



'''

dot_data = StringIO()

export_graphviz(dtree, 

                out_file=dot_data,  

                filled=True, 

                feature_names=list(df_train),

                class_names=['die','survive'],

                special_characters=True)



graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

graph.write_pdf("tree.pdf")

'''
graph
dtree.feature_importances_
y_test = df_test['Survived'].values

X_test = df_test.drop('Survived', 1)



y_predict = dtree.predict(X_test)



y_predict
from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_predict)
y_test
acc = 0

for i in range(len(y_test)):

    if y_predict[i] == y_test[i]:

        acc += 1

acc/len(y_test) ## must be the same as the above result
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(bootstrap=True, n_estimators=1000, max_depth=4)

clf.fit(df_train, y)  
y_predict = clf.predict(X_test)

accuracy_score(y_test, y_predict)
from sklearn import model_selection, metrics
def scorer(model, X, y):

    preds = model.predict(X)

    return metrics.accuracy_score(y, preds)
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]  ## try different n_estimators

cv_results = []



for estimator in n_estimators:

    rf = RandomForestClassifier(n_estimators=estimator)

    acc = model_selection.cross_val_score(rf, df_train, y, cv=10, scoring=scorer)

    cv_results.append(acc.mean())
line1= plt.plot(n_estimators, cv_results, 'b', label="cross validated accuracy")

plt.ylabel('accuracy')

plt.xlabel('n_estimators')

plt.legend()

plt.show()
best_n_estimators = n_estimators[cv_results.index(max(cv_results))]

print ("best_n_estimators: ", best_n_estimators)

print ("best accuracy: ", max(cv_results))
cv_results = []

max_depths = np.linspace(1, 32, 32, endpoint=True)  ## try different max_depths

for max_depth in max_depths:

    rf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth=max_depth)

    acc = model_selection.cross_val_score(rf, df_train, y, cv=10, scoring=scorer)

    cv_results.append(acc.mean())
line1= plt.plot(max_depths, cv_results, 'b', label="cross validated accuracy")

plt.ylabel('accuracy')

plt.xlabel('max_depths')

plt.legend()

plt.show()
best_max_depths = max_depths[cv_results.index(max(cv_results))]

print ("best_max_depths:", best_max_depths)

print ("best accuracy: ", max(cv_results))
cv_results = []

min_samples_splits = [2,3,5,10,20,30,40,50,60,70,80]  ## try different min_samples_splits

for min_samples_split in min_samples_splits:

    rf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth=best_max_depths,min_samples_split=min_samples_split)

    acc = model_selection.cross_val_score(rf, df_train, y, cv=10, scoring=scorer)

    cv_results.append(acc.mean())
line1= plt.plot(min_samples_splits, cv_results, 'b', label="cross validated accuracy")

plt.ylabel('accuracy')

plt.xlabel('min_samples_splits')

plt.legend()

plt.show()
best_min_samples_splits = min_samples_splits[cv_results.index(max(cv_results))]

print ("best_min_samples_splits: ", best_min_samples_splits)

print ("best accuracy", max(cv_results))
rf = RandomForestClassifier(n_estimators = best_n_estimators, max_depth=best_max_depths,min_samples_split=best_min_samples_splits)

rf.fit(df_train, y)

train_pred = rf.predict(df_train)

print ("Training acc:", accuracy_score(y, train_pred))

y_pred = rf.predict(X_test)

print ("Testing acc:", accuracy_score(y_test, y_pred))