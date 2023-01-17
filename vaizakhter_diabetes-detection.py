import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix,log_loss

import matplotlib.pyplot as plt

import statistics 

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
df.head()
print("dimension of diabetes data: {}".format(df.shape))
print(df.groupby('Outcome').size())
sns.countplot(df['Outcome'],label="Count")
plt.plot(df['Outcome'],df['SkinThickness'])
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

ROWS, COLS = 2, 4

fig, ax = plt.subplots(ROWS, COLS, figsize=(18,8) )

row, col = 0, 0

for i, feature in enumerate(features):

    if col == COLS - 1:

        row += 1

    col = i % COLS

    

    df[feature].hist(bins=35, color='green', alpha=0.5, ax=ax[row, col]).set_title(feature)  
df.Glucose.replace(0, np.nan, inplace=True)

df.Glucose.replace(np.nan, df['Glucose'].median(), inplace=True)

df.BloodPressure.replace(0, np.nan, inplace=True)

df.BloodPressure.replace(np.nan, df['BloodPressure'].median(), inplace=True)

df.SkinThickness.replace(0, np.nan, inplace=True)

df.SkinThickness.replace(np.nan, df['SkinThickness'].median(), inplace=True)

df.Insulin.replace(0, np.nan, inplace=True)

df.Insulin.replace(np.nan, df['Insulin'].median(), inplace=True)

df.BMI.replace(0, np.nan, inplace=True)

df.BMI.replace(np.nan, df['BMI'].median(), inplace=True)
features = [ 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

ROWS, COLS = 2, 3

fig, ax = plt.subplots(ROWS, COLS, figsize=(18,8) )

row, col = 0, 0

for i, feature in enumerate(features):

    if col == COLS - 1:

        row += 1

    col = i % COLS

    

    df[feature].hist(bins=35, color='green', alpha=0.5, ax=ax[row, col]).set_title(feature) 
X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'Outcome'], df['Outcome'], stratify=df['Outcome'], random_state=66)
training_accuracy = []

test_accuracy = []

neighbors_number = range(1, 11)

for n_neighbors in neighbors_number:

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn.fit(X_train, y_train)

    training_accuracy.append(knn.score(X_train, y_train))

    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(neighbors_number, training_accuracy, label="training accuracy")

plt.plot(neighbors_number, test_accuracy, label="test accuracy")

plt.ylabel("Accuracy")

plt.xlabel("n_neighbors")

plt.legend()

plt.savefig('knn_compare_model')
knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train, y_train)

print('Accuracy of K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))

print('Accuracy of K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
model = RandomForestClassifier(n_estimators=1000)

model.fit(X_train, y_train)

predictions = cross_val_predict(model, X_test, y_test, cv=5)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
kf = StratifiedKFold(n_splits=5, shuffle=True)

def baseline_report(model, X_train, X_test, y_train, y_test, name):

    model.fit(X_train, y_train)

    accuracy     = np.mean(cross_val_score(model, X_train, y_train, cv=kf, scoring='accuracy'))

    precision    = np.mean(cross_val_score(model, X_train, y_train, cv=kf, scoring='precision'))

    recall       = np.mean(cross_val_score(model, X_train, y_train, cv=kf, scoring='recall'))

    f1score      = np.mean(cross_val_score(model, X_train, y_train, cv=kf, scoring='f1'))

    rocauc       = np.mean(cross_val_score(model, X_train, y_train, cv=kf, scoring='roc_auc'))

    y_pred = model.predict(X_test)

    logloss      = log_loss(y_test, y_pred)   # SVC & LinearSVC unable to use cvs



    df_model = pd.DataFrame({'model'        : [name],

                             'accuracy'     : [accuracy],

                             'precision'    : [precision],

                             'recall'       : [recall],

                             'f1score'      : [f1score],

                             'rocauc'       : [rocauc],

                             'logloss'      : [logloss],

                             'timetaken'    : [0]       })   # timetaken: to be used for comparison later

    return df_model
gnb = GaussianNB()

logit = LogisticRegression()

knn = KNeighborsClassifier()

decisiontree = DecisionTreeClassifier()

randomforest = RandomForestClassifier()

svc = SVC()
df_models = pd.concat([baseline_report(gnb, X_train, X_test, y_train, y_test, 'GaussianNB'),

                       baseline_report(logit, X_train, X_test, y_train, y_test, 'LogisticRegression'),

                       baseline_report(knn, X_train, X_test, y_train, y_test, 'KNN'),

                       baseline_report(decisiontree, X_train, X_test, y_train, y_test, 'DecisionTree'),

                       baseline_report(randomforest, X_train, X_test, y_train, y_test, 'RandomForest'),

                       baseline_report(svc, X_train, X_test, y_train, y_test, 'SVC'),

                       ], axis=0).reset_index()

df_models = df_models.drop('index', axis=1)

df_models