#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

#sns.set()

sns.set_style('whitegrid')
df = pd.read_csv('../input/voice.csv')

df.shape
df.head(1)
df.columns
corr = df.corr()

corr
corr['meanfreq'].drop('meanfreq', axis=0).sort_values(ascending=False)
corr['meanfreq'] == corr['meanfreq'].drop('meanfreq', axis=0).max()
corr['meanfreq'].drop('meanfreq', axis=0)[corr['meanfreq'] == corr['meanfreq'].drop('meanfreq', axis=0).max()]
for i in df.columns:

    print([corr[i].drop(i, axis=0)[corr[i] == corr[i].drop(i, axis=0).max()]])
plt.figure(figsize=(8,8))

ax = sns.heatmap(

    corr, 

    vmax=.8,

    square=True,

)
df.isnull().sum()
print("Number of male: {}".format(df[df.label == 'male'].shape[0]))

print("Number of female: {}".format(df[df.label == 'female'].shape[0]))
#X=df.iloc[:, :-1]

#X.head()
X = df.drop('label', axis=1)

X.head(1)
from sklearn.preprocessing import LabelEncoder

y = df['label']

# Encode label category

# male -> 1

# female -> 0

gender_encoder = LabelEncoder()

y = gender_encoder.fit_transform(y)

y.shape
gender_encoder.classes_
y
# Scale the data to be between -1 and 1

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

X
data = [[0, 0], [0, 0], [1, 1], [1, 1]]

scaler = StandardScaler()

print(scaler.fit(data))

print(scaler.mean_)

print(scaler.transform(data))

print(scaler.transform([[2, 2]]))
((0.5 **2) * 4 / 4) ** 0.5
(0 - 0.5)/0.25
a = np.array(data)

a
np.std(a, axis=0)
np.mean(a, axis=0)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.svm import SVC

from sklearn import metrics

svc=SVC() #Default hyperparameters

svc.fit(X_train,y_train)
y_pred_train=svc.predict(X_train)

y_pred=svc.predict(X_test)

print('Accuracy Score for train:', metrics.accuracy_score(y_train,y_pred_train))

print('Accuracy Score for test:', metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='linear')

svc.fit(X_train,y_train)
y_pred_train=svc.predict(X_train)

y_pred=svc.predict(X_test)

print('Accuracy Score for train:', metrics.accuracy_score(y_train,y_pred_train))

print('Accuracy Score for test:', metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='rbf')

svc.fit(X_train,y_train)
y_pred_train=svc.predict(X_train)

y_pred=svc.predict(X_test)

print('Accuracy Score for train:', metrics.accuracy_score(y_train,y_pred_train))

print('Accuracy Score for test:', metrics.accuracy_score(y_test,y_pred))
svc=SVC(kernel='poly')

svc.fit(X_train,y_train)
y_pred_train=svc.predict(X_train)

y_pred=svc.predict(X_test)

print('Accuracy Score for train:', metrics.accuracy_score(y_train,y_pred_train))

print('Accuracy Score for test:', metrics.accuracy_score(y_test,y_pred))
from sklearn.model_selection import cross_val_score

svc=SVC(kernel='linear')

scores = cross_val_score(svc,X,y, cv=10, scoring='accuracy')#cv is cross validation

print(scores)
print(scores.mean())
svc=SVC(kernel='rbf', gamma='auto')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print(scores)

print(scores.mean())
svc=SVC(kernel='poly',gamma='auto')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy') #cv is cross validation

print(scores)

print(scores.mean())
C_range = list(range(1, 26))

acc_score=[]

for c in C_range:

    svc = SVC(kernel='linear', C=c)

    score = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(score.mean())

print(acc_score)
import matplotlib.pyplot as plt

%matplotlib inline
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(C_range, acc_score)

plt.xticks(np.arange(0,27,2))

plt.xlabel('Value of C for SVC')

plt.ylabel('Cross-Validated Accuracy');
C_range=list(np.arange(0.1,6,0.1))

acc_score=[]

for c in C_range:

    svc = SVC(kernel='linear', C=c)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)    
# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

C_values=list(np.arange(0.1,6,0.1))

plt.plot(C_values,acc_score)

plt.xticks(np.arange(0.0,6,0.3))

plt.xlabel('Value of C for SVC ')

plt.ylabel('Cross-Validated Accuracy')
gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]

acc_score=[]

for g in gamma_range:

    svc = SVC(kernel='rbf', gamma=g)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)    
gamma_range=[0.0001,0.001,0.01,0.1,1,10,100]



# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(gamma_range,acc_score)

plt.xlabel('Value of gamma for SVC ')

plt.xticks(np.arange(0.0001,100,5))

plt.ylabel('Cross-Validated Accuracy')
gamma_range=[0.0001,0.001,0.01,0.1]

acc_score=[]

for g in gamma_range:

    svc = SVC(kernel='rbf', gamma=g)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)    

    
gamma_range=[0.0001,0.001,0.01,0.1]



# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(gamma_range,acc_score)

plt.xlabel('Value of gamma for SVC ')

plt.ylabel('Cross-Validated Accuracy')
gamma_range=[0.01,0.02,0.03,0.04,0.05]

acc_score=[]

for g in gamma_range:

    svc = SVC(kernel='rbf', gamma=g)

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)    
gamma_range=[0.01,0.02,0.03,0.04,0.05]



# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(gamma_range,acc_score)

plt.xlabel('Value of gamma for SVC ')

plt.ylabel('Cross-Validated Accuracy')
degree=[2,3,4,5,6]

acc_score=[]

for d in degree:

    svc = SVC(kernel='poly', degree=d, gamma='auto')

    scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

    acc_score.append(scores.mean())

print(acc_score)    
degree=[2,3,4,5,6]



# plot the value of C for SVM (x-axis) versus the cross-validated accuracy (y-axis)

plt.plot(degree,acc_score,color='r')

plt.xlabel('degrees for SVC ')

plt.ylabel('Cross-Validated Accuracy')
from sklearn.svm import SVC

svc= SVC(kernel='linear',C=0.1)

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

accuracy_score= metrics.accuracy_score(y_test,y_predict)

print(accuracy_score)
from sklearn.model_selection import cross_val_score

svc=SVC(kernel='linear',C=0.1)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(scores)

print(scores.mean())
from sklearn.svm import SVC

svc= SVC(kernel='rbf',gamma=0.01)

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

metrics.accuracy_score(y_test,y_predict)
svc=SVC(kernel='linear',gamma=0.01)

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(scores)

print(scores.mean())
from sklearn.svm import SVC

svc= SVC(kernel='poly',degree=3, gamma='auto')

svc.fit(X_train,y_train)

y_predict=svc.predict(X_test)

accuracy_score= metrics.accuracy_score(y_test,y_predict)

print(accuracy_score)
svc=SVC(kernel='poly',degree=3, gamma='auto')

scores = cross_val_score(svc, X, y, cv=10, scoring='accuracy')

print(scores)

print(scores.mean())
from sklearn.svm import SVC

svm_model = SVC()
tuned_parameters = {'C':(np.arange(0.1, 1, 0.1)), 'kernel':['linear'],

                    'C':(np.arange(0.1, 1, 0.1)), 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel':['rbf'],

                    'C':(np.arange(0.1, 1, 0.1)), 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel':['poly'], 'degree': [2,3,4]}
from sklearn.model_selection import GridSearchCV

model_svm = GridSearchCV(svm_model, tuned_parameters, cv=10, scoring='accuracy')
%%time

model_svm.fit(X_train, y_train)
print(model_svm.best_score_)

print(model_svm.best_params_)

print(model_svm.best_estimator_)
from sklearn.model_selection import GridSearchCV

model_svm = GridSearchCV(svm_model, tuned_parameters)
model_svm.fit(X_train, y_train)
print(model_svm.best_params_)

print(model_svm.best_estimator_)
y_pred = model_svm.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
svc = SVC()

svc.fit(X_train, y_train)
y_pred=svc.predict(X_test)

print(metrics.accuracy_score(y_pred, y_test))
param_grid = {'C':(np.arange(0.1, 1, 0.1)), 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel':['rbf']}

model = GridSearchCV(SVC(), param_grid)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
print(model.best_params_)

print(model.best_estimator_)
param_grid = {'C':(np.arange(0.1, 1, 0.1)), 'kernel':['linear']}

model = GridSearchCV(SVC(), param_grid)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
print(model.best_params_)

print(model.best_estimator_)
param_grid = {'C':(np.arange(0.1, 1, 0.1)), 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel':['poly'], 'degree': [2,3,4]}

model = GridSearchCV(SVC(), param_grid)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
# re-run the rbf kernel

param_grid = {'C':(np.arange(0.1, 1, 0.1)), 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel':['rbf']}

model = GridSearchCV(SVC(), param_grid)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
#for original training data score

y_pred_train = model.predict(X_train)

print(metrics.accuracy_score(y_train, y_pred_train))
#score for original training data

print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
# run the rbf kernel with K-fold

param_grid = {'C':(np.arange(0.1, 1, 0.1)), 'gamma': [0.01,0.02,0.03,0.04,0.05], 'kernel':['rbf']}

model = GridSearchCV(SVC(), param_grid, cv=10, scoring='accuracy')

model.fit(X_train, y_train)

print(model.best_score_)

print(model.best_params_)

print(model.best_estimator_)
model.best_estimator_
# We used k-fold to produce the best model and we will continue to predict X_test data to be able to get the accuracy score.

model_k_fold = model.best_estimator_

y_pred_k_fold = model_k_fold.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred_k_fold))