# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt 
import seaborn as sns
plt.style.use("bmh")
df = pd.read_csv('/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')
df.head(3)
df.shape
df.head(3)
fig, ax = plt.subplots(4, 1,figsize=(15,20))
sns.countplot(y='Country', data=df, ax=ax[0])
sns.countplot(x='Sex', data=df, ax=ax[1])
sns.countplot(x='Category', data=df, ax=ax[2])
sns.countplot(x='Survived', data=df, ax=ax[3])
fig, ax = plt.subplots(3, 1,figsize=(15,20))
sns.countplot(y='Country',hue='Survived', data=df, ax=ax[0])
sns.countplot(x='Sex',hue='Survived', data=df, ax=ax[1])
sns.countplot(x='Category',hue='Survived', data=df, ax=ax[2])
df.head(3)
fig, ax = plt.subplots(1, 2,figsize=(15,8))

sns.countplot(y='Country',hue='Sex', data=df, ax= ax[0])
sns.countplot(y='Country',hue='Category', data=df, ax= ax[1])
g = sns.FacetGrid(df, hue="Sex", aspect=3)
g.map(sns.kdeplot, "Age", shade=True)
g.set(xlim=(0, 80))
g = sns.FacetGrid(df, hue="Category", aspect=3)
g.map(sns.kdeplot, "Age", shade=True)
g.set(xlim=(0, 80))
g = sns.FacetGrid(df, hue="Survived", aspect=3)
g.map(sns.kdeplot, "Age", shade=True)
g.set(xlim=(0, 80))
g = sns.FacetGrid(df, hue="Country", aspect=3)
g.map(sns.kdeplot, "Age", shade=True)
g.set(xlim=(0, 80))
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
feature = ['Sex', 'Age','Category']
frame = df[feature]
from sklearn.preprocessing import LabelEncoder
def label(value):
    label = LabelEncoder().fit(value)
    return label.transform(value)
frame['Sex'] = label(frame['Sex'])
frame['Category'] = label(frame['Category'])
frame.head(3)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(frame, df.Survived, test_size = 0.3, random_state = 40)
len(y_train), len(y_val)
import xgboost as xgb
from sklearn.metrics import mean_squared_error
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params['eta'] = 0.02
params['max_depth'] = 4

d_train = xgb.DMatrix(X_train, label=y_train)
d_valid = xgb.DMatrix(X_val, label=y_val)

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
model1 = xgb.cv(params, d_train,  num_boost_round=500, early_stopping_rounds=100)
model = xgb.train(params, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=10)
model1.loc[30:,["train-logloss-mean", "test-logloss-mean"]].plot()
xgb.plot_importance(model)
plt.rcParams['figure.figsize'] = [5, 5]
plt.title('Xgboost Survived Prediction')
plt.show()
from sklearn import preprocessing 
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import log_loss 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, average_precision_score 
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report 
k_fold = StratifiedKFold(n_splits=5,shuffle=True,random_state=2018)

penalty = 'l2'
C = 1.0
class_weight = 'balanced'
random_state = 2018
solver = 'liblinear'
n_jobs = 1

logReg = LogisticRegression(penalty=penalty, C=C, 
            class_weight=class_weight, random_state=random_state, 
                            solver=solver, n_jobs=n_jobs)

trainingScores = []
cvScores = []
predictionsBasedOnKFolds = pd.DataFrame(data=[],
                                        index=y_train.index,columns=[0,1])

model = logReg

for train_index, cv_index in k_fold.split(np.zeros(len(X_train))
                                          ,y_train.ravel()):
    X_train_fold, X_cv_fold = X_train.iloc[train_index,:], \
        X_train.iloc[cv_index,:]
    y_train_fold, y_cv_fold = y_train.iloc[train_index], \
        y_train.iloc[cv_index]
    
    model.fit(X_train_fold, y_train_fold)
    loglossTraining = log_loss(y_train_fold, 
                               model.predict_proba(X_train_fold)[:,1])
    trainingScores.append(loglossTraining)
    
    predictionsBasedOnKFolds.loc[X_cv_fold.index,:] = \
        model.predict_proba(X_cv_fold)  
    loglossCV = log_loss(y_cv_fold, 
                         predictionsBasedOnKFolds.loc[X_cv_fold.index,1])
    cvScores.append(loglossCV)
    
    print('Training Log Loss: ', loglossTraining)
    print('CV Log Loss: ', loglossCV)
    
loglossLogisticRegression = log_loss(y_train, 
                                     predictionsBasedOnKFolds.loc[:,1])
print('Logistic Regression Log Loss: ', loglossLogisticRegression)
preds = pd.concat([y_train,predictionsBasedOnKFolds.loc[:,1]], axis=1)
preds.columns = ['trueLabel','prediction']
predictionsBasedOnKFoldsLogisticRegression = preds.copy()
precision, recall, thresholds = precision_recall_curve(preds['trueLabel'],
                                                       preds['prediction'])
average_precision = average_precision_score(preds['trueLabel'],
                                            preds['prediction'])
plt.step(recall, precision, color='k', alpha=0.7, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.3, color='k')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])

plt.title('Precision-Recall curve: Average Precision = {0:0.2f}'.format(
          average_precision))
fpr, tpr, thresholds = roc_curve(preds['trueLabel'],preds['prediction'])
areaUnderROC = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='r', lw=2, label='ROC curve')
plt.plot([0, 1], [0, 1], color='k', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic: \
          Area under the curve = {0:0.2f}'.format(areaUnderROC))
plt.legend(loc="lower right")
plt.show()
X = frame.values 
y = df.Survived.values
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(16, activation='relu', input_shape=(3,)))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X, y, epochs=50, batch_size=1, verbose=1)
model.evaluate(X,y)[1]