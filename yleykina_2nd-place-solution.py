import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
train_csv = "../input/traindata/traindata.csv"
test_csv = "../input/ucu-machine-learning-inclass/testdata.csv"
X_Train = pd.read_csv(train_csv, usecols = range(1, 17))
y_Train = pd.read_csv(train_csv, usecols = [17])
X_Submit = pd.read_csv(test_csv, usecols = range(1, 17))
print('X_Train: ', X_Train.shape)
print('y_Train: ',y_Train.shape)
print(y_Train['y'].value_counts())
print('\nX_Submit: ', X_Submit.shape)
X_Train.head()
frames = [X_Train, X_Submit]
X_join = pd.concat(frames, keys=['train', 'test'])

# !pip install pandas-profiling
# import pandas_profiling
# pandas_profiling.ProfileReport(X_join)
# Remove 'previous' because of the correlation
X_Train_cleaned = X_Train.loc[:, X_Train.columns != 'previous']
X_Submit_cleaned = X_Submit.loc[:, X_Submit.columns != 'previous']

# One-hot encoding
frames = [X_Train_cleaned, X_Submit_cleaned]
X_join = pd.concat(frames, keys=['train', 'test'])
X_join_dummies = pd.get_dummies(X_join)

# pandas_profiling.ProfileReport(X_join_dummies)
X_Train = X_join_dummies.loc['train']
X_Submit = X_join_dummies.loc['test']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_Train, y_Train, test_size=0.1, random_state=42, stratify=y_Train)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_test_transformed = np.hstack((1 - y_test.reshape(y_test.size,1),
                                y_test.reshape(y_test.size,1)))

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost

XGB_Classifier = xgboost.XGBClassifier()

CV_SSS = StratifiedShuffleSplit(n_splits=5, test_size=0.33, random_state=None)

xgb_param_grid = {
                  'max_depth': np.arange(2, 40, 2),
                  'learning_rate': np.arange(0.2, 2, 0.2),
                  'n_estimators': np.arange(20, 400, 20),
                  'reg_alpha': np.arange(0, 2, 0.2),
                  'reg_lambda': np.arange(0, 2, 0.2)
                 }

random_grid_XGB_CV = RandomizedSearchCV(XGB_Classifier,
                                        xgb_param_grid,
                                        scoring = 'roc_auc',
                                        cv = CV_SSS,
                                        n_iter = 1) # change it to 10 or more
random_grid_XGB_CV.fit(X_train, y_train)
print(random_grid_XGB_CV.best_score_)
print(random_grid_XGB_CV.best_params_)
# XGB_Classifier = random_grid_XGB_CV.best_estimator_
XGB_Classifier = xgboost.XGBClassifier(params={'reg_lambda': 1.2, 'reg_alpha': 1.6, 'n_estimators': 60, 'max_depth': 30, 'learning_rate': 0.4})
XGB_Classifier.fit(X_train, np.ravel(y_train))
print(XGB_Classifier.feature_importances_)

y_score = XGB_Classifier.predict_proba(X_test)
auc_score = roc_auc_score(y_test_transformed, y_score, average='weighted')
print('roc-auc = {}'.format(round(auc_score, 4)))
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], _ = roc_curve(y_test_transformed[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
lw = 2

plt.plot(fpr[1], tpr[1], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[1])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
submission_prediction = XGB_Classifier.predict_proba(X_Submit)[:,1]
print(submission_prediction)
import csv
from datetime import datetime as dt
import pytz

time_now = dt.now(pytz.timezone('Etc/GMT-3')).strftime("%Y%m%dT%H%M")
submission_file = 'submission_{}_auc_{}.csv'.format(time_now, round(auc_score, 3))
with open(submission_file, 'w') as csvfile:
    fieldnames = ['id', 'y']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(X_Submit.shape[0]):
            writer.writerow({'id': i+1, 'y': np.round(submission_prediction, 3)[i]})
print(submission_file)
