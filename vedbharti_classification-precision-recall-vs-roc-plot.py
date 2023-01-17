import numpy as np

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_curve

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.datasets import make_classification

import collections

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib inline
def CreateDatasetTrainModel(samplesize, colcount, weight, testsize):

    '''generate dataset based on the weights provided,  plot target distribution, split test train set and perform the grid search '''

    X, y = make_classification(n_samples = samplesize, n_features = colcount, n_classes = 2, weights = weight, random_state = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = testsize, random_state = 1)

    print(f'\nDataset created with class weight {weight}\n')

    w = collections.Counter(y)

    plt.bar(w.keys(), w.values())

    plt.title('Target Distribution')

    # Hyperparameter tuning using cross-validation and grid search

    steps = [('scaler', StandardScaler()), 

        ('logreg', LogisticRegression(penalty = 'l1', solver = 'saga', tol = 1e-6,

                                      max_iter = int(1e6), warm_start = True, n_jobs = -1))] 

    pipeline = Pipeline(steps)

    param_grid = {'logreg__C': np.arange(0., 1, 0.1)}

    logreg_cv = GridSearchCV(pipeline, param_grid, cv = 5,  n_jobs = -1)

    

    logreg_cv.fit(X_train, y_train) 

    

    # best parameter and score

    print('Best score:', logreg_cv.best_score_)

    print('\nbest parameter:',logreg_cv.best_params_)

    

    return X_train, X_test, y_train, y_test
def plotFeatures(model):

    '''Plot feature selection'''

    lasso_coef = model.coef_.reshape(-1,1)

    plt.figure(figsize = (15,5))

    plt.plot([0,24],[0,0])

    _ = plt.plot(range(25), lasso_coef, linestyle='--', marker='x', color='r')

    _ = plt.xticks(range(25), range(25))

    _ = plt.ylabel('Coefficients')

    plt.xlabel('Features', fontsize = 14)

    plt.ylabel('Coefficients', fontsize = 14)

    plt.xticks(size = 12)

    plt.yticks(size = 12)

    plt.title('Feature Coefficients from Lasso Logistic Regression', fontsize = 22)

    plt.show();
def PrecisionRecall(y, ypred):

    '''Plot precision and recall plot based on test and predicted target values'''

    precision, recall, thresholds = precision_recall_curve(y, ypred)

    plt.figure(figsize = (10,8))

    plt.plot(recall, precision)

    plt.plot([0, 1], [0.5, 0.5], linestyle = '--')  #baseline

    plt.xlabel('Recall', fontsize = 14)

    plt.ylabel('Precision', fontsize = 14)

    plt.title('Lasso Logistic Regression Precision-Recall Curve', fontsize = 22)
def plotROC(y, ypred):

    fpr, tpr, thresholds = roc_curve(y, ypred)

    plt.figure(figsize = (10,8))

    plt.plot([0, 1], [0, 1], linestyle = '--') # baseline 

    plt.plot(fpr, tpr)

    plt.xlabel('False Positive Rate', fontsize = 14)

    plt.ylabel('True Positive Rate', fontsize = 14)

    plt.title('Lasso Logistic Regression ROC Curve', fontsize = 22)

    plt.show();
X_train, X_test, y_train, y_test = CreateDatasetTrainModel(100000, 25, [0.5,0.5], 0.3)
# Fit lasso logistic regression using the best parameter

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

logreg = LogisticRegression(penalty = 'l1', solver = 'saga', tol = 1e-6,  max_iter = int(1e6),

                            warm_start = True, C = 0.2)

logreg.fit(X_train_scaled, y_train)
plotFeatures(logreg)
# Make predictions using the test dataset

X_test_scaled = scaler.transform(X_test)

y_pred_prob = logreg.predict_proba(X_test_scaled)[:,1]  # return probabilities for the positive outcome only
plotROC(y_test, y_pred_prob)
# area under the curve ROC

round(roc_auc_score(y_test, y_pred_prob), 2)
# Precision-Recall Curve 

PrecisionRecall(y_test, y_pred_prob)
# area under the curve for precision and recall

round(average_precision_score(y_test, y_pred_prob), 2)
X_train, X_test, y_train, y_test = CreateDatasetTrainModel(100000, 25, [0.99,0.01], 0.3)
# Fit lasso logistic regression using the best parameter

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)

logreg = LogisticRegression(penalty = 'l1', solver = 'saga', tol = 1e-6,  max_iter = int(1e6),

                            warm_start = True, C = 0.4)

logreg.fit(X_train_scaled, y_train)
plotFeatures(logreg)
# Make predictions using the test dataset

X_test_scaled = scaler.transform(X_test)

y_pred_prob = logreg.predict_proba(X_test_scaled)[:,1]  # return probabilities for the positive outcome only
# Plot ROC Curve

plotROC(y_test, y_pred_prob)
# area under the curve ROC

round(roc_auc_score(y_test, y_pred_prob), 2)
# Precision-Recall Curve

PrecisionRecall(y_test, y_pred_prob)
# area under the curve

round(average_precision_score(y_test, y_pred_prob), 2)