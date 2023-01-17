# Loading all necessary modules



import warnings

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy import interp

from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve, train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc



pd.set_option('display.max_columns', 500)

%matplotlib inline



# ignore annoying warnings

warnings.filterwarnings('ignore')
# loading our data and taking a quick look at it



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

train_df.head()
test_df.head()
print('Train_df shape: ', train_df.shape)

print('Test_df shape: ', test_df.shape)
train_df.describe()
corr = train_df.corr()['target'].sort_values(ascending = False)
corr.head(20)
corr.tail(20)
Y_train = train_df['target']

X_train = train_df.drop(['target', 'id'], axis = 1)

X_test = test_df.drop('id', axis = 1)



print(Y_train.shape)

print(X_train.shape)

print(X_test.shape)
log = LogisticRegression(penalty = 'l1', random_state = 42)



# Setting parameters for GridSearchCV

params = {'solver': ['liblinear', 'saga'], 

          'C': [0.001, 0.1, 1, 10, 50], 

          'tol': [0.00001, 0.0001, 0.001, 0.005], 

          'class_weight': ['balanced', None]}



log_gs = GridSearchCV(log, params, cv = StratifiedKFold(n_splits = 5), verbose = 1, n_jobs = -1, scoring = 'roc_auc')



# Fitting our model

log_gs.fit(X_train, Y_train)



# Looking for best estimator

log_best = log_gs.best_estimator_



print(log_best)

print(log_gs.best_score_)
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,

                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):

    """Generate a simple plot of the test and training learning curve"""

    plt.figure()

    plt.title(title)

    if ylim is not None:

        plt.ylim(*ylim)

    plt.xlabel("Training examples")

    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")



    plt.legend(loc="best")

    return plt
g = plot_learning_curve(log_best,"LR learning curves",X_train,Y_train,cv=StratifiedKFold(n_splits = 5))
def plot_roc(clf, X = X_train, y = Y_train, n = 6):

    '''Plotting ROC curves with cross validation'''

    

    cv = StratifiedKFold(n_splits=n)

    classifier = clf



    tprs = []

    aucs = []

    mean_fpr = np.linspace(0, 1, 100)



    i = 0

    plt.figure(figsize = (8, 7))

    for train, test in cv.split(X, y):

        probas_ = classifier.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])

        # Compute ROC curve and area the curve

        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

        tprs.append(interp(mean_fpr, fpr, tpr))

        tprs[-1][0] = 0.0

        roc_auc = auc(fpr, tpr)

        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.3,

                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))



        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

             label='Chance', alpha=.8)



    mean_tpr = np.mean(tprs, axis=0)

    mean_tpr[-1] = 1.0

    mean_auc = auc(mean_fpr, mean_tpr)

    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b',

             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),

             lw=2, alpha=.8)



    std_tpr = np.std(tprs, axis=0)

    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                     label=r'$\pm$ 1 std. dev.')



    plt.xlim([-0.05, 1.05])

    plt.ylim([-0.05, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()
plot_roc(log_best)
log_p = LogisticRegression(class_weight='balanced', penalty='l1', C=0.1, solver='saga', random_state = 42)

g = plot_learning_curve(log_p,"LR learning curves",X_train,Y_train,cv=StratifiedKFold(n_splits = 5))
plot_roc(log_p)
# This model will give us 0.847 public score



log_p.fit(X_train, Y_train)

log_preds = log_p.predict_proba(X_test)[:, 1]



# predict_proba returns 2-dimensional array, where 1st value is a probability that target value = 0,

# 2nd value - probability that target value = 1, we need only 2nd values.
# Prepairing submission

submission = pd.DataFrame({

    'id': test_df['id'],

    'target': log_preds

})



submission.to_csv('submission.csv', index = False)