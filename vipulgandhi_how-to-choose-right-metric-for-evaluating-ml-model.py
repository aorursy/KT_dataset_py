import pandas as pd

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold



import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')





diabetes_data = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')



X =  diabetes_data.drop(["Outcome"],axis = 1)

y = diabetes_data["Outcome"]



# Train multiple models with various hyperparameters using the training set, select the model and hyperparameters that perform best on the validation set.

# Once model type and hyperparameters have been selected, train final model using these hyperparameters on the full training set, the generalized error is finally measured on the test set.

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 56)



# StratifiedKFold class performs stratified sampling to produce folds that contain a representative ratio of each class.

cv = StratifiedKFold(n_splits=10, shuffle = False, random_state = 76)



# Logistic Regression

clf_logreg = LogisticRegression()

# fit model

clf_logreg.fit(X_train, y_train)

# Make class predictions for the validation set.

y_pred_class_logreg = cross_val_predict(clf_logreg, X_train, y_train, cv = cv)

# predicted probabilities for class 1, probabilities of positive class

y_pred_prob_logreg = cross_val_predict(clf_logreg, X_train, y_train, cv = cv, method="predict_proba")

y_pred_prob_logreg_class1 = y_pred_prob_logreg[:, 1]



# SGD Classifier

clf_SGD = SGDClassifier()

# fit model

clf_SGD.fit(X_train, y_train)

# make class predictions for the validation set

y_pred_class_SGD = cross_val_predict(clf_SGD, X_train, y_train, cv = cv)

# predicted probabilities for class 1

y_pred_prob_SGD = cross_val_predict(clf_SGD, X_train, y_train, cv = cv, method="decision_function")



# Random Forest Classifier

clf_rfc = RandomForestClassifier()

# fit model

clf_rfc.fit(X_train, y_train)

# make class predictions for the validation set

y_pred_class_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv)

# predicted probabilities for class 1

y_pred_prob_rfc = cross_val_predict(clf_rfc, X_train, y_train, cv = cv, method="predict_proba")

y_pred_prob_rfc_class1 = y_pred_prob_rfc[:, 1]
from sklearn.base import BaseEstimator

import numpy as np



class BaseClassifier(BaseEstimator):

    def fit(self, X, y=None):

        pass

    def predict(self, X):

        return np.zeros((len(X), 1), dtype=bool)

    

base_clf = BaseClassifier()

cross_val_score(base_clf, X_train, y_train, cv=10, scoring="accuracy").mean()





# Method 2

# calculate null accuracy (for binary / multi-class classification problems)

# null_accuracy = y_train.value_counts().head(1) / len(y_train)
# calculate accuracy



acc_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'accuracy').mean()

acc_SGD = cross_val_score(clf_SGD, X_train, y_train, cv = cv, scoring = 'accuracy').mean()

acc_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'accuracy').mean()



acc_logreg, acc_SGD, acc_rfc
# calculate logloss



logloss_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()

logloss_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()



# SGDClassifier's hinge loss doesn't support probability estimates.

# We can set SGDClassifier as the base estimator in Scikit-learn's CalibratedClassifierCV, which will generate probability estimates.



from sklearn.calibration import CalibratedClassifierCV



new_clf_SGD = CalibratedClassifierCV(clf_SGD)

new_clf_SGD.fit(X_train, y_train)

logloss_SGD = cross_val_score(new_clf_SGD, X_train, y_train, cv = cv, scoring = 'neg_log_loss').mean()



logloss_logreg, logloss_SGD, logloss_rfc
# IMPORTANT: first argument is true values, second argument is predicted probabilities



# we pass y_test and y_pred_prob

# we do not use y_pred_class, because it will give incorrect results without generating an error

# roc_curve returns 3 objects false positive rate(fpr), true positive rate(tpr), thresholds



fpr_logreg, tpr_logreg, thresholds_logreg = metrics.roc_curve(y_train, y_pred_prob_logreg_class1)

fpr_rfc, tpr_rfc, thresholds_rfc = metrics.roc_curve(y_train, y_pred_prob_rfc_class1)

fpr_SGD, tpr_SGD, thresholds_SGD = metrics.roc_curve(y_train, y_pred_prob_SGD)



plt.plot(fpr_logreg, tpr_logreg, label="logreg")

plt.plot(fpr_rfc, tpr_rfc, label="rfc")

plt.plot(fpr_SGD, tpr_SGD, label="SGD")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.legend(loc="lower right", fontsize=10)

plt.grid(True)
# define a function that accepts a threshold and prints sensitivity and specificity

def evaluate_threshold(tpr, fpr,clf_threshold, threshold):

    print('Sensitivity:', tpr[clf_threshold > threshold][-1])

    print('Specificity:', 1 - fpr[clf_threshold > threshold][-1])
# Logistic Regression

evaluate_threshold(tpr_logreg, fpr_logreg, thresholds_logreg, 0.2), evaluate_threshold(tpr_logreg, fpr_logreg, thresholds_logreg, 0.8)
# Random Forest Classifier

evaluate_threshold(tpr_rfc, fpr_rfc, thresholds_rfc, 0.2), evaluate_threshold(tpr_rfc, fpr_rfc, thresholds_rfc, 0.8)
# SGD

evaluate_threshold(tpr_SGD, fpr_SGD, thresholds_SGD, 0.2), evaluate_threshold(tpr_SGD, fpr_SGD, thresholds_SGD, 0.8)
# IMPORTANT: first argument is true values, second argument is predicted probabilities

# print(metrics.roc_auc_score(y_test, y_pred_prob))
roc_auc_logreg = cross_val_score(clf_logreg, X_train, y_train, cv = cv, scoring = 'roc_auc').mean()

roc_auc_SGD = cross_val_score(clf_SGD, X_train, y_train, cv = cv, scoring = 'roc_auc').mean()

roc_auc_rfc = cross_val_score(clf_rfc, X_train, y_train, cv = cv, scoring = 'roc_auc').mean()



roc_auc_logreg, roc_auc_SGD, roc_auc_rfc
logreg_matrix = metrics.confusion_matrix(y_train, y_pred_class_logreg)

print(logreg_matrix)
SGD_matrix = metrics.confusion_matrix(y_train, y_pred_class_SGD)

print(SGD_matrix)
rfc_matrix = metrics.confusion_matrix(y_train, y_pred_class_rfc)

print(rfc_matrix)
report_logreg = metrics.classification_report(y_train, y_pred_class_logreg)   

report_SGD = metrics.classification_report(y_train, y_pred_class_SGD)

report_rfc = metrics.classification_report(y_train, y_pred_class_rfc)

print("report_logreg " +  "\n" + report_logreg,"report_SGD "  +  "\n" +  report_SGD,"report_rfc "  +  "\n" +  report_rfc, sep = "\n")
y_decision_function_scores = clf_logreg.decision_function(X_train)

y_decision_function_scores[6]
threshold = 0

y_decision_function_pred = (y_decision_function_scores[6] > threshold)

y_decision_function_pred
threshold = 2

y_decision_function_pred = (y_decision_function_scores[6] > threshold)

y_decision_function_pred
from sklearn.metrics import precision_recall_curve



precisions, recalls, thresholds = precision_recall_curve(y_train, y_pred_prob_logreg_class1)



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")

    plt.xlabel("Threshold")

    plt.legend(loc="upper left")

    plt.ylim([0, 1])

    

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
from sklearn.metrics import precision_score, recall_score



y_pred_90 = (y_pred_prob_logreg_class1 > 0.32)



precisionScore = precision_score(y_train, y_pred_90)

recallScore = recall_score(y_train, y_pred_90)

precisionScore, recallScore
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

import matplotlib.pyplot as plt



import warnings

warnings.filterwarnings('ignore')



names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

housing_data = pd.read_csv('../input/boston-house-prices/housing.csv', delim_whitespace=True, names=names)

housing_data.head(2)



X =  housing_data.drop(["MEDV"],axis = 1)

y = housing_data["MEDV"]



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)



model = LinearRegression()



# fit model

model.fit(X_train, y_train)



# make class predictions for the testing set

y_pred_class = model.predict(X_test)
# calculate Mean Absolute Error



print(metrics.mean_absolute_error(y_test, y_pred_class))
# calculate Mean Squared Error



print(metrics.mean_squared_error(y_test, y_pred_class))
# calculate Root Mean Squared Error



from math import sqrt



print(sqrt(metrics.mean_squared_error(y_test, y_pred_class)))
# calculate Mean Squared Log Error



print(metrics.mean_squared_log_error(y_test, y_pred_class))
# calculate R2 score



print(metrics.r2_score(y_test, y_pred_class))
import statsmodels.api as sm



X_train_2 = sm.add_constant(X_train) 

est = sm.OLS(y_train, X_train_2)

est2 = est.fit()



print("summary()\n",est2.summary())
from nltk.translate.bleu_score import sentence_bleu

reference = [['the', 'cat',"is","sitting","on","the","mat"]]

Machine_translation_1 = ["on",'the',"mat","is","a","cat"]

Machine_translation_2 = ["there",'is',"cat","sitting","cat"]

Machine_translation_3 = ['the', 'cat',"is","sitting","on","the","tam"]

score1 = sentence_bleu(reference, Machine_translation_1)

score2 = sentence_bleu(reference, Machine_translation_2)

score3 = sentence_bleu(reference, Machine_translation_3)

score1, score2, score3
# from sklearn.multiclass import OneVsOneClassifier



# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))

# ovo_clf.fit(X_train, y_train)
# Dataset - MNIST



# from sklearn.neighbors import KNeighborsClassifier



# y_train_large = (y_train >= 7)

# y_train_odd = (y_train % 2 == 1)

# y_multilabel = np.c_[y_train_large, y_train_odd]

# knn_clf = KNeighborsClassifier()

# knn_clf.fit(X_train, y_multilabel)
# knn_clf.predict([5])



# output: array([[False, True]], dtype=bool)
# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_train, cv = 10)

# f1_score(y_train, y_train_knn_pred, average="macro")