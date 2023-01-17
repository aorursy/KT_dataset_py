from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1)

mnist.keys()
X, y = mnist['data'], mnist['target']

X.shape, y.shape
import matplotlib as mpl

import matplotlib.pyplot as plt



some_digit = X[0]

some_digit_image = some_digit.reshape(28, 28)



plt.imshow(some_digit_image, cmap='binary')

plt.axis('off')

plt.show
import numpy as np



y[0] ## is a string must turn this into number

y = y.astype(np.uint8)

y[0]
## creating the train and the test set as always, the MNIST dataset is 

## already split into a training set(the first 60,000 images) and a test

## set (the last 10,000) images.



X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
## first of all i'll try to identify if the number is a 5

y_train_5 = (y_train == 5) ## labels of the train set, 60,000 in size

y_test_5 = (y_test == 5) ## labels of test set, 10,000 in size

y_train_5, y_test_5
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([X_train[0]])
from sklearn.model_selection import StratifiedKFold # performs stratified sampling to produce

from sklearn.base import clone                      # folds that contain a representative ratio

                                                    # of each class

skfolds = StratifiedKFold(n_splits=3, random_state=42)



for train_index, test_index in skfolds.split(X_train, y_train_5): # Generate indices to split 

    clone_clf = clone(sgd_clf) ## using clone()                   # data into training and test set.

    X_train_folds = X_train[train_index]

    y_train_folds = y_train_5[train_index]

    X_test_fold = X_train[test_index]

    y_test_fold = y_train_5[test_index]

    

    print('train_index:', train_index)

    print('test_index:', test_index)

    

    clone_clf.fit(X_train_folds, y_train_folds)

    y_pred = clone_clf.predict(X_test_fold)

    n_correct = sum(y_pred == y_test_fold)

    print('score: ', n_correct / len(y_pred), '\n-----------') ## outputs the ratio of correct predictions
#help(skfolds.split)
from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
from sklearn.base import BaseEstimator



class Never5Classifier(BaseEstimator):

    def fit(self, X, y=None):

        return self

    def predict(self, X):

        return np.zeros((len(X), 1), dtype=bool)



never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy')

## it will output about 90% accuracy, as there are around 90% non fives in the data set
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

#with this, i can get a prediction for each instance in the training set

y_train_pred
from sklearn.metrics import confusion_matrix

confusion_matrix(y_train_5, y_train_pred)
y_train_5_perfect_predictions = y_train_5

confusion_matrix(y_train_5, y_train_5_perfect_predictions)
from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_train_5, y_train_pred) ## when it claims it is a 5, the sgd_clf is correct in (precision*100)% of the time

recall = recall_score(y_train_5, y_train_pred) ## (1 - recall) says % how many true 5s were not spot by the classifier

precision, recall                              ## that is it only detects (recall*100)% of the 5s    
from sklearn.metrics import f1_score

f1_score(y_train_5, y_train_pred)
## instead of calling the classifier's predict() method, you can call its decision_function() method,

## which returns a score for each instance, and then use any threshold you want to make predictions

## based on those scores:

y_scores = sgd_clf.decision_function([some_digit])

y_scores
threshold = 0  #threshold from the tradeoff between Precision and Recall

y_some_digit_pred = (y_scores > threshold)

y_some_digit_pred
threshold = 8000  #threshold from the tradeoff between Precision and Recall

y_some_digit_pred = (y_scores > threshold)

y_some_digit_pred
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,

                             method='decision_function')

#wtih 'decision_function' as the method, i'll get the scores of all instances in the training set

y_scores
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')

    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')

    plt.xlabel('Threshold')

    plt.legend(loc='upper left')

    plt.ylim([0,1])





plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()
plt.plot(recalls, precisions)

plt.xlabel('Recalls')

plt.ylabel('Precisions')

plt.show()
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

# to make predictions (on the training set for now), instead of calling the classifier's predict() method,

# i'll run this code:

y_train_pred_90 = (y_scores >= threshold_90_precision)
prec_s = precision_score(y_train_5, y_train_pred_90)

rec_s = recall_score(y_train_5, y_train_pred_90)

prec_s, rec_s
from sklearn.metrics import roc_curve



false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train_5, y_scores)

fpr, tpr = false_positive_rate, true_positive_rate



def plot_roc_curve(fpr, tpr, label=None):

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--') #Dashed diagonal line

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate (Recall)')



plot_roc_curve(fpr, tpr)

plt.show()
from sklearn.metrics import roc_auc_score

roc_auc_score(y_train_5, y_scores)
## The predict_proba() method returns an array containing a row per instance

## and a column per class, each containing the probability that the given 

## instance belongs to the given class



from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(random_state=42)

y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,

                                    method='predict_proba')
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)
plt.plot(fpr, tpr, 'b:', label='SGD')

plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")

plt.legend(loc='lower right')

plt.show()
roc_auc_score(y_train_5, y_scores_forest)
from sklearn.svm import SVC

svm_clf = SVC()

svm_clf.fit(X_train, y_train)

## under the hood scikit-learn is using the OvO strategy: it trained 45 binary classifiers,

## when we call .predicti() it gets their decision scores for the image, and selected the class 

## that won the most duels. 
svm_clf.predict([some_digit])
some_digit_scores = svm_clf.decision_function([some_digit])

some_digit_scores
svm_clf.classes_
from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC())

ovr_clf.fit(X_train, y_train)
sgd_clf.fit(X_train, y_train)

sgd_clf.predict([some_digit])
sgd_clf.decision_function([some_digit])
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy')
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy')
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)

conf_mx = confusion_matrix(y_train, y_train_pred)

conf_mx
plt.matshow(conf_mx, cmap=plt.cm.gray)

plt.show()