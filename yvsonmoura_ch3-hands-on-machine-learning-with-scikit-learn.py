## Page 87/88

# Classification
# MNIST Dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
mnist.keys()

X, y = mnist["data"], mnist["target"]
X.shape #(70000, 784)
y.shape #(70000, )

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
# More Images
import numpy as np

some_digit0 = X[0:5]
some_digit1 = X[5:10]
some_digit2 = X[15:20]
some_digit3 = X[25:30]
some_digit4 = X[35:40]
some_digit_image0 = some_digit0.reshape(140, 28)
some_digit_image1 = some_digit1.reshape(140, 28)
some_digit_image2 = some_digit2.reshape(140, 28)
some_digit_image3 = some_digit3.reshape(140, 28)
some_digit_image4 = some_digit4.reshape(140, 28)
image = np.c_[some_digit_image0, some_digit_image1, some_digit_image2, some_digit_image3, some_digit_image4]

plt.imshow(image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
## Page 89/90

import numpy as np

y = y.astype(np.uint8) # converting from 
some_digit = X[0]

# Train data and Test data
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training a Binary Classifier
y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

# Stochastic Gradient Descent (SGD)
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])
## Page 91/92

# Measuring Accuracy Using Cross-Validation - Implementing Cross-Validation
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

skfolds = StratifiedKFold(n_splits=3, random_state=42)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_fold = X_train[test_index]
    y_test_fold = y_train_5[test_index]
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495
    
# Using cross_val_score function
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy") # returns array([0.96355, 0.93795, 0.95615])

# Analyzing number 5 classification with cross_val_score function
from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
    
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")
## Page 92/93/94/95

# Confusion Matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)

y_train_perfect_predictions = y_train_5 # pretend we reached perfection
confusion_matrix(y_train_5, y_train_perfect_predictions)

# Precision = TP/(TP + FP)
# Recall = TP/(TP + FN)

from sklearn.metrics import precision_score, recall_score

precision_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1522) = 0,7290
recall_score(y_train_5, y_train_pred) # == 4096 / (4096 + 1325) = 0,7555

# F1 = harmonic_mean(precision, recall) = 2/((1/precision)+(1/recall))
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)
## Page 96

# Decision threshold and precision/recall tradeoff
y_scores = sgd_clf.decision_function([some_digit])
y_scores

threshold = 0 # SGDClassifier uses a threshold equal to 0,
y_some_digit_pred = (y_scores > threshold) 

threshold = 8000
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred # array([False]) - This confirms that raising the threshold decreases recall.

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")

# Now with these scores you can compute precision and recall for all possible thresholds using the precision_recall_curve() function
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

## Page 97

plt.style.use('ggplot')

threshold_40_recalls = thresholds[np.argmax(recalls <= 0.48)] # ~ 
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] # ~ 

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(12,6))
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.plot(threshold_90_precision, 0.9, 'o', ms=5, c='red')
    plt.plot(threshold_40_recalls, 0.48, 'o', ms=5, c='red')
    plt.hlines(y=0.9, xmin=-110000, xmax=3500, colors='red', linestyle='dotted')
    plt.hlines(y=0.48, xmin=-110000, xmax=3500, colors='red', linestyle='dotted')
    plt.vlines(x=3500, ymin=0, ymax=0.9, colors='red', linestyle='dotted')
    plt.xlabel('Threshold')
    plt.legend()
    # highlight the threshold, add the legend, axis label and grid

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()
## Page 98

# Using 90% Precision to predict 
y_train_pred_90 = (y_scores >= threshold_90_precision)

precision_score(y_train_5, y_train_pred_90) # 0.9000380083618396
recall_score(y_train_5, y_train_pred_90) # 0.4368197749492714
## Page 99/100

# The ROC Curve - The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.ylabel('True Positive Ratio - Sensitivity (Recall)')
    plt.xlabel('False Positive Ratio (Specificity)')
    plt.legend()
    
plot_roc_curve(fpr, tpr, label='ROC Curve')
plt.show()

# ROC Area Under the Curve (ROC AUC = 1 - Perfect classifier // ROC AUC = 0.5 Purely radom classifier)
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5, y_scores)
## Page 101/102

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class  # ROC curve, you need scores, not probabilities.
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

roc_auc_score(y_train_5, y_scores_forest) # 0.9983436731328145
## Page 102/103

# Multiclass Classification - distinguishing among more than two classes
sgd_clf.fit(X_train, y_train) # y_train, not y_train_5
sgd_clf.predict([some_digit])

some_digit_scores = sgd_clf.decision_function([some_digit])
some_digit_scores

np.argmax(some_digit_scores)

sgd_clf.classes_

sgd_clf.classes_[5]


## Page 104

# Classifiers - OneVsOneClassifier & OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)

forest_clf.fit(X_train, y_train)
forest_clf.predict([some_digit])
forest_clf.predict_proba([some_digit])

cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# Improving accuracy by scaling the inputs with StandardScaler
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")
## Page 105/106

# Error Analysis
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# The plot of the confusion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# Comparing errors among the classes
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
## Page 107

# Plot_digits function
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = plt.cm.binary, **options)
    plt.axis("off")

# Analyzing 3s and 5s
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()
## Page 108

# Multilabel Classification
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()

# This code creates a y_multilabel array containing two target labels for each digit image # 
# FIRST indicates whether or not the digit is large (7, 8, or 9) and SECOND indicates whether or not it is odd.
knn_clf.fit(X_train, y_multilabel)

knn_clf.predict([some_digit]) # testing number 5 - It gets it right!

## Page 109a

# This code computes the average F1 score across all labels
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro")
## Page 109b/110

# Multioutput Classification

# Let’s build a system that removes noise from images
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise

noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise

y_train_mod = X_train
y_test_mod = X_test

# Plotting actual image and noisy image
cl_c = 4
X_four_actual = X_train[(y_train == cl_c)]
X_four_noisy = X_train_mod[(y_train == cl_c)]
plt.subplot(122); plot_digits(X_four_actual[:4],  images_per_row=2) 
plt.subplot(121); plot_digits(X_four_noisy[:4],  images_per_row=2)

## Page 110

# Noise cleaning - Previous image cleaned.
knn_clf.fit(X_train_mod, y_train_mod)
index_fours = np.where(y_train == 4)[0][0:4] # index number = 9
clean_digit0 = knn_clf.predict([X_train_mod[index_fours[0]]])
clean_digit1 = knn_clf.predict([X_train_mod[index_fours[1]]])
clean_digit2 = knn_clf.predict([X_train_mod[index_fours[2]]])
clean_digit3 = knn_clf.predict([X_train_mod[index_fours[3]]])
clean_digits = np.r_[clean_digit0, clean_digit1, clean_digit2, clean_digit3]
plot_digits(clean_digits, images_per_row=2)
