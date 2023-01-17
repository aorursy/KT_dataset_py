import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from xgboost import XGBClassifier

from sklearn.metrics import mean_absolute_error



# apply ignore

import warnings

warnings.filterwarnings('ignore')
#load train data

train_data = pd.read_csv('../input/learn-together/train.csv')

train_data.head()
from sklearn.model_selection import train_test_split



# Select columns 

selected_features = [cname for cname in train_data.columns if cname not in ['Id','Cover_Type']]



X = train_data[selected_features]

y = train_data.Cover_Type



# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.15, random_state=0)
# Define the model 

# multiclass classification objective ('multi:softmax', 'multi:softprob')

xgb_model = XGBClassifier(max_depth=50, learning_rate=0.04, 

                          objective='multi:softmax', num_class=7)

# Train and evaluate.

# multiclass classification eval_metric ('merror', 'mlogloss')

evalmetric = 'merror'

xgb_model.fit(X_train, y_train, eval_metric=[evalmetric], eval_set=[(X_train, y_train),(X_valid, y_valid)], verbose=False)
# Import the library

import matplotlib.pyplot as plt

%matplotlib inline



def visualize_acuracy(xgb, metric):

    # Plot and display the performance evaluation

    xgb_eval = xgb.evals_result()

    eval_steps = range(len(xgb_eval['validation_0'][metric]))

    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(8, 6))

    ax.plot(eval_steps, [1-x for x in xgb_eval['validation_0'][metric]], label='Train')

    ax.plot(eval_steps, [1-x for x in xgb_eval['validation_1'][metric]], label='Test')

    ax.legend()

    ax.set_title('Accuracy')

    ax.set_xlabel('Number of iterations')
visualize_acuracy(xgb_model, evalmetric)    
from sklearn.metrics import accuracy_score

def score_accuracy(xgb, X, y):

    # run trained model.

    y_pred = xgb.predict(X)

    # Check the accuracy of the trained model.

    accuracy = accuracy_score(y, y_pred)

    print("Accuracy: %.1f%%" % (accuracy * 100.0))
score_accuracy(xgb_model, X_valid, y_valid)
from sklearn.model_selection import cross_val_score

def score_mean(xgb, X, y):

    accuracies = cross_val_score(estimator = xgb, X = X, y = y, cv = 6)

    print("Mean_XGB_Acc : ", accuracies.mean())
score_mean(xgb_model, X_valid, y_valid)
from sklearn.metrics import classification_report

def score_report(xgb, X, y):

    # run trained model.

    y_pred = xgb.predict(X)

    print(classification_report(y, y_pred))
score_report(xgb_model, X_valid, y_valid)
import seaborn as sns

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(model, X, y, normalized=True, cmap='bone'):

    # run trained model.

    y_pred = model.predict(X)

    classes = np.sort(y.unique()) # depends (y should have all labels)

    # run trained model.

    cm = confusion_matrix(y, y_pred)

    # run trained model.

    plt.figure(figsize=[7, 6])

    norm_cm = cm

    if normalized:

        norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(norm_cm, annot=cm, fmt='g', xticklabels=classes, yticklabels=classes, cmap=cmap)

        plt.savefig('confusion-matrix.png')
plot_confusion_matrix(xgb_model, X_valid, y_valid)
# Search for the best parameters.

from sklearn.model_selection import GridSearchCV

# set up parameter grid.

parameters = { 'n_estimators': np.arange(50, 150, 10)}

clf = GridSearchCV(xgb_model, parameters, scoring='accuracy', cv=5, n_jobs=-1, refit=True)

clf.fit(X, y)
# Display the accuracy of best parameter combination on the test set.

print("Best score: %.1f%%" % (clf.best_score_*100))

print("Best parameter set: %s" % (clf.best_params_))
xgb_best = clf.best_estimator_
# read test data file using pandas

test_data = pd.read_csv('../input/learn-together/test.csv')



# make predictions 

test_preds = xgb_best.predict(test_data[selected_features])



# save to submit

output = pd.DataFrame({'Id': test_data.Id,

                       'Cover_Type': test_preds})

output.to_csv('submission.csv', index=False)
output.head()