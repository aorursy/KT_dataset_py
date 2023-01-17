from __future__ import division # In case of Python 2.7.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import probplot

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score

from itertools import cycle

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier



plt.rcParams['font.size'] = 12

plt.rcParams['figure.figsize'] = (7, 4)

plt.rcParams['image.interpolation'] = 'nearest'

plt.rcParams['image.cmap'] = 'gray'

%matplotlib inline
data = pd.read_csv('../input/creditcard.csv')

data.head(n = 5)
np.isnan(data).any().values.reshape(1, -1)
print ('No. of genuine transactions: {0:g}\nNo. of fraudulent transactions: {1:g}'.format( \

        np.sum(data['Class'] == 0), np.sum(data['Class'] == 1)))
features = data.columns

features = [str(s) for s in features]

label = features[-1]

features = features[1:-1] # Time is only an indicator instead of a useful feature.



data = data[features + [label]]
fig, ax = plt.subplots(6, 5, figsize = (12, 12))

k = 0

for i in range(6):

    for j in range(5):

        ax = plt.subplot(6, 5, k+1)

        probplot(data.iloc[:, k], dist = 'norm', plot = plt)

        ax.get_lines()[0].set_markersize(3)

        ax.get_lines()[0].set_markerfacecolor('None')

        ax.get_lines()[0].set_markeredgecolor('k')

        ax.get_lines()[1].set_linewidth(2)

        ax.set_xticks([])

        ax.set_xlabel('')

        ax.set_yticks([])

        ax.set_ylabel('')

        ax.set_title('')

        k += 1



plt.tight_layout()

plt.subplots_adjust(wspace = 0, hspace = 0)    
scaler = StandardScaler().fit(data[features])

scaler_mean = scaler.mean_

scaler_scale = scaler.scale_

data[features] = scaler.transform(data[features])
train_data, val_test_data = train_test_split(data, test_size = 0.4, random_state = 1)

val_data, test_data = train_test_split(val_test_data, test_size = 0.5, random_state = 1)
X = train_data[features]

y = train_data[label]



sm = SMOTE(ratio = 'auto', kind = 'regular', random_state = 1)

sample_X, sample_y = sm.fit_sample(X, y)

sample = pd.concat([pd.DataFrame(sample_X), pd.DataFrame(sample_y)], axis = 1)

sample.columns = features + [label]
print ('No. of fraudulent transactions: {0:g}\nNo. of genuine transactions: {1:g}'.format( \

        np.sum(sample['Class'] == 0), np.sum(sample['Class'] == 1)))
sum(test_data[label] == 0) / float(len(test_data))
def tune_clf(models, param_name, param_values, train_data, val_data, features, label):

    auc_scores = []

    param_cycle = cycle(param_values)

    for model in models:

        model.fit(train_data[features], train_data[label])

        probs = model.predict_proba(val_data[features])

        auc_scores.append(roc_auc_score(y_true = val_data[label], y_score = probs[:, 1]))

        """ If you don't mind seeing the output, un-comment the following line. """

        #print(param_name + ' = {0:g}, auc = {1:.4f}'.format(next(param_cycle), 

        #auc_scores[-1])) 

    

    plt.plot(param_values, auc_scores, 'b-o', linewidth = 2)

    plt.xlabel(param_name)

    plt.ylabel('auc')

    plt.tight_layout()

    plt.show()

    

    return models, auc_scores 
param_values = np.logspace(2, 4, 21)

models = []

for param_value in param_values:

    models.append(LogisticRegression(random_state = 1, penalty = 'l1', 

                                     C = 1.0/param_value))

    

models, auc_scores = tune_clf(models, 'l1', param_values, sample, val_data, 

                              features, label) 
param_values = np.linspace(3000, 5000, 41)

models = []

for param_value in param_values:

    models.append(LogisticRegression(random_state = 0, penalty = 'l1', 

                                     C = 1.0/param_value))



models, auc_scores = tune_clf(models, 'l1', param_values, sample, val_data, 

                              features, label)
def model_evaluation(model, feature_matrix, target):

    probs = model.predict_proba(feature_matrix)[:, 1]

    (fpr, tpr, thresholds) = roc_curve(y_true = target, y_score = probs)

    auc_score = auc(x = fpr, y = tpr)

    fig, ax = plt.subplots()

    ax.plot(fpr, tpr, 'r-', linewidth = 2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth = 1)

    plt.title('ROC curve with AUC = {0:.3f}'.format(auc_score))

    plt.xlabel('fpr')

    plt.ylabel('tpr')

    plt.axis([-0.01, 1.01, -0.01, 1.01])

    plt.tight_layout()

    

    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc_score}  
idx = np.where(auc_scores == np.max(auc_scores))[0][0]

best_l1 = param_values[idx]

lr_lasso = models[idx]

roc = model_evaluation(lr_lasso, test_data[features], test_data[label])
def plot_roc(thresholds, fpr, tpr):

    fig, ax = plt.subplots()

    plt.plot(roc['thresholds'], roc['tpr'], 'r-', linewidth = 2, label = 'tpr')

    plt.plot(roc['thresholds'], roc['fpr'], 'b-', linewidth = 2, label = 'fpr')

    plt.axis([-0.01, 1.01, -0.01, 1.01])

    plt.legend(loc = 'best')

    plt.xlabel('threshold')

    plt.ylabel('tpr, fpr')

    plt.tight_layout()
plot_roc(roc['thresholds'], roc['fpr'], roc['tpr'])
def model_prediction(model, threshold, feature_matrix, target):

    probs = model.predict_proba(feature_matrix)[:, 1]

    preds = np.where(probs > threshold, 1, 0)

    tn, fp, fn, tp = confusion_matrix(y_true = target, y_pred = preds).ravel()

    precision = tp / (tp + fp)

    recall = tp / (tp + fn)

    accuracy = (tp + tn) / float(len(target))    

    return {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision': precision, 

                   'recall': recall, 'accuracy': accuracy}
metrics = model_prediction(lr_lasso, 0.8, test_data[features], test_data[label])

metrics
selected_features = list(np.array(features)[lr_lasso.coef_[0, :] != 0])

print (selected_features)
param_values = np.logspace(-3, 4, 21)

models = []

for param_value in param_values:

    models.append(LogisticRegression(random_state = 1, penalty = 'l2', 

                                     C = 1.0/param_value))



models, auc_scores = tune_clf(models, 'l2', param_values, sample, val_data, 

                              selected_features, label) 
param_values = np.arange(1, 21, 1)

models = []

for param_value in param_values:

    models.append(DecisionTreeClassifier(random_state = 1, criterion = 'entropy', 

                                         max_depth = param_value))



models, auc_scores = tune_clf(models, 'max_depth', param_values, sample, val_data, 

                              features, label)
idx = np.where(auc_scores == np.max(auc_scores))[0][0]

best_depth = param_values[idx]

best_tree = models[idx]

roc = model_evaluation(best_tree, test_data[features], test_data[label])
rf = RandomForestClassifier(random_state = 1, n_estimators = 200, max_depth = 5, 

                            bootstrap = True, criterion = 'entropy')

rf.fit(sample[features], sample[label])
tree_clumps = rf.estimators_

auc_scores = []

avg_probs = np.zeros(val_data.shape[0])

for i in range(len(tree_clumps)):

    probs = tree_clumps[i].predict_proba(val_data[features])[:, 1]

    avg_probs += probs

    auc_scores.append(roc_auc_score(val_data[label], avg_probs / float(i+1)))

    #print('No. of stumps = {0:g}, auc = {1:.5f}'.format(i, auc_scores[-1]))    

    

fig, ax = plt.subplots()

ax.plot(np.arange(1, 201), auc_scores, 'r-', linewidth = 2)

plt.xlabel('n_estimators')

plt.ylabel('AUC')

plt.tight_layout()
roc = model_evaluation(rf, test_data[features], test_data[label])
plot_roc(roc['thresholds'], roc['fpr'], roc['tpr'])
metrics = model_prediction(rf, 0.5, test_data[features], test_data[label])

metrics
adb = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(random_state = 1, \

                        max_depth = 1,criterion = 'entropy'), algorithm = 'SAMME', \

                        n_estimators = 100)

adb.fit(np.array(sample[features]), np.array(sample[label]))
weights = adb.estimator_weights_

tree_clumps = adb.estimators_

weighted_probs = 0.0

auc_scores = []

for i in range(len(weights)):

    probs = tree_clumps[i].predict_proba(val_data[features])[:, 1]

    weighted_probs += weights[i] * probs

    auc_scores.append(roc_auc_score(val_data[label], 

                                    weighted_probs / sum(weights[0:i+1])))

    #print('No. of stumps = {0:g}, auc = {1:.5f}'.format(i, auc_values[-1]))



fig, ax = plt.subplots()

ax.plot(np.arange(1, len(weights)+1), auc_scores, 'r-', linewidth = 2)

plt.xlabel('n_estimators')

plt.ylabel('AUC')

plt.tight_layout()
idx = np.where(auc_scores == np.max(auc_scores))[0][0]

weighted_probs = np.zeros(len(test_data))



for i in range(idx+1):

    probs = tree_clumps[i].predict_proba(test_data[features])[:, 1]

    weighted_probs += weights[i] * probs

weighted_probs = weighted_probs/sum(weights[:idx+1])



preds = np.where(weighted_probs > 0.5, 1, 0)

tn, fp, fn, tp = confusion_matrix(y_true = test_data[label], y_pred = preds).ravel()

precision = tp / (tp + fp)

recall = tp / (tp + fn)

accuracy = (tp + tn) / float(test_data.shape[0])



(fpr, tpr, thresholds) = roc_curve(y_true = test_data[label], y_score = weighted_probs)

auc_value = auc(x = fpr, y = tpr)

fig, ax = plt.subplots()

ax.plot(fpr, tpr, 'r-', linewidth = 2)

ax.plot([0, 1], [0, 1], 'k--', linewidth = 1)

plt.title('ROC curve with AUC = {0:.3f}'.format(auc_value))

plt.xlabel('fpr')

plt.ylabel('tpr')

plt.axis([-0.01, 1.01, -0.01, 1.01])

plt.tight_layout()

plt.show()



metrics = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp, 'precision': precision, 

                   'recall': recall, 'accuracy': accuracy}    

metrics
estimator = list(np.arange(1, 101))

#plt.plot(estimator, adb.estimator_weights_, 'b-', linewidth = 2)

plt.bar(estimator, adb.estimator_weights_, align = 'center')

plt.axis([-1, 101, 0, 3])

plt.xlabel('estimator')

plt.ylabel('weight')

plt.tight_layout()