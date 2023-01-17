import numpy as np 

import pandas as pd 

from sklearn.model_selection import RandomizedSearchCV

import lightgbm

import matplotlib.pyplot as plt
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")



train = train.drop(["Id"], axis = 1)



test_ids = test["Id"]

test = test.drop(["Id"], axis = 1)
from sklearn.ensemble import RandomForestClassifier



def random_RF(X,y,n_iter):

    # Number of trees in random forest

    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

    # Number of features to consider at every split

    max_features = ['auto', 'sqrt']

    # Maximum number of levels in tree

    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

    max_depth.append(None)

    # Minimum number of samples required to split a node

    min_samples_split = [2, 5, 10]

    # Minimum number of samples required at each leaf node

    min_samples_leaf = [1, 2, 4]

    # Method of selecting samples for training each tree

    bootstrap = [True, False]

    # Create the random grid

    random_grid = {'n_estimators': n_estimators,

                   'max_features': max_features,

                   'max_depth': max_depth,

                   'min_samples_split': min_samples_split,

                   'min_samples_leaf': min_samples_leaf,

                   'bootstrap': bootstrap}



    rfc = RandomForestClassifier(n_jobs=-1)

    rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = n_iter, cv = 5, verbose=1, random_state=111, n_jobs = -1, scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, refit='NLL')



    # Fit the random search model

    rf_random.fit(X, y)

    return rf_random.best_estimator_, rf_random.cv_results_, rf_random.best_params_
n_iter=10 #iters for random search

X,y=train.drop(['Cover_Type'], axis=1), train['Cover_Type']

m,s = X.mean(0), X.std(0)

s[s==0]=1

X = (X-m)/s

trained_model, results, params=random_RF(X,y,n_iter=n_iter)

import pickle

with open('params.pickle', 'wb') as handle:

    pickle.dump(params, handle)
plt.figure(figsize=(13, 13))

plt.title("RandomSearchCV evaluating using multiple scorers simultaneously",

          fontsize=16)



plt.xlabel("iters")

plt.ylabel("Score")



ax = plt.gca()

#ax.set_xlim(0, 402)

ax.set_ylim(0.5, 0.85)



# Get the regular numpy array from the MaskedArray

X_axis = np.arange(0,n_iter)



scoring={'NLL':'neg_log_loss', 'Accuracy':'accuracy'}

for scorer, color in zip(sorted(scoring, reverse=True), ['g', 'k']):

    sample = 'test'

    if scorer == 'NLL':

        sample_score_mean = -1 * results['mean_%s_%s' % (sample, scorer)]

    else:

        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]

    sample_score_std = results['std_%s_%s' % (sample, scorer)]

    ax.fill_between(X_axis, sample_score_mean - sample_score_std,

                    sample_score_mean + sample_score_std,

                    alpha=0.1 if sample == 'test' else 0, color=color)

    ax.plot(X_axis, sample_score_mean,  color=color,

            alpha=0.4,

            label="%s (%s)" % (scorer, sample))

    

    if scorer=='NLL':

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]

        best_score = -1*results['mean_test_%s' % scorer][best_index]

    if scorer=='Accuracy':

        best_score = results['mean_test_%s' % scorer][best_index]

        

        



    # Plot a dotted vertical line at the best score for that scorer marked by x

    ax.plot([X_axis[best_index], ] * 2, [0, best_score],

            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)



    # Annotate the best score for that scorer

    if scorer=="NLL":

        ax.annotate("%0.2f" % best_score,

                   (X_axis[best_index]+0.05, best_score + 0.005))

    else:

        ax.annotate("%0.2f" % best_score,

                    (X_axis[best_index]+0.05, best_score - 0.01))



plt.legend(loc="best")

plt.grid(False)

plt.show()
test = (test - m)/s

test_pred = trained_model.predict(test)
# Save test predictions to file

output = pd.DataFrame({'Id': test_ids,

                       'Cover_Type': test_pred})

output.to_csv('rf_submission.csv', index=False)