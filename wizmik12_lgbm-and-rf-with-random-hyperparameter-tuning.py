import numpy as np 

import pandas as pd 

from sklearn.model_selection import RandomizedSearchCV

import lightgbm

import matplotlib.pyplot as plt
train = pd.read_csv("/kaggle/input/train.csv")

test = pd.read_csv("/kaggle/input/test.csv")



train = train.drop(["Id"], axis = 1)



test_ids = test["Id"]

test = test.drop(["Id"], axis = 1)
def random_lgbm(X,y,n_iter=250):

    n_estimators = [int(x) for x in np.linspace(start = 5, stop = 1000, num = 100)]

    random_grid={"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

 "max_depth"        : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],

 "min_child_weight" : [ 1, 3, 5, 7 ],

 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],

    "n_estimators": n_estimators}

    random_grid = {

    'learning_rate': [0.005, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ],

        "max_depth"        : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],

    'n_estimators': n_estimators,

    'num_leaves': [6,8,12,16],

    'boosting_type' : ['gbdt'],

 "min_child_weight" : [ 1, 3, 5, 7 ],

    'objective' : ['multiclass'],

     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

    'random_state' : [501], # Updated from 'seed'

    'colsample_bytree' : [0.3, 0.4, 0.65, 0.66],

    'subsample' : [0.7,0.75],

    'reg_alpha' : [1,1.2],

    'reg_lambda' : [1,1.2,1.4],

    }

    lgbm_model = lightgbm.LGBMClassifier(n_jobs=-1)

    lgbm_random = RandomizedSearchCV(estimator = lgbm_model, param_distributions = random_grid, n_iter = n_iter, cv = 3, verbose=1, random_state=111, n_jobs = -1, scoring = {'NLL':'neg_log_loss', 'Accuracy':'accuracy'}, refit='NLL', return_train_score=False)

    # Fit the random search model

    lgbm_random.fit(X, y)

    return lgbm_random.best_estimator_, lgbm_random.cv_results_
n_iter=150

X,y=train.drop(['Cover_Type'], axis=1), train['Cover_Type']

m,s = X.mean(0), X.std(0)

s[s==0]=1

X = (X-m)/s

trained_model, results=random_lgbm(X,y,n_iter=n_iter)
plt.figure(figsize=(13, 13))

plt.title("RandomSearchCV evaluating using multiple scorers simultaneously",

          fontsize=16)



plt.xlabel("iters")

plt.ylabel("Score")



ax = plt.gca()

#ax.set_xlim(0, 402)

ax.set_ylim(0.2, 1.2)



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

output.to_csv('lgbm_submission.csv', index=False)