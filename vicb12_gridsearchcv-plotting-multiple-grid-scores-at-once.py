from sklearn import tree

from sklearn import model_selection

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.datasets import load_digits

import warnings

warnings.filterwarnings('ignore')



digits = load_digits()

X, y = digits.data, digits.target

Algo = [['DecisionTreeClassifier', tree.DecisionTreeClassifier(),  # algorithm

             'max_depth', [1, 2, 4, 6, 8, 10, 12, 14, 18, 20, 22, 24, 26, 28, 30],  # Parameter1

             'max_features', ['sqrt', 'log2', None],  # Parameter2

                 'criterion', ['gini', 'entropy']]]  # Parameter3





def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, title):

    # Get Test Scores Mean and std for each grid search



    grid_param_1 = list(str(e) for e in grid_param_1)

    grid_param_2 = list(str(e) for e in grid_param_2)

    scores_mean = cv_results['mean_test_score']

    scores_std = cv_results['std_test_score']

    params_set = cv_results['params']



    scores_organized = {}

    std_organized = {}

    std_upper = {}

    std_lower = {}

    for p2 in grid_param_2:

        scores_organized[p2] = []

        std_organized[p2] = []

        std_upper[p2] = []

        std_lower[p2] = []

        for p1 in grid_param_1:

            for i in range(len(params_set)):

                if str(params_set[i][name_param_1]) == str(p1) and str(params_set[i][name_param_2]) == str(p2):

                    mean = scores_mean[i]

                    std = scores_std[i]

                    scores_organized[p2].append(mean)

                    std_organized[p2].append(std)

                    std_upper[p2].append(mean + std)

                    std_lower[p2].append(mean - std)



    _, ax = plt.subplots(1, 1)



    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)

    # plot means

    for key in scores_organized.keys():

        ax.plot(grid_param_1, scores_organized[key], '-o', label= name_param_2 + ': ' + str(key))

        ax.fill_between(grid_param_1, std_lower[key], std_upper[key], alpha=0.1)



    ax.set_title(title)

    ax.set_xlabel(name_param_1)

    ax.set_ylabel('CV Average Score')

    ax.legend(loc="best")

    ax.grid('on')

    plt.show()



dataset = 'Digits'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cv_split = model_selection.KFold(n_splits=10, random_state=2)



for i in range(len(Algo)):



    name = Algo[0][0]

    alg = Algo[0][1]

    param_1_name = Algo[0][2]

    param_1_range = Algo[0][3]

    param_2_name = Algo[0][4]

    param_2_range = Algo[0][5]

    param_3_name = Algo[0][6]

    param_3_range = Algo[0][7]



    for p in param_3_range:

        # grid search

        param = {

            param_1_name: param_1_range,

            param_2_name: param_2_range,

            param_3_name: [p]

        }

        grid_test = GridSearchCV(alg, param_grid=param, scoring='accuracy', cv=cv_split)

        grid_test.fit(X_train, y_train)

        plot_grid_search(grid_test.cv_results_, param[param_1_name], param[param_2_name], param_1_name, param_2_name, dataset + ' GridSearch Scores: ' + name + ', ' + param_3_name + '=' + str(p))



    param = {

        param_1_name: param_1_range,

        param_2_name: param_2_range,

        param_3_name: param_3_range

    }

    grid_final = GridSearchCV(alg, param_grid=param, scoring='accuracy', cv=cv_split)

    grid_final.fit(X_train, y_train)

    best_params = grid_final.best_params_

    alg.set_params(**best_params)

    print('best_params:', best_params)
