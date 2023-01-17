# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.



import matplotlib.pyplot as plt

# Pretty display for notebooks

%matplotlib inline



import seaborn as sns

sns.set(style="darkgrid")
data = pd.read_csv("../input/Admission_Predict.csv")
data.head()
data.describe()
data.columns
# Fix space in last column

data.columns = [c.strip() for c in data.columns]
data.columns
sns.regplot(x="GRE Score", y="Chance of Admit", data=data)
sns.pairplot(data.iloc[:, 1:], kind="reg")
sns.pairplot(data.iloc[:, 1:])
from sklearn.model_selection import train_test_split



features = data[["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]]

chances = data["Chance of Admit"]



X_train, X_test, y_train, y_test = train_test_split(features, chances, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import learning_curve



depths = [1,2,3,4,6,10]



fig = plt.figure(figsize=(15,8))



for k, depth in enumerate(depths):



    regressor = DecisionTreeRegressor(max_depth = depth)



    train_sizes, train_scores, test_scores = learning_curve(regressor, X_train, y_train, 

                                                            cv=10, n_jobs=-1, scoring = 'r2')



    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    # Plot

    ax = fig.add_subplot(2, 3, k+1)

    ax.set_title(f"max_depth = {depth}")

    ax.set_xlabel("Training examples")

    ax.set_ylabel("Score")

    ax.set_ylim([-0.05, 1.05])

#     plt.grid()



    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")

    

fig.tight_layout()
from sklearn.model_selection import GridSearchCV



regressor = DecisionTreeRegressor(random_state=42)

params = {"max_depth": range(1,11)}



grid = GridSearchCV(regressor, params, scoring='r2', cv=10)

grid = grid.fit(X_train, y_train)



print(grid.best_params_)

print(grid.best_score_)

# print(grid.cv_results_)
best_estimator = grid.best_estimator_
predictions = best_estimator.predict(X_test)

predictions
from sklearn.metrics import r2_score



r2_score(y_test, predictions)