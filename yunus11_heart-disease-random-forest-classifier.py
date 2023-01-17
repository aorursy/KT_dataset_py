import os

import pandas as pd



HEART_PATH = "../input/"



def load_housing_data(heart_path=HEART_PATH):

    csv_path = os.path.join(heart_path, "heart.csv")

    return pd.read_csv(csv_path)
heart = load_housing_data()

heart.info()
%matplotlib inline

import matplotlib.pyplot as plt



heart.hist(bins=50, figsize=(20,15))

plt.show()
from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(heart, heart["target"]):

    strat_train_set = heart.loc[train_index]

    strat_test_set = heart.loc[test_index]
heart = strat_train_set.copy()

corr = heart.corr()

corr["target"].sort_values(ascending=False)
heart = strat_train_set.drop("target", axis=1)

heart_labels = strat_train_set["target"].copy()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(heart)

heart_scaled = scaler.transform(heart)

heart_scaled
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV



param_grid = [

        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

        ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,

                            scoring='neg_mean_squared_error')

grid_search.fit(heart_scaled, heart_labels)
from sklearn.metrics import precision_score, recall_score

import numpy as np



heart_test = strat_train_set.drop("target", axis=1)

heart_test_labels = strat_train_set["target"].copy()



scaler = StandardScaler()

scaler.fit(heart_test)

heart_test_scaled = scaler.transform(heart_test)



y_pred = grid_search.best_estimator_.predict(heart_test_scaled)

print("Precision: {:.2f}%".format(100 * precision_score(heart_test_labels.values, np.around(y_pred))))

print("Recall: {:.2f}%".format(100 * recall_score(heart_test_labels.values, np.around(y_pred))))