# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_validate



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/sarcasm-detection-2020-mentoring-proj-part-1/sarcasm_prepped_data.csv")

data.head(10)
data["label"].value_counts()
train_data, test_data = train_test_split(data, test_size=0.25, random_state=42)

test_data.shape
train_data.head()
basic_train_y = train_data["label"]

basic_train_X = train_data[["score", "comment_length", "date_year", "date_month", "comment_day"]]



basic_test_y = test_data["label"]

basic_test_X = test_data[["score", "comment_length", "date_year", "date_month", "comment_day"]]



basic_train_X.head()
from sklearn.linear_model import LogisticRegression



log_reg_model = LogisticRegression(random_state=42)

log_reg_model = log_reg_model.fit(basic_train_X, basic_train_y)

log_reg_model
cross_validate(log_reg_model, basic_train_X, basic_train_y, cv=5, scoring="accuracy")
from sklearn.model_selection import GridSearchCV



log_reg_model = LogisticRegression(random_state=42, penalty="elasticnet", solver="saga", max_iter=2000, n_jobs=-1)

param_grid = {"l1_ratio": [0.0, 0.25, 0.50, 0.75, 1.0]}



grid = GridSearchCV(log_reg_model, param_grid, scoring="accuracy", cv=3, n_jobs=-1)

grid.fit(basic_train_X, basic_train_y)
print(grid.best_score_)

print(grid.best_params_)
log_reg_model = LogisticRegression(random_state=42, penalty="elasticnet", l1_ratio=0.0, solver="saga", max_iter=2000, n_jobs=-1)

log_reg_model = log_reg_model.fit(basic_train_X, basic_train_y)

score = log_reg_model.score(basic_test_X, basic_test_y)

score
train_data.to_csv("sarcasm_train_split.csv", index=False)

test_data.to_csv("sarcasm_test_split.csv", index=False)