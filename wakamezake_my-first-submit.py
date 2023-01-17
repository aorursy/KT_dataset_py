%ls ../input
import pandas as pd

import seaborn as sns

from pathlib import Path

%matplotlib inline
root = Path("../input")

train = pd.read_csv(root.joinpath("train.csv"))

test = pd.read_csv(root.joinpath("test.csv"))

submit = pd.read_csv(root.joinpath("sample_submission.csv"))
import lightgbm as lgb

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train.drop(columns=["id", "target"]), train.target, test_size=0.2, random_state=42)
param = {

    "objective": "binary",

    "seed": 4032

}

num_round = 1000
lgb_train = lgb.Dataset(X_train, label=y_train)

lgb_val = lgb.Dataset(X_val, label=y_val)
model = lgb.train(param, lgb_train, num_round, valid_sets = [lgb_train, lgb_val], verbose_eval=-1, early_stopping_rounds = 200)
pred = model.predict(test.drop(columns=["id"]))
from sklearn.metrics import roc_auc_score
pred[pred >= 0.5] = 1

pred[pred < 0.5] = 0
submit.target = pred
submit.to_csv("submit.csv", index=False)