%ls ../input
import pandas as pd

from pathlib import Path
root = Path("../input")
train = pd.read_csv(root.joinpath("train.csv"))

test = pd.read_csv(root.joinpath("test.csv"))
train_id = train.ID_code

test_id = test.ID_code

target = train.target

train.drop(columns=["ID_code", "target"], inplace=True)

test.drop(columns=["ID_code"], inplace=True)
from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score

from catboost import Pool, CatBoostClassifier
model = CatBoostClassifier(loss_function="Logloss",

                           eval_metric="AUC",

                           task_type="GPU",

                           learning_rate=0.01,

                           iterations=10000,

                           random_seed=42,

                           od_type="Iter",

                           depth=10,

                           early_stopping_rounds=500

                          )
n_split = 5

kf = KFold(n_splits=n_split, random_state=42, shuffle=True)
y_valid_pred = 0 * target

y_test_pred = 0
for idx, (train_index, valid_index) in enumerate(kf.split(train)):

    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]

    X_train, X_valid = train.iloc[train_index,:], train.iloc[valid_index,:]

    _train = Pool(X_train, label=y_train)

    _valid = Pool(X_valid, label=y_valid)

    print( "\nFold ", idx)

    fit_model = model.fit(_train,

                          eval_set=_valid,

                          use_best_model=True,

                          verbose=200,

                          plot=True

                         )

    pred = fit_model.predict_proba(X_valid)[:,1]

    print( "  auc = ", roc_auc_score(y_valid, pred) )

    y_valid_pred.iloc[valid_index] = pred

    y_test_pred += fit_model.predict_proba(test)[:,1]

y_test_pred /= n_split
submission = pd.read_csv(root.joinpath("sample_submission.csv"))

submission['target'] = y_test_pred

submission.to_csv('submission.csv', index=False)