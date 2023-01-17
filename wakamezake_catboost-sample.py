import catboost

import numpy as np

import pandas as pd

from catboost import Pool, cv

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split, StratifiedKFold
catboost.__version__
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
train_df.head()
train_df.fillna(-999, inplace=True)

test_df.fillna(-999, inplace=True)
y = train_df["Survived"]

X = train_df.drop('Survived', axis=1)
X.shape, y.shape
X.head()
categorical_features_indices = np.where(X.dtypes != np.float)[0]
categorical_features_indices
train_pool = Pool(data=X,

                  label=y,

                  cat_features=categorical_features_indices)

# valid_pool = Pool(data=X_val,

#                   label=y_val,

#                   cat_features=categorical_features_indices)
params = {"iterations": 100,

          "depth": 2,

          "loss_function": "Logloss",

          "eval_metric" : "AUC",

          "verbose": False}
scores = cv(train_pool,

            params,

            stratified=True,

            fold_count=n_splits,

            plot=True)
scores
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=1234)
model = catboost.CatBoostClassifier(use_best_model=True,

                                  eval_metric = 'AUC', random_seed=42)

model.fit(X_train, y_train, 

        cat_features=categorical_features_indices,

        eval_set=(X_val, y_val),

         verbose_eval=500)
y_val_pred = model.predict(X_val)

print('accuracy : %.2f' % roc_auc_score(y_val_pred, y_val))
n_splits = 4

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
params = {"use_best_model": True,

          "eval_metric": "AUC", 

          "random_seed": 42}
def cat_cv(train, target, test, cat_features, train_params, fold_schema):

    oof = np.zeros(len(train))

    predictions = np.zeros(len(test))

    for fold_idx, (trn_idx, val_idx) in enumerate(fold_schema.split(train, target)):

        print('Fold {}/{}'.format(fold_idx + 1, fold_schema.n_splits))

        train_pool = Pool(train.iloc[trn_idx], label=target.iloc[trn_idx],

                          cat_features=cat_features)

        valid_pool = Pool(train.iloc[val_idx], label=target.iloc[val_idx],

                          cat_features=cat_features)

        model = catboost.CatBoostClassifier(**params)

        model.fit(train_pool, eval_set=valid_pool, verbose_eval=500)

        oof[val_idx] = model.predict(train.iloc[val_idx])

        predictions += model.predict(test) / fold_schema.n_splits

        print("AUC: {}".format(roc_auc_score(target[val_idx], oof[val_idx])))

    return predictions
predictions = cat_cv(train=X, target=y, test=test_df,

                     cat_features=categorical_features_indices,

                     train_params=params, fold_schema=skf)
predictions[predictions >= 0.5] = 1

predictions[predictions < 0.5] = 0
test = pd.DataFrame()

test['PassengerId'] = test_df['PassengerId']

test['Survived'] = predictions.astype(int)

# 予測値をintに変換

test['Survived'] = test['Survived'].astype('int')

# 保存

test.to_csv('./benchmark_catboost.csv', index=False)