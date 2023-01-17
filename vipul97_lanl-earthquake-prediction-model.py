import numpy as np

import pandas as pd

import warnings



warnings.filterwarnings("ignore")

warnings.simplefilter(action='ignore', category=FutureWarning)



from catboost import CatBoostRegressor

from joblib import Parallel, delayed

from tensorflow import keras



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler
X_train_1 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_X_features_865.csv")

X_train_2 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_X_features_865.csv")

y_1 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_y.csv", index_col=False, header=None)

y_2 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_y.csv", index_col=False, header=None)
X_train = pd.concat([X_train_1, X_train_2], axis=0)

X_train = X_train.reset_index(drop=True)

X_train.shape
X_train.head()
y = pd.concat([y_1, y_2], axis=0)

y = y.reset_index(drop=True)

y.shape
y_train = pd.Series(y[0].values)
X_test = pd.read_csv("../input/lanl-master-s-features-creating-0/test_X_features_10.csv")
scaler = StandardScaler()

train_columns = X_train.columns



X_train[train_columns] = scaler.fit_transform(X_train[train_columns])

X_test[train_columns] = scaler.transform(X_test[train_columns])
train_columns = X_train.columns

n_fold = 5
%%time

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



oof = np.zeros(len(X_train))

train_score = []

cat_predictions = np.zeros(len(X_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train.values)):

    print("fold {}".format(fold_))



    X_tr, X_val = X_train[train_columns].iloc[trn_idx], X_train[train_columns].iloc[val_idx]

    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]



    model = CatBoostRegressor(n_estimators=25000, verbose=-1, objective="MAE", loss_function="MAE", boosting_type="Ordered", task_type="GPU")

    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=2500, early_stopping_rounds=500)

    oof[val_idx] = model.predict(X_val)



    cat_predictions += model.predict(X_test[train_columns]) / folds.n_splits

    train_score.append(model.best_score_['learn']["MAE"])



cv_score = mean_absolute_error(y_train, oof)

print(f"After {n_fold}: test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}", end=" ")
def create_model(input_dim=10):

    model = keras.Sequential()

    model.add(keras.layers.Dense(256, activation="relu", input_dim=input_dim))

    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(128, activation="relu"))

    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(96, activation="relu"))

    model.add(keras.layers.Dropout(0.3))

    model.add(keras.layers.Dense(1, activation="linear"))



    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    model.compile(optimizer=optimizer, loss='mae')



    return model



patience = 50

call_ES = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
%%time

folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)



NN_oof = np.zeros(len(X_train))

train_score = []

NN_predictions = np.zeros(len(X_test))



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train.values)):

    print("fold {}".format(fold_))

    

    X_tr, X_val = X_train[train_columns].iloc[trn_idx], X_train[train_columns].iloc[val_idx]

    y_tr, y_val = y_train.iloc[trn_idx], y_train.iloc[val_idx]

    

    model = create_model(X_train.shape[-1])

    model.fit(X_tr, y_tr, epochs=500, batch_size=32, verbose=0, callbacks=[call_ES,], validation_data=[X_val, y_val])

    

    NN_oof[val_idx] = model.predict(X_val)[:, 0]

    

    NN_predictions += model.predict(X_test[train_columns])[:, 0] / folds.n_splits

    history = model.history.history

    tr_loss = history["loss"]

    val_loss = history["val_loss"]

    print(f"loss: {tr_loss[-patience]:.3f} | val_loss: {val_loss[-patience]:.3f} | diff: {val_loss[-patience]-tr_loss[-patience]:.3f}")

    train_score.append(tr_loss[-patience])

    

cv_score = mean_absolute_error(y_train, NN_oof)

print(f"After {n_fold}: test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}", end=" ")
Scirpus_predictions = pd.read_csv("../input/andrews-new-script-plus-a-genetic-program-model/gpI.csv")

Scirpus_predictions.head()
submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')



submission["time_to_failure"] = (cat_predictions + NN_predictions + Scirpus_predictions.time_to_failure.values) / 3

submission.to_csv('submission.csv', index=False)

submission.head()