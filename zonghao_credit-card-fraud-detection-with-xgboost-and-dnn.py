import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../input/creditcardfraud/creditcard.csv")
df.head()
X = df.iloc[:, 1:-1].values
y = df['Class'].values
count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar', color=['C0','C1'])
plt.title('Counts of Fraud/Normal')
plt.xticks(range(2), ['Normal', 'Fraud'], rotation = 0)
plt.xlabel("Class")
plt.ylabel("Counts")
plt.show()
scale_pos_weight = y.shape[0] / y.sum() - 1
print('Counts of normal transactions / Counts of fraudulent transactions :', scale_pos_weight)
from mlxtend.plotting import heatmap

cm = np.corrcoef(df.values.T)
hm = heatmap(cm, row_names = df.columns, column_names = df.columns, figsize = (20, 20))
plt.title('Correlations Between the Different Features of the Data', fontsize = 20)
plt.show()
import matplotlib.pyplot as plt

plt.scatter(X[y == 0, 13], X[y == 0, 16], marker = 'o', linewidth=1, edgecolor='black', label = 'Normal')
plt.scatter(X[y == 1, 13], X[y == 1, 16], marker = 'o', linewidth=1, edgecolor='black', label = 'Fraud')
plt.title('Fraudulent and Normal Transactions')
plt.xlabel('V14')
plt.ylabel('V17')
plt.legend()
plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = y)
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y_train)
dtest = xgb.DMatrix(X_test, label = y_test)

param = { 'verbosity': 2,
          'objective': 'binary:logistic',
          'eval_metric': 'aucpr',
          'scale_pos_weight': scale_pos_weight,
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'tree_method': 'gpu_hist',
          'eta': 0.1,
          'max_depth': 5,
          'gamma': 0,
          'min_child_weight': 1 }

bst = xgb.cv(param, dtrain, nfold = 3, num_boost_round = 1000, early_stopping_rounds = 50)
import matplotlib.pyplot as plt

plt.plot(bst.iloc[:, 0], label = 'train')
plt.plot(bst.iloc[:, 2], label = 'test')
plt.legend(loc = 'lower right')
plt.xlabel('Runs')
plt.ylabel('AUPRC')
plt.title('The Area Under the Precision-Recall Curve (AUPRC)')
plt.show()
best_xgb = xgb.train(param, dtrain, num_boost_round = bst.shape[0])

fig, ax = plt.subplots(figsize = (6, 8))
xgb.plot_importance(best_xgb, ax = ax)
plt.show()
fig, ax = plt.subplots(figsize = (25, 10))
xgb.plot_tree(best_xgb, ax = ax)
plt.show()
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

param = {'learning_rate': 0.1,
         'verbosity': 2,
         'objective': 'binary:logistic',
         'tree_method': 'gpu_hist',
         'scale_pos_weight': scale_pos_weight,
         'n_estimators': 300}
xgb_grid = {'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0],
            'subsample': [0.8],
            'colsample_bytree': [0.8] }
xgbc = XGBClassifier(**param)
xgbc_cv = GridSearchCV(estimator = xgbc, param_grid = xgb_grid, cv = 3, scoring = 'average_precision', n_jobs = -1, verbose = 2)
xgbc_cv.fit(X_train, y_train)
print('Best parameters: ', xgbc_cv.best_params_)
print('Best score: ', xgbc_cv.best_score_)
best_xgbc = XGBClassifier(scale_pos_weight = scale_pos_weight,
                          objective = 'binary:logistic',
                          tree_method = 'gpu_hist',
                          max_depth = 6,
                          min_child_weight = 1,
                          gamma = 0,
                          subsample = 0.6,
                          colsample_bytree = 0.6,
                          alpha = 0,
                          learning_rate = 0.01,
                          n_estimators = 2000)

best_xgbc.fit(X_train, y_train)
y_pred = best_xgbc.predict(X_train)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

y_pred_proba_xgb = best_xgbc.predict_proba(X_test)[:, 1]
precision_xgb, recall_xgb, threshold_xgb = precision_recall_curve(y_test, y_pred_proba_xgb)
plt.plot(recall_xgb, precision_xgb, label = 'XGBoost (PRAUC = {:.3f})'.format(auc(recall_xgb, precision_xgb)))
plt.title('The Precison-Recall Curve of the XGBoost model')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.show()
from sklearn.model_selection import train_test_split

X = df.iloc[:, 1:-1].values
y = df['Class'].values
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = y)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 0.2, random_state = 1, stratify = y_train_val)
from sklearn.utils import resample

def Balance(X, y, random_state = None):
    X_minority = X[y == 1]
    X_majority = X[y == 0]
    
    X_minority_resample = resample(X_minority, replace = True, n_samples = X_majority.shape[0], random_state = random_state)
    X_resampled = np.vstack((X_minority_resample, X_majority))
    
    y_minority_resampled = np.ones((X_majority.shape[0], 1), dtype = int)
    y_majority = np.zeros((X_majority.shape[0], 1), dtype = int)
    y_resampled = np.vstack((y_minority_resampled, y_majority))
    
    data = np.hstack((X_resampled, y_resampled))
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1].astype(int)
    return X, y
X_train_all, y_train_all = Balance(X_train_val, y_train_val, random_state = 1)
X_train_balance, y_train_balance = Balance(X_train, y_train, random_state = 1)
X_val_balance, y_val_balance = Balance(X_val, y_val, random_state = 1)
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(32, activation = 'relu', input_shape = (29,)))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()
from keras.utils.vis_utils import plot_model

plot_model(model)
from keras import optimizers
from keras import metrics

model.compile(optimizer = optimizers.RMSprop(), loss = 'binary_crossentropy', metrics = [metrics.AUC(curve = 'PR')])
history = model.fit(X_train_balance, y_train_balance, epochs = 20, batch_size = 512, validation_data = (X_val, y_val))
train_auc = history.history['auc']
val_auc = history.history['val_auc']
epochs = range(1, len(train_auc) + 1)

plt.plot(epochs, train_auc, 'bo', label = 'Training AUPRC')
plt.plot(epochs, val_auc, 'b', label = 'Validation AUPRC')
plt.title('Training and validation AUPRC')
plt.xlabel('Epochs')
plt.ylabel('AUPRC')
plt.ylim([0.5, 1.05])
plt.legend(loc = 'lower right')
plt.show()
from keras import models
from keras import layers
from keras import optimizers
from keras import metrics

model = models.Sequential()
model.add(layers.Dense(32, activation = 'relu', input_shape = (29,)))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(8, activation = 'relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(4, activation = 'relu'))
model.add(layers.Dense(1, activation = 'sigmoid'))
model.summary()
model.compile(optimizer = optimizers.RMSprop(), loss = 'binary_crossentropy', metrics = [metrics.AUC(curve = 'PR')])
history = model.fit(X_train_all, y_train_all, epochs = 5, batch_size = 512)
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

y_pred_proba = model.predict(X_test)
precision_dnn, recall_dnn, threshold_dnn = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_dnn, precision_dnn, label = 'DNN (PRAUC = {:.3f})'.format(auc(recall_dnn, precision_dnn)))
plt.title('The Precison-Recall Curve of the Deep Neural Network (DNN)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.show()
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

y_pred_proba = 0.9 * best_xgbc.predict_proba(X_test)[:, 1][:, np.newaxis] + 0.1 * model.predict(X_test)
precision, recall, threshold = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall, precision, label = 'Ensemble (PRAUC = {:.3f})'.format(auc(recall, precision)))
plt.title('The Precison-Recall Curve of the Ensemble of XGBoost and DNN')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.show()
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc

y_pred_proba_xgb = best_xgbc.predict_proba(X_test)[:, 1][:, np.newaxis]
y_pred_proba_dnn = model.predict(X_test)
y_pred_proba = 0.9 * y_pred_proba_xgb + 0.1 * y_pred_proba_dnn
precision_xgb, recall_xgb, threshold_xgb = precision_recall_curve(y_test, y_pred_proba_xgb)
precision_dnn, recall_dnn, threshold_dnn = precision_recall_curve(y_test, y_pred_proba_dnn)
precision, recall, threshold = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_xgb, precision_xgb, label = 'XGBoost (PRAUC = {:.3f})'.format(auc(recall_xgb, precision_xgb)))
plt.plot(recall_dnn, precision_dnn, label = 'DNN (PRAUC = {:.3f})'.format(auc(recall_dnn, precision_dnn)))
plt.plot(recall, precision, label = 'Ensemble (PRAUC = {:.3f})'.format(auc(recall, precision)))
plt.title('The Precison-Recall Curves of the XGBoost, DNN, and Ensemble Models')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc = 'lower left')
plt.show()