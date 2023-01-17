import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from zipfile import ZipFile

zip_file = ZipFile('../input/the-winton-stock-market-challenge/train.csv.zip')
df = pd.read_csv(zip_file.open('train.csv'))
df.head()
zip_file = ZipFile('../input/the-winton-stock-market-challenge/test_2.csv.zip')
new_df = pd.read_csv(zip_file.open('test_2.csv'))
new_df.head()
fig, ax = plt.subplots(figsize = (15, 5))
df_na = (df.isnull().sum() / len(df))
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending = False)[: 10]
ax.bar(range(df_na.size), df_na, width = 0.5)
plt.xticks(range(df_na.size), df_na.index, rotation = 0)
plt.ylim([0, 1])
plt.title('Top ten features with the most missing values')
plt.ylabel('Missing ratio')
plt.show()
ret = df.loc[:, 'Ret_2':'Ret_120'].sum(1)
new_ret = new_df.loc[:, 'Ret_2':'Ret_120'].sum(1)

X = np.hstack((df.loc[:, 'Feature_1':'Ret_MinusOne'].values, ret.values[:, np.newaxis]))
ts = df.loc[:, 'Ret_2':'Ret_120'].values
y = df.loc[:, 'Ret_PlusOne':'Ret_PlusTwo'].values
y_ts = df.loc[:, 'Ret_121':'Ret_180'].values

new_X = np.hstack((new_df.loc[:, 'Feature_1':'Ret_MinusOne'].values, new_ret.values[:, np.newaxis]))
new_ts = new_df.loc[:, 'Ret_2':'Ret_120'].values
from sklearn.impute import SimpleImputer

imr = SimpleImputer(strategy = 'mean')
X = imr.fit_transform(X)
new_X = imr.transform(new_X)

ts = imr.fit_transform(ts)
new_ts = imr.transform(new_ts)
fig, ax = plt.subplots(figsize = (10, 5))
plt.plot(ts[0, :], label = 'Stock 1')
plt.plot(ts[1, :], label = 'Stock 2')
plt.plot(ts[2, :], label = 'Stock 3')
plt.ylim([-0.004, 0.004])
plt.xlim([0, 118])
plt.ylabel('Stock returns')
plt.xlabel('Minutes')
plt.title('Examples of the stock returns in the first 120 minutes of the current day')
plt.legend()
plt.show()
from mlxtend.plotting import heatmap

cm = np.corrcoef(np.hstack([X, y]).T)
cols = list(df.columns[1:28]) + ['Ret', 'Ret_PlusOne', 'Ret_PlusTwo']
hm = heatmap(cm, row_names = cols, column_names = cols, figsize = (20, 20))
plt.title('Correlations Between the Different Features of the Data', fontsize = 20)
plt.show()
plt.scatter(df['Feature_3'], df['Feature_11'], marker = 'o', s = 1) #, linewidth = 1, edgecolor = 'black')
plt.title('Correlation between Feature_3 and Feature_11')
plt.xlabel('Feature_3')
plt.ylabel('Feature_11')
plt.show()
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA

#scaler = StandardScaler()
#X[:, :25] = scaler.fit_transform(X[:, :25])
#pca = PCA(n_components = 25)
#X[:, :25] = pca.fit_transform(X[:, :25])
X_train = X[:30000, :]
X_val = X[30000:35000, :]
X_test = X[35000:, :]
y_train = y[:30000, :]
y_val = y[30000:35000, :]
y_test = y[35000:, :]
X_train_val = X[:35000, :]
y_train_val = y[:35000, :]
y1_train = y_train[:, 0]
y2_train = y_train[:, 1]
y1_val = y_val[:, 0]
y2_val = y_val[:, 1]
y1_test = y_test[:, 0]
y2_test = y_test[:, 1]
y1_train_val = y_train_val[:, 0]
y2_train_val = y_train_val[:, 1]
y1 = y[:, 0]
y2 = y[:, 1]
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label = y1_train)

param = { 'verbosity': 0,
          'objective': 'reg:pseudohubererror',
          'eval_metric': 'mae',
          'subsample': 0.8,
          'colsample_bytree': 0.8,
          'tree_method': 'gpu_hist',
          'eta': 0.1,
          'max_depth': 5,
          'gamma': 0,
          'min_child_weight': 1 }

bst = xgb.cv(param, dtrain, nfold = 3, num_boost_round = 1000, early_stopping_rounds = 50)
fig = plt.figure(figsize = (12, 4))
fig.suptitle('The Mean Absolute Error (MAE) of the training data and validation data')
plt.subplot(121)
plt.plot(bst['train-mae-mean'], label = 'train')
plt.plot(bst['test-mae-mean'], label = 'validation')
plt.xlabel('Runs')
plt.ylabel('MAE')
plt.legend()
plt.subplot(122)
plt.plot(bst['train-mae-mean'], label = 'train')
plt.plot(bst['test-mae-mean'], label = 'validation')
plt.yscale('log')
plt.xlabel('Runs')
plt.ylabel('MAE in log scale')
plt.legend()
plt.subplots_adjust(wspace = 0.3)
plt.show()
xgb_med = xgb.train(param, dtrain, num_boost_round = bst.shape[0])

fig, ax = plt.subplots(figsize = (6, 8))
xgb.plot_importance(xgb_med, ax = ax)
plt.show()
fig, ax = plt.subplots()
xgb.plot_tree(xgb_med, ax = ax)
plt.show()
select = [1, 5, 6, 13, 14, 18, 22, 25, 26, 27]

X_train = X[:30000, select]
X_val = X[30000:35000, select]
X_test = X[35000:, select]
X_train_val = X[:35000, select]
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

def mysearch(X_train, y_train, X_val, y_val, param, param1, param2 = None, estimator = XGBRegressor, score = mean_absolute_error):
    best_score = 10000000.
    best_param = { **param }
    para = { **param }
    key1 = list(param1.keys())[0]
    if param2 is not None:
        key2 = list(param2.keys())[0]
        for parama in param1[key1]:
            for paramb in param2[key2]:
                para[key1] = parama
                para[key2] = paramb
                est = estimator(**para)
                est.fit(X_train, y_train)
                y_pred = est.predict(X_val)
                current_score = score(y_pred, y_val)
                #print('The current score: ', current_score)
                #print('The current parameter: {} = {}, {} = {}'.format(key1, parama, key2, paramb))
                if (current_score < best_score):
                    best_score = current_score
                    best_param[key1] = parama
                    best_param[key2] = paramb
        #print('The best score: ', best_score)
        #print('The best parameter: {} = {}, {} = {}'.format(key1, best_param[key1], key2, best_param[key2]))
    else:
        for parama in param1[key1]:
            para[key1] = parama
            est = estimator(**para)
            est.fit(X_train, y_train)
            y_pred = est.predict(X_val)
            current_score = score(y_pred, y_val)
            print('The current score: ', current_score)
            print('The current parameter: {} = {}'.format(key1, parama))
            if (current_score < best_score):
                best_score = current_score
                best_param[key1] = parama
        #print('The best score: ', best_score)
        #print('The best parameter: {} = {}'.format(key1, best_param[key1]))
    return best_score, best_param
param = {'learning_rate': 0.1,
         'verbosity': 0,
         'objective': 'reg:pseudohubererror',
         'tree_method': 'gpu_hist',
         'n_estimators': 100,
         'n_jobs': -1,
         'gamma': 0,
         'subsample': 0.8,
         'colsample_bytree': 0.8,
         'alpha': 0}
param1 = { 'max_depth': [1, 3, 5] }
param2 = { 'min_child_weight': [1, 3, 5] }
score, bst_param = mysearch(X_train, y1_train, X_val, y1_val, param, param1, param2)
print('The best score is:', score)
print('The best parameter is:', bst_param)
best_xgbr = XGBRegressor( objective = 'reg:pseudohubererror',
                          tree_method = 'gpu_hist',
                          max_depth = 5,
                          min_child_weight = 3,
                          gamma = 0,
                          subsample = 0.9,
                          colsample_bytree = 0.9,
                          alpha = 0,
                          learning_rate = 0.01,
                          n_estimators = 700)
best_xgbr.fit(X_train_val, y1_train_val)
y1_test_pred = best_xgbr.predict(X_test)

best_xgbr.fit(X_train_val, y2_train_val)
y2_test_pred = best_xgbr.predict(X_test)
from sklearn.metrics import mean_absolute_error

benchmark = [np.abs(y1_test).mean(), np.abs(y2_test).mean()]
fitted = [mean_absolute_error(y1_test_pred, y_test[:, 0]), mean_absolute_error(y2_test_pred, y_test[:, 1])]

index = np.arange(2)
bar_width = 0.35
plt.bar(index, benchmark, bar_width, label = 'All-zero prediction')
plt.bar(index + bar_width, fitted, bar_width, label = 'XGBoost regressor')
plt.xticks(index + bar_width / 2, ['Ret_PlusOne', 'Ret_PlusTwo'])
plt.title('Comparison of the XGBoost regressor and the all-zero prediction')
plt.xlabel('Stock returns in the two following days')
plt.ylabel('MAE of the pridiction')
plt.ylim([0, 0.021])
plt.legend()
plt.show()
best_xgbr.fit(X[:, select], y[:, 0])
y1_new_pred = best_xgbr.predict(new_X[:, select])

best_xgbr.fit(X[:, select], y[:, 1])
y2_new_pred = best_xgbr.predict(new_X[:, select])
y1_new_pred = 0.1 * y1_new_pred + 0.9 * np.median(y[:, 0])
y2_new_pred = 0.1 * y2_new_pred + 0.9 * np.median(y[:, 1])

ts_new_pred = np.zeros((new_X.shape[0], 60))

zip_file = ZipFile('../input/the-winton-stock-market-challenge/sample_submission_2.csv.zip')
sub = pd.read_csv(zip_file.open('sample_submission_2.csv'))
sub['Predicted'] = np.hstack([ts_new_pred, y1_new_pred[:, np.newaxis], np.median(y[:, 1])* np.ones((new_X.shape[0], 1))]).flatten()
#sub.to_csv('submission.csv', index = False)