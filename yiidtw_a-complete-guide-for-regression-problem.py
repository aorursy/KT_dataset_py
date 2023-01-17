import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
train_csv = pd.read_csv('../input/train.csv')
final_csv = pd.read_csv('../input/test.csv')
# I won't call it test set here, since we're going to create our test set and validation set
# and pick up the strongest model to make prediction in the "test.csv"
# first we wanna describe the target columns to make sure we're dealing with regression problem (predict value)
# and we'll look into more detail when we have to do feature engineering
train_csv['SalePrice'].describe()
print("How many feature candidates do we have? %d" % (len(train_csv.columns) - 1))
# first we'll visualize null count
null_in_train_csv = train_csv.isnull().sum()
null_in_train_csv = null_in_train_csv[null_in_train_csv > 0]
null_in_train_csv.sort_values(inplace=True)
null_in_train_csv.plot.bar()
# visualize correlation map
sns.heatmap(train_csv.corr(), vmax=.8, square=True);
arr_train_cor = train_csv.corr()['SalePrice']
idx_train_cor_gt0 = arr_train_cor[arr_train_cor > 0].sort_values(ascending=False).index.tolist()
print("How many feature candidates have positive correlation with SalePrice(including itself)? %d" % len(idx_train_cor_gt0))
# we shall list them all, and pick up those we're interested
arr_train_cor[idx_train_cor_gt0]
idx_meta = ['SalePrice','GrLivArea', 'MasVnrArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'OverallQual', 'Fireplaces', 'GarageCars']
train_meta = train_csv[idx_meta].copy()
train_meta.head(n=5)
null_in_masvnrarea = train_meta[train_meta['MasVnrArea'].isnull()].index.tolist()
zero_in_masvnrarea = train_meta['MasVnrArea'][train_meta['MasVnrArea'] == 0].index.tolist()
print("How many null value in MasVnrArea? %d / 1460" % len(null_in_masvnrarea))
print("How many zero value in MasVnrArea? %d / 1460" % len(zero_in_masvnrarea))
# we'll fill in the null value with 0 from the analysis above
train_meta['MasVnrArea'][null_in_masvnrarea] = 0
print("How many null value in MasVnrArea after filling in null value? %d / 1460" % train_meta['MasVnrArea'].isnull().sum())
# overview
sns.pairplot(train_meta)
# GrLivArea
train_meta[(train_meta['GrLivArea'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()
# TotalBsmtSF
train_meta[(train_meta['TotalBsmtSF'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()
train_meta[(train_meta['1stFlrSF'] > 4000) & (train_meta['SalePrice'] < 200000)].index.tolist()
# Thus, we'll remove [523, 1298]
train_clean = train_meta.drop([523,1298])
nonzero_in_masvnrarea = train_clean['MasVnrArea'][train_clean['MasVnrArea'] != 0].index.tolist()
print("How many non-zero value in MasVnrArea now? %d / 1458" % len(nonzero_in_masvnrarea))
# I'll categorize into zero and non-zero
train_clean['has_MasVnrArea'] = 0
train_clean['has_MasVnrArea'][nonzero_in_masvnrarea] = 1
train_clean['TotalBsmtSF'][train_clean['TotalBsmtSF'] > 0].describe()
bins_totalbsmtsf = [-1, 1, 1004, 4000]
train_clean['binned_TotalBsmtSF'] = np.digitize(train_clean['TotalBsmtSF'], bins_totalbsmtsf)
train_clean['1stFlrSF'].describe()
bins_1stflrsf = [0, 882, 1086, 1390, 4000]
train_clean['binned_1stFlrSF'] = np.digitize(train_clean['1stFlrSF'], bins_1stflrsf)
train_clean['2ndFlrSF'][train_clean['2ndFlrSF'] > 0].describe()
bins_2ndflrsf = [-1, 1, 625, 772, 924, 4000]
train_clean['binned_2ndFlrSF'] = np.digitize(train_clean['2ndFlrSF'], bins_2ndflrsf)
train_clean['SFcross'] = (train_clean['binned_TotalBsmtSF'] - 1) * (5 * 4) + (train_clean['binned_1stFlrSF'] - 1) * 5 + train_clean['binned_2ndFlrSF']
def draw2by2log(arr):
    fig = plt.figure();
    plt.subplot(2,2,1)
    sns.distplot(arr, fit=norm);
    plt.subplot(2,2,3)
    stats.probplot(arr, plot=plt);
    plt.subplot(2,2,2)
    sns.distplot(np.log(arr), fit=norm);
    plt.subplot(2,2,4)
    stats.probplot(np.log(arr), plot=plt);
draw2by2log(train_clean['SalePrice'])
draw2by2log(train_clean['GrLivArea'])
train_clean.head(n=5)
idx_tree = ['SalePrice', 'GrLivArea', 'OverallQual', 'Fireplaces', 'GarageCars', 'has_MasVnrArea', 'SFcross']
train_tree = train_clean[idx_tree]
train_tree.head(n=5)
sns.pairplot(train_tree)
print("Max Fireplaces value in train.csv: %d, in test.csv: %d" % (train_csv['Fireplaces'].max(), final_csv['Fireplaces'].max()) )
print("Min Fireplaces value in train.csv: %d, in test.csv: %d" % (train_csv['Fireplaces'].min(), final_csv['Fireplaces'].min()) )
print("Max GarageCars value in train.csv: %d, in test.csv: %d" % (train_csv['GarageCars'].max(), final_csv['GarageCars'].max()) )
print("Min GarageCars value in train.csv: %d, in test.csv: %d" % (train_csv['GarageCars'].min(), final_csv['GarageCars'].min()) )
dummy_fields = ['OverallQual', 'Fireplaces', 'GarageCars', 'has_MasVnrArea', 'SFcross']
train_dist = train_tree[['SalePrice', 'OverallQual', 'GrLivArea']].copy()
for field in dummy_fields:
    dummies = pd.get_dummies(train_tree.loc[:, field], prefix=field)
    train_dist = pd.concat([train_dist, dummies], axis = 1)
train_dist['GarageCars_5'] = 0
train_dist['Fireplaces_4'] = 0
train_dist.head(n=5)
print("The dimension for the input of distance-based model is %d x %d" % (train_dist.shape[0], train_dist.shape[1] - 1))
# SalePrice is not input, so minus one
from sklearn.model_selection import train_test_split
random_state = 7
xt_train_test, xt_valid, yt_train_test, yt_valid = train_test_split(train_tree['SalePrice'], train_tree.drop('SalePrice', axis=1), test_size=.2, random_state=random_state)
xd_train_test, xd_valid, yd_train_test, yd_valid = train_test_split(train_dist['SalePrice'], train_dist.drop('SalePrice', axis=1), test_size=.2, random_state=random_state)
xt_train, xt_test, yt_train, yt_test = train_test_split(yt_train_test, xt_train_test, test_size=.2, random_state=random_state)
xd_train, xd_test, yd_train, yd_test = train_test_split(yd_train_test, xd_train_test, test_size=.2, random_state=random_state)
print("number of training set: %d\nnumber of testing set: %d\nnumber of validation set: %d\ntotal: %d" % (len(xt_train), len(xt_test), len(xt_valid), (len(xt_train)+len(xt_test)+len(xt_valid))))
def rmse(arr1, arr2):
    return np.sqrt(np.mean((arr1-arr2)**2))
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(xd_train, yd_train)
yd_lm = lm.predict(xd_test)
rmse_linear = rmse(yd_test, yd_lm)
sns.regplot(yd_test, yd_lm)
print("RMSE for Linear Regression Model in sklearn: %.2f" % rmse_linear)
import keras
from keras.models import Sequential
from keras.layers import Dense
def baseline_nn_model(dims):
    model = Sequential()
    model.add(Dense(dims, input_dim=dims,kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def larger_nn_model(dims):
    model = Sequential()
    model.add(Dense(dims, input_dim=dims,kernel_initializer='normal', activation='relu'))
    model.add(Dense(35, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model
def use_keras_nn_model(nn_model, x, y, xx, yy, epoch):
    print("start training")
    for step in range(epoch + 1):
        cost = nn_model.train_on_batch(x.as_matrix(), y.as_matrix())
        if step % 100 == 0:
            print("train cost: %.2f" % cost)
    print("start testing")
    yy_predict = nn_model.predict(xx.as_matrix()).reshape(len(yy),)
    res = rmse(yy, yy_predict)
    sns.regplot(yy, yy_predict)
    print("RMSE for NN Model in Keras(Tensorflow): %.2f" % res)
    return res
rmse_baselinenn = use_keras_nn_model(baseline_nn_model(xd_train.shape[1]), xd_train, yd_train, xd_test, yd_test, 700)
rmse_largernn = use_keras_nn_model(larger_nn_model(xd_train.shape[1]), xd_train, yd_train, xd_test, yd_test, 500)
rmse_nn = min(rmse_baselinenn, rmse_largernn)
import xgboost as xgb
from xgboost import plot_importance
params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()
dtrain = xgb.DMatrix(xt_train, yt_train)
dtest = xgb.DMatrix(xt_test)
num_rounds = 500
xgb_model = xgb.train(plst, dtrain, num_rounds)
yt_xgb = xgb_model.predict(dtest)
rmse_xgb = rmse(yt_test, yt_xgb)
sns.regplot(yt_test, yt_xgb)
print("RMSE for xgboost: %.2f" % rmse_xgb)
plot_importance(xgb_model)
# it shows that the feature crossing is actually working
print("The minimum RMSE goes to: %.2f" % min([rmse_linear, rmse_nn, rmse_xgb]))
# xgboost turns out to be a better model here
idx_clean_final = idx_meta.copy()
idx_clean_final.remove('SalePrice')
final_clean = final_csv[idx_clean_final]
final_clean.head(n=5)
final_clean['binned_TotalBsmtSF'] = np.digitize(final_clean['TotalBsmtSF'], bins_totalbsmtsf)
final_clean['binned_1stFlrSF'] = np.digitize(final_clean['1stFlrSF'], bins_1stflrsf)
final_clean['binned_2ndFlrSF'] = np.digitize(final_clean['2ndFlrSF'], bins_2ndflrsf)
final_clean['SFcross'] = (final_clean['binned_TotalBsmtSF'] - 1) * (5 * 4) + (final_clean['binned_1stFlrSF'] - 1) * 5 + final_clean['binned_2ndFlrSF']
final_clean['has_MasVnrArea'] = (final_clean['MasVnrArea'] > 0).astype(float)
final_clean.head(n=5)
idx_tree_final = idx_tree.copy()
idx_tree_final.remove('SalePrice')
final_tree = final_clean[idx_tree_final]
final_tree.head(n=5)
dtest_final = xgb.DMatrix(final_tree)
yt_final = xgb_model.predict(dtest_final)
summission = pd.concat([final_csv['Id'], pd.DataFrame(yt_final)], axis=1)
summission.columns = ['Id', 'SalePrice']
sns.distplot(summission['SalePrice'])
summission.to_csv('summission.csv', encoding='utf-8', index = False)