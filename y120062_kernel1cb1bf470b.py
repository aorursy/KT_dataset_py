import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import keras

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

from keras.activations import relu

from keras.models import Sequential, Model

from keras.layers import Dense, Input, Conv1D, GlobalMaxPool1D, MaxPooling1D, GlobalAveragePooling1D, Dropout

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score, KFold

from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from keras.callbacks import ModelCheckpoint

from sklearn.ensemble import IsolationForest

from sklearn.metrics import mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LassoCV

import os

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



train_ID = train['Id']

test_ID = test['Id']



train['MSSubClass'] = train['MSSubClass'].apply(str)

train['OverallCond'] = train['OverallCond'].astype(str)

test['MSSubClass'] = test['MSSubClass'].apply(str)

test['OverallCond'] = test['OverallCond'].astype(str)

train.drop("Id", axis = 1, inplace = True)

test.drop("Id", axis = 1, inplace = True)

k = 13

corrmat = abs(train.corr(method='spearman'))

corr_cols = corrmat.nlargest(k, 'SalePrice').index
ntrain = train.shape[0]

ntest = test.shape[0]

train_y = train['SalePrice']

all_data = pd.concat((train, test)).reset_index(drop=True)
all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

all_data['BsmtQual'] = all_data['BsmtQual'].fillna('None')

all_data['BsmtCond'] = all_data['BsmtCond'].fillna('None')

all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('None')

all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('None')

all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('None')

all_data['GarageType'] = all_data['GarageType'].fillna('None')

all_data['GarageFinish'] = all_data['GarageFinish'].fillna('None')

all_data['GarageQual'] = all_data['GarageQual'].fillna('None')

all_data['GarageCond'] = all_data['GarageCond'].fillna('None')
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)

all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)

all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)

all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)

all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)

all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)

all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)

all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0)

all_data['GarageCars'] = all_data['GarageCars'].fillna(0)

all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['MasVnrType'] = all_data['MasVnrType'].fillna(all_data['MasVnrType'].mode()[0])

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Functional'] = all_data['Functional'].fillna(all_data['Functional'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data.drop(['MiscVal'], axis=1, inplace=True)

all_data.drop(['MiscFeature'], axis=1, inplace=True)

all_data.drop(['Utilities'], axis=1, inplace=True)
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)

all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
corr_cols = corr_cols.drop('SalePrice')

corr_cols = corr_cols.append(pd.Index(['haspool', 'hasgarage', 'hasbsmt', 'hasfireplace', 'OverallCond', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'BsmtCond', 'BsmtQual', 'FireplaceQu']))
corr_cols
all_data_1 = all_data[corr_cols]
one_hot_encoding = all_data.copy()

one_hot_encoding = pd.get_dummies(one_hot_encoding)

one_hot_encoding_1 = all_data_1.copy()

one_hot_encoding_1 = pd.get_dummies(one_hot_encoding_1)



#len_train

one_hot_encoding_1.head()
#train = all_data[:ntrain]

#test = all_data[ntrain:]

train = one_hot_encoding[:ntrain]

test = one_hot_encoding[ntrain:]

train_1 = one_hot_encoding_1[:ntrain]

test_1 = one_hot_encoding_1[ntrain:]

train_1['SalePrice'] = train_y

clf = IsolationForest(max_samples = 100, random_state = 42)

clf.fit(train_1)

y_noano = clf.predict(train_1)

y_noano = pd.DataFrame(y_noano, columns = ['Top'])

y_noano[y_noano['Top'] == 1].index.values



train_1 = train_1.iloc[y_noano[y_noano['Top'] == 1].index.values]

train_1.reset_index(drop = True, inplace = True)

print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])

print("Number of rows without outliers:", train_1.shape[0])

train_1_y = train_1['SalePrice']

train_1.drop(['SalePrice'], axis=1, inplace=True)
train['SalePrice'] = train_y

clf = IsolationForest(max_samples = 100, random_state = 42)

clf.fit(train)

y_noano = clf.predict(train)

y_noano = pd.DataFrame(y_noano, columns = ['Top'])

y_noano[y_noano['Top'] == 1].index.values



train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]

train.reset_index(drop = True, inplace = True)

print("Number of Outliers:", y_noano[y_noano['Top'] == -1].shape[0])

print("Number of rows without outliers:", train.shape[0])

train_y = train['SalePrice']

train.drop(['SalePrice'], axis=1, inplace=True)
cols = train.columns

numeric_cols = train.select_dtypes(exclude=['object']).columns

categorical_cols = cols.drop(numeric_cols)

print(numeric_cols)

print(categorical_cols)

n_train = train[numeric_cols]

n_test = test[numeric_cols]

c_train = train[categorical_cols]

c_test = test[categorical_cols]
scaler_1 = MinMaxScaler()

scaler_1_y = MinMaxScaler()

train_1 = scaler_1.fit_transform(train_1)

test_1 = scaler_1.transform(test_1)

train_1_y = train_1_y.values.reshape(-1, 1)

train_1_y = scaler_1_y.fit_transform(train_1_y)

scaler = MinMaxScaler()

scaler_y = MinMaxScaler()

n_train = scaler.fit_transform(n_train)

n_test = scaler.transform(n_test)

train_y = train_y.values.reshape(-1, 1)

train_y = scaler_y.fit_transform(train_y)

def build_model():

    model = Sequential()



    # The Input Layer :

    model.add(Dense(512, input_dim=n_train.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(128, kernel_initializer='normal', activation='relu'))

    model.add(Dense(128, kernel_initializer='normal', activation='relu'))

    model.add(Dense(64, kernel_initializer='normal', activation='relu'))

    model.add(Dense(32, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())

    return model
def build_model_1():

    model = Sequential()



    # The Input Layer :

    model.add(Dense(512, input_dim=train_1.shape[1], kernel_initializer='normal', activation='relu'))

    model.add(Dense(256, kernel_initializer='normal', activation='relu'))

    model.add(Dense(128, kernel_initializer='normal', activation='relu'))

    model.add(Dense(64, kernel_initializer='normal', activation='relu'))

    model.add(Dense(32, kernel_initializer='normal', activation='relu'))

    model.add(Dense(1, kernel_initializer='normal'))

    # Compile model

    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta())

    return model
#model = build_model()
estimator = KerasRegressor(build_fn=build_model, epochs=130, batch_size=5, verbose=1)
kfold = KFold(n_splits=5, random_state=42)

results = cross_val_score(estimator, n_train, train_y, cv=kfold, scoring="neg_mean_squared_error")

print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
print("Results: %.4f (%.4f) RMSLE" % (results.mean(), results.std()))
def make_submission(prediction, sub_name):

  my_submission = pd.DataFrame({'Id':test_ID,'SalePrice':prediction})

  my_submission.to_csv('{}.csv'.format(sub_name),index=False)

  print('A submission file has been made')

estimator.fit(n_train, train_y)

predictions = estimator.predict(n_test)

predictions = predictions.reshape(-1, 1)
predictions = scaler_y.inverse_transform(predictions)
make_submission(predictions[:,0],'submission.csv')
tree = DecisionTreeRegressor()

result_tree = cross_val_score(tree, n_train, train_y, cv=kfold, scoring="neg_mean_squared_error")
alpha = [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

lasso = LassoCV(max_iter=1e7, alphas = alpha, random_state = 42)

result_lasso = cross_val_score(lasso, n_train, train_y, cv=kfold, scoring="neg_mean_squared_error")
print("Results tree: %.4f (%.4f) RMSLE" % (result_tree.mean(), result_tree.std()))

print("Results lr: %.4f (%.4f) RMSLE" % (result_lasso.mean(), result_lasso.std()))
tree = tree.fit(n_train, train_y)

lasso = lasso.fit(n_train, train_y)



pred_tree = tree.predict(n_test)

pred_lr = lasso.predict(n_test)
pred_tree = pred_tree.reshape(-1, 1)

pred_tree = scaler_y.inverse_transform(pred_tree)

make_submission(pred_tree[:,0],'tree.csv')
pred_lr = pred_lr.reshape(-1, 1)

pred_lr = scaler_y.inverse_transform(pred_lr)

make_submission(pred_lr[:,0],'lasso.csv')