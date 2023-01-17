import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv")



import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import OneHotEncoder
X = pd.read_csv("../input/train.csv", index_col = 'Id')

X_test = pd.read_csv("../input/test.csv", index_col = 'Id')



X.dropna(axis = 0, subset = ['SalePrice'], inplace = True)

y = X['SalePrice']

X.drop(['SalePrice'], axis = 1, inplace = True)



# temp = pd.concat([X, X_test], axis = 0)



# temp.drop(null_cols, axis = 1, inplace = True)



# temp = pd.concat([temp[numerical_cols], temp[categorical_cols]], axis = 1)



numeric_imputer = SimpleImputer(strategy = 'median')

categorical_imputer = SimpleImputer(strategy = 'most_frequent')



# temp1 = pd.DataFrame(numeric_imputer.fit_transform(temp[numerical_cols]))

# temp2 = pd.DataFrame(categorical_imputer.fit_transform(temp[categorical_cols]))



encoder = OneHotEncoder(handle_unknown = 'ignore', sparse = False)

# Oh_cols = pd.DataFrame(encoder.fit_transform(temp2))



# num = temp2.drop(temp2.columns, axis = 1)

# oh_train = pd.concat([num, Oh_cols], axis = 1)



# temp = pd.concat([temp1, Oh_cols], axis = 1)

# temp.head()



# from sklearn.preprocessing import StandardScaler



# scaler = StandardScaler()

# # temp = pd.DataFrame(scaler.fit_transform(temp))



# X, X_test = temp.iloc[: X.shape[0], :], temp.iloc[X.shape[0] :, :]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)



null_cols = [col for col in X_train.columns if X_train[col].isnull().sum() > 300]



X_train.drop(columns = null_cols, inplace = True)

X_valid.drop(columns = null_cols, inplace = True)

X_test.drop(columns = null_cols, inplace = True)



categorical_cols = X_train.select_dtypes(include = 'object').columns

numerical_cols = X_train.select_dtypes(exclude = 'object').columns



imp_num_X_train = pd.DataFrame(numeric_imputer.fit_transform(X_train[numerical_cols]))

imp_num_X_valid = pd.DataFrame(numeric_imputer.fit_transform(X_valid[numerical_cols]))

imp_num_X_test = pd.DataFrame(numeric_imputer.fit_transform(X_test[numerical_cols]))



imp_cat_X_train = pd.DataFrame(categorical_imputer.fit_transform(X_train[categorical_cols]))

imp_cat_X_valid = pd.DataFrame(categorical_imputer.fit_transform(X_valid[categorical_cols]))

imp_cat_X_test = pd.DataFrame(categorical_imputer.fit_transform(X_test[categorical_cols]))



oh_X_train = pd.DataFrame(encoder.fit_transform(imp_cat_X_train))

oh_X_valid = pd.DataFrame(encoder.transform(imp_cat_X_valid))

oh_X_test = pd.DataFrame(encoder.transform(imp_cat_X_test))



X_train = pd.concat([imp_num_X_train, oh_X_train], axis = 1)

X_train.columns = [i for i in range(0, X_train.shape[1])]

X_valid = pd.concat([imp_num_X_valid, oh_X_valid], axis = 1)

X_valid.columns = [i for i in range(0, X_train.shape[1])]

X_test = pd.concat([imp_num_X_test, oh_X_test], axis = 1)

X_test.columns = [i for i in range(0, X_train.shape[1])]



# X_train = pd.DataFrame(scaler.fit_transform(X_train))

# X_test = pd.DataFrame(scaler.fit_transform(X_test))

# X_valid = pd.DataFrame(scaler.fit_transform(X_valid))
from sklearn.ensemble import RandomForestRegressor



model = RandomForestRegressor(random_state = 0)

model.fit(X_train, y_train)



pred = model.predict(X_valid)

print(mean_absolute_error(pred, y_valid))



pred_rfregressor = model.predict(X_test)

temp = pd.read_csv('../input/test.csv', index_col = 'Id')

pd.DataFrame({'Id': temp.index, 'SalePrice': pred_rfregressor}).to_csv('submission.csv', index = False)
# from sklearn.linear_model import SGDRegressor



# model = SGDRegressor(max_iter = 10000, penalty = 'elasticnet', alpha = 3000000, learning_rate = 'optimal').fit(X_train, y_train)

# pred = model.predict(X_valid)

# print(mean_absolute_error(pred, y_valid))
from xgboost import XGBRegressor



model_xgboost = XGBRegressor(random_state = 0, learning_rate = 0.09, n_etimators = 2000, reg_alpha = 100000)

model_xgboost.fit(X_train, y_train, early_stopping_rounds = 10, eval_set = [(X_valid, y_valid)])

pred = model_xgboost.predict(X_valid)

print(mean_absolute_error(pred, y_valid))



pred_xgboost = model_xgboost.predict(X_test)

pd.DataFrame({'Id': temp.index, 'SalePrice': pred_xgboost}).to_csv('submission.csv', index = False)
from sklearn.svm import LinearSVR



model = LinearSVR(random_state = 0, max_iter = 100000, C = 0.05)

model.fit(X_train, y_train)

preds = model.predict(X_valid)

print(mean_absolute_error(preds, y_valid))
from sklearn.svm import SVR



model = SVR(kernel = 'rbf').fit(X_train, y_train)

preds = model.predict(X_valid)

print(mean_absolute_error(preds, y_valid))