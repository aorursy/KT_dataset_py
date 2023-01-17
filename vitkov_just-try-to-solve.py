import pandas as pd

import xgboost as xgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
df_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

df_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

df_sample_submission  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
columns = list()

for col in df_train.columns:

    if df_train[col].dtype in ['int64', 'float64']:

        columns.append(col)



columns.remove('Id')

columns.remove('SalePrice')
X = df_train[columns]

X.fillna(0, inplace=True)

y = df_train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_mse = 0

best_max_depth = 0

for max_depth in range(1,20):

    model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=max_depth, learning_rate=0.1)

    model_xgb.fit(X_train, y_train)

    mse = mean_squared_error(model_xgb.predict(X_test), y_test)

    if best_mse == 0:

        best_mse = mse

        best_max_depth = max_depth

    elif mse < best_mse:

        best_mse = mse

        best_max_depth = max_depth

    print(max_depth, mean_squared_error(model_xgb.predict(X_test), y_test))
model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=best_max_depth, learning_rate=0.1)

model_xgb.fit(X, y)
df_sample_submission['SalePrice'] = model_xgb.predict(df_test[columns].fillna(0))

df_sample_submission.head()
df_sample_submission.to_csv('submission.csv', index=False)