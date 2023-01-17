
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
d1 = pd.read_csv("../input/Admission_Predict.csv")
d2 = pd.read_csv('../input/Admission_Predict_Ver1.1.csv')
d1.pop('Serial No.')
d2.pop('Serial No.')
data = d1.append(d2, ignore_index=True)
print(data.columns)
Y = data['Chance of Admit ']

X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,
                            max_depth=5, alpha=10, n_estimators=10)
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))

data_matrix = xgb.DMatrix(data=data, label=Y)
params = {"objective":"reg:linear", 'colsample_bytree': 0.3, 'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}
cv_results = xgb.cv(dtrain=data_matrix, params=params, nfold=3,
                        num_boost_round=50, early_stopping_rounds=10, metrics="rmse", as_pandas=True, seed=123)
print(cv_results.head())
print((cv_results["test-rmse-mean"]).tail(1))
xgb.plot_importance(xg_reg)
plt.rcParams["figure.figsize"] = [5,5]
plt.show()
