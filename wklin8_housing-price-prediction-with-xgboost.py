import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

print(f"xgboost version: {xgb.__version__}")

from xgboost import XGBRegressor

%matplotlib inline
import matplotlib.pyplot as plt
train_raw = pd.read_csv("../input/train.csv")
test_raw = pd.read_csv("../input/test.csv")

train_raw_corr = train_raw.corr()
all_columns = train_raw.columns
numeric_columns = train_raw_corr.columns
non_numeric_columns = all_columns[~all_columns.isin(numeric_columns)]

label = 'SalePrice'

## only use the numeric column if its correlation with label is greater than this threshold
correlation_threshold = 0.2

# only use categorical column if pct of nan values less than this threshold
na_column_threshold = 0.4

sns.distplot(train_raw["SalePrice"])
significant_correlation = train_raw_corr["SalePrice"].abs().sort_values(ascending=False) >= correlation_threshold

significant_numeric_columns = significant_correlation[significant_correlation].index

fig, ax = plt.subplots(figsize=(10,10))
_ = sns.heatmap(train_raw[significant_numeric_columns].corr(), annot=True, ax=ax)
# pair plot of the first 6 correlations
sns.pairplot(train_raw[significant_numeric_columns[:6]])
na_non_numeric = (train_raw[non_numeric_columns].isna().sum() / train_raw[non_numeric_columns].isna().count() < na_column_threshold)
significant_non_numeric_columns = na_non_numeric[na_non_numeric].index

x = significant_non_numeric_columns[2]
f, ax = plt.subplots(figsize=(8, 6))
fig = sns.boxplot(x=x, y=label, data=train_raw)
fig.axis(ymin=0, ymax=800000);
%%time
significant_columns = np.concatenate([significant_numeric_columns, significant_non_numeric_columns])
test = test_raw.copy()
test[label] = 0

train_n_test = pd.concat([train_raw, test], ignore_index=True, sort=False)
train_n_test = pd.get_dummies(train_n_test, dummy_na=True)

train = train_n_test.query('Id <= @train_raw.shape[0]')
test = train_n_test.query('Id > @train_raw.shape[0]').drop(label, axis=1)

val_pct = 0.2
X_train, X_test, y_train, y_test = train_test_split(train.drop(["Id", label], axis=1).values, train_raw[label].values, test_size=val_pct, random_state=42)

model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_test, y_test)], verbose=False)
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error;
from math import sqrt

predict = model.predict(X_test)
print("Mean Absolute Error : " + str(mean_absolute_error(predict, y_test)))

rms = sqrt(mean_squared_error(np.log(y_test), np.log(model.predict(X_test))))
print("RMSE : " + str(rms))

submission = test[["Id"]].copy()
submission["SalePrice"] = model.predict(test.drop("Id", axis=1).values)
submission.to_csv("submission_xgboost_simple.csv", index=False)
