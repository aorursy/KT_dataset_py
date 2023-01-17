import pandas as pd 
data = pd.read_csv("../input/automobile-dataset/Automobile_data.csv")
data.price.unique()
import numpy as np
for row in range(data.shape[0]):
    for col in range(data.shape[1]):
        if data.iloc[row, col] == "?":
            data.iloc[row, col] = np.nan
data[["stroke", "horsepower", "peak-rpm", "bore"]] = data[["stroke", "horsepower", "peak-rpm", "bore"]].astype("float64")
data = data.dropna(axis=0, subset=["price"])
y_train_full = data.price
X_train_full = data.drop("price", axis=1)
y_train_full = y_train_full.astype("float64")
num_cols = [col for col in X_train_full.columns 
            if X_train_full[col].dtype in ["int64", "float64"]]
cat_cols = [col for col in X_train_full.columns
            if X_train_full[col].dtype == "object"
           and X_train_full[col].nunique() < 25]
X_train_full.isna().sum()
from sklearn.impute import SimpleImputer
num_imputer = SimpleImputer(strategy="most_frequent")
imp_num_train = pd.DataFrame(num_imputer.fit_transform(X_train_full[num_cols]))
imp_num_train.columns = num_cols
imputer = SimpleImputer(strategy="most_frequent")
imp_cat_train = pd.DataFrame(imputer.fit_transform(X_train_full[cat_cols]))
imp_cat_train.columns = cat_cols
from sklearn.preprocessing import OneHotEncoder
oh_enc = OneHotEncoder(sparse=False)
oh_cat_train = pd.DataFrame(oh_enc.fit_transform(imp_cat_train))
oh_cat_train.index = imp_cat_train.index
oh_X_train = pd.concat([imp_num_train, oh_cat_train], axis=1)
# oh_X_train[num_cols].info()
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
reg = xgb.XGBRegressor()
reg_cv = GridSearchCV(reg, 
                      {"max_depth": np.arange(2, 11, 2), 
                       "n_estimators": np.arange(10, 50, 5)}, 
                      cv=4, 
                      verbose=1)
reg_cv.fit(oh_X_train, y_train_full)
print(reg_cv.best_params_, reg_cv.best_score_)
best_params = reg_cv.best_params_
reg = xgb.XGBRegressor(max_depth=best_params["max_depth"], 
                       n_estimators=best_params["n_estimators"])
reg.fit(oh_X_train, y_train_full)
plt = xgb.plot_importance(reg)
