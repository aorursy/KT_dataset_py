import os

import numpy as np

import pandas as pd

import joblib

from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
_input_path = "../input/1056lab-used-cars-price-prediction/"

os.listdir(_input_path)
df_train = pd.read_csv(os.path.join(_input_path, "train.csv"), index_col=0)

df_test = pd.read_csv(os.path.join(_input_path, "test.csv"), index_col=0)
df_train.head()
df_test.head()
df_train["Price"].hist()
plt.hist(np.log1p(df_train["Price"]))
df_train["Engine"].str[-3:].value_counts()
df_test["Engine"].str[-3:].value_counts()
df_train["Year"].hist()
cnt_nan = df_train.isnull().sum()

plt.figure(figsize=(20, 5))

plt.barh(cnt_nan.index, cnt_nan.values)

plt.title("Train NaN Count")



cnt_nan = df_train.isnull().sum()

plt.figure(figsize=(20, 5))

plt.barh(cnt_nan.index, cnt_nan.values)

plt.title("Test NaN Count")
df_train.isnull().sum()



plt.figure(figsize=(20, 5))

sns.heatmap(df_train.drop("Price", axis=1).isnull(), cbar=False)

plt.title("Train NaN")

plt.show()



plt.figure(figsize=(20, 5))

sns.heatmap(df_test.isnull(), cbar=False)

plt.title("Test NaN")

plt.show()
df_train[df_train[["Engine", "Power"]].isnull().sum(axis=1) >= 1]
drop_origin_cols = []
df_train["Engine_num"] = df_train["Engine"].str[: -3].astype(np.float16) # NaNがintに出来ないのでfloatへ

df_test["Engine_num"] = df_test["Engine"].str[: -3].astype(np.float16)

if "Engine" not in drop_origin_cols: drop_origin_cols.append("Engine") 
df_train["Engine_num"].median()
df_train["Mileage_num"] = df_train["Mileage"].apply(lambda x: str(x).split(" ")[0]).astype(np.float)

df_test["Mileage_num"] = df_test["Mileage"].apply(lambda x: str(x).split(" ")[0]).astype(np.float)

if "Mileage" not in drop_origin_cols: drop_origin_cols.append("Mileage") 
df_train["Power"].str[-4: ].value_counts()
df_train["Power_num"] = df_train["Power"].str[: -4].replace({"null": np.nan}).astype(np.float)

df_test["Power_num"] = df_test["Power"].str[: -4].replace({"null": np.nan}).astype(np.float)

if "Power" not in drop_origin_cols: drop_origin_cols.append("Power") 
df_train.drop(drop_origin_cols, axis=1, inplace=True)

df_test.drop(drop_origin_cols, axis=1, inplace=True)
# メーカ名の抽出

df_train["maker"] = df_train["Name"].apply(lambda x: x.split(" ")[0])

df_test["maker"] = df_test["Name"].apply(lambda x: x.split(" ")[0])
print('Train Makers')

print(df_train['maker'].unique(), '\n')



print('Test Makers')

print(df_test['maker'].unique(), '\n')



print(set(df_train['maker'].unique()) - set(df_test['maker'].unique()))

print(set(df_test['maker'].unique()) - set(df_train['maker'].unique()))
# maker_rep = {'ISUZU': 'Isuzu', 'Bentley': 'Mercedes-Benz', 'Smart': 'Volkswagen'}

maker_rep = {'ISUZU': 'Isuzu'}

df_train['maker'].replace(maker_rep, inplace=True)

df_test['maker'].replace(maker_rep, inplace=True)
from sklearn.preprocessing import LabelEncoder

import category_encoders as ce



# LabelEncode

# le = LabelEncoder()

le = ce.OrdinalEncoder()

df_train["LE_maker"] = le.fit_transform(df_train["maker"])

df_test["LE_maker"] = le.transform(df_test["maker"])



# category型に変換

df_train["LE_maker"] = df_train["LE_maker"].astype("category")

df_test["LE_maker"] = df_test["LE_maker"].astype("category")



# Frequency Encode

fe_maker = df_train["maker"].value_counts().to_dict()

for i in list(set(df_test['maker'].unique()) - set(df_train['maker'].unique())): # Testにしかないmaker

    fe_maker[i] = 0

df_train["FE_maker"] = df_train["maker"].map(fe_maker)

df_test["FE_maker"] = df_test["maker"].map(fe_maker)



df_train.drop(["maker"], axis=1, inplace=True)

df_test.drop(["maker"], axis=1, inplace=True)
# 車のシリーズ?を抽出

df_train["series"] = df_train["Name"].apply(lambda x: x.split(" ")[1])

df_test["series"] = df_test["Name"].apply(lambda x: x.split(" ")[1])



le = ce.OrdinalEncoder()

df_train["LE_series"] = le.fit_transform(df_train["series"])

df_test["LE_series"] = le.transform(df_test["series"])



df_train["LE_series"] = df_train["LE_series"].astype("category")

df_test["LE_series"] = df_test["LE_series"].astype("category")



# Frequency Encode

fe_series = df_train["series"].value_counts().to_dict()

for i in list(set(df_test['series'].unique()) - set(df_train['series'].unique())): # Testにしかないmaker

    fe_series[i] = 0

df_train["FE_series"] = df_train["series"].map(fe_maker)

df_test["FE_series"] = df_test["series"].map(fe_maker)



df_train.drop(["series"], axis=1, inplace=True)

df_test.drop(["series"], axis=1, inplace=True)
le = LabelEncoder()

df_train["Location"] = le.fit_transform(df_train["Location"])

df_test["Location"] = le.transform(df_test["Location"])



df_train["Location"] = df_train["Location"].astype("category")

df_test["Location"] = df_test["Location"].astype("category")
le = ce.OrdinalEncoder()

df_train["Fuel_Type"] = le.fit_transform(df_train["Fuel_Type"])

df_test["Fuel_Type"] = le.transform(df_test["Fuel_Type"])



df_train["Fuel_Type"] = df_train["Fuel_Type"].astype("category")

df_test["Fuel_Type"] = df_test["Fuel_Type"].astype("category")
le = LabelEncoder()

df_train["Transmission"] = le.fit_transform(df_train["Transmission"])

df_test["Transmission"] = le.transform(df_test["Transmission"])
df_train["Transmission"] = df_train["Transmission"].astype("category")

df_test["Transmission"] = df_test["Transmission"].astype("category")
le = LabelEncoder()

df_train["Owner_Type"] = le.fit_transform(df_train["Owner_Type"])

df_test["Owner_Type"] = le.transform(df_test["Owner_Type"])



df_train["Owner_Type"] = df_train["Owner_Type"].astype("category")

df_test["Owner_Type"] = df_test["Owner_Type"].astype("category")
df_train.isnull().sum()
df_train["Seats"].fillna(-9999, inplace=True)

df_test["Seats"].fillna(-9999, inplace=True)
df_train["Power_num"].fillna(-9999, inplace=True)

df_test["Power_num"].fillna(-9999, inplace=True)
df_train["Engine_num"].fillna(-9999, inplace=True)

df_test["Engine_num"].fillna(-9999, inplace=True)
df_train["New_Price"].fillna(-9999, inplace=True)

df_test["New_Price"].fillna(-9999, inplace=True)
df_train
drop_cols = ["Name", "New_Price"]



X = df_train.drop(drop_cols + ["Price"], axis=1)

y = np.log1p(df_train["Price"])

X_test = df_test.drop(drop_cols, axis=1)
cat_cols = [ind for ind, typ in (df_test.dtypes=="category").items() if typ]
import lightgbm as lgb

lgb_reg = lgb.LGBMRegressor()

lgb_reg.fit(X, y)

lgb_predict = np.expm1(lgb_reg.predict(X_test))
lgb_submit = pd.read_csv(os.path.join(_input_path, "sampleSubmission.csv"))

lgb_submit["Price"] = lgb_predict

lgb_submit.to_csv("lgb_submit.csv", index=False)
import catboost as cat



cat_X = X.copy()

cat_X[cat_cols] = cat_X[cat_cols].astype(np.int16)

cat_X_test = X_test.copy()

cat_X_test[cat_cols] = cat_X_test[cat_cols].astype(np.int16)



cat_reg = cat.CatBoostRegressor()

cat_reg.fit(cat_X, y, cat_features=cat_cols, verbose=False)

cat_predict = np.expm1(cat_reg.predict(cat_X_test))
cat_submit = pd.read_csv(os.path.join(_input_path, "sampleSubmission.csv"))

cat_submit["Price"] = cat_predict

cat_submit.to_csv("cat_submit.csv", index=False)