# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cs = pd.read_csv("/kaggle/input/used-car-dataset-ford-and-mercedes/vw.csv").columns
df = pd.DataFrame(columns=cs+["brand"])
dfs = []

brands_dict = {
    "vw": "VW",
    "merc": "Mercedes",
}
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if not "unclean" in filename and not "focus" in filename and not "cclass" in filename:
            p = os.path.join(dirname, filename)
            brand = filename.replace(".csv", "")
            if brand in brands_dict:
                brand = brands_dict[brand]
            new_df = pd.read_csv(p)
            new_df["brand"] = [brand for i in range(len(new_df))]
            if "tax(£)" in new_df.columns:
                new_df.rename(columns={"tax(£)": "tax"}, inplace=True)
            
            dfs.append(new_df)
df = pd.concat(dfs)          
df.describe()
df.head()


df.head()
df.groupby("model")["price"].mean()
models = list(df["model"])
brands = list(df["brand"])
new_model_names = [f"{brand} {model}" for model, brand in zip(models, brands)]
df["model"] = new_model_names
df.head()
numerical_df = df.copy()
def make_column_numerical(df, column):
    uniques = list(df[column].unique())
    df[column] = df[column].apply(lambda x: uniques.index(x))
    return df
def make_numerical(df):
    num = df.copy()
    num = make_column_numerical(num, "transmission")
    num = make_column_numerical(num, "fuelType")
    return num
#numerical_df = make_numerical(numerical_df)

df["model"] == models[0]
list(map(lambda x: int(x), list(df["model"] == models[0])))

def make_ready_for_regression(df):
    models = df["model"].unique()
    numerical = make_numerical(df)
    target = numerical["price"]
    numerical.drop("price", axis=1, inplace=True)
    numerical.drop("brand", axis=1, inplace=True)
    for model in models:
        is_model = list(map(lambda x: int(x), list(numerical["model"] == model)))
        numerical[model] = is_model
    numerical.drop("model", axis=1, inplace=True)
    X = numerical.values
    y = target
    return X, y
df.head()
df.isna().sum()
num = make_numerical(df)
num.head()
X, y = make_ready_for_regression(df)
import random
from sklearn.model_selection import train_test_split
random.shuffle(zip(X, y))
train_X, test_X, train_y, test_y = train_test_split(
 X, y, test_size=0.33, random_state=42)
from sklearn.linear_model import LinearRegression
train_X[0]
model = LinearRegression()
model = model.fit(train_X, train_y)
model.score(test_X, test_y)
from sklearn.metrics import mean_squared_error, mean_absolute_error
y = model.predict(test_X)
mean_squared_error(test_y, y)
mean_absolute_error(test_y, y)

model.coef_
numerical.values()