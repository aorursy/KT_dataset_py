# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/train.csv")

data = df.drop(columns='SalePrice')

label = df['SalePrice'].values
X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.1, random_state=113)
num_cols = []

cat_cols = []

for name in X_train.columns:

    if (name == "Id"):

        continue

    d_type = X_train.dtypes[name]

    if np.issubdtype(d_type, np.number):

        num_cols.append(name)

    else:

        cat_cols.append(name)
# category col

cat_si_step = ('si', SimpleImputer(strategy='constant',fill_value='MISSING'))

cat_ohe_step = ('ohe', OneHotEncoder(sparse=False, handle_unknown='ignore'))

cat_steps = [cat_si_step, cat_ohe_step]

cat_pipe = Pipeline(cat_steps)

num_si_step = ('si', SimpleImputer(strategy='median'))

num_ss_step = ('ss', StandardScaler())

num_pipe = Pipeline([num_si_step, num_ss_step])

full_ct = ColumnTransformer(transformers=[("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])

ml_pipe = Pipeline([('transform', full_ct), ('regression', RandomForestRegressor())])
ml_pipe.fit(X_train, y_train)
ml_pipe.score(X_val, y_val)
evaluate_df = pd.read_csv("../input/test.csv")

evaluate_df = evaluate_df.set_index('Id')

evaluate_id = evaluate_df.index
pred = ml_pipe.predict(evaluate_df)
submission = pd.DataFrame({

    'Id': evaluate_id,

    'SalePrice': pred

})
submission.head()
submission.to_csv("submission.csv", index=False)