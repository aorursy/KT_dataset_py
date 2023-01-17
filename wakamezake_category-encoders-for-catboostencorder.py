import numpy as np

import pandas as pd

import category_encoders as ce

from sklearn.datasets import load_boston, load_wine
ce.__version__
# load data

bunch = load_boston()

df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
target_cols = ['CHAS', 'RAD']
df.head()
df[target_cols].head()
enc = ce.CatBoostEncoder(cols=['CHAS', 'RAD']).fit(df, bunch.target)

numeric_dataset = enc.transform(df)
numeric_dataset.head()
numeric_dataset[target_cols].head()
# load data

bunch = load_wine()

df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
df.head()
enc = ce.CatBoostEncoder().fit(df, bunch.target)

numeric_dataset = enc.transform(df)
numeric_dataset.head()