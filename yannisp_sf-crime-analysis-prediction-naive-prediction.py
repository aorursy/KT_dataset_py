import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])
test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')
train.drop_duplicates(inplace=True)
train.replace({'X': -120.5, 'Y': 90.0}, np.NaN, inplace=True)

imp = SimpleImputer(strategy='mean')

for district in train['PdDistrict'].unique():
    train.loc[train['PdDistrict'] == district, ['X', 'Y']] = imp.fit_transform(
        train.loc[train['PdDistrict'] == district, ['X', 'Y']])
naive_vals = train.groupby('Category').count().iloc[:,0]/train.shape[0]
n_rows = test.shape[0]

submission = pd.DataFrame(
    np.repeat(np.array(naive_vals), n_rows).reshape(39, n_rows).transpose(),
    columns=naive_vals.index)

submission.to_csv('naive.csv', index_label='Id')