import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import json
from pandas.io.json import json_normalize
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields

JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

def load_df(csv_path):
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'})

    json_part = df[JSON_COLUMNS]
    df = df.drop(JSON_COLUMNS, axis=1)
    normed_json_part = []
    for col in JSON_COLUMNS:
        col_as_df = json_normalize(json_part[col])
        col_as_df.rename(columns=lambda x: f'{col}.{x}', inplace=True)
        normed_json_part.append(col_as_df)
    df = pd.concat([df] + normed_json_part, axis=1)

    return df
train = load_df('../input/train.csv')
train[['visitId', 'visitStartTime']].head()
mismatch = train['visitId'].astype(str) != train['visitStartTime'].astype(str)
mismatch.sum(), mismatch.mean()
# confirm `sessionId` == `fullVisitorId` + '_' + `visitId`
((train['fullVisitorId'] + '_' + train['visitId'].astype(str)) == train['sessionId']).all()
train['sessionId'].value_counts().max(), (train['sessionId'].value_counts() > 1).sum()
duplicates = train[train['sessionId'].map(train['sessionId'].value_counts()) > 1].sort_values(by='sessionId')
duplicates.head()
(train.loc[mismatch, 'visitId'] < train.loc[mismatch, 'visitStartTime']).all()
diff = train.loc[mismatch, 'visitStartTime'] - train.loc[mismatch, 'visitId']
diff.hist()
diff.value_counts().iloc[:20]
diff.max()
train = train.sort_values(by='visitStartTime')
# convert to `str` so that the series will not be converted to `float` automatically.
train['previous_visitStartTime'] = train['visitStartTime'].astype(str).groupby(train['fullVisitorId']).shift(1).fillna('-1').astype(np.int64)
train = train.sort_index()
((train['previous_visitStartTime'] == train['visitId']) & mismatch).sum()
duplicates = train[train['sessionId'].map(train['sessionId'].value_counts()) > 1].sort_values(by=['sessionId', 'date'])
(duplicates['visitId'] == duplicates['previous_visitStartTime']).sum()
(train['fullVisitorId'] + '_' + train['visitStartTime'].astype(str)).nunique() == train.shape[0]
