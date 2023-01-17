import pandas as pd
from pandas.core.common import array_equivalent


def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = {}
    duplicated = {}
    for t, v in groups.items():
        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)
        for i in range(lcs):
            if cs[i] in duplicated:
                continue
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if array_equivalent(ia, ja):
                    duplicated[cs[i]] = True
                    duplicated[cs[j]] = True
                    if cs[i] not in dups:
                        dups[cs[i]] = []
                    dups[cs[i]].append(cs[j])
    return dups
TRAIN_N_ROWS = 5000
train_df = pd.read_csv('../input/train.csv',
                 nrows=TRAIN_N_ROWS,
                 dtype={8: object, 9: object, 10: object, 11: object, 12: object,
                        43: object, 196: object, 214: object,
                        225: object, 228: object, 229: object,
                        231: object, 235: object, 238: object},
                 parse_dates=['VAR_0073', 'VAR_0075',
                              'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159',
                              'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169',
                              'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179',
                              'VAR_0204', 'VAR_0217'],
                 date_parser=lambda t: pd.to_datetime(t, format='%d%b%y:%H:%M:%S'))
TEST_N_ROWS = 5000
test_df = pd.read_csv('../input/test.csv',
                 nrows=TEST_N_ROWS,
                 dtype={8: object, 9: object, 10: object, 11: object, 12: object,
                        43: object, 196: object, 214: object,
                        225: object, 228: object, 229: object,
                        231: object, 235: object, 238: object},
                 parse_dates=['VAR_0073', 'VAR_0075',
                              'VAR_0156', 'VAR_0157', 'VAR_0158', 'VAR_0159',
                              'VAR_0166', 'VAR_0167', 'VAR_0168', 'VAR_0169',
                              'VAR_0176', 'VAR_0177', 'VAR_0178', 'VAR_0179',
                              'VAR_0204', 'VAR_0217'],
                 date_parser=lambda t: pd.to_datetime(t, format='%d%b%y:%H:%M:%S'))
num_train_df = train_df.select_dtypes(include=['number'], exclude=['datetime64']).drop(['ID', 'target'], axis=1)
duplicate_columns(num_train_df)
num_test_df = test_df.select_dtypes(include=['number'], exclude=['datetime64']).drop(['ID'], axis=1)
duplicate_columns(num_test_df)
str_train_df = train_df.select_dtypes(include=['object', 'datetime64'])
duplicate_columns(str_train_df)
str_test_df = test_df.select_dtypes(include=['object', 'datetime64'])
duplicate_columns(str_test_df)
