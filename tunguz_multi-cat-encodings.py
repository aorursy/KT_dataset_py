# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

from tqdm import tqdm_notebook

from sklearn.model_selection import KFold





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv', index_col='id')

test = pd.read_csv('../input/cat-in-the-dat-ii/test.csv', index_col='id')
train.shape
train.head()
test.shape
test.head()
target = train['target'].values

del train['target']
np.save('target', target)
columns = train.columns



for cc in tqdm_notebook(columns):

    train[cc] = train[cc].fillna(train[cc].mode()[0])

    test[cc] = test[cc].fillna(test[cc].mode()[0])
np.save('columns', columns.values)
%%time



X_train = train.copy()

X_test = test.copy()



for cc in tqdm_notebook(columns):

    le = LabelEncoder()

    le.fit(list(train[cc].values)+list(test[cc].values))

    X_train[cc] = le.transform(train[cc].values)

    X_test[cc] = le.transform(test[cc].values)
np.save('X_train_le', X_train)

np.save('X_test_le', X_test)
%%time

X_train = train.copy()

X_test = test.copy()



ohe = OneHotEncoder(dtype='uint16', handle_unknown="ignore")

ohe.fit(train)

X_train = ohe.transform(train)

X_test = ohe.transform(test)
X_train.shape
X_test.shape
np.save('X_train_ohe', X_train)

np.save('X_test_ohe', X_test)
X_train
# Reading the data

X = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

Xt = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")



# Separating target and ids

y = X.target.values

id_train = X.id

id_test = Xt.id



X.drop(['id', 'target'], axis=1, inplace=True)

Xt.drop(['id'], axis=1, inplace=True)



# Classifying variables in binary, high and low cardinality nominal, ordinal and dates

binary_vars = [c for c in X.columns if 'bin_' in c]



nominal_vars = [c for c in X.columns if 'nom_' in c]

high_cardinality = [c for c in nominal_vars if len(X[c].unique()) > 16]

low_cardinality = [c for c in nominal_vars if len(X[c].unique()) <= 16]



ordinal_vars = [c for c in X.columns if 'ord_' in c]



time_vars = ['day', 'month']
# Some feature engineering

X['ord_5_1'] = X['ord_5'].apply(lambda x: x[0] if type(x) == str else np.nan)

X['ord_5_2'] = X['ord_5'].apply(lambda x: x[1] if type(x) == str else np.nan)

Xt['ord_5_1'] = Xt['ord_5'].apply(lambda x: x[0] if type(x) == str else np.nan)

Xt['ord_5_2'] = Xt['ord_5'].apply(lambda x: x[1] if type(x) == str else np.nan)



ordinal_vars += ['ord_5_1', 'ord_5_2']
# Converting ordinal labels into ordered values

ordinals = {

    'ord_1' : {

        'Novice' : 0,

        'Contributor' : 1,

        'Expert' : 2,

        'Master' : 3,

        'Grandmaster' : 4

    },

    'ord_2' : {

        'Freezing' : 0,

        'Cold' : 1,

        'Warm' : 2,

        'Hot' : 3,

        'Boiling Hot' : 4,

        'Lava Hot' : 5

    }

}



def return_order(X, Xt, var_name):

    mode = X[var_name].mode()[0]

    el = sorted(set(X[var_name].fillna(mode).unique())|set(Xt[var_name].fillna(mode).unique()))

    return {v:e for e, v in enumerate(el)}



for mapped_var in ordinal_vars:

    if mapped_var not in ordinals:

        mapped_values = return_order(X, Xt, mapped_var)

        X[mapped_var + '_num'] = X[mapped_var].replace(mapped_values)

        Xt[mapped_var + '_num'] = Xt[mapped_var].replace(mapped_values)

    else:

        X[mapped_var + '_num'] = X[mapped_var].replace(ordinals[mapped_var])

        Xt[mapped_var + '_num'] = Xt[mapped_var].replace(ordinals[mapped_var])
# Transforming all the labels of all variables

from sklearn.preprocessing import LabelEncoder



label_encoders = [LabelEncoder() for _ in range(X.shape[1])]



for col, column in enumerate(X.columns):

    unique_values = pd.Series(X[column].append(Xt[column]).unique())

    unique_values = unique_values[unique_values.notnull()]

    label_encoders[col].fit(unique_values)

    X.loc[X[column].notnull(), column] = label_encoders[col].transform(X.loc[X[column].notnull(), column])

    Xt.loc[Xt[column].notnull(), column] = label_encoders[col].transform(Xt.loc[Xt[column].notnull(), column])
# Dealing with any residual missing value

X = X.fillna(-1)

Xt = Xt.fillna(-1)
# Enconding frequencies instead of labels (so we have some numeric variables)

def frequency_encoding(column, df, df_test=None):

    frequencies = df[column].value_counts().reset_index()

    df_values = df[[column]].merge(frequencies, how='left', 

                                   left_on=column, right_on='index').iloc[:,-1].values

    if df_test is not None:

        df_test_values = df_test[[column]].merge(frequencies, how='left', 

                                                 left_on=column, right_on='index').fillna(1).iloc[:,-1].values

    else:

        df_test_values = None

    return df_values, df_test_values



for column in X.columns:

    train_values, test_values = frequency_encoding(column, X, Xt)

    X[column+'_counts'] = train_values

    Xt[column+'_counts'] = test_values
# Target encoding of selected variables

X['fold_column'] = 0

n_splits = 5

kf = KFold(n_splits=n_splits, random_state=137)



import category_encoders as cat_encs



cat_feat_to_encode = binary_vars + ordinal_vars + nominal_vars + time_vars

smoothing = 0.3



enc_x = np.zeros(X[cat_feat_to_encode].shape)



for i, (tr_idx, oof_idx) in enumerate(kf.split(X, y)):

    encoder = cat_encs.TargetEncoder(cols=cat_feat_to_encode, smoothing=smoothing)

    

    X.loc[oof_idx, 'fold_column'] = i

    

    encoder.fit(X[cat_feat_to_encode].iloc[tr_idx], y[tr_idx])

    enc_x[oof_idx, :] = encoder.transform(X[cat_feat_to_encode].iloc[oof_idx], y[oof_idx])

    

encoder.fit(X[cat_feat_to_encode], y)

enc_xt = encoder.transform(Xt[cat_feat_to_encode]).values



for idx, new_var in enumerate(cat_feat_to_encode):

    new_var = new_var + '_enc'

    X[new_var] = enc_x[:,idx]

    Xt[new_var] = enc_xt[:, idx]
oof_idx
# Setting all to dtype float32

X = X.astype(np.float32)

Xt = Xt.astype(np.float32)



# Defining categorical variables

cat_features = nominal_vars + ordinal_vars



# Setting categorical variables to int64

X[cat_features] = X[cat_features].astype(np.int64)

Xt[cat_features] = Xt[cat_features].astype(np.int64)
X.head()
X['target'] = y
X.to_csv('X_train_te.csv', index=False)

Xt.to_csv('X_test_te.csv', index=False)