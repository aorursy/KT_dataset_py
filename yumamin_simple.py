import numpy as np

import pandas as pd



from sklearn.experimental import enable_iterative_imputer

from sklearn.impute import SimpleImputer, IterativeImputer



from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
file_train = "/kaggle/input/loan-pred/train_ctrUa4K.csv"

file_test = "/kaggle/input/loan-pred/test_lAUu6dG.csv"
train = pd.read_csv(file_train)

test = pd.read_csv(file_test)
train.shape
train.isnull().sum()
train.nunique()
train.info()
train.columns
impute_feature_num = ['Dependents', 'LoanAmount', 'Loan_Amount_Term']

impute_feature_cat = ['Gender', 'Married', 'Self_Employed', 'Credit_History']
train[impute_feature_cat].head()
simple_imputer_cat = SimpleImputer(strategy='most_frequent')

simple_imputer_num = SimpleImputer(strategy='mean')



def first_stage(df):

    df = df.copy()

    df['Dependents'] = df['Dependents'].apply(lambda x: int(x.strip

                                                            ('+')) if isinstance(x, str) else x)

    return df



def second_simple_impute(df, imputer, cols, test=False):

    df = df.copy()

    if not test:

        imputer.fit(df[cols])

    df[cols] = imputer.transform(df[cols])

    return df



def third_fix_data(df):

    df = df.copy()

    df['Dependents'] = df['Dependents'].astype(int)

    return df
pp_train = first_stage(train)

pp_train = second_simple_impute(pp_train, simple_imputer_cat, impute_feature_cat)

pp_train = second_simple_impute(pp_train, simple_imputer_num, impute_feature_num)

pp_train = third_fix_data(pp_train)
pp_train.isnull().sum()
pp_train.head()
train.columns
encode_features_ord = ['Gender', 'Married', 'Education', 'Self_Employed']

encode_features_ohe = ['Property_Area']
orde = OrdinalEncoder()

ohe = OneHotEncoder(dtype=np.int8)



def fourth_encode(df, encoder, cols, test=False):

    df = df.copy()

    if not test:

        encoder.fit(df[cols])

    df[cols] = encoder.transform(df[cols])

    return df



def fourth_encode_ohe(df, encoder, cols, test=False):

    df = df.copy()

    if not test:

        encoder.fit(df[cols])

    ohe_data = encoder.transform(df[cols]).toarray()

    ohe_cols = ["Property_Area_{item}" for item in list(encoder.categories_[0])]

    df = df.drop(cols, axis=1)

    return pd.concat([df, pd.DataFrame(ohe_data, columns=ohe_cols)], axis=1)
pp_trian = fourth_encode(pp_train, orde, encode_features_ord)

pp_train = fourth_encode_ohe(pp_train, ohe, encode_features_ohe)
pp_train.head()