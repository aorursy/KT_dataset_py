import numpy as np

import pandas as pd

train_data = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/train.csv')

test_data = pd.read_csv('/kaggle/input/cat-in-the-dat-ii/test.csv')
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values

    summary['Total'] = df.count().values   

    summary['Missing Percentage']=(summary['Missing']/summary['Total'])*100

    summary['Uniques'] = df.nunique().values

    summary['Uniques_val'] = [df[col].unique() for col in df.columns]

    return summary



resumetable(test_data)
def fillna_sample(df):

    for col in df.columns:

        df.loc[df[col].isna(),col] = df[col][-df[col].isna()].sample(n= df[col].isna().sum()).values

fillna_sample(train_data)

fillna_sample(test_data)
train_label = train_data['target']

train_data.drop(columns=['id', 'target'], axis=1, inplace=True)

test_id = test_data['id']

test_data.drop(columns=['id'], axis=1, inplace=True)
convert_ord = { "ord_1": {"Novice":0, "Contributor":1, "Expert":3, "Master":4, "Grandmaster":5},

                "ord_2": {"Freezing": 0, "Cold": 1, "Warm":3, "Hot":4, "Lava Hot":5, "Boiling Hot":6} }

train_data.replace(convert_ord, inplace=True)

test_data.replace(convert_ord, inplace=True)
need_dummy_cols = ["nom_0", "nom_1", "nom_2", "nom_3", "nom_4", "bin_3", "bin_4"]

train_data = pd.get_dummies(train_data, columns=need_dummy_cols, drop_first=True)

test_data = pd.get_dummies(test_data, columns=need_dummy_cols, drop_first=True)
from sklearn.preprocessing import LabelEncoder

lb_bin = LabelEncoder()

categorical_cols = train_data.select_dtypes(include=['object']).columns

for col in categorical_cols:

    train_data[col] = lb_bin.fit_transform(train_data[col])

    test_data[col] = lb_bin.fit_transform(test_data[col])
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

train_data = scaler.fit_transform(train_data)

test_data = scaler.fit_transform(test_data)
from sklearn.linear_model import LogisticRegression

lrclf = LogisticRegression(C=5)

lrclf.fit(train_data, train_label)

lrclf_pred = lrclf.predict_proba(test_data)
submission = pd.DataFrame({'id': test_id, 'target': lrclf_pred[:,1]})

submission.to_csv('submission.csv', index=False)