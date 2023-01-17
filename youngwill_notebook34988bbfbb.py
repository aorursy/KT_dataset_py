# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score
def grouped_impute(df, col, group_by, how='mean'):

    df[col] = df[col].fillna(df.groupby(group_by)[col].transform(how))

    df[col]= df[col].fillna(df[col].mean()) 

    return df

    

def setup_xy(df):

    df['Sex'] = df['Sex'].apply(lambda s: 1 if s == 'Male' else  0)

    

    for col in ['Age', 'Fare']:

        df = grouped_impute(df, col, ['Sex', 'Pclass'])

    

    try:

        y = df['Survived'].values

    except KeyError:

        y = None

    return df[['Pclass', 'Sex', 'Age', 'Fare']], y
train_df = pd.read_csv('../input/train.csv')
x_train, y_train = setup_xy(train_df)
clf = GradientBoostingClassifier(max_depth=10, n_estimators=500, learning_rate=0.1, min_samples_leaf=20)

clf.fit(x_train, y_train)

clf.score(x_train, y_train)
test_df = pd.read_csv('../input/test.csv')

x_test, _ = setup_xy(test_df)
out_df = pd.DataFrame({'Survived': clf.predict(x_test)})

out_df['PassnegerId'] = test_df['PassengerId']

out_df.to_csv('predictions.csv', index_label=False, index=False)