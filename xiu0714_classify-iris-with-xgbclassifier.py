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
def birdview(data):

    print("----------Head 5 Record----------")

    print(data.head(5))

    print("\n-----------Information-----------")

    print(data.info())

    print("\n-----------Data Types-----------")

    print(data.dtypes)

    print("\n----------Missing value-----------")

    print(data.isnull().sum())

    print("\n----------Null value-----------")

    print(data.isna().sum())

    print("\n----------Shape of Data----------")

    print(data.shape)

    

def graph_insight(data):

    print(set(data.dtypes.tolist()))

    df_num = data.select_dtypes(include = ['float64', 'int64'])

    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);



def distribution_insight(data, column):

    print("===================================")

    print("Min Value:", data[column].min())

    print("Max Value:", data[column].max())

    print("Average Value:", data[column].mean())

    print("Center Point of Data (median):", data[column].median())

    print("===================================")

    # sns.boxplot(data[column])
iris = '../input/irisdataset/Iris.csv'

df = pd.read_csv(iris)



birdview(df)

df.iris.value_counts()
graph_insight(df)
from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()

labels = lb.fit_transform(df.iris)

labels 
df.drop(columns=['iris'], inplace=True)

df.head()
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(df, labels, test_size=0.2, random_state=0)

eval_set = [(X_train, y_train), (X_val, y_val)]
from xgboost import XGBClassifier

model = XGBClassifier()

model.fit(X_train, y_train, eval_set=eval_set,

          early_stopping_rounds=10, eval_metric='merror',

          verbose=True)
from sklearn.metrics import accuracy_score



y_preds = model.predict(X_val)

print("Accuracy score", accuracy_score(y_val, y_preds))