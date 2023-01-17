# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

import seaborn as sns
input_path='/kaggle/input/infopulsehackathon/'

train_df=pd.read_csv(input_path+'train.csv')

test_df=pd.read_csv(input_path+'test.csv')
print(train_df.shape, test_df.shape)
cat_cols=train_df.select_dtypes('object').columns

lbl_enc=LabelEncoder()

for col in cat_cols:

    train_df[col]=lbl_enc.fit_transform(train_df[col])

    test_df[col]=lbl_enc.transform(test_df[col])
x_train, y_train = train_df.drop(columns=['Energy_consumption']), train_df['Energy_consumption']
pca_decomp=PCA(n_components=5, copy=True)

x_train_decomp=pca_decomp.fit_transform(x_train)

x_test_decomp=pca_decomp.transform(test_df)
print(x_train_decomp.shape, x_test_decomp.shape)

pca_decomp.explained_variance_ratio_.cumsum()[-1]
lin_reg=LinearRegression()

lin_reg.fit(x_train_decomp, y_train.values)

y_pred=lin_reg.predict(x_test_decomp)
sns.distplot(y_train)

sns.distplot(y_pred)
submission=pd.read_csv(input_path+'sample_submission.csv')

submission['Energy_consumption']=y_pred

submission.to_csv('submission.csv', index=False)