import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
X = data.dropna(axis=1).select_dtypes('number').drop(['Id','SalePrice'],axis=1)

y = data['SalePrice']

test_X = test[X.columns]
missing_val_count_by_column = (test_X.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
import matplotlib.pyplot as plt

import numpy as np



for xaxis in X.columns:

    print(xaxis)

    print(X[xaxis].values)

    fig, ax = plt.subplots()

    ax.plot(X[xaxis].values, y,'.')



    ax.set(xlabel=xaxis, ylabel='Sale Price ($)',

       title='Sale Price vs '+ xaxis)

    ax.grid()



    plt.show()
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()

model.fit(X,y)
# predict results

results = model.predict(test_X.fillna(0))

print(results)
submission = pd.read_csv("../input/home-data-for-ml-course/sample_submission.csv")

print(submission.head())

submission.loc[test_X.index.values, "SalePrice"] = results

print(submission.head())
submission.to_csv("submission.csv",index=False)