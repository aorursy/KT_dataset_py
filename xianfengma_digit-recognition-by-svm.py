# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import svm



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# read training data and test data

data_train = pd.read_csv("../input/train.csv")

#data_train.head(10)



data_test = pd.read_csv("../input/test.csv")

#data_test.head(10)
X = data_train.drop("label", axis=1)

Y = data_train["label"]

lin_clf = svm.LinearSVC()

lin_clf.fit(X,Y)
lin_clf.score(X, Y)
Predict_test = lin_clf.predict(data_test)


submission = pd.DataFrame({

        "ImageId": data_test.index.values + 1,

        "Label": Predict_test

    })

submission.to_csv('digit_recognition.csv', index=False)