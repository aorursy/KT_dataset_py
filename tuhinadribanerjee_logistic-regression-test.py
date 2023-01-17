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
import pandas as pd

diabetes = pd.read_csv("../input/diabetes.csv")


from sklearn.linear_model import LogisticRegression





X = diabetes.iloc[:, :-1]

y = diabetes.iloc[:, -1]





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
(diabetes)
x=diabetes.iloc[:,0:8]

y=diabetes.iloc[:,8]



(x)

(y)
prediction = model.predict
function = LogisticRegression()



function.fit(X_train, y_train)

y_pred = function.predict(X_test)

print("Pred_Regression : ",function.score(X_test, y_test)*100)