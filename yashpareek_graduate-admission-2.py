# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn

from sklearn import linear_model

import math

import matplotlib.pyplot as pyplot

from matplotlib import style

import seaborn as sns

from sklearn.metrics import mean_absolute_error

sns.set()





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")



data.drop('Serial No.', axis='columns', inplace=True) # Drop unwanted column

data.head()

gscore = data[["GRE Score"]]

fig=sns.distplot(gscore,color='blue',kde=False)

pyplot.title("GRE Score")

pyplot.show()



Tscore = data[["TOEFL Score"]]

fig=sns.distplot(Tscore,color='r',kde=False)

pyplot.title("TOEFL Score")

pyplot.show()



UnivRating = data[["University Rating"]]

fig=sns.distplot(UnivRating,color='black',kde=False)

pyplot.title("University Rating")

pyplot.show()
sns.distplot(data['Chance of Admit '])
predict = 'Chance of Admit '



X = np.array(data.drop([predict], axis=1))

y = np.array(data[predict])



x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.12,random_state=1)



linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)

acc = linear.score(x_test, y_test)





print("Model Accuracy: " + str(acc))

predictions = linear.predict(x_test)

sns.regplot(x=y_test, y=predictions)







for x in range(len(predictions)):

    print(predictions[x], "    ",y_test[x]) #Print the prediction and then the original value

    
