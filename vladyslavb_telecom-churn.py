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
df = pd.read_csv('/kaggle/input/churn-in-telecoms-dataset/bigml_59c28831336c6604c800002a.csv')

df.head()
import matplotlib.pyplot as plt

import seaborn as sns
df.describe()
df.info()
df['churn'].value_counts(normalize = True)
df['churn'].value_counts(normalize = True).plot(kind = 'bar')
df["state"].value_counts()
df["area code"].value_counts()
df_1 = df.drop(["state","phone number"], axis = 1)
df_2 = df_1.copy()



df_2["area code"] = df['area code'].map({415:0,510:1,408:2})

df_2["international plan"] = df['international plan'].map({"yes":1,"no":0})

df_2["voice mail plan"] = df['voice mail plan'].map({"yes":1,"no":0})

df_2["churn"] = df['churn'].map({False:0,True:1})
df_2
df_3 = df_2.drop(["churn"], axis = 1)



df_3
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(df_3)



X
from sklearn.model_selection import train_test_split



#X найден ранее

y = df_2['churn']



X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.25, random_state = 12)
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_valid)
from sklearn.metrics import accuracy_score

accuracy_score(y_valid, y_pred)
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score



kf = KFold(n_splits=5, shuffle=True, random_state=42)

print(cross_val_score(knn, X, y, cv = kf, scoring = 'accuracy'))
from sklearn.model_selection import GridSearchCV



knn_params = {'n_neighbors': np.arange(1, 21)}

knn_grid = GridSearchCV(knn, knn_params, scoring = 'accuracy', cv = kf)

knn_grid.fit(X, y)
knn_grid.best_estimator_
knn_grid.best_score_
knn_grid.best_params_
pd.DataFrame(knn_grid.cv_results_)
import matplotlib.pyplot as plt



results_df = pd.DataFrame(knn_grid.cv_results_)

plt.plot(results_df['param_n_neighbors'], results_df['mean_test_score'])



plt.xlabel('neighbors')

plt.ylabel('test_score')

plt.show()
p_params = {'p': np.linspace(1,10,200)}

knn_1 = KNeighborsClassifier(n_neighbors=7, weights = 'distance', n_jobs = -1)

knn_cv = GridSearchCV(knn_1, p_params, cv = kf, scoring='accuracy', verbose = 100)

knn_cv.fit(X,y)
pd.DataFrame(knn_cv.cv_results_)
knn_cv.best_estimator_
knn_cv.best_score_
knn_cv.best_params_
from sklearn.neighbors import RadiusNeighborsClassifier



rnclassifier = RadiusNeighborsClassifier(radius=5)

rnclassifier.fit(X_train, y_train)



rnclassifier.score(X_valid, y_valid)
from sklearn.neighbors import RadiusNeighborsRegressor



rnregressor = RadiusNeighborsRegressor(radius=5)

rnregressor.fit(X_train, y_train)



rnregressor.score(X_valid, y_valid)
from sklearn.neighbors import NearestCentroid



ncentroid = NearestCentroid()

ncentroid.fit(X_train, y_train)



ncentroid.score(X_valid, y_valid)