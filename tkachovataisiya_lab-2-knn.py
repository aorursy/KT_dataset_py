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
#Загрузка данных

df = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')

df.head()
df.keys()
#df.info()
import seaborn as sns

sns.catplot(x = "deposit", kind = "count", palette = "ch:.25", data = df)
# Выведем процентное соотношение

df['deposit'].value_counts(normalize=True)
# print(df['deposit'].shape) # Всего 11162 строк

df['deposit']
from sklearn. preprocessing import LabelEncoder



le = LabelEncoder()

le.fit(df.default)

df['default_le'] = le.transform(df.default)



# ручная альтернатива

#dct = {'no': 0, 'yes': 1} # словарь для кодировки 

#df['default_le'] = df['default'].map(dct)



le_1 = LabelEncoder()

le_1.fit(df.housing)

df['housing_le'] = le_1.transform(df.housing)



le_2 = LabelEncoder()

le_2.fit(df.housing)

df['loan_le'] = le_2.transform(df.loan)



le_3 = LabelEncoder()

le_3.fit(df.housing)

df['deposit_le'] = le_3.transform(df.deposit)



dct_1 = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec':12}

df['month_le'] = df['month'].map(dct_1)



#dct_2 = {'married': 1, 'single': 2, 'divorced': 3}

#df['marital_le'] = df['marital'].map(dct_2)



#dct_3 = {'secondary': 1, 'tertiary': 2, 'primary': 3, 'unknown': 4}

#df['education_le'] = df['education'].map(dct_3)



#dct_4 = {'unknown': 0, 'cellular': 1, 'telephone': 2}

#df['contact_le'] = df['contact'].map(dct_4)



#dct_5 = {'unknown': 0, 'other': 1, 'failure': 2, 'success': 3}

#df['poutcome_le'] = df['poutcome'].map(dct_5)



df.head().T
df_1 = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')

df_1['default'] = df['default_le']

df_1['housing'] = df['housing_le']

df_1['loan'] = df['loan_le']

df_1['deposit'] = df['deposit_le']

df_1['month'] = df['month_le']



#df_1['marital'] = df['marital_le']

#df_1['education'] = df['education_le']

#df_1['contact'] = df['contact_le']

#df_1['poutcome'] = df['poutcome_le']



# get_dummies

df_1 = pd.get_dummies(df_1, columns=['marital', 'education', 'contact', 'poutcome'])



df_1.head().T
#df_1.info()
#df_2 = df_1.drop(['job'], axis = 1)

df_2 = pd.get_dummies(df_1, columns=['job']) 



df_2.head().T
#df_2.info()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



# Создание X - вся таблица без target (deposit), а y - target (deposit).



y = df_2['deposit']

df_3 = df_2.drop('deposit', axis = 1)



X = df_3



X_new = scaler.fit_transform(X)

X_new
# Разбиение

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, 

                                                      y, 

                                                      test_size=0.25, 

                                                      random_state=20) 



#random_state. Controls the shuffling applied to the data before applying the split. 
print(X_train.shape, y_train.shape)

print( X_valid.shape, y_valid.shape)
#Обучение классификатора

# Создаём представителя класса модели, задаём необходимые гиперпараметры



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
# Обучаем модель на обучающей выборке

knn.fit(X_train, y_train)
# Строим предсказания на основе обученной модели

y_pred = knn.predict(X_valid)

y_pred
#Функция mean_squared_error вычисляет среднеквадратичную ошибку, метрику риска, 

#соответствующую ожидаемому значению квадратичной ошибки или убытка.



from sklearn.metrics import mean_squared_error

mean_squared_error(y_valid, y_pred)
# Вычисляем метрику

knn.score(X_valid, y_valid)
# Ещё один способ для вычисления метрики

from sklearn.metrics import accuracy_score

print(accuracy_score(y_valid, y_pred))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_valid, y_pred))
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits = 5, shuffle = True, random_state = 42)



scores = cross_val_score(knn, X, y, cv = kf, scoring = 'accuracy')

scores.mean()
scores = cross_val_score(knn, X, y, cv = kf, scoring = 'f1')

scores.mean()
from sklearn.model_selection import GridSearchCV

knn_params = {'n_neighbors': np.arange(1, 51)}

knn_grid = GridSearchCV(knn, 

                        knn_params, 

                        scoring='f1',

                        cv = kf)

knn_grid.fit(X_train, y_train)
print("Best_estimator: ", knn_grid.best_estimator_)

print("Cross-validated score of the best_estimator: ", knn_grid.best_score_)

print( "Best_index_ while the best_score_ attribute will not be available: ", knn_grid.best_params_)
cv_results = pd.DataFrame(knn_grid.cv_results_)

cv_results.T
# Предсказания на тестовой выборке для оптимального числа соседей

y_pred = knn_grid.predict(X_valid)

accuracy_score(y_valid, y_pred)
best_knn = KNeighborsClassifier(n_neighbors=13)

y_pred = best_knn.fit(X_train, y_train).predict(X_valid)

accuracy_score(y_valid, y_pred)
import matplotlib.pyplot as plt

plt.plot(cv_results["param_n_neighbors"],cv_results["mean_test_score"])



plt.xlabel('Number of neighbors')

plt.ylabel('Test accuracy')

plt.title('F1 score')

plt.show()
knn_params = {"p": np.linspace(1,10, 200)}



knn = KNeighborsClassifier(n_neighbors = 13, weights = "distance", n_jobs = -1)

knn.fit(X_train, y_train)



cv = GridSearchCV(knn, knn_params, cv = kf, scoring="f1")

cv.fit(X_train, y_train)


print("Лучшее значение:", cv.best_score_)

print(cv.best_estimator_)
cv_result = pd.DataFrame(cv.cv_results_)

cv_result
plt.plot(cv_result["param_p"],cv_result["mean_test_score"])
from sklearn.neighbors import NearestCentroid

clf = NearestCentroid()

clf.fit(X_train, y_train)

NearestCentroid()



y_pred_nc = clf.predict(X_valid)

y_pred_nc
mean_squared_error(y_valid, y_pred_nc)
print(accuracy_score(y_valid, y_pred_nc))
print(confusion_matrix(y_valid, y_pred_nc))