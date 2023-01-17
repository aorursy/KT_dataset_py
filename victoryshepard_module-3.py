import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline
df = pd.read_csv('main_task.csv')
df_test = pd.read_csv('kaggle_task.csv')
sample_submission = pd.read_csv('sample_submission.csv')
df.head()
df.info()
df['cuisine_style_isNAN'] = pd.isna(df['Cuisine Style']).astype('uint8')
df['number_of_reviews_isNAN'] = pd.isna(df['Number of Reviews']).astype('uint8')

df['Cuisine Style']=df['Cuisine Style'].fillna("")
df = df.fillna(0)
df=df.drop(columns=['URL_TA','ID_TA', 'Reviews'])
df.head()
def calc_mean_score(row):
    return df[df['City']==row['City']]['Rating'].mean()
df['city_mean']=df.apply(calc_mean_score, axis=1)
df['Price Range'].value_counts()
df.head()
df=pd.get_dummies(df, columns=['Price Range'])
df.columns=['Restaurant_id','city', 'cuisine_style','ranking','Rating', 'review_numbers', 'cuisine_style_isNAN', 'number_of_reviews_isNAN', 'city_mean','low_price','mid_price','high_price','no_price']
df
df.cuisine_style.value_counts().head(30)
pd.Series(cuisine).value_counts().sort_values()
count = []
for i in df.cuisine_style.values:
    if i == 0:
        count.append(1)
    else:
        count.append(len(i.split(',')))

df['count_cuisine'] = count
df.head()
df.head()
df['rest_counts'] = df.city.apply(lambda x: df.city.value_counts()[x])
df['relative_rank'] = df.ranking / df.rest_counts
df=df.drop(columns=['ranking','rest_counts','city','cuisine_style'])
df
sns.heatmap(df.corr())
# Х - данные с информацией о ресторанах, у - целевая переменная (рейтинги ресторанов)
X = df.drop(['Restaurant_id', 'Rating'], axis = 1)
y = df['Rating']
# Загружаем специальный инструмент для разбивки:
from sklearn.model_selection import train_test_split
# Наборы данных с меткой "train" будут использоваться для обучения модели, "test" - для тестирования.
# Для тестирования мы будем использовать 25% от исходного датасета.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель
regr = RandomForestRegressor(n_estimators=100)

# Обучаем модель на тестовом наборе данных
regr.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = regr.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data = df.drop(['Rating','Restaurant_id'], axis=1)
sample_submission
predict_submission = regr.predict(test_data[:10000])
predict_submission = np.round(predict_submission*2)/2
predict_submission
sample_submission['Rating'] = predict_submission
sample_submission.to_csv("submission.csv", index=False)
sample_submission.head(10)