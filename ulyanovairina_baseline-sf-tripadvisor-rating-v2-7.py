# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler



import collections

import re

import datetime

import json



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42

# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
def round_of_rating(number):

    # Округляем до 0.5

    return np.round(number * 2) / 2
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()

df_train.head(5)
df_test.info()

df_test.head(5)
sample_submission.info()

sample_submission.head(5)
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_NAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True) 
data['Reviews_NaN'] = pd.isna(data['Reviews']).astype('uint8')

# заполним пропуски значением '[[], []]'

data['Reviews'] = data['Reviews'].fillna('[[], []]')
data['Cuisine_Style_NaN'] = pd.isna(data['Cuisine Style']).astype('uint8') 

# заполним пропуски значением 'other_style'

data['Cuisine Style'] = data['Cuisine Style'].fillna("['Other']")
# сохраним информацию о пропусках чтобы не потерять

data['Price_Range_NaN'] = pd.isna(data['Price Range']).astype('uint8') 

# поэтому заполняем пропуски 0

data['Price Range'].fillna(0, inplace=True) 
data.Restaurant_id.value_counts()
def change_id(x):

    if 'id_' in str(x):

        return str(x).replace('id_', '')

    else: return x

    

data['Restaurant_id'] = data['Restaurant_id'].apply(change_id)

data['Restaurant_id'] = pd.to_numeric(data['Restaurant_id'])
data.City.value_counts()
data['City_origin'] = data['City']
data['Cuisine Style'] = data['Cuisine Style'].apply(lambda x: re.findall('\w+\s*\w+\s*\w+', str(x)))
# заполним значения в переменной по словарю

dict_Price_Range = {'$':1,'$$ - $$$':2,'$$$$':3}

data['Price Range']=data['Price Range'].map(lambda x: dict_Price_Range.get(x,x))



data['Price Range'].value_counts()
def fill_na_reviews(x):

    if x == '[[], []]':

        return None

    else:

        return x

    

data['Reviews'] = data['Reviews'].fillna(fill_na_reviews)
# выделим даты из обзора

data['Review_date'] = data.Reviews.apply(lambda x : [0] if pd.isna(x) else x[2:-2].split('], [')[1][1:-1].split("', '"))
data.drop(['URL_TA'], axis=1, inplace=True)
def change_id_TA(x):

    if 'd' in str(x):

        return str(x).replace('d', '')

    else: return x



# заменим на число

data['ID_TA'] = data['ID_TA'].apply(change_id_TA)

data['ID_TA'] = pd.to_numeric(data['ID_TA'])
data.columns
# встречаемые кухни

cuisines = set()



for i in data['Cuisine Style']:

    for j in i:

        cuisines.add(j)
cuisines
# частота встречаемости

type_cousine = {}  # создаём пустой словарь для хранения информации о кухнях

for item in cuisines:  # перебираем список кухонь

    type_cousine[item] = 0 # добавляем в словарь ключ, соответствующий очередной кухне



for i in data['Cuisine Style']:   # перебираем список кухонь

    for j in i:   # и список кухонь в каждом ресторане

        type_cousine[j] += 1   # увеличиваем значение нужного ключа в словаре на 1
type_cousine
# топ кухонь

top_cuisine = []

for key, value in type_cousine.items():

    if value > 3000:

        top_cuisine.append(key)

top_cuisine
def most_popular_cuisine(x):

    for element in top_cuisine:

        if element in x:

            return 1

        else:

            continue

            

# новое поле "Наличие популярной кухни"            

data['most_popular_cuisine'] = data['Cuisine Style'].apply(most_popular_cuisine)
data['most_popular_cuisine'].fillna(0, inplace = True)
# добавим новый признак "Количество кухонь в ресторане"

data['cuisine_counts'] = data['Cuisine Style'].apply(lambda x: len(x))
# максимальная дата отзыва

data['max_Review_date'] = pd.to_datetime(data['Review_date'].apply(lambda x: max(x)))

# первая дата отзыва

data['first_Review_date'] = pd.to_datetime(data['Review_date'].apply(lambda x : x[0]))

# вторая дата отзыва

data['second_Review_date'] = pd.to_datetime(data['Review_date'].apply(lambda x: x[1] if len(x) == 2 else ''))

# новое поле "разница в датах отзыва"

data['rew_delta'] = np.abs(data['first_Review_date'] - data['second_Review_date'])

#  "разница в датах отзыва" в днях

data['rew_delta'] = data['rew_delta'].apply(lambda x: x.days)



data['rew_delta'].fillna(value=round(data.rew_delta.mean()), inplace=True)



# новое поле "разница между текущей датой и последней датой отзывы"

data['cur_rew_delta'] = datetime.datetime.now() - data['max_Review_date']

data['cur_rew_delta'] = data['cur_rew_delta'].apply(lambda x: x.days)

data['cur_rew_delta'].fillna(value=round(data.cur_rew_delta.mean()), inplace=True)



# пустые значения

data['cur_rew_delta'] = data['cur_rew_delta'].fillna(data['cur_rew_delta'].median())

data['rew_delta'] = data['rew_delta'].fillna(data['rew_delta'].median())
data.drop(['max_Review_date','first_Review_date','second_Review_date'], axis=1, inplace=True)
data['City_origin'].value_counts()
dict_Сity_population= {'London' : 8908, 'Paris' : 2206, 'Madrid' : 3223, 'Barcelona' : 1620, 

                        'Berlin' : 6010, 'Milan' : 1366, 'Rome' : 2872, 'Prague' : 1308, 

                        'Lisbon' : 506, 'Vienna' : 1888, 'Amsterdam' : 860, 'Brussels' : 179, 

                        'Hamburg' : 1841, 'Munich' : 1457, 'Lyon' : 506, 'Stockholm' : 961, 

                        'Budapest' : 1752, 'Warsaw' : 1764, 'Dublin' : 553, 

                        'Copenhagen' : 616, 'Athens' : 665, 'Edinburgh' : 513, 

                        'Zurich' : 415, 'Oporto' : 240, 'Geneva' : 201, 'Krakow' : 769, 

                        'Oslo' : 681, 'Helsinki' : 643, 'Bratislava' : 426, 

                        'Luxembourg' : 119, 'Ljubljana' : 284}



# Население города из справочника

data['Сity_population'] = data.apply(lambda row: dict_Сity_population[row['City_origin']], axis = 1)
# Ресторанов на город

data['Rest_Cnt_Per_City'] = data['City_origin'].map(data.groupby('City_origin')['Restaurant_id'].count().to_dict())

# Людей на ресторан

data['Cnt_rest_per_person'] = data['Сity_population']/data['Rest_Cnt_Per_City']
# Рейтинг на город

data['Ranking_per_city'] = data['Ranking'] / data['Cnt_rest_per_person']
data.info()
correlation = data[data['sample'] == 1][['Ranking', 'Price Range','Number of Reviews', 'Rating','most_popular_cuisine', 

                                         'cuisine_counts', 'cur_rew_delta', 'rew_delta', 'Сity_population', 'Rest_Cnt_Per_City']].corr()

plt.figure(figsize=(20, 10))

sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
mn = data.groupby('City_origin')['Ranking'].mean()

st = data.groupby('City_origin')['Ranking'].std()

# Стандартизировать рейтинг

data['Std_Ranking'] = (data['Ranking'] - data['City_origin'].map(mn))/data['City_origin'].map(st)
data.info()
data.drop(['Restaurant_id','Cuisine_Style_NaN', 'Price_Range_NaN', 'Cuisine Style', 'Reviews', 'ID_TA', 

           'Number_of_Reviews_NAN','Reviews_NaN', 'City_origin', 'City', 'Review_date'], axis=1, inplace=True, errors='ignore')
# Теперь выделим тестовую часть

train_data = data.query('sample == 1').drop(['sample'], axis=1)

test_data = data.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# проверяем

test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = round_of_rating(model.predict(X_test))
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
test_data.sample(10)
test_data = test_data.drop(['Rating'], axis=1)
sample_submission
predict_submission = round_of_rating(model.predict(test_data))
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)