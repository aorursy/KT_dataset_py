# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



import ast

import re



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer



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
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_train.info()
df_train.head(5)
df_test.info()
df_test.head(5)
sample_submission.head(5)
sample_submission.info()
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
data.info()
data.sample(5)
data.Reviews[1]
data.URL_TA[2]
data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')

data['Price_Range_isNAN'] = pd.isna(data['Price Range']).astype('uint8')

data['Cuisine_Style_isNAN'] = pd.isna(data['Cuisine Style']).astype('uint8')
data['Number_of_Reviews_isNAN']
plt.rcParams['figure.figsize'] = (10,7)

df_train['Ranking'].hist(bins=100)
df_train['City'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)
# посмотрим на топ 10 городов

for x in (df_train['City'].value_counts())[0:10].index:

    df_train['Ranking'][df_train['City'] == x].hist(bins=100)

plt.show()
df_train['Rating'].value_counts(ascending=True).plot(kind='barh')
df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)
df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)
plt.rcParams['figure.figsize'] = (10,7)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
data['Price Range'].value_counts()
# Заполняем пропуски $$ - $$$

data['Price Range'].fillna('$$ - $$$', inplace=True)

# Кодируем числами 1, 2, 3

prices_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3}

data['Price'] = data['Price Range'].map(prices_dict)
# Попробуем One-Hot Encoding

#data = pd.get_dummies(data, columns=['Price Range',], dummy_na=True)
# Нормализуем по количеству ресторанов в городе

data['Ranking_Norm'] = data.groupby('City')['Ranking'].apply(lambda x: x / x.max())
plt.rcParams['figure.figsize'] = (10,7)

data['Ranking_Norm'].hist(bins=100)
City_Capital = {

    'London': 1, 'Paris': 1, 'Madrid': 1, 'Barcelona': 0, 'Berlin': 1, 'Milan': 0, 'Rome': 1,

    'Prague': 1, 'Lisbon': 1, 'Vienna': 1, 'Amsterdam': 1, 'Brussels': 1, 'Hamburg': 0, 'Munich': 0,

    'Lyon': 0, 'Stockholm': 1, 'Budapest': 1, 'Warsaw': 1, 'Dublin': 1, 'Copenhagen': 1, 'Athens': 1,

    'Edinburgh': 1, 'Zurich': 1, 'Oporto': 0, 'Geneva': 1, 'Krakow': 1, 'Oslo': 1, 'Helsinki': 1,

    'Bratislava': 1, 'Luxembourg': 1, 'Ljubljana': 1

    }



City_Population = {

    'London': 8173900, 'Paris': 2240621, 'Madrid': 3155360, 'Barcelona': 1593075, 'Berlin': 3326002,

    'Milan': 1331586, 'Rome': 2870493, 'Prague': 1272690, 'Lisbon': 547733, 'Vienna': 1765649,

    'Amsterdam': 825080, 'Brussels': 144784, 'Hamburg': 1718187, 'Munich': 1364920, 'Lyon': 496343,

    'Stockholm': 1981263, 'Budapest': 1744665, 'Warsaw': 1720398, 'Dublin': 506211 , 'Copenhagen': 1246611,

    'Athens': 3168846, 'Edinburgh': 476100, 'Zurich': 402275, 'Oporto': 221800, 'Geneva': 196150, 'Krakow': 756183,

    'Oslo': 673469, 'Helsinki': 574579, 'Bratislava': 413192, 'Luxembourg': 576249, 'Ljubljana': 277554

    }



City_Country = {

    'London': 'UK', 'Paris': 'France', 'Madrid': 'Spain', 'Barcelona': 'Spain', 'Berlin': 'Germany', 'Milan': 'Italy',

    'Rome': 'Italy', 'Prague': 'Czech', 'Lisbon': 'Portugalia', 'Vienna': 'Austria', 'Amsterdam': 'Nederlands',

    'Brussels': 'Belgium', 'Hamburg': 'Germany', 'Munich': 'Germany', 'Lyon': 'France', 'Stockholm': 'Sweden',

    'Budapest': 'Hungary', 'Warsaw': 'Poland', 'Dublin': 'Ireland' , 'Copenhagen': 'Denmark', 'Athens': 'Greece',

    'Edinburgh': 'Schotland', 'Zurich': 'Switzerland', 'Oporto': 'Portugalia', 'Geneva': 'Switzerland', 'Krakow': 'Poland',

    'Oslo': 'Norway', 'Helsinki': 'Finland', 'Bratislava': 'Slovakia', 'Luxembourg': 'Luxembourg', 'Ljubljana': 'Slovenija'

    }
data['Capital'] = data['City'].map(City_Capital)

data['Population'] = data['City'].map(City_Population)

data['Country'] = data['City'].map(City_Country)
data.sample(5)
# Далее заполняем пропуски 0

# data['Number of Reviews'].fillna(0, inplace=True)

# Заполняем средним

# data['Number of Reviews'] = data.groupby('City')['Number of Reviews'].apply(lambda x:x.fillna(x.mean()))

# Заполняем медианным

data['Number of Reviews'] = data.groupby('City')['Number of Reviews'].apply(lambda x:x.fillna(x.median()))
data['Number_of_Reviews_City'] = data['Number of Reviews'] / data['Population']
data['Cuisine Style'].fillna("['IsNan']", inplace=True)
def cuisine_style_list(cuisine_style):

    return ast.literal_eval(cuisine_style)
data['Cuisines_Count'] = data['Cuisine Style'].apply(lambda x: len(cuisine_style_list(x)))
mlb = MultiLabelBinarizer()

Cuisines_df = pd.DataFrame(mlb.fit_transform(data['Cuisine Style'].apply(cuisine_style_list)),

                       columns=mlb.classes_, index=data.index)
Cuisines_df.sample(5)
pd.concat([data, Cuisines_df], axis=1)
CURRENT_DATE = pd.to_datetime('17/03/2020')
data['Reviews'].fillna("[[], []]", inplace=True)
data['Reviews'][1]
data['dates'] = data['Reviews'].apply(lambda x: re.findall(r'(\d\d/\d\d/\d\d\d\d)', x))

data['dates'] = data['dates'].apply(lambda x: list(map(pd.to_datetime,x))) 
data['days'] = data['dates'].apply(lambda x: abs((x[0]-x[1]).days) if len(x) == 2 else 0)
data['max_date'] = data['dates'].apply(lambda x: max(x) if len(x) > 0 else CURRENT_DATE)

data['today_days'] = CURRENT_DATE - data['max_date']

data['today_days'] = data['today_days'].apply(lambda x: x.days)

data.drop(['max_date'], axis = 1, inplace=True)
data.sample(5)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
# Попробуем Label Encoding

# le = LabelEncoder()

# le.fit(data['City'])

# data['City Code'] = le.transform(data['City'])
data = pd.get_dummies(data, columns=[ 'Country',], dummy_na=True)
# Попробуем Label Encoding

# le = LabelEncoder()

# le.fit(data['Country'])

# data['Country Code'] = le.transform(data['Country'])
data.columns
data.sample(5)
# на всякий случай, заново подгружаем данные

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем

data.info()
def preproc_data(df_input):

    '''includes several functions to pre-process the predictor data.'''

    

    df_output = df_input.copy()

    

    # ################### 1. NAN ############################################################## 

    df_output['Number_of_Reviews_isNAN'] = pd.isna(df_output['Number of Reviews']).astype('uint8')

    df_output['Price_Range_isNAN'] = pd.isna(df_output['Price Range']).astype('uint8')

    df_output['Cuisine_Style_isNAN'] = pd.isna(df_output['Cuisine Style']).astype('uint8')    

    

    # ################### 2. Encoding, Feature Engineering ##############################################

  

    # #### Price Range #### #

    

    df_output['Price Range'].fillna('$$ - $$$', inplace=True)

    prices_dict = {'$': 1, '$$ - $$$': 2, '$$$$': 3}

    df_output['Price'] = df_output['Price Range'].map(prices_dict)

    

    

    # #### Ranking #### #

    

    df_output['Ranking_Norm'] = df_output.groupby('City')['Ranking'].apply(lambda x: x / x.max())   

    df_output.drop(['Ranking'], axis = 1, inplace=True)

    

    

    # #### About City #### #

    

    df_output['Capital'] = df_output['City'].map(City_Capital)

    df_output['Population'] = df_output['City'].map(City_Population)

    df_output['Country'] = df_output['City'].map(City_Country)    

    

    

    # #### Number of Reviews #### #

    

    # Далее заполняем пропуски 0

    df_output['Number of Reviews'].fillna(0, inplace=True)

    # Заполняем средним

    #df_output['Number of Reviews'] = df_output.groupby('City')['Number of Reviews'].apply(lambda x:x.fillna(x.mean()))

    # Заполняем медианным

    #df_output['Number of Reviews'] = df_output.groupby('City')['Number of Reviews'].apply(lambda x:x.fillna(x.median())) 

    

    df_output['Number_of_Reviews_City'] = df_output['Number of Reviews'] / df_output['Population']

    

    #df_output.drop(['Number of Reviews'], axis = 1, inplace=True)

   



    # #### Cuisine Style #### #

    

    df_output['Cuisine Style'].fillna("['IsNan']", inplace=True)

    df_output['Cuisines_Count'] = df_output['Cuisine Style'].apply(lambda x: len(cuisine_style_list(x)))

    

    mlb = MultiLabelBinarizer()

    Cuisines_df = pd.DataFrame(mlb.fit_transform(df_output['Cuisine Style'].apply(cuisine_style_list)),

                   columns=mlb.classes_, index=df_output.index) 

    df_output = pd.concat([df_output, Cuisines_df], axis=1)

    

    

    # #### Reviews #### #

    

    df_output['Reviews'].fillna("[[], []]", inplace=True)

    

    df_output['dates'] = df_output['Reviews'].apply(lambda x: re.findall(r'(\d\d/\d\d/\d\d\d\d)', x))

    df_output['dates'] = df_output['dates'].apply(lambda x: list(map(pd.to_datetime,x))) 

    

    df_output['days'] = df_output['dates'].apply(lambda x: abs((x[0]-x[1]).days) if len(x) == 2 else 0)

    

    df_output['max_date'] = df_output['dates'].apply(lambda x: max(x) if len(x) > 0 else CURRENT_DATE)

    df_output['today_days'] = CURRENT_DATE - df_output['max_date']

    df_output['today_days'] = df_output['today_days'].apply(lambda x: x.days)

    df_output.drop(['max_date'], axis = 1, inplace=True)

    

    

    # #### City #### #

    

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True)   

    #le = LabelEncoder()

    #le.fit(df_output['City'])

    #df_output['City Code'] = le.transform(df_output['City'])

    

    

    # #### Country #### #

    

    #df_output = pd.get_dummies(df_output, columns=[ 'Country',], dummy_na=True)    

    le = LabelEncoder()

    le.fit(df_output['Country'])

    df_output['Country Code'] = le.transform(df_output['Country'])

    

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)

df_preproc.sample(10)
df_preproc.info()
def round_pred(y_pred):

    new_y_predict = []

    

    for y in y_pred:

        

        if y < 0.25:

             new_y_predict.append(0.0)

        elif y < 0.75:

             new_y_predict.append(0.5)

        elif y < 1.25:

             new_y_predict.append(1.0)

        elif y < 1.75:

             new_y_predict.append(1.5)

        elif y < 2.25:

             new_y_predict.append(2.0)

        elif y < 2.75:

             new_y_predict.append(2.5)

        elif y < 3.25:

             new_y_predict.append(3.0)

        elif y < 3.75:

             new_y_predict.append(3.5)

        elif y < 4.25:

             new_y_predict.append(4.0)

        elif y < 4.75:

             new_y_predict.append(4.5)

        else:

             new_y_predict.append(5.0)

    

    return(new_y_predict)
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



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

y_pred = model.predict(X_test)
# Округляем

y_pred = round_pred(y_pred) 
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
# Обучаем модель на полном тестовом наборе данных

model.fit(X_train, y_train)
predict_submission = model.predict(test_data)
# Округляем

predict_submission = round_pred(predict_submission)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)