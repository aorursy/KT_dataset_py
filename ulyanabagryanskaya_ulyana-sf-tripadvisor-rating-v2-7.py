# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re



import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



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
# Для примера я возьму столбец Number of Reviews

data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
data['Number_of_Reviews_isNAN']
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...

data['Number of Reviews'].fillna(0, inplace=True)
data['Cuisine Style isNaN'] = pd.isna(data['Cuisine Style']).astype('uint8')

data['Price Range isNaN'] = pd.isna(data['Price Range']).astype('uint8')
data['Price Range isNaN'] = pd.isna(data['Price Range']).astype('uint8')

data.sample(5)
data.nunique(dropna=False)
addata = data['City'].value_counts().rename_axis('City').to_frame(name='Rest')

data = data.merge(addata, on='City', how='left')

data['New Rank'] = data['Ranking'] / data['Rest']

data.sample(5)
# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

#data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
data['Price Range'].value_counts()
# Ваша обработка 'Price Range'

data['Price Range'].value_counts() # самое популярное -- $$ - $$$ -- заменим пропуски на него) 

data['Price Range'] = data['Price Range'].fillna('$$ - $$$')

data['Price Range'].isna().sum() # чекаем, точно ли заменилось, потому что - очевидно - писал невротик
data['Price Range'] = data['Price Range'].replace('$', '1')

data['Price Range'] = data['Price Range'].replace('$$ - $$$', '2')

data['Price Range'] = data['Price Range'].replace('$$$$', '3')
# Обработка признака Cuisine Style

data['Cuisine Style'].isna().sum()

k = data['Cuisine Style'].str.split("', '").str.len().fillna(1)

data['Style Amount'] = k

data.sample(5)
pattern = re.compile('\d{2}/\d{2}/\d{4}')

reviews=[]

for i in data['Reviews']:

    reviews.append(re.findall(pattern, str(i)))

rev = pd.DataFrame(reviews).dropna()

rev.columns=['date1', 'date2']

rev['date1'] = pd.to_datetime(rev['date1']) 

rev['date2'] = pd.to_datetime(rev['date2']) 

rev['dd']= rev['date1']-rev['date2']



data['date1'] = rev['date1'].max() - rev['date1']

data['date1'] = data['date1'].apply(lambda x: x.days)

data['date1'] = data['date1'].fillna(0)

data['date1'] = data['date1'].apply(lambda x: int(x))



data['date2'] = rev['date1'].max() - rev['date2']

data['date2'] = data['date2'].apply(lambda x: x.days)

data['date2'] = data['date2'].fillna(0)

data['date2'] = data['date2'].apply(lambda x: int(x))



data['dd'] = data['date2'] - data['date1']



data.head()
# Добавим еще признак: City_Population

City_Population = {

    'London': 9304016, 'Paris': 2140526, 'Madrid': 3348536,'Barcelona': 1620343, 'Berlin': 3748148,

    'Milan': 1404239, 'Rome': 2856133, 'Prague': 1324277, 'Lisbon': 506654, 'Vienna': 1911728,

    'Amsterdam': 873555, 'Brussels': 1209000, 'Hamburg': 1841179, 'Munich': 1471508, 'Lyon': 515695,

    'Stockholm': 974073, 'Budapest': 1752286, 'Warsaw': 1790658, 'Dublin': 554554, 'Copenhagen': 626508,

    'Athens': 664046, 'Edinburgh': 524930, 'Zurich': 415367, 'Oporto': 237559, 'Geneva': 201818,

    'Krakow': 779115, 'Oslo': 693491, 'Helsinki': 648042, 'Bratislava': 432864, 'Luxembourg': 613894,

    'Ljubljana': 292988

}

data['City_Population'] = data['City'].map(City_Population)

data.sample(5)
Country = {

    'Amsterdam': 'Netherlands', 'Athens': 'Greece', 'Barcelona': 'Spain', 'Berlin': 'Germany',

    'Bratislava': 'Slovakia', 'Brussels': 'Belgium', 'Budapest': 'Hungary', 'Copenhagen': 'Denmark',

    'Dublin': 'Ireland', 'Edinburgh': 'United Kingdom', 'Geneva': 'Switzerland', 'Hamburg': 'Germany',

    'Helsinki': 'Finland', 'Krakow': 'Poland', 'Lisbon': 'Portugal', 'Ljubljana': 'Slovenia',

    'London': 'United Kingdom', 'Luxembourg': 'Luxembourg', 'Lyon': 'France', 'Madrid': 'Spain',

    'Milan': 'Italy', 'Munich': 'Germany', 'Oporto': 'Portugal', 'Oslo': 'Norway', 'Paris': 'France',

    'Prague': 'Czechia', 'Rome': 'Italy', 'Stockholm': 'Sweden', 'Vienna': 'Austria','Warsaw': 'Poland',

    'Zurich': 'Switzerland'

}



data['Country'] = data['City'].map(Country)



Capital = ['Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna', 'Rome', 'Madrid',

       'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen','Amsterdam', 'Lisbon', 'Prague', 'Oslo',

       'Helsinki', 'Ljubljana', 'Athens', 'Luxembourg']



data['Is_capital'] = data['City'].apply(lambda row: row in Capital).astype('uint8')



data.sample(5)
# Добавим признак - сколько тратит житель страны на рестораны: Costs

Costs = {

    'Belgium': 1280.0, 'Bulgaria': 350.0, 'Czechia': 840.0, 'Denmark': 1500.0,  'Germany': 1100.0,

    'Estonia': 830.0, 'Ireland': 3350.0, 'Greece': 1425.0, 'Spain': 2270.0, 'France': 1370.0,

    'Italy': 1820.0, 'Cyprus': 2940.0, 'Latvia': 600.0, 'Lithuania': 410.0, 'Luxembourg': 2350.0,

    'Hungary': 630.0, 'Malta': 2820.0, 'Netherlands': 1690.0, 'Austria': 3080.0, 'Poland': 270.0,

    'Portugal': 1800.0, 'Romania': 210.0, 'Slovenia': 930.0, 'Slovakia': 550.0, 'Finland': 1430.0,

    'Sweden': 1420.0, 'United Kingdom': 2120.0, 'Iceland': 4710.0, 'Norway': 1940.0,

    'Switzerland': 1425.0

     }



data['Costs'] = data['Country'].map(Costs)



data.sample(5)
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
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
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

    

    # ################### 1. Предобработка ############################################################## 

    # убираем не нужные для модели признаки

    df_output.drop(['Restaurant_id','ID_TA',], axis = 1, inplace=True)

    df_output['City1'] = df_output['City']

    

    

    # ################### 2. NAN ############################################################## 

    

    df_output['Number_of_Reviews_isNAN'] = pd.isna(df_output['Number of Reviews']).astype('uint8')

    df_output['Number of Reviews'].fillna(0, inplace=True)

    

    df_output['Cuisine Style isNaN'] = pd.isna(df_output['Cuisine Style']).astype('uint8')

    df_output['Price Range isNaN'] = pd.isna(df_output['Price Range']).astype('uint8')

    

    # тут ваш код по обработке NAN 

    df_output['Price Range'] = df_output['Price Range'].fillna('$$ - $$$')

    

    

    # ################### 3. Encoding ############################################################## 

    # Price Range

    df_output['Price Range'] = df_output['Price Range'].replace('$', '1')

    df_output['Price Range'] = df_output['Price Range'].replace('$$ - $$$', '2')

    df_output['Price Range'] = df_output['Price Range'].replace('$$$$', '3')



    

    # ################### 4. Feature Engineering ####################################################

    

    # Amount of restaurants per city

    addata = df_output['City1'].value_counts().rename_axis('City1').to_frame(name='Rest')

    df_output = df_output.merge(addata, on='City1', how='left')

    df_output['New Rank'] = df_output['Ranking'] / df_output['Rest']



    # Amount of cuisine style per restaurant

    k = df_output['Cuisine Style'].str.split("', '").str.len().fillna(1)

    df_output['Style Amount'] = k

    

    # Добавляем 3 новых столбца: 

    # время между последним комментарием в датасете и последнем в ресторане: date1 

    # время между предпоследним комментарием в датасете и предпоследнем в ресторане: date2 

    # время между комментариями: dd

    

    pattern = re.compile('\d{2}/\d{2}/\d{4}')

    reviews=[]

    for i in df_output['Reviews']:

        reviews.append(re.findall(pattern, str(i)))

    rev = pd.DataFrame(reviews).dropna()

    rev.columns=['date1', 'date2']

    rev['date1'] = pd.to_datetime(rev['date1']) 

    rev['date2'] = pd.to_datetime(rev['date2']) 

    rev['dd']= rev['date1']-rev['date2']



    df_output['date1'] = rev['date1'].max() - rev['date1']

    df_output['date1'] = df_output['date1'].apply(lambda x: x.days)

    df_output['date1'] = df_output['date1'].fillna(0)

    df_output['date1'] = df_output['date1'].apply(lambda x: int(x))



    df_output['date2'] = rev['date1'].max() - rev['date2']

    df_output['date2'] = df_output['date2'].apply(lambda x: x.days)

    df_output['date2'] = df_output['date2'].fillna(0)

    df_output['date2'] = df_output['date2'].apply(lambda x: int(x))

    

    # Добавим еще признак: City_Population

    

    City_Population = {

    'London': 9304016, 'Paris': 2140526, 'Madrid': 3348536,'Barcelona': 1620343, 'Berlin': 3748148,

    'Milan': 1404239, 'Rome': 2856133, 'Prague': 1324277, 'Lisbon': 506654, 'Vienna': 1911728,

    'Amsterdam': 873555, 'Brussels': 1209000, 'Hamburg': 1841179, 'Munich': 1471508, 'Lyon': 515695,

    'Stockholm': 974073, 'Budapest': 1752286, 'Warsaw': 1790658, 'Dublin': 554554, 'Copenhagen': 626508,

    'Athens': 664046, 'Edinburgh': 524930, 'Zurich': 415367, 'Oporto': 237559, 'Geneva': 201818,

    'Krakow': 779115, 'Oslo': 693491, 'Helsinki': 648042, 'Bratislava': 432864, 'Luxembourg': 613894,

    'Ljubljana': 292988

    }

    

    df_output['City_Population'] = df_output['City1'].map(City_Population)

    

    # Добавим признак Is_capital: столица ли город, в котором находится ресторан. 

    # Для этого сделаем еще словарь

    

    Country = {

    'Amsterdam': 'Netherlands', 'Athens': 'Greece', 'Barcelona': 'Spain', 'Berlin': 'Germany',

    'Bratislava': 'Slovakia', 'Brussels': 'Belgium', 'Budapest': 'Hungary', 'Copenhagen': 'Denmark',

    'Dublin': 'Ireland', 'Edinburgh': 'United Kingdom', 'Geneva': 'Switzerland', 'Hamburg': 'Germany',

    'Helsinki': 'Finland', 'Krakow': 'Poland', 'Lisbon': 'Portugal', 'Ljubljana': 'Slovenia',

    'London': 'United Kingdom', 'Luxembourg': 'Luxembourg', 'Lyon': 'France', 'Madrid': 'Spain',

    'Milan': 'Italy', 'Munich': 'Germany', 'Oporto': 'Portugal', 'Oslo': 'Norway', 'Paris': 'France',

    'Prague': 'Czechia', 'Rome': 'Italy', 'Stockholm': 'Sweden', 'Vienna': 'Austria','Warsaw': 'Poland',

    'Zurich': 'Switzerland'

    }



    df_output['Country'] = df_output['City1'].map(Country)



    Capital = ['Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna', 'Rome', 'Madrid',

       'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen','Amsterdam', 'Lisbon', 'Prague', 'Oslo',

       'Helsinki', 'Ljubljana', 'Athens', 'Luxembourg']



    df_output['Is_capital'] = df_output['City1'].apply(lambda row: row in Capital).astype('uint8')

    

    # Добавим признак - сколько тратит житель страны на рестораны: Costs



    Costs = {

    'Belgium': 1280.0, 'Bulgaria': 350.0, 'Czechia': 840.0, 'Denmark': 1500.0,  'Germany': 1100.0,

    'Estonia': 830.0, 'Ireland': 3350.0, 'Greece': 1425.0, 'Spain': 2270.0, 'France': 1370.0,

    'Italy': 1820.0, 'Cyprus': 2940.0, 'Latvia': 600.0, 'Lithuania': 410.0, 'Luxembourg': 2350.0,

    'Hungary': 630.0, 'Malta': 2820.0, 'Netherlands': 1690.0, 'Austria': 3080.0, 'Poland': 270.0,

    'Portugal': 1800.0, 'Romania': 210.0, 'Slovenia': 930.0, 'Slovakia': 550.0, 'Finland': 1430.0,

    'Sweden': 1420.0, 'United Kingdom': 2120.0, 'Iceland': 4710.0, 'Norway': 1940.0,

    'Switzerland': 1425.0

     }



    df_output['Costs'] = df_output['Country'].map(Costs)

    

    df_output = pd.get_dummies(df_output, columns=[ 'City'], dummy_na=True)

    

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберем их и удалим

    

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)

df_preproc.sample(10)
df_preproc.info()
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
predict_submission = model.predict(test_data)
predict_submission
sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)