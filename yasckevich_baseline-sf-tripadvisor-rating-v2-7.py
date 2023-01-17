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
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
col = ['Restaurant_id', 'City', 'Cuisine_Style', 'Ranking', 'Price_Range', 'Number_of_Reviews', 'Reviews', 'URL_TA', 'ID_TA','sample','Rating']

data.columns = col
data.info(100000)
data
data.Restaurant_id.value_counts()

# пропусков нет, все ок



print(data.Restaurant_id.describe())

# по условию - в одному id  могут быть прикреплены несколько ресторанов одной сети,

# поэтому уникальных значений меньше чем записей



data.Restaurant_id = [i[3:] for i in data.Restaurant_id]

data.Restaurant_id = [int(i) for i in data.Restaurant_id]



print(' ')

print(len(data.Restaurant_id))

print(type(data.Restaurant_id[1]))



# превратили столбец в численное выражение
a=data.Restaurant_id.value_counts()

a=dict(a)

a=pd.DataFrame({'count':a}, columns=['count'])



A=a.index

B=a.values



data['chain']= data['Restaurant_id'].replace(A, B)
data.City.value_counts()

# пропусков нет, города не повторяются, проблем с ракладкой или регистром нет, все ок



print(data.City.describe())



A = list(data.City.value_counts().keys())

B = range(0, len(A))

dict_city = dict(zip(A, B))

# словарь со значениями городов



data['City_ind'] = data['City'].replace(A, B)



print(' ')

print(len(data.City_ind))

print(type(data.City_ind[1]))



# аномалий нет + мы заменили города на числовые индефикаторы
A = list(data.City.value_counts().keys())

data1=pd.DataFrame()

data1['City'] = data['City']

data1['Ranking']=data['Ranking']

data1=pd.DataFrame(data1.groupby(['City']).max())

data2=data1['Ranking']

A=list(data2.keys())

B=list(data2)



data['Len_rest_in_city'] = data['City'].replace(A, B)



data['Std_ranking']=data['Ranking']/data['Len_rest_in_city']



data.Std_ranking.describe()
B = [8961989, 2148271, 3266126, 1636762, 3769495, 1397852, 4110000, 1324277, 506654, 1888776, 860124,

     179277, 1899160, 1471508, 481181, 975904, 1752286, 1790658, 1173179, 794128, 664046, 513210, 424008,

     237591, 201818, 779115, 693491, 643272, 432862, 626108, 295504]

dict_res = dict(zip(A, B))

# словарь со кол-вом жителей



data['Residents'] = data['City'].replace(A, B)



print(len(data.Residents))

print(type(data.Residents[1]))
data['Country'] = data.City

data['Country'] = data['Country'].replace('London', 'GreatBritain')

data['Country'] = data['Country'].replace('Paris', 'France')

data['Country'] = data['Country'].replace('Madrid', 'Spain')

data['Country'] = data['Country'].replace('Barcelona', 'Spain')

data['Country'] = data['Country'].replace('Berlin', 'Germany')

data['Country'] = data['Country'].replace('Milan', 'Italy')

data['Country'] = data['Country'].replace('Rome', 'Italy')

data['Country'] = data['Country'].replace('Prague', 'CzechRepublic')

data['Country'] = data['Country'].replace('Lisbon', 'Portugal')

data['Country'] = data['Country'].replace('Vienna', 'Austria')

data['Country'] = data['Country'].replace('Amsterdam', 'Netherlands')

data['Country'] = data['Country'].replace('Brussels', 'Belgium')

data['Country'] = data['Country'].replace('Hamburg', 'Germany')

data['Country'] = data['Country'].replace('Munich', 'Germany')

data['Country'] = data['Country'].replace('Lyon', 'France')

data['Country'] = data['Country'].replace('Stockholm', 'Sweden')

data['Country'] = data['Country'].replace('Budapest', 'Hungary')

data['Country'] = data['Country'].replace('Warsaw', 'Poland')

data['Country'] = data['Country'].replace('Dublin', 'Irland')

data['Country'] = data['Country'].replace('Copenhagen', 'Denmark')

data['Country'] = data['Country'].replace('Athens', 'Greece')

data['Country'] = data['Country'].replace('Edinburgh', 'Scotland')

data['Country'] = data['Country'].replace('Zurich', 'Switzeland')

data['Country'] = data['Country'].replace('Oporto', 'Portugal')

data['Country'] = data['Country'].replace('Geneva', 'Switzeland')

data['Country'] = data['Country'].replace('Krakow', 'Poland')

data['Country'] = data['Country'].replace('Oslo', 'Norway')

data['Country'] = data['Country'].replace('Helsinki', 'Finland')

data['Country'] = data['Country'].replace('Bratislava', 'Slovakia')

data['Country'] = data['Country'].replace('Luxembourg', 'Luxembourg')

data['Country'] = data['Country'].replace('Ljubljana', 'Slovenia')



A = list(data.Country.value_counts().keys())

B = range(0, len(A))

dict_country = dict(zip(A, B))

# словарь со значениями стран





data['Country_ind'] = data['Country'].replace(A, B)



print(len(data.Country_ind))

print(type(data.Country_ind[1]))
data['Rew_of'] = data['Number_of_Reviews']/data['Residents']

data['Rew_of'] = data['Rew_of'].fillna(0)







print(len(data.Rew_of))

print(type(data.Rew_of[1]))
data['Cuisine_Style'] = data['Cuisine_Style'].fillna('""No_info"')

# заменяем пропуски



new = pd.DataFrame(data.Cuisine_Style.dropna())

a = list(new.Cuisine_Style)

b = list()



def l(x):

    i = 0

    for g in x:

        f = x[i].split(',')

        v = 0

        for g in f:

            h = f[v][2:-1].replace("'", '')

            v = +1

            b.append(h)

        i += 1

        

l(a)



from collections import Counter



coun=Counter(b)

coun=dict(coun)

coun=pd.DataFrame({'count':coun}, columns=['count'])

a=coun['count'].mean()



b=list(coun.query('count > @a').index)

b



#оставляем только самые популярные кухни



def find_item(cell):

    if item in cell:

        return 1

    return 0





for item in b:

    data[item] = data['Cuisine_Style'].apply(find_item)



data['Cuisine_Style'] = data['Cuisine_Style'].apply(lambda x: len(x))



len(data['Cuisine_Style'])

# пропуски заменены, строки преобразованы в столбцы
print(data.Price_Range.describe())



data['Price_Range'] = data['Price_Range'].replace('$', 1)

data['Price_Range'] = data['Price_Range'].replace('$$ - $$$', 2)

data['Price_Range'] = data['Price_Range'].replace('$$$$', 3)



a = data['Price_Range'].mean()

a = int(a)

data['Price_Range'] = data['Price_Range'].fillna(a)



print('')

print(len(data.Price_Range))

print(type(data.Price_Range[1]))

# заменили буквенные индефикаторы на численные
print(data.Reviews.describe())



from datetime import datetime, date, time

data['Reviews'] = data.Reviews.replace("[[], []]", 'No_info')

# заменяем пропуски



data['Last_rew'] = data['Reviews']



data['Last_rew']=data['Last_rew'].str[-27:-17]



now = datetime.now()



#base['Last_rew'][base.Last_rew.str.contains("]")]=now

data['Last_rew'][data.Last_rew.str.contains("]")==True] = now

data['Last_rew'] = data['Last_rew'].fillna(now)



# приравниваем строки без даты к сегодня



data['Last_rew'] = [pd.to_datetime(i) for i in data.Last_rew]



# добавляем сколько прошло времени с момента последнего отзыва



data['Last_rew_data'] = data['Last_rew']-now

data['Last_rew_data'] = [i.total_seconds() for i in data.Last_rew_data]

data['Last_rew_data'] = data['Last_rew_data']

data['Last_rew_data'] = data['Last_rew_data'].fillna(0)







print('')

print(len(data.Last_rew_data))

print(type(data.Last_rew_data[1]))
from datetime import datetime, date, time



m=data['Number_of_Reviews'].mean()



print(data['Number_of_Reviews'].describe())

# отрицательных чисел нет



data['Number_of_Reviews'].value_counts()

# но есть пропуски, нужно проверить, что где есть пропуски - там действительно нет отзывов



data['Number_of_Reviews'] = data['Number_of_Reviews'].fillna('No_info')

# меняем пропуски на Not_info



a = data.query('Reviews == "No_info" & Number_of_Reviews == "No_info"')

b = list(a.index)

data['Number_of_Reviews'][b] = 0

# где нет значения кол-ва отзывов и превью отзывов - ставим 0



len(data['Number_of_Reviews'])

# заменили предполагаемые пропуски, в некоторые смогли подставить значения. С 2,5+ тысяч снизили пропуски до 900+



data['Number_of_Reviews'] = data['Price_Range'].replace('No_info', m)



# пропуски, которые не смогли заполнить, заполняем средним значением



#data.Number_of_Reviews = [float(i) for i in data.Number_of_Reviews]



print('')

print(len(data.Number_of_Reviews))

print(type(data.Number_of_Reviews[1]))
A = list(data.City.value_counts().keys())

data1=pd.DataFrame()

data1['City'] = data['City']

data1['Number_of_Reviews']=data['Number_of_Reviews']



data1=pd.DataFrame(data1.groupby(['City']).sum())

data2=data1['Number_of_Reviews']

A=list(data2.keys())

B=list(data2)







data['Len_rew'] = data['City'].replace(A, B)



data['Std_num_rew']=data['Number_of_Reviews']/data['Len_rew']
print(data['ID_TA'].describe())



data['ID_TA']=data['ID_TA'].str[1:]

data.ID_TA = [float(i) for i in data.ID_TA]



print('')

print(len(data.ID_TA))

print(type(data.ID_TA[1]))
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

sns.heatmap(data[col].drop(['sample'], axis=1).corr(),)
data = data.drop('URL_TA', axis=1)

data = data.drop('Reviews', axis=1)

data = data.drop('Last_rew', axis=1)

data = data.drop('City', axis=1)

data = data.drop('Country', axis=1)

data = data.drop('Restaurant_id', axis=1)
#data = data.drop('Len_rest_in_city', axis=1)
data = data.drop('chain', axis=1)
data = data.drop('Cuisine_Style', axis=1)
plt.rcParams['figure.figsize'] = (15,10)

sns.heatmap(data.drop(['sample'], axis=1).corr(),)
pd.options.display.max_info_columns 

data.info(verbose=True, max_cols=False, null_counts=True)
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
len(test_data)
len(sample_submission)
predict_submission = model.predict(test_data)



len(predict_submission)
def round_nearest(x, a):

    return round(x / a) * a



sample_submission['Rating'] = predict_submission.round(1)

sample_submission['Rating'] = round_nearest(sample_submission['Rating'], 0.5)





sample_submission.head(10)



sample_submission.to_csv('submission.csv', index=False)