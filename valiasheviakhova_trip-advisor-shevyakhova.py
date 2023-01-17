import pandas as pd  # импортируем все нужные библиотеки

import matplotlib.pyplot as plt

import seaborn as sns

from itertools import combinations

import numpy as np



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split
# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!

RANDOM_SEED = 42
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:

!pip freeze > requirements.txt
DATA_DIR = '/kaggle/input/trpadvsr/'

df_train = pd.read_csv('/kaggle/input/trpdvsr/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'kaggle_task VS.csv')

sample_submission = pd.read_csv(DATA_DIR+'sample_submission VS.csv')
df_train.head()
sample_submission.info()
df_train.info() #здесь мы имеем дело с базой данных Trip Advisor, содержащей 10 столбцов и 40 тыс. строк -

#данные по 40 тыс. ресторанов, разделенные на 10 параметров
# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



df = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
df.info() #проверка - да все объединилось, как надо.
df.head()
#проверим колонку "город" на наличие пропусков

df.City.isna().sum()
#дальше категориальные признаки нужно заменить числовыми. Сначала посмотрим на список городов и проверим,

#нет ли неправильных написаний или каких-то других багов, которые могут дублировать значения

df['City'].value_counts()
#Далее я попробовала сократить список значимых городов, но это никак не повлияло на модель, 

#поэтому я не вставляю этот шаг в финальное решение:

#Всего у нас 40 тыс. записей, и если следовать формуле Парето 80/20, то нас интересуют только те города, которые

#вносят вклад в 80% записей. это - 32 тыс. записей

#df['City'].value_counts()[:16].sum() # это - первые 16 городов, то есть половина всей выборки
#city_short = df['City'].value_counts()[:16]
#city_short = city_short.reset_index()

#city_short['index'].tolist()
#Оставим в базе только первые 16 городов - примерно половину, а остальные назовем other

#city_set = set()

#for item in city_short['index']:

 #   city_set.add(item)
#len(city_set)
#def find_item(cell):

 #   if item in cell:

  #      return 1

   # return 0

#функция, которая вернет dummies в колонки с популярными городами
#for item in city_set:

 #   df[item] = df['City'].apply(find_item)
#далее применим get_dummies:

df = pd.get_dummies(df, columns=['City'], dummy_na=True)
display(df.head())
df.info()
df['City_nan'].value_counts() #поскольку распределения нет, удалим эту колонку.
df = df.drop('City_nan', axis=1)
#посмотрим, есть ли пропуски

df.info()
df['Cuisine Style'][0] #посмотрим, что надпись в ячейке организована как список разных видов кухонь
df['Cuisine Style'] = df['Cuisine Style'].fillna('X]') #заменим пустые значения на X с лишней скобкой в конце для более простой обработки в дальнейшем
df['Cuisine Style'] = df['Cuisine Style'].apply(lambda x: x[:-1]) #удалим лишние скобки в конце каждой строки. здесь нам и пригодилась скобка у empty
df['Cuisine Style'] = df['Cuisine Style'].apply(lambda x: x.split(',')) #разделим кажду строку на элементы списка
df['Cuisine Style'][1]
len(df['Cuisine Style'][0]) #проверим, разбились ли строки на список
df['Cuisine Style'][0] #проверим, как выглядит список. Есть лишние кавычки.
#напишем функцию, которая очищает все элементы всех строк от лишних кавычек и запятых

def cut_c(line):

    new_list=[]

    for element in line:

        if element=='X':

            return 'X'

        else:

            new_list.append(element[2:-1])

    return new_list
df['Cuisine new'] = df['Cuisine Style'].apply(cut_c) #применим функцию
df['Cuisine new'] #ура! Получилось! я избавилась от лишних скобок и всего прочего!!!
cuisine_freq= pd.DataFrame(df['Cuisine new'].tolist()).stack().value_counts()

# здесь я смотрю на распределение разных видов кухонь по популярности
cuisine_freq = cuisine_freq.reset_index()
cuisine_freq.columns = ['index', 'freq']
pd.DataFrame(df['Cuisine new'].tolist()).stack().value_counts().sum()

#посмотрим, сколько всего разных упоминаний кухонь в датасете
pd.DataFrame(df['Cuisine new'].tolist()).stack().value_counts().sum() *0.8 #запуталась, как посчитать этот процентиль формулами,

#поэтому делаю вручную: 80% от всех упоминаний кухнь это - 105012.8
cuisine_freq.freq[:23].sum()
f_tolist = cuisine_freq['index'][0:23].tolist()
#Оставим в базе только первые 23 вида кухни - они составляют 80% всех упоминаний - а остальные назовем other
cuisine_short_set = set()

for item in f_tolist:

    cuisine_short_set.add(item)
def find_item(cell):

    if item in cell:

        return 1

    return 0

#функция, которая вернет dummies в колонки с типами кухонь
for item in cuisine_short_set:

    df[item] = df['Cuisine new'].apply(find_item)
df.shape # к датасету прибавилось 23 колонки
data = df.copy() #сделаем копию на всякий пожарный
import datetime

from datetime import datetime, timedelta

# здесь я работаю с датами, которые содержатся в колонке Review: я их выделяю и перевожу в формат to_datetime
data['Reviews'].isna() # есть два пропуска
data['Reviews'] = data['Reviews'].fillna('no data')
data['Reviews'].isna().sum()
data['Reviews'] = data['Reviews'].apply(lambda x: x.split(',')) #сначала я разделяю каждую запись в Review на элементы списка
len(data['Reviews'][1]) #проверим: теперь каждая запись - это список из нескольких элементов
data['Reviews'][1] #первые два элемента - это текст отзыва, а вторые два - даты, когда был оставлен отзыв. Они-то нам и нужны!
def cut_last(line): # эта функция выделит мне последние даты из каждой строки.

    return line[-1][2:-3]
data['Reviews_lastdate'] = data['Reviews'].apply(cut_last) #применяю функцию к колонке и переношу все выделенные даты в отдельную колонку
data['Reviews_lastdate'].isna().sum()
data['Reviews_lastdate'] = data['Reviews_lastdate'].apply(lambda x: None if x == 'data' else x)
def cut_first(line): # а эта функция выделит мне первые даты из каждой строки.

    if len(line)>=2:

        return line[-2][3:-1]
data['Reviews_firstdate'] = data['Reviews'].apply(cut_first)
data['Reviews_firstdate'] = data['Reviews_firstdate'].apply(lambda x: x.lower() if x!= None else None) 

#с этой колонкой больше сложностей: в ней иногда попадается текст, написанный чем попало. поэтому на всякий случай

#я перевожу весь текст в строчные буквы
def cut_again(line):

    if len(line)<2:

        return None

    elif len(line)>=2:

        if line[0] == "'":

            return line[1:]

    else:

        return line
data['Reviews_lastdate'] = data['Reviews_lastdate'].apply(cut_again)
data['Reviews_lastdate'] = pd.to_datetime(data['Reviews_lastdate']) #перевожу данные в первой колонке в формат to_datetime
def replace(line):

    if 'e' in line:

        return None

    if 'a' in line:

        return None

    if 'r' in line:

        return None

    if 'i' in line:

        return None

    if 'o' in line:

        return None

    if 'u' in line:

        return None

    if 't' in line:

        return None

    if 'w' in line:

        return None

    if '厅' in line:

        return None

    if 'm' in line:

        return None

    if '/' not in line:

        return None

    return line

#ищу текст в колонке и заменяю его на None
data[data['Reviews_firstdate'].isna()==True]
data['Reviews_firstdate'] = data['Reviews_firstdate'].fillna('o') 

#пока я none-значения не заменила на букву, у меня не работала функция....
data['Reviews_firstdate'] = data['Reviews_firstdate'].apply(replace) 

#теперь я могу применитьфункцию по удалению текста
data['Reviews_firstdate'].isna().sum()
data['Reviews_firstdate'].isna().sum()

#проверим, работает ли функция  - да, работает, в колонке появились NONE-значения
data['Reviews_firstdate'] = pd.to_datetime(data['Reviews_firstdate'])

#переведем в to_datetime
data['Reviews_deltadates'] = data['Reviews_lastdate'] - data['Reviews_firstdate']
display(data.head())
print(data['Reviews_deltadates'].describe(datetime_is_numeric=True)) # ну и вызовем статистику

print()

print(data['Reviews_firstdate'].describe(datetime_is_numeric=True))

print()

print(data['Reviews_lastdate'].describe(datetime_is_numeric=True))
data['ordinal_firstdate']=data['Reviews_firstdate'].map(datetime.toordinal)

#поскольку линейная регрессия не умеет работает с данными типа "даты", переведем все даты в ординальный формат
data['ordinal_lastdate']=data['Reviews_lastdate'].map(datetime.toordinal)
data['Reviews_lastdate'][10]
data['ordinal_timedelta'] = data['ordinal_lastdate'] - data['ordinal_firstdate']

#здесь я создаю новую колонку - разница во времени между последним и первым отзывом. Возможно, это важно.
data['ordinal_timedelta'] 

# и проверим, как это выглядит. интересно же. Отрицательные числа говорят о том, что

# иногда дата в колонке firstdate оказывается более свежей, чем в колонке lastdate
latest = data.copy() 

#здесь я хочу создать еще одну колонку, в которой содержатся только самые свежие даты отзывов,

#но специального метода не знаю, поэтому сделаю это как получается :) создаю копию датасета
latest = latest.loc[latest['ordinal_firstdate'] >= latest['ordinal_lastdate']]

#выделяю в датасете только те строки, где дата из первой колонки- позже даты из второй колонки
latest
latest['latest_review_date'] = latest['ordinal_firstdate']

# создаю новую колонку, в которую переношу самые свежие даты из ordinal first_date
latest2 = data.copy() #дальше повторяю всю ту же процедуру для другой колонки:
latest2 = latest2.loc[latest2['ordinal_firstdate'] < latest2['ordinal_lastdate']]

#выделяю только те строки, где самые свежие даты содержатся в колонке lastdate
latest2['latest_review_date'] = latest2['ordinal_lastdate']
new_df = pd.concat([latest, latest2], ignore_index=True) #объединяю два датасета, так что у меня теперь появляется новая целая колонка
new_df.info()
## хотя я и потратила кучу сил на то, чтобы создать эту отдельную колонку с самыми свежими датами ревью, однако она

## она никак не улучшает модель.  Лучшие результаты достигаются с колонкой ordinal_lastdate или ordinal_timedelta
new_df['Price Range'].isnull().sum() #посмотрим на количество пропусков
new_df['Price Range'].value_counts() # а это распределение разных ценовых категорий по датасету
def price_range(x): # функция, которая меняет категориальное значение числовым. 

    if x == '$':

        return 1

    if x == '$$ - $$$':

        return 2.5

    if x == '$$$$':

        return 4   
new_df['price_range_index'] = new_df['Price Range'].apply(price_range) #создаю новую колонку с числовыми значениями
#попробуем заменить NaN в пропорциональном соотношении:

#общая сумма всех заполненных ячеек:

print(f'sum = {23041+7816+1782}')

print(f'1 = {7816/32639}')

print(f'2.5={23041/32639}')

print(f'4 = {1782/32639}')

print('Количество пропусков: ', new_df['price_range_index'].isna().sum())
#посчитаем, сколько каких значений нам нужно отдать пустым ячейкам:

print(f'1: {round(17359*0.23946)}')

print(f'2.5: {round(17359*0.705934)}')

print(f'4: {round(17359*0.05459)}')
new_df[new_df['price_range_index'].isna()][0:4157]
new_df['price_range_index'] = new_df['price_range_index'][0:13235].fillna(1)
new_df[new_df['price_range_index'].isna()][0:12255]
new_df['price_range_index'] = new_df['price_range_index'][0:25490].fillna(2.5) #заполним 9790 ячеек значением 2.5
new_df['price_range_index'] = new_df['price_range_index'].fillna(4) #остальные ячейки заполним значением 4
new_df['price_range_index'].isna().sum()  # проверяем - все пустые ячейки заполнены
new_df['Number of Reviews'].isna().sum() #посмотрим на пропуски в этой колонке
new_df['Number of Reviews'] = new_df['Number of Reviews'].fillna(0)

#заполним их нулями - раз отзывов нет, значит, скорее всего, их, и правда, нет
new_df['Number of Reviews'].isna().sum() # проверим, как все заполнилось
new_df['Number of Reviews'].median() #посмотрим на медианное значение по количеству отзывов
new_df['Number of Reviews'].describe() # виден очень большой разброс между медианным и средним значением
new_df['Number of Reviews'].hist(bins=50)
sns.boxplot(new_df['Number of Reviews']); #посмотрим на выбросы на графике:
IQR = new_df['Number of Reviews'].quantile(0.75) - new_df['Number of Reviews'].quantile(0.25)

perc25 = new_df['Number of Reviews'].quantile(0.25)

perc75 = new_df['Number of Reviews'].quantile(0.75)

print('25 квантиль -', perc25, '75 квантиль -', perc75)

print(f'верхняя граница выбросов: {perc75+1.5*IQR}')
new_df['Ranking'].isna().sum() #посмотрим на пропуски здесь
new_df['Ranking'].describe()
new_df['Ranking'].hist();
sns.boxplot(new_df['Ranking']); #похоже, есть выбросы
IQR = new_df.Ranking.quantile(0.75) - new_df.Ranking.quantile(0.25)

perc25 = new_df.Ranking.quantile(0.25)

perc75 = new_df.Ranking.quantile(0.75)

print(f'верхняя граница выбросов: {perc75+1.5*IQR}')
#new_df.Ranking = new_df.Ranking.apply(lambda x: 0 if x >11645 else x)
new_df.Rating.isna().sum()# в целевой переменной выбросов нет
new_df.Rating.describe() 
new_df.Rating.hist(bins=20)
sns.boxplot(new_df.Rating)

#видно, что в основном рестораны получают средние оценки, с крепкой "четверкой" по медиане
new_df.info()

#нужно избавиться от столбцов с категориальными переменными и переменными типа datetime:
new_df = new_df.drop(['Restaurant_id', 'Cuisine new', 'Reviews_lastdate', 'Reviews_firstdate', 'Reviews_deltadates','Reviews', 'Price Range','Cuisine Style', 'ordinal_firstdate', 'ordinal_timedelta', 'latest_review_date', 'URL_TA', 'ID_TA'], axis=1)
# Теперь выделим тестовую часть

train_data = new_df.query('sample == 1').drop(['sample'], axis=1)

test_data = new_df.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)
test_data.shape
sample_submission
# Воспользуемся специальной функцией train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

# Загружаем специальный инструмент для разбивки:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
# проверяем

test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели
# Создаём модель

regr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)



# Обучаем модель на тестовом наборе данных

regr.fit(X_train, y_train)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = regr.predict(X_test)
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
test_data = test_data.drop(['Rating'], axis=1) #возьмем датасет, выделенный специально для тестирования модели, 
predict_submission = regr.predict(test_data) #применим предсказательную модель к тестовому датасету
predict_submission
sample_submission['Rating'] = predict_submission #перезапишем столбец Rating с учетом новых данных
sample_submission
sample_submission.to_csv('submission_draft11.csv', index=False) # и запишем новый результат в файл