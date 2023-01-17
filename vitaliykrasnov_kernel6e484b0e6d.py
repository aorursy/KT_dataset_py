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
DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating'

df_train = pd.read_csv(DATA_DIR+'/main_task.csv')

df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')

# ВАЖНО! для корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
#================================================

# "Number of reviews"

#================================================

nor_mean = data.groupby(["City"])['Number of Reviews'].mean()

# функция возвращает среднее кол-во отзывов по городу если Nan или текущее значение кол-ва отзывов

def number_of_reviews_filter(row):

    if  pd.isna(row['Number of Reviews']): 

        return float(round(nor_mean[row['City']]))

    else:

        return row['Number of Reviews']

# создаём новую колонку с кол-вом отзывов без Nan     

nor_new = data.apply(number_of_reviews_filter, axis=1)

# присваиваем новую колонку старой     

data['Number of Reviews'] = nor_new

#================================================

# "Price Range"

#================================================

# все уникальные значения диапазонов цен переводим в цифры

# больше всего ресторанов в средней ценовой категории, поэтому заполняем пропущенные значение 

# значением "2"

data['Price Range'].fillna(2, inplace=True)

def price_range_filter(p):

    if str(p) == '$$ - $$$':

        return 2

    elif str(p) == '$$$$':

        return 3

    elif str(p) == '$': 

        return 1

    else:

        return p

# добавляем в дата фрейм новую колонку с числовыми признаками диапазонов цен    

data['price_range'] = data['Price Range'].apply(lambda x: price_range_filter(x))

#================================================

# "Cuisine Style"

#================================================

# возвращаем список типов кухонь для ресторана

def cuisine_filter(c):

    c = c[1:-1].replace("'","").split(',')

    c = [i.strip() for i in c]

    return c

# для значений Nan принимаем кол-во типов кухонь равное 1        

# в новой колонке для кухонь меняем список типов кухонь на их кол-во, для Nan значений возвращаем 1

cuisine_series = data['Cuisine Style'].apply(lambda c: 1 if pd.isna(c) else len(cuisine_filter(c)))

# добавляем новую колонку с количестом типов кухонь для каждого ресторана

data['cuisine_style'] = cuisine_series 



# реализуем концепцию "dummy variables" - концепия для типов кухонь не взлетела, на графике самых важных признаков нет признаков по типам кухни

#cuisine_series2 = df['Cuisine Style'].fillna(1)

#cuisine_series2 = data['Cuisine Style'].apply(lambda c: [] if pd.isna(c) else cuisine_filter(c))

# добавляем новую колонку со значением являющимся строкой со списком типов кухонь для каждого ресторана

#data['cuisine_style_arr'] = cuisine_series2 



# в рамках заданий по модулю находим список наиболее часто встречающихся (НЧВ) типов кухонь и берём 

# те, которые встречаются больше 1000 раз

#most_freq_cuisine = ['Vegetarian Friendly' ,'European' ,'Mediterranean' ,'Italian' ,'Vegan Options' ,'Gluten Free Options' ,'Bar' ,'French' ,'Asian' ,'Pizza' ,'Spanish' ,'Pub' ,'Cafe' ,'Fast Food' ,'British' ,'International' ,'Seafood' ,'Japanese' ,'Central European' ,'American' ,'Sushi' ,'Chinese' ,'Portuguese' ,'Indian' ,'Middle Eastern']

# добавляем колонки из списка НЧВ со значениями 0

#for x in most_freq_cuisine:

#    data[x] = 0

# функция добавляет для каждой строки дата фрейма колонки с именами ранее определённых НЧВ типов кухонь в данном

# ресторане и со значением 1 если такой тип кухни есть у данного ресторана и 0 если нет

# также есть колонка с типом кухни "Other" в которой стоит 1 если ни один из типов кухни в ресторане 

# не попал в список

# def set_dummy_vars(row):

#     c_list = row['cuisine_style_arr']

#     if len(c_list) == 0:

#         for mfc in most_freq_cuisine:    

#             row[mfc] = 0

#         row['Other'] = 1

#     else:

#         other_cuisine = 0

#         for mfc in most_freq_cuisine:

#             if mfc in c_list:

#                 row[mfc] = 1

#             else:

#                 other_cuisine += 1

#                 row[mfc] = 0

#         if other_cuisine > 0:

#             row['Other'] = 1

#         else:

#             row['Other'] = 0

#     return row       

# data = data.apply(set_dummy_vars, axis=1)

# удаляем колонку 'cuisine_style_arr'

# data = data.drop(['cuisine_style_arr'], axis=1)



#================================================

# "Reviews"

#================================================

import re, numpy as np

from datetime import datetime

# функция превращает строки с датами отзывов по каждому ресторану в список дат

def review_filter(r):

    date_list = list()

    r = str(r)

    lr = re.findall('\d\d/\d\d/\d\d\d\d', r)

    for r in lr:

        review_date = datetime.strptime(r.strip(), '%m/%d/%Y')

        date_list.append(review_date)

    return date_list    



# получаем серию по всем ресторанам со списком дат отзывов    

review_series = data['Reviews'].apply(lambda r: review_filter(r))



# ищем самый свежий отзыв

dt_max = datetime(1970, 1, 1)

for review_date_list in review_series:

    if len(review_date_list) != 0:

        for dt in review_date_list:

            if dt > dt_max:

                dt_max = dt

                

# заполняем список значениями количества дней между самым старым и самым свежим отзывом по ресторану

days_between = []

for review_date_list in review_series:

    if len(review_date_list) == 0:

        days_between.append(None)

    elif len(review_date_list) == 1:

        days_between.append((dt_max - review_date_list[0]).days) 

    else: # len(review_date_list) > 1

        min_date = np.min(np.array(review_date_list))

        max_date = np.max(np.array(review_date_list))

        days_between.append((max_date - min_date).days)

data["days_between_reviews"] = pd.Series(days_between)



# убираем Nan из колонки "days_between_reviews"

dbr_mean = data.groupby(["City"])['days_between_reviews'].mean()

def dbr_filter(row):

    if  pd.isna(row['days_between_reviews']): 

        return float(round(dbr_mean[row['City']]))

    else:

        return row['days_between_reviews']

# создаём новую колонку без Nan     

dbr_new = data.apply(dbr_filter, axis=1)

# присваиваем новую колонку старой     

data['days_between_reviews'] = dbr_new

# считаем удовлетворённость клиентов по отзывам

import nltk

#nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia



def sentiment_scores(sentence): 

    sid_obj = sia(lexicon_file="/kaggle/input/sf-dst-restaurant-rating/vader_lexicon.txt") 

    sentiment_dict = sid_obj.polarity_scores(sentence) 

    return sentiment_dict['compound']



def sentiment_filter(r):

    if type(r) == float:

        return 0

    rev_str = r.split('], [')[0]

    if len(rev_str) == 2:

        return 0

    else:

        rev_str = rev_str[2:-1].replace(',','').replace("'",'')

        return sentiment_scores(rev_str)



#data['sentiment'] = data['Reviews'].apply(sentiment_filter)

#print(data['sentiment'])    

#================================================

# "City"

#================================================

# создадим колонку указывающую является ли город столицей

capitals = ['Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna', 'Rome', 'Madrid', 'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen', 'Amsterdam', 'Lisbon', 'Prague', 'Oslo', 'Helsinki', 'Ljubljana',

 'Athens', 'Luxembourg', 'Edinburgh']

data['capital'] = data['City'].apply(lambda c: 1 if c in capitals else 0)



# делаем из городов dummy variables

# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True) # pd.get_dummies удаляет колонку City из дата сета
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

    

    # ################### 4. Feature Engineering ####################################################

    # тут ваш код не генерацию новых фитчей

    # ....

    

    

    #================================================

    # "Number of reviews"

    #================================================

    nor_mean = df_output.groupby(["City"])['Number of Reviews'].mean()

    # функция возвращает среднее кол-во отзывов по городу если Nan или текущее значение кол-ва отзывов

    def number_of_reviews_filter(row):

        if  pd.isna(row['Number of Reviews']): 

            return float(round(nor_mean[row['City']]))

        else:

            return row['Number of Reviews']

    # создаём новую колонку с кол-вом отзывов без Nan     

    nor_new = df_output.apply(number_of_reviews_filter, axis=1)

    # присваиваем новую колонку старой     

    df_output['Number of Reviews'] = nor_new



    #================================================

    # "Price Range"

    #================================================

    # все уникальные значения диапазонов цен переводим в цифры

    # больше всего ресторанов в средней ценовой категории, поэтому заполняем пропущенные значение 

    # значением "2"

    df_output['Price Range'].fillna(2, inplace=True)

    def price_range_filter(p):

        if str(p) == '$$ - $$$':

            return 2

        elif str(p) == '$$$$':

            return 3

        elif str(p) == '$':

            return 1

        else:

            return p                                                                                                                              

    # добавляем в дата фрейм новую колонку с числовыми признаками диапазонов цен    

    df_output['price_range'] = df_output['Price Range'].apply(lambda x: price_range_filter(x))

    

    #================================================

    # работаем с колонкой "Cuisine Style"

    #================================================

    # возвращаем список типов кухонь для ресторана

    # для значений Nan принимаем кол-во типов кухонь равное 1

    #cuisine_series = df['Cuisine Style'].fillna(1)

    def cuisine_filter(c):

        c = c[1:-1].replace("'","").split(',')

        c = [i.strip() for i in c]

        return c

    # в новой колонке для кухонь меняем список типов кухонь на их кол-во, для Nan значений возвращаем 1

    cuisine_series = df_output['Cuisine Style'].apply(lambda c: 1 if pd.isna(c) else len(cuisine_filter(c)))

    # добавляем новую колонку с количестом типов кухонь для каждого ресторана

    df_output['cuisine_style'] = cuisine_series 



    # реализуем концепцию "dummy variables"

    #cuisine_series2 = df_output['Cuisine Style'].apply(lambda c: [] if pd.isna(c) else cuisine_filter(c))

    # добавляем новую колонку со значением являющимся строкой со списком типов кухонь для каждого ресторана

    #df_output['cuisine_style_arr'] = cuisine_series2 



    # в рамках заданий по модулю находим список наиболее часто встречающихся (НЧВ) типов кухонь и берём 

    # те, которые встречаются больше 1000 раз

    # most_freq_cuisine = ['Vegetarian Friendly','European','Mediterranean','Italian']

    #most_freq_cuisine = ['Vegetarian Friendly' ,'European' ,'Mediterranean' ,'Italian' ,'Vegan Options' ,'Gluten Free Options' ,'Bar' ,'French' ,'Asian' ,'Pizza' ,'Spanish' ,'Pub' ,'Cafe' ,'Fast Food' ,'British' ,'International' ,'Seafood' ,'Japanese' ,'Central European' ,'American' ,'Sushi' ,'Chinese' ,'Portuguese' ,'Indian' ,'Middle Eastern']

    # добавляем колонки из списка НЧВ со значениями 0

    #for x in most_freq_cuisine:

    #    df_output[x] = 0

    # функция добавляет для каждой строки дата фрейма колонки с именами ранее определённых НЧВ типов кухонь в данном

    # ресторане и со значением 1 если такой тип кухни есть у данного ресторана и 0 если нет

    # также есть колонка с типом кухни "Other" в которой стоит 1 если ни один из типов кухни в ресторане 

    # не попал в список

#     def set_dummy_vars(row):

#         c_list = row['cuisine_style_arr']

#         if len(c_list) == 0:

#             for mfc in most_freq_cuisine:    

#                 row[mfc] = 0

#             row['Other'] = 1

#         else:

#             other_cuisine = 0

#             for mfc in most_freq_cuisine:

#                 if mfc in c_list:

#                     row[mfc] = 1

#                 else:

#                     other_cuisine += 1

#                     row[mfc] = 0

#             if other_cuisine > 0:

#                 row['Other'] = 1

#             else:

#                 row['Other'] = 0

#         return row       

#     df_output = df_output.apply(set_dummy_vars, axis=1)

    # удаляем колонку 'cuisine_style_arr'

#     df_output = df_output.drop(['cuisine_style_arr'], axis=1)



    #================================================

    # "Reviews"

    #================================================

    import re, numpy as np

    from datetime import datetime

    # функция превращает строки с датами отзывов по каждому ресторану в список дат

    def review_filter(r):

        date_list = list()

        r = str(r)

        lr = re.findall('\d\d/\d\d/\d\d\d\d', r)

        for r in lr:

            review_date = datetime.strptime(r.strip(), '%m/%d/%Y')

            date_list.append(review_date)

        return date_list    



    # получаем серию по всем ресторанам со списком дат отзывов    

    review_series = df_output['Reviews'].apply(lambda r: review_filter(r))



    # ищем самый свежий отзыв

    dt_max = datetime(1970, 1, 1)

    for review_date_list in review_series:

        if len(review_date_list) != 0:

            for dt in review_date_list:

                if dt > dt_max:

                    dt_max = dt

                    

    # заполняем список значениями количества дней между самым старым и самым свежим отзывом по ресторану

    days_between = []

    for review_date_list in review_series:

        if len(review_date_list) == 0:

            days_between.append(None)

        elif len(review_date_list) == 1:

            days_between.append((dt_max - review_date_list[0]).days) 

        else: # len(review_date_list) > 1

            min_date = np.min(np.array(review_date_list))

            max_date = np.max(np.array(review_date_list))

            days_between.append((max_date - min_date).days)

    df_output["days_between_reviews"] = pd.Series(days_between)



    # убираем Nan из колонки "days_between_reviews"

    dbr_mean = df_output.groupby(["City"])['days_between_reviews'].mean()

    def dbr_filter(row):

        if  pd.isna(row['days_between_reviews']): 

            return float(round(dbr_mean[row['City']]))

        else:

            return row['days_between_reviews']

    # создаём новую колонку без Nan     

    dbr_new = df_output.apply(dbr_filter, axis=1)

    # присваиваем новую колонку старой     

    df_output['days_between_reviews'] = dbr_new

    

    # считаем удовлетворённость клиентов по отзывам

    import nltk

    #nltk.download('vader_lexicon')

    from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia



    def sentiment_scores(sentence): 

        sid_obj = sia(lexicon_file="/kaggle/input/sf-dst-restaurant-rating/vader_lexicon.txt") 

        sentiment_dict = sid_obj.polarity_scores(sentence) 

        return sentiment_dict['compound']



    def sentiment_filter(r):

        if type(r) == float:

            return 0

        rev_str = r.split('], [')[0]

        if len(rev_str) == 2:

            return 0

        else:

            rev_str = rev_str[2:-1].replace(',','').replace("'",'')

            return sentiment_scores(rev_str)



    df_output['sentiment'] = df_output['Reviews'].apply(sentiment_filter)

    

    

    ## создание признака "количество ресторанов в городе"

    df_output['Rest per City'] = df_output['City'].map(df_output.groupby(['City'])['City'].count().to_dict())            

    ## создание признака "относительный рэнкинг"

    df_output['Relative Ranking'] = df_output['Ranking'] / df_output['Rest per City']                                                           

    

    #================================================

    # "City"

    #================================================

    # создадим колонку указывающую является ли город столицей

    capitals = ['Paris', 'Stockholm', 'London', 'Berlin', 'Bratislava', 'Vienna', 'Rome', 'Madrid', 'Dublin', 'Brussels', 'Warsaw', 'Budapest', 'Copenhagen', 'Amsterdam', 'Lisbon', 'Prague', 'Oslo', 'Helsinki', 'Ljubljana',

     'Athens', 'Luxembourg', 'Edinburgh']

    df_output['capital'] = df_output['City'].apply(lambda c: 1 if c in capitals else 0)

    

    # делаем из городов dummy variables

    # для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na

    df_output = pd.get_dummies(df_output, columns=[ 'City',], dummy_na=True) # pd.get_dummies удаляет колонку City из дата сета                                              

                                                              

                                                                                      

    # ################### 5. Clean #################################################### 

    # убираем признаки которые еще не успели обработать, 

    # модель на признаках с dtypes "object" обучаться не будет, просто выберим их и удалим

    object_columns = [s for s in df_output.columns if df_output[s].dtypes == 'object']

    df_output.drop(object_columns, axis = 1, inplace=True)

    

    return df_output
df_preproc = preproc_data(data)
df_preproc.info()
# Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data.Rating.values            # наш таргет

X = train_data.drop(['Rating'], axis=1)



# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

# выделим 20% данных на валидацию (параметр test_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_SEED)



# проверяем

test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape
# Импортируем необходимые библиотеки:

from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели

from sklearn import metrics # инструменты для оценки точности модели

# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)

model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

# Обучаем модель на тестовом наборе данных

model.fit(X_train, y_train)

#model.fit(X, y)



# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.

# Предсказанные значения записываем в переменную y_pred

y_pred = model.predict(X_test)
# Округляем полученные значения рейтингов!!!!!!!!

def round_d(rec):

    if rec <0.25:

        return 0

    elif 0.25<rec<=0.75:

        return 0.5

    elif 0.75<rec<=1.25:

        return 1

    elif 1.25<rec<=1.75:

        return 1.5

    elif 1.75<rec<=2.25:

        return 2

    elif 2.25<rec<=2.75:

        return 2.5

    elif 2.75<rec<=3.25:

        return 3

    elif 3.25<rec<=3.75:

        return 3.5

    elif 3.75<rec<=4.25:

        return 4

    elif 4.25<rec<=4.75:

        return 4.5

    else:

        return 5

    

for i in range(y_pred.size):

    y_pred[i]=round_d(y_pred[i])
# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))

# в RandomForestRegressor есть возможность вывести самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(15).plot(kind='barh')
# Обучаем модель на всех исходных данных

model.fit(X, y)
# Запускаем на тестовых данных

test_df = df_preproc.query('sample == 0').drop(['sample'], axis=1)

test_df = test_df.drop(['Rating'], axis=1)

predict_submission = model.predict(test_df)
# Округление результата

sample_submission['Rating'] = predict_submission

sample_submission['Rating'] = sample_submission['Rating'].apply(round_d)
# Выгрузка в файл

sample_submission.to_csv('submission2.csv', index=False)