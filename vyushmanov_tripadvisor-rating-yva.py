# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import re 

import plotly

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

from scipy.stats import ttest_ind

from itertools import combinations

from collections import Counter



from sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder



from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

%matplotlib inline



# Загружаем специальный удобный инструмент для разделения датасета:

from sklearn.model_selection import train_test_split



# Загружаем набор собственных функций

import myfunction as mf



# Сервисные функции

pd.set_option('display.max_rows', 50) # выведем больше строк

pd.set_option('display.max_columns', 100) # выведем больше колонок

import warnings; warnings.simplefilter('ignore') #  отключение вывода предупреждающих сообщений



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# фиксируйте RANDOM_SEED и версию пакетов, чтобы эксперименты были воспроизводимы:

RANDOM_SEED = 42

!pip freeze > requirements.txt
# Читаем датасеты

DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'

df_train = pd.read_csv(DATA_DIR + '/main_task.csv')

df_test = pd.read_csv(DATA_DIR + 'kaggle_task.csv')

sample_submission = pd.read_csv(DATA_DIR + '/sample_submission.csv')



# ВАЖНО! для корректной обработки признаков объединяем трейн и тест в один датасет

df_train['sample'] = 1 # помечаем где у нас трейн

df_test['sample'] = 0 # помечаем где у нас тест

df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями



data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем



# Выводим сводку о содержании датасета

brief_columns = ['Признак', '#', 'тип данных', '% заполнения', '# пропусков', '# уникальных', 'диапазон значений / примеры']

mf.brief_summary(df_train, brief_columns)

data = mf.drop_dublle(data, ['ID_TA', 'sample'])
# вывод структуры уникальных и сетевых ресторанов

# Добавление столбца с количеством ресторанов в сети

    

mf.view_count_in_chain(data)

data = mf.calc_count_in_chain(data) 
# строка преобразована в список

data = mf.string_to_list_distribution(data, 'City')

# выведем распределение ресторанов по городам, сохраним общую численность ресторанов в городе в признаке restorans_in_city 

data = mf.view_horiz_bar_n_table(data, 'City', 'restaurants_in_city') 
# выделены часто встречающиеся и редкие значения City



# из внешних источников датасет дополнен сведениями city_is_the_capital, population_city, country

data = mf.city_expansion_features(data)
# заполним пропуски значением 'Empty'

# создадим признак empty_cuisine_style, в котором '1' обозначим пустые значения Cuisine Style, для остальных применим '0'



# закодируем значения в переменной до их преобразования - признак code_cuisine_style



# преобразуем строку с перечислением в список - признак list_cuisine_style



# выделим редко встречающиеся кухни, обозначим unique_style. Считаем, что редкие кухни - последние 23 (встречаются реже 20 раз)



# посчитаем количество заявленных кухонь - признак count_cuisine_style. В случае отсутствия информации

# о количестве кухонь используем медианное значение

# выведем гистограммы и боксплоты для линейных и логарифмических значений признака



# добавим признак, рассчитанный как натуральный логарифм количества кухонь

    

# выделим редко встречающиеся кухни. Последнее количество unique_border - уникальные кухни.

data = mf.string_to_list_distribution(data, 'Cuisine Style')

data = mf.cuisine_distribution(data, 'list_cuisine_style')

data = mf.rife_rare_distribution(data, 'list_cuisine_style',.2,.03)

data = mf.localisation_cuisine_country(data) # идею позаимствовал у (с)Rezinko Mikhail

data = mf.view_histogrm_n_boxplot(data, 'count_cuisine_style')
# определение наиболее частых и наиболее редких вариантов признака

data = mf.rife_rare_distribution(data, 'list_cuisine_style', .25, .02)
# посмотрим на распределение признака в 10 крупнейших городах:

mf.view_attribute_based_distribution(data, 'Ranking', 'City', 10)
# произведем сквозное ранжирование равномерно распределив ранг ресторанов в рамках города

# значение нового признака сохраним в total_ranking



# в качестве альтернативного способа преобразования признака используем стандартизацию и сохраняем результат в standard_ranking



data = mf.ranking_distribution(data)

data = mf.add_ranking_distribution(data)



mf.view_attribute_based_distribution(data, 'total_ranking', 'City', 10)

mf.view_attribute_based_distribution(data, 'standard_ranking', 'City', 10)

mf.view_attribute_based_distribution(data, 'norm_ranking_on_population', 'City', 10)

mf.view_attribute_based_distribution(data, 'norm_ranking_on_tourists', 'City', 10)

mf.view_attribute_based_distribution(data, 'norm_ranking_on_max_rank', 'City', 10)

mf.view_attribute_based_distribution(data, 'norm_ranking_on_restaurant', 'City', 10)
# создадим признак empty_price_range, в котором '1' обозначим пустые значения Price Range, для остальных применим '0'



# перекодируем признак по словарю {'$':1, '$$ - $$$':2, '$$$$':3}



# заполним пропущенные значения модой, т.е. 2



data = mf.price_distribution(data, 2)

mf.view_price_info(data)

data = mf.mean_price_in_city(data)

mf.view_mean_price(data)
# приведем наименование столбца к стандартному виду

data.rename(columns={'Number of Reviews': 'number_of_reviews'}, inplace=True)



# зафиксируем строки с пустыми значениями

data['empty_number_of_reviews'] = pd.isna(data['number_of_reviews']).astype('float64')



# вывод гистограмм и таблицы

data = mf.view_histogrm_n_boxplot(data, 'number_of_reviews')

# добавлен признак, рассчитанный как натуральный логарифм номера ревю
# исследуем вляние различных признаков на распределение log_number_of_reviews:

mf.view_attribute_based_distribution(data, 'log_number_of_reviews', 'City', 5)

mf.view_attribute_based_distribution(data, 'log_number_of_reviews', 'Price Range', 4)

mf.view_attribute_based_distribution(data, 'log_number_of_reviews', 'count_cuisine_style', 8)



# в целях детального изучения распределения вновь созданного признака log_number_of_reviews выведем подробную гистограмму и расчет выбросов 

mf.view_histogram_n_outliers(data, 'number_of_reviews', 'log', 160)
# выбросы зафиксированы в 10 наблюдениях, удалим их, предварительно сохранив информацию о них

data['outliers_number_of_reviews'] = pd.DataFrame(data['log_number_of_reviews'] > 8.56).astype('float64')

data.loc[data['log_number_of_reviews'] > 8.56, 'number_of_reviews']=None
# строковая переменная преобразована в словари используемых в отзывах слов, которые сохранены в признак review_words_list

# строки с пустыми отзывами отмечены 1 в признаке empty_review



# произведен подсчет слов, имеющих позитивную и негативную окраску. Количество таких слов сохранено в признаки count_pos_words

# и count_neg_words соответственно. В случае присутствия в отзывах слова 'not' из количества негативных слов вычитается 1.



data = mf.review_text_distribution(data)
# выделим из Reviews информацию о датах размещения отзывов

# рассчитаем количество дней, прошедших между публикациями отзывов

# определим для каждого ресторана, положение самого свежего отзыва на временном луче, нулевая отметка которого соответствует 

# дню выхода самой первой публикации 

data = mf.data_review_distribution(data)



# произведем сравнение распределений количества дней после новейшей публикации и периодом между публикациями

fig = px.scatter(data[data['review_date_count'] == 2], x="review_date_min", y="review_date_delta",

                marginal_x='histogram', marginal_y='histogram',

                trendline='ols', trendline_color_override='darkblue')

fig.show()
data = mf.view_histogrm_n_boxplot(data, 'review_date_min')

data = mf.view_histogrm_n_boxplot(data, 'review_date_delta')
# исследуем распределение признаков review_date_olded и review_date_delta    

mf.view_histogram_n_outliers(data, 'review_date_min', 'all', 200)

mf.view_histogram_n_outliers(data, 'review_date_delta', 'lin', 200)
# бесконечности и пропуски заменены

data.replace(np.inf, 1, inplace=True)

data.replace(-np.inf, 0, inplace=True)

data = data.fillna(0)



mf.show_heatmap(data[data['sample'] == 1].drop(['sample'], axis=1))
# Из признаков, полученных вокруг Ranking оставляем standard_ranking как имеющий наибольшую корреляцию

data.drop(['total_ranking'], axis=1, inplace=True, errors='ignore')



# Из пар линейного значения и логарифма выбираем имеющие наибольшую корреляцию, остальные удаляем

data.drop(['log_count_cuisine_style', 'log_number_of_reviews', 'log_review_date_min', 'log_review_date_delta'], axis=1, inplace=True, errors='ignore')



# Признак outliers_number_of_reviews не имеет корреляции с целевой переменной - удаляем

data.drop(['outliers_number_of_reviews'], axis=1, inplace=True, errors='ignore')



# произведение модулей признаков empty_review и review_date_count, первичные признаки удалены

#data['empty_review_date_count'] = data['empty_review'] + data['review_date_count']

#data.drop(['empty_review'], axis=1, inplace=True, errors='ignore')



mf.show_heatmap(data[data['sample'] == 1].drop(['sample'], axis=1))
#list_for_pca = ['empty_review', 'review_date_count', 'empty_number_of_reviews', 'count_review_words', 'code_review_words', 'number_of_reviews']

#data = mf.pca_distribution(data, list_for_pca, 'pca_review')



#list_for_pca = ['code_city', 'restaurants_in_city', 'population_city', 'code_country', 'count_city_tourists', 'count_in_chain', 'city_is_the_capital']

#data = mf.pca_distribution(data, list_for_pca, 'pca_city', [0,1,1,0,0,1,0])



#list_for_pca = ['code_cuisine_style', 'empty_cuisine_style', 'rare_cuisine_style', 'local_cuisine']

#data = mf.pca_distribution(data, list_for_pca, 'pca_cuisine')



list_for_pca = ['Ranking', 'count_in_chain', 'standard_ranking', 'restaurants_in_city', 'population_city']

data = mf.pca_distribution(data, list_for_pca, 'pca_ranking', [1,0,1,0,0])

data.sample(2)
# количество ресторанов на 1000 человек населения города, первичные признаки удалены

#data['restaurant_on_population'] = data['restaurants_in_city'] / data['population_city'] / 1000



# число туристов на одного жителя города

#data['tourists_on_population'] = data['count_city_tourists'] / data['population_city']



# частное от деления Ranking на количество ресторанов в городе (restaurants_in_city)

#data['ranking_on_count_restaurant'] = data['Ranking'] / data['restaurants_in_city']



# количество отзывов на 10 000 жителей

#data['review_on_population'] = data['number_of_reviews'] / data['population_city'] / 10000



# количество отзывов на 100 000 туристов

#data['review_on_tourists'] = data['number_of_reviews'] / data['population_city'] / 100000



# отношение количества туристов к жителям

#data['tourists_on_population'] = data['count_city_tourists'] / data['population_city']



#data.drop(['city_is_the_capital', 'code_cuisine_style', 'local_cuisine', 'empty_number_of_reviews',

#          'review_date_count', 'empty_review_date_count', 'count_neg_words'], axis=1, inplace=True, errors='ignore')



#mf.show_heatmap(data[data['sample'] == 1].drop(['sample'], axis=1))

data = mf.prep_dummies(data, 'name_chain', 1, 'ch_')#

#data = mf.prep_dummies(data, 'list_city')

#data = mf.prep_dummies(data, 'list_country')

#data = mf.prep_dummies(data, 'list_cuisine_style', .85, 'cs_')

#positive_words, negative_words = mf.read_positive_words()

#data = mf.prep_dummies(data, 'list_review_words', .3, 'w_', positive_words) # обработаны только слова из "позитивного списка"
data = mf.read_dataframes() # чтение файлов и формирование исходного датасета

data = mf.drop_dublle(data, ['ID_TA', 'sample']) # удаление дублей

data.rename(columns={'Number of Reviews': 'number_of_reviews'}, inplace=True)

#data['empty_number_of_reviews'] = pd.isna(data['number_of_reviews']).astype('float64')

data = mf.string_to_list_distribution(data, 'City') # строка преобразована в список 'list_city'



data = mf.ranking_distribution(data)

data = mf.city_expansion_features(data)

#data = mf.add_ranking_distribution(data)

#list_for_pca = ['total_ranking', 'standard_ranking', 'norm_ranking_on_max_rank', 

#                'norm_ranking_on_restaurant', 'norm_ranking_on_population', 'norm_ranking_on_tourists']

#data = mf.pca_distribution(data, list_for_pca, 'pca_norm_ranking', [1,1,1,1,1,1])



#data['log_number_of_reviews'] = np.log1p(data['number_of_reviews'])



data = mf.calc_count_in_chain(data)

data = mf.prep_dummies(data, 'name_chain', 1, 'ch_') # преобразование признаков в dummy-переменные

data = mf.prep_dummies(data, 'list_city') # преобразование признаков в dummy-переменные

data = mf.review_text_distribution(data)

positive_words, negative_words = mf.read_positive_words()

data = mf.prep_dummies(data, 'list_review_words', .3, 'w_', positive_words) # обработаны только слова из "позитивного списка"



#data = mf.price_distribution(data, 2)

#data = mf.mean_price_in_city(data)



data = mf.string_to_list_distribution(data, 'Cuisine Style')

data = mf.cuisine_distribution(data, 'list_cuisine_style')

#data = mf.rife_rare_distribution(data, 'list_cuisine_style',.3,.01)

#data = mf.city_expansion_features(data)

#data = mf.localisation_cuisine_country(data)

#data = mf.prep_dummies(data, 'list_cuisine_style', .85, 'cs_')



data = mf.data_review_distribution(data)



#data['outliers_date_min'] = pd.DataFrame(data['review_date_min'] > 1122.5).astype('float64')

#data.loc[data['review_date_min'] > 1122.5, 'number_of_reviews']=None

#data['outliers_date_delta'] = pd.DataFrame(data['review_date_delta'] > 355.5).astype('float64')

#data.loc[data['review_date_delta'] > 355.5, 'number_of_reviews']=None



#data['empty_number_of_reviews'] = pd.isna(data['number_of_reviews']).astype('float64')



#data = mf.ranking_distribution(data)

list_for_pca = ['Ranking', 'standard_ranking', 'total_ranking']

data = mf.pca_distribution(data, list_for_pca, 'pca_ranking',[1,1,1])

data['ranking_power'] = data['Ranking']* data['Ranking']

data['ranking_copy'] = data['Ranking']



#data['log_number_of_reviews'] = np.log1p(data['number_of_reviews'])

#data['log_review_date_min'] = np.log1p(data['review_date_min'])

#data['log_review_date_delta'] = np.log1p(data['review_date_delta'])



#data['outliers_number_of_reviews'] = pd.DataFrame(data['log_number_of_reviews'] > 8.56).astype('float64')

#data.loc[data['log_number_of_reviews'] > 8.56, 'number_of_reviews']=None



data.drop(['count_in_chain', 'code_review_words',

           'count_pos_words', 'count_neg_words'], axis=1, inplace=True, errors='ignore')

data.drop(['count_city_tourists'], axis=1, inplace=True, errors='ignore')
data = data.fillna(0)

# произведена мин-макс стандартизация (за исключением списка столбцов)

data = mf.normalisation(data, MinMaxScaler(), ['Rating', 'sample'])



df_preproc = mf.delete_string_sign(data)
display(df_preproc.sample(2))

display(data.describe().head(1))
# Теперь выделим тестовую часть



train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)

test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)



y = train_data[['Rating']]           # наш таргет

X = train_data.drop(['Rating'], axis=1)



# Воспользуемся специальной функцией train_test_split для разбивки тестовых данных

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

y_pred = np.array([5.0 if x>5 else x for x in list(np.round(y_pred * 2) / 2)])



# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются

# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

MAE = metrics.mean_absolute_error(y_test, y_pred)

try: title = 'MAE: '+str(MAE)+' <-- '+str(old_MAE)

except: title = 'MAE: '+str(MAE)

old_MAE = MAE



layout =go.Layout(

              autosize=False,

              width=1000,

              height=500)

fig = go.Figure(layout = layout)

fig.add_trace(go.Bar(x = model.feature_importances_, y = X.columns, orientation='h')), 

fig.update_layout(title = title, title_x = 0.5,

                  yaxis={'categoryorder':'total descending'},

                  margin = dict(l=200, r=100, t=50, b=0), showlegend=False)

fig.update_yaxes(range=(-.5, 20.5))

fig.show()
'''

# блок тестирования оптимального набора признаков

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

list_importance_sign = list(feat_importances.nlargest(len(train_data.columns)-1).index)

min_MAE = round(MAE,3)

print(f"min_MAE = {min_MAE}")

remove_list = []

log = []

delta =0.001

for i in range(0,len(list_importance_sign),1):

    col = list_importance_sign[i]

    print(f"{i}.{col}")

#     ###

    train_data = data.query('sample == 1').drop(['sample'], axis=1)

    test_data = data.query('sample == 0').drop(['sample'], axis=1)



    y = train_data.Rating.values            # наш таргет

    X = train_data.drop(['Rating']+[col], axis=1)



    # Воспользуемся специальной функцие train_test_split для разбивки тестовых данных

    # выделим 20% данных на валидацию (параметр test_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    print(test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape)



    model.fit(X_train, y_train)



    y_pred = model.predict(X_test)



    y_pred = np.array([5.0 if x>5 else x for x in list(np.round(y_pred * 2) / 2)])

    temp_MAE = metrics.mean_absolute_error(y_test, y_pred)

#     ###

    print(temp_MAE)

    log.append([col, temp_MAE])

    if round(temp_MAE,3) <= min_MAE-delta:

        remove_list.append(col)

        print(f"удаляем:= {col}")

    else:

        print(f"не удаляем:= {col}")

print(f"i={i}")

print(f"remove_list: {remove_list}")

print(f"log_list: {log}")

'''
test_data = test_data.drop(['Rating'], axis=1)

predict_submission = model.predict(test_data)

predict_submission = np.array([5.0 if x>5 else x for x in list(np.round(predict_submission * 2) / 2)])



sample_submission['Rating'] = predict_submission

sample_submission.to_csv('submission.csv', index=False)

sample_submission.head(10)