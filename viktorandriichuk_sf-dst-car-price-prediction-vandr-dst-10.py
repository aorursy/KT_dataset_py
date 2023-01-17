import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

import datetime
import sys
import re

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, BaggingRegressor, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import ExtraTreeRegressor
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from sklearn.base import clone

from tqdm import tqdm # обеспечивает простой индикатор выполнения для операций pandas
%matplotlib inline
# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
!pip freeze > requirements.txt
DIR_TRAIN  = '../input/train-data-autoru/' # подключил к ноутбуку свой внешний датасет
DIR_TEST   = '../input/sf-dst-car-price/'
!ls ../input/
test = pd.read_csv(DIR_TEST+'test.csv')
sample_submission = pd.read_csv(DIR_TEST+'sample_submission.csv')
test.sample(5)
test.info()
test.columns = ([
    'body_type',
    'brand',
    'color',
    'fuel_type',
    'model_date_begin',
    'name',
    'number_of_doors',
    'production_date',
    'vehicle_configuration',
    'vehicle_transmission',
    'engine_displacement',
    'engine_power',
    'description',
    'mileage',
    'equipment',
    'gear_type',
    'steering_wheel',
    'not_damage',
    'owners',
    'technical_passport',
    'custom_clear',
    'ownership',
    'id',
])
test.info()
def col_info(col):
    print('Количество пропусков: {}'.format(col.isnull().sum()))
    print('{},'.format(col.describe()))
    print('Распределение:\n{},'.format(col.value_counts()))
    col.hist()
def outliers_iqr_short(ys):
    # Определяет номера значений с отклонением больше, чем iqr
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - 1.5*iqr
    upper_bound = quartile_3 + 1.5*iqr
    return np.where((ys > upper_bound) | (ys < lower_bound))[0]

def outliers_iqr_long(ys): # Передаем на вход признак-столбец датафрейма
    # Находим необходимые параметры
    median = ys.median()
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print('Медиана: {},'.format(median),'25-й перцентиль: {},'.format(quartile_1), '75-й перцентиль: {},'.format(quartile_3)
      , "IQR: {}, ".format(iqr),"Границы выбросов: [{}, {}].".format(lower_bound,upper_bound))
    # Отбрасываем значения, лежещие за пределами границ, рисуем график
#     ys.loc[ys.between(lower_bound, upper_bound)].hist(bins = int(upper_bound-lower_bound), range = (lower_bound, upper_bound), label = 'IQR')
#     plt.legend();
    # На выход отдаем отфильтрованные значения
    
    first_rec = ys.mask((ys > upper_bound) | (ys < lower_bound))
    second_rec = np.where((ys > upper_bound) | (ys < lower_bound))[0]
    
    return first_rec
# Функция очистки от выбросов
def delete_outliers_iqr(df, column):
    # Считаем первый и третий квартили
    first_quartile = df[column].describe()['25%']
    third_quartile = df[column].describe()['75%']

    # IQR (Межквартильный размах)
    iqr = third_quartile - first_quartile

    print(first_quartile, third_quartile, iqr)

    # Удаляем то, что выпадает за границы IQR
    #     df_tmp = df.copy()
    df = df[(df[column] > (first_quartile - 3 * iqr)) &
                (df[column] < (third_quartile + 3 * iqr))]

    df[column].hist()
    df[column].describe()

    df = df.loc[df[column].between(first_quartile - 1.5*iqr, third_quartile + 1.5*iqr)]
    df.info()
col_info(test.model_date_begin)
test['model_date_begin'] = pd.to_datetime(test['model_date_begin'], format='%Y').dt.year
col_info(test.model_date_begin)
test.model_date_begin.min()
test.model_date_begin.max()
test['model_age'] = datetime.datetime.now().year - test['model_date_begin']
col_info(test.model_age)
test[test['model_age'] > 40]
test['model_age'] = test['model_age'].apply(lambda x: 20 if x > 19 else x)
test['model_age'] = test['model_age'].astype('str')
col_info(test.production_date)
test['production_date'] = pd.to_datetime(test['production_date'], format='%Y').dt.year
test['car_age'] = datetime.datetime.now().year - test['production_date']
col_info(test.car_age)
test.sample(2)
test['mileage_per_year'] = test['mileage'] / test['car_age']
def cat_mileage_per_year(x):
    if x < 10000: x = 1
    elif 10000 <= x < 20000: x = 2
    elif 20000 <= x < 30000: x = 3
    elif 30000 <= x: x = 4
    return x


test['mileage_per_year'] = test['mileage_per_year'].apply(lambda x: cat_mileage_per_year(x))
test['mileage_per_year'] = test['mileage_per_year'].astype('str')
col_info(test.mileage_per_year)
def cat_age(x):
    if x == 1 : x = 1 # типа, если автомобилю 1 год, то добавим его в категорию 1
    elif 2 <= x < 4: x = 2 # типа, если автомобилю от 2 до 3 лет включительно, то добавим его в категорию 2 и тд.
    elif 4 <= x < 6: x = 3
    elif 6 <= x < 10: x = 4
    elif 10 <= x < 16: x = 5
    elif 16 <= x < 20: x = 6
    elif 20 <= x: x = 7
    return x 


test['car_age'] = test['car_age'].apply(lambda x: cat_age(x))

test['car_age'] = test['car_age'].astype('str')
col_info(test.number_of_doors)
test['number_of_doors'] = test['number_of_doors'].astype(int)
col_info(test.number_of_doors)
col_info(test.custom_clear)
test['custom_clear'] = test['custom_clear'].apply(lambda x: "1" if x == "Растаможен" else "0").astype(int)
test['custom_clear'].value_counts()
test.sample(2)
test = test.drop('custom_clear', axis=1)
test = test.drop('id', axis=1)
test.sample(2)
col_info(test.body_type)
test[['body_type','number_of_doors']]
test.groupby('body_type').agg({'number_of_doors': 'value_counts'})
# Новый датафрейм с разбитыми данными из нужно мне колонки - сделаю так, чтобы это было решением не частным, а для всех случаев
body_type_tmp = test['body_type'].str.split(" ", n = 1, expand = True) 
  
# Возьму теперь первую часть из нового созданного датафрейма, там будет нужное мне название типа кузова, и создам колонку в полном датафрейме
test['body_type']= body_type_tmp[0]
test.sample(2)
# словарь для кодировки
dic_body_type = {
    'внедорожник': 'allroad',
    'кабриолет': 'cabriolet',
    'компактвэн': 'compact_van',
    'купе': 'coupe',
    'лифтбек': 'liftback',
    'родстер': 'roadster',
    'седан': 'sedan',
    'универсал': 'station_wagon',
    'хэтчбек': 'hatchback'
}

test['body_type'] = test['body_type'].map(dic_body_type)
test.groupby('body_type').agg({'number_of_doors': 'value_counts'})
# test = pd.get_dummies(test, columns=[ 'body_type',], dummy_na=True)
test.sample(5)
col_info(test.color)
dict_color = {
    'чёрный': 'black',
    'белый': 'white',
    'синий': 'blue',
    'коричневый': 'brown',
    'красный': 'red',
    'серый': 'grey',
    'бежевый': 'beige',
    'серебристый': 'silver',
    'золотистый': 'gold',
    'оранжевый': 'orange',
    'голубой': 'mid_blue',
    'пурпурный': 'purple',
    'жёлтый': 'yellow',
    'зелёный': 'green',
    'фиолетовый': 'violet'
}

test['color'] = test['color'].map(dict_color)
test.sample(5)
col_info(test.fuel_type)
dict_fuel_type = {
    'дизель': 'diesel',
    'бензин': 'gasoline',
    'гибрид': 'hybrid',
    'электро': 'electro'
}

test['fuel_type'] = test['fuel_type'].map(dict_fuel_type)
col_info(test.vehicle_transmission)
dict_vehicle_transmission = {
    'автоматическая': 'AT',
    'механическая': 'MT',
    'роботизированная': 'AMT'
}

test['vehicle_transmission'] = test['vehicle_transmission'].map(dict_vehicle_transmission)
test.info()
test = test.drop('brand', axis=1)
col_info(test.engine_displacement)
test['engine_displacement'] = test.engine_displacement.apply(lambda x: x.replace('LTR', '').replace(' ', ''))
test[test['engine_displacement'] == 'undefined']
test[test['engine_displacement'] == 'undefined']['equipment'].values
test[test['engine_displacement'] == 'undefined']['vehicle_configuration'].values
test.loc[test['engine_displacement'] == 'undefined', 'engine_displacement'] = 0
test['engine_displacement'] = test['engine_displacement'].astype(float)
col_info(test.vehicle_configuration)
# test = test.drop('vehicle_configuration', axis=1)
col_info(test.engine_power)
test.groupby('name').agg({'engine_power': 'value_counts'})
# Новый датафрейм с разбитыми данными из нужно мне колонки - сделаю так, чтобы это было решением не частным, а для всех случаев
engine_power_tmp = test['engine_power'].str.split(" ", expand = True) 
engine_power_tmp[1].value_counts()
# Возьму данные из первой колонки и подкорректирую признак с мощностью двигателя
test['engine_power']= engine_power_tmp[0]

# ну и приведу колонку сразу к числовому типу
test['engine_power'] = test['engine_power'].astype(int)
cars_name_for_search_model = test[['name', 'model_date_begin', 'body_type', 'engine_displacement', 'fuel_type', 'vehicle_transmission', 'engine_power']]
cars_name_for_search_model.to_excel("cars_name_for_search_model.xlsx",
             sheet_name='cars_name_for_search_model')  

cars_name_for_search_model
cars_name_for_search_model2 = name_tmp
cars_name_for_search_model2.to_excel("cars_name_for_search_model2.xlsx",
             sheet_name='cars_name_for_search_model')  

cars_name_for_search_model2
# Новый датафрейм с разбитыми данными из нужной мне колонки - сделаю так, чтобы это было решением не частным, а для всех случаев
name_tmp = test['name'].str.split(" ", n = 2, expand = True) 
# name_tmp.sample(10)
# name_tmp[1].replace(to_replace='[0-9].', value='', inplace=True, regex=True)
name_tmp.sample(10)
name_tmp
# попробуем склеить первые две колонки, чтобы получить название модели автомобиля, но при этом будем соблюдать некоторые условия:
# во второй должно содержаться xDrive, Competition, sDrive, Package, тогда это будет добавлено к первой колонке
# в первой колонке может быть все, что угодно
# конъюнкция двух признаков
def make_conj(data, feature1, feature2):
    data['tmp_1'] = np.where(~data[feature2].str.contains('xDrive|Competition|sDrive|Package'),
                                 '', data[feature2])
    
    data['tmp_0'] = data[feature1]
    
    data['tmp'] = data['tmp_0'] + ' ' + data['tmp_1']
                    
    return (data)

# выполним корректировку названия
make_conj(name_tmp, 0, 1)
name_tmp.sample(10)
name_tmp.drop([0, 1, 2], axis=1, inplace=True,)
test['name'] = name_tmp['tmp']
test['name'] = test['name'].apply(lambda x: x.replace('xDrive20d', '20d xDrive'))
test['name'] = test['name'].apply(lambda x: x.replace('sDrive18i', '18i sDrive'))
test['name'] = test['name'].apply(lambda x: x.replace('sDrive20i', '20i sDrive'))
test['name'] = test['name'].apply(lambda x: x.replace('xDrive18d', '18d xDrive'))
test['name'] = test['name'].apply(lambda x: x.replace('xDrive20', '20 xDrive'))
test['name'] = test['name'].apply(lambda x: x.replace('20 xDrivei', '20i xDrive'))
test['name'] = test['name'].str.strip()
col_info(test.name)
print(test['description'][2])
print(len(test['description'][2]))
test['description_score'] = test['description'].apply(lambda x: len(x) / 100).astype(float)
test = test.drop('description', axis=1)
test = test.drop('description_score', axis=1)
test.sample(10)
col_info(test.mileage)
test[test['mileage'] < 10000]
test['mileage'] = test['mileage'].astype(int)
test['exp_year'] = test.mileage.apply(lambda x: x//17000 if x<=400000 else 400000//17000+1)
def cat_mileage(x):
    if x < 25000: x = 1
    elif 25000 <= x < 50000: x = 2
    elif 50000 <= x < 75000: x = 3
    elif 75000 <= x < 100000: x = 4
    elif 100000 <= x < 125000: x = 5
    elif 125000 <= x < 150000: x = 6
    elif 150000 <= x < 175000: x = 7
    elif 175000 <= x < 200000: x = 8
    elif 200000 <= x < 225000: x = 9
    elif 225000 <= x < 250000: x = 10
    elif 250000 <= x < 275000: x = 11
    elif 275000 <= x < 300000: x = 12
    elif 300000 <= x < 325000: x = 13
    elif 325000 <= x < 350000: x = 14
    elif 350000 <= x < 375000: x = 15
    elif 375000 <= x < 400000: x = 16
    elif 400000 <= x: x = 17
    return x    


test['mileage'] = test['mileage'].apply(lambda x: cat_mileage(x))

test['mileage'] = test['mileage'].astype('str')
col_info(test.equipment)
def get_test_features(equipment):
    # Создаем пустой список, в который будут добавляться все фичи
    all_features = []
    for data in equipment:
        # Находим все слова между кавычками
        features=re.findall(r'\"(.+?)\"',data)
        # Добавляем в общий список
        all_features.extend(features)
    # Удаляем дубликаты
    all_features = list(dict.fromkeys(all_features))
    return all_features
test_features = get_test_features(test.equipment)
# Удаляем лишние записи
for bad_feature in ['name','Безопасность','values','Комфорт','Мультимедиа','Обзор','Салон','Защита от угона','Элементы экстерьера']:
    test_features.remove(bad_feature)    
print('Всего уникальных фич:', len(test_features))
print(test_features)
def get_features_test(equipment): 
    features=re.findall(r'\"(.+?)\"',equipment)  
    return features
test['equipment'] = test['equipment'].apply(lambda x: get_features_test(x))
test.sample(5)
col_info(test.gear_type)
dict_gear_type = {
    'задний': 'rear',
    'полный': '4w',
    'передний': 'front'
}

test['gear_type'] = test['gear_type'].map(dict_gear_type)
col_info(test.steering_wheel)
dict_steering_wheel = {
    'Левый': 'left',
    'Правый': 'right'
}

test['steering_wheel'] = test['steering_wheel'].map(dict_steering_wheel)
col_info(test.not_damage)
dict_not_damage = {
    'Не требует ремонта': '1'
}

test['not_damage'] = test['not_damage'].map(dict_not_damage)

# ну и приведу колонку сразу к числовому типу
test['not_damage'] = test['not_damage'].astype(int)
col_info(test.owners)
# через словарь почему-то не захотело корректно менять. пришлочь через вхождение.
test['owners'] = np.where(test['owners'].str.contains('1'),
                                 '1', test['owners'])
test['owners'] = np.where(test['owners'].str.contains('2'),
                                 '2', test['owners'])
test['owners'] = np.where(test['owners'].str.contains('3'),
                                 '3', test['owners'])

# ну и приведу колонку сразу к числовому типу
test['owners'] = test['owners'].astype(int)
test['owners']
test.info()
col_info(test.technical_passport)
dict_technical_passport = {
    'Оригинал': 'original',
    'Дубликат': 'duplicate'
}

test['technical_passport'] = test['technical_passport'].map(dict_technical_passport)
# Новый датафрейм с разбитыми данными из нужно мне колонки - сделаю так, чтобы это было решением не частным, а для всех случаев
ownership_tmp = test['ownership'].str.split(" ", n = 4, expand = True) 
  
# Возьму теперь первую часть из нового созданного датафрейма, там будет нужное мне название типа кузова, и создам колонку в полном датафрейме
# test['body_type']= body_type_tmp[0]
ownership_tmp
ownership_tmp
ownership_tmp = ownership_tmp.fillna(0)                             
ownership_tmp.drop([1, 2, 4], axis=1, inplace=True,)
ownership_tmp[0] = ownership_tmp[0].apply(lambda x: int(x) * 12).astype(int)
# ну и приведу колонку 3 (месяц) сразу к числовому типу
ownership_tmp[3] = ownership_tmp[3].astype(int)
test['ownership'] = ownership_tmp[0] + ownership_tmp[3]
test = test.drop('ownership', axis=1)
test = test.drop('vehicle_configuration', axis=1)
test.info()
def test_series(df):
    conditions = [
        (df['name'] == '116') # 1ER
        | (df['name'] == '116i xDrive')
        | (df['name'] == '116i')
        | (df['name'] == '118d xDrive')
        | (df['name'] == '118d')
        | (df['name'] == '118i xDrive')
        | (df['name'] == '118i')
        | (df['name'] == '120d xDrive')
        | (df['name'] == '120d')
        | (df['name'] == '120i xDrive')
        | (df['name'] == '120i')
        | (df['name'] == '125i xDrive')
        | (df['name'] == '125i')
        | (df['name'] == '130i xDrive')
        | (df['name'] == '130i')
        | (df['name'] == '135i xDrive')
        | (df['name'] == '135i')
        | (df['name'] == 'M135i xDrive')
        | (df['name'] == 'M135i')
        | (df['name'] == 'Active'),
        (df['name'] == '1.5') # 2ER
        | (df['name'] == '218i')
        | (df['name'] == '218i xDrive')
        | (df['name'] == '220i')
        | (df['name'] == '220i xDrive')
        | (df['name'] == 'M235i')
        | (df['name'] == 'M235i xDrive'),
        (df['name'] == '316')  # 3ER
        | (df['name'] == '316i')
        | (df['name'] == '316i xDrive')
        | (df['name'] == '318d')
        | (df['name'] == '318d xDrive')
        | (df['name'] == '318i')
        | (df['name'] == '318i xDrive')
        | (df['name'] == '320d')
        | (df['name'] == '320d xDrive')
        | (df['name'] == '320i')
        | (df['name'] == '320i xDrive')
        | (df['name'] == '323i')
        | (df['name'] == '323i xDrive')
        | (df['name'] == '325i')
        | (df['name'] == '325i xDrive')
        | (df['name'] == '325xi')
        | (df['name'] == '325xi xDrive')
        | (df['name'] == '328i')
        | (df['name'] == '328i xDrive')
        | (df['name'] == '330d')
        | (df['name'] == '330d xDrive')
        | (df['name'] == '330i')
        | (df['name'] == '330i xDrive')
        | (df['name'] == '330xd')
        | (df['name'] == '330xd xDrive')
        | (df['name'] == '330xi')
        | (df['name'] == '330xi xDrive')
        | (df['name'] == '335i')
        | (df['name'] == '335i xDrive')
        | (df['name'] == '335xi')
        | (df['name'] == '335xi xDrive')
        | (df['name'] == '340i')
        | (df['name'] == '340i xDrive')
        | (df['name'] == 'M340i')
        | (df['name'] == 'M340i xDrive'),
        (df['name'] == '316')  # 4
        | (df['name'] == '420d')
        | (df['name'] == '420d xDrive')
        | (df['name'] == '420i')
        | (df['name'] == '420i xDrive')
        | (df['name'] == '428i')
        | (df['name'] == '428i xDrive')
        | (df['name'] == '430i')
        | (df['name'] == '430i xDrive')
        | (df['name'] == '440i'),
        (df['name'] == '316')  # 5ER
        | (df['name'] == '518')
        | (df['name'] == '520Li')
        | (df['name'] == '520Li xDrive')
        | (df['name'] == '520d')
        | (df['name'] == '520d xDrive')
        | (df['name'] == '520i')
        | (df['name'] == '520i xDrive')
        | (df['name'] == '523i')
        | (df['name'] == '523i xDrive')
        | (df['name'] == '525d')
        | (df['name'] == '525d xDrive')
        | (df['name'] == '525i')
        | (df['name'] == '525i xDrive')
        | (df['name'] == '525xd')
        | (df['name'] == '525xd xDrive')
        | (df['name'] == '525xi')
        | (df['name'] == '525xi xDrive')
        | (df['name'] == '528Li')
        | (df['name'] == '528Li xDrive')
        | (df['name'] == '528i')
        | (df['name'] == '528i xDrive')
        | (df['name'] == '530Li')
        | (df['name'] == '530Li xDrive')
        | (df['name'] == '530d')
        | (df['name'] == '530d xDrive')
        | (df['name'] == '530i')
        | (df['name'] == '530i xDrive')
        | (df['name'] == '530xd')
        | (df['name'] == '530xd xDrive')
        | (df['name'] == '530xi')
        | (df['name'] == '530xi xDrive')
        | (df['name'] == '535d')
        | (df['name'] == '535d xDrive')
        | (df['name'] == '535i')
        | (df['name'] == '535i xDrive')
        | (df['name'] == '540i')
        | (df['name'] == '540i xDrive')
        | (df['name'] == '545i')
        | (df['name'] == '545i xDrive')
        | (df['name'] == '550i')
        | (df['name'] == '550i xDrive')
        | (df['name'] == 'M550d')
        | (df['name'] == 'M550i')
        | (df['name'] == 'M550i xDrive')
        | (df['name'] == 'M550d xDrive'),
        (df['name'] == '620d')  # 6ER
        | (df['name'] == '620d xDrive')
        | (df['name'] == '630d')
        | (df['name'] == '630d xDrive')
        | (df['name'] == '630i')
        | (df['name'] == '630i xDrive')
        | (df['name'] == '640d')
        | (df['name'] == '640d xDrive')
        | (df['name'] == '640i')
        | (df['name'] == '640i xDrive')
        | (df['name'] == '645i')
        | (df['name'] == '645i xDrive')
        | (df['name'] == '650i')
        | (df['name'] == '650i xDrive'),
        (df['name'] == '725Ld')  # 7ER
        | (df['name'] == '725Ld xDrive')
        | (df['name'] == '728i')
        | (df['name'] == '728i xDrive')
        | (df['name'] == '730Ld')
        | (df['name'] == '730Ld xDrive')
        | (df['name'] == '730Li')
        | (df['name'] == '730Li xDrive')
        | (df['name'] == '730d')
        | (df['name'] == '730d xDrive')
        | (df['name'] == '730i')
        | (df['name'] == '730i xDrive')
        | (df['name'] == '735Li')
        | (df['name'] == '735Li xDrive')
        | (df['name'] == '735i')
        | (df['name'] == '735i xDrive')
        | (df['name'] == '740Ld')
        | (df['name'] == '740Ld xDrive')
        | (df['name'] == '740Li')
        | (df['name'] == '740Li xDrive')
        | (df['name'] == '740d')
        | (df['name'] == '740d xDrive')
        | (df['name'] == '740i')
        | (df['name'] == '740i xDrive')
        | (df['name'] == '745Li')
        | (df['name'] == '745Li xDrive')
        | (df['name'] == '745i')
        | (df['name'] == '745i xDrive')
        | (df['name'] == '750Ld')
        | (df['name'] == '750Ld xDrive')
        | (df['name'] == '750Li')
        | (df['name'] == '750Li xDrive')
        | (df['name'] == '750d')
        | (df['name'] == '750d xDrive')
        | (df['name'] == '750i')
        | (df['name'] == '750i xDrive')
        | (df['name'] == '760Li')
        | (df['name'] == '760Li xDrive')
        | (df['name'] == '760i')
        | (df['name'] == 'M760Li')
        | (df['name'] == 'M760Li xDrive'),
        (df['name'] == '840d')  # 8ER
        | (df['name'] == '840d xDrive')
        | (df['name'] == 'M850i')
        | (df['name'] == 'M850i xDrive'),
        (df['name'] == '1.5hyb')  # I8
        | (df['name'] == 'Electro'),
        (df['name'] == 'Competition'),  # M5
        (df['name'] == 'Competition Package'),  # M4
        ((df['name'] == '18d') & (df['model_age'] == '1'))  # X1
        | ((df['name'] == '18d') & (df['model_age'] == '5'))
        | ((df['name'] == '18d') & (df['model_age'] == '11'))
        | ((df['name'] == '18i') & (df['model_age'] == '5'))
        | ((df['name'] == '18i') & (df['model_age'] == '8'))
        | ((df['name'] == '18i') & (df['model_age'] == '11'))
        | ((df['name'] == '20d') & (df['model_age'] == '2'))
        | ((df['name'] == '20d') & (df['model_age'] == '5'))
        | ((df['name'] == '20d') & (df['model_age'] == '8'))
        | ((df['name'] == '20d') & (df['model_age'] == '11'))
        | ((df['name'] == '20i') & (df['model_age'] == '1'))
        | ((df['name'] == '20i') & (df['model_age'] == '5'))
        | ((df['name'] == '20i') & (df['model_age'] == '8'))
        | ((df['name'] == '20i') & (df['model_age'] == '11') & (df['number_of_doors'] == '5'))
        | (df['name'] == '23d')
        | ((df['name'] == '25i') & (df['model_age'] == '11'))
        | ((df['name'] == '28i') & (df['model_age'] == '8'))
        | ((df['name'] == '28i') & (df['model_age'] == '11'))
        | (df['name'] == '20d xDrive'),
        ((df['name'] == '18d') & (df['model_age'] == '3'))  # X2
        | ((df['name'] == '18i') & (df['model_age'] == '3'))
        | (df['name'] == '18i sDrive')
        | (df['name'] == '18d xDrive')
        | (df['name'] == '20i xDrive')
        | (df['name'] == '20i sDrive'),
        ((df['name'] == '20d') & (df['model_age'] == '3'))  # X3
        | ((df['name'] == '20d') & (df['model_age'] == '10'))
        | ((df['name'] == '20d') & (df['model_age'] == '14'))
        | ((df['name'] == '20i') & (df['model_age'] == '3'))
        | ((df['name'] == '20i') & (df['model_age'] == '6'))
        | (df['name'] == '20i xDrive')
        | ((df['name'] == '20i') & (df['model_age'] == '10'))
        | ((df['name'] == '20i') & (df['model_age'] == '14'))
        | ((df['name'] == '25i') & (df['model_age'] == '14'))
        | ((df['name'] == '25i') & (df['model_age'] == '17'))
        | ((df['name'] == '28i') & (df['model_age'] == '10'))
        | ((df['name'] == '30d') & (df['model_age'] == '3'))
        | (df['name'] == '30d xDrive')
        | ((df['name'] == '30d') & (df['model_age'] == '17'))
        | ((df['name'] == '30i') & (df['model_age'] == '3'))
        | ((df['name'] == '30i') & (df['model_age'] == '17'))
        | (df['name'] == '30i xDrive')
        | ((df['name'] == 'M40d') & (df['model_age'] == '3'))
        | ((df['name'] == '35d') & (df['model_age'] == '10'))
        | ((df['name'] == '35d') & (df['model_age'] == '14'))
        | ((df['name'] == '35i') & (df['model_age'] == '7'))
        | (df['name'] == '35i xDrive')
        | (df['name'] == '35d xDrive')
        | ((df['name'] == '35i') & (df['model_age'] == '10')),
        ((df['name'] == '20d') & (df['model_age'] == '6'))  # X4
        | ((df['name'] == '20i') & (df['model_age'] == '2'))
        | ((df['name'] == '28i') & (df['model_age'] == '6'))
        | (df['name'] == '28i xDrive')
        | ((df['name'] == '30i') & (df['model_age'] == '2'))
        | ((df['name'] == 'M40d') & (df['model_age'] == '2'))
        | (df['name'] == 'M40i'),
        (df['name'] == '25d')  # X5
        | (df['name'] == '3.0')
        | (df['name'] == '3.0d')
        | (df['name'] == '3.0i')
        | (df['name'] == '3.0sd')
        | (df['name'] == '3.0si')
        | ((df['name'] == '30d') & (df['model_age'] == '7'))
        | ((df['name'] == '30d') & (df['model_age'] == '10'))
        | ((df['name'] == '30d') & (df['model_age'] == '14'))
        | ((df['name'] == '30i') & (df['model_age'] == '14'))        
        | (df['name'] == '4.0')
        | (df['name'] == '4.4')
        | (df['name'] == '4.4i')
        | (df['name'] == '4.6is')
        | (df['name'] == '4.8i')
        | (df['name'] == '4.8is')
        | ((df['name'] == '40d') & (df['model_age'] == '7'))
        | ((df['name'] == '40d') & (df['model_age'] == '10'))
        | (df['name'] == '40e')
        | (df['name'] == '48i')
        | (df['name'] == '5.0')
        | ((df['name'] == '50i') & (df['model_age'] == '7'))
        | ((df['name'] == '50i') & (df['model_age'] == '10'))
        | ((df['name'] == 'M50d') & (df['model_age'] == '7'))
        | ((df['name'] == 'M50d') & (df['model_age'] == '10')),
        ((df['name'] == '30d') & (df['model_age'] == '1'))  # X6
        | ((df['name'] == '30d') & (df['model_age'] == '6'))
        | ((df['name'] == '30d') & (df['model_age'] == '8'))
        | ((df['name'] == '30d') & (df['model_age'] == '13'))
        | ((df['name'] == '35d') & (df['model_age'] == '13'))
        | (df['name'] == '35d')
        | ((df['name'] == '35i') & (df['model_age'] == '6'))
        | ((df['name'] == '35i') & (df['model_age'] == '8'))
        | ((df['name'] == '35i') & (df['model_age'] == '13'))
        | ((df['name'] == '40d') & (df['model_age'] == '6'))
        | ((df['name'] == '40d') & (df['model_age'] == '8'))
        | ((df['name'] == '40d') & (df['model_age'] == '13'))
        | ((df['name'] == '40i') & (df['model_age'] == '1'))
        | ((df['name'] == '50i') & (df['model_age'] == '6'))
        | ((df['name'] == '50i') & (df['model_age'] == '8'))
        | ((df['name'] == '50i') & (df['model_age'] == '13'))
        | (df['name'] == 'ActiveHybrid')
        | ((df['name'] == 'M50d') & (df['model_age'] == '1'))
        | ((df['name'] == 'M50d') & (df['model_age'] == '6'))
        | ((df['name'] == 'M50d') & (df['model_age'] == '8'))
        | (df['name'] == 'M50i'),
        ((df['name'] == '30d') & (df['model_age'] == '2'))  # X7
        | ((df['name'] == '40i') & (df['model_age'] == '2'))
        | ((df['name'] == 'M50d') & (df['model_age'] == '2')),
        (df['name'] == '2.0')  # Z4
        | (df['name'] == '2.2')
        | ((df['name'] == '20i') & (df['model_age'] == '7'))
        | (df['name'] == '20i')
        | ((df['name'] == '20i') & (df['model_age'] == '11') & (df['number_of_doors'] == '2'))
        | (df['name'] == '23i')
        | ((df['name'] == '28i') & (df['model_age'] == '7'))
        | ((df['name'] == '30i') & (df['model_age'] == '11'))
        | ((df['name'] == '35i') & (df['model_age'] == '11'))
        | (df['name'] == '35is')
        | (df['name'] == 'CS'),
    ]
    choices = ['1ER','2ER','3ER','4','5ER','6ER','7ER','8ER','I8','M5','M4','X1','X2','X3','X4','X5','X6','X7','Z4']
    df['series'] = np.select(conditions, choices, default='default')
    return df['series']
test['series'] = test_series(test)
test.series.value_counts()
test[test['series'] == 'default']
# Сделаю это категориальным признаком
test['covid'] = '0'
train = pd.read_csv(DIR_TRAIN+'train_data_auto_ru.csv') # мой подготовленный датасет для обучения модели
train.sample(5)
train.info()
train = train.rename(columns={'pts_origin': 'technical_passport', 'complectation': 'equipment', 'model_name': 'series'})
train = train.drop(columns=['brand'])
train['production_date'] = pd.to_datetime(train['production_date'], format='%Y').dt.year

train['car_age'] = datetime.datetime.now().year - train['production_date']
train['mileage_per_year'] = train['mileage'] / train['car_age']

train['mileage_per_year'] = train['mileage_per_year'].apply(lambda x: cat_mileage_per_year(x))
train['mileage_per_year'] = train['mileage_per_year'].astype('str')
# Отберу нужные цены и умножу на нужное число, чтобы сравнить
train.query('car_age > 2 & car_age < 6')['price'] * 0.98
# Сделаю нужное преобразование уже в рабочем датафрейме
train.loc[(train['car_age'] > 2) & (train['car_age'] < 6), 'price'] *= 0.98
# Можно так еще
mask = train.eval("2 < car_age < 6")
train.loc[mask, 'price'] *= 0.98
col_info(train.mileage_per_year)
train['car_age'] = train['car_age'].apply(lambda x: cat_age(x))

train['car_age'] = train['car_age'].astype('str')
train['exp_year'] = train.mileage.apply(lambda x: x//17000 if x<=400000 else 400000//17000+1)
train['mileage'] = train['mileage'].apply(lambda x: cat_mileage(x))
train['mileage'] = train['mileage'].astype('str')
col_info(train.gear_type)
dict_gear_type = {
    'REAR_DRIVE': 'rear',
    'ALL_WHEEL_DRIVE': '4w',
    'FORWARD_CONTROL': 'front'
}

train['gear_type'] = train['gear_type'].map(dict_gear_type)
col_info(train.steering_wheel)
dict_steering_wheel = {
    'LEFT': 'left',
    'RIGHT': 'right'
}

train['steering_wheel'] = train['steering_wheel'].map(dict_steering_wheel)
def get_autoru_features(equipment):
    # Создаем пустой список, в который будут добавляться все фичи
    autoru_features = []
    for data in equipment:
        # Находим все слова между кавычками
        features=re.findall(r'\'(.+?)\'',data)
        # Добавляем в общий список
        autoru_features.extend(features)
    # Удаляем дубликаты
    autoru_features = list(dict.fromkeys(autoru_features))
    return autoru_features
autoru_features = get_autoru_features(train.equipment)  
print('Всего уникальных фич:', len(autoru_features))
print(autoru_features)
def get_features_train(equipment):      
    features=re.findall(r'\'(.+?)\'',equipment)
    return features
train['equipment'] = train['equipment'].apply(lambda x: get_features_train(x))
train.sample(5)
train_equipment = train.copy()

# Функция для отображения фич в записи
def find_item(cell):
    if item in cell:
        return 1
    return 0
# Создаем набор фич
features = set()
for featurelist in train_equipment['equipment']:
    for feature in featurelist:
        features.add(feature)
# Cоздаем столбцы с фичами и заполняем 0 или 1
for item in features:
    train_equipment[item] = train_equipment['equipment'].apply(find_item)
cor_target = abs(train_equipment.corr()["price"])
relevant_features = cor_target[cor_target>0.3] # Выбираем фичи с значением модуля коэффициента корреляции > 0.3
relevant_features
equipment_list = [
    'high-beam-assist',
    'front-seats-heat-vent',
    'bluetooth',
    'laser-lights',
    'light-sensor',
    'adaptive-light',
    'apple-carplay',
    'electro-trunk',
    'third-row-seats',
    'keyless-entry',
    'activ-suspension',
    'multizone-climate-control',
    'body-kit',
    'projection-display',
    'start-stop-function',
    'start-button',
    'paint-metallic',
    'wheel-heat',
    'passenger-seat-electric',
    'rain-sensor',
    'navigation',
    'voice-recognition',
    'multi-wheel',
    'airbag-curtain',
    'glonass',
    'auto-mirrors',
    'usb',
    'power-latching-doors',
    'led-lights',
    'auto-park',
    'bas',
    'isofix',
    'tyre-pressure',
    'air-suspension',
    'decorative-interior-lighting',
    '360-camera',
    'wireless-charger',
    'electronic-gage-panel',
]
equipment_dict = {
    'Система управления дальним светом': 'high-beam-assist',
    'Вентиляция передних сидений': 'front-seats-heat-vent',
    'Bluetooth': 'bluetooth',
    'Лазерные фары': 'laser-lights',
    'Датчик света': 'light-sensor',
    'Система адаптивного освещения': 'adaptive-light',
    'CarPlay': 'apple-carplay',
    'Электропривод крышки багажника': 'electro-trunk',
    'Третий ряд сидений': 'third-row-seats',
    'Система доступа без ключа': 'keyless-entry',
    'Активная подвеска': 'activ-suspension',
    'Климат-контроль многозонный': 'multizone-climate-control',
    'Обвес кузова': 'body-kit',
    'Проекционный дисплей': 'projection-display',
    'Система «старт-стоп»': 'start-stop-function',
    'Запуск двигателя с кнопки': 'start-button',
    'Металлик': 'paint-metallic',
    'Обогрев рулевого колеса': 'wheel-heat',
    'Электрорегулировка передних сидений': 'passenger-seat-electric',
    'Датчик дождя': 'rain-sensor',
    'Навигационная система': 'navigation',
    'Голосовое управление': 'voice-recognition',
    'Мультифункциональное рулевое колесо': 'multi-wheel',
    'Подушки безопасности оконные (шторки)': 'airbag-curtain',
    'ЭРА-ГЛОНАСС': 'glonass',
    'Электроскладывание зеркал': 'auto-mirrors',
    'USB': 'usb',
    'Доводчик дверей': 'power-latching-doors',
    'Светодиодные фары': 'led-lights',
    'Система автоматической парковки': 'auto-park',
    'Система помощи при торможении (BAS, EBD)': 'bas',
    'Крепление детского кресла (передний ряд) ISOFIX': 'isofix',
    'Датчик давления в шинах': 'tyre-pressure',
    'Пневмоподвеска': 'air-suspension',
    'Декоративная подсветка салона': 'decorative-interior-lighting',
    'Камера 360°': '360-camera',
    'Беспроводная зарядка для смартфона': 'wireless-charger',
    'Электронная приборная панель': 'electronic-gage-panel',
}
def test_feature_change(x):
    x = [equipment_dict.get(a) if equipment_dict.get(a) else a for a in x]
    return x
test['equipment'] = test['equipment'].apply(lambda x: test_feature_change(x))
def main_feature(x):
    super_features = []
    for equipment in x:
        if equipment in equipment_list:
            super_features.append(equipment)
    x = super_features
    return x
train['equipment'] = train['equipment'].apply(lambda x: main_feature(x))
test['equipment'] = test['equipment'].apply(lambda x: main_feature(x))
for item in equipment_list:
    train[item] = train['equipment'].apply(find_item)
for item in equipment_list:
    test[item] = test['equipment'].apply(find_item)
col_info(train.not_damage)
train['not_damage'] = train['not_damage'].astype(int)
col_info(train.technical_passport)
train['technical_passport']
dict_technical_passport = {
    'ORIGINAL': 'original',
    'DUPLICATE': 'duplicate'
}

train['technical_passport'] = train['technical_passport'].map(dict_technical_passport)

train['technical_passport'] = train['technical_passport'].fillna('Unknown')
col_info(train.custom_clear)
train['custom_clear'] = train['custom_clear'].astype(int)
train = train.drop(columns=['custom_clear'])
col_info(train.body_type)
col_info(train.model_date_begin)
train['model_date_begin'] = pd.to_datetime(train['model_date_begin'], format='%Y').dt.year

train['model_age'] = datetime.datetime.now().year - train['model_date_begin']
train['model_age'] = train['model_age'].apply(lambda x: 20 if x > 19 else x)
train['model_age'] = train['model_age'].astype('str')
col_info(train.number_of_doors)
col_info(train.fuel_type)
dict_fuel_type = {
    'DIESEL': 'diesel',
    'GASOLINE': 'gasoline',
    'HYBRID': 'hybrid',
    'ELECTRO': 'electro'
}

train['fuel_type'] = train['fuel_type'].map(dict_fuel_type)
col_info(train.vehicle_transmission)
dict_vehicle_transmission = {
    'AUTOMATIC': 'AT',
    'MECHANICAL': 'MT',
    'ROBOT': 'AMT'
}

train['vehicle_transmission'] = train['vehicle_transmission'].map(dict_vehicle_transmission)
col_info(train.engine_displacement)
train['engine_displacement'] = train['engine_displacement'].astype(float)
train['engine_displacement'] = round((train['engine_displacement'] / 1000), 1)
col_info(train.engine_power)
col_info(train.owners)
train['owners'] = train['owners'].fillna(1)
# приведу к числовому виду
train['owners'] = train['owners'].astype(int)
col_info(train.engine_power)
col_info(train.name)
# Новый датафрейм с разбитыми данными из нужной мне колонки - сделаю так, чтобы это было решением не частным, а для всех случаев
name_tmp = train['name'].str.split(" ", n = 2, expand = True) 


# попробуем склеить первые две колонки, чтобы получить название модели автомобиля, но при этом будем соблюдать некоторые условия:
# во второй должно содержаться xDrive, Competition, sDrive, Package, тогда это будет добавлено к первой колонке
# в первой колонке может быть все, что угодно
# конъюнкция двух признаков
def make_conj(data, feature1, feature2):
    data['tmp_1'] = np.where(~data[feature2].str.contains('xDrive|Competition|sDrive|Package'),
                                 '', data[feature2])
    
    data['tmp_0'] = data[feature1]
    
    data['tmp'] = data['tmp_0'] + ' ' + data['tmp_1']
                    
    return (data)

# выполним корректировку названия
make_conj(name_tmp, 0, 1)

# выполним корректировку названия
make_conj(name_tmp, 0, 1)


train['name'] = name_tmp['tmp']

train['name'] = train['name'].apply(lambda x: x.replace('xDrive20d', '20d xDrive'))
train['name'] = train['name'].apply(lambda x: x.replace('sDrive18i', '18i sDrive'))
train['name'] = train['name'].apply(lambda x: x.replace('sDrive20i', '20i sDrive'))
train['name'] = train['name'].apply(lambda x: x.replace('xDrive18d', '18d xDrive'))
train['name'] = train['name'].apply(lambda x: x.replace('xDrive20', '20 xDrive'))
train['name'] = train['name'].apply(lambda x: x.replace('20 xDrivei', '20i xDrive'))
train['name'] = train['name'].str.strip()
name_tmp.sample(10)
train[['name', 'engine_displacement', 'fuel_type', 'vehicle_transmission', 'engine_power']].sample(20)
train[train['name'] == ' ']
train = train.loc[train['name'] != ' ']
col_info(train.series)
cars_name_for_search_model3 = train[['name', 'series', 'model_date_begin', 'body_type', 'engine_displacement', 'fuel_type', 'vehicle_transmission', 'engine_power', 'price']]
cars_name_for_search_model3.to_excel("cars_name_for_search_model3.xlsx",
             sheet_name='cars_name_for_search_model')  

cars_name_for_search_model3
cars_name_for_search_model3_groupby_series = cars_name_for_search_model3.groupby(['series','name','model_date_begin','body_type','engine_displacement','fuel_type','engine_power']).agg({'price': 'sum'}).sort_values(by=['series'], ascending=False).reset_index()
cars_name_for_search_model3_groupby_series.to_excel("cars_name_for_search_model3_groupby_series.xlsx",
             sheet_name='model_groupby_series')  

cars_name_for_search_model3_groupby_series
dic_model = {
    '1 серия': '1ER',
    '1M': '1ER',
    '2ACTIVETOURER': '2ER',
    '2 серия': '2ER',
    '2 серия Grand Tourer': '2ER',
    '2 серия Active Tourer': '2ER',
    '2GRANDTOURER': '2ER',
    '3 серия': '3ER',
    '4': '4ER',
    '4 серия': '4ER',
    '5 серия': '5ER',
    '6 серия': '6ER',
    '7 серия': '7ER',
    '8 серия': '8ER',
    'i8': 'I8',
    'i3': 'I3',
    'M2': '2ER',
    'M3': '3ER',
    'M4': '4ER',
    'M5': '5ER',
    'M6': '6ER',
    'M8': '8ER',
    'X1': 'X1',
    'X2': 'X2',
    'X3': 'X3',
    'X3 M': 'X3',
    'X4': 'X4',
    'X4 M': 'X4',
    'X5': 'X5',
    'X5 M': 'X5',
    'X6': 'X6',
    'X6 M': 'X6',
    'X7': 'X7',
    'Z1': 'Z1',
    'Z3': 'Z3',
    'Z3 M': 'Z3',
    'Z4': 'Z4',
    'Z4 M': 'Z4',
    'Z8': 'Z8',
}
train['series'] = train['series'].map(dic_model)
col_info(train.series)
col_info(train.model_date_end)
train[['name', 'model_date_end']].groupby('name').agg({'model_date_end': 'value_counts'})
a = train[['name', 'model_date_end']].groupby('name').agg({'model_date_end': 'value_counts'}).to_dict(orient='split')
print(type(a))
print(a)
train = train.drop(columns=['model_date_end'])
from skimage import color


def color_hex_replace(hex):


    peaked_color = '#' + f"{hex}"
    
#     print(peaked_color)

    # Initialize a dictionary where the key is the RGB value as hex string, and the value is the color name
    # https://en.wikipedia.org/wiki/List_of_colors:_A%E2%80%93F

    colors_dict = {
        "000000":"black",
        "C0C0C0":"silver",
        "808080":"grey",
        "FFFFFF":"white",
        "800000":"maroon",
        "FF0000":"red",
        "800080":"purple",
        "FF00FF":"fuchsia",
        "008000":"green",
        "006400":"darkgreen",
        "00FF00":"lime",
        "808000":"olive",
        "FFFF00":"yellow",
        "000080":"navy",
        "0000FF":"blue",
        "008080":"teal",
        "00FFFF":"aqua",
        "a52a2a":"brown",
        "f5f5dc":"beige",
        "ffd700":"gold",
        "ffa500":"orange",
        "0000cd":"mid_blue",
        "ee82ee":"violet"
    }
    
    
    # Get a list of color values in hex string format
    hex_rgb_colors = list(colors_dict.keys())

    # https://stackoverflow.com/questions/39908314/slice-all-strings-in-a-list, https://stackoverflow.com/questions/9210525/how-do-i-convert-hex-to-decimal-in-python
    r = [int(hex[0:2], 16) for hex in hex_rgb_colors]  # List of red elements.
    g = [int(hex[2:4], 16) for hex in hex_rgb_colors]  # List of green elements.
    b = [int(hex[4:6], 16) for hex in hex_rgb_colors]  # List of blue elements.

    r = np.asarray(r, np.uint8)  # Convert r from list to array (of uint8 elements)
    g = np.asarray(g, np.uint8)  # Convert g from list to array
    b = np.asarray(b, np.uint8)  # Convert b from list to array

    rgb = np.dstack((r, g, b)) #Stack r,g,b across third dimention - create to 3D array (of R,G,B elements).

    # Convert from sRGB color spave to LAB color space
    # https://stackoverflow.com/questions/13405956/convert-an-image-rgb-lab-with-python
    lab = color.rgb2lab(rgb)



    # Convert peaked color from sRGB color spave to LAB color space
    # peaked_color = '#673429ff'
    peaked_rgb = np.asarray([int(peaked_color[1:3], 16), int(peaked_color[3:5], 16), int(peaked_color[5:7], 16)], np.uint8)
    peaked_rgb = np.dstack((peaked_rgb[0], peaked_rgb[1], peaked_rgb[2]))
    peaked_lab = color.rgb2lab(peaked_rgb)

    # Compute Euclidean distance from peaked_lab to each element of lab
    lab_dist = ((lab[:,:,0] - peaked_lab[:,:,0])**2 + (lab[:,:,1] - peaked_lab[:,:,1])**2 + (lab[:,:,2] - peaked_lab[:,:,2])**2)**0.5

    # Get the index of the minimum distance
    min_index = lab_dist.argmin()

    # Get hex string of the color with the minimum Euclidean distance (minimum distance in LAB color space)
    peaked_closest_hex = hex_rgb_colors[min_index]

    # Get color name from the dictionary
    peaked_color_name = colors_dict[peaked_closest_hex]

    return peaked_color_name
        


color_hex_replace('ffffff')
train['color'] = train['color'].apply(color_hex_replace)
col_info(train.color)
train['color'] = train['color'].apply(lambda x: x.lower())
train['color']
col_info(train.body_type)
# словарь для кодировки
dic_body_type = {
    'Внедорожник 5 дв.': 'allroad',
    'Кабриолет': 'cabriolet',
    'Компактвэн': 'compact_van',
    'Компактвэн Gran Tourer': 'compact_van_gran_tourer',
    'Купе': 'coupe',
    'Купе-хардтоп': 'coupe_hardtop',
    'Лифтбек': 'liftback',
    'Лифтбек Gran Turismo': 'liftback_gran_turismo',
    'Родстер': 'roadster',
    'Седан': 'sedan',
    'Седан 2 дв.': 'sedan',
    'Седан Long': 'sedan_long',
    'Универсал 5 дв.': 'station_wagon',
    'Хэтчбек 5 дв.': 'hatchback',
    'Хэтчбек 3 дв.': 'hatchback',
    'Хэтчбек 3 дв. Compact': 'hatchback_compact'
}

train['body_type'] = train['body_type'].map(dic_body_type)
train = train.dropna()
train['covid'] = '1'
import seaborn as sns

sns.distplot(train.price.values)
np.median(train.price.values)
col_info(train.price)
train['price'] = train['price'].astype(int)
train['price'] = train['price'] * 0.91
train.info()
print('test:\n', test['technical_passport'].sample(10))
print('train:\n', train['technical_passport'].sample(10))
display(train.sample(5))
display(test.sample(5))
train.drop(['equipment'], axis=1, inplace=True,)
test.drop(['equipment'], axis=1, inplace=True,)
# Укажу, какую колонку нужно исключить из списка (в тесте нет колонки цены)
excluded_feats = ['price']

# Запоминаем порядок колонок
column_list = [f_ for f_ in train.columns if f_ not in excluded_feats]


# Устанавливаем порядок колонок как для трейновой выборки, иначе предсказания неверные.
test = test[column_list]
VAL_SIZE   = 0.33   # 33%
N_FOLDS    = 5

# RANDOM_SEED
RANDOM_SEED = 42
X = train.drop(['price'], axis=1,)
y = train.price.values
X_sub = test.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
X_train.info()
N_FOLDS    = 5

# CATBOOST
ITERATIONS = 2000
LR         = 0.1

# RANDOM_SEED
RANDOM_SEED = 42
cat_features_ids = np.where(X_train.dtypes == object)[0].tolist()
cat_features_ids
categorical_features_names = ['body_type', 'color', 'fuel_type', 'name', 'vehicle_transmission', 'mileage', 'gear_type',
                              'steering_wheel',  'technical_passport', 'model_age', 'car_age']
model = CatBoostRegressor(iterations = ITERATIONS,
                          learning_rate = LR,
                          random_seed = RANDOM_SEED,
                          eval_metric='MAPE',
                          custom_metric=['R2', 'MAE']
                         )
model.fit(X_train, y_train,
         cat_features=categorical_features_names,
         eval_set=(X_test, y_test),
         verbose_eval=100,
         use_best_model=True,
         plot=True
         )
model.save_model('catboost_single_model_baseline.model')
from matplotlib import pyplot as plt

idx = np.argsort(model.feature_importances_)

plt.figure(figsize=(17,8))

sns.barplot(x=model.feature_importances_[idx], y=np.array(model.feature_names_)[idx])
predict_submission = model.predict(X_sub)
predict_submission
VERSION = 20
sample_submission['price'] = predict_submission
sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)
sample_submission.head(10)
def cat_model(y_train, X_train, X_test, y_test):
    model = CatBoostRegressor(iterations = ITERATIONS,
                              learning_rate = LR,
                              eval_metric='MAPE',
                              random_seed = RANDOM_SEED,)
    model.fit(X_train, y_train,
              cat_features=cat_features_ids,
              eval_set=(X_test, y_test),
              verbose=False,
              use_best_model=True,
              plot=False)
    
    return(model)


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))
submissions = pd.DataFrame(0,columns=["sub_1"], index=sample_submission.index) # куда пишем предикты по каждой модели
score_ls = []
splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))

for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS,):
    # use the indexes to extract the folds in the train and validation data
    X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
    # model for this fold
    model = cat_model(y_train, X_train, X_test, y_test,)
    # score model on test
    test_predict = model.predict(X_test)
    test_score = mape(y_test, test_predict)
    score_ls.append(test_score)
    print(f"{idx+1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")
    # submissions
    submissions[f'sub_{idx+1}'] = model.predict(X_sub)
    model.save_model(f'catboost_fold_{idx+1}.model')
    
print(f'Mean Score: {np.mean(score_ls):0.3f}')
print(f'Std Score: {np.std(score_ls):0.4f}')
print(f'Max Score: {np.max(score_ls):0.3f}')
print(f'Min Score: {np.min(score_ls):0.3f}')
submissions.head(10)
submissions['blend'] = (submissions.sum(axis=1))/len(submissions.columns)
sample_submission['price'] = submissions['blend'].values
sample_submission.to_csv(f'submission_blend_v{VERSION}.csv', index=False)
sample_submission.head(10)
new_train = train.copy()
new_test = test.copy()
new_test['price'] = 0
col_info(new_test.name)
len(new_test.name.tolist())
test_car_name_list = new_test.name.tolist()
test_car_name_list = set(test_car_name_list)
len(test_car_name_list)
new_train = new_train.query('name == @test_car_name_list')
new_test
new_train.info()
delete_outliers_iqr(new_train, 'price')
new_test['dataset'] = 'test'
new_train['dataset'] = 'train'
# Запоминаем порядок колонок
column_list = [f_ for f_ in new_test.columns]


# Устанавливаем порядок колонок как для трейновой выборки, иначе предсказания неверные.
new_train = new_train[column_list]
new_big_df = pd.concat([new_train, new_test], ignore_index=True)
new_big_df
cat_features_ids = np.where(new_big_df.dtypes == float)[0].tolist()

cat_features_ids
new_big_df.info()
new_big_df.iloc[:,[9, 60]]


col_info(new_big_df.mileage_per_year)
new_big_df['engine_displacement'] = (new_big_df['engine_displacement'] * 1000).astype('int')
new_big_df['engine_displacement']
new_big_df['price'] = new_big_df['price'].astype(int)
new_big_df['model_age'] = new_big_df['model_age'].astype(int)
new_big_df['model_age']
new_big_df['engine_displacement']
def luxury_tax(df):
    conditions = [
        ((df['name'] == 'M240i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))  # 1.1
        | ((df['name'] == 'M235i xDrive') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000) & (df['body_type'] <= 'coupe'))
        | ((df['name'] == '330d xDrive') & (df['model_age'] == 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == 'M340i xDrive') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '340i xDrive') & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '330i xDrive') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '340i xDrive') & (df['model_age'] > 1) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000) & (df['body_type'] <= 'liftback_gran_turismo'))
        | ((df['name'] == '430i') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000) & (df['body_type'] <= 'coupe'))
        | ((df['name'] == '430i xDrive') & (df['model_age'] > 1) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000) & (df['body_type'] <= 'cabriolet'))
        | ((df['name'] == '430i') & (df['model_age'] > 1) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000) & (df['body_type'] <= 'cabriolet'))
        | ((df['name'] == '420d') & (df['model_age'] <= 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 2000) & (df['body_type'] <= 'cabriolet'))
        | ((df['name'] == '430i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000) & (df['body_type'] <= 'coupe'))
        | ((df['name'] == '440i xDrive') & (df['model_age'] > 1) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000) & (df['body_type'] <= 'coupe'))
        | ((df['name'] == '440i') & (df['model_age'] > 1) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000) & (df['body_type'] <= 'cabriolet'))
        | ((df['name'] == '520i') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '520d') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '520d xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '530d xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '530i') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '530i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '540i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '630d xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '630i') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '640d xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '640i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '730i') & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline'))
        | ((df['name'] == 'M2') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M2 Competition') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M550d xDrive') & (df['model_age'] > 1) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M550i xDrive') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M40i') & (df['series'] <= 'X3') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M40d') & (df['series'] <= 'X3') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '20d xDrive') & (df['series'] <= 'X3') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '20i xDrive') & (df['series'] <= 'X3') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '30i xDrive') & (df['series'] <= 'X3') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '30d xDrive') & (df['series'] <= 'X3') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M40i') & (df['series'] <= 'X4') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M40d') & (df['series'] <= 'X4') & (df['model_age'] <= 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '20i xDrive') & (df['series'] <= 'X4') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '20d xDrive') & (df['series'] <= 'X4') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '20i xDrive M Sport') & (df['series'] <= 'X4') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '28i xDrive') & (df['series'] <= 'X4') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '30i xDrive') & (df['series'] <= 'X4') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '30d xDrive') & (df['series'] <= 'X4') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '35d xDrive') & (df['series'] <= 'X4') & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '35i xDrive') & (df['series'] <= 'X4') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '25d xDrive Business') & (df['series'] <= 'X5') & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '30d xDrive') & (df['series'] <= 'X5') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '35i xDrive') & (df['series'] <= 'X5') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '40i xDrive') & (df['series'] <= 'X5') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '40d xDrive') & (df['series'] <= 'X5') & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '40e xDrive') & (df['series'] <= 'X5') & (df['model_age'] <= 3) & (df['fuel_type'] == 'hybrid'))
        | ((df['name'] == '50i xDrive') & (df['series'] <= 'X5') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == '30d xDrive') & (df['series'] <= 'X6') & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '20i sDrive') & (df['series'] <= 'Z4') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '30i sDrive') & (df['series'] <= 'Z4') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == 'M40i') & (df['series'] <= 'Z4') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M4') & (df['model_age'] > 1) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000) & (df['body_type'] == 'coupe'))
        | ((df['name'] == '120Ah') & (df['model_age'] > 0) & (df['model_age'] <= 1) & (df['fuel_type'] == 'electro'))
        | ((df['name'] == '620d xDrive') & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 2000)),
        ((df['name'] == 'М4') & (df['model_age'] <= 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))  # 2
        | ((df['name'] == 'М') & (df['series'] <= 'X3') & (df['model_age'] <= 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M40d') & (df['series'] <= 'X4') & (df['model_age'] <= 1) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M40i') & (df['series'] <= 'X4') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M') & (df['series'] <= 'X4') & (df['model_age'] <= 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '640i xDrive') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000) & (df['body_type'] == 'cabriolet'))
        | ((df['name'] == '640i') & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000) & (df['body_type'] == 'cabriolet'))
        | ((df['name'] == '650i xDrive') & (df['model_age'] > 2) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'coupe'))
        | ((df['name'] == '650i xDrive') & (df['model_age'] > 2) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'cabriolet'))
        | ((df['name'] == '650i xDrive') & (df['model_age'] > 2) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'coupe'))
        | ((df['name'] == '650i') & (df['model_age'] > 2) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'cabriolet'))
        | ((df['name'] == '730i') & (df['model_age'] <= 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '730d xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 5) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '730Ld xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 5) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '740d xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 5) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '740Ld xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 5) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '740Le xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 4) & (df['fuel_type'] == 'hybrid') & (df['engine_displacement'] <= 2000))
        | ((df['name'] == '740Li xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 4) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '750d xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '750i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == '750Ld xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '750Li xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M550d xDrive') & (df['model_age'] <= 1) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M550i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M5') & (df['model_age'] > 2) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'sedan'))
        | ((df['name'] == 'M5 Competition') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M6') & (df['model_age'] > 3) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'coupe'))
        | ((df['name'] == 'M6') & (df['model_age'] <= 4) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'cabriolet'))
        | ((df['name'] == 'M6') & (df['model_age'] > 2) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'coupe'))
        | ((df['name'] == 'M760Li xDrive') & (df['model_age'] <= 4) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 7000))
        | ((df['name'] == '50i xDrive') & (df['series'] <= 'X5') & (df['model_age'] == 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M') & (df['series'] <= 'X5') & (df['model_age'] > 0) & (df['model_age'] <= 5) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M50i') & (df['series'] <= 'X5') & (df['model_age'] <= 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M50d') & (df['series'] <= 'X5') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M50d xDrive') & (df['series'] <= 'X5') & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '30d xDrive Exclusive') & (df['series'] <= 'X5') & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '30d xDrive Pure Experience') & (df['series'] <= 'X5') & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '40d xDrive M Sport') & (df['series'] <= 'X5') & (df['model_age'] > 3) & (df['model_age'] <= 5) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '35i xDrive') & (df['series'] <= 'X6') & (df['model_age'] == 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '40i xDrive') & (df['series'] <= 'X6') & (df['model_age'] <= 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '30d xDrive') & (df['series'] <= 'X6') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M') & (df['series'] <= 'X6') & (df['model_age'] > 0) & (df['model_age'] <= 5) & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M50i') & (df['series'] <= 'X6') & (df['model_age'] <= 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == 'M50d') & (df['series'] <= 'X6') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M50d xDrive') & (df['series'] <= 'X6') & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '30d xDrive Luxury') & (df['series'] <= 'X6') & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '30d xDrive M Sport') & (df['series'] <= 'X6') & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '30d xDrive Pure Extravagance') & (df['series'] <= 'X6') & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '40d xDrive') & (df['series'] <= 'X6') & (df['model_age'] > 1) & (df['model_age'] <= 3) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '40d xDrive M Sport') & (df['series'] <= 'X6') & (df['model_age'] <= 4) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '50i xDrive') & (df['series'] <= 'X6') & (df['model_age'] > 1) & (df['model_age'] <= 4) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500))
        | ((df['name'] == '30d xDrive') & (df['series'] <= 'X7') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '40i xDrive') & (df['series'] <= 'X7') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == 'M50d') & (df['series'] <= 'X7') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000))
        | ((df['name'] == '840i xDrive') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000) & (df['body_type'] == 'cabriolet'))
        | ((df['name'] == '840i xDrive') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000) & (df['body_type'] == 'coupe'))
        | ((df['name'] == '840d xDrive') & (df['model_age'] == 1) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000) & (df['body_type'] == 'coupe'))
        | ((df['name'] == '840d xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'diesel') & (df['engine_displacement'] <= 3000) & (df['body_type'] == 'coupe'))
        | ((df['name'] == '840i xDrive') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 3000) & (df['body_type'] == 'coupe'))
        | ((df['name'] == 'M850i xDrive') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'coupe'))
        | ((df['name'] == 'M850i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'coupe'))
        | ((df['name'] == 'M850i xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'cabriolet'))
        | ((df['name'] == 'i8') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'hybrid') & (df['engine_displacement'] <= 1500) & (df['body_type'] == 'coupe')),
        ((df['name'] == 'M8') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'coupe'))  # 3
        | ((df['name'] == 'M8') & (df['model_age'] == 1) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 4500) & (df['body_type'] == 'cabriolet'))
        | ((df['name'] == 'i8') & (df['model_age'] > 2) & (df['model_age'] <= 4) & (df['fuel_type'] == 'hybrid') & (df['engine_displacement'] <= 1500))
        | ((df['name'] == 'i8') & (df['model_age'] > 0) & (df['model_age'] <= 2) & (df['fuel_type'] == 'hybrid') & (df['engine_displacement'] <= 1500) & (df['body_type'] == 'roadster'))
        | ((df['name'] == 'M760Li xDrive') & (df['model_age'] > 0) & (df['model_age'] <= 3) & (df['fuel_type'] == 'gasoline') & (df['engine_displacement'] <= 6600) & (df['body_type'] == 'roadster')),
    ]
    choices = [1.1,2,3]
    df['luxury_tax'] = np.select(conditions, choices, default=1)
    return df['luxury_tax']
new_big_df['luxury_tax'] = luxury_tax(new_big_df)
col_info(new_big_df.luxury_tax)
def make_car_tax(row):
    if row.engine_power <= 100:
            row.car_tax = row.engine_power * row.luxury_tax * 12
            return row.car_tax
        
    elif 100 < row.engine_power <= 125:
        row.car_tax = row.engine_power * row.luxury_tax * 20
        return row.car_tax
    
    elif 125 < row.engine_power <= 150:
        row.car_tax = row.engine_power * row.luxury_tax * 24
        return row.car_tax
    
    elif 150 < row.engine_power <= 175:
        row.car_tax = row.engine_power * row.luxury_tax * 42
        return row.car_tax
    
    elif 175 < row.engine_power <= 200:
        row.car_tax = row.engine_power * row.luxury_tax * 43
        return row.car_tax

    elif 200 < row.engine_power <= 225:
        row.car_tax = row.engine_power * row.luxury_tax * 58
        return row.car_tax
    
    elif 225 < row.engine_power <= 250:
        row.car_tax = row.engine_power * row.luxury_tax * 68
        return row.car_tax
    
    elif row.engine_power > 250:
        row.car_tax = row.engine_power * row.luxury_tax * 132
        return row.car_tax
  
 
new_big_df['car_tax'] = new_big_df.apply(lambda row: make_car_tax(row), axis=1)
new_big_df['model_age'] = new_big_df['model_age'].astype('str')
new_big_df['car_tax'] = new_big_df['car_tax'].astype(int)
new_big_df['luxury_tax'] = new_big_df['luxury_tax'].astype('str')
col_info(new_big_df.car_tax)
equipments_list = [
    'high-beam-assist',
    'front-seats-heat-vent',
    'bluetooth',
    'laser-lights',
    'light-sensor',
    'adaptive-light',
    'apple-carplay',
    'electro-trunk',
    'third-row-seats',
    'keyless-entry',
    'activ-suspension',
    'multizone-climate-control',
    'body-kit',
    'projection-display',
    'start-stop-function',
    'start-button',
    'paint-metallic',
    'wheel-heat',
    'passenger-seat-electric',
    'rain-sensor',
    'navigation',
    'voice-recognition',
    'multi-wheel',
    'airbag-curtain',
    'glonass',
    'auto-mirrors',
    'usb',
    'power-latching-doors',
    'led-lights',
    'auto-park',
    'bas',
    'isofix',
    'tyre-pressure',
    'air-suspension',
    'decorative-interior-lighting',
    '360-camera',
    'wireless-charger',
    'electronic-gage-panel']
new_big_df['equipments_count'] = new_big_df[equipments_list].sum(axis=1).astype('int')
col_info(new_big_df.not_damage)
new_big_df.drop(['not_damage'], axis=1, inplace=True,)
# #encoders for categorical data
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import category_encoders as ce

# from itertools import combinations

# object_cols = new_big_df.select_dtypes('object').columns
# low_cardinality_cols = [col for col in object_cols if new_big_df[col].nunique() < 15]
# # low_cardinality_cols.append('price')
# interactions = pd.DataFrame(index=new_big_df.index)

# # Генерирую комбинации фичей
# for features in combinations(low_cardinality_cols,2):
    
#     new_interaction = new_big_df[features[0]].map(str)+"_"+new_big_df[features[1]].map(str)
    
#     encoder = LabelEncoder()
#     interactions["_".join(features)] = encoder.fit_transform(new_interaction)
# low_cardinality_cols
# interactions
# new_big_df = new_big_df.join(interactions) #добавлю теперь новые колонки в датасет
new_big_df.info()
new_big_df.to_csv('new_big_df.csv')  
# def make_car_quality(row):
#     if row.series == '3ER' and 1997 <= row.model_date_begin < 2006:
#         row.quality = '1'
#         return row.quality
        
#     elif row.name == '116i' or row.name == '116' or row.name == '116d':
#         row.quality = '1'
#         return row.quality
    
#     elif row.series == 'X5':
#         row.quality = '1'
#         return row.quality
    
#     elif row.series == '3ER' and 2005 < row.model_date_begin <= 2013:
#         row.quality = '1'
#         return row.quality
    
#     elif row.series == '5ER' and 2003 <= row.model_date_begin <= 2009:
#         row.quality = '1'
#         return row.quality
    
#     else:
#         row.quality = '0'
#         return row.quality
  
 
# new_big_df['quality'] = new_big_df.apply(lambda row: make_car_quality(row), axis=1)
# col_info(new_big_df.quality)
# def make_car_popular(row):
#     if row.series == '3ER' and row.series == 'sedan':
#         row.popularity = '1'
#         return row.popularity
    
#     elif row.series == '5ER' and row.series == 'sedan':
#         row.popularity = '1'
#         return row.popularity
        
#     elif row.series == 'X5' or row.series == 'X6' or row.series == 'X3':
#         row.popularity = '1'
#         return row.popularity
    
#     else:
#         row.popularity = '0'
#         return row.popularity
  
 
# new_big_df['popularity'] = new_big_df.apply(lambda row: make_car_popular(row), axis=1)
# def make_car_rating(row):
#     if row.series == '6ER':
#         row.rating = 9.3
#         return row.rating
    
#     elif row.series == 'X6':
#         row.rating = 9.2
#         return row.rating
    
#     elif row.series == 'X5':
#         row.rating = 9.1
#         return row.rating
    
#     elif row.series == 'X3':
#         row.rating = 9.0
#         return row.rating
    
#     elif row.series == '7ER':
#         row.rating = 9.0
#         return row.rating
    
#     elif row.series == 'Z4':
#         row.rating = 8.9
#         return row.rating
    
#     elif row.series == '5ER':
#         row.rating = 8.8
#         return row.rating
    
#     elif row.series == '1ER':
#         row.rating = 8.8
#         return row.rating
    
#     elif row.series == 'X4':
#         row.rating = 8.7
#         return row.rating
    
#     elif row.series == 'X1':
#         row.rating = 8.7
#         return row.rating
    
#     elif row.series == '3ER':
#         row.rating = 8.7
#         return row.rating
    
#     else:
#         row.rating = '0'
#         return row.rating
  
 
# new_big_df['rating'] = new_big_df.apply(lambda row: make_car_rating(row), axis=1)
# new_big_df['rating'] = new_big_df['rating'].astype('float')
# col_info(new_big_df.rating)
new_big_df = new_big_df.drop('series', axis=1)
new_big_df = new_big_df.drop('model_date_begin', axis=1)
new_big_df.drop(['covid'], axis=1, inplace=True,)
new_big_df.info()
new_big_df['name'] = new_big_df.name.apply(lambda x: x.replace('i', '').replace('d', ''))
new_big_df.sample(15)
cat_features_ids = np.where(new_big_df.dtypes == object)[0].tolist()

cat_features_ids
categorical_features_names = ['body_type', 'color', 'fuel_type', 'name', 'vehicle_transmission', 'mileage', 'gear_type',
                              'steering_wheel',  'technical_passport', 'car_age', 'mileage_per_year', 'model_age', 'luxury_tax']
new_big_df.info()
for column in categorical_features_names:
    dummies_train = pd.get_dummies(new_big_df[column], prefix = new_big_df[column].name)

    # Удаляем исходный столбец и добавляем dummies
    new_big_df = new_big_df.drop(new_big_df[column].name, axis=1).join(dummies_train)

new_big_df.to_csv('new_big_df.csv', index=False)  
new_big_df['price']
VAL_SIZE   = 0.33   # 33%
N_FOLDS    = 5

# CATBOOST
ITERATIONS = 2000
LR         = 0.1

# RANDOM_SEED
RANDOM_SEED = 42
dataset_train = 'train'
dataset_test = 'test'

train_preproc = new_big_df.query('@dataset_train in dataset').drop(['dataset'], axis=1)
X_sub = new_big_df.query('@dataset_test in dataset').drop(['dataset', 'price'], axis=1)


# # Запоминаем порядок колонок
column_list = X_sub.columns

X = train_preproc.drop(['price'], axis=1,)

# # Устанавливаем порядок колонок как для тестовой выборки, иначе предсказания неверные.
X = X[column_list]

y = train_preproc.price.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True, random_state=RANDOM_SEED)
def compute_meta_feature(model, X_train, X_test, y_train, cv):
   
    X_meta_train = np.zeros_like(y_train, dtype = np.float32)
    for train_fold_index, predict_fold_index in cv.split(X_train):
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]
        
        folded_model = clone(model)
        folded_model.fit(X_fold_train, y_fold_train)
        X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)
        
    meta_model = clone(model)
    meta_model.fit(X_train, y_train)
    
    X_meta_test = meta_model.predict_proba(X_test)[:,1]
    
    return X_meta_train, X_meta_test
cv = KFold(n_splits=N_FOLDS, shuffle=True)
# 1 - Catboost

cat_features_ids = np.where(X.dtypes == object)[0].tolist()

X_meta_train_features = []
X_meta_test_features = []

model = CatBoostRegressor(iterations = ITERATIONS,
                          learning_rate = LR,
                          random_seed = RANDOM_SEED,
                          eval_metric='MAPE',
                          custom_metric=['R2', 'MAE'],
                          loss_function = 'RMSE'
                         )

X_meta_train = np.zeros_like(y, dtype = np.float32)
X_meta_test = np.zeros(len(X_sub), dtype = np.float32)
for train_fold_index, predict_fold_index in cv.split(X):
    X_fold_train, X_fold_predict = X.iloc[train_fold_index], X.iloc[predict_fold_index]
    y_fold_train = y[train_fold_index]

    folded_model = clone(model)
    folded_model.fit(X_fold_train, y_fold_train,
                     cat_features=cat_features_ids,
                     eval_set=(X_test, y_test),
                     verbose_eval=1000,
                     use_best_model=True,
                     plot=False
)
    X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)
    X_meta_test += folded_model.predict(X_sub)

X_meta_test = X_meta_test / N_FOLDS

X_meta_train_features.append(X_meta_train)
X_meta_test_features.append(X_meta_test)

print(model.get_best_score())
# 2 - RandomForestRegressor

model = RandomForestRegressor(n_estimators=400, random_state=42)

X_meta_train = np.zeros_like(y, dtype = np.float32)
X_train_num = X
X_sub_num = X_sub

for train_fold_index, predict_fold_index in cv.split(X_train_num):
    X_fold_train, X_fold_predict = X_train_num.iloc[train_fold_index], X_train_num.iloc[predict_fold_index]
    y_fold_train = y[train_fold_index]

    folded_model = clone(model)
    folded_model.fit(X_fold_train, y_fold_train)
    X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)

meta_model = clone(model)
meta_model.fit(X_train_num, y)

X_meta_test = meta_model.predict(X_sub_num)

X_meta_train_features.append(X_meta_train)
X_meta_test_features.append(X_meta_test)
# 3 LinearRegression

model = LinearRegression(normalize = True)

X_meta_train = np.zeros_like(y, dtype = np.float32)

for train_fold_index, predict_fold_index in cv.split(X_train_num):
    X_fold_train, X_fold_predict = X_train_num.iloc[train_fold_index], X_train_num.iloc[predict_fold_index]
    y_fold_train = y[train_fold_index]

    folded_model = clone(model)
    folded_model.fit(X_fold_train, y_fold_train)
    X_meta_train[predict_fold_index] = folded_model.predict(X_fold_predict)

meta_model = clone(model)
meta_model.fit(X_train_num, y)

X_meta_test = meta_model.predict(X_sub_num)

X_meta_train_features.append(X_meta_train)
X_meta_test_features.append(X_meta_test)
stacked_features_train = np.vstack(X_meta_train_features[:2]).T
stacked_features_test = np.vstack(X_meta_test_features[:2]).T
final_model = LinearRegression()
final_model.fit(stacked_features_train, y)
VERSION = 65
sample_submission['price'] = np.floor(final_model.predict(stacked_features_test) / 10000) * 10000 
sample_submission.to_csv(f'submission_stack_v{VERSION}.csv', index=False)
sample_submission.head(10)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from collections import defaultdict
from sklearn.model_selection import KFold


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))

def print_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    print(f'RMSE = {rmse:.2f}, MAE = {mae:.2f}, R-sq = {r2:.2f}, MAPE = {mape:.2f} ')
def compute_meta_feature(clf, X_train, X_test, y_train, cv):
    """
    Computes meta-features usinf the classifier cls
    
    :arg model: scikit-learn classifier
    :arg X_train, y_train: training set
    :arg X_test: testing set
    :arg cv: cross-validation folding
    """
    
    X_meta_train = np.zeros_like(y_train, dtype = np.float32)
    X_meta_test = np.zeros(len(X_test), dtype=np.float32)
    for train_fold_index, predict_fold_index in cv.split(X_train):
        X_fold_train, X_fold_predict = X_train[train_fold_index], X_train[predict_fold_index]
        y_fold_train = y_train[train_fold_index]
        
        folded_clf = clone(clf)
        folded_clf.fit(X_fold_train, y_fold_train)
            
        
        X_meta_train[predict_fold_index] = folded_clf.predict(X_fold_predict)
        
        print_regression_metrics(X_meta_train[predict_fold_index], y_train[predict_fold_index])
        X_meta_test += folded_clf.predict(X_test)
    
    n = cv.n_splits
    X_meta_test = X_meta_test / n
    
    return X_meta_train, X_meta_test
def generate_meta_features(regressors, X_train, X_test, y_train, cv):
   
    features = [
        compute_meta_feature(clf, X_train, X_test, y_train, cv)
        for clf in tqdm(regressors)
    ]
    
    stacked_features_train = np.stack([
        features_train for features_train, features_test in features
    ], axis=-1)

    stacked_features_test = np.stack([
        features_test for features_train, features_test in features
    ], axis=-1)
    
    return stacked_features_train, stacked_features_test
cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def compute_metric(clf, X_train=X_train, y_train=y_train, X_test=X_test):
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)
    return print_regression_metrics(y_test, y_test_pred)
from sklearn.base import clone

from sklearn.preprocessing import StandardScaler
# Стандартизируем данные:
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
test = scaler.fit_transform(X_sub)

stacked_features_train, stacked_features_test = generate_meta_features([
    RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
    BaggingRegressor(ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_SEED)),
    CatBoostRegressor(loss_function = 'MAE',
                         eval_metric = 'MAPE',
                         learning_rate=0.005,
                         iterations=4500,
                         l2_leaf_reg=2,
                         depth=6,
                         bootstrap_type = 'Bayesian',
                         random_seed=42,
                         od_type='Iter',
                         od_wait=100)
    ], X_train, test, y_train, cv)


#Строим мета-алгоритм

final_model = LinearRegression()
final_model.fit(stacked_features_train, y_train)
VERSION = 42
y_pred = np.round((final_model.predict(stacked_features_test)/1000))*1000

sample_submission['price'] =  y_pred
sample_submission.to_csv(f'submission_stack_v{VERSION}.csv', index=False)

sample_submission.head(10)
dataset_train = 'train'
dataset_test = 'test'

train_data = new_big_df.query('@dataset_train in dataset').drop(['dataset'], axis=1)
test_data = new_big_df.query('@dataset_test in dataset').drop(['dataset', 'price'], axis=1)


# # Запоминаем порядок колонок
column_list = test_data.columns

X_train = train_data.drop(['price'], axis=1)

# # Устанавливаем порядок колонок как для тестовой выборки, иначе предсказания неверные.
X_train = X_train[column_list]
X_test = test_data
y = np.log(train_data.price.values)
import datetime as dt
from vecstack import stacking

from sklearn.linear_model import LinearRegression 
from sklearn.ensemble import ExtraTreesRegressor    
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred-y_true)/y_true))


# Configure models
RANDOM_SEED = 42


lr = LinearRegression(normalize=True, n_jobs=-1)

etc = ExtraTreesRegressor(n_estimators=500,  n_jobs=-1,
                          random_state=RANDOM_SEED)  # max_depth=5,
catb = CatBoostRegressor(iterations=3500,
                                 learning_rate=0.05,
                                 random_seed=RANDOM_SEED,
                                 eval_metric='MAPE',
                                 verbose = 500
                                 )
rf = RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1,
                           n_estimators=500)  # , max_depth=3

knn = KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=-1)


print("Finished setting up regressors at ", dt.datetime.now())

# Initialize 1-st level models.
models = [catb, rf, etc, knn]

# Compute stacking features
S_train, S_test = stacking(models, X_train, y, X_test,
                           regression=True, metric=mape, n_folds=4,
                           shuffle=True, random_state=RANDOM_SEED, verbose=2)

# Initialize 2-nd level model
model = lr

# Fit 2-nd level model
model = model.fit(S_train, y)

# Predict
y_test_pred = np.exp(model.predict(S_test))
VERSION = 80
sample_submission['price'] = y_test_pred
# sample_submission['price'] = sample_submission['price'].apply(lambda x: round(x/1000)*1000)
sample_submission.to_csv(f'submission_stack_v{VERSION}.csv', index=False)
sample_submission.head(10)
import matplotlib.pyplot as plt

num=new_train.select_dtypes(exclude='object')
numcorr=num.corr()
f,ax=plt.subplots(figsize=(17,1))
sns.heatmap(numcorr.sort_values(by=['price'], ascending=False).head(1), cmap='Blues')
plt.title(" Numerical features correlation with the price", weight='bold', fontsize=18)
plt.xticks(weight='bold')
plt.yticks(weight='bold', color='dodgerblue', rotation=0)


plt.show()
num=numcorr['price'].sort_values(ascending=False).head(10).to_frame()

cm = sns.light_palette("cyan", as_cmap=True)

s = num.style.background_gradient(cmap=cm)
s
# Все возможные типы топлива
fuel_type_uniq = new_train['fuel_type'].unique()
fuel_type_uniq
# Формирование dummy- столбцов для топлива
new_train_extend = pd.concat([new_train, 
                              pd.get_dummies(new_train['fuel_type'])], 
                             axis=1)
# Коэффициенты корреляции
fuel_type_corr = new_train_extend[fuel_type_uniq].corr()
fuel_type_corr
# Тепловая карта
sns.set(font_scale=1)
plt.subplots(figsize=(10, 10))
sns.heatmap(fuel_type_corr, square=True, 
           annot=True, fmt=".1f", linewidths=0.1, cmap="RdBu")
### Drop fuel_type_gasoline
new_big_df.drop('fuel_type_gasoline', axis=1, inplace=True)
fig = plt.figure(figsize=(7, 7))
plt.grid(True)
plt.title('Price')
sns.boxplot(x = new_train['price'])
Q1 =  new_train['price'].quantile(0.25)
Q3 = new_train['price'].quantile(0.75)
IQR = Q3 - Q1

new_train.query("@Q1 - 1.5*@IQR < price < @Q3 + 1.5*@IQR").boxplot(column="price")
outliers_price = outliers_iqr_long(new_train.price)

print(len(outliers_price))
col_info(new_train.price)
new_train_corr = new_train.corr()
new_train_corr
# new_train_corr.style.background_gradient(cmap='coolwarm')
new_train_corr.style.background_gradient(cmap='coolwarm').set_precision(2)
# Create correlation matrix
corr_matrix = new_train.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

print(to_drop)

# Drop features 
# new_big_df.drop(to_drop, axis=1, inplace=True)
for c in new_train.columns[:-1]:
    plt.figure(figsize=(20,5))
    plt.title("{} vs. \nprice".format(c),fontsize=16)
    plt.scatter(x=new_train[c],y=new_train['price'],color='blue',edgecolor='k')
    plt.grid(True)
    plt.xlabel(c,fontsize=14)
    plt.ylabel('Стоимость', fontsize=14)

    plt.show()
new_big_df.columns.tolist()
new_big_df[['price', 'name']].groupby('name').sum()
new_big_df[new_big_df['name'] == '']
new_big_df['name'].value_counts()
new_big_df['name'].unique()
col_info(new_big_df.price)
# Функция очистки от выбросов
def delete_outliers_iqr(df, column):
    # Считаем первый и третий квартили
    first_quartile = df[column].describe()['25%']
    third_quartile = df[column].describe()['75%']

    # IQR (Межквартильный размах)
    iqr = third_quartile - first_quartile

    print(first_quartile, third_quartile, iqr)

    # Удаляем то, что выпадает за границы IQR
    #     df_tmp = df.copy()
    df = df[(df[column] > (first_quartile - 3 * iqr)) &
                (df[column] < (third_quartile + 3 * iqr))]

    df[column].hist()
    df[column].describe()

    df = df.loc[df[column].between(first_quartile - 1.5*iqr, third_quartile + 1.5*iqr)]
    df.info()

var = 'car_age'
data = pd.concat([new_train['price'], new_train[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 10))
fig = sns.boxplot(x=var, y="price", data=data)
fig.axis(ymin=0, ymax=165);
plt.xticks(rotation=90);
#plot graph of feature importances for better visualization

plt.figure(figsize = (12,8))
feat_importances = pd.Series(etc.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()