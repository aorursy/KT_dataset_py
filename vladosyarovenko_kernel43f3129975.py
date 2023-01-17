# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
file_path='../input/coronavirusdataset/Case.csv'
data = pd.read_csv(file_path,index_col="case_id", parse_dates=True)
#убираем дубли
data.drop_duplicates(inplace=True)
#группируем данные по провинции и суммируем случаи заражения в граппах
datas=data.groupby(['province']).confirmed.sum().reset_index()
plt.figure(figsize=(35,10))
#строим столбчатый график
sns.barplot(x=datas.province, y=datas.confirmed)
plt.ylabel("Количество инфицированных")
plt.xlabel("провинции")


file_path='../input/coronavirusdataset/TimeGender.csv'
data = pd.read_csv(file_path,index_col="date", parse_dates=True)
data.drop_duplicates(inplace=True)
#группируем данные по дате и полу
datas=data.groupby(['date','sex']).confirmed.first().reset_index()
#настройка размера графика
plt.figure(figsize=(35,10))

#настройка графика с линиями по 2м полам
plt.plot( datas.groupby(['date']).date.first(), datas[datas['sex']=='male']['confirmed'], marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.plot( datas.groupby(['date']).date.first(), datas[datas['sex']=='female']['confirmed'], marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)
plt.legend()


file_path='../input/coronavirusdataset/TimeAge.csv'
dataa = pd.read_csv(file_path,index_col="date", parse_dates=True)
#убираем дупли
dataa.drop_duplicates(inplace=True)
#группировка по возрасту
datass=dataa.groupby(['age']).confirmed.sum().reset_index()
plt.figure(figsize=(35,10))

#очищаем данные для построения(нужны в числовом виде)
datass['age'] = datass['age'].map(lambda x: x.rstrip('s'))
datass['age'] = pd.to_numeric(datass['age'])

#ломаный график(возраст и количетсво заражений за весь месяц для конкретного возраста)
plt.plot(datass['age'], datass['confirmed'], marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)










#построение НС
# загружаем данные
file_path='../input/google-play-store-apps/googleplaystore.csv'
data = pd.read_csv(file_path)
dataAI=data
#приводим к числовому виду
dataAI['Installs'] = dataAI['Installs'].map(lambda x: x.replace(',',''))
dataAI['Installs'] = dataAI['Installs'].map(lambda x: x.replace('+',''))
dataAI['Price'] = dataAI['Price'].map(lambda x: x.replace('$',''))
dataAI['Size'] = dataAI['Size'].map(lambda x: x.replace('M',''))


#простая проверка на то,что данные можно предстваить как число,иначе удаляем строчку с неправильными данными
for index, row in dataAI.iterrows():
    if not row['Reviews'].isnumeric() or not row['Installs'].isnumeric() or not row['Size'].isnumeric() or not row['Price'].isnumeric():
        dataAI.drop(index, inplace=True)

#заполняем 0 если есть None
pd.DataFrame(dataAI).fillna(0,inplace=True)   
#набор критерий для поиска
features = ['Reviews','Installs','Price','Size']
X=dataAI[features]
#выходные значения
y = dataAI.Rating
#разбиваем данные на 2 категории(для тренировки,для обучения)
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#Создаем модель
rf_model = RandomForestRegressor(random_state=1)
#Обучаем
rf_model.fit(train_X, train_y)
#вывод
rf_val_predictions = rf_model.predict(val_X)
#средняя ошибка
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
val_X.reset_index(inplace=True)
pd.options.display.float_format = '{:,.1f}'.format
#записываем данные
output = pd.DataFrame({'Installs':val_X.Installs,'Reviews':val_X.Reviews,'Price:':val_X.Price,'confirmed': rf_val_predictions})
print(rf_val_mae)
print(output)