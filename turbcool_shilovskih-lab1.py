#Импорт библиотек:

import csv as csv

import pandas as pd

import numpy as np

import datetime as dt

from calendar import monthrange



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
#Импорт данных:

customers = pd.read_csv('../input/mlhse/CASC_Constant.csv')

orders = pd.read_csv('../input/mlhse/casc-resto.csv', sep=';', dtype = {

    'CustomerID': 'int64',

    'Restaurant': 'int64',

    'Quantity': 'int64',

    'SummBasic': 'float64',

    'SummAfterPointsUsage': 'float64'

}, parse_dates=['RKDate'], header=0, decimal=",")

customers.rename(columns={'CustomerId': 'CustomerID'}, inplace=True)
#Visited - клиент пришёл хотя бы раз за период START_DATE - END_DATE?

START_DATE = np.datetime64('2017-07-01')

END_DATE = np.datetime64('2017-12-31')

visited = orders.groupby('CustomerID').agg(

    Visited=('RKDate', lambda x: x.where((START_DATE <= x) & (x <= END_DATE)).count() > 0)

)
#Расчёт статистики по покупкам, сделанным до START_DATE (для покупателя):

prevOrders = orders.where(orders.RKDate < START_DATE)

customerStats = prevOrders.groupby('CustomerID').agg(

    Recency=('RKDate', lambda x: START_DATE - x.max()),

    Frequency=('RKDate', lambda x: x.groupby([x.dt.month]).agg('count').mean()/30), #среднее за день

    MonetaryValue=('SummBasic', 'mean')

)
#Объединяем все статистики в одну таблицу:

customerStats = customerStats.merge(customers, on='CustomerID').merge(visited, on='CustomerID').dropna()



#Подготавливаем данные к использованию в моделях:

customerStats['Visited'] = customerStats['Visited'].astype(int)

customerStats['Recency'] = customerStats['Recency'].dt.days

customerStats['Age'] = customerStats['Age'].astype(int)

customerStats['Sex'] = (customerStats['Sex'] == 'Female').astype(int)

#Доп. характеристики:

customerStats['SubscribedEmail'] = customerStats['SubscribedEmail'].astype(int)

customerStats['SubscribedPush'] = customerStats['SubscribedPush'].astype(int)
#Данные без доп. характеристик:

y = customerStats[['Visited']].values.ravel()

x = customerStats[['Age', 'Sex', 'Recency', 'Frequency', 'MonetaryValue']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)



#Данные с доп. характеристиками:

y_alt = customerStats[['Visited']].values.ravel()

x_alt = customerStats[['Age', 'Sex', 'Recency', 'Frequency', 'MonetaryValue', 'SubscribedEmail', 'SubscribedPush']]

X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(x_alt, y_alt, test_size=0.2)
#Модель без доп. характеристик:

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

scoreTrain = logreg.score(X_train, y_train)

scoreTest = logreg.score(X_test, y_test)

print('До добавления доп. характеристик:')

print(f"Score train: {scoreTrain*100:.2f}%\nScore test: {scoreTest*100:.2f}% \n")



#Модель с доп. характеристиками:

logreg_alt = LogisticRegression()

logreg.fit(X_train2, y_train2)

scoreTrain_alt = logreg.score(X_train_alt, y_train_alt)

scoreTest_alt = logreg.score(X_test_alt, y_test_alt)

print('После добавления доп. характеристик:')

print(f"Score train: {scoreTrain_alt*100:.2f}%\nScore test: {scoreTest_alt*100:.2f}%\n")

print(f"Преимущество: {(scoreTest_alt - scoreTest)*100:.2f}%")
# Метод k-средних

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))