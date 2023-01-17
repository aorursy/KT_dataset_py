import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.model_selection import train_test_split,cross_val_score, cross_val_predict

from sklearn.metrics import confusion_matrix

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier

%matplotlib inline

# открываем датасет

data = pd.read_csv('/kaggle/input/airbag-and-other-influences/nassCDS.csv')

print(data.head())
# Строим графики

ax = data.groupby('dead')['Unnamed: 0'].nunique().plot(kind='bar')

ax.xaxis.set_label_text("")

ax.set_title("Исход аварий")

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(

                                    ncols=2,

                                    nrows=2,

                                    figsize=(15, 10))

fig.suptitle('Соотношение факторов', fontsize=16)

for ax in [(ax1,'dvcat', 'Скорость'),

           (ax2, 'airbag', 'Подушка безопасности'),

           (ax3, 'seatbelt', 'Ремень безопасности'),

           (ax4, 'frontal', 'Лобовое столкновение')]:

    data.groupby(ax[1])['Unnamed: 0'].nunique().plot(kind='bar', ax = ax[0])

    ax[0].xaxis.set_label_text("")

    ax[0].set_title(ax[2])

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(

                                    ncols=2,

                                    nrows=2,

                                    figsize=(15, 10))

fig.suptitle('Соотношение смертельных исходов при различных факторах', fontsize=16)

for ax in [(ax1,'dvcat', 'Скорость'),

           (ax2, 'airbag', 'Подушка безопасности'),

           (ax3, 'seatbelt', 'Ремень безопасности'),

           (ax4, 'frontal', 'Лобовое столкновение')]:

    data.query('dead == "dead"').groupby(ax[1])['Unnamed: 0'].nunique().plot(kind='bar', ax = ax[0])

    ax[0].xaxis.set_label_text("")

    ax[0].set_title(ax[2])

plt.show()
# Прослеживается прямая зависимость между скоростью автомобиля и наличием ремня безопасности и вероятностью смертельного исхода

# Переименовываем первый столбец, поскольку в датасете столбец, содержащий id инцидента, не проименован

data.rename(columns={'Unnamed: 0': 'AccId'}, inplace=True)

# Даем категории dead конкретные численные значения

data = data.replace({'dead': 1, 'alive': 0})

# Смотрим на незаполненные ячейки

data.isnull().sum()
sns.heatmap(data.isnull(), cbar = False).set_title("Карта отсутствующих значений")
# Заменяем пустые ячейки в yearVeh автомобился средним годом выпуска по датасету.

data['yearVeh'] = data['yearVeh'].replace({np.nan: np.mean(data['yearVeh'])})

# Год инцидента, тяжесть полученных увечий и внутренний id инцидента не учитываем за отсутствием релевантности

data.drop(['yearacc', 'injSeverity', 'caseid'], axis=1, inplace=True)

# Преобразуем категории в код

for cat in ('dvcat', 'airbag','sex','occRole', 'deploy', 'abcat', 'seatbelt'):

    data[cat] = preprocessing.LabelEncoder().fit_transform(data[cat])

# В столбце года выпуска  yearVeh присутствуют пустые ячейки

data.info()
# Столбцы dead и yearVeh с целыми значениями имеют отличный от int тип

# Исправляем это

for cat in ('dead', 'yearVeh'):

    data[cat] = data[cat].astype(int)

# Просматриваем корреляцию признаков с dead

correlations_data = data.corr()['dead'].sort_values()

print(correlations_data)
# Коэффициенты парной корреляции достаточно малы (<0.5)

# Нормализованная БД выглядит следующим образом:

print(data.head())
# Создаем копию датасета и делим ее на features и targets

features = data.copy()

targets = features['dead']

features.drop(['dead'], axis=1)

# Делим features и targets на обучающие и тестовые выборки

x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2,random_state = 42)

# Наивный Байес

clf = GaussianNB()

clf.fit(x_train, np.ravel(y_train))

print(f"Точность НБ: {round(clf.score(x_test, y_test) * 100, 4 )}%")
result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')

print(f"Оценка кросс-валидации для НБ: {round(result_rf.mean()*100, 4)}%")

y_pred = cross_val_predict(clf,x_train,y_train,cv=10)

sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f',cmap="summer")

plt.title('Матрица ошибок для НБ', y=1.05, size=20)
# Случайный лес

clf = RandomForestClassifier(criterion='entropy',

                            n_estimators=700,

                            min_samples_split=10,

                            min_samples_leaf=1,

                            max_features='auto',

                            oob_score=True,

                            random_state=1,

                            n_jobs=-1)

clf.fit(x_train, np.ravel(y_train))

print(f"Точность СЛ: {round(clf.score(x_test, y_test) * 100, 4)}%")

result_rf=cross_val_score(clf,x_train,y_train,cv=10,scoring='accuracy')

print(f'Оценка кросс-валидации для СЛ {round(result_rf.mean()*100, 4)}')

y_pred = cross_val_predict(clf,x_train,y_train,cv=10)

sns.heatmap(confusion_matrix(y_train,y_pred),annot=True,fmt='3.0f', cmap="summer")

plt.title('Матрица ошибок для СЛ', y=1.05, size=20)
# Создаем предсказание по таблице с отсутствующими данными по смертности

# result = clf.predict(X_to_be_predicted)