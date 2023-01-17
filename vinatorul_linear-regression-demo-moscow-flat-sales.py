# подключение библиотек для работы с данными и визуализации

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



# загрузка данных из файла

file_path = "/kaggle/input/price-of-flats-in-moscow/flats_moscow.csv"

data = pd.read_csv(file_path)
# Описание данных, позволяет сразу узнать медианные значения, количество записей по каждому полю, максимум, минимум итд

data.describe()
# target - целевая функция

target = data.price

# строим график, чтобы посмотреть на распределение целевой функции

sns.distplot(target)

plt.show()
# Поскольку распределение оказалось сильно сдвинуто - работаем с логарифмами

sns.distplot(np.log(target))

plt.show()
# Создаём фигуру 10 на 5 дюймов

fig = plt.figure(figsize=(10, 5))

# Прикидываем сколько графиков нужно разместить, в моём случае 2, поэтому я выбрал 1 строку и 2 стобца

# Строим первый график в разметке (1 строка, 2 стобца)

# График распределения totsp - общей площади квартиры, кв.м.

fig.add_subplot(1, 2, 1)

sns.distplot(data.totsp)

# Строим второй график в разметке (1 строка, 2 стобца)

# График распределения livetsp - жилой площади квартиры, кв.м.

fig.add_subplot(1, 2, 2)

sns.distplot(data.livesp)



plt.tight_layout()

plt.show()
# аналогично целевой функции распределение показателей сильно сдвинуты, поэтому работаем с логарифмами

fig = plt.figure(figsize=(10, 5))

fig.add_subplot(1, 2, 1)

sns.distplot(np.log(data.totsp))

fig.add_subplot(1, 2, 2)

sns.distplot(np.log(data.livesp))

    

plt.tight_layout()

plt.show()

# с помощью numpy для всеого списка площадей вычисляем список прологарифмированных площадей

main = np.log(data.livesp)

# аналогично для всего списка целевых значений вычисляем список прологарифмированных значений

target_log = np.log(target)
# Создаём фигуру 10 на 10 дюймов, бубем использовать разметку 2x2 (2 строки, 2 столбца)

fig = plt.figure(figsize=(10, 10))

# На первом графике обе величины без изменений

fig.add_subplot(2, 2, 1)

sns.scatterplot(data.livesp, target)

# На втором графике величина livesp прологарифмирована, целевая функция без изменений

fig.add_subplot(2, 2, 2)

sns.scatterplot(main, target)

# На третьем графике величина livesp без изменений, целевая функция прологарифмирована

fig.add_subplot(2, 2, 3)

sns.scatterplot(data.livesp, target_log)

# На четвёртом графике величина, и livesp, и целевая функция прологарифмированы

fig.add_subplot(2, 2, 4)

sns.scatterplot(main, target_log)

    

plt.tight_layout()

plt.show()
# будем строить прямую по двум точкам, для этого возьмём минимальное и максимальное значения по x

min_x = main.min()

max_x = main.max()
# и в этих точках найдём значение функции h

t0 = 0

t1 = 1

hmn = t0 + t1 * min_x

hmx = t0 + t1 * max_x



# Построим два графика на одном

fig, ax = plt.subplots()

# декартова плоскость с размеченными точками

sns.scatterplot(main, target_log, ax=ax)

# прямая, соответствующая функции h

sns.lineplot(x=[min_x, max_x], y=[hmn, hmx], ax=ax)
# Вычислим значение J

J = 0

for i in range(len(data.livesp)):

    J += (np.exp(t0 + t1 * main[i]) - data.price[i])**2

J /= 2 * len(data.livesp)

print(J)
sns.regplot(main, target_log)

plt.show()
# Наивная попытка подбора коэффициентов

t1_arr = []

t2_arr = []

J_arr = []

best_t1 = -1

best_t2 = -1

best_J = 1000000

for i in np.arange(-2.5, 2.5, 0.1):

    for j in np.arange(0.0, 2.0, 0.1):

        tmp_J = 0

        for k in range(len(data.livesp)):

            tmp_J += (np.exp(i + j * main[k]) - data.price[k])**2

        tmp_J /= 2 * len(data.livesp)

        if tmp_J < best_J:

            best_i = i

            best_j = j

            best_J = tmp_J

        t1_arr.append(i)

        t2_arr.append(j)

        J_arr.append(min(tmp_J, 10000))

print(best_J)

print(i)

print(j)
# 3D отрисовка подбора коэффициентов.

# По оси X откладывается коэффициент theta0, по оси Y откладывается коэффициент theta1, а по Z откладывается значение J(theta1, theta)

import plotly.offline as py

import plotly.graph_objs as go

py.init_notebook_mode(connected=True)



trace1 = go.Scatter3d(

    x=t1_arr,

    y=t2_arr,

    z=J_arr,

    mode='markers',

    marker=dict(

        size=5,

        line=dict(

            color='rgba(217, 217, 217, 0.14)',

            width=0.1

        ),

        opacity=1

    ),

    name = 'price'

)

fig = go.Figure(data=trace1)

py.iplot(fig)
t0_best = t0

t1_best = t1

hmn = t0_best + t1_best * min_x

hmx = t0_best + t1_best * max_x



# Построим два графика на одном

fig, ax = plt.subplots()

# декартова плоскость с размеченными точками

sns.scatterplot(main, target_log, ax=ax)

# прямая, соответствующая функции h

sns.lineplot(x=[min_x, max_x], y=[hmn, hmx], ax=ax)

plt.show()
t0_best = t0

t1_best = t1

alpha = 0.0001

# желаемая точность

eps = 0.0001

# количество шагов, которое понадобилось сделать для достижения желаемой точности

c = 1



# Высляем частную производную функции J по t_0 во всех точках выборки, суммируем

J_t0 = 0

for i in range(len(data.livesp)):

    J_t0+= (t0_best + t1_best * main[i] - target_log[i])

# делим на количество

J_t0 /= len(data.livesp)

# умножаем на шаг обучения

J_t0 *= alpha



# Высляем частную производную функции J по t_1 во всех точках выборки, суммируем

J_t1 = 0

for i in range(len(data.livesp)):

    J_t1 += (t0_best + t1_best * main[i] - target_log[i]) * main[i]

# делим на количество

J_t1 /= len(data.livesp)

# умножаем на шаг обучения

J_t1 *= alpha



# обновляем значения t0 и t1

t0_best = t0_best - J_t0

t1_best = t1_best - J_t1

print(J_t0, J_t1)

# выводим на экран лучшие значения t0 и t1

print(t0_best, t1_best)

# отображаем график

hmn = t0_best + t1_best * min_x

hmx = t0_best + t1_best * max_x

fig, ax = plt.subplots()

sns.scatterplot(main, target_log, ax=ax)

sns.lineplot(x=[min_x, max_x], y=[hmn, hmx], ax=ax)

plt.show()
# Возьмём шаг обучения 0.01

alpha = 0.01

# желаемая точность

eps = 0.0001

# количество шагов, которое понадобилось сделать для достижения желаемой точности

c = 1



# Высляем частную производную функции J по t_0 во всех точках выборки, суммируем

J_t0 = 0

for i in range(len(data.livesp)):

    J_t0+= (t0_best + t1_best * main[i] - target_log[i])

# делим на количество

J_t0 /= len(data.livesp)

# умножаем на шаг обучения

J_t0 *= alpha



# Высляем частную производную функции J по t_1 во всех точках выборки, суммируем

J_t1 = 0

for i in range(len(data.livesp)):

    J_t1 += (t0_best + t1_best * main[i] - target_log[i]) * main[i]

# делим на количество

J_t1 /= len(data.livesp)

# умножаем на шаг обучения

J_t1 *= alpha



# обновляем значения t0 и t1

t0_best = t0_best - J_t0

t1_best = t1_best - J_t1



# запускаем цикл, пока J_t0 и J_t1 больше желаемой точности

# значения J_t0 и J_t1 показывают, на сколько изменились t0_best и t1_best

# и если это изменение меньшн eps, то мы достигли нужной точности

while (abs(J_t0) > eps or abs(J_t1) > eps):

    # Высляем частную производную функции J по t_0 во всех точках выборки, суммируем

    J_t0 = 0

    for i in range(len(data.livesp)):

        J_t0+= (t0_best + t1_best * main[i] - target_log[i])

    # делим на количество

    J_t0 /= len(data.livesp)

    # умножаем на шаг обучения

    J_t0 *= alpha



    # Высляем частную производную функции J по t_1 во всех точках выборки, суммируем

    J_t1 = 0

    for i in range(len(data.livesp)):

        J_t1 += (t0_best + t1_best * main[i] - target_log[i]) * main[i]

    # делим на количество

    J_t1 /= len(data.livesp)

    # умножаем на шаг обучения

    J_t1 *= alpha

    # обновляем значения t0 и t1

    t0_best = t0_best - J_t0

    t1_best = t1_best - J_t1

    c += 1

# выводим на экран количество шагов

print(c)

# выводим на экрнан значения J_t0 и J_t1, чтобы убедиться в достигнутой точности

print(J_t0, J_t1)

# выводим на экран лучшие значения t0 и t1

print(t0_best, t1_best)

# отображаем график

hmn = t0_best + t1_best * min_x

hmx = t0_best + t1_best * max_x

fig, ax = plt.subplots()

sns.scatterplot(main, target_log, ax=ax)

sns.lineplot(x=[min_x, max_x], y=[hmn, hmx], ax=ax)



plt.show()
# Вычислим значение J в точке начального приближения t0, t1 

# в моём случае (0, 1)

# если работа шла с логарифмическими величинами, то нужно не забыть взять экспоненту e^(h(x))

# если вы работали с исходными величинами, то np.exp в вашем случае не будет

J = 0

for i in range(len(data.livesp)):

    J += (np.exp(t0 + t1 * main[i]) - data.price[i])**2

J /= 2 * len(data.livesp)

print(J)



# Вычислим значение J в точке t0_best, t1_best

# аналогично если работа шла с логарифмическими величинами, то нужно не забыть взять экспоненту e^(h(x))

# если вы работали с исходными величинами, то np.exp в вашем случае не будет

J = 0

for i in range(len(data.livesp)):

    J += (np.exp(t0_best + t1_best * main[i]) - data.price[i])**2

J /= 2 * len(data.livesp)

print(J)
# Построим графики других параметром, чтобы выбрать пригодные для построения регрессии

attribs = data.select_dtypes(exclude='object').drop('price', axis=1).copy()



# построим все 10 графиков на одной фигуре

fig = plt.figure(figsize=(20,25))

for i in range(len(attribs.columns)):

    fig.add_subplot(4, 3, i+1)

    # перед построением удаляем NaN значения

    sns.distplot(attribs.iloc[:, i].dropna())

    plt.xlabel(attribs.columns[i])

    

plt.tight_layout()

plt.show()
features = ['totsp', 'livesp', 'kitsp', 'dist', 'metrdist', 'walk', 'brick', 'code']
X_corr = data[features + ['price']]

X_corr.describe()



x_corr_info =  X_corr.corr()

f, ax = plt.subplots(figsize=(14, 12))

sns.heatmap(x_corr_info, annot=True)

plt.show()
features = ['totsp', 'livesp', 'kitsp', 'dist', 'brick']

print(len(features))

theta0 = 0

thetaJ = [1 for i in range(len(features))]
# Вычислим значение J

thetaJnp = np.array(thetaJ)

X = data[features]

#print(X)

htheta = thetaJnp.dot(np.transpose(X))

J = 0

for i in range(len(htheta)):

    J += (t0 + htheta[i] - data['price'][i])**2

J /= 2 * len(htheta)

print(J)
# Возьмём шаг обучения 0.0001

alpha = 0.00022

# желаемая точность

eps = 0.0001

# количество шагов, которое понадобилось сделать для достижения желаемой точности

c = 1



X = data[features]



# задаём начальное приближение

theta0_best = 0

theta_best = [1 for i in range(len(features))]

htheta_best = np.array(theta_best).dot(np.transpose(X))



cont = True

# запускаем цикл, пока хотя бы одно значение частной производной больше eps

while True:  

    # Высляем частную производную функции J по t_0 во всех точках выборки, суммируем

    J_theta0 = 0

    for i in range(len(htheta_best)):

        J_theta0 += (theta0_best + htheta_best[i] - data.price[i])

    # делим на количество

    J_theta0 /= len(htheta_best)

    # умножаем на шаг обучения

    J_theta0 *= alpha



    J_theta = [0 for i in range(len(features))]

    for j in range(len(features)):

        # Высляем частную производную функции J по t_j во всех точках выборки, суммируем

        J_theta[j] = 0

        for i in range(len(htheta_best)):

            J_theta[j] += (theta0_best + htheta_best[i] - data.price[i]) * data[features[j]][i]

        # делим на количество

        J_theta[j] /= len(htheta_best)

        # умножаем на шаг обучения

        J_theta[j] *= alpha



    # обновляем значение коэффициентов тета

    theta0_best = theta0_best - J_theta0

    theta_best = np.subtract(theta_best, J_theta)

    htheta_best = np.array(theta_best).dot(np.transpose(X))

    # пересчитываем оценочную функцию

    J = 0

    for i in range(len(htheta_best)):

        J += (theta0_best + htheta_best[i] - data['price'][i])**2

    J /= 2 * len(htheta_best)

    # Выводим на экран J, чтобы убедиться, что оценочная функция улучшается

    # Если при выводе значения будут увеличиваться, то цикл нужно остановить и скоректировать alpha

    print(J)

    # Для примера делаем отсечение в 200 шагов, иначе слишком долго сходится

    c += 1

    if c == 200:

        break

    # значения J_theta0 и J_theta показывают, на сколько изменились theta0_best и theta_best

    # и если это изменение меньшн eps, то мы достигли нужной точности

    continue_loop = False

    for i in range(len(J_theta)):

        if abs(J_theta[i]) > eps:

            continue_loop = True

            break

    continue_loop = continue_loop or abs(J_theta0) > eps

    if not continue_loop:

        break   



print('t0 =', theta0_best)

for i in range(len(theta_best)):

    print('t', i + 1, ' = ', theta_best[i], sep="")
X1 = data[features][:len(data[features]) // 2]

X2 = data[features][len(data[features]) // 2:]

y1 = data['price'][:len(data[features]) // 2]

y2 = data['price'][len(data[features]) // 2:]
from sklearn.linear_model import LinearRegression

# Обучаем модель на наборе X1 y1

lreg = LinearRegression().fit(X1, y1)

print(lreg.score(X1, y1))

print(lreg.score(X2, y2))
# Находим значение функции J на наборе X1

y_pred1 = lreg.predict(X1)

J = 0

for i in range(len(y_pred1)):

    J += (y1[i] - y_pred1[i])**2

J /= 2 * len(y_pred1)

print(J)

# Находим значение функции J на наборе X2

y_pred2 = lreg.predict(X2)

J = 0

for i in range(len(y_pred2)):

    J += (y2[i + len(y_pred1)] - y_pred2[i])**2

J /= 2 * len(y_pred2)

print(J)
from sklearn.ensemble import GradientBoostingRegressor

# Обучаем модель на наборе X1 y1

gbr = GradientBoostingRegressor(learning_rate=0.02).fit(X1, y1)

print(gbr.score(X1, y1))

print(gbr.score(X2, y2))
# Находим значение функции J на наборе X1

y_pred_gbr1 = gbr.predict(X1)

J = 0

for i in range(len(y_pred_gbr1)):

    J += (y1[i] - y_pred_gbr1[i])**2

J /= 2 * len(y_pred_gbr1)

print(J)

# Находим значение функции J на наборе X2

y_pred_gbr2 = gbr.predict(X2)

J = 0

for i in range(len(y_pred_gbr2)):

    J += (y2[i + len(y_pred_gbr1)] - y_pred_gbr2[i])**2

J /= 2 * len(y_pred_gbr2)

print(J)
from sklearn.ensemble import AdaBoostRegressor

# Обучаем модель на наборе X1 y1

ada = AdaBoostRegressor().fit(X1, y1)

print(ada.score(X1, y1))

print(ada.score(X2, y2))
# Находим значение функции J на наборе X1

y_pred_ada1 = ada.predict(X1)

J = 0

for i in range(len(y_pred_ada1)):

    J += (y1[i] - y_pred_ada1[i])**2

J /= 2 * len(y_pred_ada1)

print(J)

# Находим значение функции J на наборе X2

y_pred_ada2 = ada.predict(X2)

J = 0

for i in range(len(y_pred_ada2)):

    J += (y2[i + len(y_pred_ada1)] - y_pred_ada2[i])**2

J /= 2 * len(y_pred_ada2)

print(J)