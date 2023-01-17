#импортирую библиотеки

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split 

from sklearn.linear_model import LinearRegression

from sklearn.dummy import DummyRegressor

df = pd.read_csv("../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.shape
df.head()
df.describe()
#имя столбцов

column = df.columns

names = column[:-1]
#загружаю данные с 11 признаками/качествами

features = np.array(df[column[:-1]])

#ну и отдельно оценку вина = целевую функцию

target = np.array(df[column[-1]])

#и делю на тренировочные и тестовые данные

features_train, features_test, target_train,target_test = train_test_split(features,target,test_size=0.1,random_state=1)

#объект базовой регрессионной модели

dummy_regression = DummyRegressor(strategy='mean')

#тренировка базовой модели

model = dummy_regression.fit(features_train,target_train)

#прогноз целевой функции на тестовых признаках

target_predicted = model.predict(features_test)
#сделала оценку коэф-том детерминации

dummy_r2 = model.score(features_test,target_test)

print('Оценка коэффициента детерминации:',dummy_r2)

#строю график

def plot(regression_coef):

    plt.title("модельные коэффициенты")

    plt.bar(range(features.shape[1]),regression_coef)

    plt.xticks(range(features.shape[1]),names,rotation=90)

    plt.show()
#создаю объект линейной регресии

lin_regression = LinearRegression()

#подгонка линейной регрессии

model = lin_regression.fit(features_train,target_train)

#прогноз целевой функции на тестовых признаках

target_predicted = model.predict(features_test)

#оценю модель коэф-ом детерминации

lin_r2 = model.score(features_test,target_test)

print('Оценка коэффициента  детерминации:',lin_r2)
#эффект единичного изменения на вектор целей, модельные коэффициенты

lin_regression_coef = model.coef_

plot(lin_regression_coef)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

import seaborn as sns
#оформляю матрицу ошибок

def matrix_plot():

    matrix = confusion_matrix(target_test,target_predicted)

    matrix_name = sorted(set(target_predicted) | set(target_test)) #создание осей для х и у - это оценки

    df = pd.DataFrame(matrix,index=matrix_name,columns=matrix_name)

    sns.heatmap(df,annot=True,cbar=None,cmap="Blues")

    plt.title("Матрица ошибок")

    plt.ylabel("Истинный класс")

    plt.xlabel("Предсказанный класс")

    plt.show()
log_regression = LogisticRegression(random_state=0,class_weight="balanced")

model = log_regression.fit(features_train,target_train)

target_predicted = model.predict(features_test) #прогноз целевой функции на тестовых признаках

log_r2 = model.score(features_test,target_test) #оценка модели коэф-ом детерминации

print('Оценка коэффициента  детерминации:',log_r2)

matrix_plot()
from sklearn.tree import DecisionTreeRegressor
decisiontree_regression = DecisionTreeRegressor(random_state=0) 

model = decisiontree_regression.fit(features_train,target_train)

target_predicted = model.predict(features_test)

decisiontree_r2 = model.score(features_test,target_test)

print('Оценка коэффициента  детерминации:',decisiontree_r2)

matrix_plot()
from sklearn.ensemble import RandomForestRegressor

def importance(model_x):

    importances = model_x.feature_importances_

    #строю график важность признаков

    indices = np.argsort(importances)[::-1] #сортировка в нисходящем порядке

    names_in_plot = [names[i] for i in indices] #потом перераспределяю имена

    plt.title("Важность признаков") 

    plt.bar(range(features.shape[1]),importances[indices])

    plt.xticks(range(features.shape[1]),names_in_plot,rotation=90)

    plt.show()
randomforest_regression = RandomForestRegressor(random_state=0,n_jobs=-1) #объект 

model = randomforest_regression.fit(features_train,target_train)

randomforest_regression = model.predict(features_test)

randomforest_r2 = model.score(features_test,target_test)

print('Оценка коэффициента  детерминации:',randomforest_r2)

importance(model)
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier

from sklearn.neighbors import RadiusNeighborsClassifier,KNeighborsClassifier

from sklearn.dummy import DummyClassifier

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import learning_curve
def classifier(method):

    model = method.fit(features_train,target_train)

    importance(model)

    print('Точность предсказаний трен', accuracy_score(target_train,model.predict(features_train)))

    print('Точность предсказаний тест', accuracy_score(target_test,model.predict(features_test)))

    train_sizes, train_scores, test_scores = learning_curve (method,features_train,target_train,cv=10,scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.01,1.0,50))

    train_mean = np.mean(train_scores,axis=1)

    test_std = np.std(test_scores,axis=1)

    test_mean = np.mean(test_scores,axis=1)

    plt.plot(train_sizes,train_mean,'--',color="#111111",label="Тренировочная оценка")

    plt.plot(train_sizes,test_mean,color="#111111",label="Перекрестно-проверочная оценка")

    plt.fill_between(train_sizes,test_mean-test_std,test_mean+test_std,color="#DDDDDD")

    plt.title("Кривая заучивания")

    plt.xlabel("Размер тренировочного набора")

    plt.ylabel("Оценка точности")

    plt.grid()

    plt.legend(loc="best")

    plt.tight_layout()

    plt.show()
decisiontree = DecisionTreeClassifier(random_state=0)

randomforest = RandomForestClassifier(random_state=0,n_jobs=-1)

ABS = AdaBoostClassifier(random_state=0)

gaussonnb = GaussianNB()
classifier(decisiontree)
classifier(randomforest)
classifier(ABS)