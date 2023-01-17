# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/iris/Iris.csv')



print(data.dtypes)

data.head()
data.drop('Id', axis=1, inplace=True)



print(data.count())

data['Species'].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns



fig, axs = plt.subplots(3, 2, figsize=(13, 10))

plt.subplots_adjust(hspace=0.5)



setosa = data.loc[data['Species'] == 'Iris-setosa']

versicolor = data.loc[data['Species'] == 'Iris-versicolor']

verginica = data.loc[data['Species'] == 'Iris-virginica']



def plot_data(axs, xcolumn, ycolumn, title, xlabel, ylabel, classes=(setosa, versicolor, verginica), coord=(0, 0)):

    ax = axs[coord[0], coord[1]]

    ax.plot(classes[0][xcolumn], classes[0][ycolumn], 'ob', label='Setosa')

    ax.plot(classes[1][xcolumn], classes[1][ycolumn], 'og', label='Versicolor')

    ax.plot(classes[2][xcolumn], classes[2][ycolumn], 'or', label='Verginica')

    ax.set_title(title)

    ax.set_xlabel(xlabel)

    ax.set_ylabel(ylabel)

    

common = dict({

    'xlabel': 'Length',

    'ylabel': 'Width'

})



def plot_dataset():

    common = dict({

        'xlabel': 'Length',

        'ylabel': 'Width'

    })



    plot_data(axs, xcolumn='SepalLengthCm', ycolumn='SepalWidthCm',

              title='Sepal length vs width (cm)', coord=(0, 0), **common)



    plot_data(axs, xcolumn='PetalLengthCm', ycolumn='PetalWidthCm',

              title='Petal length vs width (cm)', coord=(0, 1), **common)



    plot_data(axs, xcolumn='SepalLengthCm', ycolumn='PetalLengthCm',

              title='Sepal length vs petal length (cm)', coord=(1, 0), **common)



    plot_data(axs, xcolumn='SepalLengthCm', ycolumn='PetalWidthCm',

              title='Sepal length vs petal width (cm)', coord=(1, 1), **common)



    plot_data(axs, xcolumn='SepalWidthCm', ycolumn='PetalLengthCm',

              title='Sepal width vs petal length (cm)', coord=(2, 0), **common)



    plot_data(axs, xcolumn='SepalWidthCm', ycolumn='PetalWidthCm',

              title='Sepal width vs petal width (cm)', coord=(2, 1), **common)



    

plot_dataset()

plt.legend(loc=(1.1, 3.66))

plt.show()
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

plt.subplots_adjust(hspace=0.3)



common = dict({

    'hist': True,

    'kde': True,

    'norm_hist': True

})



sns.distplot(setosa['SepalLengthCm'], ax=axs[0, 0], label='Setosa', **common)

sns.distplot(versicolor['SepalLengthCm'], ax=axs[0, 0], label='Versicolor', **common)

sns.distplot(verginica['SepalLengthCm'], ax=axs[0, 0], label='Verginica', **common)

axs[0, 0].set_title('SepalLengthCm')



sns.distplot(setosa['SepalWidthCm'], ax=axs[0, 1], label='Setosa', **common)

sns.distplot(versicolor['SepalWidthCm'], ax=axs[0, 1], label='Versicolor', **common)

sns.distplot(verginica['SepalWidthCm'], ax=axs[0, 1], label='Verginica', **common)

axs[0, 1].set_title('SepalWidthCm')



sns.distplot(setosa['PetalLengthCm'], ax=axs[1, 0], label='Setosa', **common)

sns.distplot(versicolor['PetalLengthCm'], ax=axs[1, 0], label='Versicolor', **common)

sns.distplot(verginica['PetalLengthCm'], ax=axs[1, 0], label='Verginica', **common)

axs[1, 0].set_title('PetalLengthCm')



sns.distplot(setosa['PetalWidthCm'], ax=axs[1, 1], label='Setosa', **common)

sns.distplot(versicolor['PetalWidthCm'], ax=axs[1, 1], label='Versicolor', **common)

sns.distplot(verginica['PetalWidthCm'], ax=axs[1, 1], label='Verginica', **common)

axs[1, 1].set_title('PetalWidthCm')



plt.legend(loc=(1.1, 2.1))

plt.show()
X = data.drop(['Species'], axis=1)

corr = X.corr()



sns.heatmap(corr, annot=True)
fig, ax = plt.subplots(2, 2, figsize=(10, 10))



feat = np.array(X.columns).reshape((2, 2))



for i in range(2):

    for j in range(2):

        sns.boxplot(x='Species', y=feat[i][j], data=data, ax=ax[i][j])
from pandas.plotting import parallel_coordinates, radviz



plt.figure(figsize=(15, 6))

parallel_coordinates(data, 'Species')
plt.figure(figsize=(15, 6))

radviz(data, 'Species')
from sklearn.model_selection import train_test_split



# X = data.drop(['Species', 'Id'], axis=1)

y = data['Species']



X.shape, y.shape
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score



def evaluate(model, X, y, k=4):

    """

    Evaluate model performance with k-fold cross validation

    """

    kf = KFold(n_splits=k)

    predictions = []

    

    for train_index, test_index in kf.split(X):

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]

        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        

        model.fit(X_train, y_train)

        

        prediction = model.predict(X_test)

        score = accuracy_score(y_test, prediction)

        predictions.append(score)

        

    return np.array(predictions).sum() / len(predictions)
from sklearn.neighbors import KNeighborsClassifier



for weights in ['uniform', 'distance']:

    print('weights =', weights)

    

    for k in [1, 3, 5, 7, 9, 11]:

        model = KNeighborsClassifier(n_neighbors=k, weights=weights)

        print('k = {}, {}'.format(k, evaluate(model, X, y)))



    print('')
from sklearn.svm import SVC



common = dict({"random_state": 42})



svm_linear = SVC(kernel='linear', C=1.0, **common)

svm_rbf = SVC(kernel='rbf', C=3.0, gamma=1.5, **common)

svm_poly = SVC(kernel='poly', C=1.0, gamma=0.1, **common)



print('SVM linear kernel, ', evaluate(svm_linear, X, y))

print('SVM rbf kernel, ', evaluate(svm_rbf, X, y))

print('SVM poly kernel, ', evaluate(svm_poly, X, y))
from sklearn.linear_model import LogisticRegression



scores = []



cs = np.linspace(0.1, 5.0, 10)



for C in cs:

    model = LogisticRegression(penalty='l2', C=C, multi_class='multinomial', solver='newton-cg')

    score = evaluate(model, X, y)

    scores.append(score)

    

    print('C = {}, acc = {}'.format(C, score))



plt.plot(scores)

plt.show()
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier(

    random_state=0,

    criterion='entropy',

    max_depth=2,

    max_features=2)



evaluate(dt, X, y)
from sklearn.ensemble import AdaBoostClassifier



adaboost = AdaBoostClassifier(

    random_state=42,

    base_estimator=None, 

    n_estimators=50, 

    learning_rate=1.0)



evaluate(adaboost, X, y)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(

    random_state=42,

    n_estimators=50,

    max_features=2)



evaluate(rf, X, y)
from sklearn.ensemble import BaggingClassifier



bag = BaggingClassifier(

    random_state=42,

    base_estimator=None,

    n_estimators=2)



evaluate(bag, X, y)
import math



scaled_ev = lambda m, X: math.exp(evaluate(m, X, y) * 10)



knn = KNeighborsClassifier(n_neighbors=1, weights='distance')

logres = LogisticRegression(penalty='l2', C=2.82, multi_class='multinomial', solver='newton-cg')



def compare_models(X, ev=scaled_ev):

    results = [ev(knn, X), ev(svm_linear, X), ev(logres, X), ev(dt, X), ev(rf, X), ev(adaboost, X), ev(bag, X)]

    titles = ('KNN', 'SVM (linear kernel)', 'LogisticRegression', 'DecisionTree', 'RandomForest', 'AdaBoost', 'Bagging')

    indexes = np.arange(7)

    width = 0.5



    plt.figure(figsize=(14, 5))



    plt.bar(indexes, results, width)

    plt.title('Comparison of model performance')

    plt.ylabel('Accuracy (scaled)')

    plt.xticks(indexes, titles)

    

compare_models(X)

plt.show()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



y_encoded = le.fit_transform(y)



def plot_decision_boundaries(X, y, ax, title, model):

    reduced_data = X.iloc[:, :2]

    model.fit(reduced_data, y)

    

    h = .5

    x_min, x_max = reduced_data.iloc[:, 0].min() - 1, reduced_data.iloc[:, 0].max() + 1

    y_min, y_max = reduced_data.iloc[:, 1].min() - 1, reduced_data.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))



    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    



    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1

    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                         np.arange(y_min, y_max, 0.1))



    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)



    ax.contourf(xx, yy, Z, alpha=0.4)

    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, alpha=0.8)

    ax.set_title(title)

    ax.set_xlabel('Sepal length (cm)')

    ax.set_ylabel('Sepal width (cm)')

    

    return ax



fig, axs = plt.subplots(2, 3, figsize=(16, 8))

plt.subplots_adjust(hspace=0.3)



ax_models = [

    [('DecisionTree', dt), ('LogisticRegression', logres), ('KNearestNeighbors', knn)],

    [('SVM (linear kernel)', svm_linear), ('AdaBoost', adaboost), ('Bagging', bag)]

]



for i in range(2):

    for j in range(3):

        title, model = ax_models[i][j]

        plot_decision_boundaries(X, y_encoded, axs[i, j], title, model)
normalize = lambda x: (x - x.mean()) / x.std()

regular_ev = lambda m, X: evaluate(m, X, y)



# Normalize

X_norm = X.copy()

X_norm['SepalWidthCm'] = normalize(X_norm['SepalWidthCm'])

X_norm['SepalLengthCm'] = normalize(X_norm['SepalLengthCm'])

X_norm['PetalWidthCm'] = normalize(X_norm['PetalWidthCm'])

X_norm['PetalLengthCm'] = normalize(X_norm['PetalLengthCm'])



compare_models(X_norm)

plt.show()
from keras.optimizers import RMSprop

from keras.models import Model

from keras.layers import Input, Dense



def create_model():

    x = Input(shape=(4,))

    h_1 = Dense(16, activation='relu')(x)

    output = Dense(3, activation='softmax')(h_1)

    

    return Model(x, output)



model = create_model()



model.compile(RMSprop(learning_rate=0.15), loss='categorical_crossentropy', metrics=['acc'])



y_dummies = pd.get_dummies(y)



train_X, test_X, train_y, test_y = train_test_split(X_norm, y_dummies, test_size=0.25)



history = model.fit(train_X, train_y, epochs=10, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_X, test_y)



print('\ntest loss = {}, test accuracy = {}'.format(test_loss, test_acc))



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))



ax1.plot(history.history['loss'], label='Train loss')

ax1.plot(history.history['val_loss'], label='Validation loss')

ax1.legend()



ax2.plot(history.history['acc'], label='Train accuracy')

ax2.plot(history.history['val_acc'], label='Validation accuracy')

ax2.legend()



plt.show()