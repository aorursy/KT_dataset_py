import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn import preprocessing

%matplotlib inline
# load csv data

iris = pd.read_csv("../input/Iris.csv")

iris.info()
# change the target from string to int

iris['Species'] = iris['Species'].astype('category')

iris['target'] = pd.Categorical.from_array(iris['Species']).codes

iris['target'].value_counts()
# scatter plot using pandas



ax = iris[iris.Species=='Iris-setosa'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 

                                                    color='red', label='setosa')

iris[iris.Species=='Iris-versicolor'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 

                                                color='green', label='versicolor', ax=ax)

iris[iris.Species=='Iris-virginica'].plot.scatter(x='SepalLengthCm', y='SepalWidthCm', 

                                                color='blue', label='virginica', ax=ax)

ax.set_title("scatter")
# scatter plot using plotly

import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()
trace_0 = go.Scatter(

    x = iris[iris.Species=='Iris-setosa']['SepalLengthCm'],

    y = iris[iris.Species=='Iris-setosa']['SepalWidthCm'],

    mode = 'markers',

    marker = dict(

        size = 10,

        symbol = 'x'

    ),

    name = 'setosa'

)



trace_1 = go.Scatter(

    x = iris[iris.Species=='Iris-versicolor']['SepalLengthCm'],

    y = iris[iris.Species=='Iris-versicolor']['SepalWidthCm'],

    mode = 'markers',

    marker = dict(

        size = 10,

        symbol = 'square-dot'

    ),

    name = 'versicolor'

)



trace_2 = go.Scatter(

    x = iris[iris.Species=='Iris-virginica']['SepalLengthCm'],

    y = iris[iris.Species=='Iris-virginica']['SepalWidthCm'],

    mode = 'markers',

    marker = dict(

        size = 10,

        symbol = "circle"

    ),

    name = 'virginica'

)

layout = dict(

    title = 'Scatter Plot',

    xaxis = dict(

        title = 'SepalLengthCm'

    ),

    yaxis = dict(

        title = 'SepalWidthCm'

    )

)

data = [trace_0, trace_1, trace_2]

fig1 = dict(data=data, layout=layout)

iplot(fig1, filename='scatter')
# Paired plot using seaborn

sns.set()

sns.pairplot(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']],

             hue="Species", diag_kind="kde")
# kdeplot using seaborn

sns.set(style='darkgrid')

setosa = iris.query("Species=='Iris-setosa'")

versicolor = iris.query("Species=='Iris-versicolor'")

virginica = iris.query("Species=='Iris-virginica'")

f, ax = plt.subplots(figsize=(8, 8))

# ax.set_aspect('equal')

ax = sns.kdeplot(setosa.SepalWidthCm, setosa.SepalLengthCm, cmap="Greens",

          shade=True, shade_lowest=False)

ax = sns.kdeplot(versicolor.SepalWidthCm, versicolor.SepalLengthCm, cmap="Reds",

          shade=True, shade_lowest=False)

ax = sns.kdeplot(virginica.SepalWidthCm, virginica.SepalLengthCm, cmap="Blues",

          shade=True, shade_lowest=False)



red = sns.color_palette("Reds")[-2]

blue = sns.color_palette("Blues")[-2]

green = sns.color_palette("Greens")[-2]

ax.text(2.5, 8.2, "virginica", size=16, color=blue)

ax.text(1.8, 6.5, "versicolor", size=16, color=red)

ax.text(3.8, 4.5, "setosa", size=16, color=green)
# PCA and Lda

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



data = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

target = iris['target'].values



pca = PCA(n_components=2)

iris['pca1'] = pca.fit(data).transform(data)[:, 0]

iris['pca2'] = pca.fit(data).transform(data)[:, 1]



lda = LDA(n_components=2)

iris['lda1'] = lda.fit(data, target).transform(data)[:, 0]

iris['lda2'] = lda.fit(data, target).transform(data)[:, 1]
# kdeplot after pca and lda

setosa = iris.query("Species=='Iris-setosa'")

versicolor = iris.query("Species=='Iris-versicolor'")

virginica = iris.query("Species=='Iris-virginica'")

f, ax = plt.subplots(figsize=(8, 8))

# ax.set_aspect('equal')

plt.figure(1)

ax = sns.kdeplot(setosa.pca1, setosa.pca2, cmap="Greens",

          shade=True, shade_lowest=False)

ax = sns.kdeplot(versicolor.pca1, versicolor.pca2, cmap="Reds",

          shade=True, shade_lowest=False)

ax = sns.kdeplot(virginica.pca1, virginica.pca2, cmap="Blues",

          shade=True, shade_lowest=False)



red = sns.color_palette("Reds")[-2]

blue = sns.color_palette("Blues")[-2]

green = sns.color_palette("Greens")[-2]

ax.text(2, 1.0, "virginica", size=16, color=blue)

ax.text(-1, 1.5, "versicolor", size=16, color=red)

ax.text(-2.5, 0.5, "setosa", size=16, color=green)

ax.set_title('PCA')



plt.figure(2)

ax1 = sns.kdeplot(setosa.lda1, setosa.lda2, cmap="Greens",

          shade=True, shade_lowest=False)

ax1 = sns.kdeplot(versicolor.lda1, versicolor.lda2, cmap="Reds",

          shade=True, shade_lowest=False)

ax1 = sns.kdeplot(virginica.lda1, virginica.lda2, cmap="Blues",

          shade=True, shade_lowest=False)



red = sns.color_palette("Reds")[-2]

blue = sns.color_palette("Blues")[-2]

green = sns.color_palette("Greens")[-2]

ax1.text(7, 1.0, "virginica", size=16, color=blue)

ax1.text(-1, 1.5, "versicolor", size=16, color=red)

ax1.text(-5.8, 0.5, "setosa", size=16, color=green)

ax1.set_title('LDA')
# SVM before LDA

from sklearn import svm

X = iris[['SepalLengthCm', 'SepalWidthCm']].values

y = iris['target'].values



h = 0.02

c = 1.0



# fit model

svc_lin = svm.SVC(kernel='linear', C=c).fit(X, y)

svc_rbf = svm.SVC(kernel='rbf', C=c, gamma=0.7).fit(X, y)

svc_poly = svm.SVC(kernel='poly', C=c, degree=3).fit(X, y)



svc = svm.LinearSVC(C=c).fit(X, y)



# create a mesh to plot in

x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1

y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))



titles = ['SVC with linear model',

          'SVC with rbf model',

          'SVC with poly model',

          'LinearSVC (linear model)']





plt.figure(3)

for i, clf in enumerate((svc_lin, svc_rbf, svc_poly, svc)):

    plt.subplot(2, 2, i+1)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)



    plt.scatter(X[:, 0], X[:, 1], c=y*100, cmap=plt.cm.Paired)

    plt.xlabel('Sepal length')

    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())



    plt.xticks(())

    plt.yticks(())

    plt.title(titles[i])

# SVM after LDA

from sklearn import svm

X = iris[['lda1', 'lda2']].values

y = iris['target'].values



h = 0.02

c = 1.0



# fit model

svc_lin = svm.SVC(kernel='linear', C=c).fit(X, y)

svc_rbf = svm.SVC(kernel='rbf', C=c, gamma=0.7).fit(X, y)

svc_poly = svm.SVC(kernel='poly', C=c, degree=3).fit(X, y)



svc = svm.LinearSVC(C=c).fit(X, y)



# create a mesh to plot in

x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1

y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))



titles = ['SVC with linear model',

          'SVC with rbf model',

          'SVC with poly model',

          'LinearSVC (linear model)']



plt.figure(3)

for i, clf in enumerate((svc_lin, svc_rbf, svc_poly, svc)):

    plt.subplot(2, 2, i+1)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)



    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)



    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)



    plt.scatter(X[:, 0], X[:, 1], c=y*100, cmap=plt.cm.Paired)

    plt.xlabel('lda1')

    plt.ylabel('lda2')

    plt.xlim(xx.min(), xx.max())

    plt.ylim(yy.min(), yy.max())



    plt.xticks(())

    plt.yticks(())

    plt.title(titles[i])
# make a complete Neural Network to classify the iris dataset

import tensorflow as tf

import numpy as np



from sklearn.model_selection import train_test_split



# first divide the dataset into train(130) dataset and test(20) dataset randomly



data = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values

target = iris['target'].values



train_data, test_data, train_target, test_target = train_test_split(

     data, target, test_size=20.0/150, random_state=1

)



train_target = np.int64(train_target)

test_target = np.int64(test_target)



feature_columns = [tf.contrib.layers.real_valued_column('', dimension=4)]



# Build 3 layer DNN with 10, 20, 10 units respectively

classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,

                                            hidden_units=[10, 20, 10],

                                            n_classes=3,

                                            model_dir="./iris_model")



# Fit model

classifier.fit(x=train_data,

               y=train_target,

               steps=2000)



accuracy_score = classifier.evaluate(x=test_data,

                                     y=test_target)["accuracy"]



print('Accuracy: {0:f}'.format(accuracy_score))



# Classify two new flower samples

new_samples = test_data



y = list(classifier.predict(new_samples, as_iterable=True))



print('Predictions: {}'.format(str(y)))
test_data, test_target