# numpy
import numpy
print('numpy: %s' % numpy.__version__)
# scipy
import scipy
print('scipy: %s' % scipy.__version__)
# matplotlib
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
# pandas
import pandas
print('pandas: %s' % pandas.__version__)
# scikit-learn
import sklearn
print('sklearn: %s' % sklearn.__version__)
from pandas import read_csv
train_file = '../input/higgs-boson-data/higgs_train_10k.csv'
test_file = '../input/higgs-boson-data/higgs_test_5k.csv'
names = [
    'response',
    'x1',
    'x2',
    'x3',
    'x4',
    'x5',
    'x6',
    'x7',
    'x8',
    'x9',
    'x10',
    'x11',
    'x12',
    'x13',
    'x14',
    'x15',
    'x16',
    'x17',
    'x18',
    'x19',
    'x20',
    'x21',
    'x22',
    'x23',
    'x24',
    'x25',
    'x26',
    'x27',
    'x28']
train_data = read_csv(train_file, names=names)
test_data = read_csv(test_file, names=names)
print(train_data.shape)
print(test_data.shape)
# give the peek into the dataset
peek = train_data.head(20)
print(peek)
# datatype of each feataure
types = train_data.dtypes
print(types)
#base statistics for data
from pandas import set_option
set_option('display.width', 100)
set_option('precision', 5)
description = train_data.describe()
print(description)
# class distribution for train and test
train_data_class = train_data.groupby('response').size()
print(train_data_class)
test_data_class = test_data.groupby('response').size()
print(test_data_class)
# pearsons correlation to understand feature independence
correlations = train_data.corr(method='pearson')
print(correlations)
# visualization of correlations
import matplotlib.pyplot as pyplot
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,29,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
pyplot.rcParams['figure.figsize'] = (27,27)
pyplot.show()
# visualization using pyplot and histograms of training and testing data
from matplotlib import pyplot
pyplot.rcParams['figure.figsize'] = (13,13)
train_data.hist()
pyplot.show()
test_data.hist()
pyplot.show()
# boxplot visualization of train and test data
pyplot.rcParams['figure.figsize'] = (12,12)
train_data.plot(kind='box', subplots=True, layout=(6,6), sharex=False, sharey=False)
pyplot.show()
test_data.plot(kind='box', subplots=True, layout=(6,6), sharex=False, sharey=False)
pyplot.rcParams['figure.figsize'] = (12,12)
pyplot.show()
# train data
train_array = train_data.values
# separate array into input and output variables
X_train = train_array[:,1:28]
y_train = train_array[:,0]
# test data
test_array = test_data.values
# separate array into input and output variables
X_test = test_array[:,1:28]
y_test = test_array[:,0]
from time import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter

from sklearn import manifold
methods = ['standard', 'ltsa', 'hessian', 'modified']
labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']

n_neighbors = 10
n_components = 2
color=y_train

for i, method in enumerate(methods):
    t0 = time()
    Ytransformed = manifold.Isomap(n_neighbors, n_components).fit_transform(X_train)
    t1 = time()
    print("Isomap: %.2g sec" % (t1 - t0))
    ax = fig.add_subplot(257)
    plt.scatter(Ytransformed[:, 0], Ytransformed[:, 1],c=color, cmap=plt.cm.Spectral)
    plt.title(labels[i])
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.show()


t0 = time()
mds = manifold.MDS(n_components, max_iter=100, n_init=1)
Ytransformed = mds.fit_transform(X_train)
t1 = time()
print("MDS: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(258)
plt.scatter(Ytransformed[:, 0], Ytransformed[:, 1], c=color,cmap=plt.cm.Spectral)
plt.title("MDS (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.show()


t0 = time()
se = manifold.SpectralEmbedding(n_components=n_components,
                                n_neighbors=n_neighbors)
Ytransformed = se.fit_transform(X_train)
t1 = time()
print("SpectralEmbedding: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(259)
plt.scatter(Ytransformed[:, 0], Ytransformed[:, 1], c=color,cmap=plt.cm.Spectral)
plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')
plt.show()

t0 = time()
tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
Ytransformed = tsne.fit_transform(X_train)
t1 = time()
print("t-SNE: %.2g sec" % (t1 - t0))
ax = fig.add_subplot(2, 5, 10)
plt.scatter(Ytransformed[:, 0], Ytransformed[:, 1], c=color,cmap=plt.cm.Spectral)
plt.title("t-SNE (%.2g sec)" % (t1 - t0))
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

plt.show()
# Feature decomposition  with PCA
from sklearn.decomposition import PCA
# feature extraction
pca = PCA(n_components=2)
fit = pca.fit(X_train)
projected = pca.fit_transform(X_train)

pyplot.scatter(projected[:, 0], projected[:, 1],
               c=y_train, edgecolor='none', alpha=0.5)
pyplot.xlabel('PCA component 1')
pyplot.ylabel('PCA component 2')
pyplot.rcParams['figure.figsize'] = (8, 8)
pyplot.colorbar()
pyplot.show()
pca = PCA(n_components=25)
fit = pca.fit(X_train)
pyplot.plot(numpy.cumsum(fit.explained_variance_ratio_))
pyplot.xlabel('number of components')
pyplot.ylabel('cumulative explained variance')
pyplot.show()
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
from sklearn import preprocessing
from sklearn.feature_selection import chi2

min_max_scaler = preprocessing.MinMaxScaler()
scaler = min_max_scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
chi2_score = chi2(X_train_scaled, y_train)[0]
features = [
    'x1',
    'x2',
    'x3',
    'x4',
    'x5',
    'x6',
    'x7',
    'x8',
    'x9',
    'x10',
    'x11',
    'x12',
    'x13',
    'x14',
    'x15',
    'x16',
    'x17',
    'x18',
    'x19',
    'x20',
    'x21',
    'x22',
    'x23',
    'x24',
    'x25',
    'x26',
    'x27',
    'x28']
fscores = zip(features, chi2_score)
wchi2 = sorted(fscores, key=lambda x: x[1], reverse=True)
scores_labels = numpy.asarray(wchi2)
print(scores_labels)
label = [row[0] for row in scores_labels]
print(label)
score = [row[1] for row in scores_labels]
print(score)
y_pos = numpy.arange(len(score))
yrange = range(len(score))
print(yrange)
# perform grid search to find the best parameter for Logistic Regression,
# Perceptron, Naive Bayes, LDA algorithm
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

# using roc AUC as scoring
scoring = 'accuracy'

# Naive Bayes
naiveBayes = GaussianNB()
nbscore = cross_val_score(naiveBayes, X_train, y_train, cv=3, scoring=scoring)
print('Naive Bayes CV score =', np.mean(nbscore))


# penalty
penalties = numpy.array(['l1', 'l2'])
# C for logistic regression
c_values = numpy.array([1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001])
# max iteration
iters = numpy.array([100, 150])
LR_param_grid = {'penalty': penalties, 'C': c_values, 'max_iter': iters}

# logistic regression as algorithm
gridLogisticRegression = LogisticRegression()
# Using GridSearchCV on Training Data for LR
grid = GridSearchCV(
    estimator=gridLogisticRegression,
    param_grid=LR_param_grid,
    scoring=scoring)
grid.fit(X_train, y_train)
print('LR CVScore ', grid.best_score_)
print('LR Penalty', grid.best_estimator_.penalty)
print('LR C', grid.best_estimator_.C)
print('LR Max Iterations', grid.best_estimator_.max_iter)


# Perceptron
# Using GridSearchCV on Training Data for perceptron
# alphas
alphas = numpy.array([0.001, 0.0001, 0.00001, 0.000001])
# iterations
pereptorn_param_grid = {'alpha': alphas, 'max_iter': iters}
grid = GridSearchCV(
    estimator=Perceptron(),
    param_grid=pereptorn_param_grid,
    scoring=scoring)
grid.fit(X_train, y_train)
print('Perceptron CVScore ', grid.best_score_)
print('Perceptron alpha', grid.best_estimator_.alpha)
print('Perceptron Max Iterations', grid.best_estimator_.max_iter)

# LDA
tols = numpy.array([0.001, 0.00001, 0.001])
lda_param_grid = {'tol': tols}
grid = GridSearchCV(
    estimator=LinearDiscriminantAnalysis(),
    param_grid=lda_param_grid,
    scoring=scoring)
grid.fit(X_train, y_train)
print('LDA CVScore ', grid.best_score_)
print('LDA tol', grid.best_estimator_.tol)
from sklearn.svm import SVC
import numpy
# gamma parameter in SVM
gammas = numpy.array([1, 0.1, 0.01, 0.001])
# C for logistic regression
c_values = numpy.array([100, 1, 0.1, 0.01])
svm_param_grid = {'gamma': gammas, 'C': c_values}
svm = SVC(kernel='rbf')
scoring = 'accuracy'
grid = GridSearchCV(estimator=svm, param_grid=svm_param_grid, scoring=scoring)
grid.fit(X_train, y_train)
print(grid.best_score_)
print(grid.best_estimator_.gamma)
print(grid.best_estimator_.C)
# Modified the Code for changes
# Original Authors: Robert McGibbon, Joel Nothman, Guillaume Lemaitre


from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing

# transform the features using MinMaxScaler as many are negatives
min_max_scaler = preprocessing.MinMaxScaler()
scaler = min_max_scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)

print(__doc__)

pipe = Pipeline([
    ('reduce_dim', PCA()),
    ('classify', LogisticRegression())
])

N_FEATURES_OPTIONS = [10, 15, 20]
C_OPTIONS = [0.001, 0.1, 1, 10, 100, 1000]
max_iter_OPTIONS = [100, 150]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=10)],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__max_iter':max_iter_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS,
        'classify__max_iter':max_iter_OPTIONS
    },
]
reducer_labels = ['PCA', 'KBest(chi2)']

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
grid.fit(X_train_scaled, y_train)

mean_scores = np.array(grid.cv_results_['mean_test_score'])
# scores are in the order of param_grid iteration, which is alphabetical
mean_scores = mean_scores.reshape(len(C_OPTIONS), -1, len(N_FEATURES_OPTIONS))
# select score for best C
mean_scores = mean_scores.max(axis=0)
bar_offsets = (np.arange(len(N_FEATURES_OPTIONS)) *
               (len(reducer_labels) + 1) + .5)

plt.figure()
COLORS = ['tomato', 'darkolivegreen', 'lightsteelblue']
for i, (label, reducer_scores) in enumerate(zip(reducer_labels, mean_scores)):
    plt.bar(bar_offsets + i, reducer_scores, label=label, color=COLORS[i])

plt.title("Comparing feature reduction techniques")
plt.xlabel('Reduced number of features')
plt.xticks(bar_offsets + len(reducer_labels) / 2, N_FEATURES_OPTIONS)
plt.ylabel('Classification accuracy')
plt.ylim((0, 1))
plt.legend(loc='upper left')
plt.show()
# learning curves
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve


def plot_learning_curve(estimator, name, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title('Learning Curves for ' + name)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("No. Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="b")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


estimator = LogisticRegression(C=0.1, penalty='l1', max_iter=100)
plot_learning_curve(estimator, 'Tuned Logistic Regression', X_train, y_train)
plt.rcParams['figure.figsize'] = (7, 7)
plt.show()
estimator = SVC(C=100, gamma=0.01, kernel='rbf')
plot_learning_curve(estimator, 'Tuned SVM', X_train, y_train)
plt.rcParams['figure.figsize'] = (7, 7)
plt.show()