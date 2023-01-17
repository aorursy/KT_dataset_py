import pandas as pd 

import sklearn

from sklearn.utils import shuffle 

import warnings

warnings.filterwarnings("ignore")



wine = pd.read_csv("../input/wine-quality/winequalityN.csv")

wine = sklearn.utils.shuffle(wine)



#change type to binary

wine['type_bin'] = [0 if x == 'white' else 1 for x in wine.type]

wine.drop("type", axis= 1)



# add classification in quality

wine['quality_class'] = "Very good"

wine['quality_class'][wine.quality <= 8] = 'Good'

wine['quality_class'][wine.quality <= 7] = 'Average'

wine['quality_class'][wine.quality <= 5] = 'Bad'

wine['quality_class'][wine.quality <= 3] = 'Terrible'



print("Data loaded and variables added!")
Sum = wine.isnull().sum()

Percentage = (wine.isnull().sum()/wine.isnull().count())

values = pd.DataFrame([Sum,Percentage])

values.rename(index={0: 'Sum', 1: 'Percentage'}, inplace=True)

print(values)



wine.dropna(inplace=True)
import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt

corr_matrix = wine.corr()

dropSelf = np.zeros_like(corr_matrix)

dropSelf[np.triu_indices_from(dropSelf)] = True

sns.heatmap(corr_matrix, cmap=sns.diverging_palette(220, 10, as_cmap=True), annot=True, fmt=".2f", mask=dropSelf)

plt.title('Correlation Matrix')

plt.show()

sns.pairplot(wine, kind="scatter", hue="quality_class", palette="Set1")

plt.show()
from sklearn.linear_model import LogisticRegression

from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeRegressor

wine = pd.read_csv("../input/wine-quality/winequalityN.csv")

models = []

models.append(['LR', LogisticRegression(solver='lbfgs', multi_class='multinomial')])

models.append(['SVM', svm.SVC(decision_function_shape="ovo")])

models.append(['RF', RandomForestClassifier(n_estimators=1000, max_depth=10)])

models.append(['NN', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(150, 10))])

models.append(['KNN', KNeighborsClassifier()])

models.append(['DTC', DecisionTreeClassifier()])
from sklearn.model_selection import train_test_split

from sklearn import preprocessing 



featuresList = list(wine.columns)

featuresNotSelected = ['quality', 'quality_class','type']

features = list(set(featuresList).difference(set(featuresNotSelected)))

X_stand = preprocessing.scale(wine[features])

X_normal = preprocessing.normalize(X_stand)

y = wine['quality_class']

train_X, val_X, train_y, val_y = train_test_split(X_normal, y)
from sklearn.metrics import accuracy_score

import pandas as pd

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

results = []



for name, wine_model in models:

    wine_model.fit(train_X, train_y)

    pred = wine_model.predict(val_X)

    acc = accuracy_score(val_y, pred)

    precision = precision_score(val_y, pred, average=None)

    recall = recall_score(val_y, pred, average= None)

    error_Rate = 1- acc

    print('Model tested: {}'.format(name))

    print('Accuracy= {}'.format(acc))

    print('Error Rate= {}'.format(error_Rate))

    print('Recall Rate= {}'.format(recall))

    print("Precision Rate: {}".format(precision))

    print()

    results.append((name, acc, precision, error_Rate, recall))


