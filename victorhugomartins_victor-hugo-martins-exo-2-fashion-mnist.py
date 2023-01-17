# Directive pour afficher les graphiques dans Jupyter

%matplotlib inline
# Pandas : librairie de manipulation de données

# NumPy : librairie de calcul scientifique

# MatPlotLib : librairie de visualisation et graphiques

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



from sklearn import metrics

from sklearn import preprocessing

from sklearn import model_selection

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score,auc, accuracy_score



from sklearn.preprocessing import StandardScaler, MinMaxScaler



from sklearn.model_selection import train_test_split



from IPython.core.display import HTML # permet d'afficher du code html dans jupyter
from sklearn.model_selection import learning_curve

def plot_learning_curve(est, X_train, y_train) :

    train_sizes, train_scores, test_scores = learning_curve(estimator=est, X=X_train, y=y_train, train_sizes=np.linspace(0.1, 1.0, 10),

                                                        cv=5,

                                                        n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)

    train_std = np.std(train_scores, axis=1)

    test_mean = np.mean(test_scores, axis=1)

    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(8,10))

    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')

    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,color='green', linestyle='--',marker='s', markersize=5,label='validation accuracy')

    plt.fill_between(train_sizes,test_mean + test_std,test_mean - test_std,alpha=0.15, color='green')

    plt.grid(b='on')

    plt.xlabel('Number of training samples')

    plt.ylabel('Accuracy')

    plt.legend(loc='lower right')

    plt.ylim([0.6, 1.0])

    plt.show()
def plot_roc_curve(est,X_test,y_test) :

    probas = est.predict_proba(X_test)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test,probas[:, 1])

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.figure(figsize=(8,8))

    plt.title('Receiver Operating Characteristic')

    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)

    plt.legend(loc='lower right')

    plt.plot([0,1],[0,1],'r--')        # plus mauvaise courbe

    plt.plot([0,0,1],[0,1,1],'g:')     # meilleure courbe

    plt.xlim([-0.05,1.2])

    plt.ylim([-0.05,1.2])

    plt.ylabel('Taux de vrais positifs')

    plt.xlabel('Taux de faux positifs')

    plt.show
df = pd.read_csv('../input/fashionmnist-train/fashion-mnist_train.csv')
df.head()
labels = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt",

          "Sneaker","Bag","Ankle boot"]
print(labels[df.label[0]])
data_train = df.sample(frac=0.4, random_state=1)# 80% des données avec frac=0.8

data_test = df.drop(data_train.index)     # le reste des données pour le test
X_train = data_train.drop(['label'], axis=1)

y_train = data_train['label']

X_test = data_test.drop(['label'], axis=1)

y_test = data_test['label']
# On transforme la première ligne en une matrice 28x28 et on la affiche

X = df.drop(['label'], axis=1)

Y = df['label']

x1 = np.array(X)

image = x1[0].reshape(28,28)



plt.imshow(image)
n_samples = len(df.index)

image = x1.reshape(n_samples,28,28)
# et pour afficher les 50 premières images

plt.figure(figsize=(8,16))

for i in range(0,49) :

    plt.subplot(10,5,i+1)

    plt.axis('off')

    plt.imshow(image[i])

    plt.title(Y[i])
from sklearn import ensemble

rf = ensemble.RandomForestClassifier()

rf.fit(X_train, y_train)

y_rf = rf.predict(X_test)
rf_score = accuracy_score(y_test, y_rf)

print(rf_score)
pd.crosstab(y_test, y_rf, rownames=['Reel'], colnames=['Prediction'], margins=True)
print(classification_report(y_test, y_rf))
from sklearn import tree

dtc = tree.DecisionTreeClassifier()

dtc.fit(X_train,y_train)

y_dtc = dtc.predict(X_test)

print(accuracy_score(y_test, y_dtc))
pd.crosstab(y_test, y_dtc, rownames=['Reel'], colnames=['Prediction'], margins=True)
print(classification_report(y_test, y_dtc))