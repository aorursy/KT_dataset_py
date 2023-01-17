import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler,normalize,label_binarize

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score,confusion_matrix, roc_curve, auc, f1_score, precision_score,recall_score, roc_auc_score

from time import time

df = pd.read_csv('../input/3wine-classification-dataset/wine.csv')

df.head()
sns.set(style="ticks", color_codes=True)

sns.pairplot(df,hue="Wine", markers=["o", "s", "D"])
y = df.iloc[:,0]

x = df.iloc[:,1:]



sc = StandardScaler()

x = sc.fit_transform(x)



x_train,x_test,y_train,y_test = train_test_split(x,y,train_size = 0.7,random_state = 42)

warnings.filterwarnings('ignore')

def evaluateClassifier(x,y,y_pred,y_score):

    cm = pd.DataFrame(

        confusion_matrix(y, y_pred),

        columns=['Predicted Wine 1', 'Predicted Wine 2','Predicted Wine 3'],

        index=['True Wine 1', 'True Wine 2','True Wine 3']

    )

    print('\nConfusion Matrix: \n')

    sns.set(font_scale=1.4) # for label size

    sns.heatmap(cm, annot=True, annot_kws={"size": 16}) # font size

    plt.show()

    w1 = cm['Predicted Wine 1']['True Wine 1'] / (cm['Predicted Wine 1']['True Wine 1'] + cm['Predicted Wine 2']['True Wine 1'] + cm['Predicted Wine 3']['True Wine 1'])

    w2 = cm['Predicted Wine 2']['True Wine 2'] / (cm['Predicted Wine 1']['True Wine 2'] + cm['Predicted Wine 2']['True Wine 2'] + cm['Predicted Wine 3']['True Wine 2'])

    w3 = cm['Predicted Wine 3']['True Wine 3'] / (cm['Predicted Wine 1']['True Wine 3'] + cm['Predicted Wine 2']['True Wine 3'] + cm['Predicted Wine 3']['True Wine 3'])

    print('\nClasswise accuracy: ')

    print('\nWine 1: ',w1 * 100)

    print('\nWine 2: ',w2 * 100)

    print('\nWine 3: ',w3 * 100)

    

    indices = ['Accuracy','Precision','F1 score','Recall  score']

    eval = pd.DataFrame([accuracy_score(y,y_pred) * 100,precision_score(y,y_pred,average = 'macro') * 100,f1_score(y,y_pred,average = 'macro') * 100,recall_score(y,y_pred,average = 'macro') * 100],columns=['Value'],index=indices)

    eval.index.name = 'Metrics'

    print('\n',eval)

    y = label_binarize(y, classes = range(1,4))

    for i in range(3):

        fpr, tpr, _ = roc_curve(y[:, i], y_score[:, i])

        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label = 'AUC = %0.2f' % roc_auc)        

    plt.plot([0, 1], [0, 1], 'k--')

    plt.xlim([-0.2, 1.05])

    plt.ylim([0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.legend()

    plt.title('ROC curve')

    plt.show()
gnb = GaussianNB()

start = time()

gnb.fit(x_train,y_train)

stop = time()

print('Training time: ',stop - start)

evaluateClassifier(x_train,y_train,gnb.predict(x_train),gnb.predict_proba(x_train))

evaluateClassifier(x_test,y_test,gnb.predict(x_test),gnb.predict_proba(x_test))
def dtree_grid_search(X,y,nfolds):

    param_grid = { 'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}

    dtree_model=DecisionTreeClassifier()

    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv = nfolds)

    dtree_gscv.fit(X, y)

    return dtree_gscv.best_params_



optimal_params = dtree_grid_search(x_train,y_train,5)



dtc = DecisionTreeClassifier(criterion = optimal_params['criterion'],max_depth = optimal_params['max_depth'])

start = time()

dtc.fit(x_train,y_train)

stop = time()

print('Training time: ',stop - start)

evaluateClassifier(x_train,y_train,dtc.predict(x_train),dtc.predict_proba(x_train))
optimal_params = dtree_grid_search(x_test,y_test,5)



dtc = DecisionTreeClassifier(criterion = optimal_params['criterion'],max_depth = optimal_params['max_depth'])

dtc.fit(x_train, y_train)

evaluateClassifier(x_test,y_test,dtc.predict(x_test),dtc.predict_proba(x_test))
def knn_grid_search(X,y,nfolds):

    param_grid = { 'n_neighbors':range(25)}

    knn_model = KNeighborsClassifier()

    knn_gscv = GridSearchCV(knn_model, param_grid, cv = nfolds)

    knn_gscv.fit(X, y)

    return knn_gscv.best_params_



optimal_params = knn_grid_search(x_train,y_train,5)



knn = KNeighborsClassifier(n_neighbors = optimal_params['n_neighbors'])

start = time()

knn.fit(x_train,y_train)

stop = time()

print('Training time: ',stop - start)

evaluateClassifier(x_train,y_train,knn.predict(x_train),knn.predict_proba(x_train))
optimal_params = knn_grid_search(x_test,y_test,5)



knn = KNeighborsClassifier(n_neighbors = optimal_params['n_neighbors'])

knn.fit(x_train, y_train)

evaluateClassifier(x_test,y_test,knn.predict(x_test),knn.predict_proba(x_test))