from sklearn.datasets import make_blobs

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
X,y = make_blobs(n_samples=1000, 

                 n_features=2,

                 cluster_std=0.8, 

                 centers = 5,

                 center_box = (-5,5),

                 random_state=10)
plt.figure(figsize=(7,7))

sns.set_style('whitegrid')

sns.scatterplot(X[:,0],X[:,1],hue=y,palette='gist_rainbow')

plt.title("Original Data Plotted")

plt.show()
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import AdaBoostClassifier



models_dic = {'Decision Tree':DecisionTreeClassifier(),

              'Random Forest':RandomForestClassifier(),

              'Logistic Regression':LogisticRegression(),

              'SVM':SVC(),

              'KNN':KNeighborsClassifier(),

              'Naive Bayes':GaussianNB(),

              'AdaBoost':AdaBoostClassifier()}

model_names = list(models_dic.keys())



for model in model_names:

    

    model_instance = models_dic[model]

    model_instance.fit(X,y)

    

    curr_x1 = []

    curr_x2 = []

    curr_y = []

    

    for j in np.linspace(-6,5,100):

        for k in np.linspace(-6,5,100):

            curr_x1.append(j)

            curr_x2.append(k)

            curr_y.append(model_instance.predict(np.array([[j,k]]))[0])

            

    plt.figure(figsize=(7,7))

    sns.scatterplot(curr_x1,curr_x2,hue=curr_y,palette='gist_rainbow',

                   alpha=0.5)

    plt.title(model)

    plt.show()