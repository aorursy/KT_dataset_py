import pandas as pd

import numpy as np

import math as m

import random as r

n = 800 #number of data points

innerRadius = 0.3 #radius of inner circle

outerRadius = 0.5 #radius of outer circle

z = 0 #gaussian multiplier

data = {'x1':[],'x2':[],'label':[]}

for i in range(n):

    x1 = r.randint(1,100_000)/100_000

    x2 = r.randint(1,100_000)/100_000

    coef = abs(np.random.normal(scale=z))

    if (x1-0.5)**2 + (x2-0.5)**2 > (innerRadius-coef)**2:

        if (x1-0.5)**2 + (x2-0.5)**2 < (outerRadius-coef)**2:

            label = 1

        else:

            label = 0

    else:

        label = 0

    data['x1'].append(x1)

    data['x2'].append(x2)

    data['label'].append(label)

data = pd.DataFrame(data)
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(7,7))

sns.set_style('whitegrid')

sns.scatterplot(data['x1'],data['x2'],hue=data['label'],palette='gist_rainbow',

               alpha=0.5)

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

    model_instance.fit(data[['x1','x2']],data['label'])

    

    curr_x1 = []

    curr_x2 = []

    curr_y = []

    

    for j in np.linspace(0,1,100):

        for k in np.linspace(0,1,100):

            curr_x1.append(j)

            curr_x2.append(k)

            curr_y.append(model_instance.predict(np.array([[j,k]]))[0])

            

    plt.figure(figsize=(7,7))

    sns.scatterplot(curr_x1,curr_x2,hue=curr_y,palette='gist_rainbow',alpha=0.5)

    plt.title(model)

    plt.show()