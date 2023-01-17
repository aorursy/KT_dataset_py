import pandas as pd

import numpy as np

dim1 = 20 #x-dimension

dim2 = 20 #y-dimension

data = {'x1':[],'x2':[],'label':[]}

for xcoor in range(dim1):

    for ycoor in range(dim2):

        if xcoor % 2 == 0 and ycoor % 2 == 0:

            label = 1

        elif xcoor % 2 == 1 and ycoor % 2 == 1:

            label = 1

        else:

            label = 0

        data['x1'].append(xcoor)

        data['x2'].append(ycoor)

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

    

    for j in np.linspace(0,40,300):

        for k in np.linspace(0,40,300):

            curr_x1.append(j)

            curr_x2.append(k)

            curr_y.append(model_instance.predict(np.array([[j,k]]))[0])

            

    plt.figure(figsize=(7,7))

    sns.scatterplot(curr_x1,curr_x2,hue=curr_y,palette='magma',alpha=0.5)

    plt.title(model)

    plt.show()