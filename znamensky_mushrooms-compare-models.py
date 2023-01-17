import numpy as np

import pandas as pd

import seaborn as sns

sns.set_palette('husl')

import matplotlib.pyplot as plt

%matplotlib inline





import pandas_profiling



from sklearn.preprocessing import LabelEncoder

from itertools import combinations



from sklearn.model_selection import train_test_split



from sklearn import metrics

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier  # for K nearest neighbours

from sklearn import svm  #for Support Vector Machine (SVM) Algorithm

from sklearn import metrics #for checking the model accuracy

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.ensemble import RandomForestClassifier



import xgboost as xgb

from xgboost.sklearn import XGBClassifier

from catboost import Pool, CatBoostClassifier





from scipy.stats import norm

from scipy import stats

from scipy.stats import skew 

from sklearn.preprocessing import StandardScaler





data=pd.read_csv('../input/mushroom-classification/mushrooms.csv')

y=data['class'].copy()

#y=data['class'].copy()

#data=data.drop(['class'],axis=1)

#All columns are object

#22 columns



#data.nunique()

print('Proportion of e/p','\n',

      data['class'].value_counts())



#s.unstack(level=0)

for color in ['cap-color','gill-color','stalk-color-above-ring','stalk-color-below-ring','veil-color','spore-print-color']:

    a=(pd.DataFrame(data.groupby([color,'class'])['class'].count()).unstack())

    fig = plt.figure()

    a.plot(kind='bar', legend=True, figsize=(9,3), title=color)
data1=data.drop(['class'],axis=1)
data1= pd.get_dummies(data1)
data1.columns
def model_mass_calc(X,y):



    #Some parameters



    svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)



    #Split



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

    #Standartize



    sc = StandardScaler()

    sc.fit(X_train)

    X_train_std = sc.transform(X_train)

    X_test_std = sc.transform(X_test)

    a=[]

   

    #Search knn_param

    a_index=list(range(1,11))

    knn=[1,2,3,4,5,6,7,8,9,10]

    a=[]

    for i in knn:

        model=KNeighborsClassifier(n_neighbors=i) 

        model.fit(X_train_std, y_train)

        prediction=model.predict(X_test_std)

        a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))





    #Max_Score_KNN

    knn=pd.DataFrame(knn)

    a=pd.DataFrame(a)

    knn_data=pd.concat([knn,a],axis=1)

    knn_data.columns=['Neig','Score']

    knn_take=int(knn_data[knn_data['Score']==knn_data['Score'].max()][:1]['Neig'])



    #model

    #SolveLater How to write names automat

    x=['CatB','XGB','RandomF','NB','svm.SVC','Log','DTr',str('KN='+str(knn_take))]

    #Form for cycle



    models=[CatBoostClassifier(),XGBClassifier(),RandomForestClassifier(),GaussianNB(),svm,LogisticRegression(),DecisionTreeClassifier(),KNeighborsClassifier(n_neighbors=knn_take)]

    a_index=list(range(1,len(models)+1))

    a=[]

    for model in models:



        model.fit(X_train_std, y_train)

        prediction=model.predict(X_test_std)

        a.append(pd.Series(metrics.accuracy_score(prediction,y_test)))

    plt.plot(x, a)

    #plt.xticks(x)

    #MAX_Score+Model

    x=pd.DataFrame(x)

    a=pd.DataFrame(a)

    all_scores=pd.concat([x,a],axis=1)

    all_scores.columns=['model','Score']

    print('Max_score:',all_scores[all_scores['Score']==all_scores['Score'].max()])
model_mass_calc(data1,y)