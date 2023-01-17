# Gerekli Kütüphaneleri Yüklenmesi Yapılıyor


import numpy as np # linear algebra
import pandas as pd # veri işleme

#Görselleştirme Kütüphaneleri
import seaborn as sns
import matplotlib.pyplot as plt

#Machine Learning tools
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report

import os
import warnings

# Çıktılarda karmaşıklığa sebep olduğu için uyarılırı iptal ediyoruz
warnings.filterwarnings("ignore")
print(os.listdir("../input"))
#Veri setinin yüklemesi yapılıyor
dataset=pd.read_csv("../input/winequality-red.csv")
dataset.head()
#Kaç farklı kalite puanı olduğunu öğrenelim
print("Kalite puanları:",dataset['quality'].unique())
dataset.info()
#Farklı kalite puanlarının görselleştirilmesi
sns.countplot(dataset['quality'])
#Özelliklerin kalite puanları ile ilişkisini göstermek için kullanılacak
#çizim türleri
def draw_multivarient_plot(dataset, rows, cols, plot_type):
    column_names=dataset.columns.values
    number_of_column=len(column_names)
    fig, axarr=plt.subplots(rows,cols, figsize=(22,16))

    counter=0
    for i in range(rows):
        for j in range(cols):
            if 'violin' in plot_type:
                sns.violinplot(x='quality', y=column_names[counter],data=dataset, ax=axarr[i][j])
            elif 'box'in plot_type :
                sns.boxplot(x='quality', y=column_names[counter],data=dataset, ax=axarr[i][j])
            elif 'point' in plot_type:
                sns.pointplot(x='quality',y=column_names[counter],data=dataset, ax=axarr[i][j])
            elif 'bar' in plot_type:
                sns.barplot(x='quality',y=column_names[counter],data=dataset, ax=axarr[i][j])
                
            counter+=1
            if counter==(number_of_column-1,):
                break
draw_multivarient_plot(dataset,4,3,"box")
draw_multivarient_plot(dataset,4,3,"violin")
draw_multivarient_plot(dataset,4,3,"pointplot")
draw_multivarient_plot(dataset,4,3,"bar")

def get_models():
    models=[]
    models.append(("LR",LogisticRegression()))
    models.append(("NB",GaussianNB()))
    models.append(("KNN",KNeighborsClassifier()))
    models.append(("DT",DecisionTreeClassifier()))
    models.append(("SVM rbf",SVC()))
    models.append(("SVM linear",SVC(kernel='linear')))
    
    return models

def cross_validation_scores_for_various_ml_models(X_cv, y_cv):
    print("Çapraz Doğrulama Başarı Oranları".upper())
    models=get_models()


    results=[]
    names= []

    for name, model in models:
        kfold=KFold(n_splits=10,random_state=22)
        cv_result=cross_val_score(model,X_cv, y_cv, cv=kfold,scoring="accuracy")
        names.append(name)
        results.append(cv_result)
        print("{} modelinin çapraz doğrulaması yapıldı, başarı oranı:{:0.2f}".format(name, cv_result.mean()))

   
dataset_temp=dataset.copy(deep=True)
X=dataset.drop('quality', axis=1)
y=dataset['quality']

X=StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.3, random_state=42)


cross_validation_scores_for_various_ml_models(X, y)

def SVM_GridSearch(X_train, X_test, y_train, y_test):
    best_score=0
    gammas=[0.001, 0.01, 0.1, 1, 10, 100]
    Cs=[0.001, 0.01, 0.1, 1, 10, 100]
    
    for gamma in gammas:
        for C in Cs:
            svm=SVC(kernel='rbf',gamma=gamma, C=C)
            svm.fit(X_train, y_train)
            
            
            score=svm.score(X_test, y_test)
            
            if score>best_score:
                y_pred=svm.predict(X_test)
                best_score=score
                best_params={'C':100, 'gamma':gamma}
        
    print("best score:",best_score)
    print("best params:",best_params)
    print("classification reports:\n",classification_report(y_test, y_pred))
SVM_GridSearch(X_train, X_test, y_train, y_test)



dataset_temp.loc[(dataset_temp['quality']==3),'quality']=1
dataset_temp.loc[(dataset_temp['quality']==4),'quality']=1

dataset_temp.loc[(dataset_temp['quality']==5),'quality']=2
dataset_temp.loc[(dataset_temp['quality']==6),'quality']=2

dataset_temp.loc[(dataset_temp['quality']==7),'quality']=3
dataset_temp.loc[(dataset_temp['quality']==8),'quality']=3

draw_multivarient_plot(dataset_temp,4,3,"violin")

draw_multivarient_plot(dataset_temp,4,3,"box")
X_temp=dataset_temp.drop('quality', axis=1)
y_temp=dataset_temp['quality']
X=StandardScaler().fit_transform(X)

X_train_temp, X_test_temp, y_train_temp, y_test_temp=train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)


cross_validation_scores_for_various_ml_models(X_temp, y_temp)
SVM_GridSearch(X_train_temp, X_test_temp, y_train_temp, y_test_temp)