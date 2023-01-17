import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

import pickle

import pandas_profiling

import tensorflow as ts
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv',header=[0])
profile = df.profile_report(title='Report - Churn Modeling')

profile
# Tabela cruzada

tb = df.pivot_table(['Age','EstimatedSalary','CreditScore','Balance','Tenure','NumOfProducts'],

               index=['Geography','Gender','Exited','IsActiveMember'],

               aggfunc='mean',

               margins=True).reset_index().round(0)

tb
# Boxplots com foco no target

df.boxplot(column='Age', by='Exited')

df.boxplot(column='Balance', by='Exited')

df.boxplot(column='CreditScore', by='Exited')

df.boxplot(column='EstimatedSalary', by='Exited')

df.boxplot(column='NumOfProducts', by='Exited')

df.boxplot(column='Tenure', by='Exited')
df_model = df.filter(['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard',

                      'IsActiveMember','EstimatedSalary','Exited'])
df_model['Age_bin'] = pd.qcut(df_model['Age'], 4, labels=False)

df_model['EstimatedSalary_bin'] = pd.qcut(df_model['EstimatedSalary'], 4, labels=False)

df_model['CreditScore_bin'] = pd.qcut(df_model['CreditScore'], 4, labels=False)

df_model['Tenure_bin'] = pd.qcut(df_model['Tenure'], 4, labels=False)

df_model.head()
df_model['Age_bin']= df_model['Age_bin'].astype(str)

df_model['EstimatedSalary_bin']= df_model['EstimatedSalary_bin'].astype(str)

df_model['CreditScore_bin']= df_model['CreditScore_bin'].astype(str) 

df_model['Tenure_bin']= df_model['Tenure_bin'].astype(str)
df_model1 = pd.get_dummies(df_model)

df_model1.head(2)
#df_model1.columns

df_model2 =df_model1.filter(['NumOfProducts','HasCrCard','IsActiveMember','Exited','Geography_France',

       'Geography_Germany', 'Geography_Spain', 'Gender_Female', 'Gender_Male',

       'Age_bin_0', 'Age_bin_1', 'Age_bin_2', 'Age_bin_3',

       'EstimatedSalary_bin_0', 'EstimatedSalary_bin_1',

       'EstimatedSalary_bin_2', 'EstimatedSalary_bin_3', 'CreditScore_bin_0',

       'CreditScore_bin_1', 'CreditScore_bin_2', 'CreditScore_bin_3',

       'Tenure_bin_0', 'Tenure_bin_1', 'Tenure_bin_2', 'Tenure_bin_3'])

df_model2.head()
sns.set(rc={'figure.figsize':(7,4)})

sns.heatmap(df_model2.corr())

plt.title('Mapa de correlações')

plt.show()
from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split

from sklearn.metrics import confusion_matrix, classification_report 

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import MultinomialNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
xtr, xval, ytr, yval = train_test_split(df_model2.drop('Exited',axis=1),

                                                    df_model2['Exited'],

                                                    test_size=0.3,

                                                    random_state=67)
sel = SelectKBest(f_classif, k=10).fit(xtr,ytr)

selecao = list(xtr.columns[sel.get_support()])

print(selecao)
xtr = xtr.filter(selecao)

xval = xval.filter(selecao)
models = [] 

models.append(('ADA', AdaBoostClassifier())) 

models.append(('GB', GradientBoostingClassifier())) 

models.append(('RF', RandomForestClassifier())) 

models.append(('CART', DecisionTreeClassifier())) 

models.append(('RGL', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('NB', MultinomialNB()))



# Avalia os algoritmos

results = [] 

names = [] 



for name, model in models: 

    cv_results = cross_val_score(model, xtr, ytr, cv=10, scoring="accuracy") 

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()) 

    print(msg)



# Compara os algoritmos

sns.set(rc={'figure.figsize':(5, 5)})

fig = plt.figure() 

fig.suptitle('Comparação de modelos') 

ax = fig.add_subplot(111) 

plt.boxplot(results) 

ax.set_xticklabels(names) 

plt.show()
# Grid search: Metricas

scoring = 'accuracy'

kfold = KFold(n_splits=10, random_state=8)

model = GradientBoostingClassifier(random_state = 8)



# Grid search: parâmetros

param_grid = {

    'n_estimators': [20,50,100,150],

    'learning_rate': [0.04, 0.03, 0.01],

    'max_depth': [3,4,5],

    #'min_samples_split': [0.0050, 0.0040, 0.0035, 0.0010],

    #'subsample':[0.6,0.7,0.8,0.9],

    #'max_features': ['sqrt', 'log2']

}



# Execução do grid search

CV_model = GridSearchCV(estimator=model, param_grid=param_grid,cv=kfold,scoring=scoring)

CV_model_result = CV_model.fit(xtr, ytr)



# Print resultados

print("Best: %f using %s" % (CV_model_result.best_score_, CV_model_result.best_params_))
baseline = GradientBoostingClassifier(**CV_model_result.best_params_)

baseline.fit(xtr,ytr)
p = baseline.predict(xval)
cmx = confusion_matrix(yval, p)

print(cmx)
print(classification_report(yval, p))
sns.set(rc={'figure.figsize':(8, 8)})

features = xtr.columns

importances = baseline.feature_importances_

indices = np.argsort(importances)



plt.title('Importancia das variáveis')

plt.barh(range(len(indices)), importances[indices], color='b', align='center')

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Importancia relativa')

plt.show()