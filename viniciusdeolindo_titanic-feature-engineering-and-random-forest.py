import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import re
train = pd.read_csv("/kaggle/input/titanic/train.csv",header = [0])
test = pd.read_csv("/kaggle/input/titanic/test.csv",header = [0])
train1 = train.drop('Survived',axis=1)
df = pd.concat([train1,test])
profile = df.profile_report(title='Report - Titanic',minimal=True)
profile
# Avaliação da variável da idade
df['Age'] = df['Age'].fillna(method='ffill')
df['Age_bin'] = pd.qcut(df['Age'], 10, labels=False)
df['Age_bin']= df['Age_bin'].round(0).astype(str)
# Avaliação de variável de Fare
df['Fare'] = df['Fare'].fillna(method = 'ffill')
df['Fare_bin'] = pd.qcut(df['Fare'], 10, labels=False)
df['Fare_bin']= df['Fare_bin'].round(0).astype(str)
# Avaliação da variável Nome (Título)
df['Title'] = [nameStr[1].strip().split('.')[0] for nameStr in df['Name'].str.split(',')]
# Tamanho da familia
df['FamilySize'] =  df['SibSp'] + df['Parch'] + 1
# Avaliação da variável Cabine
df['IsCabinDataEmpty'] = 0
df.loc[df['Cabin'].isnull(),'IsCabinDataEmpty'] = 1
# Avaliação da variável PClass
df['Pclass'] = df['Pclass'].astype(str)
# Seleção das variáveis
df2 = df.filter(['Pclass', 'Sex', 'Embarked', 'Age_bin', 'Fare_bin', 'Title', 'FamilySize', 'IsCabinDataEmpty'])
# Transformação de variáveis categóricas para númericas
df3 = pd.get_dummies(df2)
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (confusion_matrix, precision_recall_curve, 
                             auc,roc_curve,recall_score, classification_report,
                             f1_score, precision_recall_fscore_support)
treino = df3.iloc[0:891]
teste = df3.iloc[891:1309]
xtr, xval, ytr, yval = train_test_split(treino,
                                        train['Survived'],
                                        test_size=0.2,
                                        random_state=67)
%%time
sel = SelectKBest(mutual_info_classif, k=15).fit(xtr,ytr)
selecao = list(xtr.columns[sel.get_support()])
print(selecao)
xtr = xtr.filter(selecao)
xval = xval.filter(selecao)
# Random Search Cross Validation
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
%%time
# Use the random grid to search for best hyperparameters
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,n_iter = 100,cv = 5,verbose=2,random_state=42,n_jobs = -1)
rf_random.fit(xtr, ytr)
baseline = RandomForestClassifier(**rf_random.best_params_)
baseline.fit(xtr,ytr)
p = baseline.predict(xval)
# Calculo das probabilidade das classes e as métricas para curva ROC.
y_pred_prob = baseline.predict_proba(xval)[:,1]
fpr,tpr,thresholds = roc_curve(yval,y_pred_prob)
roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# Curva ROC.
plt.plot(fpr, tpr)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('Curva ROC - Survived in Titanic')
plt.xlabel('False Positive Rate (1 — Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

# Calculo de metricas e thresholds.
i = np.arange(len(tpr)) # index for df
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),
                    'tpr' : pd.Series(tpr, index = i), 
                    '1-fpr' : pd.Series(1-fpr, index = i), 
                    'tf' : pd.Series(tpr - (1-fpr), index = i), 
                    'threshold' : pd.Series(thresholds, index = i)})
tab_metricas = roc.iloc[(roc.tf-0).abs().argsort()[:1]]
print(tab_metricas)

# Threshold: O corte ideal seria onde tpr é alto e fpr 
# é baixo tpr - (1-fpr) é zero ou quase zero é o ponto de corte ideal.
t = tab_metricas.iloc[0].values[4]
y_pred = [1 if e > t else 0 for e in y_pred_prob]

# Construção do plot da matriz de confusão
LABELS = ['not_survived', 'survived']
conf_matrix = confusion_matrix(yval, p)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Matriz de confusão")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.grid(False)
plt.show()

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
y_pred = baseline.predict(teste.filter(selecao))
submission_df = pd.DataFrame(columns=['PassengerId', 'Survived'])
submission_df['PassengerId'] = test['PassengerId']
submission_df['Survived'] = y_pred
submission_df.to_csv('submissions.csv', header=True, index=False)