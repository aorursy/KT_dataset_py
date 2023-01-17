# IMportação das bibliotecas basicas



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os



# Abaixo listamos os arquivos da base





for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

#Leitura do arquivo

df = pd.read_csv('/kaggle/input/hmeq-data/hmeq.csv')
# Abaixo poemos veriricar o conteúdo da base

df
#Verificamos agora os tipos de dados

df.info()
# Verificando os valores nulos

df.isnull().sum()
#Imputação de dados nas colunas

df.loc[df['MORTDUE'].isnull(),'MORTDUE'] = 0

df.loc[df['VALUE'].isnull(),'VALUE'] = 0

df.loc[df['JOB'].isnull(),'JOB'] = 'None'

df.loc[df['REASON'].isnull(),'REASON'] = 'Other'

df.loc[df['YOJ'].isnull(),'YOJ'] = 0

df.loc[df['DEROG'].isnull(),'DEROG'] = 0

df.loc[df['DELINQ'].isnull(),'DELINQ'] = 0

df.loc[df['CLAGE'].isnull(),'CLAGE'] = 0

df.loc[df['NINQ'].isnull(),'NINQ'] = 0

df.loc[df['CLNO'].isnull(),'CLNO'] = 0

df.loc[df['DEBTINC'].isnull(),'DEBTINC'] = 0
#Listagem das classes de JOB

df['JOB'].unique()
#Listagem das classes de REASON

df['REASON'].unique()
#Definição da função para transformar a informação textual da coluna REASON em codificacao numerica

def REASONN (row):

   if row['JOB'] == 'Other':

      return 0

   if row['JOB'] == 'HomeImp':

      return 1

   if row['JOB'] == 'DebtCon':

      return 2

   return 3

df['REASONN'] = df.apply (lambda row: REASONN(row), axis=1)
#Definição da função para transformar informar textual da coluna JOB em codificacao numerica

def JOBN (row):

   if row['JOB'] == 'Other':

      return 0

   if row['JOB'] == 'Office':

      return 1

   if row['JOB'] == 'Sales':

      return 2

   if row['JOB'] == 'Mgr':

      return 3

   if row['JOB'] == 'ProfExe':

      return 4

   if row['JOB'] == 'Self':

      return 5

   return 6

df['JOBN'] = df.apply (lambda row: JOBN(row), axis=1)
#Contagem de valores nulos na base

df.isnull().sum()
# Separando as colunas para a construção do modelo

feats = [c for c in df.columns if c not in ['BAD','REASON','JOB']]
#Conteudo da base de dados

df.T
#Dividindo a base em treino e teste

from sklearn.model_selection import train_test_split



train,test = train_test_split(df, test_size=0.20, random_state=42)



train.shape,valid.shape,test.shape
# Instanciando o random forest classifier

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_jobs=-1, n_estimators=200, oob_score=True, random_state=42)
#Aplicando o RandomForest à base

rf.fit(train[feats], train['BAD'])
# Prevendo o BAD de teste usando o modelo treinado

y_test_pred = rf.predict(test[feats]).astype(int)
#Importação do pacote para aferição d acuracia

from sklearn.metrics import accuracy_score
# Medida da acurácia

accuracy_score(test['BAD'], y_test_pred)
#Vamos usar agora um método de medida mais confiável - AUROC - Area sob a curva ROC

#Geramos as probabilidades das classes na previsão (necessário para a rotina de medida AUROC)

y_test_prob = rf.predict_proba(test[feats])
#Pega so uma coluna para efetuar o teste

y_test_prob = [p[1] for p in y_test_prob]
#Importando o pacote para a medida da acuracia AUROC

from sklearn.metrics import roc_auc_score
#Medida da acurácia  usando a area sob a curva ROC - AUROC

roc_auc_score(test['BAD'], y_test_prob) 
#Importação de paaotes para plotagem de graficos

from sklearn import metrics

import matplotlib.pyplot as plt
#Curva ROC para os dados originais





fpr, tpr, threshold = metrics.roc_curve(test['BAD'], y_test_prob)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)



#plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), label="diagonal")

    

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Prevendo os dados para a matriz de confusão

cnf_matrix = metrics.confusion_matrix(test['BAD'], y_test_pred)

cnf_matrix
# import required modules

# is scikit's classifier.predict() using 0.5 by default?



#In probabilistic classifiers, yes. It's the only sensible threshold from a mathematical viewpoint, as others have explained.

import seaborn as sns 



%matplotlib inline

fig, ax = plt.subplots()

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Matriz Confusão', y=1.1)

plt.ylabel('Real')

plt.xlabel('Predito')
#Por ultimo geramos o relatorio com medidas da qualidade das predições

classific = metrics.classification_report(test['BAD'], y_test_pred)
print(classific)
#Importação da biblioteca para reamostragem

from imblearn.over_sampling import SMOTE, ADASYN   #reamostragem com a rotina SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(train[feats], train['BAD'])
y_resampled.value_counts()


rf2 = RandomForestClassifier(n_jobs=-1, n_estimators=200, oob_score=True, random_state=42)
#Aplicando o RandomForest à base

rf2.fit(X_resampled, y_resampled)
# Prevendo o BAD para a base com reamostragem SMOTE

y_test_pred_2 = rf2.predict(test[feats]).astype(int)
# Medida da acurácia

accuracy_score(test['BAD'], y_test_pred_2)
#Vamos usar agora um método de medida mais confiável - AUROC - Area sob a curva ROC

#Geramos as probabilidades das classes na previsão (necessário para a rotina de medida AUROC)

y_test_prob_2 = rf2.predict_proba(test[feats])
#Pega so uma coluna para efetuar o teste

y_test_prob_2 = [p[1] for p in y_test_prob_2]
#Medida da acurácia  usando a area sob a curva ROC - AUROC

roc_auc_score(test['BAD'], y_test_prob_2) 
#Curva ROC para os dados reamostrados





fpr, tpr, threshold = metrics.roc_curve(test['BAD'], y_test_prob_2)

roc_auc = metrics.auc(fpr, tpr)



# method I: plt



plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)



#plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), label="diagonal")

    

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
#Prevendo os dados para a matriz de confusão

cnf_matrix = metrics.confusion_matrix(test['BAD'], y_test_pred_2)

cnf_matrix
%matplotlib inline

fig, ax = plt.subplots()

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Matriz Confusão', y=1.1)

plt.ylabel('Real')

plt.xlabel('Predito')
#Geramos o relatorio com medidas da qualidade das predições

classific = metrics.classification_report(test['BAD'], y_test_pred_2)

print(classific)