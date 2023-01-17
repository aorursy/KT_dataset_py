#The right way to ovversample
#https://beckernick.github.io/oversampling-modeling/
    
#How to Handle Imbalanced Classes in Machine Learning    
#https://elitedatascience.com/imbalanced-classes

#Over-sampling
#https://imbalanced-learn.readthedocs.io/en/stable/over_sampling.html

#https://datascienceplus.com/selecting-categorical-features-in-customer-attrition-prediction-using-python/
#https://www.datacamp.com/community/tutorials/understanding-logistic-regression-python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

from imblearn.over_sampling import SMOTE, ADASYN   #reamostragem com a rotina SMOTE
from sklearn.utils import resample #resample with replacement.

#Importando pacote de regressao e de  medida de acuracia
from sklearn.linear_model import LogisticRegression

#Importa o pacote para medida de acurácia
from sklearn.metrics import roc_auc_score

from sklearn.metrics import accuracy_score


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#leitura da base
df = pd.read_csv('/kaggle/input/agoravaimesmo/basefinalagoravaimesmo.csv',sep=';')

#df2 = pd.read_csv('/kaggle/input/base-completa/PES2015.txt',sep=';')
#define o numero de casa decimais a serem visualizadas = 0
pd.options.display.float_format = '{:.0f}'.format
df.T
#Imputação de dados nas colunas
df.loc[df['TinhaCartAssin'].isnull(),'TinhaCartAssin'] = 0
df.loc[df['EraServPubEst'].isnull(),'EraServPubEst'] = 0
df.loc[df['EraServPubEstAnt'].isnull(),'EraServPubEstAnt'] = 0
df.loc[df['RendMenPerc'].isnull(),'RendMenPerc'] = 0
df.loc[df['EsferaAtual'].isnull(),'EsferaAtual'] = 0
df.loc[df['EsferaAnt'].isnull(),'EsferaAnt'] = 0
df.loc[df['EstadoCivil'].isnull(),'EstadoCivil'] = 9

df.info()
#Lista de colunas para o filtro
#colunas = ['UF','Sexo','Idade','Raça','EstadoCivil','EsferaAtual','EraServPubEst','EsferaAnt','EraServPubEstAnt','TinhaCartAssin','QtdAnosEmpAnt','PosOcu358','RendMenPerc','Aposentado','ocupacao','atividade','GrAnterior']
#Lista de colunas para o filtro sem dados atuais de emprego
#EsferaAnt foi retirado porque tem forte correlação com EraServPubEstAnt
colunas_ant = ['UF','Sexo','Idade','Raça','EstadoCivil','EraServPubEstAnt','QtdAnosEmpAnt','GrAnterior']
#Tabela de dorrelações
df_merged2 = df.loc[:,['UF','Sexo','Idade','Raça','EstadoCivil','EraServPubEstAnt','QtdAnosEmpAnt','GrAnterior','EsferaAnt','Mobilidade']]
df_merged2.to_csv('df_merged2.csv')
df_merged2.corr()
df_merged2
df['Mobilidade'].value_counts()
#Apiicação do filtro
X = df.loc[:,colunas_ant]
X.T
df[df['Mobilidade'].notnull()]['Mobilidade'].value_counts()
#Importa a bibilioteca de split
from sklearn.model_selection import train_test_split

#Separando a variavel dependente
y = df['Mobilidade']

#verificando o tamanho das classes na base completa
y.value_counts().to_frame()
#verificando a proporcao das classes na base completa
(y.value_counts(normalize=True) * 100).to_frame()
#separando os dados de teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25)
#df_test = pd.concat([X_test, y_test], axis=1)
#verificando a frequencia das classes nos dados de teste
y_test.value_counts().to_frame()
##verificando a proporcao das classes nos dados de teste
(y_test.value_counts(normalize=True) * 100).to_frame()
# dividindo novamente os dados de treino em treino e validação
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25)
#verificando a proporcao das classes nos dados de treino
(y_train.value_counts(normalize=True) * 100).to_frame()
#verificando a quantidade das classes nos dados de treino
y_train.value_counts().to_frame()
##verificando a proporcao das classes nos dados de validação
(y_val.value_counts(normalize=True) * 100).to_frame()
#verificando a quantidade das classes nos dados de validação
y_val.value_counts().to_frame()
#oversampling os dados de treino

X_train.info()
acuracia = pd.DataFrame({
                'Treino Score':    [0.0, 0.0, 0.0],
                'Treino Auroc':    [0.0, 0.0, 0.0],
                'Validação Score': [0.0, 0.0, 0.0],
                'Validação Auroc': [0.0, 0.0, 0.0],
                'Teste Score':     [0.0, 0.0, 0.0],
                'Teste Auroc':     [0.0, 0.0, 0.0]
                },
                index=['Base Original', 'Reamostragem (RESAMPLE)', 'Reamostragem (SMOTE)'])
acuracia['Treino Score']['Base Original']

#Efetua a regressao logistica co os dados de treinamento
reg1 = LogisticRegression().fit(X_train, y_train)
def f_coef(reg,base_x):
    intercept = pd.DataFrame ([['Intercept',reg.intercept_[0]]], columns = ['Variável','Coeficiente'])
    coefficients =  pd.concat([pd.DataFrame(base_x.columns),pd.DataFrame(np.transpose(reg.coef_))], axis = 1)
    coefficients.columns = ['Variável','Coeficiente']
    return intercept.append(coefficients, ignore_index = True)
f_coef(reg1,X_train)
intercept = pd.DataFrame ([['Intercept',reg1.intercept_[0]]], columns = ['Variável','Coeficiente'])
intercept
coefficients =  pd.concat([pd.DataFrame(X_train.columns),pd.DataFrame(np.transpose(reg1.coef_))], axis = 1)
coefficients.columns = ['Variável','Coeficiente']
coefficients
all_coefs = intercept.append(coefficients)
all_coefs
coefficients
#prevendo os resultados do modelo com o uso dos dados de treinamento
pred_y_1 = reg1.predict(X_train)
#mede a acurácia do modelo
accuracy_score(y_train, pred_y_1)
acuracia['Treino Score']['Base Original'] = accuracy_score(y_train, pred_y_1)
#Repetimos com dados de validação - sem reamostragem
val_y_1 = reg1.predict(X_val)
accuracy_score(y_val, val_y_1)
acuracia['Validação Score']['Base Original'] = accuracy_score(y_val, val_y_1)

#Repetimos com dados de teste - sem reamostragem
test_y_1 = reg1.predict(X_test)
accuracy_score(y_test, test_y_1)
acuracia['Teste Score']['Base Original'] = accuracy_score(y_test, test_y_1)
#Prevendo os dados para a matriz de confusão

cnf_matrix = metrics.confusion_matrix(y_test, test_y_1)
cnf_matrix

# import required modules
# is scikit's classifier.predict() using 0.5 by default?

#In probabilistic classifiers, yes. It's the only sensible threshold from a mathematical viewpoint, as others have explained.

%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matriz Confusão', y=1.1)
plt.ylabel('Real')
plt.xlabel('Predito')
#Vamos usar agora um método de medida mais confiável - AUROC
#Geramos as probabilidades das classes na previsão (necessário para a rotina de medida AUROC)
prob_y_1 = reg1.predict_proba(X_train)
prob_y_1
y_train
#Pega so uma coluna
prob_y_1 = [p[1] for p in prob_y_1]
prob_y_1
#Medida da acurácia  - sem reamostragem - AUROC
#metrics.roc_auc_score gives the area under the ROC curve. Can anyone tell me what command will find the optimal cut-off point (threshold value)?
roc_auc_score(y_train, prob_y_1) 
acuracia['Treino Auroc']['Base Original'] = roc_auc_score(y_train, prob_y_1) 
#Repetimos com dados de validação - sem reamostragem - AUROC
prob_val_y_1 = reg1.predict_proba(X_val)
prob_val_y_1 = [p[1] for p in prob_val_y_1]
roc_auc_score(y_val, prob_val_y_1)
acuracia['Validação Auroc']['Base Original'] = roc_auc_score(y_val, prob_val_y_1)
#Repetimos com dados de teste - sem reamostragem - AUROC
prob_test_y_1 = reg1.predict_proba(X_test)
prob_test_y_1 = [p[1] for p in prob_test_y_1]
roc_auc_score(y_test, prob_test_y_1)
acuracia['Teste Auroc']['Base Original'] = roc_auc_score(y_test, prob_test_y_1)
y_train.value_counts()
#Curva ROC para os dados originais


fpr, tpr, threshold = metrics.roc_curve(y_test, prob_test_y_1)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt

plt.title('Curva ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

#plt.plot(np.linspace(0,1,10), np.linspace(0,1,10), label="diagonal")
    
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Taxa de positivos verdadeiros')
plt.xlabel('Taxa de falsos positivos')
plt.show()

threshold
X_train
#1. Up-sample Minority Class
#Up-sampling is the process of randomly duplicating observations from the minority class in order to reinforce its signal.
#There are several heuristics for doing so, but the most common way is to simply resample with replacement.



#merge de X_train e y_train paraa efetuar o upscaling
df_merged = pd.concat([X_train, y_train], axis=1)

# Separate majority and minority classes
df_majority = df_merged[df_merged.Mobilidade==0]
df_minority = df_merged[df_merged.Mobilidade==1]

#Faz o upscaling (reamostragem) usando a rotina resample
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=1556,    # to match majority class
                                 random_state=123) # reproducible results

df_upsampled = pd.concat([df_majority, df_minority_upsampled])   #merge das tabelas

df_upsampled['Mobilidade'].value_counts()  # contagem das classes

#Separação de colunas
X_upsampled = df_upsampled.loc[:,colunas_ant]
#Separando a variavel dependente
y_upsampled = df_upsampled['Mobilidade']

y_upsampled.value_counts()
reg2 = LogisticRegression().fit(X_upsampled, y_upsampled)
#prevendo os resultados do modelo com o uso dos dados reamostrados
pred_y_2 = reg2.predict(X_upsampled)
#mede a acurácia do modelo
accuracy_score(y_upsampled, pred_y_2)
acuracia['Treino Score']['Reamostragem (RESAMPLE)'] = accuracy_score(y_upsampled, pred_y_2)
#Repetimos com dados de validação
val_y_2 = reg2.predict(X_val)
accuracy_score(y_val, val_y_2)
acuracia['Validação Score']['Reamostragem (RESAMPLE)'] = accuracy_score(y_val, val_y_2)
#Repetimos com dados de teste
test_y_2 = reg2.predict(X_test)
accuracy_score(y_test, test_y_2)
acuracia['Teste Score']['Reamostragem (RESAMPLE)'] = accuracy_score(y_test, test_y_2)

# MAtriz de confursão pra RESAMPLE nos dados de teste


cnf_matrix = metrics.confusion_matrix(y_test, test_y_2)
# import required modules



%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matriz Confusão Resample', y=1.1)
plt.ylabel('Real')
plt.xlabel('Predito')
#Gera as probabilidades das classes na previsão (necessário para a rotina de medida AUROC)
prob_y_2 = reg2.predict_proba(X_upsampled)
#Pega so uma coluna
prob_y_2 = [p[1] for p in prob_y_2]
#Medida da acurácia  para os dados reamostrados
roc_auc_score(y_upsampled, prob_y_2) 
acuracia['Treino Auroc']['Reamostragem (RESAMPLE)'] = roc_auc_score(y_upsampled, prob_y_2) 
#Repetimos com dados de validação
prob_val_y_2 = reg2.predict_proba(X_val)
prob_val_y_2 = [p[1] for p in prob_val_y_2]
roc_auc_score(y_val, prob_val_y_2)
acuracia['Validação Auroc']['Reamostragem (RESAMPLE)'] = roc_auc_score(y_val, prob_val_y_2)
#Repetimos com dados de teste
prob_test_y_2 = reg2.predict_proba(X_test)
prob_test_y_2 = [p[1] for p in prob_test_y_2]
roc_auc_score(y_test, prob_test_y_2)
acuracia['Teste Auroc']['Reamostragem (RESAMPLE)'] = roc_auc_score(y_test, prob_test_y_2)

#Curva ROC para os dados reamostrados com RESMAPLE


fpr, tpr, threshold = metrics.roc_curve(y_test, prob_test_y_2)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt

plt.title('Curva ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Taxa de positivos verdadeiros')
plt.xlabel('Taxa de falsos positivos')
plt.show()
#Efetuamos agora a reamostragem usando a rotina SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
X_resampled.info()
y_resampled.value_counts()
##verificando a proporcao das classes nos dados de validação
(y_resampled.value_counts(normalize=True) * 100).to_frame()
#Efetuamos aqui a regressao logistica para os dados reamostrados com SMOTE
reg3 = LogisticRegression().fit(X_resampled, y_resampled)
#prevendo os resultados do modelo com o uso dos dados reamostrados
pred_y_3 = reg3.predict(X_resampled)
#mede a acurácia do modelo
accuracy_score(y_resampled, pred_y_3)
acuracia['Treino Score']['Reamostragem (SMOTE)'] = accuracy_score(y_resampled, pred_y_3)
#Repetimos com dados de validação
val_y_3 = reg3.predict(X_val)
accuracy_score(y_val, val_y_3)
acuracia['Validação Score']['Reamostragem (SMOTE)'] = accuracy_score(y_val, val_y_3)
#Repetimos com dados de validação
test_y_3 = reg3.predict(X_test)
accuracy_score(y_test, test_y_3)
acuracia['Teste Score']['Reamostragem (SMOTE)'] = accuracy_score(y_test, test_y_3)
# MAtriz de confursão pra SMOTE nos dados de teste

cnf_matrix = metrics.confusion_matrix(y_test, test_y_3)
    
%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Matriz Confusão SMOTE', y=1.1)
plt.ylabel('Real')
plt.xlabel('Predito')
#Gera as probabilidades das classes na previsão (necessário para a rotina de medida AUROC)
prob_y_3 = reg3.predict_proba(X_resampled)
#Pega so uma coluna
prob_y_3 = [p[1] for p in prob_y_3]
#Medida da acurácia  para os dados reamostrados
roc_auc_score(y_resampled, prob_y_3) 
acuracia['Treino Auroc']['Reamostragem (SMOTE)'] = roc_auc_score(y_resampled, prob_y_3) 
#Repetimos com dados de validação
prob_val_y_3 = reg3.predict_proba(X_val)
prob_val_y_3 = [p[1] for p in prob_val_y_3]
roc_auc_score(y_val, prob_val_y_3)
acuracia['Validação Auroc']['Reamostragem (SMOTE)'] = roc_auc_score(y_val, prob_val_y_3)
#Repetimos com dados de validação
prob_test_y_3 = reg3.predict_proba(X_test)
prob_test_y_3 = [p[1] for p in prob_test_y_3]
roc_auc_score(y_test, prob_test_y_3)
acuracia['Teste Auroc']['Reamostragem (SMOTE)'] = roc_auc_score(y_test, prob_test_y_3)
#Curva ROC para os dados reamostrados com SMOTE

fpr, tpr, threshold = metrics.roc_curve(y_test, prob_test_y_3)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt

plt.title('Curva ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('Taxa de positivos verdadeiros')
plt.xlabel('Taxa de falsos positivos')
plt.show()

pd.options.display.float_format = '{:.4f}'.format
#acuracia[['Validação Score','Validação Auroc']]
acuracia
pd.options.display.float_format = '{:.4f}'.format
df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
df_upsampled.to_csv('df_upsampled.csv')
df_merged.to_csv('df_merged.csv')
df_resampled.to_csv('df_resampled.csv')

df_merged
df_upsampled.info()
X.cov()
X.corr()