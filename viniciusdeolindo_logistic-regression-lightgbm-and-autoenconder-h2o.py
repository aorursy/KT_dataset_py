# Biblotecas básicas

import numpy as np              #Biblioteca para algebra linear e calculos

import pandas as pd             #Biblioteca para manipulação de dados

import pandas_profiling         #Biblioteca para analise exploratoria de dados

import matplotlib.pyplot as plt #Biblioteca para gráficos

import seaborn as sns           #Biblioteca para gráficos mais complexos

import h2o                      #Biblioteca para conexão do framework H2O 

import lightgbm as lgb          #Biblioteca para conexão do framework lightGBM



# Pacotes para realização das modelagens de aprendizado de máquina.

from pylab import rcParams

import matplotlib.gridspec as gridspec

from scipy.stats import randint as sp_randint

from scipy.stats import uniform as sp_uniform

from sklearn.feature_selection import SelectKBest, f_classif,mutual_info_classif

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split

from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc,roc_curve, 

                             recall_score, classification_report, f1_score, precision_recall_fscore_support)

from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
%%time

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
%%time

# Report Pandas Profiling

report = df.profile_report(title='Report - Ccard Fraud',minimal=True)

report
# Estatísticas Descritivas do dataset

round(df.describe(),5)
# Histogramas com foco no target

v_features = df.iloc[:,1:29].columns

plt.figure(figsize=(12,28*4))

gs = gridspec.GridSpec(28, 1)

for i, cn in enumerate(df[v_features]):

    ax = plt.subplot(gs[i])

    sns.distplot(df[cn][df.Class == 1], bins=50)

    sns.distplot(df[cn][df.Class == 0], bins=50)

    ax.set_xlabel('')

    ax.set_title('Histograma relação de variáveis com o Target: ' + str(cn))

plt.show()
# Retirar as duplicatas

df = df.drop_duplicates()
# Retirando a coluna Time

df = df.drop(['Time'], axis=1)
x = df.drop('Class',axis=1)

y = df['Class']

xtr, xval, ytr, yval = train_test_split(x,y,test_size=0.2,random_state=67)
# Selecionar um terço das variáveis do dataset

sel = SelectKBest(mutual_info_classif, k=10).fit(xtr,ytr)

selecao = list(xtr.columns[sel.get_support()])

print(selecao)
# Dados de treino e teste com a seleção de variáveis

xtr = xtr.filter(selecao)

xval = xval.filter(selecao)
%%time

# Pesquisa Aleatória

# Parâmetros para pesquisa aleatória

logistic = LogisticRegression()

C = np.logspace(0, 4, num=10)

penalty = ['l1', 'l2']

solver = ['liblinear', 'saga']

weights = np.linspace(0.01, 0.99, 10)

hyperparameters = dict(C=C, penalty=penalty, solver=solver, class_weight = weights)



# Treinamento e seleção dos melhores hiperparamentros

randomizedsearch = RandomizedSearchCV(logistic, hyperparameters,cv = 3,verbose=2,random_state=42,n_jobs = -1,scoring="recall")

best_model_random = randomizedsearch.fit(xtr, ytr)

print(best_model_random.best_estimator_)
baseline = LogisticRegression(C=1.0, class_weight=0.8811111111111111, dual=False,

                   fit_intercept=True, intercept_scaling=1, l1_ratio=None,

                   max_iter=100, multi_class='auto', n_jobs=None, penalty='l1',

                   random_state=None, solver='saga', tol=0.0001, verbose=0,

                   warm_start=False)

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

plt.title('Curva ROC - Credit Card Detection')

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



# Threshold: O corte ideal seria onde tpr é alto e fpr é baixo tpr - (1-fpr) é zero ou quase zero é o ponto de corte ideal.

t = tab_metricas.iloc[0].values[4]

y_pred = [1 if e > t else 0 for e in y_pred_prob]



# Construção do plot da matriz de confusão

LABELS = ['Normal', 'Fraud']

conf_matrix = confusion_matrix(yval, p)

plt.figure(figsize=(8,6))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Matriz de confusão")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.grid(False)

plt.show()



# Report de métricas

report = classification_report(yval, p)

print(report)
%%time

# Pesquisa Aleatória

# Paramentros para pesquisa aleatória

param_test ={'num_leaves': sp_randint(6, 50), 

             'min_child_samples': sp_randint(100, 500), 

             'min_child_weight': [1e-5, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4],

             'subsample': sp_uniform(loc=0.2, scale=0.8), 

             'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),

             'reg_alpha': [0, 1e-1, 1, 2, 5, 7, 10, 50, 100],

             'reg_lambda': [0, 1e-1, 1, 5, 10, 20, 50, 100]}

n_HP_points_to_test = 100

clf = lgb.LGBMClassifier(max_depth=-1, random_state=314, silent=True, metric='None', n_jobs=4, n_estimators=100)

randomizedsearch = RandomizedSearchCV(

    estimator=clf, param_distributions=param_test, 

    n_iter=n_HP_points_to_test,

    scoring='roc_auc',

    cv=3,

    refit=True,

    random_state=314,

    verbose=True)



# Treinamento e seleção dos melhores hiperparâmentros

best_model_random = randomizedsearch.fit(xtr, ytr)

print(best_model_random.best_estimator_)
baseline = lgb.LGBMClassifier(boosting_type='gbdt', class_weight=None,

               colsample_bytree=0.43473068607216775, importance_type='split',

               learning_rate=0.1, max_depth=-1, metric='None',

               min_child_samples=478, min_child_weight=0.01, min_split_gain=0.0,

               n_estimators=100, n_jobs=4, num_leaves=9, objective=None,

               random_state=314, reg_alpha=1, reg_lambda=5, silent=True,

               subsample=0.4261926450859534, subsample_for_bin=200000,

               subsample_freq=0)

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

plt.title('ROC for Credit Card Detection')

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



# Threshold: O corte ideal seria onde tpr é alto e fpr é baixo tpr - (1-fpr) é zero ou quase zero é o ponto de corte ideal.

t = tab_metricas.iloc[0].values[4]

y_pred = [1 if e > t else 0 for e in y_pred_prob]



# Construção do plot da matriz de confusão

LABELS = ['Normal', 'Fraud']

conf_matrix = confusion_matrix(yval, p)

plt.figure(figsize=(8,6))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Matriz de confusão")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.grid(False)

plt.show()



# Report de métricas

report = classification_report(yval, p)

print(report)
h2o.init(max_mem_size = 4)

h2o.remove_all()
creditData_df = h2o.import_file("../input/creditcardfraud/creditcard.csv")
features= creditData_df.drop(['Time'], axis=1)

train, test = features.split_frame([0.8])

print(train.shape)

print(test.shape)
# Conversão para dataframe pandas

train_df = train.as_data_frame()

test_df = test.as_data_frame()

train_df = train_df[train_df['Class'] == 0]



# Drop a variável de target

train_df = train_df.drop(['Class'], axis=1)

Y_test_df = test_df['Class'] # true labels of the testing set

test_df = test_df.drop(['Class'], axis=1)

train_df.shape



train_h2o = h2o.H2OFrame(train_df) # converting to h2o frame

test_h2o = h2o.H2OFrame(test_df)

x = train_h2o.columns
%%time

anomaly_model = H2ODeepLearningEstimator(activation = "Tanh",

                               hidden = [29,15,15,29],

                               epochs = 500,

                               standardize = True,

                               stopping_metric = 'MSE', 

                               loss = 'automatic',

                               train_samples_per_iteration = 32,

                               shuffle_training_data = True,     

                               autoencoder = True,

                               l1 = 10e-5)

anomaly_model.train(x=x, training_frame = train_h2o)

anomaly_model._model_json['output']['variable_importances'].as_data_frame()
rcParams['figure.figsize'] = 14, 8

#plt.rcdefaults()

fig, ax = plt.subplots()

variables = anomaly_model._model_json['output']['variable_importances']['variable']

var = variables[0:10]

y_pos = np.arange(len(var))

scaled_importance = anomaly_model._model_json['output']['variable_importances']['scaled_importance']

sc = scaled_importance[0:10]

ax.barh(y_pos, sc, align='center', color='green', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(variables)

ax.invert_yaxis()

ax.set_xlabel('Importancia relativa')

ax.set_title('Importancia das variáveis')

plt.show()
scoring_history = anomaly_model.score_history()

%matplotlib inline

rcParams['figure.figsize'] = 14, 8

plt.plot(scoring_history['training_mse'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

test_rec_error = anomaly_model.anomaly(test_h2o) 



# anomaly is a H2O function which calculates the error for the dataset

# converting to pandas dataframe

test_rec_error_df = test_rec_error.as_data_frame()

# plotting the testing dataset against the error

test_rec_error_df['id']=test_rec_error_df.index

rcParams['figure.figsize'] = 14, 8

test_rec_error_df.plot(kind="scatter", x='id', y="Reconstruction.MSE")

plt.show()
predictions = anomaly_model.predict(test_h2o)

error_df = pd.DataFrame({'reconstruction_error': test_rec_error_df['Reconstruction.MSE'],

                        'true_class': Y_test_df})

error_df.describe()
fig = plt.figure()

ax = fig.add_subplot(111)

rcParams['figure.figsize'] = 14, 8

normal_error_df = error_df[(error_df['true_class']== 0) & (error_df['reconstruction_error'] < 10)]

_ = ax.hist(normal_error_df.reconstruction_error.values, bins=20)
fig = plt.figure()

ax = fig.add_subplot(111)

rcParams['figure.figsize'] = 14, 8

fraud_error_df = error_df[error_df['true_class'] == 1]

_ = ax.hist(fraud_error_df.reconstruction_error.values, bins=20)
fpr, tpr, thresholds = roc_curve(error_df.true_class, error_df.reconstruction_error)

roc_auc = auc(fpr, tpr)



# Calculo de metricas e thresholds.

i = np.arange(len(tpr)) # index for df

roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),

                    'tpr' : pd.Series(tpr, index = i), 

                    '1-fpr' : pd.Series(1-fpr, index = i), 

                    'tf' : pd.Series(tpr - (1-fpr), index = i), 

                    'threshold' : pd.Series(thresholds, index = i)})

tab_metricas = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

print(tab_metricas)



# Threshold: O corte ideal seria onde tpr é alto e fpr é baixo tpr - (1-fpr) é zero ou quase zero é o ponto de corte ideal.

t = tab_metricas.iloc[0].values[4]



plt.title('ROC for Credit Card Detection')

plt.plot(fpr, tpr, label='AUC = %0.4f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.xlabel('False Positive Rate (1 — Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.show();
threshold = t

groups = error_df.groupby('true_class')

fig, ax = plt.subplots()

for name, group in groups:

    ax.plot(group.index, group.reconstruction_error, marker='o', ms=3.5, linestyle='',

            label= "Fraud" if name == 1 else "Normal")

ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')

ax.legend()

plt.title("Reconstruction error for different classes")

plt.ylabel("Reconstruction error")

plt.xlabel("Data point index")

plt.show();
LABELS = ['Normal', 'Fraud']

y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]

conf_matrix = confusion_matrix(error_df.true_class, y_pred)

plt.figure(figsize=(8,6))

sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");

plt.title("Matriz de confusão")

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.show()



csr = classification_report(error_df.true_class, y_pred)

print(csr)