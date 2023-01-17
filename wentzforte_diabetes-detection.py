import numpy as np

import pandas as pd

import statsmodels.api as sm

import scipy.stats as st

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.metrics import confusion_matrix

import matplotlib.mlab as mlab

%matplotlib inline



from sklearn import datasets, linear_model

from  sklearn.model_selection  import  train_test_split

from sklearn.metrics import mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')
dados_base = pd.read_csv('../input/dataset_treino.csv')

dados = dados_base.copy()

dados.describe().T
sb.countplot(x='classe',data=dados)
sb.pairplot(data=dados, hue='classe')
dados.describe().T
dados = dados.drop(columns=['id'])

dados = dados.drop(columns=['grossura_pele']) #identificado corelação com bmi

dados = dados[dados.bmi > 0]

dados = dados[dados.glicose > 0]

dados = dados[dados.pressao_sanguinea > 0]

dados = dados[dados.insulina <= 600]

dados = dados[dados.idade <= 80]

dados = dados[dados.num_gestacoes <= 15]

dados.insulina[dados.insulina == 0] = round(dados.insulina[dados.insulina > 0].mean())
dados.describe().T
fig = plt.subplots(figsize = (10,10))

sb.set(font_scale=1.5)

sb.heatmap(dados.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 10})

plt.show()
X_train, X_test, y_train, y_test = train_test_split(dados.drop(columns=['classe']), pd.DataFrame(dados.classe))
y_test.describe().T
X_train.describe().T
lm = linear_model.LinearRegression()

lm.fit(X_train, y_train)

y_pred = lm.predict(X_test)

print('Coefficients: \n', lm.coef_)

print("Mean squared error: %.2f" % mean_squared_error(y_test, y_pred))

print('Variance score: %.2f' % r2_score(y_test, y_pred))

print('Total True: %d' % lm.predict(X_test).round().sum())
y_test['classe'] = lm.predict(X_test).round()

y_test.classe = y_test.classe.astype(int)

sb.countplot(x='classe',data= y_test)
envio_base = pd.read_csv('../input/dataset_teste.csv')

envio_base.describe().T
envio = envio_base

envio = envio.drop(columns=['id'])

envio = envio.drop(columns=['grossura_pele'])



envio.bmi[envio.bmi == 0] = round(envio.bmi[(envio.bmi > 0)].mean())

envio.pressao_sanguinea[envio.pressao_sanguinea == 0] = round(envio.pressao_sanguinea[(envio.pressao_sanguinea > 0)].mean())

envio.insulina[envio.insulina == 0] = round(envio.insulina[(envio.insulina > 0) & (envio.insulina < 600)].mean())

envio.insulina[envio.insulina > 600] = round(envio.insulina[(envio.insulina > 0) & (envio.insulina < 600)].mean())



envio.describe().T
envio_final = pd.DataFrame(envio_base.id)

envio_final['classe'] = lm.predict(envio).round()

envio_final.classe[envio_final.classe < 0.1] = 0

envio_final.classe[envio_final.classe > 0] = 1

envio_final.classe = envio_final.classe.astype(int)

sb.countplot(x='classe',data=envio_final)

envio_final.describe().T
envio_final.to_csv('Submission.csv', index=False)