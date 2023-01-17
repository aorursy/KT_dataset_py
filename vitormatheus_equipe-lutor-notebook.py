import pandas as pd
import numpy as np
# import os ## Presente no Notebook original para manipulação de diretórios

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import pytz
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')
## Linha original modificada para plataforma Kaggle
# path_train, path_test = os.path.join('data', 'train.csv'), os.path.join('data', 'test.csv')

path_train, path_test = '../input/train.csv', '../input/test.csv'
data_train, data_test = pd.read_csv(path_train), pd.read_csv(path_test)

data, test, test_ids = data_train.drop(['id'], axis=1), data_test.drop(['id'], axis=1), data_test.id
target_attr = 'Classification'

data.sample(3)
data.describe()
sns.set(style="darkgrid")
ax = sns.countplot(x=target_attr, data=data);

plt.title("Histograma para Atributo Alvo", fontsize=16)
ax.set_xlabel("Classes", fontsize=14)
ax.set_ylabel("Quantidade de Observações", fontsize=14)

fig = plt.gcf()
fig.set_size_inches(18.5, 8.5)

plt.show()
correlations = data[data.columns.drop([target_attr])].corrwith(data[target_attr])
correlations.sort_values(inplace=True)

ax = plt.axes()
for x, y, label in zip(range(0, len(correlations)), correlations, correlations.keys()):
    ax.bar(x, y, label=label)

plt.title("Correlação de Pearson", fontsize=16)
plt.xlabel('Atributos', fontsize=14)
plt.ylabel('Correlação', fontsize=14)

ax.set(ylim=[-1, 1])
plt.xticks([])
plt.legend()

fig = plt.gcf()
fig.set_size_inches(18.5, 8.5)

plt.show()
negative = data.loc[data['Classification'] == 1]
positive = data.loc[data['Classification'] == 2]    

features = data.columns.drop('Classification')

for f, i in zip(features, range(1, len(features)+1)):
    plt.subplot(len(features)/3 + 1, 3, i)
    plt.title('Atributo ' + f, fontsize=14)
    
    plt.hist(negative[f], alpha=0.4, color='DarkBlue', label='Sem Câncer') 
    plt.hist(positive[f], alpha=0.6, color='DarkRed', label='Com Câncer') 
    
ax = plt.gca()    
handles, labels = ax.get_legend_handles_labels()
   
fig = plt.gcf()
fig.set_size_inches(18.5, 16.5)
fig.legend(handles, labels, prop={'size': 16})
fig.suptitle('Distribução das Características em Relação ao Atributo Alvo', fontsize=16)

plt.show()
drop_columns = [target_attr]

## Linhas abaixo utilizadas na submisão em que foram retirados os 
## atributos com menor correlação de Pearson em relação ao atributo alvo. 

# drop_columns = [target_attr, 'Adiponectin', 'Age', 'Leptin', 'MCP.1']
# test = test[test.columns.drop(['Adiponectin', 'Age', 'Leptin', 'MCP.1'])]
X, Y = data[data.columns.drop(drop_columns)], data[target_attr]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)
## Linhas abaixo utilizadas na submisão em que se utilizou de 'scalers' 
## para preparação do dataset. 

# scaler = StandardScaler()
# X, test = scaler.fit_transform(X), scaler.fit_transform(test)
alphas = np.arange(0.5, 3.5, 0.5)
ni, no = len(X_train.columns), data[target_attr].nunique()
hidden_layers = []

for alpha in alphas:
    nh = int(round(alpha * np.sqrt(ni * no)))
    
    for i in range(1, nh+1):
        if (nh - i):
            t = (i, nh-i,)
        else:
            t = (i,)
        
        hidden_layers.append(t)
params = {'activation' : ['identity', 'logistic', 'tanh', 'relu'],
          'solver' : ['lbfgs'],
          'alpha' : [1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
          'hidden_layer_sizes' : hidden_layers}
gs = GridSearchCV(MLPClassifier(), params, cv=10, n_jobs=-1, scoring='f1', return_train_score=False)
gs.fit(X, Y);
results = pd.DataFrame(gs.cv_results_).drop(['params'], axis=1)
results.sort_values('rank_test_score', inplace=True)
results.head()
best_model = gs.best_estimator_
best_model.fit(X_train, Y_train);

Y_pred = best_model.predict(X_test)
print(classification_report(Y_test, Y_pred))
sns.heatmap(confusion_matrix(Y_test, Y_pred), fmt='d', annot=True, cmap='Greys')

fig = plt.gcf()
fig.set_size_inches(10.5, 8.5)

plt.title('Matriz de Confusão para o Melhor Modelo', fontsize=16)
plt.xlabel('Valores Reais', fontsize=14);
plt.ylabel('Valores Previstos', fontsize=14);

plt.show()
best_model.fit(X, Y)
Y_pred = best_model.predict(test)
results = pd.DataFrame(data={'id' : test_ids, 'Classification': Y_pred})

sub_time = datetime.now(pytz.timezone('Etc/GMT+4'))
sub_title = 'sub_' + sub_time.strftime("%d_%m_%Y_at_%H_%M") + '.csv'

## Linhas responsáveis por salvar o arquivo, presentes no notebook original
# path_result = os.path.join('submissions', sub_title)
# results.to_csv(path_result, index=False)