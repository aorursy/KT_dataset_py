import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier, multilayer_perceptron
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
scaler = StandardScaler()
# data_train.head()
# data_test.head()
correlations = data_train[data_train.columns[:-1]].corrwith(data_train.quality)
# correlations = correlations.map(abs)
correlations.sort_values(inplace=True)

X_train_new = scaler.fit_transform(data_train[correlations.sort_values().iloc[-8:].index])
X_test_new = scaler.fit_transform(data_test[correlations.sort_values().iloc[-8:].index])

ax = correlations.plot(kind='bar')
ax.set(ylim=[-1, 1], ylabel='pearson correlation');

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

plt.show()
Y_id = data_test.id

data_train.drop('id', axis=1, inplace=True)
data_test.drop('id', axis=1, inplace=True)

X_train, X_test, Y_train = data_train[data_train.columns[:-1]], data_test, data_train.quality 
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, random_state=42, shuffle=True)
scaler = StandardScaler()

X_train_sld, X_test_sld = scaler.fit_transform(X_train), scaler.fit_transform(X_test) 
x_train_sld, x_test_sld = train_test_split(X_train_sld, random_state=42, shuffle=True)
params = {'activation' : ['relu'], 
          'solver' : ['adam'],
          'alpha' : [0.001, 1],
          'max_iter' : [700],
          'learning_rate' : ['adaptive'],
          'hidden_layer_sizes' : [(100,), (500,), (1000,), (500, 500,), (1000, 1000,)]
         }

GS = GridSearchCV(MLPClassifier(), params, cv=3, scoring='f1_micro', return_train_score=False)
GS.fit(X_train_sld, Y_train);
# help(GridSearchCV)
# help(MLPClassifier)
results_df = pd.DataFrame(GS.cv_results_).drop(['params'], axis=1)
results_df.sort_values('rank_test_score', inplace=True)
results_df.drop(['rank_test_score'], axis=1, inplace=True)

results_df.head()
best_model = GS.best_estimator_
# best_model.fit(X_train_sld, Y_train)

Y_pred = best_model.predict(X_test_sld)
best_model.fit(x_train_sld, y_train);

y_pred = best_model.predict(x_test_sld)
print("F-Score", f1_score(y_test, y_pred, average='micro'))
sns.heatmap(confusion_matrix(y_test, y_pred, labels=np.arange(0, 11, 1)), fmt='d', annot=True)

fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)

plt.xlabel('Ground Truth', fontsize=17);
plt.ylabel('Predicted Values', fontsize=17);

# fig.savefig('matrix.png')
plt.show()
results = pd.DataFrame(data={'id' : Y_id, 'quality' : Y_pred})

# results.to_csv('results\\results_5.csv', index=False)