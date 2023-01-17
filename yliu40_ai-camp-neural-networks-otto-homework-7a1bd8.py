import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
data = pd.read_csv('../input/train.csv')
columns = list(data.columns)[1:-1]
X = data[columns]
X.head(2)
y = np.ravel(data['target'])
y[:10]
data['target'].value_counts().plot(kind='bar')
for i in range(9):
    plt.subplot(3,3,i+1)
    data[data['target']=='Class_'+str(i+1)].feat_20.hist()
plt.scatter(data['feat_19'],data['feat_20'])
figure = plt.figure()
ax = figure.add_subplot(111)
cax = ax.matshow(X.corr())
figure.colorbar(cax)
plt.show()
num_fea = X.shape[1]
print (num_fea)
model = MLPClassifier(solver = 'lbfgs',hidden_layer_sizes=(30,10), random_state=1, alpha=1e-5, verbose=True)
model.fit(X,y)
model.intercepts_
print(model.coefs_[0].shape)
print(model.coefs_[1].shape)
print(model.coefs_[2].shape)
pred = model.predict(X)
pred
model.score(X,y)
sum(pred == y) / len(y)
test_data = pd.read_csv('../input/test.csv')
Xtest = test_data[test_data.columns[1:]]
ytest = model.predict_proba(Xtest)
ytest[:2]
solution = pd.DataFrame(ytest, columns=['Class_1','Class_2','Class_3','Class_4','Class_5','Class_6','Class_7','Class_8','Class_9'])
solution['id'] = test_data['id']
solution.to_csv('./otto_prediction.csv', index=False, header=True)
