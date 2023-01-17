import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import log_loss

from keras.models import Sequential

from keras.layers import Dense

from keras.utils import np_utils
train_data = pd.read_csv('../input/train.csv')



x = pd.get_dummies(train_data.drop(['Id','class'], 1))

y = pd.get_dummies(train_data['class'])
train_data.head(3)
x.head(3)
y.head(3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30)
entrada = x_train.shape[1]

saida = y_train.shape[1]
model = Sequential()



model.add(Dense(36, input_dim=entrada, activation='relu'))

model.add(Dense(saida, activation='softmax'))



model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['acc'])



model.fit(x_train, y_train, epochs=300, batch_size=100)
def medir(y_test, probpredict, index):

    

    start = index

    end = index+1



    print('-' * 40)

    print('Resposta da Amostra {}: {}'.format(index, np.array(y_test.iloc[start:end,:])))

    print('-' * 40)

    print('Probabilidade de "acc"..: {}'.format(np.array(probpredict[start:end, 0])))

    print('Probabilidade de "god"..: {}'.format(np.array(probpredict[start:end, 1])))

    print('Probabilidade de "unacc": {}'.format(np.array(probpredict[start:end, 2])))

    print('Probabilidade de "vgood": {}'.format(np.array(probpredict[start:end, 3])))
probpredict = model.predict_proba(x_test)
qtdTeste = 5



print('\nÁrea de Testes, Utilizando {} Amostras...\n'.format(qtdTeste))



for i in range(qtdTeste):

    medir(y_test, probpredict, i)
score = model.evaluate(x_test, y_test)[1] * 100

logloss = log_loss(y_test, probpredict)



auc1 = roc_auc_score(y_test.iloc[:, 0], probpredict[:, 0])

auc2 = roc_auc_score(y_test.iloc[:, 1], probpredict[:, 1])

auc3 = roc_auc_score(y_test.iloc[:, 2], probpredict[:, 2])

auc4 = roc_auc_score(y_test.iloc[:, 3], probpredict[:, 3])



geral = pd.DataFrame({

    'Class': ['acc', 'god', 'unacc', 'vgood'],

    'Auc': [auc1, auc2, auc3, auc4]

})



print()

print('-' * 40)

print('\tInformações Gerais')

print('-' * 40)

print('logloss = %.2f' % logloss)

print('-' * 40)

print('Score do Modelo = %.2f' % score)

print('-' * 40)

print('Scores Separados Utilizando AUC ROC\n\n', geral)

print('-' * 40)
test_data = pd.read_csv('../input/test.csv')



xtest = pd.get_dummies(test_data.drop('Id', 1))
result = model.predict_proba(xtest)



print('shape predicts = {}'.format(result.shape))



good = result[:, 1]

acc = result[:, 0]

unacc = result[:, 2]

vgood = result[:, 3]
pd.DataFrame({

    

    'Id' : test_data['Id'],

    'Class_vgood' : vgood,

    'Class_good' : good,

    'Class_acc' : acc,

    'Class_unacc' : unacc

    

}).to_csv('sampleSubmission.csv', index=False)
pd.read_csv('sampleSubmission.csv').head(3)