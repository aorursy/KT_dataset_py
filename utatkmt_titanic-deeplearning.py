import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers.core import Dense, Activation

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# シード値の生成

np.random.seed(0)
# 訓練データおよびテストデータの読み込み

train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")

gender_submission = pd.read_csv("../input/titanic/gender_submission.csv")
train.isnull().sum()
# Ageのコラムに存在する欠損値をAgeの平均値で埋める

train.Age.fillna(train.Age.mean(), inplace=True)

test.Age.fillna(test.Age.mean(), inplace=True)
# Embarkedのコラムを削除する

train = train.drop('Embarked', axis=1)

test = test.drop('Embarked', axis=1)
# Nameのコラムを削除する

train = train.drop('Name', axis=1)

test = test.drop('Name', axis=1)
# Cabinのコラムを削除する

train = train.drop('Cabin',axis=1)

test = test.drop('Cabin', axis=1)
# Ticketのコラムを削除する

train = train.drop('Ticket', axis=1)

test = test.drop('Ticket', axis=1)
# maleを0,femaleを1に置換する

train = train.replace({'male':0, 'female':1})

test = test.replace({'male': 0, 'female':1})
train['Fare'] = train['Fare'] / train.Fare.max()

test['Fare'] = test['Fare'] / test.Fare.max()
train['Age'] = train['Age'] / train.Age.max()

test['Age'] = test['Age'] / test.Age.max()
train['Pclass'] = train['Pclass'] / train.Pclass.max()

test['Pclass'] = test['Pclass'] / test.Pclass.max()
# SibSpとParchの要素を足したFamilyという新たな列を生成し、データの正規化を行う

train['Family'] = train['SibSp'] + train['Parch']

train['Family'] = train['Family'] / train.Family.max()

test['Family'] = test['SibSp'] + test['Parch']

test['Family'] = test['Family'] / test.Family.max()
# SibSpとParchをテーブルから削除する

train = train.drop('SibSp', axis=1)

train = train.drop('Parch', axis=1)

train = train.drop('PassengerId', axis=1)

test = test.drop('SibSp', axis=1)

test = test.drop('Parch', axis=1)

test = test.drop('PassengerId', axis=1)
X_train = train.iloc[:891][['Pclass', 'Sex', 'Age', 'Fare', 'Family']]
x = X_train.values

x_test = test.values
Y_train = train.Survived.values

Y_train
# y = np.eye(2)[Y_train.astype(int)]

len(Y_train)
# 訓練データおよび検証データの生成

N_validation = 200 # 検証データの数

x_train, x_validation, y_train, y_validation = train_test_split(x, Y_train, test_size=N_validation)
# モデル設定

n_in = len(x[0]) # 5

n_hidden = 300

n_out = 1 # 2

activation = 'relu'







early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)



# 入力層 - 隠れ層

model = Sequential()

model.add(Dense(n_hidden, input_dim=n_in))

model.add(BatchNormalization())

model.add(Activation(activation))



# 隠れ層 - 隠れ層

model.add(Dense(n_hidden))

model.add(BatchNormalization())

model.add(Activation(activation))



# 隠れ層 - 出力層

model.add(Dense(n_out))

model.add(Activation('sigmoid'))



model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999),metrics=['accuracy'])

'''

モデル学習

'''

epochs = 200

batch_size = 50



hist = model.fit(x_train, y_train, epochs=epochs,

                batch_size=batch_size,

                validation_data=(x_validation, y_validation),

                callbacks=[early_stopping])
'''

学習の進み具合を可視化

'''

val_acc = hist.history['val_accuracy']

val_loss = hist.history['val_loss']



plt.rc('font', family='serif')

fig = plt.figure()

plt.plot(range(len(val_acc)), val_acc, label='acc', color='black')

plt.plot(range(len(val_loss)), val_loss, label='loss', color='gray')

plt.xlabel('epochs')

plt.show()
'''

モデルのテスト

'''

result = model.predict(x_test)



for i in range(len(result)):

    if result[i] > 0.5:

        result[i] = 1

    else:

        result[i] = 0



        
result
sub = gender_submission

sub['Survived'] = list(map(int, result))

sub.to_csv("submission.csv", index=False)