# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def create_data(x, len_seq):

    # データを作成

    X = []  # 入力

    y = []  # 出力

    for i in range(len(x) - len_seq):

        X.append(x[i:i+len_seq])  # 入力ベクトルを追加

        y.append(x[i+len_seq])  # 出力値を追加

    return np.array(X), np.array(y)



df = pd.read_csv('../input/train.csv', index_col=0)

x = df['Min temp.'].values



len_seq = 121  # 入力シーケンス長（121日=約4ヶ月）

X, y = create_data(x, len_seq)

X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # (samples, timesteps, features)

print(X.shape)

print(y.shape)
from keras.models import Sequential

from keras.layers.core import Dense, Dropout

from keras.layers.recurrent import LSTM



model = Sequential()

model.add(LSTM(units=64, return_sequences=False, input_shape=(len_seq, 1)))

model.add(Dropout(0.2))

model.add(Dense(units=1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()
model.fit(X, y, batch_size=365, epochs=100, validation_split=0.1)
predict = model.predict(X)  # Xに対して予測

predict.flatten()  # ベクトル化



%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(y, marker='.', linestyle='None', color='c')  # yを点で出力

plt.plot(predict)  # 予測値を線で出力

plt.show()
x = df['Min temp.'].values[len(x)-len_seq:]  # 最初の予測のためのベクトル（訓練データの最後のlen_seq個）



predict = np.array([])

for i in range(365):

    X = np.reshape(x, (1, len_seq, 1))  # ベクトルxを3階テンソルに (samples, timesteps, features)

    p = model.predict(X)[0]  # Xからpを予測

    predict = np.append(predict, p)  # pをpredictに追加

    x = np.append(np.delete(x, 0), p)  # xの先頭要素を削除し、最後に予測したpを追加



%matplotlib inline

import matplotlib.pyplot as plt

plt.plot(predict)  # 予測値を線で出力

plt.show()
submit = pd.read_csv('../input/sampleSubmission.csv')

submit['Min temp.'] = predict

submit.to_csv('submission.csv', index=False)