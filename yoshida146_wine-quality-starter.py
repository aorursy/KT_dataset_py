import pandas as pd

import numpy as np

import os
input_path = '../input/1056lab-wine-quality-prediction' # 自分の環境に合わせて変更してください。

df_train = pd.read_csv(os.path.join(input_path, 'train.csv'), index_col=0)

df_test = pd.read_csv(os.path.join(input_path, 'test.csv'), index_col=0)
df_train.isnull().sum()
df_test.isnull().sum()
X = df_train.drop(['color', 'quality'], axis=1)

y = df_train['quality'].values

X_test = df_test.drop(['color'], axis=1)
X.fillna(0., inplace=True)

X_test.fillna(0., inplace=True)
from sklearn.svm import SVR



model = SVR(gamma='scale') # 警告が出るので未来のデフォルトパラメータで設定

model.fit(X, y)

predict = model.predict(X_test)
output_path = './' # 自分の環境に合わせて変更してください



submit = pd.read_csv(os.path.join(input_path, 'sampleSubmission.csv'), index_col=0)

submit['quality'] = predict

submit.to_csv('submit.csv')