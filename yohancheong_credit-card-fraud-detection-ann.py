import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_colwidth', -1) 

df = pd.read_csv('../input/creditcardfraud/creditcard.csv')

# feature columns
print(df.columns)

# feature distribution
df.describe()
df.corrwith(df.Class).plot.bar(title='corr with class', x= 'feature', y='corr (pearson)', grid=True, fontsize=12, rot=30, figsize=(15, 4))
import seaborn as sn
sn.set_style('white')
f, ax = plt.subplots(figsize=(7, 7))
mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sn.diverging_palette(220, 10, as_cmap=True)
sn.heatmap(df.corr(), mask=mask, cmap="YlGnBu", vmax=.5, square=True)
feature_cols = np.array(df.columns)
feature_cols = feature_cols[~np.isin(feature_cols, ['Class', 'Time', 'Amount'])]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

df[feature_cols].describe()
# blank values
df.isna().sum()
# oversampling a less represented group
df[df.Class == 1].shape
df = df.append([df[df.Class == 1]]*10, ignore_index=True)
df[df.Class == 1].shape
# split train test
X = df.iloc[:, df.columns.isin(feature_cols)]
y = df.iloc[:, df.columns == 'Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
import tensorflow as tf
from tensorflow.keras import Model, Sequential, backend as K
from tensorflow.keras.layers import Input, Dense, Dropout

inp = Input(shape=(len(X.columns),))
dense_1 = Dense(100, activation='relu')(inp)
dense_2 = Dense(50, activation='relu')(dense_1)
dense_3 = Dense(100, activation='relu')(dense_2)
out = Dense(1, activation='sigmoid')(dense_3)
model = Model(inp, out)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=200, epochs=500, verbose=0)
score = model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=('Non-Fraud (Actual)', 'Fraud (Actual)'), columns=('Non-Fraud (Predicted)', 'Fraud (Predicted)'))
sn.set_style('white')
plt.figure(figsize = (7,7))
sn.heatmap(df_cm, annot=True, cmap="YlGnBu")
print(df_cm)
print("Backtest Accuracy: %0.4f" % accuracy_score(y_test, y_pred))