import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

import os
os.listdir('../input')

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
brainwave_df = pd.read_csv('../input/emotions.csv')
brainwave_df.head()
brainwave_df.shape
brainwave_df.describe()
plt.figure()
sns.countplot(x=brainwave_df.label, color='blue')
plt.title('Emotional sentiment class distribution')
plt.ylabel('Class Counts')
plt.xlabel('Class Label')
plt.xticks(rotation='vertical');
label_df = brainwave_df['label']
brainwave_df.drop('label', axis = 1, inplace=True)
correlations = brainwave_df.corr(method='pearson')
correlations
skew = brainwave_df.skew()
skew
%%time

pl_random_forest = Pipeline(steps=[('random_forest', RandomForestClassifier())])
scores = cross_val_score(pl_random_forest, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for RandomForest : ', scores.mean())
%%time

pl_log_reg = Pipeline(steps=[('scaler',StandardScaler()),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])
scores = cross_val_score(pl_log_reg, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for Logistic Regression: ', scores.mean())
scaler = StandardScaler()
scaled_df = scaler.fit_transform(brainwave_df)
pca = PCA(n_components = 30)
pca_vectors = pca.fit_transform(scaled_df)
for index, var in enumerate(pca.explained_variance_ratio_):
    print("Explained Variance ratio by Principal Component ", (index+1), " : ", var)

plt.figure()
plt.plot(pca.explained_variance_ratio_)
plt.xticks(rotation='vertical')
plt.figure(figsize=(25,8))
sns.scatterplot(x=pca_vectors[:, 0], y=pca_vectors[:, 1], hue=label_df)
plt.title('Principal Components vs Class distribution', fontsize=16)
plt.ylabel('Principal Component 2', fontsize=16)
plt.xlabel('Principal Component 1', fontsize=16)
plt.xticks(rotation='vertical');
%%time
pl_log_reg_pca = Pipeline(steps=[('scaler',StandardScaler()),
                             ('pca', PCA(n_components = 2)),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])
scores = cross_val_score(pl_log_reg_pca, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for Logistic Regression with 2 Principal Components: ', scores.mean())
%%time

pl_log_reg_pca_10 = Pipeline(steps=[('scaler',StandardScaler()),
                             ('pca', PCA(n_components = 10)),
                             ('log_reg', LogisticRegression(multi_class='multinomial', solver='saga', max_iter=200))])
scores = cross_val_score(pl_log_reg_pca_10, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for Logistic Regression with 10 Principal Components: ', scores.mean())
%%time

pl_mlp = Pipeline(steps=[('scaler',StandardScaler()),
                             ('mlp_ann', MLPClassifier(hidden_layer_sizes=(1275, 637)))])
scores = cross_val_score(pl_mlp, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for ANN : ', scores.mean())
%%time

pl_svm = Pipeline(steps=[('scaler',StandardScaler()),
                             ('pl_svm', LinearSVC())])
scores = cross_val_score(pl_svm, brainwave_df, label_df, cv=10,scoring='accuracy')
print('Accuracy for Linear SVM : ', scores.mean())
%%time
pl_xgb = Pipeline(steps=
                  [('xgboost', xgb.XGBClassifier(objective='multi:softmax'))])
scores = cross_val_score(pl_xgb, brainwave_df, label_df, cv=10)
print('Accuracy for XGBoost Classifier : ', scores.mean())
# np.array(brainwave_df).shape
X = np.array(brainwave_df)
# X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
Y = np.array(label_df)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = np.resize(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.resize(X_test, (X_test.shape[0], 1, X_test.shape[1]))
label_enc = LabelEncoder()
Y_train = label_enc.fit_transform(Y_train)
Y_test = label_enc.transform(Y_test)
Y_train.shape, Y_test.shape
model = Sequential()
model.add(LSTM(120, activation='relu', input_shape=(1, 2548)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=1)
scores = model.evaluate(X_test, Y_test, verbose=0)
scores
lstmdrop = Sequential()
lstmdrop.add(LSTM(100))
lstmdrop.add(Dropout(0.2))
lstmdrop.add(Dense(1, activation='sigmoid'))
lstmdrop.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_lstm = lstmdrop.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=1)
scores_dropout = lstmdrop.evaluate(X_test, Y_test)
scores_dropout
lstm_stack = Sequential()
lstm_stack.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(1, 2548)))
lstm_stack.add(LSTM(100, activation='relu', return_sequences=True))
lstm_stack.add(LSTM(50, activation='relu', return_sequences=True))
lstm_stack.add(LSTM(25, activation='relu'))
lstm_stack.add(Dense(20, activation='relu'))
lstm_stack.add(Dense(10, activation='relu'))
lstm_stack.add(Dense(1, activation='sigmoid'))
lstm_stack.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history_stacked = lstm_stack.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=1)
scores_stacked = lstm_stack.evaluate(X_test, Y_test)
scores_stacked
conv = Sequential()
conv.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
conv.add(MaxPooling1D(pool_size=2, padding='same'))
conv.add(LSTM(100))
conv.add(Dense(1, activation='sigmoid'))
conv.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history_conv = conv.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=1)
scores_conv = conv.evaluate(X_test, Y_test)
scores_conv
conv_stack = Sequential()
conv_stack.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
conv_stack.add(MaxPooling1D(pool_size=2, padding='same'))
conv_stack.add(LSTM(100, activation='relu', return_sequences=True))
conv_stack.add(LSTM(50, activation='relu', return_sequences=True))
conv_stack.add(LSTM(25, activation='relu'))
conv_stack.add(Dense(20, activation='relu'))
conv_stack.add(Dense(10, activation='relu'))
conv_stack.add(Dense(1, activation='sigmoid'))
conv_stack.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history_conv_stack = conv_stack.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=1)
scores_conv_stack = conv_stack.evaluate(X_test, Y_test)
scores_conv_stack
history
bi = Sequential()
bi.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(1, 2548)))
bi.add(Dense(1))
bi.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history_bi = bi.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=1)
scores_bi = bi.evaluate(X_test, Y_test)
scores_bi
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('LSTM Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('LSTM Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history_lstm.history.keys())
# summarize history for accuracy
plt.plot(history_lstm.history['acc'])
plt.plot(history_lstm.history['val_acc'])
plt.title('LSTM/Dropout Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_lstm.history['loss'])
plt.plot(history_lstm.history['val_loss'])
plt.title('LSTM/Dropout Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history_stacked.history.keys())
# summarize history for accuracy
plt.plot(history_stacked.history['acc'])
plt.plot(history_stacked.history['val_acc'])
plt.title('Stacked LSTM Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_stacked.history['loss'])
plt.plot(history_stacked.history['val_loss'])
plt.title('Stacked LSTM Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history_conv.history.keys())
# summarize history for accuracy
plt.plot(history_conv.history['acc'])
plt.plot(history_conv.history['val_acc'])
plt.title('Convolutions + LSTM Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_conv.history['loss'])
plt.plot(history_conv.history['val_loss'])
plt.title('Convolutions + LSTM Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history_conv_stack.history.keys())
# summarize history for accuracy
plt.plot(history_conv_stack.history['acc'])
plt.plot(history_conv_stack.history['val_acc'])
plt.title('Convolutions + Stacked LSTM Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_conv_stack.history['loss'])
plt.plot(history_conv_stack.history['val_loss'])
plt.title('Convolutions + Stacked LSTM Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
print(history_bi.history.keys())
# summarize history for accuracy
plt.plot(history_bi.history['acc'])
plt.plot(history_bi.history['val_acc'])
plt.title('Bidirectional LSTM Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history_bi.history['loss'])
plt.plot(history_bi.history['val_loss'])
plt.title('Bidirectional LSTM Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
