import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df= pd.read_csv("commercial_vedio_data.csv")
print(df.info())
df.info
df.head()
df.tail()
df.describe()
df.isnull()
df.isnull().values.any()
df.isnull().sum(axis=0)
df = df.fillna(df.mean())

df.isnull()
df = df.fillna(0)
df.isnull()
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(15,8))
barplot=sns.distplot(df['labels'], kde=False, rug=True)
plt.figure(figsize=(15,8))
corr_plot = sns.heatmap(df.corr(),cmap="RdYlGn",annot=False)
df_corr=df.corr()
x=[]
for i in df_corr['labels']:
    x.append(i)
plt.figure(figsize=(15,45))
ax=sns.barplot(x =x , y = df_corr['labels'].index)
ax.set(xlabel='correlation values', ylabel='columns names')
plt.show()
from sklearn.neural_network import MLPClassifier
import numpy as np
df["labels"]
# to see unique val
set(df['labels'].values)
df['labels'] = df['labels'].apply(lambda x:0 if x == -1 else x)
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Specify the data 
X=df.drop("labels",axis=1)

# Specify the target labels and flatten the array 
y=df["labels"]

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_tst = accuracy_score(y_test, pred)
print("testing accuracy", accuracy_tst)
clf = MLPClassifier(alpha=1e-3,max_iter=500, hidden_layer_sizes = (16,32,64,32,16))
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
accuracy_tst = accuracy_score(y_test, pred)
print("testing accuracy", accuracy_tst)
print(X_train.shape)
import keras
from keras.models import Sequential
from keras.layers import Dense, ReLU, Activation, Dropout
model = Sequential()
model.add(Dense(64, input_dim=231,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(64,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(64,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(128,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(128,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(128,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(256,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(256,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(512,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(512,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(512,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(512,kernel_initializer='normal'))
# model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Dense(1 ,activation="sigmoid"))
opt = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
model.summary()
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=500, batch_size=128)
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
model = Sequential()
model.add(Dense(256, input_dim=231, activation='relu',kernel_initializer='normal'))
model.add(Dense(128, activation='relu',kernel_initializer='normal'))
model.add(Dense(64, activation='relu',kernel_initializer='normal'))
model.add(Dense(32, activation='relu',kernel_initializer='normal'))
#model.add(Dense(16, activation='relu',kernel_initializer='normal'))
model.add(Dense(1 ,activation="sigmoid"))
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt,metrics=['accuracy'])
model.summary()