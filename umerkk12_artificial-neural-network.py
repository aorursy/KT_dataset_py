import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df = pd.read_csv('../input/insurance/insurance.csv')
df.head()
msno.matrix(df)
df[df.isnull()].count()
df.info()
df.describe()
df.head()
Male = pd.get_dummies(df['sex'], drop_first=True)
df = pd.concat([df, Male], axis=1 )
Smoker = pd.get_dummies(df['smoker'], drop_first=True)
df = pd.concat([df, Smoker], axis=1 )
df = df.rename(columns={'yes':'Smoker'})
df['region'].unique()
region = pd.get_dummies(df['region'])
df = pd.concat([df, region], axis=1 )
#df.drop('region', axis=1,inplace=True)
#df.drop(['sex','smoker'], axis=1, inplace=True)
df.head()
plt.figure(figsize=(12,6))
sns.set_style('white')
sns.countplot(x='sex', data = df, palette='GnBu')
sns.despine(left=True)

plt.figure(figsize=(14,10))
sns.set_style('white')
sns.boxplot(x='sex', y='charges', data = df, palette='OrRd', hue='Smoker')
sns.despine(left=True)

fig, ax =plt.subplots(nrows= 1, ncols = 3, figsize= (14,6))
sns.scatterplot(x='age', y='charges', data = df, palette='coolwarm', hue='sex', ax=ax[0])
sns.scatterplot(x='age', y='charges', data = df, palette='GnBu', hue='Smoker', ax=ax[1])
sns.scatterplot(x='age', y='charges', data = df, palette='magma_r', hue='region', ax=ax[2])
sns.set_style('dark')
sns.despine(left=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
fig, ax =plt.subplots(nrows= 1, ncols = 2, figsize= (14,6))
sns.boxplot(x='region', y='charges', data = df, palette='GnBu', hue='Smoker', ax=ax[0])
sns.boxplot(x='region', y='charges', data = df, palette='coolwarm', hue='sex', ax=ax[1])
fig, ax =plt.subplots(nrows= 1, ncols = 2, figsize= (14,6))
sns.scatterplot(x='bmi', y='charges', data = df, palette='GnBu_r', hue='sex', ax=ax[0])
sns.scatterplot(x='bmi', y='charges', data = df, palette='magma', hue='Smoker', ax=ax[1])
sns.set_style('dark')
sns.despine(left=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
df.drop(['sex', 'region', 'smoker', 'southwest'], axis=1, inplace=True)
df.head()
plt.figure(figsize=(16,6))
sns.heatmap(df.corr(), cmap='OrRd')
X=df.drop('charges', axis=1)
y=df['charges']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_validate = scaler.transform(X_test)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()
model.add(Dense(units = 8, activation = 'relu'))
model.add(Dense(units = 3, activation = 'relu'))
#model.add(Dropout(0.5))

#model.add(Dense(units = 2, activation = 'relu'))
#model.add(Dense(units = 4, activation = 'relu'))
#model.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))



model.add(Dense(units = 1))
model.compile(optimizer = 'adam', loss = 'mse')
early_stop = EarlyStopping(monitor='val_loss', mode= 'min', verbose= 0, patience=15)

model.fit(x=X_train, y=y_train, epochs = 2000, validation_data=(X_test, y_test), batch_size=128, callbacks=[early_stop])
loss = pd.DataFrame(model.history.history)
loss.plot()
from sklearn.metrics import mean_squared_error
pred = model.predict(X_test)
np.sqrt(mean_squared_error(y_test,pred))
entry_1 = df[:][257:477].drop('charges', axis=1)
pred = model.predict(entry_1)
np.sqrt(mean_squared_error(df[:][257:477]['charges'], pred))