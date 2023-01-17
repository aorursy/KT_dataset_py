import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set_style('whitegrid')
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.info()
df.isnull().sum()
plt.figure(figsize=(15,3))
sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')
df.describe().transpose()
plt.figure(figsize=(12,8))
sns.distplot(df['price'])
plt.figure(figsize=(12,8))
sns.countplot(df['bedrooms'])
plt.figure(figsize=(12,8))
sns.regplot(x='price',y='sqft_living',data=df)
plt.figure(figsize=(12,8))
sns.boxplot(x='bedrooms',y='price',data=df)
plt.figure(figsize=(12,8))
sns.regplot(x='price',y='long',data=df)
plt.figure(figsize=(12,8))
sns.regplot(x='price',y='lat',data=df,color='red')
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',data=df, hue='price')
df.sort_values('price',ascending=False).head(10)
len(df)*(0.01)
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize=(12,8))
sns.scatterplot(x='long',y='lat',
                data=non_top_1_perc,hue='price',
                palette='RdYlGn',edgecolor=None,alpha=0.2)
sns.boxplot(x='waterfront',y='price',data=df)
df.head()
df.info()
df = df.drop('id',axis=1)
df.head(3)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)
df = df.drop('date',axis=1)
df.head(2)
sns.boxplot(x='year',y='price',data=df)
sns.boxplot(x='month',y='price',data=df)
df.groupby('month').mean()['price'].plot()
df.groupby('year').mean()['price'].plot()
df.columns
# https://i.pinimg.com/originals/4a/ab/31/4aab31ce95d5b8474fd2cc063f334178.jpg
# May be worth considering to remove this or feature engineer categories from it
df['zipcode'].value_counts()
df = df.drop('zipcode',axis=1)
df.head()
# could make sense due to scaling, higher should correlate to more value
df['yr_renovated'].value_counts()
# makes sense: higher=more value, leave as is
df['sqft_basement'].value_counts()
X = df.drop('price',axis=1).values
y = df['price'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # only transform the test set, no fit to prevent data leakage
X_train.shape
X_test.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1)) # 1 neuron that predicts the price (regression)

model.compile(optimizer='adam',loss='mse')
# all arrays need to be NUMPY arrays, use .values
model.fit(x=X_train,y=y_train,
          validation_data=(X_test,y_test),
          batch_size=128, # power of 2
          epochs=400)
losses = pd.DataFrame(model.history.history)
losses
losses.plot()
plt.xlabel('Epochs')
plt.ylabel('Losses')
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
X_test # as NUMPY array, used .values before
predictions = model.predict(X_test)
predictions
# ALL ARE COMPARISONS OF TRUE LABELS AND OUR PREDICTIONS
print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
explained_variance_score(y_test,predictions)
df['price'].mean()
plt.figure(figsize=(12,6))

# Our predictions
plt.scatter(y_test,predictions,label='perfect fit line')

# Perfect predictions
plt.plot(y_test,y_test,'r')

plt.xlabel('y_test (actual)')
plt.ylabel('pred')
plt.legend()
errors = y_test - predictions
plt.figure(figsize=(12,6))
sns.distplot(errors)
X = df.drop('price',axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
plt.figure(figsize=(12,6))

# Our predictions
plt.scatter(y_test,predictions)

# Perfect predictions
plt.plot(y_test,y_test,'r',label='perfect fit line')

plt.xlabel('y_test (actual)')
plt.ylabel('pred')
plt.legend()
# ALL ARE COMPARISONS OF TRUE LABELS AND OUR PREDICTIONS
print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
plt.figure(figsize=(12,6))
sns.distplot(y_test-predictions) # COMPARE DIFFERENCE IN A HISTOGRAM
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients
importance=coeffecients['Coeffecient'].values
importance
importance = lm.coef_
importance
plt.figure(figsize=(20,6))

plt.bar(X_train.columns,abs(importance))
plt.show()