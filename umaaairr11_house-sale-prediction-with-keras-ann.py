import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
df = pd.read_csv('../input/housesalesprediction/kc_house_data.csv')
df.head()
df.isnull().sum()
df.describe().transpose()
plt.figure(figsize = (12,8))
sns.distplot(df['price'])
plt.figure(figsize = (10,6))
sns.countplot(df['bedrooms'])
hist2 = [go.Histogram(x=df.yr_built,xbins=dict(start=np.min(df.yr_built),size=1,end=np.max(df.yr_built)),marker=dict(color='rgb(0,102,0)'))]

histlayout2 = go.Layout(title="Built Year Counts",xaxis=dict(title="Years"),yaxis=dict(title="Built Counts"))

histfig2 = go.Figure(data=hist2,layout=histlayout2)

iplot(histfig2)
sns.countplot(x='floors',data=df, palette='Set2')
plt.figure(figsize = (12,8))
sns.scatterplot(x='price',y='sqft_living',data=df)
plt.figure(figsize = (12,8))
sns.boxplot(x='bedrooms',y='price',data=df)
sns.boxplot(x='waterfront',y='price',data=df)
plt.figure(figsize = (12,8))
sns.scatterplot(x='price',y='long',data=df)
plt.figure(figsize = (12,8))
sns.scatterplot(x='price',y='lat',data=df)
plt.figure(figsize = (12,8))
sns.scatterplot(x='long',y='lat',data=df,hue='price')
df.sort_values('price',ascending=False).head(20)
len(df)*0.01
non_top_1_perc = df.sort_values('price',ascending=False).iloc[216:]
plt.figure(figsize = (12,8))
sns.scatterplot(x='long',y='lat',data=non_top_1_perc,hue='price',palette='RdYlGn',edgecolor=None,alpha=0.2)
df.head()
df.info()
df = df.drop('id',axis=1)
df.head()
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].apply(lambda date:date.month)
df['year'] = df['date'].apply(lambda date:date.year)
df.head(2)
plt.figure(figsize=(10,6))
sns.boxplot(x='year',y='price',data=df)
plt.figure(figsize=(10,6))
sns.boxplot(x='month',y='price',data=df)
df.groupby('month').mean()['price'].plot()
df = df.drop('date',axis=1)
df = df.drop('zipcode',axis=1)
df.head()
X = df.drop('price',axis=1)
y = df['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()

model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))
model.add(Dense(19,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(x=X_train,y=y_train.values,
          validation_data=(X_test,y_test.values),
          batch_size=128,epochs=400)
losses = pd.DataFrame(model.history.history)
losses.plot()
from sklearn.metrics import mean_squared_error,mean_absolute_error,explained_variance_score
predictions = model.predict(X_test)
mean_absolute_error(y_test,predictions)
mean_squared_error(y_test,predictions)
mean_squared_error(y_test,predictions)**0.5
explained_variance_score(y_test,predictions)
# Our predictions
plt.figure(figsize=(10,6))
plt.scatter(y_test,predictions)

# Perfect predictions
plt.plot(y_test,y_test,'r')
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
predictionslr = lr.predict(X_test)
mean_absolute_error(y_test,predictionslr)
mean_squared_error(y_test,predictionslr)
mean_squared_error(y_test,predictionslr)**0.5
explained_variance_score(y_test,predictionslr)
