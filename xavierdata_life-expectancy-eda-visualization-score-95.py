import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
df = pd.read_csv('../input/life-expectancy-who/Life Expectancy Data.csv')
df.head()
df.columns
df.rename(columns=lambda x: x.strip(), inplace=True)
df.isna().sum()
df = df.fillna(df.mean())
df_2000=(df[df.Year==2000]
    .groupby("Country")
    ["Country", "Life expectancy"]
    .median()
    .sort_values(by="Life expectancy", ascending=True))

df_2000.plot(kind='bar', figsize=(50,10), fontsize=12)
plt.title("Life expectancy per Country in 2000",fontsize=30)
plt.xlabel("Country",fontsize=15)
plt.ylabel("Life expectancy 2015",fontsize=15)
plt.show()
df_2011=(df[df.Year==2011]
    .groupby("Country")
    ["Country", "Life expectancy"]
    .median()
    .sort_values(by="Life expectancy", ascending=True))

df_2011.plot(kind='bar', figsize=(50,10), fontsize=12)
plt.title("Life expectancy per Country in 2011",fontsize=30)
plt.xlabel("Country",fontsize=15)
plt.ylabel("Life expectancy 2015",fontsize=15)
plt.show()
df_2015=(df[df.Year==2015]
    .groupby("Country")
    ["Country", "Life expectancy"]
    .median()
    .sort_values(by="Life expectancy", ascending=True))

df_2015.plot(kind='bar', figsize=(50,10), fontsize=12)
plt.title("Life expectancy per Country for 2015",fontsize=30)
plt.xlabel("Country",fontsize=15)
plt.ylabel("Life expectancy 2015",fontsize=15)
plt.show()
life_expectancy_per_country = df.groupby('Country')['Life expectancy'].mean().sort_values(ascending=True)
life_expectancy_per_country.plot(kind='bar', figsize=(50,10), fontsize=12)
plt.title("Life expectancy mean per Country from 2000 to 2015",fontsize=30)
plt.xlabel("Country",fontsize=15)
plt.ylabel("Life expectancy",fontsize=15)
plt.show()
plt.figure(figsize=(10,10))
plt.bar(df.groupby('Status')['Status'].count().index,df.groupby('Status')['Life expectancy'].mean())
plt.xlabel("Status",fontsize=15)
plt.ylabel("Life expectancy",fontsize=15)
plt.title("Life expectancy for developed and developing country",fontsize=20)
plt.show()
plt.figure(figsize=(20,7))
plt.subplot(1, 2, 1)
plt.scatter(df["Alcohol"], df["Life expectancy"])
plt.xlabel("Alcohol",fontsize=15)
plt.ylabel("Life expectancy",fontsize=15)
plt.title("Life expectancy - Alcohol",fontsize=17)

plt.subplot(1, 2, 2)
plt.scatter(df["Schooling"], df["Life expectancy"])
plt.xlabel("Schooling",fontsize=15)
plt.ylabel("Life expectancy",fontsize=15)
plt.title("Life expectancy - Schooling",fontsize=17)
df
df['Status'].value_counts()
def encode_status(x):
    if x == 'Developed':
        return 1
    else:
        return 0
df['Status'] = df['Status'].apply(encode_status)
df = pd.concat([df, pd.get_dummies(df['Country'], prefix='Country', drop_first=True)], axis=1)
df = df.drop(['Country'], axis=1)
df
X = df.drop(['Life expectancy'], axis=1)

y = df['Life expectancy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=21)
lr = LinearRegression()

lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
mean_squared_error(y_test, y_pred)

r2_score(y_test, y_pred)*100