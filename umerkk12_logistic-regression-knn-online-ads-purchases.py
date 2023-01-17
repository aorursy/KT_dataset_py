import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
df = pd.read_csv('../input/logistic-regression/Social_Network_Ads.csv')
df.head()
df.info()
msno.bar(df)
Gender = pd.get_dummies(df['Gender'], drop_first=True)

df = pd.concat([df, Gender], axis=1)
df.head()
df['Male'] = df['Male'].astype(float)
df.info()

plt.figure(figsize=(10,6))
sns.countplot(x='Gender', data=df, palette='magma', alpha=0.4)
sns.despine(left=True)

plt.figure(figsize=(10,6))
sns.boxplot(x='Gender', y='EstimatedSalary', data=df, palette='plasma')
sns.despine(left=True)
plt.figure(figsize=(14,10))
sns.scatterplot(x='EstimatedSalary', y='Age', data=df, hue=df['Purchased'], palette='GnBu', s=100, alpha=0.9)
sns.despine(left=True)
df_purc = df.groupby(df['Purchased']).mean()
df_purc = df_purc.reset_index()
fig, ax =plt.subplots(nrows= 1, ncols = 3, figsize= (14,6))
w = sns.barplot(x='Purchased', y= 'Age', data=df_purc, palette='GnBu', ax=ax[0])
i = sns.barplot(x='Purchased', y= 'EstimatedSalary', data=df_purc, palette='magma', ax=ax[1])
h = sns.barplot(x='Purchased', y= 'Male', data=df_purc, palette='GnBu', ax=ax[2])
sns.despine(left=True)

w.set_title('Purchased or Not: Average Age')
w.set_ylabel('Age')

i.set_title('Purchased or Not: Average Estimated Salary')
i.set_ylabel('Estimated Salary')

h.set_title('Purchased or Not: Male = 1, Female = 0')
h.set_ylabel('Gender')

X = df.drop(['User ID', 'Gender', 'Purchased'], axis=1)
y = df['Purchased']
X.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_data = pd.DataFrame(scaled_data, columns= X.columns)
scaled_data.head()
from sklearn.feature_selection import SelectKBest,chi2
test=SelectKBest(score_func=chi2,k=2)
fit=test.fit(X,y)
print(fit.scores_)
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(scaled_data,y, test_size=0.3)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
prediction = logreg.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction))
from sklearn.neighbors import KNeighborsClassifier
error_rate = []
for i in range(1,16):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    prediction_k = knn.predict(X_test)
    error_rate.append(np.mean(prediction_k !=y_test))
error_rate = pd.DataFrame(error_rate,range(1,16), columns=['Error Rate']).reset_index()
error_rate = error_rate.rename(columns={'index': 'K Value'})
plt.figure(figsize=(12,7))
g = sns.set(style="white")
g = sns.lineplot(x='K Value', y='Error Rate', data=error_rate, color='green')
f = sns.scatterplot(x='K Value', y='Error Rate', data=error_rate, color='red', s=100)

g.set_title('Error Rate Analysis')
g.set_ylabel('Error Rate')
g.set_xlabel('K Value')
g =sns.despine(left=True)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
prediction_5 = knn.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test, prediction_k))