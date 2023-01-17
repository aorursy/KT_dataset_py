#importing librairies and modules



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report



#ingesting the datasets



gold_staters = pd.read_csv('../input/mantis-ancient-coins/gold_staters.csv')

silver_tetradrachms = pd.read_csv('../input/mantis-ancient-coins/silver_tetradrachms.csv')

gold_aurei = pd.read_csv('../input/mantis-ancient-coins/gold_aureus.csv')

silver_denarii = pd.read_csv('../input/mantis-ancient-coins/silver_denarius.csv')



#verifying the shapes of the DataFrames



print('Number of gold staters, number of features:', gold_staters.shape)

print('Number of silver tetradrachms, number of features:', silver_tetradrachms.shape)

print('Number of gold aurei, number of features:', gold_aurei.shape)

print('Number of silver denarii, number of features:', silver_denarii.shape)
#adding the 4 dataframes together



coins = gold_staters.append([silver_denarii, silver_tetradrachms, gold_aurei])



#number of coins and features in the new dataframe



print('Total number of coins', len(coins))

print('Total number of features:', len(coins.columns))



coins.head()
#keeping the relevant columns



coins = coins[['Material', 'Denomination','Weight','Diameter']]



#grouping tags into 4 labels



coins['Label'] = coins['Material']+ '_' + coins['Denomination']

del coins['Material']

del coins ['Denomination']



#removing NaN values



coins = coins.dropna().reset_index(drop=True)



print('We have', len(coins), 'coins left in the dataset without NaN values.')

coins.head()
#computing the number of coins per label



coins_counts = coins['Label'].value_counts()

print(coins_counts)



#creating the plot



coins_counts.plot(kind='bar', figsize=(10,5), color='tan', rot=0)

plt.title('Distribution per Label', fontsize=20, color='rosybrown')

plt.ylabel('Frequency');
#computing stats per label



average_coins = coins[['Label','Weight','Diameter']].groupby('Label').describe()

average_coins
#drawing a scatter plot



plt.scatter(coins['Weight'], coins['Diameter'], c='rosybrown')

plt.title('Coins Distribution', fontsize=20, color='sandybrown')

plt.xlabel('Weight')

plt.ylabel('Diameter');
#searching for outliers in the dataset



is_outlier = (coins['Diameter'] > 200) | (coins['Weight'] > 40)

outliers = coins[is_outlier]

outliers
#dropping the outliers



coins = coins.drop([1863,2666,2667])



#drawing a new scatter plot



is_goldstater = coins['Label'].str.contains('Gold_stater')

goldstaters = coins[is_goldstater]



is_tetradrachm = coins['Label'].str.contains('Silver_tetradrachm')

tetradrachms = coins[is_tetradrachm]



is_denarius = coins['Label'].str.contains('silver_denarius')

silverdenarius = coins[is_denarius]



is_aureus = coins['Label'].str.contains('Gold_Aureus')

goldaureus = coins[is_aureus]



plt.figure(figsize=(12, 8))

plt.scatter(goldstaters['Weight'], goldstaters['Diameter'], c='yellow')

plt.scatter(tetradrachms['Weight'], tetradrachms['Diameter'], c='green')

plt.scatter(silverdenarius['Weight'], silverdenarius['Diameter'], c='blue')

plt.scatter(goldaureus['Weight'], goldaureus['Diameter'], c='red')

plt.title('Coins Distribution per Denomination', fontsize=20, color='lightcoral')

plt.xlabel('Weight')

plt.ylabel('Diameter');
#dividing data into features and labels



X = coins.iloc[:, 0:2].values

y = coins.iloc[:, 2].values



#encoding the labels from strings to numbers



le = LabelEncoder()

y = le.fit_transform(y)

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

print('Here are the codes for each label:')

print(le_name_mapping)
#dividing the dataset into training and test set (default 75/25)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)



print('Number of coins in the training split:', len(X_train))

print('Number of coins in the test split:', len(X_test))
#fitting the classifier



classifier = KNeighborsClassifier(n_neighbors=5)

classifier.fit(X_train, y_train)



#testing the classifier



y_pred = classifier.predict(X_test);
#calculating the accuracy of the model



classifier.score(X_test, y_test)
print(classification_report(y_test, y_pred))
#calculating error for k values



error = []



for i in range(1, 40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    pred_i = knn.predict(X_test)

    error.append(np.mean(pred_i != y_test))
#drawing the line plot



plt.figure(figsize=(12, 6))

plt.plot(range(1, 40), error, color='sandybrown', marker='o', markerfacecolor='sandybrown', markersize=10)

plt.title('Error Rate of k', color='rosybrown', fontsize=20)

plt.xlabel('K Value')

plt.ylabel('Mean Error');
#fitting the classifier



classifier = KNeighborsClassifier(n_neighbors=8)

classifier.fit(X_train, y_train)



#testing the classifier



y_pred = classifier.predict(X_test)



#calculating the accuracy of the model



classifier.score(X_test, y_test)
#creating color maps

cmap_light = ListedColormap(['coral', 'lemonchiffon','springgreen','lightblue'])

cmap_bold = ListedColormap(['red', 'yellow','green','blue'])



#calculating min max and limits

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),

np.arange(y_min, y_max, 0.02))



#predicting class using data and kNN classifier

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])



#putting the result into a color plot

Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 8))

plt.pcolormesh(xx, yy, Z, cmap=cmap_light)



#plotting the training points

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)

plt.xlim(xx.min(), xx.max())

plt.ylim(yy.min(), yy.max())

plt.title('Coins Classification (k=8)', fontsize=20, color='lightcoral')

plt.xlabel('Weight')

plt.ylabel('Diameter');
#let's play with random values



print('7 grams & 19 mm -- should get a gold aureus (0) :', classifier.predict([[7,19]]))

print('8 grams & 18 mm -- should get a gold stater (1) :', classifier.predict([[8,18]]))

print('16 grams & 28 mm -- should get a silver tetradrachm (2) :', classifier.predict([[16,28]]))

print('3 grams & 19 mm -- should get a silver denarius (3) :', classifier.predict([[3,19]]))