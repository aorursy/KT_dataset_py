import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/classified-dataset/Classified Data', index_col = 0)

df.head()
# Logistic Model training and Predicting



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix



X = df[['WTT', 'PTI', 'EQW', 'SBI', 'LQE', 'QWG', 'FDJ', 'PJF', 'HQE', 'NXJ']]

y = df['TARGET CLASS']



train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 100)



model = LogisticRegression()

model.fit(train_X, train_y)



prediction = model.predict(test_X)
# Finding Accuracy



print(confusion_matrix(test_y, prediction))

print('\n')

print(classification_report(test_y, prediction))
df.head()
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

scaler.fit(df.drop(['TARGET CLASS'], axis = 1))



scaled_feat = scaler.transform(df.drop(['TARGET CLASS'],axis = 1))



#Final Standardised Data

df_feat = pd.DataFrame(scaled_feat, columns = df.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, confusion_matrix



train_X,test_X, train_y, test_y = train_test_split(df_feat, df['TARGET CLASS'], test_size = 0.4, random_state = 100)





# Model training

model = KNeighborsClassifier(n_neighbors= 1) # Taking k value equals to 1

model.fit(train_X, train_y)



pred = model.predict(test_X)
# Finding Accuracy



print(confusion_matrix(test_y, pred))

print('\n')

print(classification_report(test_y, pred))
# Using elbow method



error_rate = []

for i in range(1, 60):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(train_X, train_y)

    pred_i = knn.predict(test_X)

    error_rate.append(np.mean(pred_i != test_y))
# Plotting various K values with respect to the error_rate, to find perfect K value



sns.set_style('whitegrid')

plt.figure(figsize=(10,6))

plt.plot(range(1,60), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markerfacecolor = 'red',markersize = 10)

plt.title('Error rate vs K value')

plt.xlabel('K')

plt.ylabel('Error rate')
# Prediction with K=35



model = KNeighborsClassifier(n_neighbors=35)

model.fit(train_X, train_y)



predict = model.predict(test_X)
# Finding Accuracy



print(confusion_matrix(test_y, predict))

print('\n')

print(classification_report(test_y, predict))
# Importing SVM model and Predicting 



from sklearn.svm import SVC



svm_model = SVC()

svm_model.fit(train_X, train_y)



predict = svm_model.predict(test_X)
# Accuracy 



print(confusion_matrix(test_y, predict))

print('\n')

print(classification_report(test_y, predict))
# Importing the Grid Search model and predicting



from sklearn.model_selection import GridSearchCV



parameters = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}



grid = GridSearchCV(SVC(), param_grid=parameters)



# Model Training



grid.fit(train_X, train_y)
predict = grid.predict(test_X)



# Finding Accuracy



print(confusion_matrix(test_y, predict))

print('\n')

print(classification_report(test_y, predict))