import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import *

sns.set_style('whitegrid')

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
data = pd.read_csv('../input/heart.csv', delimiter=',')
data.head(3)
# Let's look at the distribution of people by sex

male = len(data[data.sex == 1])

female = len(data[data.sex == 0])

sns.countplot('sex', hue='target', data=data)

plt.title('Heart Disease: Sex')

plt.xlabel('Sex (0 = Female, 1 = Male)')

plt.xticks(rotation=0)

plt.legend(["Haven't Disease", "Have Disease"])

plt.ylabel('Frequency')

plt.show()
# Now, let's look at the distribution of people by age.

plt.figure(figsize=(9, 9))

plt.title('Heart Disease: Age')

plt.xlabel('Age')

plt.ylabel('Qantity')

data['age'].hist(bins=20)

plt.show()
data_v = data.iloc[:, 0:13].values

print('Feature vector:', data_v[1])
#Reducing the dimension to 2 for visualization.

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(data_v)

data_2d = pca.transform(data_v)



#Building a graph on a two-dimensional matrix

colormap = np.array(['red', 'lime'])

plt.figure(figsize=(10, 10))



for i in range(0, data_2d.shape[0]):

    if data['target'][i] == 1:

        c1 = plt.scatter(data_2d[i, 0], data_2d[i, 1], c='red')

    elif data['target'][i] == 0:

        c2 = plt.scatter(data_2d[i, 0], data_2d[i, 1], c='lime')



plt.title('People distribution')

plt.legend([c1, c2], ['Sick', 'Healthy'])
plt.figure(figsize=(12, 12))

sns.heatmap(data.corr(), annot=True)

plt.show()
print(data['target'].value_counts())
plt.figure(figsize=(7, 7))

data['target'].value_counts().plot(kind='bar', label='Target')

plt.legend()

plt.title('Distribution of target')
from sklearn.utils import resample



df_majority = data[data.target==1]

df_minority = data[data.target==0]

 



df_minority_upsampled = resample(df_minority, 

                                 replace=True,     

                                 n_samples=165,    

                                 random_state=123)

 



data = pd.concat([df_majority, df_minority_upsampled])

 



data['target'].value_counts()
from keras.models import Sequential

from keras import metrics

from keras.layers.core import Dense, Activation ,Dropout

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score
data_pop = data.drop("target", axis=1)

target = data["target"]

X_train, X_test, Y_train, Y_test = train_test_split(data_pop, target, test_size=0.3, random_state=0)



from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#Keras neural network



model = Sequential()

model.add(Dense(15, init = 'uniform', activation='relu', input_dim=13))

model.add(Dense(10, init = 'uniform', activation='relu'))

model.add(Dense(6, init = 'uniform', activation='relu'))

model.add(Dense(1, init = 'uniform', activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



#Fitting

model.fit(X_train, Y_train, epochs=130)
#Testing on a test sample

Y_pred_nn = model.predict(X_test)

rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded
score_nn = round(accuracy_score(Y_pred_nn, Y_test)*100,2)

score_f1 = round(f1_score(Y_pred_nn, Y_test)*100, 2)

print("Accuracy score: " + str(score_nn) + " %")

print("F1 score: " + str(score_f1) + "%")