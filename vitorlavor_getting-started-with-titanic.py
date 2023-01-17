# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Import matplotlib for data visualisation

import seaborn as sns # Statistical data visualization



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data
print('Number of Training Examples = {}'.format(train_data.shape[0]))

print('Number of Test Examples = {}\n'.format(test_data.shape[0]))

print(train_data.columns)
train_data.info()
all_data = pd.concat([train_data, test_data], sort=True).reset_index(drop=True)



print(all_data.info())
all_data.isnull().sum()
print("% of missing values Age: ", all_data.isnull().sum()["Age"]/all_data.shape[0]*100)

print("% of missing values Cabin: ", all_data.isnull().sum()["Cabin"]/all_data.shape[0]*100)

print("% of missing values Embarked: ", all_data.isnull().sum()["Embarked"]/all_data.shape[0]*100)

print("% of missing values Fare: ", all_data.isnull().sum()["Fare"]/all_data.shape[0]*100)
# Valores Nulos na coluna Embarked



train_data[train_data['Embarked'].isnull()] 
all_data['Age'] = all_data['Age'].fillna(all_data['Age'].median()) # Modificação dos valores nulos de Age para o valor de sua mediana

all_data['Embarked'] = all_data["Embarked"].fillna("S") # Modificação dos valores nulos de Embarked por "S"

all_data = all_data.drop(columns = ['Cabin']) # Desprezar as coluna Cabin

all_data['Fare'] = all_data['Fare'].fillna(all_data['Fare'].median()) # Modificação dos valores nulos de fare para o valor de sua mediana
# Checando o número de NANs após substituições



all_data.isnull().sum()
# Dividindo o dados novamente em train e test datasets



train_data, test_data = all_data.loc[:890], all_data.loc[891:].drop(['Survived'], axis=1)

train_data["Survived"] = train_data["Survived"].astype(float).astype(int)
# Training ser Correlations



plt.figure(figsize=(20,10)) 

sns.heatmap(train_data.drop(['PassengerId'], axis=1).corr(), annot=True)
# Test set Correlations



plt.figure(figsize=(20,10)) 

sns.heatmap(test_data.drop(['PassengerId'], axis=1).corr(), annot=True)
cont_features = ['Age', 'Fare']

surv = train_data['Survived'] == 1



fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))

plt.subplots_adjust(right=1.5)



for i, feature in enumerate(cont_features):    

    # Distribution of survival in feature

    sns.distplot(train_data[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])

    sns.distplot(train_data[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

    

    # Distribution of feature in dataset

    sns.distplot(train_data[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])

    sns.distplot(test_data[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

    

    axs[0][i].set_xlabel('')

    axs[1][i].set_xlabel('')

    

    for j in range(2):        

        axs[i][j].tick_params(axis='x', labelsize=20)

        axs[i][j].tick_params(axis='y', labelsize=20)

    

    axs[0][i].legend(loc='upper right', prop={'size': 20})

    axs[1][i].legend(loc='upper right', prop={'size': 20})

    axs[0][i].set_title('Distribution of Survival in {}'.format(feature), size=20, y=1.05)



axs[1][0].set_title('Distribution of {} Feature'.format('Age'), size=20, y=1.05)

axs[1][1].set_title('Distribution of {} Feature'.format('Fare'), size=20, y=1.05)

        

plt.show()
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']



fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(20, 20))

plt.subplots_adjust(right=1.5, top=1.25)



for i, feature in enumerate(cat_features, 1):    

    plt.subplot(2, 3, i)

    sns.countplot(x=feature, hue='Survived', data=train_data)

    

    plt.xlabel('{}'.format(feature), size=20, labelpad=15)

    plt.ylabel('Passenger Count', size=20, labelpad=15)    

    plt.tick_params(axis='x', labelsize=20)

    plt.tick_params(axis='y', labelsize=20)

    

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 18})

    plt.title('Count of Survival in {} Feature'.format(feature), size=20, y=1.05)



plt.show()
train_data
test_data
cat_features = ['Pclass', 'Sex', 'Embarked']

encoded_features = []

for df in dfs:

    for feature in cat_features:

        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()

        n = df[feature].nunique()

        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]

        encoded_df = pd.DataFrame(encoded_feat, columns=cols)

        encoded_df.index = df.index

        encoded_features.append(encoded_df)

train_data = pd.concat([train_data, *encoded_features[:3]], axis=1)

test_data = pd.concat([test_data, *encoded_features[3:]], axis=1)
drop_cols_train = ['Embarked','Name', 'PassengerId', 'Pclass', 'Sex', 'Ticket','Survived']

drop_cols_test = ['Embarked','Name', 'PassengerId', 'Pclass', 'Sex', 'Ticket']
X_train = StandardScaler().fit_transform(train_data.drop(columns=drop_cols_train))

y_train = train_data['Survived'].values

X_test = StandardScaler().fit_transform(test_data.drop(columns=drop_cols_test))



print('X_train shape: {}'.format(X_train.shape))

print('y_train shape: {}'.format(y_train.shape))

print('X_test shape: {}'.format(X_test.shape))
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)

model.fit(X_train, y_train)

predictions = model.predict(X_test)



output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})

# output = pd.Series(predictions, index = test_data['PassengerId'])

output.to_csv('my_submission.csv', index=False)

print("Your submission was successfully saved!")