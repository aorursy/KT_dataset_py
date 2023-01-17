import matplotlib.pyplot as plt 

import pandas as pd 

import numpy as np 

import seaborn as sns 

import warnings



warnings.filterwarnings("ignore")

df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

print(df.head)
print(df.info()) 
copied_data = df.dropna(axis=1)
from sklearn.preprocessing import LabelBinarizer 





encoder = LabelBinarizer()

feature = copied_data['diagnosis']

encoded_feature = encoder.fit_transform(feature)

copied_data['diagnosis'] = encoded_feature

most_correlated = copied_data.corr().abs()['diagnosis'].sort_values(ascending=False)



#We will chose top 10 most correlated features

most_correlated = most_correlated[:10]

training_set = copied_data.loc[:, most_correlated.index]

print(most_correlated)

sns.catplot(x="diagnosis", y="radius_mean", data=training_set)

plt.show()
for feature in training_set.columns.values:

    sns.catplot(x='diagnosis', y=feature, data=training_set)

    
plt.figure(figsize=(10, 10))

sns.heatmap(training_set.corr(), annot=True, fmt='.0%')
corr_matrix = copied_data.corr()

print(corr_matrix['diagnosis'])
from sklearn.preprocessing import StandardScaler 

from sklearn.model_selection import train_test_split



labels = training_set['diagnosis']

new_training_set = training_set.drop('diagnosis', axis=1)



X_train, X_test, y_train, y_test = train_test_split(new_training_set, labels, test_size=0.2, random_state=0)



scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression 

from sklearn.ensemble import RandomForestClassifier 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.model_selection import cross_val_score



models = {'Logistic':LogisticRegression(), 'forest':RandomForestClassifier(n_estimators=10, criterion='gini', random_state=0),

         'tree':DecisionTreeClassifier(criterion='gini', random_state=0)}



trained_models = list()



for value in models.values():

    model = value

    model.fit(X_train, y_train)

    acc = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=10)

    print(acc.mean())

    trained_models.append(model)
from sklearn.metrics import confusion_matrix 



predictions = trained_models[0].predict(X_test)

conf_matrix = confusion_matrix(y_test, predictions)



dataframe = pd.DataFrame(conf_matrix)

sns.heatmap(dataframe, annot=True, cbar=None, cmap="Reds")

plt.title("Confusion Matrix"), plt.tight_layout()

plt.ylabel("True Class"), plt.xlabel("Predicted Class")

plt.show()
accuracy = (dataframe[0][0] + dataframe[1][1]) / (dataframe[0][1] + dataframe[1][0]+ dataframe[0][0] + dataframe[1][1]) 

print('Test accuracy:', accuracy)