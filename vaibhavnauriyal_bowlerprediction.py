import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset=pd.read_csv('/kaggle/input/cricket-player-info/cricket_bowler_information.csv')
#checking for missing values

dataset.isnull().sum()
#bowlingStyle contains categorical values, so it is best to fill the missing values with mode

dataset['bowlingStyle'].fillna(dataset['bowlingStyle'].mode()[0], inplace=True)





#rest of the missing values are replaced by medians of the respective columns

def fmedian(df,col):

    median_value=df[col].median()

    df[col].fillna(median_value, inplace=True)

    

fmedian(dataset,'consistency')

fmedian(dataset,'Average_Career')

fmedian(dataset,'Strike_rate_Career')

fmedian(dataset,'form')

fmedian(dataset,'Average_Yearly')

fmedian(dataset,'Strike_rate_Yearly')

fmedian(dataset,'opposition')

fmedian(dataset,'Average_opposition')

fmedian(dataset,'Strike_rate_opposition')

fmedian(dataset,'Strike_rate_venue')

fmedian(dataset,'venue')

fmedian(dataset,'Average_venue')
dataset.dtypes
from sklearn.preprocessing import LabelEncoder



def encode(df,col):

    le = LabelEncoder()

    df[col] = le.fit_transform(df[col])



encode(dataset,'Innings Player')

encode(dataset,'Ground')

encode(dataset,'Country')

encode(dataset,'Opposition')

encode(dataset,'Day')

encode(dataset,'bowlingStyle')
dataset.FF
from matplotlib import pyplot as plt

import seaborn as sns

corr = dataset.corr()

fig, ax = plt.subplots(figsize=(30, 18))

colormap = sns.diverging_palette(220, 10, as_cmap=True)

dropSelf = np.zeros_like(corr)

dropSelf[np.triu_indices_from(dropSelf)] = True

colormap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)

plt.title('Wickets - Features Correlations')

plt.show()
#Balls bowled and Overs Bowled are perfectly corelated, so it is necessary to remove one of them

dataset.drop('Overs_Bowled',axis=1,inplace=True)
target=dataset['Wickets_Taken']

train=dataset.drop('Wickets_Taken', axis=1)
target.value_counts()
#applying SMOTE

from imblearn.combine  import SMOTETomek

smk=SMOTETomek(random_state=42)

train_new,target_new=smk.fit_sample(train,target)
#splitting dataset into 80% train and 20% test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_new, target_new, test_size = 0.20, random_state = 0)
from sklearn.naive_bayes import GaussianNB

nb_model=GaussianNB()

nb_model.fit(X_train,y_train)

y_pred=nb_model.predict(X_test)
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



accuracy=accuracy_score(y_test,y_pred) 

precision=precision_score(y_test,y_pred,average='weighted')

recall=recall_score(y_test,y_pred,average='weighted')

f1=f1_score(y_test,y_pred,average='weighted')



print('Accuracy - {}'.format(accuracy))

print('Precision - {}'.format(precision))

print('Recall - {}'.format(recall))

print('F1 - {}'.format(f1))
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier(criterion='gini', splitter='best',

                             max_depth=16, min_samples_split=2,

                             min_samples_leaf=1, min_weight_fraction_leaf=0.0,

                             max_features=None, random_state=None,

                             max_leaf_nodes=None, min_impurity_decrease=0.0, 

                             min_impurity_split=None, class_weight=None, 

                             presort='deprecated', ccp_alpha=0.0)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



accuracy=accuracy_score(y_test,y_pred) 

precision=precision_score(y_test,y_pred,average='weighted')

recall=recall_score(y_test,y_pred,average='weighted')

f1=f1_score(y_test,y_pred,average='weighted')



print('Accuracy - {}'.format(accuracy))

print('Precision - {}'.format(precision))

print('Recall - {}'.format(recall))

print('F1 - {}'.format(f1))
from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier(n_estimators=50)

model.fit(X_train,y_train)

y_pred=model.predict(X_test)
from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import f1_score



accuracy=accuracy_score(y_test,y_pred) 

precision=precision_score(y_test,y_pred,average='weighted')

recall=recall_score(y_test,y_pred,average='weighted')

f1=f1_score(y_test,y_pred,average='weighted')



print('Accuracy - {}'.format(accuracy))

print('Precision - {}'.format(precision))

print('Recall - {}'.format(recall))

print('F1 - {}'.format(f1))