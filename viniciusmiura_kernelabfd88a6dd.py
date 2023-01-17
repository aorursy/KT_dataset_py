import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

sns.set(style='white', context='notebook', palette='deep')
treino= pd.read_csv('../input/train_data.csv',sep=r'\s*,\s*',
                     engine='python',
                     na_values="?")

#Modificando a visalização da base de treino
treino.iloc[0:20,:]
treino.drop(['Id'],axis =1 )
treino['income']=treino['income'].map({'<=50K': 0, '>50K': 1, '<=50K.': 0, '>50K.': 1})

numeric_features = ['age','fnlwgt','education.num','capital.gain','capital.loss','hours.per.week','income']

cat_features = ['workclass','education','marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# Drop the data you don't want to use
treino.drop(labels=["sex","marital.status","workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)
print('Dataset with Dropped Labels')
print(treino.head())
treino.shape
treino.head()
features = treino[["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]]
target = treino['income']
gnb = GaussianNB()

gnb.fit(features, target)
scores = cross_val_score(gnb, features, target, cv=10)

print(scores)

num_trees = 100
max_features = 3
array = treino.values
X = array[:,0:6]
Y = array[:,7]
print('Split Data: X')
print(X)
print('Split Data: Y')
print(Y)
validation_size = 0.20
seed = 7
num_folds = 10
scoring = 'accuracy'
features, X_validation, target, Y_validation = train_test_split(X,Y,
    test_size=validation_size,random_state=seed)
random_forest = RandomForestClassifier(n_estimators=250,max_features=5)
random_forest.fit(features, target)
predictions = random_forest.predict(X_validation)
print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
features.shape
decision_tree = DecisionTreeClassifier()
decision_tree.fit(features, target)
predictions = decision_tree.predict(X_validation)
print("Accuracy: %s%%" % (100*accuracy_score(Y_validation, predictions)))
teste=pd.read_csv('../input/test_data.csv',sep=r'\s*,\s*',
                     engine='python',
                     na_values="?")
teste.iloc[0:20,:]
teste.shape

teste.drop(labels=["marital.status","sex","workclass","education","occupation","relationship","race","native.country"], axis = 1, inplace = True)
features_test = teste[["age","fnlwgt","education.num","capital.gain","capital.loss","hours.per.week"]]
teste.shape
teste.head()
x_val_test = features_test
y_val_test = random_forest.predict(x_val_test)

dfSave = pd.DataFrame(data={"Id" : teste["Id"], "income" : y_val_test})
dfSave['Id'] = dfSave['Id'].astype(int)
dfSave["income"] = dfSave["income"].map({0:'<=50K', 1:'>50K'})
pd.DataFrame(dfSave[["Id", "income"]], columns = ["Id", "income"]).to_csv("Output.csv", index=False)
dfSave


