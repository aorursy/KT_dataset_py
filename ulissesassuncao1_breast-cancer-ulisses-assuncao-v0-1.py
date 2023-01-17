import pandas as pd



train = pd.read_csv('../input/data.csv')

train.head()
import seaborn as sns

import matplotlib.pyplot as plt



g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'radius_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'texture_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'perimeter_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'area_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'smoothness_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'compactness_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'concavity_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'concave points_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'symmetry_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'fractal_dimension_mean', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'radius_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'texture_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'perimeter_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'area_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'smoothness_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'compactness_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'concavity_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'concave points_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'symmetry_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'fractal_dimension_se', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'radius_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'texture_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'perimeter_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'area_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'smoothness_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'compactness_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'concavity_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'concave points_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'symmetry_worst', bins=20)
g = sns.FacetGrid( train, col='diagnosis')

g.map(plt.hist, 'fractal_dimension_worst', bins=20)
train.drop(['perimeter_mean',

           'texture_se',

           'concavity_se',

           'compactness_se',

           'symmetry_se',

           'radius_se',

           'fractal_dimension_se',

           'smoothness_worst',

           'smoothness_mean',

           'symmetry_mean',

           'symmetry_worst'],axis=1,)

train=train.drop(['Unnamed: 32'],axis=1)
train.head()
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

train['diagnosis'] = labelencoder.fit_transform(train['diagnosis'])



train.head()
classe = train['diagnosis']

atributos = train.drop('diagnosis', axis=1)

atributos.head()
from sklearn.model_selection import train_test_split

atributos_train, atributos_test, class_train, class_test = train_test_split(atributos, classe, test_size = 0.25 )

atributos_train.describe()
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=3, random_state =0)

model = dtree.fit(atributos_train, class_train)
from sklearn.metrics import accuracy_score

classe_pred = model.predict(atributos_test)

acc = accuracy_score(class_test, classe_pred)

print("My Decision Tree acc is {}".format(acc))