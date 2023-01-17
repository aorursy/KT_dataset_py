# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dfs = {file.split('.')[0] : pd.read_csv("../input/"+file) for file in os.listdir("../input")}

for name, df in dfs.items():

    print(name)

    print(df.head(1))

    print(df.info())
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



import logging

logging.getLogger('sklearn').setLevel(logging.ERROR) # Ex: module_name = seaborn

import warnings

warnings.filterwarnings("ignore")



from dask.distributed import Client

client = Client(n_workers=4, threads_per_worker=1)

client

categorial_column_types = ['object','datetime']

intervals = df.select_dtypes(exclude=categorial_column_types).columns.values.tolist()

categorials = df.select_dtypes(include=categorial_column_types).columns.values.tolist()

intervals, categorials
le = preprocessing.LabelEncoder()
df = dfs['train']

test = dfs['test']

# df, test = df.dropna(), test.dropna()

df, test = df.interpolate(method='linear'), test.interpolate(method='linear')
features = ['Pclass', 'Age']

target = ['Survived']

x_train, x_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.33, random_state=42)

from sklearn import svm



model1 = svm.OneClassSVM()
from tpot import TPOTClassifier



model2 = TPOTClassifier(

    generations=1,

    population_size=10,

    cv=2,

    n_jobs=-1,

    random_state=0,

    verbosity=0,

    use_dask=True

)
from sklearn.ensemble import AdaBoostClassifier

from sklearn.tree import DecisionTreeClassifier



model3 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),

                          n_estimators=300, random_state=7)
from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier



clf1 = DecisionTreeClassifier(max_depth=4)

clf2 = KNeighborsClassifier(n_neighbors=7)

clf3 = SVC(gamma=.1, kernel='rbf', probability=True)

model4 = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),

                                    ('svc', clf3)],

                        voting='soft', weights=[2, 1, 2])
from sklearn.neural_network import MLPClassifier



model5 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,

                    solver='sgd', verbose=10, tol=1e-4, random_state=1,

                    learning_rate_init=.1)
model = model5
model.fit(x_train, y_train)
y_true = model.predict(x_test)
def transform_classes(raw_classes : 'np.array') -> 'np.array':

    transform_classes = {-1:0}

    func = np.vectorize(lambda x: transform_classes.get(x, x))

    return func(raw_classes)
str(accuracy_score(y_true, transform_classes(y_test)) * 100.0) + '%'
predictions = model.predict(test[features])

predictions
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived': transform_classes(predictions)})

submission.head()
filename = 'Titanic Predictions 5.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)