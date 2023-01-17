# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)







# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import h2o

from h2o.automl import H2OAutoML

h2o.init(max_mem_size='16G')
data = h2o.import_file('/kaggle/input/titanic/train.csv')



# If you only want to use a few features, manually listing them is better.

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']



# If you have a large number of features (uncomment below line), it is better to just pull out the list of columns. But make sure that you drop all unnecessary features if you do that.

#features = data.columns



output = 'Survived'
aml = H2OAutoML(max_models=100, max_runtime_secs=1500, seed=1)

aml.train(x=features, y=output, training_frame=data)
lb = aml.leaderboard

lb.head()
#Read the test data

titanic_test = h2o.import_file('/kaggle/input/titanic/test.csv')



#Make predictions on the test data. You don't need to feed in the columns you are using. H2O will automatically select them based on the columns in the training data.

preds = aml.predict(titanic_test)





titanic_test['Survived'] = preds

sub = titanic_test[['PassengerId', 'Survived']]



#Converting H2o Frame to pandas DataFrame for submission.

subs = sub.as_data_frame(use_pandas=True)
subs['Survived'] = subs['Survived'] > 0.5

subs['Survived'] = subs['Survived'].astype(int)

subs.to_csv('h2o_sub.csv', index=False)
!pip install tpot
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
#Separate the target

y = train['Survived']
combined = pd.concat([train, test])

len_train = len(train)

len_test = len(test)
combined.drop(['Survived', 'PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)
combined['Age'] = combined['Age'].fillna(-999)

combined['Fare'] = combined['Fare'].fillna(-999)

combined['Embarked'] = combined['Embarked'].fillna('Unkown')



combined = pd.get_dummies(combined)
pd.isnull(combined).any()
train = combined.head(len_train)

test = combined.tail(len_test)

print("Train shape is " + str(train.shape))

print("Test shape is " + str(train.shape))
from tpot import TPOTClassifier



tpot = TPOTClassifier(generations=5, verbosity=2, random_state=None)



tpot.fit(train, y)
a = tpot.predict(test)

sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')

sub['Survived'] = a

sub.to_csv('tpot_sub.csv', index=False)
tpot.export('tpot_titanic_pipeline.py')