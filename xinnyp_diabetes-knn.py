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
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head(3)
df.describe()
df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] =  df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
df.describe()
%matplotlib inline
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')

p = df.hist(figsize=(6,10))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df.drop('Outcome', axis=1)
y = df.Outcome
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=42, stratify=y)
X_train.head()
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

features_mean_impute = ['Glucose', 'BloodPressure']
features_median_impute = ['SkinThickness', 'Insulin', 'BMI']

imputer_mean =  SimpleImputer(strategy='mean')
imputer_median =SimpleImputer(strategy='median')

preprocessor= ColumnTransformer(transformers=[('imputer_mean', imputer_mean, features_mean_impute), 
                                                ('imputer_median', imputer_median, features_median_impute)])

# X_train = preprocessor.fit(X_train)
# X_test = preprocessor.fit(X_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error

scaler = StandardScaler()

test_scores = []


for i in range(1,15):
    model = KNeighborsClassifier(n_neighbors=i)
    classifiers = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('scaler', scaler),
                                  ('model', model)])

    classifiers.fit(X_train, y_train)
    test_scores.append(classifiers.score(X_test,y_test))
max_test_score = max(test_scores)
test_scores_ind = [i for i, v in enumerate(test_scores) if v == max_test_score]
print('Max test score {} % and k = {}'.format(max_test_score*100,list(map(lambda x: x+1, test_scores_ind))))
plt.figure(figsize=(12,5))
p = sns.lineplot(range(1,15),test_scores,marker='*',label='Test Score')
