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
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use("bmh")
df = pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head(3)
plt.rcParams["figure.figsize"] = (20,14)
sns.countplot(x='age', data=df, palette='Wistia')
sns.countplot(x='age', hue='anaemia', data=df)
plt.title('Age and anaemia')
sns.countplot(x='anaemia', hue='DEATH_EVENT', data=df)
sns.countplot(x='diabetes', hue='DEATH_EVENT', data=df)
plt.rcParams["figure.figsize"] = (18,10)
sns.countplot(x='ejection_fraction', hue='DEATH_EVENT', data=df)
plt.rcParams["figure.figsize"] = (18,10)
sns.countplot(x='serum_creatinine', hue='DEATH_EVENT', data=df)
plt.rcParams["figure.figsize"] = (25,13)
sns.countplot(x='serum_sodium', hue='DEATH_EVENT', data=df)
plt.figure(figsize=(12,10), dpi= 80)
sns.heatmap(df.corr(), xticklabels=df.corr().columns, yticklabels=df.corr().columns, cmap='RdYlGn', center=0, annot=True)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

df.head(3)
!pip install pycaret
from pycaret.classification import *
df.head(2)
env = setup(data = df, 
             target = 'DEATH_EVENT',
            #ignore_features = ['anaemia'],
             silent = True,
            remove_outliers = True,
            normalize = True)
model_results=compare_models()
model_results
compare_models()
xgb = create_model('xgboost')
plot_model(estimator = xgb, plot = 'feature')
plot_model(estimator = xgb, plot = 'auc')
plot_model(estimator = xgb, plot = 'confusion_matrix')
predictions = predict_model(xgb, data=df)
predictions.head()