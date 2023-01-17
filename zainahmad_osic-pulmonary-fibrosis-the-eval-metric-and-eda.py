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
import pydicom

import matplotlib.pyplot as plt

import seaborn as sns

train_labels = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_labels = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

sample_sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')
sample_sub.head()
sample_sub['Id'] = [x.split('_')[0] for x in sample_sub['Patient_Week']]

sample_sub.head()
sample_sub['Week'] = [x.split('_')[1] for x in sample_sub['Patient_Week']]

sample_sub.head()
weeks = sample_sub['Week'].unique()

print(len(weeks))

weeks
total_patients_sub = len(sample_sub['Id'].unique())

number_of_weeks_per_patient = {x : len(sample_sub[sample_sub['Id'] == x]) for x in sample_sub['Id'].unique()}



print('total number of patients in submission file : ' , total_patients_sub)

print('no of weeks for eevery patient id')

number_of_weeks_per_patient
train_labels.columns
number_of_patients_train = len(train_labels['Patient'].unique())

print('number of patients in train : ' , number_of_patients_train)


unique_values_for_each_attribute = {x: len(train_labels[x].unique()) for x in train_labels.columns}

unique_values_for_each_attribute
test_labels.head()
matched = False

count = 0

for p in sample_sub['Id'].unique():

    if p in test_labels['Patient'].unique():

        count +=1



if count == len(test_labels['Patient'].unique()):

    print('Patient in submission and test are same -- total : ' , count)

else :

    print('only %d patient s are bothe in test and submission ' , count)

    
sns.distplot(train_labels['FVC'])

print('mean FVC : ' + str(np.mean(train_labels['FVC'])) + '\n  std FVC : ' + str(np.std(train_labels['FVC'])) )
sns.distplot(train_labels['Weeks'])