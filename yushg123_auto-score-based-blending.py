# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        #print(os.path.join(dirname, filename))

        pass



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ridge = pd.read_csv('/kaggle/input/trends-master-ensemble/submission_ridge.csv')

modra = pd.read_csv('/kaggle/input/trends-master-ensemble/sub.csv')

tunguz_n_IC20 = pd.read_csv('/kaggle/input/rapids-ensemblenoic20/submission_rapids_ensemble (1).csv')

rapids = pd.read_csv('/kaggle/input/trends-master-ensemble/submission1.csv')

bagging = pd.read_csv('/kaggle/input/baggingregressor-rapids-ensemble/submission_rapids_ensemble_with_baggingregressor.csv')

SS_skewed = pd.read_csv('/kaggle/input/rapids-ensemblenoic20/06163.csv')

stack = pd.read_csv('/kaggle/input/trends-multi-layer-model/submission.csv')
all_subs = pd.DataFrame(columns=['Id', 'modra', 'bagging', 'SS_skewed']) #'rapids', 'ridge', , 'tunguz'

all_subs['Id'] = ridge['Id']

all_subs['modra'] = modra['Predicted']

all_subs['bagging'] = bagging['Predicted']

all_subs['SS_skewed'] = SS_skewed['Predicted']
all_subs['stack'] = all_subs.merge(stack, how='left', left_on='Id', right_on='Id')['Predicted']
import math

print(math.e)
all_subs
scores = {

    'modra' : math.e ** (5.0 - 0.33),

    'bagging': math.e ** (5.0 - 0.31),

    'SS_skewed' : math.e ** (5.0 - 0.1),

    'stack' : math.e ** (5.0 - 0.0)

}
scores
columns = list(all_subs.columns)

columns.remove('Id')



total = 0

for col in columns:

    total += scores[col]

total
weight_sum = 0



for col in columns:

    weight = scores[col] / total

    weight_sum += weight

    print(weight)

    all_subs[col] = all_subs[col] * weight

weight_sum
all_subs['Predicted'] =  + all_subs['modra'] + all_subs['bagging'] + all_subs['SS_skewed'] + all_subs['stack']  #+ all_subs['rapids'] + all_subs['ridge'] + all_subs['tunguz']
all_subs
sub = all_subs[['Id', 'Predicted']]

sub
sub.to_csv('blend_sub.csv', index=False)