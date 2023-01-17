import pandas as pd

import numpy as np

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import os

from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix, hstack

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import StratifiedKFold

from sklearn.metrics import log_loss
datadir = '../input'

gatrain = pd.read_csv(os.path.join(datadir,'gender_age_train.csv'),

                      index_col='device_id')

gatest = pd.read_csv(os.path.join(datadir,'gender_age_test.csv'),

                     index_col = 'device_id')

phone = pd.read_csv(os.path.join(datadir,'phone_brand_device_model.csv'))

# Get rid of duplicate device ids in phone

phone = phone.drop_duplicates('device_id',keep='first').set_index('device_id')
phone.head()
gatrain['brand'] = phone['phone_brand']

gatrain['model'] = phone['device_model']
gatest['brand'] = phone['phone_brand']

gatest['model'] = phone['device_model']
modelnum = pd.DataFrame(phone['device_model'].value_counts().sort_values(ascending=False))

#modelnum[:15].plot(kind='bar')

modelnum.head()

#print(len(modelnum))
gatest['conf'] = 0

gatest['group'] = 0

part0 = 23

part = part0 * 50

N1 = 74645

N2 = 112071

r0 = 50

test_try = gatest[N2-((part+r0)*75):N2-(part*75)]

nn = 0

for i in range(len(test_try)):

    pos = N2-((part+r0)*75)+i

    tpos = int(pos / N2 * N1)

    train_try = gatrain[tpos-150:tpos+150]

    modelname = test_try.iloc[i]['model']

    match = train_try[train_try['model']==modelname]

    if len(match)==0:

        continue

    match = match.iloc[0]

    nn += 1

    match['conf'] = 1/np.log(modelnum.loc[modelname][0]+1)

    test_try.iloc[i, 2] = match['conf']

    test_try.iloc[i, 3] = match['group']

    if i%75==0:

        print(pos)

print(nn)

gatest[N2-((part+r0)*75):N2-(part*75)] = test_try
gatest.head()
gatest[N2-((part+r0)*75):N2-(part*75)].to_csv('leaky.csv',index=True)