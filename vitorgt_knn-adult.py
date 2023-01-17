import csv

import numpy as np

import pandas as pd



import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score



adultNA = pd.read_csv('../input/adult-pmr3508/train_data.csv',

                      index_col=['Id'], na_values="?")

adultTest = pd.read_csv('../input/adult-pmr3508/test_data.csv',

                        index_col=['Id'], na_values="?")



print("Adult with NA shape:", adultNA.shape)

print("Adult testing shape:", adultTest.shape)



adultNA.head()
uniques = {

    'workclass': [],

    'marital.status': [],

    'occupation': [],

    'relationship': [],

    'race': [],

    'sex': [],

    'native.country': [],

    'income': [],

}



uniquesTest = {}



for keys in uniques:

    adultNA.loc[:, (keys+'.num')], uniques[keys] = pd.factorize(

        adultNA.loc[:, (keys)], sort=True)

    if keys != 'income':

        adultTest.loc[:, (keys+'.num')], uniquesTest[keys] = pd.factorize(

            adultTest.loc[:, (keys)], sort=True)
adultNA.describe()
adultNA.hist(bins=100, figsize=(20, 20))
corrmat = adultNA.corr()

threshold = 0.1

plt.figure(figsize=(25, 10))

sns.heatmap(corrmat[abs(corrmat[:]) > threshold],

            vmax=1., vmin=-1., annot=True).xaxis.tick_top()
plt.figure(figsize=(20, 20))

sns.pairplot(adultNA, vars=['age', 'education.num', 'marital.status.num',

                            'relationship.num', 'sex.num', 'capital.gain',

                            'capital.loss', 'hours.per.week'], hue='income')
adultNA.loc[:, ('native.country')].value_counts(normalize=True)
adult = adultNA.dropna()



print("Adult with NA shape:", adultNA.shape)

print("Adult without NA shape:", adult.shape)
X1adult = adult.loc[:, ('age', 'education.num', 'marital.status.num',

                        'relationship.num', 'sex.num', 'capital.gain',

                        'capital.loss', 'hours.per.week')]



# this second classifier will have two classes more: 'race' and 'occupation'

X2adult = adult.loc[:, ('age', 'education.num', 'marital.status.num',

                        'relationship.num', 'sex.num', 'capital.gain',

                        'capital.loss', 'hours.per.week', 'race.num',

                        'occupation.num')]



Yadult = adult.loc[:, ('income')]



testX1adult = adultTest.loc[:, ('age', 'education.num', 'marital.status.num',

                                'relationship.num', 'sex.num', 'capital.gain',

                                'capital.loss', 'hours.per.week')]



testX2adult = adultTest.loc[:, ('age', 'education.num', 'marital.status.num',

                                'relationship.num', 'sex.num', 'capital.gain',

                                'capital.loss', 'hours.per.week', 'race.num',

                                'occupation.num')]
%%time



knn = {

    'best': {

        'score': 0,

        'k': 0

    }

}



for k in range(10, 28):

    knn[k] = {}

    knn[k]['classifier'] = KNeighborsClassifier(k, metric='manhattan')

    knn[k]['score'] = np.mean(cross_val_score(

        knn[k]['classifier'], X2adult, Yadult, cv=10, scoring="accuracy"))

    if knn[k]['score'] > knn['best']['score']:

        knn['best']['score'] = knn[k]['score']

        knn['best']['k'] = k





knn[knn['best']['k']]['classifier'].fit(X2adult, Yadult)



print("Best accuracy:", knn['best']['score'], "K =", knn['best']['k'])
%%time



prediction = knn[knn['best']['k']]['classifier'].predict(testX2adult)
result = pd.DataFrame({'income': prediction})

result.to_csv("submission.csv", index=True, index_label='Id')

result