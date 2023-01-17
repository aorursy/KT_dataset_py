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
import warnings; warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
FX = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')

FX.drop('Unnamed: 0', axis=1, inplace=True)

# rename long feature names to ISO Currency Code
FX.rename(columns={
    'AUSTRALIA - AUSTRALIAN DOLLAR/US$':            'AUD',
    'EURO AREA - EURO/US$':                         'EUR',
    'NEW ZEALAND - NEW ZELAND DOLLAR/US$':          'NZD',
    'UNITED KINGDOM - UNITED KINGDOM POUND/US$':    'GBP',
    'BRAZIL - REAL/US$':                            'BRL',
    'CANADA - CANADIAN DOLLAR/US$':                 'CAD',
    'CHINA - YUAN/US$':                             'CNY',
    'HONG KONG - HONG KONG DOLLAR/US$':             'HKD',
    'INDIA - INDIAN RUPEE/US$':                     'INR',
    'KOREA - WON/US$':                              'KRW',
    'MEXICO - MEXICAN PESO/US$':                    'MXN',
    'SOUTH AFRICA - RAND/US$':                      'ZAR',
    'SINGAPORE - SINGAPORE DOLLAR/US$':             'SGD',
    'DENMARK - DANISH KRONE/US$':                   'DKK',
    'JAPAN - YEN/US$':                              'JPY',
    'MALAYSIA - RINGGIT/US$':                       'MYR',
    'NORWAY - NORWEGIAN KRONE/US$':                 'NOK',
    'SWEDEN - KRONA/US$':                           'SEK',
    'SRI LANKA - SRI LANKAN RUPEE/US$':             'LKR',
    'SWITZERLAND - FRANC/US$':                      'CHF',
    'TAIWAN - NEW TAIWAN DOLLAR/US$':               'TWD',
    'THAILAND - BAHT/US$':                          'THB'
}, inplace=True)

FX.replace('ND', np.nan, inplace=True)
FX.dropna(inplace=True)
FX['Time Serie'] = pd.to_datetime(FX['Time Serie'])

ccys = ['AUD','EUR','NZD','GBP','BRL','CAD','CNY','HKD','INR','KRW','MXN',
        'ZAR','SGD','DKK','JPY','MYR','NOK','SEK','LKR','CHF','TWD','THB']

FX0 = FX.copy()
FX.set_index('Time Serie',inplace=True)

# FX: daily %change, FX0: Log scaled index
for c in ccys:
    FX[c] = FX[c].astype('float')
    FX[c] = FX[c].pct_change()
    FX0[c] = FX0[c].astype('float')
    FX0[c] = np.log(FX0[c]/FX0.loc[0, c])

FX.dropna(inplace=True)
FX0.set_index('Time Serie', inplace=True)
def plot_linechart(feat, title, axpos):
    sns.lineplot(data=FX0[feat], ax=axpos)
    axpos.set_title(title)
    axpos.set_xlabel(None)
    axpos.set_ylabel('Log Scale')

fig, ax = plt.subplots(5, 1, sharey=True, sharex=True, figsize=(16,20))
plot_linechart(['HKD','CNY','MYR'], 'Currencies including Dollar-pegged period in dataset', ax[0])
plot_linechart(['EUR','DKK','NOK','SEK','CHF','GBP'], 'European Currenies', ax[1])
plot_linechart(['KRW','SGD','TWD','THB','JPY'], 'East Asian Currencies', ax[2])
plot_linechart(['AUD','NZD','INR','LKR'], 'Currencies in South Asia and Oceania', ax[3])
plot_linechart(['BRL','CAD','MXN','ZAR'], 'Americas and Africa', ax[4])
# hereafter, limit features non-Dollar-pegged Currencies
FX1 = FX.drop(['HKD', 'CNY', 'MYR'], axis=1)

# sort features by correlations to EUR
ordered_ccy = FX1.corr()['EUR'].sort_values(ascending=False).index

fig, ax = plt.subplots(figsize=(16,5))
sns.boxplot(data=FX1[ordered_ccy], ax=ax)
ax.set_title('%change Distribution');
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(FX1[ordered_ccy].corr(), square=True, cmap='coolwarm', center=0, ax=ax);
def plot_dendrogram(model, **kwargs):
    label_list = FX1.columns.to_list()
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, labels=label_list, **kwargs)

furthest_euc = AgglomerativeClustering(distance_threshold=0, n_clusters=None,
                                       linkage='complete').fit(FX1.T)
fig,ax = plt.subplots(figsize=(15,6))
plot_dendrogram(furthest_euc, truncate_mode='level', ax=ax)
ax.set_title('Hierarchical Clustering based on Euclid Norm');
furthest_cos = AgglomerativeClustering(distance_threshold=0, n_clusters=None,
                                       linkage='complete', affinity='cosine').fit(FX1.T)
fig,ax = plt.subplots(figsize=(15,6))
plot_dendrogram(furthest_cos, truncate_mode='level', ax=ax)
ax.set_title('Hierarchical Clustering based on Cosine Similality');