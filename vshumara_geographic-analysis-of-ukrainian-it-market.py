# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

from collections import Counter

from mpl_toolkits.basemap import Basemap





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
df_codes = pd.read_csv("../input/ukrain-postal-codes/ua_post_utf8_v1.csv").drop_duplicates(subset="postal_code")

df_fop = pd.read_csv("../input/ukraine-register-of-individual-entrepreneurs/edr_fop_utf-8.csv")



df_fop = df_fop.loc[(df_fop['KVED'].str.startswith('63')) | (df_fop['KVED'].str.startswith('62'))]

df_fop['postal_code'] = df_fop['ADDRESS'].str[:5]

df_codes['postal_code'] = df_codes['postal_code'].astype(str).apply(lambda x: '0' * (5 - len(x)) + x if len(x) < 5 else x)
df_full = pd.merge(df_fop.assign(postal_code=df_fop.postal_code.astype(str)), 

          df_codes.assign(postal_code=df_codes.postal_code.astype(str)), 

          how='left', on='postal_code')



print('NAN count = ' + str(df_full['longitude'].isna().sum()))
df_nan = df_full[pd.isnull(df_full['latitude'])][['postal_code', 'ADDRESS']]

df_nan['postal4'] = df_nan.postal_code.str[:4]

df_codes['postal4'] = df_codes.postal_code.str[:4]

df_codes = df_codes.drop_duplicates(subset="postal4")



df_nan_merge = pd.merge(df_nan, df_codes, how='left', on='postal4')



df_full.loc[pd.isnull(df_full['latitude']),['latitude', 'longitude', 'admin_name1']] = df_nan_merge[['latitude', 'longitude', 'admin_name1']].values



print(df_full['latitude'].isna().sum())
df_full = df_full.dropna(subset = ['longitude'])



df_act = df_full[df_full['STAN'] == 'зареєстровано']

df_inact = df_full[df_full['STAN'] == 'припинено']



print('NA count after cleanup = ' + str(df_full['latitude'].isna().sum()))

print('Total count after cleanup = ' + str(len(df_full.index)))
def createmap(sizex, sizey, plot):

    plot.figure(figsize=(sizex, sizey), dpi= 80, facecolor='w', edgecolor='k')

    m = Basemap(projection='aeqd', resolution = 'i', lon_0 = 31.0, lat_0 = 48.7, width = 1400000, height = 1000000)

    m.drawmapboundary()

    m.drawcoastlines(linewidth=0.5, linestyle='solid', color='k', antialiased=1)

    m.drawcountries(linewidth=0.5, linestyle='solid', color='k', antialiased=1)



    return m
plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k')



df_full.groupby('STAN').admin_name1.count().sort_values().plot(kind = 'bar')



plt.title('Count distribution by Status')

plt.xlabel('Status')

plt.ylabel('Count')

plt.grid(axis='y', alpha=0.75)



for i, v in enumerate(df_full.groupby('STAN').admin_name1.count().sort_values().values):

    plt.text(i, v + 1000, str(v), color='blue', ha = 'center', fontsize = 16)



plt.show()
#onlyActive

m = createmap(18, 16, plt)

x, y = m(df_act['longitude'].tolist(), df_act['latitude'].tolist())

m.scatter(x, y, marker='.', color='r', alpha = 0.05)

plt.show()



#onlyInactive

m = createmap(18, 16, plt)

x, y = m(df_inact['longitude'].tolist(), df_inact['latitude'].tolist())

m.scatter(x, y, marker='.', color='r', alpha = 0.05)

plt.show()
fig = plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k')



ax = fig.add_subplot(1, 1, 1)

minor_ticks = np.arange(0, 42000, 2000)

ax.set_xticks(minor_ticks, minor=True)

ax.grid(which='minor', axis='x')

ax.grid(which='minor', alpha=0.25)



values = df_act.groupby('admin_name1').admin_name1.count().sort_values()

values.plot(kind = 'barh')



plt.title('Count distribution by Regions')

plt.xlabel('Count')

plt.ylabel('Region')



#for i, v in enumerate(values.values):

#    plt.text(v, i, "" + str(np.round(v / 1000, 1)), color='blue', va = 'center')



plt.show()
fig = plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k')



ax = fig.add_subplot(1, 1, 1)

minor_ticks = np.arange(0, 42000, 2000)

ax.set_xticks(minor_ticks, minor=True)

ax.grid(which='minor', axis='x')

ax.grid(which='minor', alpha=0.25)



sns.countplot(y='admin_name1', hue='STAN', data=df_full[['admin_name1', 'STAN']], order = df_full['admin_name1'].value_counts().index);



plt.show()
fig = plt.figure(figsize=(9, 8), dpi= 80, facecolor='w', edgecolor='k')



ax = fig.add_subplot(1, 1, 1)

minor_ticks = np.arange(0, 50, 5)

ax.set_xticks(minor_ticks, minor=True)

ax.grid(which='minor', axis='x')

ax.grid(which='minor', alpha=0.25)



ratio = df_full.groupby(['admin_name1'])['admin_name1'].count().to_frame().assign(ratio = lambda x: np.round(df_inact.groupby(['admin_name1'])['admin_name1'].count().loc[x['admin_name1'].index] / x['admin_name1'] * 100,1))



ratio['ratio'].sort_values().plot(kind = 'barh')



plt.title('Ratio Inactiv/total distribution by Regions')

plt.xlabel('Ratio %')

plt.ylabel('Region')



plt.show()

m = createmap(18, 16, plt)



kmeans = KMeans(n_clusters = 15)

kmeans.fit(np.column_stack((df_act['longitude'].tolist(),df_act['latitude'].tolist())))

y_kmeans = kmeans.predict(np.column_stack((df_act['longitude'].tolist(),df_act['latitude'].tolist())))

centers = kmeans.cluster_centers_



maxr = np.sqrt(np.sqrt(40000))

maxv = np.sqrt(np.sqrt(max(list(Counter(y_kmeans).values()))))

s1 = pd.Series(dict(sorted(Counter(y_kmeans).items()))).to_frame()

s1['radius'] = (np.sqrt(np.sqrt(s1[0])) * maxr / maxv) ** 4



x, y = m(df_act['longitude'].tolist(), df_act['latitude'].tolist())

x1, y1 = m(centers[:, 0], centers[:, 1])



m.scatter(x, y, c = y_kmeans, marker='o', alpha = 0.1)

m.scatter(x1, y1, c='red', marker='.', s = s1['radius'])



plt.show()