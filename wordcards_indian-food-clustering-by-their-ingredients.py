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

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()



from sklearn.cluster import KMeans



!pip install --q fuzzywuzzy

from fuzzywuzzy import fuzz
food = pd.read_csv('../input/indian-food-101/indian_food.csv').set_index('name')

food.shape
food.head()
fig, ax = plt.subplots(1,4, sharey=True,figsize=(14,3.5))

plt.tight_layout()

for i, f in enumerate(['diet', 'flavor_profile','course', 'region']):

    sns.countplot(food[f], ax=ax[i])

    ax[i].tick_params(axis='x', labelrotation=45)
fig, ax = plt.subplots(figsize=(14,3))

sns.countplot(food['state'], ax=ax)

ax.tick_params(axis='x', labelrotation=90)
ing_dic = {}



for f in food.index:

    ing_list = food.at[f, 'ingredients'].split(', ')

    for i in ing_list:

        i = i.lower()

        if i[0] == ' ':

            i = i[1:]

        if i[-1] ==' ':

            i = i[:-1]

        if i not in ing_dic:

            ing_dic[i] = 1

        else:

            ing_dic[i] += 1



ing_df = pd.DataFrame.from_dict(ing_dic, orient='index').rename(columns={0:'count'})
ing_df.sort_index().loc['red': 'red0'].T
ing_list = ing_df.sort_values('count').index.to_list()



n = 0



for i in range(len(ing_list)-1):

    for j in range(i+1, len(ing_list)):

        ratio = fuzz.ratio(ing_list[i], ing_list[j])

        if n == 30:

            break

        if ratio > 70:

            print(ing_list[i], ', ', ing_list[j], '\t', ratio)

            n += 1
for i in range(len(ing_list)-1):

    for j in range(i+1, len(ing_list)):

        ratio = fuzz.ratio(ing_list[i], ing_list[j])

        if ratio > 80:

            print('"', ing_list[i], '": "',ing_list[j], '"\t', ratio)
similar_ing_dic = {

    "red chili": "red chilli",

    "greens":"green",

    "drumstick":"drumsticks",

    "thin rice flakes":"beaten rice flakes",

    "chana daal":"chana da ",

    "whole urad dal":"white urad dal",

    "bell pepper":"bell peppers",

    "frozen green peas":"green peas" ,

    "fresh green peas":"green peas",

    "chilli": "chillies",

    "fish fillets": "fish fillet",

    "mustard seed": "mustard seeds",

    "peanut":"peanuts",

    "red chillies":"red chilli",

    "dried fruits":"dry fruits",

    "almond":"almonds",

    "carrots":"carrot",

    "yoghurt":"yogurt",

    "chenna":"chhena",

    "green chillies":"green chilies",

    "green chilli":"green chilies",

    "green chili":"green chilies",

    "potatoes":"potato",

    "tomatoes":"tomato"

}
new_ing_dic = {}



for f in food.index:

    tmp_list = food.at[f, 'ingredients'].split(', ')

    for i in tmp_list:

        i = i.lower()

        if i[0] == ' ':

            i = i[1:]

        if i[-1] ==' ':

            i = i[:-1]

        if i in similar_ing_dic:

            i = similar_ing_dic[i]

        if i not in new_ing_dic:

            new_ing_dic[i] = 1

        else:

            new_ing_dic[i] += 1

            

new_ing_df = pd.DataFrame.from_dict(new_ing_dic, orient='index').rename(columns={0:'count'})

new_ing_df.sort_index().loc['red':'red0'].T
BoI_df = pd.DataFrame(np.zeros(len(food)*len(new_ing_dic)).reshape(

    len(food), len(new_ing_dic)).astype(int), index=food.index, columns=new_ing_df.index)



for f in food.index:

    tmp_list = food.at[f, 'ingredients'].split(', ')

    for i in tmp_list:

        i = i.lower()

        if i[0] == ' ':

            i = i[1:]

        if i[-1] ==' ':

            i = i[:-1]

        if i in similar_ing_dic:

            i = similar_ing_dic[i]

        BoI_df.at[f, i]=1



BoI_df.head()
km = KMeans(n_clusters=5,random_state=0)

clust5 = pd.DataFrame(km.fit(BoI_df).labels_, index=BoI_df.index).rename(columns={0:'grp'})
sns.countplot(y=clust5['grp'], orient='h');
fig,ax = plt.subplots(4,figsize=(12,16))

for i, f in enumerate(['flavor_profile','course','region','diet']):

    pd.crosstab(clust5['grp'],food[f],normalize='index')[::-1].plot.barh(stacked=True,ax=ax[i])

    ax[i].set_title(f)
freq_ing = BoI_df.sum().sort_values(ascending=False)[:10].index.to_list()

freq_ing_df = pd.merge(clust5, BoI_df[freq_ing],

                       how='inner', left_index=True, right_index=True).groupby('grp').sum()

tot_ing_df = pd.DataFrame(BoI_df.sum()).rename(columns={0:'total'})



for c in freq_ing_df.columns:

    freq_ing_df[c] = freq_ing_df[c]/tot_ing_df.at[c, 'total']



freq_ing_df.T[::-1].plot.barh(stacked=True, figsize=(12,8),

                              title='Share of frequently used ingredient');
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(14,4))

sns.swarmplot(y=clust5['grp'], x=food['prep_time'], orient='h', ax=ax[0])

sns.swarmplot(y=clust5['grp'], x=food['cook_time'], orient='h', ax=ax[1]);
print(clust5[clust5['grp']==0].index.to_list())
print(clust5[clust5['grp']==1].index.to_list())
print(clust5[clust5['grp']==2].index.to_list())
print(clust5[clust5['grp']==3].index.to_list())
print(clust5[clust5['grp']==4].index.to_list())