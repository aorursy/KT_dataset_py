import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')



try:

    t_file = pd.read_csv('../input/questions.csv', encoding='ISO-8859-1')

    print('File load: Success')

except:

    print('File load: Failed')
from nltk.corpus import stopwords

stop = stopwords.words('english')

print(stop)
t_file = t_file.dropna()

t_file['question1'] = t_file['question1'].str.lower().str.split()

t_file['question2'] = t_file['question2'].str.lower().str.split()

t_file['question1'] = t_file['question1'].apply(lambda x: [item for item in x if item not in stop])

t_file['question2'] = t_file['question2'].apply(lambda x: [item for item in x if item not in stop])
t_file['Common'] = t_file.apply(lambda row: len(list(set(row['question1']).intersection(row['question2']))), axis=1)

t_file['Average'] = t_file.apply(lambda row: 0.5*(len(row['question1'])+len(row['question2'])), axis=1)

t_file['Percentage'] = t_file.apply(lambda row: row['Common']*100.0/row['Average'], axis=1)
y = t_file['Percentage'][t_file['is_duplicate']==0].values

x = t_file['Average'][t_file['is_duplicate']==0].values



fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(7, 4))

fig.subplots_adjust(hspace=0.5, left=0.07, right=0.93)

ax = axs[0]

hb = ax.hexbin(x, y, gridsize=70, bins='log', cmap='inferno')

ax.axis([0, 20, 0, 100])

ax.set_title("Duplicates")

cb = fig.colorbar(hb, ax=ax)

cb.set_label('log10(N)')





y = t_file['Percentage'][t_file['is_duplicate']==1].values

x = t_file['Average'][t_file['is_duplicate']==1].values

ax = axs[1]

hb = ax.hexbin(x, y, gridsize=70, bins='log', cmap='inferno')

ax.axis([0, 20, 0, 100])

ax.set_title("Not duplicates")

cb = fig.colorbar(hb, ax=ax)

cb.set_label('log10(N)')



plt.show()
x = t_file['Percentage'][t_file['is_duplicate']==0].values

y = t_file['qid1'][t_file['is_duplicate']==0].values

area = t_file['Average'][t_file['is_duplicate']==0].values



plt.scatter(x, y, s=area*3, c='r', alpha=0.1)



x = t_file['Percentage'][t_file['is_duplicate']==1].values

y = t_file['qid1'][t_file['is_duplicate']==1].values

area = t_file['Average'][t_file['is_duplicate']==1].values



plt.scatter(x, y, s=area*3, c='b', alpha=0.1)







plt.show()