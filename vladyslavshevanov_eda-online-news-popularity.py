# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={'figure.figsize':(10, 8)}); # you can change this if needed 
df = pd.read_csv('../input/OnlineNewsPopularityReduced.csv', sep=',')
df.info()
df.head().T
df.groupby(['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday','weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday']).size()
df.loc[df['weekday_is_monday'] == 1, 'days'] = 'Monday'

df.loc[df['weekday_is_tuesday'] == 1, 'days'] = 'Tuesday'

df.loc[df['weekday_is_wednesday'] == 1, 'days'] = 'Wednesday'

df.loc[df['weekday_is_thursday'] == 1, 'days'] = 'Thursday'

df.loc[df['weekday_is_friday'] == 1, 'days'] = 'Friday'

df.loc[df['weekday_is_saturday'] == 1, 'days'] = 'Saturday'

df.loc[df['weekday_is_sunday'] == 1, 'days'] = 'Sunday'
df.groupby(['days']).size().plot(kind='bar')

plt.show()
df['n_tokens_title'].hist(bins=20);
from scipy.stats import spearmanr

correl = spearmanr(df['n_tokens_title'], df['shares'])

print('Сorrelation:', correl[0], 'p-value:', correl[1])
correl_img = spearmanr(df['num_imgs'], df['shares'])

print('Images corellation', correl_img[0], 'p-value', correl_img[1])

correl_video = spearmanr(df['num_videos'], df['shares'])

print('Videos correllation', correl_video[0], 'p-value', correl_video[1])
df.groupby('is_weekend')['shares'].mean().plot(kind='bar') 

plt.show();
correl= spearmanr(df['n_tokens_content'], df['shares'])

print('Tokens correlation:', correl[0], 'p-value:', correl[1])