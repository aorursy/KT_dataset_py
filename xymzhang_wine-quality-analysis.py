import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import seaborn as sns
from collections import Counter
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
wine = pd.read_csv('../input/winequality-red.csv')
wine.info()
wine['alcohol_2'] = wine['alcohol'].apply(lambda x: 'high' if x >10 else 'low')
g = sns.FacetGrid(wine, col='quality', hue='alcohol_2', col_wrap=3, height=4)
g.map(plt.scatter, 'density','chlorides')
plt.legend()
corr = wine.corr()
sns.heatmap(corr, cmap='YlGnBu', center=0, vmin=-1, vmax=1)
plt.savefig('heatmap.png',dpi=600)
wine_train = wine.sample(frac=0.8, random_state=666)
wine_com = pd.concat([wine, wine_train], axis=0).reset_index()
wine_test = wine_com.drop_duplicates(keep=False)
