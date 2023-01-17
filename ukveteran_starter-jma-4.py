import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from matplotlib import cm

sns.set_style('ticks')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
df1 = pd.read_csv('../input/pdb_data_no_dups.csv')

df2 = pd.read_csv('../input/pdb_data_seq.csv')

df = pd.read_csv('../input/pdb_data_no_dups.csv').merge(pd.read_csv('../input/pdb_data_seq.csv'), how='inner', on='structureId').drop_duplicates(["sequence"])
df.head()
print(df.residueCount_x.quantile(0.9))

df.residueCount_x.describe()
plt.matshow(df.corr())

plt.colorbar()

plt.show()