import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import holoviews as hv

from holoviews import dim

hv.extension('bokeh')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

pd.set_option('display.max_columns', 100)
data = pd.read_csv('/kaggle/input/student-grade-prediction/student-mat.csv')


data.head()
data.info()
data.isna().sum()
data.describe()
def boxplot(x):

    g1 = hv.BoxWhisker(data, x, 'G1').opts(box_fill_color=dim(x).str(), cmap='Set1')

    g2 = hv.BoxWhisker(data, x, 'G2').opts(box_fill_color=dim(x).str(), cmap='Set1')

    g3 = hv.BoxWhisker(data, x, 'G3').opts(box_fill_color=dim(x).str(), cmap='Set1')

    return (g1 + g2 + g3).opts(title=f'{x} boxplot')
boxplot('age')
boxplot('sex')
boxplot('internet')
boxplot('Fjob')
boxplot('Mjob')
boxplot('goout')
sns.pairplot(data)
corr_data = data.corr()

corr_data
sns.heatmap(corr_data)