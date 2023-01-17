import matplotlib.pyplot as plt

plt.style.use('ggplot')



import seaborn as sns

sns.set_palette('husl')



# Bokeh

from bokeh.io import show, output_notebook

from bokeh.palettes import Spectral9

from bokeh.plotting import figure

output_notebook() # if you want to have an output in notebook



import pandas as pd

import numpy as np



import warnings

warnings.filterwarnings('ignore')
data=pd.read_csv('/kaggle/input/fifa19/data.csv',parse_dates=['Joined'])
pd.set_option('display.max_columns',200)

pd.set_option('display.float_format','{:.4f}'.format)
data.columns
fig=plt.figure(figsize=(10,4))

fig.add_subplot(1,1,1)



sns.boxenplot(data['International Reputation'], data['Age'], palette = 'Greys')

#sns.stripplot(data['International Reputation'], data['Age'], size=4, jitter=True, color="gray")

plt.title('Age boxplot by reputation level', fontsize = 20)

plt.show()
data.loc[data['International Reputation']==5,:]
data.describe(include='O')
data['Body Type'].value_counts()
data['Preferred Foot'].value_counts()
data_type=data.loc[data['Body Type'].isin(['Normal','Lean','Stocky'] ),:]
fig=plt.figure(figsize=(10,5))

fig.add_subplot(1,1,1)



sns.set(style="whitegrid", palette="pastel", color_codes=True)



sns.violinplot(x="Body Type", y="Age", hue="Preferred Foot",

               split=True, inner="quart",

               palette={"Right": "y", "Left": "b"},

               data=data_type)

sns.despine(left=True)
data['Work Rate'].value_counts()
fig=plt.figure(figsize=(15,5))

fig.add_subplot(1,1,1)



sns.boxplot(x="Work Rate", y="Age",data=data)

sns.despine(offset=10, trim=True)
sns.set(style="darkgrid")



g = sns.FacetGrid(data=data, row='Preferred Foot', col='Work Rate', margin_titles=True)

bins = np.linspace(0, 60, 13)

g.map(plt.hist,"Age", color="steelblue", bins=bins)
data['Joined_yr']=data['Joined'].dt.year

data_j=data.loc[data['Joined_yr'].isna()==False,:]

data_j['Joined_yr']=data_j['Joined_yr'].astype(int)
grid = sns.JointGrid(data['Age'], data['Joined_yr'], space=0, height=6, ratio=50)

grid.plot_joint(plt.scatter, color="b")

grid.plot_marginals(sns.rugplot, height=1, color="g")
fig, ax = plt.subplots(2,1,figsize=(18,8))



sns.set(style="ticks", palette="pastel")

sns.boxplot(x="Joined_yr", y="Age", data=data_j, ax=ax[0])

sns.boxplot(x="Joined_yr", y="Age", hue="International Reputation", data=data_j, ax=ax[1])

sns.despine(offset=10, trim=True)
g = sns.lmplot(x="Age", y="Joined_yr", hue="Work Rate", truncate=True, height=5, data=data)

#g.set_axis_labels("Age(yr)", "Joined_yr")