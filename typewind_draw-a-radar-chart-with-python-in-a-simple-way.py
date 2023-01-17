%matplotlib inline

import pandas as pd

import seaborn as sns

import numpy as np

df=pd.read_csv("../input/Pokemon.csv")

df.head()
labels=np.array(['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'])

stats=df.loc[386,labels].values
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

stats=np.concatenate((stats,[stats[0]]))

angles=np.concatenate((angles,[angles[0]]))
fig=sns.plt.figure()

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, stats, 'o-', linewidth=2)

ax.fill(angles, stats, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title([df.loc[386,"Name"]])

ax.grid(True)