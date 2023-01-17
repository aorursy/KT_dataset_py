import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



#ignore warning messages 

import warnings

warnings.filterwarnings('ignore') 



# set seaborn

sns.set()
iris = pd.read_csv('/kaggle/input/iris/Iris.csv')

iris.head()
iris.shape
dataset = iris.drop(['Species','Id'],axis = 1)

dataset.head()
from sklearn.preprocessing import StandardScaler



standard = StandardScaler()

cleanDataSet = pd.DataFrame(standard.fit_transform(dataset))

cleanDataSet.head()
!pip install minisom

from minisom import MiniSom    

from matplotlib.gridspec import GridSpec



som = MiniSom(7, 7, 4, sigma=0.25,neighborhood_function='gaussian') 

som.train_random(cleanDataSet.to_numpy(), 30000) # trains the SOM with 100 iterations
target = iris.Species.astype('category').cat.codes

labels_map = som.labels_map(cleanDataSet.to_numpy(), target)

label_names = np.unique(target)



plt.figure(figsize=(7, 7))

the_grid = GridSpec(7, 7)



for position in labels_map.keys():

    label_fracs = [labels_map[position][l] for l in label_names]

    plt.subplot(the_grid[6-position[1], position[0]], aspect=1)

    patches, texts = plt.pie(label_fracs)

plt.legend(patches, label_names, bbox_to_anchor=(0, 1.5), ncol=3)



plt.show()
plt.figure(figsize=(7, 7))

frequencies = np.zeros((7, 7))

for position, values in som.win_map(cleanDataSet.to_numpy()).items():

    frequencies[position[0], position[1]] = len(values)

plt.pcolor(frequencies, cmap='Blues')

plt.colorbar()

plt.show()