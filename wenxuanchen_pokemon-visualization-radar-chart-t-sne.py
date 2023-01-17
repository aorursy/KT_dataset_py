import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



TYPE_LIST = ['Grass','Fire','Water','Bug','Normal','Poison',

            'Electric','Ground','Fairy','Fighting','Psychic',

            'Rock','Ghost','Ice','Dragon','Dark','Steel','Flying']



COLOR_LIST = ['#8ED752', '#F95643', '#53AFFE', '#C3D221', '#BBBDAF', '#AD5CA2', 

              '#F8E64E', '#F0CA42', '#F9AEFE', '#A35449', '#FB61B4', '#CDBD72', 

              '#7673DA', '#66EBFF', '#8B76FF', '#8E6856', '#C3C1D7', '#75A4F9']



# The colors are copied from this script: https://www.kaggle.com/ndrewgele/d/abcsds/pokemon/visualizing-pok-mon-stats-with-seaborn

# The colors look reasonable in this map: For example, Green for Grass, Red for Fire, Blue for Water...

COLOR_MAP = dict(zip(TYPE_LIST, COLOR_LIST))





# A radar chart example: http://datascience.stackexchange.com/questions/6084/how-do-i-create-a-complex-radar-chart

def _scale_data(data, ranges):

    (x1, x2), d = ranges[0], data[0]

    return [(d - y1) / (y2 - y1) * (x2 - x1) + x1 for d, (y1, y2) in zip(data, ranges)]



class RaderChart():

    def __init__(self, fig, variables, ranges, n_ordinate_levels = 6):

        angles = np.arange(0, 360, 360./len(variables))



        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar = True, label = "axes{}".format(i)) for i in range(len(variables))]

        _, text = axes[0].set_thetagrids(angles, labels = variables)

        

        for txt, angle in zip(text, angles):

            txt.set_rotation(angle - 90)

        

        for ax in axes[1:]:

            ax.patch.set_visible(False)

            ax.xaxis.set_visible(False)

            ax.grid("off")

        

        for i, ax in enumerate(axes):

            grid = np.linspace(*ranges[i], num = n_ordinate_levels)

            grid_label = [""]+[str(int(x)) for x in grid[1:]]

            ax.set_rgrids(grid, labels = grid_label, angle = angles[i])

            ax.set_ylim(*ranges[i])

        

        self.angle = np.deg2rad(np.r_[angles, angles[0]])

        self.ranges = ranges

        self.ax = axes[0]



    def plot(self, data, *args, **kw):

        sdata = _scale_data(data, self.ranges)

        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)



    def fill(self, data, *args, **kw):

        sdata = _scale_data(data, self.ranges)

        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)



    def legend(self, *args, **kw):

        self.ax.legend(*args, **kw)



# select display colors according to Pokemon's Type 1

def select_color(types):

    colors = [None] * len(types)

    used_colors = set()

    for i, t in enumerate(types):

        curr = COLOR_MAP[t]

        if curr not in used_colors:

            colors[i] = curr

            used_colors.add(curr)

    unused_colors = set(COLOR_LIST) - used_colors

    for i, c in enumerate(colors):

        if not c:

            try:

                colors[i] = unused_colors.pop()

            except:

                raise Exception('Attempt to visualize too many pokemons. No more colors available.')

    return colors



df = pd.read_csv('../input/Pokemon.csv')



# In this order, 

# HP, Defense and Sp. Def will show on left; They represent defense abilities

# Speed, Attack and Sp. Atk will show on right; They represent attack abilities

# Attack and Defense, Sp. Atk and Sp. Def will show on opposite positions

use_attributes = ['Speed', 'Sp. Atk', 'Defense', 'HP', 'Sp. Def', 'Attack']

# choose the pokemons you like

use_pokemons = ['Venusaur', 'Charizard', 'Blastoise', 'Pikachu']



df_plot = df[df['Name'].map(lambda x:x in use_pokemons)==True]

datas = df_plot[use_attributes].values 

ranges = [[2**-20, df_plot[attr].max()] for attr in use_attributes]

colors = select_color(df_plot['Type 1']) # select colors based on pokemon Type 1 



fig = plt.figure(figsize=(10, 10))

radar = RaderChart(fig, use_attributes, ranges)

for data, color, pokemon in zip(datas, colors, use_pokemons):

    radar.plot(data, color = color, label = pokemon)

    radar.fill(data, alpha = 0.1, color = color)

    radar.legend(loc = 1, fontsize = 'small')

plt.title('Base Stats of '+(', '.join(use_pokemons[:-1])+' and '+use_pokemons[-1] if len(use_pokemons)>1 else use_pokemons[0]))

plt.show() 
df_mega = df[df['Name'].map(lambda x:'Mega ' in x)==True].reset_index(drop=True)



# Names of Mega-Evolved Pokemons are too long. Truncate them to normal form

df_mega['Name'] = df_mega['Name'].map(lambda x:x[x.index('Mega ')+5:])



use_attributes = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']

data = df_mega[use_attributes].values



from sklearn.manifold import TSNE

model = TSNE(n_components=2, random_state=42)

tsne = pd.DataFrame(model.fit_transform(data))



graph = sns.jointplot(x=0, y=1, data=tsne, kind='reg', marginal_kws=dict(bins=20, rug=True), size = 10, xlim=(-200, 300), ylim=(-200, 200))

graph.ax_joint.cla()



plt.sca(graph.ax_joint)

x, y, colors = tsne[0], tsne[1], df_mega['Type 1'].map(lambda x:COLOR_MAP[x])

plt.scatter(x, y, color = colors)

plt.xlim(-200, 250)

plt.ylim(-200, 200)



for i, txt in enumerate(df_mega['Name']):

    plt.annotate(txt, (x[i], y[i]))

plt.show()