release_filter = ['Kaladesh']

keeps = ['name', 'colorIdentity', 'colors', 'type', 'types', 'cmc', 'power', 'toughness', 'legalities']

colorIdentity_map = {'B': 'Black', 'G': 'Green', 'R': 'Red', 'U': 'Blue', 'W': 'White'}
import pandas as pd

import numpy as np

from numpy.random import random

from math import ceil



import ggplot as gg

from ggplot import aes



import matplotlib.pyplot as plt

from matplotlib.path import Path

from matplotlib.spines import Spine

from matplotlib.projections.polar import PolarAxes

from matplotlib.projections import register_projection



from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

from plotly.tools import FigureFactory as FF

from plotly import tools



import seaborn as sns

init_notebook_mode()
raw = pd.read_json('../input/AllSets-x.json')

raw.shape
# Capture release filter

all_releases = raw.loc[['name', 'releaseDate']].transpose().sort_values('releaseDate', ascending=False)

release_mapping = {key: value for (key, value) in zip(raw.loc['name'], raw.columns)}



try:

    releases = [release_mapping[i] for i in release_filter]

except KeyError:

    print('Error: Please check that you have spelled the releases correctly.')

    raise

except NameError:    

    release_filter = ''

except:

    print('Error: Unknown error.  Check everything!')

    

if not len(release_filter):    

    print('Warning: Could not find the release_filter variable.  Using all releases.')

    releases = release_mapping.values



# Some data-fu to get all cards into a single table along with a couple of release columns.

mtg = []

for col in releases:

    release = pd.DataFrame(raw[col]['cards'])

    release = release.loc[:, keeps]

    release['releaseName'] = raw[col]['name']

    release['releaseDate'] = raw[col]['releaseDate']

    mtg.append(release)

mtg = pd.concat(mtg)

del release, raw   

mtg.shape
# remove promo cards that aren't used in normal play

# Edit 2016-09-30: Could be null because it's too new to have a ruling on it.

mtg_nulls = mtg.loc[mtg.legalities.isnull()]

mtg = mtg.loc[~mtg.legalities.isnull()]



# remove cards that are banned in any game type

mtg = mtg.loc[mtg.legalities.apply(lambda x: sum(['Banned' in i.values() for i in x])) == 0]

mtg = pd.concat([mtg, mtg_nulls])

mtg.drop('legalities', axis=1, inplace=True)

del mtg_nulls



# remove tokens without types

mtg = mtg.loc[~mtg.types.apply(lambda x: isinstance(x, float))]



# Power and toughness that depends on board state or mana cannot be resolved

mtg[['power', 'toughness']] = mtg[['power', 'toughness']].apply(lambda x: pd.to_numeric(x, errors='coerce'))



mtg.shape
# Combine colorIdentity and colors

mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colors'] = mtg.loc[(mtg.colors.isnull()) & (mtg.colorIdentity.notnull()), 'colorIdentity'].apply(lambda x: [colorIdentity_map[i] for i in x])

mtg['colorsCount'] = 0

mtg.loc[mtg.colors.notnull(), 'colorsCount'] = mtg.colors[mtg.colors.notnull()].apply(len)

mtg.loc[mtg.colors.isnull(), 'colors'] = ['Colorless']

mtg['colorsStr'] = mtg.colors.apply(lambda x: ''.join(x))



# Include colorless and multi-color.

mtg['manaColors'] = mtg['colorsStr']

mtg.loc[mtg.colorsCount>1, 'manaColors'] = 'Multi'
mono_colors = np.sort(mtg.colorsStr[mtg.colorsCount==1].unique()).tolist()



for color in mono_colors:

    mtg[color] = mtg.colors.apply(lambda x: color in x)



corr = [mtg.loc[(mtg[color] == True), mono_colors].apply(np.sum) for color in mono_colors]

corr = pd.DataFrame(corr, index=mono_colors)

corr
corr = [mtg.loc[(mtg[color] == True), mono_colors].apply(np.mean) for color in mono_colors]

corr = pd.DataFrame(corr, index=mono_colors)



vmax = ceil(max(corr[corr<1].max()) * 100) / 100

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr, mask=mask, vmax=vmax, square=True)
creatures = mtg.loc[

    (mtg.type.str.contains('Creature', case=False))    

    & (mtg.cmc.notnull())

    & (mtg.power.notnull())

    & (mtg.toughness.notnull())

    & (0 <= mtg.cmc) & (mtg.cmc < 16)

    & (0 <= mtg.power) & (mtg.power < 16)

    & (0 <= mtg.toughness) & (mtg.toughness < 16)    

    , ['name', 'cmc', 'power', 'toughness', 'manaColors']

]



creatures['cmc2'] = creatures['cmc'] + random(creatures.shape[0])

creatures['power2'] = creatures['power'] + random(creatures.shape[0])

creatures['toughness2'] = creatures['toughness'] + random(creatures.shape[0])



creatures['cmc'] = round(creatures['cmc'])

creatures['power'] = round(creatures['power'])

creatures['toughness'] = round(creatures['toughness'])

creatures.shape
colors = np.sort(mtg.manaColors.unique()).tolist()

plotly_colors = ['rgb(75,75,75)', 'rgb(70,150,225)', 'rgb(150, 150, 150)', 'rgb(75,175,25)', 'rgb(125, 75, 125)', 'rgb(225,70,25)', 'rgb(200,200,150)']

color_map = {key: value for (key, value) in zip(colors, plotly_colors)}



creatures['text'] = creatures[['power', 'toughness', 'cmc']].apply(lambda x: ' '.join(x.astype(int).astype(str)), axis=1)

creatures['text'] = creatures['name'] + '<br>' + '(' + creatures['text'] + ')'





data = [

    go.Scatter3d(

        x = creatures.loc[creatures.manaColors == color, 'power2'],    

        y = creatures.loc[creatures.manaColors == color, 'manaColors'],    

        z = creatures.loc[creatures.manaColors == color, 'toughness2'],

        mode='markers',

        name=color,

        text=creatures.loc[creatures.manaColors == color, 'text'],

        hoverinfo='text',

        marker=dict(   

            color=plotly_color,   

            size=6,

            opacity=0.6,

            line=dict(width=1)

        )

    )

    for color, plotly_color in zip(colors, plotly_colors)

]





layout = dict(height=900, width=900, title='Creatures', hovermode='closest')

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='Creatures')
gg.ggplot(gg.aes(x='cmc', color='manaColors'), data=creatures) + gg.geom_bar(size=1) + gg.facet_wrap('manaColors')
gg.ggplot(gg.aes(x='power', color='manaColors'), data=creatures) + gg.geom_bar(size=1) + gg.facet_wrap('manaColors') 
gg.ggplot(gg.aes(x='toughness', color='manaColors'), data=creatures) + gg.geom_bar(size=1) + gg.facet_wrap('manaColors')
def radar_factory(num_vars, frame='circle'):

    """Create a radar chart with `num_vars` axes.



    This function creates a RadarAxes projection and registers it.



    Parameters

    ----------

    num_vars : int

        Number of variables for radar chart.

    frame : {'circle' | 'polygon'}

        Shape of frame surrounding axes.



    """

    # calculate evenly-spaced axis angles

    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    # rotate theta such that the first axis is at the top

    theta += np.pi/2



    def draw_poly_patch(self):

        verts = unit_poly_verts(theta)

        return plt.Polygon(verts, closed=True, edgecolor='k')



    def draw_circle_patch(self):

        # unit circle centered on (0.5, 0.5)

        return plt.Circle((0.5, 0.5), 0.5)



    patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}

    if frame not in patch_dict:

        raise ValueError('unknown value for `frame`: %s' % frame)



    class RadarAxes(PolarAxes):



        name = 'radar'

        # use 1 line segment to connect specified points

        RESOLUTION = 1

        # define draw_frame method

        draw_patch = patch_dict[frame]



        def fill(self, *args, **kwargs):

            """Override fill so that line is closed by default"""

            closed = kwargs.pop('closed', True)

            return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)



        def plot(self, *args, **kwargs):

            """Override plot so that line is closed by default"""

            lines = super(RadarAxes, self).plot(*args, **kwargs)

            for line in lines:

                self._close_line(line)



        def _close_line(self, line):

            x, y = line.get_data()

            # FIXME: markers at x[0], y[0] get doubled-up

            if x[0] != x[-1]:

                x = np.concatenate((x, [x[0]]))

                y = np.concatenate((y, [y[0]]))

                line.set_data(x, y)



        def set_varlabels(self, labels):

            self.set_thetagrids(np.degrees(theta), labels)



        def _gen_axes_patch(self):

            return self.draw_patch()



        def _gen_axes_spines(self):

            if frame == 'circle':

                return PolarAxes._gen_axes_spines(self)

            # The following is a hack to get the spines (i.e. the axes frame)

            # to draw correctly for a polygon frame.



            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.

            spine_type = 'circle'

            verts = unit_poly_verts(theta)

            # close off polygon by repeating first vertex

            verts.append(verts[0])

            path = Path(verts)



            spine = Spine(self, spine_type, path)

            spine.set_transform(self.transAxes)

            return {'polar': spine}



    register_projection(RadarAxes)

    return theta





def unit_poly_verts(theta):

    """Return vertices of polygon for subplot axes.



    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)

    """

    x0, y0, r = [0.5] * 3

    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]

    return verts
spoke_labels = ['Artifact', 'Creature', 'Enchantment', 'Instant', 'Sorcery']

for col in spoke_labels:

    mtg[col] = mtg.types.apply(lambda x: col in x)



# Do not double count creatures

# Some double counting of artifacts and enchantments exist but it's rare.

mtg['Creature'] = mtg['Creature'] & ~(mtg['Artifact'] | mtg['Enchantment'])



# Is there a better way of grouping?

grouped = mtg.loc[(mtg.types.apply(lambda x: len(set(x).intersection(set(spoke_labels))))>0)].groupby('colorsStr')

data = grouped[spoke_labels].agg(np.mean)



theta = radar_factory(len(spoke_labels), frame='polygon')

cols = 3
colors = np.sort(mtg.colorsStr[mtg.colorsCount<=1].unique()).tolist()

rows = ceil(len(colors) / cols)

fig = plt.figure(figsize=(10, rows*5))

fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.5, bottom=0.05)



for n, color in enumerate(colors):

    ax = fig.add_subplot(rows, cols, n + 1, projection='radar')

    plt.rgrids(np.linspace(.1, 1, 10).tolist())

    ax.set_title(color, weight='bold', size='medium', position=(0.5, 1.1),

                 horizontalalignment='center', verticalalignment='center')



    dat = data.loc[data.index==color, :].iloc[0].tolist()

    if color != 'Colorless':        

        ax.plot(theta, dat, color=color.lower())

        ax.fill(theta, dat, facecolor=color.lower(), alpha=0.5)

    else:

        ax.plot(theta, dat, color='grey')

        ax.fill(theta, dat, facecolor='grey', alpha=0.5)

        

    ax.set_varlabels(spoke_labels)
# Values are percentages so 100% artifacts does not mean there are lots of artifacts of that color.

colors = np.sort(mtg.colorsStr[mtg.colorsCount>1].unique()).tolist()

colors = [i for i in colors if i in data.index.values]



rows = ceil(len(colors) / cols)

fig = plt.figure(figsize=(10, rows*5))

fig.subplots_adjust(wspace=0.3, hspace=0.3, top=0.5, bottom=0.05)



for n, color in enumerate(colors):

    ax = fig.add_subplot(rows, cols, n + 1, projection='radar')

    plt.rgrids(np.linspace(.1, 1, 10).tolist())

    ax.set_title(color, weight='bold', size='medium', position=(0.5, 1.1),

                 horizontalalignment='center', verticalalignment='center')



    dat = data.loc[data.index==color, :].iloc[0].tolist()

    ax.plot(theta, dat, color='y')

    ax.fill(theta, dat, facecolor='y', alpha=0.5)

    ax.set_varlabels(spoke_labels)
with pd.option_context('display.max_rows', 999, 'display.max_columns', 3):

    print(all_releases)