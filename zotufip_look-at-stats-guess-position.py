%matplotlib inline

import numpy as np

import pandas as pd

import sqlalchemy as sql

import matplotlib.pyplot as plt

import sklearn as sl
engine = sql.create_engine("sqlite:///../input/database.sqlite")

with engine.connect() as conn, conn.begin():

    player_stats = pd.read_sql_table("Player_Stats",conn)

    players = pd.read_sql_table("Player",conn)

    match_data = pd.read_sql_table("Match",conn)
match_data = match_data[match_data['season'] == '2015/2016']

data_sliced = match_data.iloc[:,11:77]

data_sliced = data_sliced.dropna(0)

player_data = player_stats.merge(players,on="player_api_id")

positions = pd.DataFrame(columns=['X','Y','player_id'])

for i in range(22):

    tmp = data_sliced[[i,i+22,i+44]]

    tmp.columns = ['X','Y','player_id']

    positions = positions.append(tmp,ignore_index=True)
dist = positions.groupby(['X','Y']).size()

distdf= dist.reset_index()

distdf.loc[distdf.Y==1.0,'X']=5.0 #moving goalkeeper position to (5,1) from ()

distdf

s = list(map(lambda x: 5 + x/10,distdf[0]))

plt.ylim(0,12)

plt.xlim(0,10)

plt.scatter(distdf['X'],distdf['Y'],s=s)
distdf = distdf[distdf['Y']!=1.0]

s = list(map(lambda x: 5 + x/10,distdf[0]))

distdf

plt.ylim(2,12)

plt.xlim(0,10)

plt.scatter(distdf['X'],distdf['Y'],s=s)

plt.plot([3.5,3.5],[0,12],c='black')

plt.plot([6.5,6.5],[0,12],c='black')

plt.plot([0,10],[4,4],c='black')

plt.plot([0,10],[9.5,9.5],c='black')
def x_pos_map(x):

    return 0 if 3.5<x<6.5 else 1 # 0 is center, 1 is side



def y_pos_map(y):

    if y<4 : return 0 # defender

    elif y<9.5 : return 1 #midfielder

    else : return 2 #striker



pos_labels = ['Central Defender','Central Midfielder','Central Striker','Side Defender','Side Midfielder','Side Striker']

    

labeled_positions = pd.DataFrame(columns=['X','Y','player_id'])

labeled_positions['X'] = positions['X'].map(x_pos_map)

labeled_positions['Y'] = positions['Y'].map(y_pos_map)

labeled_positions['player_id'] = positions['player_id']

player_positions = labeled_positions.groupby(['player_id','X','Y'])

player_positions = player_positions.size().groupby(level=0).apply(lambda x: x/float(x.sum()))

player_positions_df = player_positions.reset_index()

player_positions_df = player_positions_df[player_positions_df[0]>0.5]

label_column = player_positions_df.apply(lambda x: int(x[2]+x[1]*3) ,axis=1)

labeled_players = pd.DataFrame(columns=['player_api_id','pos'])

labeled_players['player_api_id'] = player_positions_df['player_id']

labeled_players['pos'] = label_column

labeled_players.head()
player_data_filtered = player_data[player_data['date_stat']>'2015-01-01'].sort_values(by=['player_api_id','date_stat'],ascending=[1,0]).groupby('player_api_id', as_index=False).first()

dataset = player_data_filtered.merge(labeled_players,on='player_api_id')

X = dataset.iloc[:,9:36]

Y = dataset['pos']
pos_dist = dataset.groupby(['pos']).size()

pos_dist.apply(lambda x: x/float(pos_dist.sum()))
from sklearn.preprocessing import normalize

X_norm = normalize(X)

from sklearn.cross_validation import train_test_split

X_t,X_v,Y_t,Y_v = train_test_split(X_norm,Y,test_size=0.3)


from sklearn.linear_model import LogisticRegression

logit = LogisticRegression(solver='lbfgs')

logit.fit(X_t,Y_t)

logit.score(X_v,Y_v)
"""

Example of creating a radar chart (a.k.a. a spider or star chart) [1]_.



Although this example allows a frame of either 'circle' or 'polygon', polygon

frames don't have proper gridlines (the lines are circles instead of polygons).

It's possible to get a polygon grid by setting GRIDLINE_INTERPOLATION_STEPS in

matplotlib.axis to the desired number of vertices, but the orientation of the

polygon is not aligned with the radial axes.



.. [1] http://en.wikipedia.org/wiki/Radar_chart

"""

import numpy as np



import matplotlib.pyplot as plt

from matplotlib.path import Path

from matplotlib.spines import Spine

from matplotlib.projections.polar import PolarAxes

from matplotlib.projections import register_projection





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
from numpy import amin,amax



test = logit.coef_

matmin = amin(amin(test))

test = test + abs(matmin)

matmax = amax(amax(test))

zeropoint = abs(matmin) / matmax

test = test / matmax

zeropoint
N = 27

theta = radar_factory(N, frame='polygon')

data = test

colors = ['b', 'r', 'g', 'm', 'y', 'c']

labels = X.columns

fig = plt.figure(figsize=(12, 17))

for n , d in enumerate(data):

            ax = fig.add_subplot(3, 2, n+ 1, projection='radar')

            ax.set_title(pos_labels[n], weight='bold', size='medium', position=(0.5, 1.1),

                     horizontalalignment='center', verticalalignment='center')

            plt.rgrids([0.2,0.4,zeropoint,0.6,0.8])

            ax.plot(theta, d, color=colors[n])

            ax.fill(theta, d, facecolor=colors[n], alpha=0.25)

            ax.set_varlabels(labels)

plt.show()