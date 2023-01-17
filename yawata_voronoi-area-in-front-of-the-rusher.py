import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



def new_X(x_coordinate, play_direction):

        if play_direction == 'left':

            return 120.0 - x_coordinate

        else:

            return x_coordinate



def new_line(rush_team, field_position, yardline):

    if rush_team == field_position:

            # offense starting at X = 0 plus the 10 yard endzone plus the line of scrimmage

        return 10.0 + yardline

    else:

        return 60.0 + (50 - yardline)



def new_orientation(angle, play_direction):

    if play_direction == 'left':

        new_angle = 360.0 - angle

        if new_angle == 360.0:

            new_angle = 0.0

        return new_angle

    else:

        return angle



def euclidean_distance(x1,y1,x2,y2):

    x_diff = (x1-x2)**2

    y_diff = (y1-y2)**2

    return np.sqrt(x_diff + y_diff)



def back_direction(orientation):

    if orientation > 180.0:

        return 1

    else:

        return 0





def update_yardline(df):

    new_yardline = df[df['NflId'] == df['NflIdRusher']]

    new_yardline['YardLine'] = new_yardline[['PossessionTeam','FieldPosition','YardLine']].apply(lambda x: new_line(x[0],x[1],x[2]), axis=1)

    new_yardline = new_yardline[['GameId','PlayId','YardLine']]

    return new_yardline



def update_orientation(df, yardline):

    df['X'] = df[['X','PlayDirection']].apply(lambda x: new_X(x[0],x[1]), axis=1)

    df['Orientation'] = df[['Orientation','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

    df['Dir'] = df[['Dir','PlayDirection']].apply(lambda x: new_orientation(x[0],x[1]), axis=1)

    df = df.drop('YardLine', axis=1)

    df = pd.merge(df, yardline, on=['GameId','PlayId'], how='inner')

    return df





def get_update_orientarion():

    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})

    yards = train.Yards

    yardline = update_yardline(train)

    train = update_orientation(train, yardline)

    return train



train = get_update_orientarion()
import numpy as np

import matplotlib.pyplot as plt

from scipy.spatial import Voronoi

from shapely.geometry import box, Polygon





def display_formation(train, PlayId, visualize=True):

    train_pl = train.loc[train.loc[:,'PlayId']==PlayId,:]

    rusher = (train_pl.loc[train_pl.loc[:,'NflId']==train_pl.loc[:,'NflIdRusher'],:]).index[0]

    yard = train_pl.loc[rusher,'Yards']

    of = train_pl.loc[rusher,'Team']

    

    train_of = train_pl.loc[train_pl.loc[:,'Team']==of,:]

    train_df = train_pl.loc[train_pl.loc[:,'Team']!=of,:]

    if visualize:

        plt.figure(figsize=(25,12))

        plt.xlim(0,120+80)

        plt.ylim(0,53.3)

        plt.scatter(train_of.loc[:,'X'],train_of.loc[:,'Y'])

        plt.scatter(train_df.loc[:,'X'],train_df.loc[:,'Y'])

    for ii,i in enumerate(train_of.index):

        if visualize:

            plt.annotate(train_of.loc[i,'Position'],(train_of.loc[i,'X'],train_of.loc[i,'Y']))

            plt.annotate('', xy=[train_of.loc[i,'X']+0.5*np.sin(train_of.loc[i,'Dir']/180*np.pi)*train_of.loc[i,'S'],

                                 train_of.loc[i,'Y']+0.5*np.cos(train_of.loc[i,'Dir']/180*np.pi)*train_of.loc[i,'S']], 

                         xytext=[train_of.loc[i,'X'],train_of.loc[i,'Y']],

                         arrowprops=dict(shrink=0, width=1, headwidth=3, headlength=3, 

                         connectionstyle='arc3',facecolor='gray', edgecolor='green'))

    for i in train_df.index:

        if visualize:

            plt.annotate(train_df.loc[i,'Position'],(train_df.loc[i,'X'],train_df.loc[i,'Y']))

            plt.annotate('', xy=[train_df.loc[i,'X']+0.5*np.sin(train_df.loc[i,'Dir']/180*np.pi)*train_df.loc[i,'S'],

                                 train_df.loc[i,'Y']+0.5*np.cos(train_df.loc[i,'Dir']/180*np.pi)*train_df.loc[i,'S']], 

                         xytext=[train_df.loc[i,'X'],train_df.loc[i,'Y']],

                         arrowprops=dict(shrink=0, width=1, headwidth=3, headlength=3, 

                         connectionstyle='arc3',facecolor='gray', edgecolor='black'))

    rusher = (train_pl.loc[train_pl.loc[:,'NflId']==train_pl.loc[:,'NflIdRusher'],:]).index[0]

    yard = train_pl.loc[rusher,'Yards']

    team = train_pl.loc[rusher,'Team']

    x,y = train_pl.loc[rusher,['X','Y']]

    for i,c in enumerate(train_pl.columns[:25]):

        if visualize:

            plt.annotate(c+'  '+str(train_pl.loc[rusher,c]),(125,51.5-i*2),size=12)

    for i,c in enumerate(train_pl.columns[25:]):

        if visualize:

            plt.annotate(c+'  '+str(train_pl.loc[rusher,c]),(160,51-i*2),size=12)

    #if train_pl.loc[train_pl.index[0],'PossessionTeam'] == train_pl.loc[train_pl.index[0],'FieldPosition']:

    if visualize:

        plt.plot((train_pl.loc[train_pl.index[0],'YardLine'],train_pl.loc[train_pl.index[0],'YardLine']),(0,53.3))

        plt.plot((train_pl.loc[train_pl.index[0],'YardLine']+yard,train_pl.loc[train_pl.index[0],'YardLine']+yard),(0,53.3),color='red')

    #else:

        #plt.plot((110-train_pl.loc[train_pl.index[0],'YardLine'],110-train_pl.loc[train_pl.index[0],'YardLine']),(0,53.3))

    if visualize:

        plt.savefig('formation.png')

        plt.show()

    return train_of.loc[:,['X','Y']].values, train_df.loc[:,['X','Y']].values, x, y, train_of.loc[rusher,:]

    #print(train_home.head())

    #print(train_away.head())

play_ls = list(set(train.loc[:,'PlayId']))

vs = []

print(len(play_ls))

xy_pair = [[5,10],[10,10],[20,10]]

area_feature = pd.DataFrame(columns=(['PlayId']+\

                                     ['areaup_{0}_{1}'.format(str(x),str(y)) for x,y in xy_pair]+['areadown_{0}_{1}'.format(str(x),str(y)) for x,y in xy_pair]))





def C_finite_polygons_2d(vor, radius=None):



    if vor.points.shape[1] != 2:

        raise ValueError("Requires 2D input")



    new_regions = []

    new_vertices = vor.vertices.tolist()



    center = vor.points.mean(axis=0)

    if radius is None:

        radius = vor.points.ptp().max()



    # Construct a map containing all ridges for a given point

    all_ridges = {}

    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):

        all_ridges.setdefault(p1, []).append((p2, v1, v2))

        all_ridges.setdefault(p2, []).append((p1, v1, v2))



    # Reconstruct infinite regions

    for p1, region in enumerate(vor.point_region):

        vertices = vor.regions[region]



        if all(v >= 0 for v in vertices):

            # finite region

            new_regions.append(vertices)

            continue



        # reconstruct a non-finite region

        ridges = all_ridges[p1]

        new_region = [v for v in vertices if v >= 0]



        for p2, v1, v2 in ridges:

            if v2 < 0:

                v1, v2 = v2, v1

            if v1 >= 0:

                # finite ridge: already in the region

                continue



            # Compute the missing endpoint of an infinite ridge



            t = vor.points[p2] - vor.points[p1] # tangent

            t /= np.linalg.norm(t)

            n = np.array([-t[1], t[0]])  # normal



            midpoint = vor.points[[p1, p2]].mean(axis=0)

            direction = np.sign(np.dot(midpoint - center, n)) * n

            far_point = vor.vertices[v2] + direction * radius



            new_region.append(len(new_vertices))

            new_vertices.append(far_point.tolist())



        # sort region counterclockwise

        vs = np.asarray([new_vertices[v] for v in new_region])

        c = vs.mean(axis=0)

        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])

        new_region = np.array(new_region)[np.argsort(angles)]



        # finish

        new_regions.append(new_region.tolist())



    return new_regions, np.asarray(new_vertices)



def cover_area(of,df,x,y, xy_size=[[5.01,5.01]], visualize=True):

    pts = np.concatenate([of,df],axis=0)

    vor = Voronoi(pts)

    regions, vertices = C_finite_polygons_2d(vor,1000)

    if visualize:

        plt.figure(figsize=(10,10))

    vs = []

    for x_size, y_size in xy_size:

        area = Polygon(np.array([[x,y-y_size/2],[x+x_size,y-y_size/2],[x+x_size,y+y_size/2],[x,y+y_size/2]]))

        v = 0

        for region in regions[:11]:

            polygon = vertices[region]

            p = Polygon(polygon)

            if p.intersects(area):

                p = Polygon(p.intersection(area))

                v += p.area

                xx, yy = p.exterior.coords.xy

                if visualize:

                    plt.fill(xx,yy, alpha=0.4, color='green')

        if visualize:

            plt.scatter(pts[:11,0], pts[:11,1],c='blue')

            plt.scatter(pts[11:,0], pts[11:,1],c='red')

            plt.xlim(vor.min_bound[0] - 10, vor.max_bound[0] + 10)

            plt.ylim(vor.min_bound[1] - 10, vor.max_bound[1] + 10)

            plt.savefig('area.png')

            plt.show()

        vs.append(v)

    return vs



play_ls = list(set(train.loc[:,'PlayId']))

of, df, x, y,rusher = display_formation(train, play_ls[0], visualize=False)

v = cover_area(of,df,x,y,xy_size=[[10.01,10.01]], visualize = True)