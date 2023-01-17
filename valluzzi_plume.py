# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



def XYrot(Xs,Ys,Xr,Yr,alpha):

    #Xs,Ys = coordinate della sorgente

    #Xr,Yr = coordinate del recettore

    #Xi,Yi = coordinate relative del recettore rispetto alla sorgente

    #alpha = angolo dal Nord (clockwise)

    #theta = 90 -alpha (angolo da est Counterclockwise)

    Xi,Yi = Xr-Xs,Yr-Ys

    theta = np.radians( 90- alpha )

    #Matrice di rotazione

    Rot = np.array(

    [[np.cos(theta), -np.sin(theta)],

     [np.sin(theta),  np.cos(theta)]])

    return np.array([Xi,Yi]).dot(Rot)





def FromCSV(X,Y,filename):

    import pandas as pd

    df =  pd.read_csv(filename,sep=',')



    xmin = df["X"].min()

    xmax = df["X"].max()

    ymin = df["Y"].min()

    ymax = df["Y"].max()



    if ( X < xmin or X > xmax ):

        return 0.0

    if ( Y < ymin or Y > ymax ):

        return 0.0



    #j1 e j2 sono gli indici di x1,x2

    j1     = df.loc[df['X'] <= X, 'X'].idxmax()

    j2     = df.loc[df['X'] >= X, 'X'].idxmin()

    x1     = df['X'][j1]

    x2     = df['X'][j2]

    df1    = df[ df['X'] == x1]

    df2    = df[ df['X'] == x2]

    df     = pd.concat([df1,df2]) #selezione dei valor tra x1,x2



    i1     = df.loc[df['Y'] <= Y, 'Y'].idxmax()

    i2     = df.loc[df['Y'] >= Y, 'Y'].idxmin()

    y1     = df['Y'][i1]

    y2     = df['Y'][i2]

    df1    = df[ df['Y'] == y1]

    df2    = df[ df['Y'] == y2]

    df     = pd.concat([df1,df2])



    print(df)

    

    return df['Z'].max()

#-------------------------------------------------------------------------------

#   __main__ 

#-------------------------------------------------------------------------------

filename = "../input/plume.csv"



df = pd.read_csv(filename,sep=',')

x    = df['X'].as_matrix().reshape((100,100))

y    = df['Y'].as_matrix().reshape((100,100))

z    = df['Z'].as_matrix().reshape((100,100))



fig = plt.figure()

ax = fig.add_subplot(111)

c = ax.contourf(x,y,z,100)

fig.show()

#receptor 

alpha =30

Xi,Yi = 50,50

X,Y = XYrot(0,0,Xi,Yi,alpha)

print("({0},{1}) rotated of {2} degrees from North clockwise => ({3},{4})".format(Xi,Yi,alpha,X,Y))



print('Conc=',FromCSV(X,Y,filename))


