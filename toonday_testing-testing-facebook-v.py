# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Load the training data set
data = pd.read_csv("../input/train.csv")
places = data['place_id']
features = data.drop(['place_id', 'row_id'], axis = 1)

#== DATA EXPLORATION
dta = data
dta['freq'] = dta.groupby('place_id')['x'].transform('count')
dta['hour'] = (dta['time'] / 60) % 24
dta['weekday'] = (dta['time'] / (60*24)) % 7
dta['month'] = (dta['time'] / (60*24*30)) % 12
dta['year'] = dta['time'] / (60*24*365)
dta['day'] = (dta['time'] / (60*24)) % 365
#dta['x_combo_y'] = ((dta['x']*1)+(dta['y']*10))/(10.049876)
print( dta[:5])

# DATA VISUALIZATION
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plotly import tools

#fig = plt.figure(figsize=(16,10))
halfSize = int(len(range(100, 140, 40)) / 2)

import math
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import OneVsOneClassifier

iCount = 0
for i in range(100, 140, 40):
    print("i = ", i)
    
    # create grid
    start_index = 1
    a = 10
    N = i#math.sqrt(n)
    n = N*N#10000
    modVal = N - start_index + 1
    celldata = np.zeros((n,), dtype=[('index', 'i4'), ('row', 'i4'),('col', 'i4')])
    gridcell = pd.DataFrame(celldata)

    print( "N: ",N )

    cellValuesCombo = np.asarray(list(itertools.product(range(1, int(N)+1),range(1, int(N)+1))))
    gridcell['row'] = cellValuesCombo[:,0]
    gridcell['col'] = cellValuesCombo[:,1]
    gridcell['index'] = gridcell['row'] + (modVal*gridcell['col'])

    dta['gridcell_x'] = (dta['x']*N/a) + 1
    dta['gridcell_y'] = (dta['y']*N/a) + 1
    dta['gridcell_index'] = dta['gridcell_x'] + (modVal*dta['gridcell_y'])

    #-- Filter data
    xy = dta[dta['gridcell_index'] == gridcell.iloc[0,0]]
    
    """
    xy['hour'] = (dta['time'] / 60) % 24
    xy['weekday'] = (dta['time'] / (60*24)) % 7
    xy['month'] = (dta['time'] / (60*24*30)) % 12
    xy['year'] = dta['time'] / (60*24*365)
    xy['day'] = (dta['time'] / (60*24)) % 365
    """
    
    xy_sub = xy[['x', 'y', 'place_id', 'time', 'accuracy']]
    vw = xy[['x', 'y']]
    tu = xy[['x', 'y', 'time']]

    print( "xy: ",xy.shape )
    print( "halfSize: ",halfSize )
    print( dta[:5] )


    #-- reduce dimensionality of features
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    #-- plotting x against y with colors for each place
    temp_x = iCount % halfSize if (halfSize>0) else 0 #remainder
    temp_y = iCount / halfSize if (halfSize>0) else 0 #quotient
    #axarr[temp_x, temp_y].scatter(xy['x'], xy['y'], xy['time'], c=xy['place_id'], linewidth=0.0)#
    iCount = iCount+1
    
    """
    projCount = 220 + iCount
    print( "projCount: ",projCount )
    ax = fig.add_subplot(projCount, projection='3d')
    ax.scatter(xy['x'], xy['y'], xy['time'], c=xy['place_id'], linewidth=0.0)
    """
#plt.show()



