import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
riots_data = pd.read_csv("../input/los-angeles-1992-riot-deaths-from-la-times/la-riots-deaths.csv")
riots_data.head()

#dtype of the dataset
type(riots_data)

#list of column names
riots_data.columns 
# BBox=((riots_data.lat.min(), riots_data.lat.max(), riots_data.lon.min(), riots_data.lon.max()))
#defines the area of the map that will include all the spatial points 

#Creating a list with additional area to cover all points into the map
BBox=[-118.4917, -117.6106, 33.7299, 34.3871]

BBox
#import map layer extracted based on the lat and long values
la_map = plt.imread('../input/map-zoom-out/map.png')

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(riots_data.lat,riots_data.lon, zorder=2, alpha= 0.5, c='r', s=60)
ax.set_title('Deaths across LA during the 1992 riots')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(la_map, zorder=1, extent=BBox,aspect='auto')
plt.show()