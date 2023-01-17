# libraries and data

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import pandas as pd

 

# Dataset:

a = pd.DataFrame({ 'group' : np.repeat('A',500), 'value': np.random.normal(10, 5, 500) })

b = pd.DataFrame({ 'group' : np.repeat('B',500), 'value': np.random.normal(13, 1.2, 500) })

c = pd.DataFrame({ 'group' : np.repeat('B',500), 'value': np.random.normal(18, 1.2, 500) })

d = pd.DataFrame({ 'group' : np.repeat('C',20), 'value': np.random.normal(25, 4, 20) })

e = pd.DataFrame({ 'group' : np.repeat('D',100), 'value': np.random.uniform(12, size=100) })

df=a.append(b).append(c).append(d).append(e)

 

# Usual boxplot

sns.boxplot(x='group', y='value', data=df)
# libraries and data

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import pandas as pd

plt.style.use('seaborn')

 

# Dataset:

df=pd.DataFrame({'x': np.random.normal(10, 1.2, 20000), 'y': np.random.normal(10, 1.2, 20000), 'group': np.repeat('A',20000) })

tmp1=pd.DataFrame({'x': np.random.normal(14.5, 1.2, 20000), 'y': np.random.normal(14.5, 1.2, 20000), 'group': np.repeat('B',20000) })

tmp2=pd.DataFrame({'x': np.random.normal(9.5, 1.5, 20000), 'y': np.random.normal(15.5, 1.5, 20000), 'group': np.repeat('C',20000) })

df=df.append(tmp1).append(tmp2)

 

# plot

plt.plot( 'x', 'y', data=df, linestyle='', marker='o')

plt.xlabel('Value of X')

plt.ylabel('Value of Y')

plt.title('Overplotting looks like that:', loc='left')
# libraries

import matplotlib.pyplot as plt

import numpy as np

 

# create data

x = np.random.normal(size=50000)

y = (x * 3 + np.random.normal(size=50000)) * 5

 

# Make the plot

plt.hexbin(x, y, gridsize=(15,15) )

plt.show()

 

# We can control the size of the bins:

plt.hexbin(x, y, gridsize=(150,150) )

plt.show()
# libraries

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

# Dataset

df=pd.DataFrame({'X': range(1,101), 'Y': np.random.randn(100)*15+range(1,101), 'Z': (np.random.randn(100)*15+range(1,101))*2 })

 

# plot

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(df['X'], df['Y'], df['Z'], c='skyblue', s=60)

ax.view_init(30, 185)

plt.show()
# library & dataset

from matplotlib import pyplot as plt

import numpy as np

 

# create data

x = np.random.rand(15)

y = x+np.random.rand(15)

z = x+np.random.rand(15)

z=z*z

 

# Use it with a call in cmap

plt.scatter(x, y, s=z*2000, c=x, cmap="BuPu", alpha=0.4, edgecolors="grey", linewidth=2)

 

# You can reverse it:

plt.scatter(x, y, s=z*2000, c=x, cmap="BuPu_r", alpha=0.4, edgecolors="grey", linewidth=2)

 

# OTHER: viridis / inferno / plasma / magma

plt.scatter(x, y, s=z*2000, c=x, cmap="plasma", alpha=0.4, edgecolors="grey", linewidth=2)
# libraries

from mpl_toolkits.basemap import Basemap

import numpy as np

import matplotlib.pyplot as plt

 

# Make a data frame with the GPS of a few cities:

data = pd.DataFrame({

'lat':[-58, 2, 145, 30.32, -4.03, -73.57, 36.82, -38.5],

'lon':[-34, 49, -38, 59.93, 5.33, 45.52, -1.29, -12.97],

'name':['Buenos Aires', 'Paris', 'melbourne', 'St Petersbourg', 'Abidjan', 'Montreal', 'Nairobi', 'Salvador']

})

 

# A basic map

m=Basemap(llcrnrlon=-160, llcrnrlat=-75,urcrnrlon=160,urcrnrlat=80)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha=0.7, lake_color='grey')

m.drawcoastlines(linewidth=0.1, color="white")

 

# Add a marker per city of the data frame!

m.plot(data['lat'], data['lon'], linestyle='none', marker="o", markersize=16, alpha=0.6, c="orange", markeredgecolor="black", markeredgewidth=1)
# libraries

import matplotlib.pyplot as plt

import numpy as np

 

# create data

x = np.random.rand(40)

y = np.random.rand(40)

z = np.random.rand(40)

 

# use the scatter function

plt.scatter(x, y, s=z*1000, alpha=0.5)

plt.show()
# Import the library

import matplotlib.pyplot as plt

from matplotlib_venn import venn3

 

# Make the diagram

venn3(subsets = (10, 8, 22, 6,9,4,2))

plt.show()
#libraries

import matplotlib

import matplotlib.pyplot as plt

import pandas as pd

import squarify # pip install squarify (algorithm for treemap)&lt;/pre&gt;

 

# Create a dataset:

my_values=[i**3 for i in range(1,100)]

 

# create a color palette, mapped to these values

cmap = matplotlib.cm.Blues

mini=min(my_values)

maxi=max(my_values)

norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)

colors = [cmap(norm(value)) for value in my_values]

 

# Change color

squarify.plot(sizes=my_values, alpha=.8, color=colors )

plt.axis('off')

plt.show()
# libraries

import pandas as pd

import matplotlib.pyplot as plt

 

# Create a dataframe

value1=np.random.uniform(size=20)

value2=value1+np.random.uniform(size=20)/4

df = pd.DataFrame({'group':list(map(chr, range(65, 85))), 'value1':value1 , 'value2':value2 })

 

# Reorder it following the values of the first value:

ordered_df = df.sort_values(by='value1')

my_range=range(1,len(df.index)+1)

 

# The vertical plot is made using the hline function

# I load the seaborn library only to benefit the nice looking feature

import seaborn as sns

plt.hlines(y=my_range, xmin=ordered_df['value1'], xmax=ordered_df['value2'], color='grey', alpha=0.4)

plt.scatter(ordered_df['value1'], my_range, color='skyblue', alpha=1, label='value1')

plt.scatter(ordered_df['value2'], my_range, color='green', alpha=0.4 , label='value2')

plt.legend()

 

# Add title and axis names

plt.yticks(my_range, ordered_df['group'])

plt.title("Comparison of the value 1 and the value 2", loc='left')

plt.xlabel('Value of the variables')

plt.ylabel('Group')
# Libraries

import matplotlib.pyplot as plt

 

# Make data: I have 3 groups and 7 subgroups

group_names=['groupA', 'groupB', 'groupC']

group_size=[12,11,30]

subgroup_names=['A.1', 'A.2', 'A.3', 'B.1', 'B.2', 'C.1', 'C.2', 'C.3', 'C.4', 'C.5']

subgroup_size=[4,3,5,6,5,10,5,5,4,6]

 

# Create colors

a, b, c=[plt.cm.Blues, plt.cm.Reds, plt.cm.Greens]

 

# First Ring (outside)

fig, ax = plt.subplots()

ax.axis('equal')

mypie, _ = ax.pie(group_size, radius=1.3, labels=group_names, colors=[a(0.6), b(0.6), c(0.6)] )

plt.setp( mypie, width=0.3, edgecolor='white')

 

# Second Ring (Inside)

mypie2, _ = ax.pie(subgroup_size, radius=1.3-0.3, labels=subgroup_names, labeldistance=0.7, colors=[a(0.5), a(0.4), a(0.3), b(0.5), b(0.4), c(0.6), c(0.5), c(0.4), c(0.3), c(0.2)])

plt.setp( mypie2, width=0.4, edgecolor='white')

plt.margins(0,0)

 

# show it

plt.show()
# libraries

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

 

# Data

df=pd.DataFrame({'x': range(1,11), 'y1': np.random.randn(10), 'y2': np.random.randn(10)+range(1,11), 'y3': np.random.randn(10)+range(11,21) })

 

# multiple line plot

plt.plot( 'x', 'y1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=4)

plt.plot( 'x', 'y2', data=df, marker='', color='olive', linewidth=2)

plt.plot( 'x', 'y3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="toto")

plt.legend()