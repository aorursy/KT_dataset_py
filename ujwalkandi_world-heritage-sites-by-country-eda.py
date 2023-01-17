# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# `plt` is an alias for the `matplotlib.pyplot` module

import matplotlib.pyplot as plt

# import seaborn library (wrapper of matplotlib)
import seaborn as sns
whs = pd.read_csv('../input/world-heritage-sites-by-country/World_Heritage_Sites_by_country.csv')
whs.head()
#splitting the Country column to get single country names  
new = whs["Country"].str.split(" ", n = 1, expand = True) 

#Loading 1st string value  
whs["Country Name"]= new[0] 

#Dropping old Country column
whs.drop(columns =["Country"], inplace = True) 

#df display
whs
#To remove null and total rows from the DataFrame
whs_mod = whs.drop([168, 169])
whs_mod
#Apply plot style - 'bmh'
plt.style.use('bmh')

#Scatterplot
fig=plt.figure()
ax=fig.add_axes([0,0,3,1])
ax.scatter(whs_mod['Country Name'], whs_mod['Total sites'], color='r', alpha=0.9)
plt.xticks(rotation = 90)
ax.set_xlabel('Countries')
ax.set_ylabel('Total Sites')
ax.set_title('World Heritage Sites by country')
plt.show()
#Apply plot style - 'ggplot'
plt.style.use('ggplot')

#Histogram to find the average number of sites in most countries
plt.figure(figsize=(15,7))
whs_mod['Total sites'].hist(bins=40, edgecolor='black')
plt.xticks(rotation = 90)
plt.xlabel("Total sites")
plt.ylabel("Average number total sites")
plt.title('World Heritage Sites')
plt.show()