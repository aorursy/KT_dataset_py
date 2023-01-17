# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/autos_new.csv")
dataset.info()
dataset.head()
dataset.tail()
np.shape(dataset['name'].unique())
np.shape(dataset['name'])
dataset=dataset.dropna()
dataset.head(10)
np.shape(dataset)
# plot image with 335111 points will be too large 
# just cut it into 10000 points
small_dataset=dataset[:10000]
sns.FacetGrid(small_dataset, hue="kilometer", size=5) \
   .map(plt.scatter, "power_ps", "dollar_price") \
   .add_legend()
plt.show()
sns.FacetGrid(small_dataset, hue="kilometer", size=5) \
   .map(plt.scatter, "registration_year", "dollar_price") \
   .add_legend()
plt.show()
subdataset=dataset.loc[dataset['brand']=='mercedes_benz']
np.shape(subdataset)
sub_mercedes_no_damage_dataset=subdataset.loc[subdataset['unrepaired_damage']!='ja']
sub_mercedes_no_damage_dataset.info()
sns.FacetGrid(sub_mercedes_no_damage_dataset, hue="kilometer", size=5) \
   .map(plt.scatter, "registration_year", "dollar_price") \
   .add_legend()
plt.show()
sns.pairplot(data=sub_mercedes_no_damage_dataset[["kilometer","registration_year","dollar_price","gearbox"]],hue="gearbox")\
    .add_legend()
plt.show()
sub_mercedes_no_damage_dataset['vehicle_type'].unique()
sub_mercedes_bus_no_damage_dataset=sub_mercedes_no_damage_dataset.loc[sub_mercedes_no_damage_dataset['vehicle_type']=='bus']
sub_mercedes_bus_diesel_no_damage_dataset=sub_mercedes_bus_no_damage_dataset.loc[sub_mercedes_bus_no_damage_dataset['fuel_type']=='diesel']
sub_mercedes_bus_diesel_no_damage_dataset.info()
np.shape(sub_mercedes_bus_diesel_no_damage_dataset['name'].unique())
sns.pairplot(data=sub_mercedes_bus_diesel_no_damage_dataset[["kilometer","registration_year","dollar_price","gearbox"]],hue="gearbox")\
    .add_legend()
plt.show()
