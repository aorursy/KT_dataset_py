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
lis = pd.read_csv('../input/listings_dec18.csv',low_memory=False)  # Using the listings_dec18.csv file to cleanse

print(lis.shape)
lis.info()
print(sum(lis.host_listings_count !=lis.host_total_listings_count),"records in column host_listings_count are different from host_total_listings_count:")

print(list(lis[lis.host_listings_count!=lis.host_total_listings_count].host_total_listings_count))

#print(sum(lis.calculated_host_listings_count !=lis.host_total_listings_count),"records in column calculated_host_listings_count are different from host_total_listings_count.")
calculated_counts=lis.groupby(['host_id']).size().reset_index(name='calculated_num_listings')

calculated_counts.head()
print('Unique values in the market column:',"\n",lis.market.unique())
lis[lis.market=='Guangzhou']
print('License:','\n',lis[(lis.license.notnull())&(lis.license != "GST")].license)
print('Host name of the three properties with license 35753401805:',list(lis[lis.license=='35753401805'].host_name))
lis[(lis.license.notnull())&(lis.license != "GST")][['host_name','host_id','license']].drop_duplicates().sort_values(['host_name'])