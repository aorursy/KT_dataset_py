import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

all_df = pd.concat([train_df, test_df], ignore_index=True)

all_df["directory"] = ['test' if np.isnan(bg) else 'train' for bg in all_df.bandgap_energy_ev.values]
geom_data = []

for row in all_df.itertuples():

    with open('../input/' + row.directory + '/' + str(row.id) + '/geometry.xyz') as f:

        text = f.read()

        geom_data.append(text)
geom_data[307] == geom_data[2153]
same_geoms = []

for i in range(len(geom_data)):

    for j in range(i):

        if geom_data[i] == geom_data[j]:

            print(j, i)

            same_geoms.append([j, i])
len(same_geoms)
all_df.loc[same_geoms[0],:]    
all_df.loc[same_geoms[1],:] 
all_df.loc[same_geoms[2],:] 
all_df.loc[same_geoms[3],:] 
all_df.loc[same_geoms[4],:] 
all_df.loc[same_geoms[5],:] 
all_df.loc[same_geoms[6],:] 
all_df.loc[same_geoms[7],:] 