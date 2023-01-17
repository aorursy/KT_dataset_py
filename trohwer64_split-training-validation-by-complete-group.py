import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



train_x = pd.read_csv('../input/X_train.csv')

train_y = pd.read_csv('../input/y_train.csv')



num_test=500



def prepare_data(t):

    def f(d):

        d=d.sort_values(by=['measurement_number'])

        return pd.DataFrame({

         'lx':[ d['linear_acceleration_X'].values ],

         'ly':[ d['linear_acceleration_Y'].values ],

         'lz':[ d['linear_acceleration_Z'].values ],

         'ax':[ d['angular_velocity_X'].values ],

         'ay':[ d['angular_velocity_Y'].values ],

         'az':[ d['angular_velocity_Z'].values ],

         'ox':[ d['orientation_X'].values ],

         'oy':[ d['orientation_Y'].values ],

         'oz':[ d['orientation_Z'].values ],

         'ow':[ d['orientation_W'].values ],

        })



    t= t.groupby('series_id').apply(f)

    return t





def split_shuffle_groups(t):

    t= t.copy()



    # select randomly some groups (should be weighted by # of samples)



    aggcol='surface' # arbitrary; just to get size

    gstat= t.groupby('group_id')[aggcol].agg(np.size)

    gstat= gstat.reset_index()



    import random

    random.shuffle



    groups = list(zip(gstat['group_id'].values, gstat[aggcol].values))

    random.shuffle(groups)

    

    test_groups= set()

    c=0

    for gid,len in groups:

        if c>=num_test: break

        c+=len

        test_groups.add(gid)

    print("test groups:", test_groups)



    ctest = [ i for i,gid in enumerate(t['group_id']) if (gid in test_groups) ]

    ctrain = [ i for i,gid in enumerate(t['group_id']) if not (gid in test_groups) ]



    random.shuffle(ctrain)

    random.shuffle(ctest)



    return t.iloc[ctrain], t.iloc[ctest]





train= prepare_data(train_x)



# merge

train=pd.merge(train,train_y[['series_id','group_id','surface']],on='series_id')



train_part_df, validation_part_df= split_shuffle_groups(train)



print("training part of training data set:", train_part_df.describe())

print("validation part of training data set:",validation_part_df.describe())
