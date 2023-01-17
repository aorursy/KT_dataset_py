# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train_x = pd.read_csv('../input/X_train.csv')

train_y = pd.read_csv('../input/y_train.csv')

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

        })



    t= t.groupby('series_id').apply(f)



    def mfft(x):

        return [ x/math.sqrt(128.0) for x in np.absolute(np.fft.fft(x)) ][1:65]



    t['lx_f']=[ mfft(x) for x in t['lx'].values ]

    t['ly_f']=[ mfft(x) for x in t['ly'].values ]

    t['lz_f']=[ mfft(x) for x in t['lz'].values ]

    t['ax_f']=[ mfft(x) for x in t['ax'].values ]

    t['ay_f']=[ mfft(x) for x in t['ay'].values ]

    t['az_f']=[ mfft(x) for x in t['az'].values ]

    return t



t=prepare_data(train_x)



t=pd.merge(t,train_y[['series_id','surface','group_id']],on='series_id')

t=t.rename(columns={"surface": "y"})



def aggf(d, feature):

    va= np.array(d[feature].tolist())

    mean= sum(va)/va.shape[0]

    var= sum([ (va[i,:]-mean)**2 for i in range(va.shape[0]) ])/va.shape[0]

    dev= [ math.sqrt(x) for x in var ]

    return pd.DataFrame({

        'mean': [ mean ],

        'dev' : [ dev ],

    })



display={

'hard_tiles_large_space':'r-.',

'concrete':'g-.',

'tiled':'b-.',



'fine_concrete':'r-',

'wood':'g-',

'carpet':'b-',

'soft_pvc':'y-',



'hard_tiles':'r--',

'soft_tiles':'g--',

}



import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8*7))

#plt.margins(x=0.0, y=0.0)

#plt.tight_layout()

# plt.figure()



features=['lx_f','ly_f','lz_f','ax_f','ay_f','az_f']

count=0



for feature in features:

    stat= t.groupby('y').apply(aggf,feature)

    stat.index= stat.index.droplevel(-1)

    b=[*range(len(stat.at['carpet','mean']))]



    count+=1

    plt.subplot(len(features)+1,1,count)

    for i,(k,v) in enumerate(display.items()):

        plt.plot(b, stat.at[k,'mean'], v, label=k)

        # plt.errorbar(b, stat.at[k,'mean'], yerr=stat.at[k,'dev'], fmt=v)

   

    leg = plt.legend(loc='best', ncol=3, mode="expand", shadow=True, fancybox=True)

    plt.title("sensor: " + feature)

    plt.xlabel("frequency component")

    plt.ylabel("amplitude")



count+=1

plt.subplot(len(features)+1,1,count)

k='concrete'

v=display[k]

feature='lz_f'

stat= t.groupby('y').apply(aggf,feature)

stat.index= stat.index.droplevel(-1)

b=[*range(len(stat.at['carpet','mean']))]



plt.errorbar(b, stat.at[k,'mean'], yerr=stat.at[k,'dev'], fmt=v)

plt.title("sample for error bars (lz_f, surface concrete)")

plt.xlabel("frequency component")

plt.ylabel("amplitude")



plt.show()


