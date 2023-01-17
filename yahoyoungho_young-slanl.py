# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_dt = pd.read_csv('../input/train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print(train_dt.head())

print(train_dt.tail())
train_acoustic_small = train_dt['acoustic_data'].values[::50]

train_ttf_small = train_dt['time_to_failure'].values[::50]
train_acoustic_small.shape
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title('Acoustic_data and time_to_failure (sampled)')

plt.plot(train_acoustic_small, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train_ttf_small, color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)

del train_ttf_small

del train_acoustic_small
train_ttf_epoch = train_dt['time_to_failure'].values[:100000]

train_acoustic_epoch = train_dt['acoustic_data'].values[:100000]
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title('Acoustic_data and time_to_failure(s) (sampled)')

plt.plot(train_acoustic_epoch, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train_ttf_epoch, color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)

#train_ttf_epoch[:100]

# occurance of ttf == 

# 1.4690996: 108,

# 1.4690998: 109,

# 1.4690999: 108,

# 1.4691: 41

unique, counts = np.unique(train_ttf_epoch[:1000], return_counts=True)

dict(zip(unique, counts))
#delete train_ttf/acoustic_epoch

del train_ttf_epoch

del train_acoustic_epoch
# sample first 1% of the data

onepercent = int(len(train_dt)*0.01)

train_ttf_one = train_dt['time_to_failure'].values[:onepercent]

train_acoustic_one = train_dt['acoustic_data'].values[:onepercent]
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title('Acoustic_data and time_to_failure(s) (sampled) one percent')

plt.plot(train_acoustic_one, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train_ttf_one, color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)

#8th percent

train_ttf_one = train_dt['time_to_failure'].values[onepercent*7:onepercent*8]

train_acoustic_one = train_dt['acoustic_data'].values[onepercent*7:onepercent*8]

train_acoustic_one.shape
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title('Acoustic_data and time_to_failure(s) (sampled) one percent')

plt.plot(train_acoustic_one, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train_ttf_one, color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)

# returns time difference between ttf is minmum and when acoustic data is unusually high

def timegap(acoustic,ttf):

    ttf_spike = np.argmax(np.abs(acoustic))

    ttf_min = np.argmin(ttf)

    return {"min_ttf":ttf[ttf_min],

           "ttf_spike":ttf[ttf_spike],

           "difference":np.abs(ttf[ttf_min]-ttf[ttf_spike])}
timegaps = []

timegaps.append(timegap(train_acoustic_one,train_ttf_one))

print(timegaps)
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title('Acoustic_data and time_to_failure (sampled)')

plt.plot(train_acoustic_small, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train_ttf_small, color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)

# going to filter out low amplitude by changing to 0

# takes in acoustic data and an int amplitude

# will change acoustic data to 0 if less than amplitude value

def low_amp_filter(acou,amp):

    toReturn = []

    i=0

    while i < len(acou):

        if np.abs(acou[i])<amp:

            toReturn.append(0)

        else: # when value is higher than amp

            #print(acou[i])

            toReturn.append(acou[i])

        i+=1

    return toReturn
acou_filtered = low_amp_filter(train_acoustic_small,800)

acou_filtered_np = np.array(acou_filtered)
fig, ax1 = plt.subplots(figsize=(16, 8))

plt.title('Acoustic_data and time_to_failure (sampled & filtered)')

plt.plot(acou_filtered, color='b')

ax1.set_ylabel('acoustic_data', color='b')

plt.legend(['acoustic_data'])

ax2 = ax1.twinx()

plt.plot(train_ttf_small, color='r')

ax2.set_ylabel('time_to_failure', color='r')

plt.legend(['time_to_failure'], loc=(0.875, 0.9))

plt.grid(False)

# get indices of ttf when it reaches the minimum

# when ttf value increases 

def ttf_low(ttf):

    toreturn = []

    i = 0

    while i < len(ttf)-2:

        diff = ttf[i+1]-ttf[i]

        #print(diff)

        if diff > 1.0:

            #store the index and the ttf value

            tmp = np.array([i,ttf[i]])

            toreturn.append(tmp)

        i+=1

    return toreturn





ttf_low_idx = ttf_low(train_ttf_small)

ttf_low_idx = np.array(ttf_low_idx)

ttf_low_idx.shape



#indices of ttf when it is right before reaching 0

ttf_low_idx[:,0]
# counts the number of high acoustic values before ttf reaches 0

"""

acou: filtered acoustic data

ttf_idx: array of indices of ttf right before it resets

"""

def acou_counter(acou,ttf_idx,ttf):

    i=0

    toreturn = []

    for indices in ttf_idx:

        #indices = index of ttf right before it resets

        tmp = []

        ttf_ind = int(indices)

        while i < ttf_ind:

            #loop thru right before i == indices -1

            #get the index of acou where not 0

            if acou[i] != 0:

                acou_dict = {

                    "index": i,

                    "acoustic_value": acou[i],

                    "acou_ttf": ttf[i],

                    "this.ttf": ttf[ttf_ind],

                    "this.ttf_ind": ttf_ind

                }

                tmp.append(acou_dict)

            i+=1

            

        toreturn.append(tmp)

    return toreturn
acou_idx = acou_counter(np.array(acou_filtered),ttf_low_idx[:,0],train_ttf_small)
acou_idx[0]
acou_idx[0][0]

print(type(acou_idx),type(acou_idx[0]),type(acou_idx[0][0]))
for i in range(len(acou_idx)):

    acou0 = acou_idx[i]

    print(str(i+1)+"th")

    for d in range(len(acou0)):

        print(acou0[d]["acou_ttf"] - acou0[d]["this.ttf"])

del acou0    
#acou_diction = list(list(dict))

def timegap_avg(acou_diction):

    toreturn = []

    for sublist in acou_diction:

        dif_list = []

        for diction in sublist:

            dif_list.append(diction["acou_ttf"] - diction["this.ttf"])

        toreturn.append(np.mean(dif_list))

    return toreturn
timeavg = timegap_avg(acou_idx)
timeavg
acou_idx[0][0].keys()