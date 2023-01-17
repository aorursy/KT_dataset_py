# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import matplotlib.patches as patches



# Any results you write to the current directory are saved as output.
thin_rate = 10



data = pd.read_csv('../input/train.csv',  dtype={'acoustic_data': np.int16 ,'time_to_failure': np.float32})

data = data[::thin_rate]
delta_time = ((data.time_to_failure[0]-data.time_to_failure[5000000])/5000000)*thin_rate

delta_time
acoustic_data_per_100 = data.acoustic_data[::10]

time_to_failure_per_100 = data.time_to_failure[::10]





fig, ax1 = plt.subplots(figsize=(20, 10))

plt.title("train_data", fontsize = 30)

ax1.plot(acoustic_data_per_100, color = "blue")

ax1.set_xlabel("index", fontsize = 20)

ax1.set_ylabel("acoustic_data", color = "blue", fontsize = 20)

ax1.tick_params(axis="x",labelsize = 10)

ax1.tick_params(axis="y", labelcolor="blue", labelsize = 15)

ax1.plot()



ax2 = ax1.twinx()

ax2.plot(time_to_failure_per_100, color = "green")

ax2.set_ylabel("time_to_failure", color = "green", fontsize = 20)

ax2.tick_params(axis="y", labelcolor="green", labelsize = 15)

ax2.plot()

data["rolling_std"] = data.acoustic_data.rolling(window=1000).std()

rolling_std_per_100 = data['rolling_std'].values[::10]

time_to_failure_per_100 = data.time_to_failure[::10]

acoustic_data_per_100 = data.acoustic_data[::10]

index = np.arange(len(acoustic_data_per_100))

bbox = {

    "facecolor" : "lightblue",

    "edgecolor" : "blue",

    "boxstyle" : "Round",

    "linewidth" : 2

}



fig, ax1 = plt.subplots(figsize=(20, 10))

plt.title("comparison between the two", fontsize = 30)

ax1.plot(index,acoustic_data_per_100, color = "blue")

ax1.set_xlabel("index", fontsize = 20)

ax1.set_ylabel("acoustic_data", color = "blue", fontsize = 20)

ax1.set_ylim(-3000,3000)

ax1.tick_params(axis="x",labelsize = 10)

ax1.tick_params(axis="y", labelcolor="blue", labelsize = 15)

ax1.plot()



ax2 = ax1.twinx()

ax2.plot(index,rolling_std_per_100, color = "red")

ax2.set_ylabel("rolling_std", color = "red", fontsize = 20)

ax2.set_ylim(-600,600)

ax2.tick_params(axis="y", labelcolor="red", labelsize = 15)

ax2.plot()



fig, ax1 = plt.subplots(figsize=(20, 10))

plt.title("cut a signal", fontsize = 30)

ax1.plot(index,acoustic_data_per_100, color = "blue")

ax1.set_xlabel("index", fontsize = 20)

ax1.set_ylabel("acoustic_data", color = "blue", fontsize = 20)

ax1.plot([44200, -1000], [44200, 1000], 'k-', lw=2)

ax1.set_xlim(44000,45000)

ax1.set_ylim(-3000,3000)

ax1.tick_params(axis="x",labelsize = 10)

ax1.tick_params(axis="y", labelcolor="blue", labelsize = 15)

ax1.plot()



ax2 = ax1.twinx()

ax2.plot(index,rolling_std_per_100, color = "red")

ax2.hlines(80, 44375, 44509, linestyle='dashed', linewidth=2)

ax2.vlines(44375, 0, 200, linestyle='dashed', linewidth=2)

ax2.vlines(44509, 0, 200, linestyle='dashed', linewidth=2)

ax2.vlines(44345, -200, 200, linestyle='solid', linewidth=2,color = "royalblue")

ax2.vlines(44479, -200, 200, linestyle='solid', linewidth=2,color = "royalblue")

ax2.text(44290,270,"cutting line", size = 20, color = "red" ,bbox = bbox)

ax2.annotate('', xy=[44345,100], xytext=[44375,100],

            arrowprops = dict(shrink=0, width=1, headwidth=8,headlength=10, connectionstyle='arc3',facecolor='royalblue', edgecolor='royalblue')

           )

ax2.annotate('', xy=[44479,100], xytext=[44509,100],

            arrowprops = dict(shrink=0, width=1, headwidth=8,headlength=10, connectionstyle='arc3',facecolor='royalblue', edgecolor='royalblue')

           )

ax2.set_ylabel("rolling_std", color = "red", fontsize = 20)

ax2.set_ylim(-600,600)

ax2.tick_params(axis="y", labelcolor="red", labelsize = 15)

ax2.plot()



del rolling_std_per_100

del acoustic_data_per_100

del time_to_failure_per_100

peak_line = 80

i= 0



start_index = []

end_index = []

sample_num = []



np_std = np.asarray(data.rolling_std)



for i in range(len(data)-1):

    if ((np_std[i] < peak_line ) & ( peak_line <= np_std[i+1])):

        start_index.append(i)

    if ((peak_line <= np_std[i]) & ( np_std[i+1] < peak_line )

    &((len(start_index) -  len(end_index)) ==  1)):

        end_index.append(i)



for i in range (len(start_index)):

    sample_num.append(-(start_index[i] - end_index[i]))



del np_std
data = data.drop("rolling_std", axis = 1)
acoustic_table = np.stack([start_index, end_index , sample_num], axis = 1) #numpy_ndarray



columns = ["start_index", "end_index","sample_num"] #pandas_DataFrame



acoustic_table = pd.DataFrame(acoustic_table , columns=columns)

acoustic_table.start_index = acoustic_table.start_index - 300

acoustic_table.end_index = acoustic_table.end_index - 300



del start_index

del end_index

del sample_num
plt.figure(figsize=(20, 20))





work = data.acoustic_data[acoustic_table.start_index[0]:acoustic_table.end_index[0]]

plt.subplot(511)

plt.title("cutted signals", fontsize = 30)

plt.plot(work,color = 'b')



work = data.acoustic_data[acoustic_table.start_index[10]:acoustic_table.end_index[10]]

plt.subplot(512)

plt.plot(work,color = 'b')



work = data.acoustic_data[acoustic_table.start_index[20]:acoustic_table.end_index[20]]

plt.subplot(513)

plt.plot(work,color = 'b')



work = data.acoustic_data[acoustic_table.start_index[30]:acoustic_table.end_index[30]]

plt.subplot(514)

plt.plot(work,color = 'b')



work = data.acoustic_data[acoustic_table.start_index[40]:acoustic_table.end_index[40]]

plt.subplot(515)

plt.plot(work,color = 'b')



del work
f = []

F = []

F_abs = []

work = []

pi = np.pi



fourier_data = []

omega_data = []

energy_data = []



for i in range(len(acoustic_table)):



    f = []

    F = []

    F_abs = []

    omega = []

    fourier_work = data.acoustic_data[acoustic_table.start_index[i]:acoustic_table.end_index[i]]

    N = len(fourier_work)

    half_N = int(N/2 + 1)

    delta_omega = 2*pi/half_N



    #fourier_data

    f.append(fourier_work)

    F.append(np.fft.fft(fourier_work)[0:half_N+1])

    F_abs.append(np.abs(F))

    

    #frequency_data (omega_data)

    omega.append(np.linspace(0, 2*pi, half_N+1))

    

    #energy_data

    U = np.dot(F_abs[0], omega[0])



    fourier_data.append(F_abs)

    omega_data.append(omega)

    energy_data.append(U)
acoustic_table["fourier_data"] = fourier_data

acoustic_table["omega_data"] = omega_data

acoustic_table["energy_data"] = energy_data



del fourier_data

del omega_data

del energy_data
np_index = np.asarray(acoustic_table.start_index)

np_time = np.asarray (data.time_to_failure)



#time_data

time_work = []

i = 0

for i in range(len(acoustic_table)):

    index_work = np_index[i]

    time_work.append(np_time[index_work])

acoustic_table["time_to_failure"] = time_work



del time_work
acoustic_table
#Border is determined sorted energy data.Border is center of the two enegy data these have most wide difference.

np_energy_data = np.asarray(acoustic_table.energy_data)

np_sort_energy_data = np.sort(np_energy_data)

np_sort_energy_data



np_sort_energy_diffdata = np.diff(np_sort_energy_data) 



high_border_index = (np.argmax(np_sort_energy_diffdata) + 1)

low_border_index = (np.argmax(np_sort_energy_diffdata))

border_energy = (np_sort_energy_data[high_border_index] + np_sort_energy_data[low_border_index])/2



#graph plot

x = (np.asarray(acoustic_table.time_to_failure))

y = (np.asarray(acoustic_table.energy_data))

r = patches.Rectangle(xy=(0, border_energy), width=0.1, height=0.5, ec='#000000', fill=False)



plt.figure(figsize=(15, 8))

plt.scatter(x = x, y = y)

plt.tick_params(axis="x", labelsize = 15)

plt.tick_params(axis="y", labelleft=False, left=False)

plt.title("The relationship of energy and time_to_failure",fontsize = 20)

plt.hlines(border_energy, 0, 10, linestyle='solid', color = "red",linewidth=2)

plt.text(4,border_energy*1.05,"border energy", size = 20, color = "red")

plt.text(0.8,border_energy*1.5,"high energy data", size = 20, color = "royalblue")

plt.xlabel("time_to_failure", fontsize = 20)

plt.ylabel("energy_data", fontsize = 20)



plt.figure(figsize=(15, 8))

plt.scatter(x = x, y = y)

plt.tick_params(axis="x", labelsize = 15)

plt.tick_params(axis="y", labelleft=False, left=False)

plt.title("Expansion of high enegy data",fontsize = 20)

plt.vlines(border_energy, 0, 10, linestyle='solid', color = "red",linewidth=2)

plt.text(0.33,border_energy*1.5,"high energy data", size = 20, color = "royalblue")

plt.xlim(0.3,0.35)

plt.ylim(border_energy,border_energy*2)

plt.xlabel("time_to_failure", fontsize = 20)

plt.ylabel("energy_data", fontsize = 20)
data["predict_time"] = np.nan



np_energy = np.asarray(acoustic_table.energy_data)

np_predict_time = np.asarray(data.predict_time)



for i in range(len(np_energy)):

    if (np_energy[i] > border_energy):

        np_predict_time[acoustic_table.start_index[i]] = 0.32

data.predict_time = np_predict_time



del np_energy

del np_predict_time
just_before_time_data = data[data.predict_time > 0]

just_before_time_data
quake_time = pd.DataFrame()



np_predict_time = np.asarray(data.predict_time)

np_wait_start_index = []

np_wait_end_index = []



np_before032_wait_end_index = np.where(np_predict_time == 0.32)

np_before032_wait_end_index = np_before032_wait_end_index[0]
i = 0

index_length_toend = int((0.32/delta_time))



for i in range(len(np_before032_wait_end_index)):

    np_wait_end_index.append(int(np_before032_wait_end_index[i] + index_length_toend))

 

 

np_wait_start_index = np_wait_end_index + np.array(1)

np_wait_start_index = np.insert(np_wait_start_index, 0, 0)

np_wait_end_index.append(np.nan)



quake_time["wait_start_index"] = np_wait_start_index

quake_time["wait_end_index"] = np_wait_end_index



del np_predict_time

del np_wait_end_index

del np_before032_wait_end_index
quake_time
np_wait_start_index = np.array(quake_time.wait_start_index)

np_wait_end_index = np.array(quake_time.wait_end_index)

np_wait_end_index = np_wait_end_index.astype(np.int32)

np_predict_time = np.array(data.predict_time)

i = 0

j = 0



for i in range(len(np_wait_end_index-1)):

    j = np_wait_start_index[i]

    k = 0

    sample_num = np_wait_end_index[i] - np_wait_start_index[i]

    while (j < np_wait_end_index[i] +1):

        np_predict_time[j] = (sample_num - k)*delta_time

        j+=1

        k+=1
data.predict_time = np_predict_time
data
predict_time_work = np_predict_time[~np.isnan(np_predict_time)]

predict_time_work = predict_time_work[::50]



index_work = np.arange(len(predict_time_work))
time_to_failure_per_100 = data.time_to_failure[::10]

index_time_to_failure = np.arange(len(time_to_failure_per_100))



predict_time_per_100 = data.predict_time[::10]

predict_time_per_100 = predict_time_per_100[~np.isnan(predict_time_per_100)]

index_predict_time = np.arange(len(predict_time_per_100))



plt.figure(figsize=(20, 10))

plt.scatter(x = index_time_to_failure, y = time_to_failure_per_100, s=5)

plt.tick_params(axis="x", labelsize = 15)

plt.tick_params(axis="y", labelsize = 15)

plt.title("Mesured time_to_failure",fontsize = 20)

plt.xlabel("index", fontsize = 20)

plt.ylabel("Mesured time_to_failure", fontsize = 20)



plt.figure(figsize=(20, 10))

plt.scatter(x = index_predict_time, y = predict_time_per_100, s=5)

plt.tick_params(axis="x", labelsize = 15)

plt.tick_params(axis="y", labelsize = 15)

plt.title("Predicted time_to_failure",fontsize = 20)

plt.xlabel("index", fontsize = 20)

plt.ylabel("Predicted time_to_failure", fontsize = 20)