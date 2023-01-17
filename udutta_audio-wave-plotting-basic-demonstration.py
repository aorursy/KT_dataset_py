import numpy as np

import pandas as pd

import math

from scipy.io import wavfile

from numpy.fft import fft

import matplotlib.pyplot as plot


import os

os.getcwd()

os.chdir('/kaggle')

os.getcwd()

os.listdir('/kaggle/input')
import os

file_path='/kaggle/input'

file_name = 'audio_piano.wav'

file=os.path.join(file_path,file_name)

print(file)




plot.rcParams['xtick.labelsize']=14

plot.rcParams['ytick.labelsize']=14

plot.style.use('seaborn')

plot.show()







threshold= 800 # its the threshold frequency. we do not want any frequency below 800 as all sounds below 800 can be considered as noise in this case



fs, snd = wavfile.read(file) # read the file with piano music, filename = audio_piano



y= snd [:,0]



plot.figure(figsize =(20,8))

n = len(y)



p = fft(y) # fft is the numpy function to apply fourier transformation on the audio file



mag = np.sqrt(p.real**2 + p.imag**2)



mag = mag*2/n



mag = mag[0:math.ceil((n)/2.0)]

freq = np.arange(0, len(mag),1.0)*(fs/n)



plot.plot(freq/1000, mag , color = 'b')

# plot.xticks(np.arange(min(freq/1000), max(freq/1000)+1, 1.0))



# Lets try creating some sine waves

a = np.arange(2,10,2) # arange (start,end,step)

print(a)

b = np.linspace(2,10,5) # linspace(start, end, number)

print(b)







def genwave(freq, amp, T, shift, sr):

    time = np.arange(0,T,T/sr)

    X = amp*np.sin(2*np.pi*freq*time+shift)

    return time, X







time, amplitude = genwave(10,1,1,0, 1000)

fig = plot.figure(figsize=(15,5))



print(len(time))

print(len(amplitude))



ax = fig.add_axes([0,0,4,4])

ax.plot(time, amplitude, c = 'b')

plot.show()

ax.set_ylim([-4,4])



plot.rcParams['xtick.labelsize']=14

plot.rcParams['ytick.labelsize']=14

plot.style.use('seaborn')

ax.plot(time, amplitude, c = 'b')

plot.grid(True, which = 'both')

plot.show()
# a simple version of genwave without the phase shift

def genwave(freq, amp, T, sr):

    time = np.arange(0,T,T/sr)

    X = amp*np.sin(2*np.pi*freq*time)

    return time, X



plot.rcParams['xtick.labelsize']=14

plot.rcParams['ytick.labelsize']=14

plot.style.use('seaborn')

f,axarr = plot.subplots(4, figsize=(20,8))

sr =1000

x,y  = genwave (500,2,1,sr)

_, y2 = genwave(100,2,1,sr)

_,y3 =genwave(250,1,1,sr) 



y_final = y+y2+y3

axarr[0].plot(x,y_final)



axarr[1].plot(x,y)

axarr[2].plot(x,y2)

axarr[3].plot(x,y3)

def gen_wave(freq, amp, T, shift, sr):

    time = np.arange(0,T,T/sr)

    X = amp*np.sin(2*np.pi*freq*time+shift)

    return time, X





plot.rcParams['xtick.labelsize']=14

plot.rcParams['ytick.labelsize']=14

plot.style.use('seaborn')

f,axarr = plot.subplots(4, figsize=(20,8))

sr =1000

x,A=  gen_wave(1,3,10,0,100)

_, A2 = gen_wave(4, 20, 10, 180,100)

# _,y3 =genwave(250,1,1,sr) 



y_final = A +A2

axarr[0].plot(x,y_final)



axarr[1].plot(x,A)

axarr[2].plot(x,A2)

# axarr[3].plot(x,y3)