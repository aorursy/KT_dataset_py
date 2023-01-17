import os

os.listdir('../input/')
filename='../input/powerplay.wav' 

import librosa 

x, sr = librosa.load(filename,sr=16000)
int(librosa.get_duration(x, sr)/60)
max_slice=5 

window_length = max_slice * sr
import IPython.display as ipd 

a=x[21*window_length:22*window_length] 

ipd.Audio(a, rate=sr)
energy = sum(abs(a**2))

print(energy)
import matplotlib.pyplot as plt 

fig = plt.figure(figsize=(14, 8)) 

ax1 = fig.add_subplot(211) 

ax1.set_xlabel('time') 

ax1.set_ylabel('Amplitude') 

ax1.plot(a)
import numpy as np

energy = np.array([sum(abs(x[i:i+window_length]**2)) for i in range(0, len(x), window_length)])
import matplotlib.pyplot as plt 

plt.hist(energy) 

plt.show()
import pandas as pd

df=pd.DataFrame(columns=['energy','start','end'])

thresh=12000

row_index=0

for i in range(len(energy)):    

    value=energy[i]

    if(value>=thresh):

        i=np.where(energy == value)[0]

        df.loc[row_index,'energy']=value

        df.loc[row_index,'start']=i[0] * 5

        df.loc[row_index,'end']=(i[0]+1) * 5

        row_index= row_index + 1

temp=[]

i=0

j=0

n=len(df) - 2

m=len(df) - 1



while(i<=n):

    j=i+1

    while(j<=m):

        if(df['end'][i] == df['start'][j]):

            df.loc[i,'end'] = df.loc[j,'end']

            temp.append(j)

            j=j+1

        else:

            i=j

            break  

df.drop(temp,axis=0,inplace=True)
!pip install moviepy
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

start=np.array(df['start'])

end=np.array(df['end'])

for i in range(len(df)):

    if(i!=0):

        start_lim = start[i] - 5

    else:

        start_lim = start[i] 

    end_lim   = end[i]   

    filename="highlight" + str(i+1) + ".mp4"

    ffmpeg_extract_subclip("../input/powerplay.mp4",start_lim,end_lim,targetname=filename)