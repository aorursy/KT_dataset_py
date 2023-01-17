
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

training_file  =  pd.read_csv('/kaggle/input/birdsong-recognition/train.csv')
training_file.head(10)
# Columns for Training File
print(training_file.columns)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
sns.set()
num = len(training_file.species.value_counts())//3
for i in range(num):
    part_species = training_file.species.value_counts()[3*i:3*(i+1)]
    labels = []
    count = []
    for key,value in part_species.items():
        labels.append(key)
        count.append(value)
    
    sns.barplot(x=labels,y=count)
    plt.show()
# Visualizing Bar Graph for Channels
print(f'No. of Mono Channels: {training_file.channels.value_counts()[0]}')
print(f'No. of Stereo Channels: {training_file.channels.value_counts()[1]}')
sns.countplot(x='channels',data=training_file)
plt.show()
# Sampling Rate Visualization
labels = []
count = []
for key,value in training_file.sampling_rate.value_counts().items():
    labels.append(key.split(' ')[0])
    count.append(value/training_file.shape[0])
    print(f'Sampling Rate: {key} , count: {value}')

sns.barplot(x=labels,y=count)
plt.ylabel('Percentage of Data')
plt.show()




!pip install ffmpy
from ffmpy import FFmpeg

f = FFmpeg(inputs={'/kaggle/input/birdsong-recognition/train_audio/aldfly/XC134874.mp3':None},
            outputs={'/kaggle/working/test.wav':'-ac 1 -ar 16000'})

f.run()


training_file
