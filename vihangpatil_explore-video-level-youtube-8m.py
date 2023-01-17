# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from IPython.display import YouTubeVideo



video_level_record = "../input/video_level/train-1.tfrecord"

frame_level_record = "../input/frame_level/train-1.tfrecord"
vid_ids = []

labels = []

mean_rgb = []

mean_audio = []



for example in tf.python_io.tf_record_iterator (video_level_record):

    tf_example = tf.train.Example.FromString(example)

    

    vid_ids.append(tf_example.features.feature['video_id'].bytes_list.value[0].decode(encoding='UTF-8'))

    labels.append(tf_example.features.feature['labels'].int64_list.value)

    mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)

    mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)
print('The total number of videos in this record',len(vid_ids))

print('First video feature length is',len(mean_rgb[0]))

print('FIrst 20 features of the video (',vid_ids[0],')')

print(mean_rgb[0][0:19])
print('mean_audio has length',len(mean_audio))

print('mean_audio feature length for first 5 video:')

print([len(x) for x in mean_audio][:5])

print('mean_rgb feature length for first 5 video:')

print([len(x) for x in mean_rgb][:5])
#Labels for first 50 videos

type(labels)

print([x[:] for x in labels][:50])

print('Lengths for first 20 labels:')

print([len(x[:]) for x in labels][:50])

print('Mean of the number of labels for this record:')

print(np.mean([len(x[:]) for x in labels]))