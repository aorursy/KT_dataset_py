#import library 

from __future__ import absolute_import, division, print_function, unicode_literals

import functools

import tensorflow as tf

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import math

import time

start_time = time.time()

from random import sample

%matplotlib inline

from tensorflow.keras.layers import Dense, Flatten, Conv2D

from tensorflow.keras import datasets, layers, models

from sklearn.metrics import classification_report

import matplotlib.pyplot as plt

from scipy import ndimage, misc

import seaborn as sns

%load_ext tensorboard

logdir='log'
df = pd.read_pickle('/kaggle/input/9type_wafer.pkl')
#type hash table

"""

failure Type mapping

mapping_type = {'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8,'unlabeld':9}

mapDim = max(300,202) min(6,21)

"""

class_names = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','none','unlabeld']
#check the size of each failureType

#failureNum 9 means 'not labled'

ftype_size = df.groupby('failureNum').size()
#sorting by failureNum



f0 = df.loc[df['failureNum']==0]

f1 = df.loc[df['failureNum']==1]

f2 = df.loc[df['failureNum']==2]

f3 = df.loc[df['failureNum']==3]

f4 = df.loc[df['failureNum']==4]

f5 = df.loc[df['failureNum']==5]

f6 = df.loc[df['failureNum']==6]

f7 = df.loc[df['failureNum']==7]

f8 = df.loc[df['failureNum']==8]

f9 = df.loc[df['failureNum']==9]

fail_df = pd.DataFrame({'fail_index':[f0,f1,f2,f3,f4,f5,f6,f7,f8,f9]}) #nested data frame

#fail_df['fail_index'].iloc[0] ,calling f0
#augmetation with rotation

#series = only wafermap column series

#times = magnitude of augmentation , which is integer (> 1),



#processing order : 1.zoom  2.rotation  3.ch2 4. series to array (s_a)



def rotation(series, times):

  origin = series

  operand = series

  rotated = pd.Series()

  for x in range(len(series)):

    

    for y in range(times-1):

      unit_angle = 360/times

      angle = unit_angle*(y+1)

      a = ndimage.rotate(operand.values[x],angle,reshape=False) #reshape option is necessary

      a = pd.Series([a])

      rotated = rotated.append(a)

      



  output = origin.append(rotated)

  return output #output is series
#convert waferMap series element into 2 channel image, pixel = [Die existence, fail existence]

#series has only waferMap column and output is series of pd



#processing order : 1.zoom  2.rotation  3.ch2 4. series to array (s_a)



def ch2(s): #s : series with only waferMap column

  

  output = pd.Series()

  

  buffer = s.values 

  data_count = len(buffer)  

  



  for d in range(data_count): #data number

    holder = np.array([])

    

    for i in range(len(buffer[d])): #row count

      for j in range(len(buffer[d][i])): #col count

        if(buffer[d][i][j]==0):

          holder = np.append(holder, [0,0])

        elif(buffer[d][i][j]==1):

          holder = np.append(holder, [1,0])  

        elif(buffer[d][i][j]== 2 or buffer[d][i][j]== 3 or buffer[d][i][j]== 4):  #after zoom or rotation, some elements are changed into 3 so, I add 3

          holder = np.append(holder, [1,1])

    

    holder = holder.reshape(len(buffer[d]),len(buffer[d][i]),2)      

    holder = [holder]

    holder = pd.Series(holder)

    output = output.append(holder)



  return output   #output is pd.Series
#size standardization with zoom

#zoom and rotation should be applied before ch2 func

#ts = integer , s = waferMap series



#processing order : 1.zoom  2.rotation  3.ch2 4. series to array (s_a)



def zoom(s, ts): # ts = target size, which is applied to both height and width

  

  rotated = pd.Series()

  b = s.values

  



  for x in range(len(b)):

    row = np.shape(b[x])[0]

    col = np.shape(b[x])[1]

    

    a = ndimage.zoom(b[x],[ts/row,ts/col])

    a = [a]

    a = pd.Series(a)    

    rotated = rotated.append(a)

  

  return rotated #rotated = series
#convert series(stadardized size by zoom) into np array image like (data numbers, height, width, channel) not (data number,)



#processing order : 1.zoom  2.rotation  3.ch2 4. series to array (s_a)



def s_a(s):#size of images must have same size 

  

  a = []

  arr = s.values

  

  image_height = (np.shape(arr[0]))[0]

  image_width = (np.shape(arr[0]))[1]

  



  for i in range(len(arr)):

    a = np.append(a,arr[i])

  

  a = a.reshape(len(arr), image_height, image_width, 2)



  return a #reshaped image array with 2channel
#dividing training data, testing data from data

#ratio = integer  less than 100, bigger than 0 

def tra_tes(df,ratio):

  size = len(df)

  mark = int(size*(ratio/100))

  

  train = df.iloc[0:mark]

  test = df.iloc[mark:size]



  return [train,test]
#make complete image array and label array

#df should has one fail type like only failunreNum:1

#d_size  is target size to augment

#z_size is size of zoomed image

def ima_lab(df,d_size,z_size):

  

  #for label

  l = len(df)

  failureNum = df['failureNum']

  holder = pd.Series()

    

  if(l<d_size):

    factor = math.ceil((d_size/l))

    l1 = failureNum

    for x in range(factor):

      holder = holder.append(l1)

  

    holder = holder[0:d_size]

    #this is process for turing series into array

    #holder = holder.values

    #holder = [holder]

    #holder = np.transpose(holder)

    label = holder



  elif(l>=d_size):

    holder = failureNum[0:d_size]

    #this is process for turing series into array

    #holder = holder.values

    #holder = [holder]

    #holder = np.transpose(holder)

    label = holder



  #for image

  waferMap = df['waferMap']

  i1 = waferMap

  

  if(l>=d_size):

    i1 = i1[0:d_size]

    i1 = zoom(i1,z_size)

    i1 = ch2(i1)

    #this is process for turing series into array

    #i1 = s_a(i1)

    image = i1

  

  elif(l<d_size):

    r_factor = math.ceil(d_size/l)

    

    i1 = zoom(i1,z_size)

    i1 = rotation(i1, r_factor)

    i1 = i1[0:d_size]

    i1 = ch2(i1)

    #this is process for turing series into array

    #i1 = s_a(i1)

    image = i1



  return [image, label] #image = series, label = series
"""

failureNum data size

0      4294

1       555

2      5189

3      9680

4      3593

5       866

6      1193

7       149

8    147431

9    638507

"""



s0 = f0.sample(n=3000, random_state = 11)

s0 = s0.sample(frac=1)

[tra0, tes0] = tra_tes(s0,80)



s1 = f1.sample(n=555, random_state = 22)

s1 = s1.sample(frac=1)

[tra1, tes1] = tra_tes(s1,80)



s2 = f2.sample(n=3000, random_state = 33)

s2 = s2.sample(frac=1)

[tra2, tes2] = tra_tes(s2,80)



s3 = f3.sample(n=3000, random_state = 44)

s3 = s3.sample(frac=1)

[tra3, tes3] = tra_tes(s3,80)



s4 = f4.sample(n=3000, random_state = 55)

s4 = s4.sample(frac=1)

[tra4, tes4] = tra_tes(s4,80)



s5 = f5.sample(n=866, random_state = 66)

s5 = s5.sample(frac=1)

[tra5, tes5] = tra_tes(s5,80)



s6 = f6.sample(n=1193, random_state = 77)

s6 = s6.sample(frac=1)

[tra6, tes6] = tra_tes(s6,80)



s7 = f7.sample(n=149, random_state = 88)

s7 = s7.sample(frac=1)

[tra7, tes7] = tra_tes(s7,80)



tra_df = pd.DataFrame({'fail_type':[tra0,tra1,tra2,tra3,tra4,tra5,tra6,tra7]})

tes_df = pd.DataFrame({'fail_type':[tes0,tes1,tes2,tes3,tes4,tes5,tes6,tes7]})
tra_i = pd.Series()

tra_l = pd.Series()



tes_i = pd.Series()

tes_l = pd.Series()

for i in range(8):

  [image, label] = ima_lab(tra_df['fail_type'].iloc[i], 3000, 25 )

  

  tra_i = tra_i.append(image)

  tra_l = tra_l.append(label)



print(np.shape(tra_i))

print(np.shape(tra_l))



for i in range(8):

  [image, label] = ima_lab(tes_df['fail_type'].iloc[i], 750, 25 )

  

  tes_i = tes_i.append(image)

  tes_l = tes_l.append(label)



print(np.shape(tes_i))

print(np.shape(tes_l))

print("--- %s seconds ---" % (time.time() - start_time))
#image data

tra_i = s_a(tra_i)

tes_i = s_a(tes_i)



#label

tra_l = tra_l.values

tra_l = [tra_l]

tra_l = np.transpose(tra_l)



for_conmat = tes_l #which is series

tes_l = tes_l.values

tes_l = [tes_l]

tes_l = np.transpose(tes_l)



print(np.shape(tra_i))

print(np.shape(tra_l))

print(np.shape(tes_i))

print(np.shape(tes_l))

print("--- %s seconds ---" % (time.time() - start_time))
model = models.Sequential()





model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(25, 25, 2)))

model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))

model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))



model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))

model.add(layers.Dense(64, activation='relu'))

model.add(layers.Dense(8))





model.summary()
model.compile(optimizer='adam',

              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),

              metrics=['accuracy'])

history = model.fit(tra_i, tra_l, epochs=10, 

                    validation_data=(tes_i, tes_l))
plt.plot(history.history['accuracy'], label='accuracy')

plt.plot(history.history['val_accuracy'], label = 'val_accuracy')

plt.xlabel('Epoch')

plt.ylabel('accuracy')

plt.ylim([0.7, 1])

plt.legend(loc='lower right')



test_loss, test_acc = model.evaluate(tes_i,  tes_l, verbose=2)



print("--- %s seconds ---" % (time.time() - start_time))
labels = for_conmat.values

y_true = labels

y_pred=model.predict_classes(tes_i)
classes=[0,1,2,3,4,5,6,7]
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()

con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)



con_mat_df = pd.DataFrame(con_mat_norm,

                     index = classes, 

                     columns = classes)



figure = plt.figure(figsize=(8, 8))

sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)

plt.tight_layout()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()
