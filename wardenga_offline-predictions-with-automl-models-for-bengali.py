import tensorflow.compat.v1 as tf #modyficatin for tensorflow 2.1 might follow soon

tf.disable_v2_behavior()

import pandas as pd

import numpy as np

import io

from matplotlib.image import imsave

import csv

import os

import time

import gc
def make_predict_batch(img,export_path):

    """

    INPUT

        -`img` list of bytes representing the images to be classified

        

    OUTPUT

        -dataframe containing the probabilities of the labels and the la

        els as columnames

    """

    

    

    with tf.Session(graph=tf.Graph()) as sess:

        tf.saved_model.loader.load(sess, ['serve'], export_path)

        graph = tf.get_default_graph()

        

        feed_dict={'Placeholder:0':img}

        y_pred=sess.run(['Softmax:0','Tile:0'],feed_dict=feed_dict)

        

        if len(img)==1:

            labels=[label.decode() for label in y_pred[1]]

        else:

            labels=[label.decode() for label in y_pred[1][0]]

        

    return pd.DataFrame(data=y_pred[0],columns=labels)
i=0 

name=f'train_image_data_{i}.parquet'

test_img = pd.read_parquet('../input/bengaliai-cv19/'+name)[0:10]

test_img.head()
height=137

width=236

#we need the directory of the saved model

dir_path='../input/trained-models/Trained_Models/tf_saved_model-Bengaliai_vowel-2020-01-27T205839579Z'



images=test_img.iloc[:, 1:].values.reshape(-1, height, width)

image_id=test_img.image_id

imagebytes=[]

for i in range(test_img.shape[0]):

    imageBytearray=io.BytesIO()

    imsave(imageBytearray,images[i],format='png')

    imagebytes.append(imageBytearray.getvalue())



res=make_predict_batch(imagebytes,dir_path)

res['image_id']=image_id

res.head()
res.drop(['image_id'],axis=1).idxmax(axis=1)
#walk the working directory to find the names of the directories

import os 

inputFolder = '../input/' 

for root, directories, filenames in os.walk(inputFolder): 

    for filename in filenames: print(os.path.join(root,filename))
def make_submit(images,height=137,width=236):

    """

    

    """

    consonant_path='../input/trained-models/Trained_Models/tf_saved_model-Bengaliai_consonant-2020-01-27T205840376Z'

    root_path='../input/trained-models/Trained_Models/tf_saved_model-Bengaliai_root-2020-01-27T205838805Z'

    vowel_path='../input/trained-models/Trained_Models/tf_saved_model-Bengaliai_vowel-2020-01-27T205839579Z'

    num=images.shape[0]

    #transform the images from a dataframe to a list of images and then bytes

    image_id=images.image_id

    images=images.iloc[:, 1:].values.reshape(-1, height, width)

    imagebytes=[]

    for i in range(num):

        imageBytearray=io.BytesIO()

        imsave(imageBytearray,images[i],format='png')

        imagebytes.append(imageBytearray.getvalue())

    

    #get the predictions from the three models - passing the bytes_list

    start_pred=time.time()

    prediction_root=make_predict_batch(imagebytes,export_path=root_path)

    prediction_consonant=make_predict_batch(imagebytes,export_path=consonant_path)

    prediction_vowel=make_predict_batch(imagebytes,export_path=vowel_path)

    end_pred=time.time()

    print('Prediction took {} seconds.'.format(end_pred-start_pred))

    

    start_sub=time.time()

    p0=prediction_root.idxmax(axis=1)

    p1=prediction_vowel.idxmax(axis=1)

    p2=prediction_consonant.idxmax(axis=1)

        

    row_id = []

    target = []

    for i in range(len(image_id)):

        row_id += [image_id.iloc[i]+'_grapheme_root', image_id.iloc[i]+'_vowel_diacritic',image_id.iloc[i]+'_consonant_diacritic']

        target += [p0[i], p1[i], p2[i]]

        

    submission_df = pd.DataFrame({'row_id': row_id, 'target': target})

    #submission_df.to_csv(name, index=False)

        

    end_sub=time.time()

    print('Writing the submission_df took {} seconds'.format(end_sub-start_sub))

    return submission_df
with open('submission.csv','w') as sub:

    writer=csv.writer(sub)

    writer.writerow(['row_id','target'])



batchsize=1000



start = time.time()

for i in range(4):

    start1 = time.time()

    name=f'test_image_data_{i}.parquet'

    print('start with '+name+'...')

    test_img = pd.read_parquet('../input/bengaliai-cv19/'+name)

    

    print('starting prediction')

    start1 = time.time()

    #split into smaler filesl

    for r in range(np.ceil(test_img.shape[0]/batchsize).astype(int)):

            

        df=make_submit(test_img[r*batchsize:np.minimum((r+1)*batchsize,test_img.shape[0]+1)])

        df.to_csv('submission.csv',mode='a',index=False,header=False)

    

    end1 = time.time()

    print(end1 - start1)

    del test_img



end = time.time()

print(end - start)
df.head()