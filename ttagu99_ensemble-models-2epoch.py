# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

import os
import os

data_path = "/kaggle/input/quickdraw-doodle-recognition/" 

print(os.listdir(data_path))
import pandas as pd

sub_df = pd.read_csv(data_path+'sample_submission.csv')

print("test data 수:",len(sub_df))

sub_df.head()
train_file_path = "/kaggle/input/quickdraw-doodle-recognition/train_raw/"
train_csvs= os.listdir(train_file_path)

print("train_raw 폴더 내 파일 수:", len(train_csvs))

print(train_csvs[:5])



file_size = 0

label_names = []



for csv_file in train_csvs:

    file_size += os.path.getsize(train_file_path + csv_file) # data file들의 용량을 계산

    label_names.append(csv_file.replace('.csv','')) 

print("파일 크기 : ", file_size//(1024*1024*1024) ,"GB")



label_names = sorted(label_names,key=lambda x : str.lower(x+'.csv')) # at kaggle notebook 
hold_out_set= 'train_k99'
import numpy as np

import json    
def preds2catids(predictions): # submission을 위해 top3 category로 변환할 함수

    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])

def map_at3(y_true, y_pred): 

    map3 = tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=1)*0.5

    map3 += tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=2)*0.17

    map3 += tf.keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)*0.33    

    return map3
ext_data_path = '/kaggle/input/doodle-model/'
import pandas as pd

import numpy as np



outputs = os.listdir(ext_data_path)

hold_out_probs = [ext_data_path+ f for f in outputs if f.find('ho_prob') >= 0 ] 

test_out_probs = [ext_data_path+ f for f in outputs if f.find('test_prob') >= 0 ] 

hold_out_probs = sorted(hold_out_probs)

test_out_probs = sorted(test_out_probs)

ho_df = pd.read_csv(ext_data_path+hold_out_set)

ho_s = []

for prob_path in hold_out_probs:

    ho = np.load(prob_path)

    ho = ho[:len(ho_df)]

    ho_s.append(ho)

targets = ho_df.y.to_numpy() # hold out target
ho_arr = np.stack(ho_s,axis=-1)

ho_arr.shape
del(ho_s)

import gc

gc.collect()
xin = tf.keras.layers.Input((len(label_names),ho_arr.shape[2]))

x = tf.keras.layers.Convolution1D(1,kernel_size=1,activation='linear',use_bias=False)(xin)

x = tf.keras.layers.Reshape(target_shape=(len(label_names),))(x)

wensemble_model = tf.keras.Model(inputs=xin, outputs=x)

wensemble_model.summary()
check_point=tf.keras.callbacks.ModelCheckpoint(monitor='map_at3',verbose=1

                               ,filepath='ensemble_w.h5',save_best_only=True,mode='max') 

wensemble_model.compile(optimizer=tf.keras.optimizers.Adam(1),loss='mse', metrics=[ map_at3])

wensemble_model.fit(x=ho_arr,y=tf.keras.utils.to_categorical(targets)

                    , epochs=20, batch_size=10000,verbose=1, callbacks=[check_point])
wensemble_model.load_weights('ensemble_w.h5')

wensemble_model.get_weights()
del(ho_arr)

gc.collect()
res = np.array(wensemble_model.get_weights()).squeeze()
ens_prob = np.zeros((len(sub_df),len(label_names)))

for i, prob_path in enumerate(test_out_probs):

    ens_prob += (np.load(prob_path) * res[i])
from collections import Counter,OrderedDict

from operator import itemgetter 



def balancing_predictions(test_prob, factor = 0.1, minfactor = 0.001, patient = 5, permit_cnt=332, max_search=10000, label_num=340):

    maxk = float('inf')

    s_cnt = np.zeros(label_num)

    for i in range(max_search):

        ctop1 = Counter(np.argmax(test_prob,axis=1))

        ctop1 = sorted(ctop1.items(), key=itemgetter(1), reverse=True)

        if maxk > ctop1[0][1]:

            maxk = ctop1[0][1]

        else:

            s_cnt[ctop1[0][0]]+=1

            if np.max(s_cnt)>patient:

                if factor< minfactor:

                    print('stop min factor')

                    break

                s_cnt=np.zeros(label_num)

                factor*=0.99

                print('reduce factor: ', factor, ', current max category num: ', ctop1[0][1])



        if ctop1[0][1] <= permit_cnt:

            print('idx: ',ctop1[0][0] ,', num: ', ctop1[0][1]) 

            break

        test_prob[:,ctop1[0][0]] *= (1.0-factor)

        

    return test_prob
bal_test_prob = balancing_predictions(ens_prob)

bal_top3 = preds2catids(bal_test_prob)

id2cat = {k: cat.replace(' ', '_') for k, cat in enumerate(label_names)}

bal_top3cats = bal_top3.replace(id2cat) 

sub_df['word'] = bal_top3cats['a'] + ' ' + bal_top3cats['b'] + ' ' + bal_top3cats['c']

bal_submission = sub_df[['key_id', 'word']]

bal_submission.to_csv('submission_bal_ens.csv', index=False)