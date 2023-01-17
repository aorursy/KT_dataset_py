!pip install transformers==3.0.2

!pip install nlp
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from nlp import load_dataset

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



np.random.seed(1234) 

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
mnli = load_dataset(path='glue', name='mnli') # loading more data from the Huggin face dataset

#snli   =  load_dataset("snli") # loading more data from the Huggin face dataset
xnli = pd.read_csv('../input/xnli-organized/xnli_df.csv') # loading pre organized XNLI dataset

xnli = xnli.rename(columns = {'Unnamed: 0': 'lang_abv', '0' : 'premise', '1': 'hypothesis', '0.1': 'label' }) # renaming the columns

xnli.head()
mnli_premise = pd.Series(mnli['train']['premise'])

mnli_hypothesis = pd.Series(mnli['train']['hypothesis'])

mnli_label = pd.Series(mnli['train']['label'])



#snli_premise = pd.Series(snli['train']['premise'])

#snli_hypothesis = pd.Series(snli['train']['hypothesis'])

#snli_label = pd.Series(snli['train']['label'])





#snli = None # cleaning memory 

mnli = None # cleaning memory





#snli_df = pd.DataFrame(pd.concat([snli_premise, snli_hypothesis, pd.Series(['en'] * len(snli_label)), snli_label], axis = 1))

mnli_df = pd.DataFrame(pd.concat([mnli_premise, mnli_hypothesis, pd.Series(['en'] * len(mnli_label)), mnli_label], axis = 1))





mnli_premise = None #more memory cleaning

mnli_hypothesis = None # more memory clenaning

mnli_label = None #more memory cleaning



#snli_premise = None #more memory cleaning

#snli_hypothesis = None # more memory clenaning

#snli_label = None #more memory cleaning
mnli_df = mnli_df.rename(columns = {0 : 'premise', 1: 'hypothesis', 2: 'lang_abv', 3: 'label' })

print(mnli_df.shape)

display(mnli_df.head())



#snli_df = snli_df.rename(columns = {0 : 'premise', 1: 'hypothesis', 2: 'lang_abv', 3: 'label' })

#print(snli_df.shape)

#display(snli_df.head())

train_df = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')

print('Traning Data, the size of the dataset is: {} \n'.format(train_df.shape))



test_df = pd.read_csv('../input/contradictory-my-dear-watson/test.csv')
train_df = pd.concat([train_df, xnli, mnli_df]) #appending the original dataset to the additional datasets

train_df = train_df[train_df['label'] != -1] #cleaning values with the wrong label

mnli_df = None

snli_df = None



print('the shape of the whole DF to be used is: ' + str(train_df.shape))
# searching for duplicates



train_df = train_df[train_df.duplicated() == False]

print('the shape of the whole DF to be used is: ' + str(train_df.shape))
import seaborn as sns

import matplotlib.pyplot as plt



fig = plt.figure(figsize = (15,5))



plt.subplot(1,2,1)

plt.title('Traning data language distribution')

sns.countplot(data = train_df, x = 'lang_abv', order = train_df['lang_abv'].value_counts().index)



plt.subplot(1,2,2)

plt.title('Test data laguage distribution')

sns.countplot(data = test_df, x = 'lang_abv', order = test_df['lang_abv'].value_counts().index)
# word count



def word_count(dataset, column):

    len_vector = []

    for text in dataset[column]:

        len_vector.append(len(text.split()))

    

    return len_vector



train_premise = word_count(train_df, 'premise')

train_hypothesis = word_count(train_df, 'hypothesis')



test_premise = word_count(test_df, 'premise')

test_hypothesis = word_count(test_df, 'hypothesis')



fig = plt.figure(figsize = (15,10))



plt.subplot(2,2,1)

plt.title('word count for train dataset premise')

sns.distplot(train_premise)



plt.subplot(2,2,2)

plt.title('word count for train dataset hypothesis')

sns.distplot(train_hypothesis)



plt.subplot(2,2,3)

plt.title('word count for test dataset premise')

sns.distplot(test_premise)



plt.subplot(2,2,4)

plt.title('word count for test dataset hypothesis')

sns.distplot(test_hypothesis)        
# looking at the countplot of the labels of the traning data set



plt.title('Label column countplot')

sns.countplot(data = train_df, x = 'label')
from transformers import BertTokenizer, TFAutoModel, AutoTokenizer

import tensorflow as tf

import keras

from tensorflow.math import softplus, tanh

from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras import Input, Model, Sequential

from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM, Embedding, GlobalAveragePooling1D

from keras.preprocessing.sequence import pad_sequences

from keras.optimizers import Adam

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from tensorflow.keras import regularizers





np.random.seed(123)

max_len = 50



# this is the model used BERT huggin face



Bert_model = "bert-large-uncased"



# tokenizer



Bert_tokenizer = BertTokenizer.from_pretrained(Bert_model)



def tokeniZer(dataset,tokenizer):

    encoded_list = [] # word id array

    type_id_list = np.zeros((dataset.shape[0], max_len)) #type id array

    mask_list = np.zeros((dataset.shape[0], max_len)) #masks array

    

    for i in range(dataset.shape[0]):

        datapoint = '[CLS] ' + dataset['premise'][i] + ' [SEP]' + dataset['hypothesis'][i] + ' [SEP]' # putting the two sentences together along with special characters

        datapoint = tokenizer.tokenize(datapoint)

        datapoint = tokenizer.convert_tokens_to_ids(datapoint)

        encoded_list.append(datapoint) 

    

    encoded_list = pad_sequences(encoded_list, maxlen = max_len, padding = 'post')

    

    for i in range(encoded_list.shape[0]):

        flag = 0

        a = encoded_list[i]

        for j in range(len(a)):

            

            #building the type_id matrix

            

            if flag == 0:

                type_id_list[i,j] = 0

            else:

                type_id_list[i,j] = 1

                

            #flag for the type_id matrix

            

            if encoded_list[i,j] == 102:

                flag = 1

            

    

            #building the mask matrix 

            

            if encoded_list[i,j] == 0:

                mask_list[i,j] = 0

            else:

                mask_list[i,j] = 1

                

    return encoded_list,mask_list,type_id_list

        

        

        
# softplus - log(exp(x)+1), function that can be used for extra layers in the models

def mish(x):

    return x*tanh(softplus(x))

get_custom_objects()["mish"] = Activation(mish)
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)



# instantiate a distribution strategy

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
# model creator



def create_BERT(random_seed):

    

    tf.random.set_seed(random_seed)

    

    with tpu_strategy.scope():

    

        transformer_encoder = TFAutoModel.from_pretrained(Bert_model)



        input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")

        input_masks = Input(shape = (max_len,), dtype = tf.int32, name = 'input_mask')

        input_type_id = Input(shape = (max_len,), dtype = tf.int32, name = 'input_type_id')



        sequence_output = transformer_encoder([input_ids, input_masks, input_type_id])[0]



        cls_token = sequence_output[:, 0, :]



        #cls_token = Dense(512, activation = 'mish')(cls_token) # this layer improves the accuracy by several points about 5%



        #cls_token = Dropout(0.2)(cls_token)



        #cls_token = Dense(256, activation  = 'mish')(cls_token)



        #cls_token = Dropout(0.3)(cls_token)



        output_layer = Dense(3, activation='softmax')(cls_token)





        model = Model(inputs=[input_ids, input_masks, input_type_id], outputs = output_layer)



        model.summary()



        model.compile(Adam(lr=1e-5), 

                loss='sparse_categorical_crossentropy', 

                metrics=['accuracy']

            )

    return model
#ensemble creation and prediction



from sklearn.utils import shuffle # shuffle dataframes

#import random

#random.seed(123) # random seed to generate random list of numbers



#number_of_models = 4 #number of BERT models to be used in the ensemble 



#randomlist = random.sample(range(10, 1000), number_of_models) #creates a list of random integers that will be used for seeding the BERT models



#history_list = [0] * number_of_models #list to save the training history of the models



callbacks = [tf.keras.callbacks.EarlyStopping(patience = 2, monitor = 'val_loss', \

                                           restore_best_weights = True, mode = 'min')]

#predictions_list = [] #list to store the predictions of each model in the ensemble



#for i in range(number_of_models):

#tf.keras.backend.clear_session() #clear session to save memory 

#BertTokenizer = AutoTokenizer.from_pretrained(Bert_model)



shuffled_data = shuffle(train_df).reset_index(drop = True)#shuffle the data to add more variance





train_df = None #clearing more memory



#input_ids_train, input_masks_train, type_id = tokeniZer(shuffled_data, Bert_tokenizer) #encode shuffled data



batch_size = 128



#Bert = create_BERT(1234) #creates a single BERT model with a random seed

#history_bert = Bert.fit([input_ids_train, input_masks_train, type_id], shuffled_data['label'],

#                         validation_split = 0.2,

#                         epochs = 30, batch_size = batch_size, callbacks = callbacks)
XLM_model = "jplu/tf-xlm-roberta-large"

xlm_tokenizer = AutoTokenizer.from_pretrained(XLM_model) #Xlm tokenizer





X_train_ids, X_train_masks, _ = tokeniZer(shuffled_data,xlm_tokenizer) #encoding input
# creating the XLM model 



def create_xlm(transformer_layer,  random_seed, learning_rate = 1e-5):

    

    tf.keras.backend.clear_session()



    tf.random.set_seed(random_seed)

    

    with tpu_strategy.scope():

    

        input_ids = Input(shape = (max_len,), dtype = tf.int32)

        input_masks = Input(shape = (max_len,), dtype = tf.int32)

        #input_type_id = Input(shape = (max_len,), dtype = tf.int32)



            #insert roberta layer

        roberta = TFAutoModel.from_pretrained(transformer_layer)

        roberta = roberta([input_ids, input_masks])[0]



        #only need <s> token here, so we extract it now

        #out = roberta[:, 0, :]

        

        # using Avg pooling instead of the CLS token only

        

        out = GlobalAveragePooling1D()(roberta)





        #two layers with mish activation

        #out = tf.keras.layers.Dense(512, activation='mish')(out)

        

        #add optional Dense layer with dropout

        #out = tf.keras.layers.Dropout(0.2)(out)

        

        #out = tf.keras.layers.Dense(256, activation='mish')(out)

        

        #out = Dropout(0.3)(out)

                



                #add our softmax layer

        out = Dense(3, activation = 'softmax')(out)



        #assemble model and compile





        model = Model(inputs = [input_ids, input_masks], outputs = out)

        model.compile(

                                optimizer = Adam(lr = learning_rate), 

                                loss = 'sparse_categorical_crossentropy', 

                                metrics = ['accuracy'])

    model.summary()

        

    return model  





Xlm = create_xlm(XLM_model ,123443334, 1e-5)
#STEPS_PER_EPOCH = int(train_df.shape[0] // batch_size)



history_xlm = Xlm.fit([X_train_ids, X_train_masks], shuffled_data['label'],

          batch_size = batch_size,

        validation_split = 0.2,

         epochs = 39, callbacks = callbacks)
# preprocessing test data



input_ids_test_xml, input_masks_test_xml, _ = tokeniZer(test_df, xlm_tokenizer)

#input_ids_test_bert, input_masks_test_bert, input_type_id_test = tokeniZer(test_df, Bert_tokenizer)

#input_ids_test_xlm1, input_masks_test_xlm1, input_type_ids_test_xlm1 = tokeniZer(TTA1, xlm_tokenizer)

#input_ids_test_xlm2, input_masks_test_xlm2, input_type_ids_test_xlm2 = tokeniZer(TTA2, xlm_tokenizer)

#input_ids_test_xlm3, input_masks_test_xlm3, input_type_ids_test_xlm3 = tokeniZer(TTA3, xlm_tokenizer)

#input_ids_test_xlm4, input_masks_test_xlm4, input_type_ids_test_xlm4 = tokeniZer(TTA4, xlm_tokenizer)



#model predictions



predictions_xlm = Xlm.predict([input_ids_test_xml, input_masks_test_xml])

#predictions_bert = Bert.predict([input_ids_test_bert, input_masks_test_bert,input_type_id_test])

#predictions_xlm1 = Xlm.predict([input_ids_test_xlm1, input_masks_test_xlm1, input_type_ids_test_xlm1])

#predictions_xlm2 = Xlm.predict([input_ids_test_xlm2, input_masks_test_xlm2, input_type_ids_test_xlm2])

#predictions_xlm3 = Xlm.predict([input_ids_test_xlm3, input_masks_test_xlm3, input_type_ids_test_xlm3])

#predictions_xlm4 = Xlm.predict([input_ids_test_xlm4, input_masks_test_xlm4, input_type_ids_test_xlm4])



predictions = predictions_xlm



#final = np.zeros(predictions.shape[0])

#for i in range(predictions.shape[0]):

final = np.argmax(predictions, axis = 1)    



submission = pd.DataFrame()    



submission['id'] = test_df['id']

submission['prediction'] = final.astype(np.int32)



submission.to_csv('submission.csv', index = False)