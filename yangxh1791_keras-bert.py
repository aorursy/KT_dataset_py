!pip install --upgrade tensorflow==2.2.0-rc2

!pip install bert-for-tf2

!pip install tensorflow-addons

!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip

!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip
import tensorflow as tf

import tensorflow_hub as hub

import bert

from tensorflow.keras.models import  Model

from tqdm import tqdm

import numpy as np

from collections import namedtuple

from imblearn.over_sampling import RandomOverSampler

import pandas as pd

from sklearn.model_selection import train_test_split

from collections import Counter

# multi-label hamming loss

import tensorflow_addons as tfa

hl = tfa.metrics.HammingLoss(mode='multilabel', threshold=0.4)
bert_layer=hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=False)

MAX_SEQ_LEN=128

input_word_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,

                                       name="input_word_ids")

input_mask = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,

                                   name="input_mask")

segment_ids = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32,

                                    name="segment_ids")

pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
@tf.function

def custom_recall(y_true,y_pred):

    y_true = tf.convert_to_tensor(y_true)

    Y_PRED = tf.cast(y_pred>=0.4,tf.float32)

    recall = tf.math.divide_no_nan(tf.math.reduce_sum(tf.math.multiply(y_true,Y_PRED)),tf.math.reduce_sum(y_true))

    return recall

x = tf.keras.layers.Dropout(0.1)(pooled_output)

x= tf.keras.layers.Dense(128, activation='relu')(x)

x = tf.keras.layers.Dropout(0.1)(x)

out = tf.keras.layers.Dense(6, activation="sigmoid", name="dense_output")(x)

model = tf.keras.models.Model(

      inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=[hl,custom_recall])

model.summary()
FullTokenizer=bert.bert_tokenization.FullTokenizer

vocab_file=bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case=bert_layer.resolved_object.do_lower_case.numpy()

tokenizer=FullTokenizer(vocab_file,do_lower_case)
# Print out RECALL metric ,Hamming loss and Confusion matrix for the results.

#

def results(pred_results,ground_truth,threshold):

    hl = tfa.metrics.HammingLoss(mode='multilabel', threshold=threshold)

    hl.update_state(ground_truth, pred_results)

    print('Hamming loss:', hl.result().numpy())

    pred_results = (pred_results>=threshold).astype(int)

    print("Positive labels RECALL-(thre:{}):  {}%".format(threshold,100*np.sum(pred_results*ground_truth)/np.sum(ground_truth)))

    cf = tfa.metrics.MultiLabelConfusionMatrix(num_classes=6)

    cf.update_state(ground_truth, pred_results)

    print('Confusion matrix:', cf.result().numpy())

    return 



# Resampling of training set based on the 'k'th label (k=0,1,..5)

#   Return: Resampled set of 100K samples.

#

def RS(k,features_train,ratio=1.0):

    train_y = features_train[:,3*128:390]

    label = train_y[:,k]

    rus = RandomOverSampler(ratio,random_state=42)

    features_train_res, _ = rus.fit_resample(features_train, label)

    np.random.shuffle(features_train_res)

    features_train_res = features_train_res[:100000,:]

    return features_train_res



def index_set(k,pattern):

    m= 2**(5-k)

    return set([i for i in range(64) if (i//m%2)!=0]).intersection(set(pattern.keys()))



def label_to_class(v):

    w = np.array([32,16,8,4,2,1],dtype=np.int16)

    return np.sum(v*w,axis=1)



def class_to_label(c):

    c=np.reshape(c,(-1,1))

    v5=c%2

    c=c//2

    v4=c%2

    c=c//2

    v3=c%2

    c=c//2

    v2=c%2

    c=c//2

    v1=c%2

    c=c//2

    v0=c%2

    return np.concatenate([v0,v1,v2,v3,v4,v5],axis=1)

def distribution(pattern):

    b=class_to_label(np.array(list(pattern.keys())))

    a=np.asarray(list(pattern.values())).reshape(-1,1)

    M = a*b

    w_vec=np.sum(M,axis=0)

    #return w_vec/np.sum(w_vec)

    return w_vec

def get_masks(tokens, max_seq_length):

    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def get_segments(tokens, max_seq_length):

    """Segments: 0 for the first sequence, 1 for the second"""

    segments = []

    current_segment_id = 0

    for token in tokens:

        segments.append(current_segment_id)

        if token == "[SEP]":

            current_segment_id = 1

    return segments + [0] * (max_seq_length - len(tokens))

def get_ids(tokens, tokenizer, max_seq_length):

    """Token ids from Tokenizer vocab"""

    token_ids = tokenizer.convert_tokens_to_ids(tokens,)

    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))

    return input_ids



def create_single_input(sentence,MAX_LEN):

    stokens = tokenizer.tokenize(sentence)

    stokens = stokens[:MAX_LEN]

    stokens = ["[CLS]"] + stokens + ["[SEP]"]

    ids = get_ids(stokens, tokenizer, MAX_SEQ_LEN)

    masks = get_masks(stokens, MAX_SEQ_LEN)

    segments = get_segments(stokens, MAX_SEQ_LEN)

    return ids,masks,segments

def create_input_array(sentences):

    input_ids, input_masks, input_segments = [], [], []

    for sentence in tqdm(sentences,position=0, leave=True):

        ids,masks,segments=create_single_input(sentence,MAX_SEQ_LEN-2)

        input_ids.append(ids)

        input_masks.append(masks)

        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), 

            np.asarray(input_masks, dtype=np.int32), 

            np.asarray(input_segments, dtype=np.int32)]

df=pd.read_csv('/kaggle/working/train.csv')

df = df.sample(frac=1)

train_sentences = df["comment_text"].fillna("CVxTz").values

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

inputs=create_input_array(train_sentences)



train_y = df[list_classes].values

features=np.concatenate(inputs,axis=1)

features=np.concatenate((features,train_y),axis=1)



features_train,features_test,train_y,_= train_test_split(features, train_y,test_size=0.2,random_state=42)

print("Train set samples:",train_y.shape[0])



print("Positive labels distribution:",distribution(Counter(label_to_class(train_y)))/train_y.shape[0])



prior = distribution(Counter(label_to_class(train_y)))/train_y.shape[0]

list_trains = []

features_train_res= RS(3,features_train)

list_trains.append(features_train_res)

features_train_res= RS(5,features_train)

list_trains.append(features_train_res)

features_train_res= RS(1,features_train)

list_trains.append(features_train_res)

features_train_res= RS(4,features_train)

list_trains.append(features_train_res)

features_train_res= RS(2,features_train)

list_trains.append(features_train_res)

features_train_res= RS(0,features_train)

list_trains.append(features_train_res)



# Concatenate six rounds of resampling,each producing 100K samples, making a total of 600K sample training set

features_train=np.concatenate(list_trains,axis=0)

np.random.shuffle(features_train)

train_y = features_train[:,3*128:390]

after = distribution(Counter(label_to_class(train_y)))/600000

print("Positive sample amplification through resampling:",after/prior)

print("Positive labels distribution after resampling:",after)



X = np.split(features_train,[128,2*128,3*128,390],axis=1)

input_ids=X[0]

input_masks=X[1]

input_segments=X[2]

train_y= X[3].astype(np.float32)



X = np.split(features_test,[128,2*128,3*128,390],axis=1)

input_ids_test=X[0]

input_masks_test=X[1]

input_segments_test=X[2]

test_y= X[3].astype(np.float32)



inputs_train=[input_ids,input_masks,input_segments]

inputs_test=[input_ids_test,input_masks_test,input_segments_test]

print("Test set samples:",test_y.shape[0])

print("Train set samples after resampling:",train_y.shape[0])
model.fit(inputs_train,train_y,epochs=2,batch_size=64,shuffle=True)

model.save_weights('/kaggle/working/chkpt')

#model.load_weights('/kaggle/input/keras-bert/chkpt')
pred_results = model.predict(inputs_test,batch_size=256)

ground_truth = test_y.astype(int)

results(pred_results,ground_truth,0.4)

results(pred_results,ground_truth,0.3)

!rm /kaggle/working/*.csv