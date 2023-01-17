# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
submission= pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
train= pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
def encode_for_bert(texts,tokenizer,max_len):
    complete_tokens=[]
    complete_masks=[]
    complete_segments=[]
    
    for text in texts:
        text=tokenizer.tokenize(text)
        text=text[:max_len-2]
        input_seq= ["[CLS]"]+text+["[SEP]"]
        padding_len= max_len-len(input_seq)
        
        tokens= tokenizer.convert_tokens_to_ids(input_seq)
        tokens=tokens+[0]*padding_len
        
        pad_masks=[1]*len(input_seq)+[0]*padding_len
        segment_ids= [0]*max_len
        
        
        complete_tokens.append(tokens)
        complete_masks.append(pad_masks)
        complete_segments.append(segment_ids)
    
    return np.array(complete_tokens),np.array(complete_masks),np.array(complete_segments)
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
    
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
import tokenization
def build_model(bert_layer, max_len):
    input_word_ids=Input(shape=(max_len,), dtype=tf.int32, name='input_word_ids')
    input_masks=Input(shape=(max_len,), dtype=tf.int32, name='input_masks')
    input_segments=Input(shape=(max_len,), dtype=tf.int32, name='input_segments')
    
    _,seq_out= bert_layer([input_word_ids,input_masks,input_segments])
    clf_out= seq_out[:,0,:]
    print(clf_out)
    out= Dense(1,activation='sigmoid')(clf_out)
    
    model= Model(inputs=[input_word_ids,input_masks,input_segments],outputs=out)
    model.compile(Adam(lr=1e-5),loss='binary_crossentropy',metrics=['accuracy'])
    
    return model

%%time
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train.head(5)
train_input= encode_for_bert(train.text.values,tokenizer,128)
test_input= encode_for_bert(test.text.values,tokenizer,128)
train_labels=train.target.values

model=build_model(bert_layer,max_len=128)
model.summary()
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

training= model.fit(train_input,train_labels,validation_split=0.15, epochs=2, callbacks=[checkpoint], batch_size=16)

model.load_weights('model.h5')
test_pred= model.predict(test_input)
submission['target']=test_pred.round().astype(int)
submission.to_csv('submission.csv', index=False)
