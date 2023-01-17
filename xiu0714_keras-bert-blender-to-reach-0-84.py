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
!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py

!pip install sentencepiece
%%time

import tensorflow_hub as hub 

import tokenization
# !pip install sentencepiece

import numpy as np

import pandas as pd

import tensorflow as tf

import tensorflow_hub as hub

import itertools

import logging



logging.basicConfig(level=logging.INFO)



class Blender():

    def __init__(self, tweet_len=50, module_url=None):

        # The length of tweets we utilizing

        self.tweet_len = tweet_len

        self.module_url = module_url



    def load_data(self):

        self.train = pd.read_csv('../input/nlp-getting-started/train.csv')

        self.test = pd.read_csv('../input/nlp-getting-started/test.csv')

        self.sub = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

        print("Data loaded")

    

    def get_bert_layer(self):

        self.bert_layer = hub.KerasLayer(self.module_url, trainable=True)

        vocab_file = self.bert_layer.resolved_object.vocab_file.asset_path.numpy()

        do_lower_case = self.bert_layer.resolved_object.do_lower_case.numpy()

        self.tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

        print("Bert layer added")



    def bert_encode(self, texts):

        all_tokens = []

        all_masks = []

        all_segments = []

        max_len = self.tweet_len

        

        for text in texts:

            text = self.tokenizer.tokenize(text)

                

            text = text[:max_len-2]

            input_sequence = ["[CLS]"] + text + ["[SEP]"]

            pad_len = max_len - len(input_sequence)

            

            tokens = self.tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len

            pad_masks = [1] * len(input_sequence) + [0] * pad_len

            segment_ids = [0] * max_len

            

            all_tokens.append(tokens)

            all_masks.append(pad_masks)

            all_segments.append(segment_ids)

        

        return np.array(all_tokens), np.array(all_masks), np.array(all_segments)



    def build_model(self):

        max_len = self.tweet_len

        input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

        input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")

        segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")



        pooled_output, sequence_output = self.bert_layer([input_word_ids, input_mask, segment_ids])

        clf_output = sequence_output[:, 0, :]

        net = tf.keras.layers.Dense(64, activation='relu')(clf_output)

        net = tf.keras.layers.Dropout(0.2)(net)

        net = tf.keras.layers.Dense(32, activation='relu')(net)

        net = tf.keras.layers.Dropout(0.2)(net)

        out = tf.keras.layers.Dense(1, activation='sigmoid')(net)

        

        model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)

        model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

        

        print("Model built")

        return model

    

    def run(self):

        train_input = self.bert_encode(self.train.text.values)

        test_input = self.bert_encode(self.test.text.values)

        train_labels = self.train.target.values



        model = self.build_model()



        checkpoint = tf.keras.callbacks.ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True, verbose=1)

        earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)



        train_history = model.fit(

            train_input, train_labels, 

            validation_split=0.1,

            epochs=30,

            callbacks=[checkpoint, earlystopping],

            batch_size=16,

            verbose=1

        )



        print("Train completed.")

        

        model.load_weights('model.h5')

        test_pred = model.predict(test_input)

        self.sub['target'] = test_pred.round().astype(int)

        export_file = 'predictions_' + '_'.join(self.module_url.split('/')[-2:]) + '_' + str(self.tweet_len) + '.csv'

        self.sub.to_csv(export_file, index=False)

        

        print(f"Predictions exported to {export_file}.")

%%time

tweet_lens = [100, 120, 140, 160]

module_urls = ['https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1',

               'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2']



# tweet_lens = [50]

# module_urls = ['https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2']



for tl, mu in itertools.product(tweet_lens, module_urls):

    try:

        b = Blender(tl, mu)

        b.load_data()

        b.get_bert_layer()

        b.run()

    except Exception as e:

        print(e)

        
import os 

def blender(threshold=0.5):

    preds = []

    basedir = '.'

    for f in os.listdir(f'{basedir}/'):

        if not (f.startswith("predictions") and f.endswith(".csv")): continue

        if len(preds) == 0:

            preds = pd.read_csv(f"{basedir}/{f}").target 

        else:

            preds += pd.read_csv(f"{basedir}/{f}").target 

    

    preds = (preds >= max(preds) * threshold).astype(int)

    

    blended = pd.read_csv('../input/nlp-getting-started/sample_submission.csv')

    blended['target'] = preds

    os.system('rm submission.csv')

    blended.to_csv('submission.csv', index=False)

    print("Export done.")



    return preds



blender()
!head -n 20 submission.csv