import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Tensorflow packages
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub

# SKlearn packages
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle, class_weight

import warnings
warnings.filterwarnings('ignore')

# setting max width option
pd.set_option('display.max_colwidth', -1)


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

def dir_watch(dirname):
    for dirname, _, filenames in os.walk(dirname):
        for filename in filenames:
            print(os.path.join(dirname, filename))

# input dir
dir_watch('/kaggle/input/')
        
# Load the dataset from csv file
cfpb_data = pd.read_csv('/kaggle/input/us-consumer-finance-complaints/consumer_complaints.csv')
cfpb_data.isnull().sum()
non_na_complaints = np.where(~cfpb_data['consumer_complaint_narrative'].isna())
len(non_na_complaints[0])
cfpb_extract = cfpb_data.loc[non_na_complaints]

# Reset the index
cfpb_extract.reset_index(inplace=True)
cfpb_extract.info()
# Interested fields
key_cols = ['product', 'consumer_complaint_narrative']

cfpb_extract[key_cols][:3]
cfpb_extract['product'].value_counts()
# Plot the target variable
cfpb_extract['product'].value_counts().plot(kind='bar')
# Train and test data will be taken as 80/20 ratio
X_train_full, X_test_full = train_test_split(cfpb_extract[key_cols], test_size=0.2, random_state=111)

# Split the train data into further as 60/20 ratio
X_train, X_valid = train_test_split(X_train_full, test_size=0.2, random_state=111)
print(f"Shape of X_train: {X_train.shape}, X_valid: {X_valid.shape}" )
class_weights = list(class_weight.compute_class_weight('balanced',
                                                      np.unique(cfpb_extract['product']),
                                                      cfpb_extract['product']))


class_weights
# Converting list to dictionary object
weights = {}

for inx, weight in enumerate(class_weights):
    weights[inx] = weight
X_train['consumer_complaint_narrative'][:2]
train_tensor = tf.data.Dataset.from_tensor_slices((X_train['consumer_complaint_narrative'].values, X_train['product'].values))
test_tensor = tf.data.Dataset.from_tensor_slices((X_test_full['consumer_complaint_narrative'].values, X_test_full['product'].values))
valid_tensor = tf.data.Dataset.from_tensor_slices((X_valid['consumer_complaint_narrative'].values, X_valid['product'].values))
for corpus, target in train_tensor.take(5):
    print("\nTarget: {} \nData: {}".format(target, corpus))
products = np.unique(cfpb_extract['product'])

products

# Method to define target static hash
def target_encoding(unique_targets):
    
    key_tensor = tf.constant(unique_targets) # class names in text format
    value_tensor = tf.constant(np.arange(0, len(unique_targets))) # index values from 0 to length of the classes
    
    hash_table = tf.lookup.StaticHashTable(
                    tf.lookup.KeyValueTensorInitializer(
                        keys = key_tensor, 
                        values = value_tensor), -1
                )
    
    return hash_table

# Target encoded table
target_encoded = target_encoding(products)

# TF function will get build in the TensorFlow graph
@tf.function
def target_enc(t):
    return target_encoded.lookup(t)


def display_batchwise(dataset, bsize=5):
    for data, label in dataset.take(bsize):
        print("Data:{}\nTarget:{}\n".format(data.numpy(), label.numpy()))
        
def one_hot_labelencoding(text, label):
    return text, tf.one_hot(target_enc(label), 11)
next(iter(train_tensor))
# Transform the labels into binary variables
train_data_f = train_tensor.map(one_hot_labelencoding)
valid_data_f = valid_tensor.map(one_hot_labelencoding)
test_data_f = test_tensor.map(one_hot_labelencoding)
train_data, train_labels = next(iter(train_data_f.batch(5)))
train_data, train_labels
pretrained_url = 'https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1'

# Hub layer for embedding the text corpus
hub_layer = hub.KerasLayer(pretrained_url, output_shape=[128], 
                          input_shape=[], 
                          dtype=tf.string, 
                          trainable=True)

# Look at the hub layer
hub_layer(train_data[:1])
def build_model(embed_layer, output_shape):
    model = tf.keras.Sequential()
    
    model.add(embed_layer)
    
    for unit in [128, 128, 64, 32]:
        model.add(tf.keras.layers.Dense(unit, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.3))
    
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    
    return model

output_shape = len(products)

# NN model
model = build_model(hub_layer, output_shape)

model.summary()
# Train the model with train and validation set

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
             optimizer='adam',
             metrics=['accuracy'])
# Shuffle the train data
# shuffle_buffer_size = 50000
train_data_f = train_data_f.shuffle(60000).batch(512) 
valid_data_f = valid_data_f.shuffle(20000).batch(512)
test_data_f = test_data_f.batch(512)
# fit the data on the model
history = model.fit(train_data_f,
                    epochs=10,
                    validation_data=valid_data_f,
                    class_weight=weights,
                   verbose=1)
results = model.evaluate(test_data_f)
test_data, test_labels = next(iter(test_data_f))
y_preds = model.predict(test_data)
y_preds.argmax(axis=1)
from sklearn.metrics import classification_report

print(classification_report(test_labels.numpy().argmax(axis=1), y_preds.argmax(axis=1)))