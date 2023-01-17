from IPython.display import YouTubeVideo
YouTubeVideo('zEOtG-ChmZE', width=800, height=400)
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import transformers 
from transformers import TFAutoModel, AutoTokenizer
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
train_data_frame=pd.read_csv("../input/contradictory-my-dear-watson/train.csv")
test_data_frame =pd.read_csv("../input/contradictory-my-dear-watson/test.csv")
sample_sub=pd.read_csv("../input/contradictory-my-dear-watson/sample_submission.csv")
def Utilize_TPUs():  
    """
    Initialize training strategy using TPU if available else using default strategy for CPU and  single GPU
    
    After the TPU is initialized, you can also use manual device placement to place the computation on a single TPU device.

    """
    try:
        
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.experimental.TPUStrategy(resolver)
        REPLICAS = strategy.num_replicas_in_sync
        print("Connected to TPU Successfully:\n TPUs Initialised with Replicas:",REPLICAS)
        
        return strategy
    
    except ValueError:
        
        print("Connection to TPU Falied")
        print("Using default strategy for CPU and single GPU")
        strategy = tf.distribute.get_strategy()
        
        return strategy
    
strategy=Utilize_TPUs()
the_chosen_one="jplu/tf-xlm-roberta-base"
max_len=80
batch_size = 16 * strategy.num_replicas_in_sync
AUTO     = tf.data.experimental.AUTOTUNE
epochs= 20
n_steps = len(train_data_frame) // batch_size
def model_baseline(strategy,transformer):
    with strategy.scope():
        transformer_encoder = TFAutoModel.from_pretrained(transformer)
        input_layer = Input(shape=(max_len,), dtype=tf.int32, name="input_layer")
        sequence_output = transformer_encoder(input_layer)[0]
        cls_token = sequence_output[:, 0, :]
        output_layer = Dense(3, activation='softmax')(cls_token)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            Adam(lr=1e-5), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model
model=model_baseline(strategy,the_chosen_one)
model.summary()
train_data_frame.head()
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import seaborn as sns
import plotly.express as px
fig = px.bar(train_data_frame, x=train_data_frame['language'])
iplot(fig)
sns.countplot(train_data_frame.label)
tokenizer = AutoTokenizer.from_pretrained(the_chosen_one)

train_data = train_data_frame[['premise', 'hypothesis']].values.tolist()

test_data = test_data_frame[['premise', 'hypothesis']].values.tolist()

train_encoded=tokenizer.batch_encode_plus(train_data,pad_to_max_length=True,max_length=max_len)
test_encoded=tokenizer.batch_encode_plus(test_data,pad_to_max_length=True,max_length=max_len)
x_train, x_valid, y_train, y_valid = train_test_split(train_encoded['input_ids'], train_data_frame.label.values, test_size=0.1)

x_test = test_encoded['input_ids']
train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train)).repeat().shuffle(2048).batch(batch_size).prefetch(AUTO))

valid_dataset = (tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size).cache().prefetch(AUTO))

test_dataset = (tf.data.Dataset.from_tensor_slices(x_test).batch(batch_size))
model.fit(train_dataset,steps_per_epoch=n_steps,validation_data=valid_dataset,epochs=epochs)
predictions = model.predict(test_dataset, verbose=1)
sample_sub['prediction'] = predictions.argmax(axis=1)
sample_sub.to_csv("submission.csv",index= False)