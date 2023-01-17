from IPython.display import YouTubeVideo

YouTubeVideo("JC84GCU7zqA")
from IPython.display import YouTubeVideo

YouTubeVideo("kBjYK3K3P6M")
import os

import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

import transformers

from transformers import TFAutoModel, AutoTokenizer,BertTokenizer,TFBertModel

from tqdm.notebook import tqdm

import plotly.express as px

from collections import Counter

import re

import string

import nltk

from nltk.corpus import stopwords

stop = stopwords.words('english')

from sklearn import model_selection
AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    print('Running on TPU ', tpu.master())

except ValueError:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
# model_name = 'jplu/tf-xlm-roberta-large'

model_name = 'bert-base-multilingual-cased'

n_epochs = 25

max_len = 100



# Our batch size will depend on number of replicas

batch_size = 16 * strategy.num_replicas_in_sync

print(batch_size)
train = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")

test = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/test.csv")

submission = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/sample_submission.csv")
print('Premise:', train.premise[0])

print('hypothesis:', train.hypothesis[0])

display(train.isnull().sum(axis = 0))

display(train.head())
def code_labels(label):

    res = 'Entailment'

    if label == 1:

        res = 'Neutral'

    elif label == 2:

        res = 'Contradiction'

    return (res)   
train["Encoded"] = train["label"].apply(code_labels)
encode = train["Encoded"].value_counts()

encode_df = pd.DataFrame({"Encode":encode.index,"frequency":encode.values})

fig = px.bar(data_frame=encode_df,x="Encode",y="frequency",color="Encode",text="frequency",title="Target Column Distribution",labels={"Encode":"Type of Relationship","frequency":"Counts"})

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()

language = train["language"].value_counts()

language_df = pd.DataFrame({"Languages":language.index,"frequency":language.values})

language_df["count_percent"] = language_df['frequency'].apply(lambda x: round(x*100/language_df.frequency.sum(),2))

fig = px.bar(data_frame=language_df,x="Languages",y="frequency",color="Languages",title="Different Language Distribution",text="frequency")

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()
test_language = test["language"].value_counts()

test_language_df = pd.DataFrame({"Languages":test_language.index,"frequency":test_language.values})

fig = px.bar(data_frame=test_language_df,x="Languages",y="frequency",color="Languages",title="Different Language Distribution",text="frequency")

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()
english_text = train[train["language"]=="English"]
english_text.head()
def clean_text(text):

    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation

    and remove words containing numbers.'''

    text = str(text).lower()

    text = re.sub('\[.*?\]', '', text)

    text = re.sub('https?://\S+|www\.\S+', '', text)

    text = re.sub('<.*?>+', '', text)

    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    text = re.sub('\n', '', text)

    text = re.sub('\w*\d\w*', '', text)

    return text
def remove_stopword(x):

    return [w for w in x if not w in stop]

english_text['temp_list'] = english_text['premise'].apply(lambda x:str(x).split())

top = Counter([item for sublist in english_text['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Blues')
fig = px.bar(temp, x="count", y="Common_words", title='Commmon Words in Selected Text', orientation='h', width=700, height=700,color='Common_words')

fig.show()
english_text['temp_list'] = english_text['temp_list'].apply(lambda x:remove_stopword(x))
top = Counter([item for sublist in english_text['temp_list'] for item in sublist])

temp = pd.DataFrame(top.most_common(20))

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Purples')

fig = px.treemap(temp, path=['Common_words'], values='count',title='Tree of Most Common Words')

fig.show()

# train["kfold"] = -1

# train = train.sample(frac=1).reset_index(drop=True)



# y = train.label.values



# kf =model_selection.StratifiedKFold(n_splits=5)



# for f,(t_,v_) in enumerate(kf.split(X=train,y=y)):

#     train.loc[v_,'kfold'] = f



# tokenizer = AutoTokenizer.from_pretrained(model_name)
model_name = 'bert-base-multilingual-cased'

max_len = 80

tokenizer = BertTokenizer.from_pretrained(model_name)
train.head()
# Convert the text so that we can feed it to `batch_encode_plus`

train_text = train[['premise', 'hypothesis']].values.tolist()

test_text = test[['premise', 'hypothesis']].values.tolist()



# Now, we use the tokenizer we loaded to encode the text

# train_encoded = tokenizer.batch_encode_plus(

#     train_text,

#     pad_to_max_length=True,

#     max_length=max_len

# )



# test_encoded = tokenizer.batch_encode_plus(

#     test_text,

#     pad_to_max_length=True,

#     max_length=max_len

# )
def quick_encode(values,maxlen):

    tokens=tokenizer.batch_encode_plus(values,max_length=maxlen,pad_to_max_length=True)

    return np.array(tokens['input_ids'])



x_train = quick_encode(train_text,maxlen=max_len)

x_test = quick_encode(test_text,maxlen=max_len)

y_train = train.label.values

    
# x_train = train_encoded["input_ids"]

# y_train = train.label.values

# x_test = test_encoded['input_ids']
# x_train, x_valid, y_train, y_valid = train_test_split(

#     train_encoded['input_ids'], train.label.values, 

#     test_size=0.2, random_state=2020

# )



# train_dataset = (

#     tf.data.Dataset

#     .from_tensor_slices((x_train, y_train))

#     .repeat()

#     .shuffle(2048)

#     .batch(batch_size)

#     .prefetch(AUTO)

# )



# valid_dataset = (

#     tf.data.Dataset

#     .from_tensor_slices((x_valid, y_valid))

#     .batch(batch_size)

#     .cache()

#     .prefetch(AUTO)

# )







def create_dist_dataset(X, y,val,batch_size=batch_size):

    

    

    dataset = tf.data.Dataset.from_tensor_slices((X,y)).shuffle(len(X))

          

    if not val:

        dataset = dataset.repeat().batch(batch_size).prefetch(AUTO)

    else:

        dataset = dataset.batch(batch_size).prefetch(AUTO)



    

    

    return dataset









test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(batch_size)

)
def build_lrfn(lr_start=0.00001, lr_max=0.00003, 

               lr_min=0.000001, lr_rampup_epochs=3, 

               lr_sustain_epochs=0, lr_exp_decay=.6):

    lr_max = lr_max * strategy.num_replicas_in_sync



    def lrfn(epoch):

        if epoch < lr_rampup_epochs:

            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start

        elif epoch < lr_rampup_epochs + lr_sustain_epochs:

            lr = lr_max

        else:

            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min

        return lr

    

    return lrfn

lrfn = build_lrfn()

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

def build_model(model_name):

    # First load the transformer layer

    transformer_encoder = TFBertModel.from_pretrained(model_name)



    # This will be the input tokens 

    input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")



    # Now, we encode the text using the transformers we just loaded

    sequence_output = transformer_encoder(input_ids)[0]



    # Only extract the token used for classification, which is <s>

    cls_token = sequence_output[:, 0, :]



    # Finally, pass it through a 3-way softmax, since there's 3 possible laels

    out = Dense(3, activation='softmax')(cls_token)



    # It's time to build and compile the model

    model = Model(inputs=input_ids, outputs=out)

    model.compile(

        Adam(lr=1e-5), 

        loss='sparse_categorical_crossentropy', 

        metrics=['accuracy']

    )

    return model
pred_test=np.zeros((test.shape[0],3))

skf = model_selection.StratifiedKFold(n_splits=5,shuffle=True,random_state=777)

val_score=[]

history=[]





for fold,(train_ind,valid_ind) in enumerate(skf.split(x_train,y_train)):

    

    if fold < 4:

    

        print("fold",fold+1)

        

       

        tf.tpu.experimental.initialize_tpu_system(tpu)

        

        train_data = create_dist_dataset(x_train[train_ind],y_train[train_ind],val=False)

        valid_data = create_dist_dataset(x_train[valid_ind],y_train[valid_ind],val=True)

    

        Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"bert-base-multilingual-cased.h5", monitor='val_loss', verbose=0, save_best_only=True,

        save_weights_only=True, mode='min')

        

        with strategy.scope():

#             transformer_layer = TFAutoModel.from_pretrained(MODEL)

            model = build_model(model_name=model_name)

            

        



        n_steps = len(train_ind)//batch_size

        print("training model {} ".format(fold+1))



        train_history = model.fit(

        train_data,

        steps_per_epoch=n_steps,

        validation_data=valid_data,

        epochs=n_epochs,callbacks=[lr_schedule,Checkpoint],verbose=1)

        

        print("Loading model...")

        model.load_weights(f"bert-base-multilingual-cased.h5")

        

        



        print("fold {} validation accuracy {}".format(fold+1,np.mean(train_history.history['val_accuracy'])))

        print("fold {} validation loss {}".format(fold+1,np.mean(train_history.history['val_loss'])))

        

        val_score.append(train_history.history['val_accuracy'])

        history.append(train_history)



        val_score.append(np.mean(train_history.history['val_accuracy']))

        

        print('predict on test....')

        preds=model.predict(test_dataset,verbose=1)



        pred_test+=preds/4

        



        

print("Mean Validation accuracy : ",np.mean(val_score))
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))



for i,hist in enumerate(history):



    plt.subplot(2,2,i+1)

    plt.plot(np.arange(n_epochs),hist.history['accuracy'],label='train accu')

    plt.plot(np.arange(n_epochs),hist.history['val_accuracy'],label='validation acc')

    plt.gca().title.set_text(f'Fold {i+1} accuracy curve')

    plt.legend()

plt.figure(figsize=(15,10))



for i,hist in enumerate(history):



    plt.subplot(2,2,i+1)

    plt.plot(np.arange(n_epochs),hist.history['loss'],label='train loss')

    plt.plot(np.arange(n_epochs),hist.history['val_loss'],label='validation loss')

    plt.gca().title.set_text(f'Fold {i+1} loss curve')

    plt.legend()
submission['prediction'] = np.argmax(pred_test,axis=1)

submission.head()

submission.to_csv("submission.csv",index=False)


# train_history = model.fit(

#     train_dataset,

#     steps_per_epoch=n_steps,

#     validation_data=valid_dataset,

#     epochs=n_epochs

# )

# test_preds = model.predict(test_dataset, verbose=1)

# submission['prediction'] = test_preds.argmax(axis=1)
# submission.to_csv('submission.csv', index=False)

# submission.head()
# hist = train_history.history

# px.line(

#     hist, x=range(1, len(hist['loss'])+1), y=['accuracy', 'val_accuracy'], 

#     title='Model Accuracy', labels={'x': 'Epoch', 'value': 'Accuracy'}

# )





# px.line(

#     hist, x=range(1, len(hist['loss'])+1), y=['loss', 'val_loss'], 

#     title='Model Loss', labels={'x': 'Epoch', 'value': 'Loss'}

# )