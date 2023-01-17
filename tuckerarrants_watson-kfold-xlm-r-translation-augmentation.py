#python basics

from matplotlib import pyplot as plt

from tqdm import tqdm

import math, os, re, time, random, json, gc

import numpy as np, pandas as pd, seaborn as sns



#deep learning basics

import tensorflow as tf

import tensorflow.keras.backend as K

import tensorflow_addons as tfa



#nlp augmentation

!pip install --quiet googletrans

from googletrans import Translator



#easy way to shuffle rows

from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score



#ignore warnings

import warnings

warnings.filterwarnings('ignore')



#get current TensorFlow version fo

print("Currently using Tensorflow version " + tf.__version__)
SEED = 34



def seed_everything(seed):

    os.environ['PYTHONHASHSEED']=str(seed)

    tf.random.set_seed(seed)

    np.random.seed(seed)

    random.seed(seed)

    

seed_everything(SEED)
DEVICE = 'TPU'



if DEVICE == "TPU":

    print("connecting to TPU...")

    try:

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

    except ValueError:

        print("Could not connect to TPU")

        tpu = None



    if tpu:

        try:

            print("initializing  TPU ...")

            tf.config.experimental_connect_to_cluster(tpu)

            tf.tpu.experimental.initialize_tpu_system(tpu)

            strategy = tf.distribute.experimental.TPUStrategy(tpu)

            print("TPU initialized")

        except _:

            print("failed to initialize TPU")

    else:

        DEVICE = "GPU"



if DEVICE != "TPU":

    print("Using default strategy for CPU and single GPU")

    strategy = tf.distribute.get_strategy()



if DEVICE == "GPU":

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    



AUTO = tf.data.experimental.AUTOTUNE

REPLICAS = strategy.num_replicas_in_sync

print(f'REPLICAS: {REPLICAS}')



#choose batch size - will depend on cores of our device

BATCH_SIZE = 16 * REPLICAS
#get CSV files

train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")

test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")



print(f'Train shape: {train.shape}')

train.head()
print(f'Test shape: {test.shape}')

test.head()
#peek at a premise/hypothesis pair and their label

print(f"Premise: {train['premise'].values[0]}")

print(f"Hypothesis: {train['hypothesis'].values[0]}")

print(f"Label: {train['label'].values[0]}")
#peek at a premise/hypothesis pair and their label

print(f"Premise: {train['premise'].values[1]}")

print(f"Hypothesis: {train['hypothesis'].values[1]}")

print(f"Label: {train['label'].values[1]}")
#explore the distribution of classes and languages

fig, ax = plt.subplots(figsize = (15, 10))



#for maximum aesthetics

palette = sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)



graph1 = sns.countplot(train['language'], hue = train['label'], palette = palette)



#set title

graph1.set_title('Distribution of Languages and Labels')



plt.tight_layout()

plt.show()
def back_translate(sequence, PROB = 1):

    languages = ['en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',

                 'sw', 'vi', 'es', 'el']

    

    #instantiate translator

    translator = Translator()

    

    #store original language so we can convert back

    org_lang = translator.detect(sequence).lang

    

    #randomly choose language to translate sequence to  

    random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])

    

    if org_lang in languages:

        #translate to new language and back to original

        translated = translator.translate(sequence, dest = random_lang).text

        #translate back to original language

        translated_back = translator.translate(translated, dest = org_lang).text

    

        #apply with certain probability

        if np.random.uniform(0, 1) <= PROB:

            output_sequence = translated_back

        else:

            output_sequence = sequence

            

    #if detected language not in our list of languages, do nothing

    else:

        output_sequence = sequence

    

    return output_sequence



#check performance

for i in range(5):

    output = back_translate('I genuinely have no idea what the output of this sequence of words will be')

    print(output)
train.head()
#offline loading of augmented datasets

train_aug = pd.read_csv('../input/contradictorywatsontwicetranslatedaug/translation_aug_train.csv')

train_aug.head()
#offline loading of augmented datasets

train_twice_aug = pd.read_csv('../input/contradictorywatsontwicetranslatedaug/twice_translated_aug_train.csv')

train_twice_aug.head()
#offline loading of augmented datasets

train_thrice_aug = pd.read_csv('../input/contradictorywatsontwicetranslatedaug/thrice_translation_aug_train.csv')

train_thrice_aug.head()
#get CSV files

train_vi = pd.read_csv("../input/contradictorytranslatedtrain/train_vi.csv")

train_hi = pd.read_csv("../input/contradictorytranslatedtrain/train_hi.csv")

train_bg = pd.read_csv("../input/contradictorytranslatedtrain/train_bg.csv")
#sanity check

train_vi.head()
#sanity check

train_hi.head()
#sanity check

train_bg.head()
#get HuggingFace transformers

!pip install --quiet transformers



#import model and Tokenizer

from transformers import TFAutoModel, AutoTokenizer



#get paths to TensorFlow XLM-RoBERTa base and large models

roberta_base = "jplu/tf-xlm-roberta-base"

roberta_large = 'jplu/tf-xlm-roberta-large'
#offline load back-translated test samples

test_bt = pd.read_csv('../input/contradictorywatsontwicetranslatedaug/translation_aug_test.csv')

test_bt_twice = pd.read_csv('../input/contradictorywatsontwicetranslatedaug/twice_translated_aug_test.csv')

test_bt_thrice = pd.read_csv('../input/contradictorywatsontwicetranslatedaug/thrice_translation_aug_test.csv')
TOKENIZER = AutoTokenizer.from_pretrained(roberta_large)



#function to encode text and convert dataset to tensor dataset

def to_tf_dataset(dataset, max_len, repeat = False, shuffle = False, labeled = True, batch_size = BATCH_SIZE):

    dataset_text = dataset[['premise', 'hypothesis']].values.tolist()

    dataset_enc = TOKENIZER.batch_encode_plus(dataset_text, pad_to_max_length = True, max_length = max_len)

    

    if labeled:

        tf_dataset = tf.data.Dataset.from_tensor_slices((dataset_enc['input_ids'], dataset['label']))

    else:

        tf_dataset = tf.data.Dataset.from_tensor_slices((dataset_enc['input_ids']))

    

    if repeat: tf_dataset = tf_dataset.repeat()  

        

    if shuffle: 

        tf_dataset = tf_dataset.shuffle(2048)

        opt = tf.data.Options()

        opt.experimental_deterministic = False

        tf_dataset = tf_dataset.with_options(opt)

        

    tf_dataset = tf_dataset.batch(batch_size)

    tf_dataset = tf_dataset.prefetch(AUTO)

    

    return tf_dataset
###########################################

#### Configuration

###########################################

LR_START = 1e-6

LR_MAX = 1e-6 * 8

LR_MIN = 1e-6

LR_RAMPUP_EPOCHS = 2

LR_SUSTAIN_EPOCHS = 0

LR_DECAY = .8



#stepwise schedule

def lrfn_step(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = LR_MAX * LR_DECAY**((epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS)//2)

    return lr





#smoothish schedule

def lrfn_smooth(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback_step = tf.keras.callbacks.LearningRateScheduler(lrfn_step, verbose = True)

lr_callback_smooth = tf.keras.callbacks.LearningRateScheduler(lrfn_smooth, verbose = True)



#visualize learning rate schedule

rng = [i for i in range(25)]

y1 = [lrfn_step(x) for x in rng]

y2 = [lrfn_smooth(x) for x in rng]

fix, ax = plt.subplots(1,2, figsize = (15, 5))

ax[0].plot(rng, y1)

ax[1].plot(rng, y2)

plt.tight_layout()

print("Learning rate schedule for step schedule: {:.3g} to {:.3g} to {:.3g}".format(y1[0], max(y1), y1[-1]))

print("Learning rate schedule for smooth schedule: {:.3g} to {:.3g} to {:.3g}".format(y2[0], max(y2), y2[-1]))
#helper function to create our model

def build_model(transformer_layer, max_len, learning_rate):

    #must use this to send to TPU cores

    with strategy.scope():

        #define input(s)

        input_ids = tf.keras.Input(shape = (max_len,), dtype = tf.int32)

        

        #insert roberta layer

        roberta = TFAutoModel.from_pretrained(transformer_layer)

        roberta = roberta(input_ids)[0]

        

        #only need <s> token here, so we extract it now

        out = roberta[:, 0, :]

        

        out = tf.keras.layers.BatchNormalization()(out)

        

        #add our softmax layer

        out = tf.keras.layers.Dense(3, activation = 'softmax')(out)

        

        #assemble model and compile

        model = tf.keras.Model(inputs = input_ids, outputs = out)

        model.compile(

                        optimizer = tf.keras.optimizers.Adam(lr = learning_rate), 

                        loss = 'sparse_categorical_crossentropy', 

                        metrics = ['accuracy'])

        

    return model  
from sklearn.model_selection import KFold, StratifiedKFold



###########################################

#### Configuration

###########################################



LR_RATE = 5e-6

EPOCHS = 15

FOLDS = 4

MAX_LEN = 85

STEPS_PER_EPOCH = len(train) // BATCH_SIZE

TTA = 4

VERBOSE = 2



############################################

#### Training

############################################



preds = np.zeros((len(test), 3))

preds_tta = np.zeros((len(test), 3))

skf = StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED)



for fold,(train_index,val_index) in enumerate(skf.split(train, train['language'])):



    #to clear TPU memory

    if DEVICE=='TPU':

        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)

    

    #build model

    K.clear_session()

    model = build_model(roberta_large, max_len = MAX_LEN, learning_rate = LR_RATE)

        

    #save best model from each fold

    sv = tf.keras.callbacks.ModelCheckpoint(f'fold-{fold}.h5', monitor = 'val_loss', verbose = 0,

                        save_best_only = True, save_weights_only = True, mode = 'min')

   

    #get our datasets

    train_ds = to_tf_dataset(train.loc[train_index], labeled = True, shuffle = True, repeat = True, max_len = MAX_LEN)

    val_ds = to_tf_dataset(train.loc[val_index], labeled = True, shuffle = False, repeat = False, max_len = MAX_LEN)





    #and go

    print('')

    print('#'*25); print('#### FOLD',fold+1)

    print('Training...'); print('')

    history = model.fit(train_ds, validation_data = val_ds, callbacks = [sv],

                        epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH,

                        verbose = VERBOSE); print('')



    

    print('Loading best model...')

    model.load_weights(f'fold-{fold}.h5')

    

############################################

#### Validation

############################################

    

    #predict validation with TTA

    print('Predicting validation with TTA...')

    

    #offline load pre-back-translated datasets

    val_df = train.loc[val_index]

    val_df_bt = train_aug.loc[val_index]

    val_df_bt_twice = train_twice_aug.loc[val_index]

    val_df_bt_thrice = train_thrice_aug.loc[val_index]

    

    #convert to tensor dataset

    val_tta1 = to_tf_dataset(val_df, shuffle = False, labeled = False, repeat = False, max_len = MAX_LEN)

    val_tta2 = to_tf_dataset(val_df_bt, shuffle = False, labeled = False, repeat = False, max_len = MAX_LEN)

    val_tta3 = to_tf_dataset(val_df_bt_twice, shuffle = False, labeled = False, repeat = False, max_len = MAX_LEN)

    val_tta4 = to_tf_dataset(val_df_bt_thrice, shuffle = False, labeled = False, repeat = False, max_len = MAX_LEN)

    

    #predict with augmentated validation sets

    val_pred1 = model.predict(val_tta1, verbose = VERBOSE)

    val_pred2 = model.predict(val_tta2, verbose = VERBOSE) 

    val_pred3 = model.predict(val_tta3, verbose = VERBOSE) 

    val_pred4 = model.predict(val_tta4, verbose = VERBOSE) 

        

    val_preds = (val_pred1 + val_pred2 + val_pred3 + val_pred4) / TTA

     

    print(f"Without TTA: {accuracy_score(val_pred1.argmax(axis = 1), val_df['label'])}")

    print(f"With TTA: {accuracy_score(val_preds.argmax(axis = 1), val_df['label'])}")

    print('')

    

############################################

#### Prediction

############################################



    #predict out of fold with TTA

    print('Predicting OOF with TTA...')

    

    #convert test to tensor dataset

    test_tta1 = to_tf_dataset(test, shuffle = False, labeled = False, repeat = False, max_len = MAX_LEN)

    test_tta2 = to_tf_dataset(test_bt, shuffle = False, labeled = False, repeat = False, max_len = MAX_LEN)

    test_tta3 = to_tf_dataset(test_bt_twice, shuffle = False, labeled = False, repeat = False, max_len = MAX_LEN)

    test_tta4 = to_tf_dataset(test_bt_thrice, shuffle = False, labeled = False, repeat = False, max_len = MAX_LEN)

    

    #predict with augmentated validation sets

    pred1 = model.predict(test_tta1, verbose = VERBOSE)

    pred2 = model.predict(test_tta2, verbose = VERBOSE) 

    pred3 = model.predict(test_tta3, verbose = VERBOSE)

    pred4 = model.predict(test_tta4, verbose = VERBOSE) 

        

    preds_tta += (pred1 + pred2 + pred3 + pred4) / TTA / FOLDS

    preds += pred1 / FOLDS



    #so we don't hit memory limits

    os.remove(f"/kaggle/working/fold-{fold}.h5")

    del model ; z = gc.collect()
USE_TTA = False
if USE_TTA:

    submission = pd.DataFrame()

    submission['id'] = test['id']

    submission['prediction'] = preds_tta.argmax(axis = 1)



else:

    submission = pd.DataFrame()

    submission['id'] = test['id']

    submission['prediction'] = preds.argmax(axis = 1)

    

#sanity check 

submission.head()
submission.to_csv('submission.csv', index = False)

print('Submission saved')