# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import time

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#1.2 Keras libraries

from keras.layers import Input, Dense

from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten,Embedding, GRU

from keras.layers.merge import concatenate

from keras.models import Model

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
from keras.utils import  plot_model



# 1.4 sklearn libraries

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split



# 1.4 For model plotting

import matplotlib.pyplot as plt

import pydot

from skimage import io

%%time

MIMIC_PATH = "../input/mimic3d"

print (os.path.join(MIMIC_PATH,'mimic3d.csv'))

#mimic_df = pd.read_csv("/kaggle/input/mimic3d/mimic3d.csv")

mimic_df = pd.read_csv(os.path.join(MIMIC_PATH,'mimic3d.csv'))
mimic_df.shape
mimic_df.head()
mimic_df.describe()
def plot_features_distribution(features, title,isLog=False):

    plt.figure(figsize=(12,6))

    plt.title(title)

    for feature in features:

        if(isLog):

            sns.distplot(np.log1p(mimic_df[feature]),kde=True,hist=False, bins=120, label=feature)

        else:

            sns.distplot(mimic_df[feature],kde=True,hist=False, bins=120, label=feature)

    plt.xlabel('')

    plt.legend()

    plt.show()
def plot_count(feature, title,size=1,df=mimic_df):

    f, ax = plt.subplots(1,1, figsize=(4*size,4))

    total = float(len(df))

    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:30], palette='Set3')

    g.set_title("Number and percentage of {}".format(title))

    if(size > 2):

        plt.xticks(rotation=90, size=8)

    for p in ax.patches:

        height = p.get_height()

        ax.text(p.get_x()+p.get_width()/2.,

                height + 3,

                '{:1.2f}%'.format(100*height/total),

                ha="center") 

    plt.show()
plot_count('gender','gender')
plot_features_distribution(['age'],'Patient age distribution')
plot_count('admit_type','admission type',2)
plot_count('insurance','patient insurance',2)
mimic_df.head()
mimic_df.tail()
mimic_df.columns.values
#Any missing value?

mimic_df.isnull().values.sum() 
mimic_df.columns[mimic_df.isnull().sum()  > 0] 
mimic_df.AdmitDiagnosis = mimic_df.AdmitDiagnosis.fillna("missing")

mimic_df.religion = mimic_df.religion.fillna("missing")

mimic_df.marital_status = mimic_df.marital_status.fillna("missing")

#Want to make sure is there any missing values?

mimic_df.isnull().values.sum()  
dtrain,  dtest = train_test_split(mimic_df, test_size=0.25)
#Which columns are 'object'

obj_columns = mimic_df.select_dtypes(include = ['object']).columns.values

obj_columns
#Which columns have numeric data

num = mimic_df.select_dtypes(include = ['int64', 'float64']).columns.values

num
#Final seven obj_columns for One Hot Encoding

obj_cols = ["gender", "admit_type", "admit_location", "insurance" ,"marital_status", 'ExpiredHospital', 'LOSgroupNum']

ohe = OneHotEncoder()

# Traing on dtrain

ohe = ohe.fit(dtrain[obj_cols])

# Transform train (dtrain) and test (dtest) data

dtrain_ohe = ohe.transform(dtrain[obj_cols])

dtest_ohe = ohe.transform(dtest[obj_cols])



dtrain_ohe.shape       

dtest_ohe.shape 
#  Label encode relegion and ethnicity

#  First 'religion'

le = LabelEncoder()

le.fit(dtrain["religion"])

dtrain["re"] = le.transform(dtrain['religion'])    # Create new column in dtrain

dtest["re"] = le.transform(dtest['religion'])      #   and in dtest
# Now 'ethnicity'

le = LabelEncoder()

le.fit(dtrain["ethnicity"])

dtrain["eth"]= le.transform(dtrain['ethnicity'])   # Create new column in dtrain

dtest["eth"]= le.transform(dtest['ethnicity'])     #   and in dtest
# Finally transform two obj_columns for tokenization

te_ad = Tokenizer()

#Train tokenizer on train data ie 'dtrain'

te_ad.fit_on_texts(mimic_df.AdmitDiagnosis.values)

#Transform both dtrain and dtest and create new columns

dtrain["ad"] = te_ad.texts_to_sequences(dtrain.AdmitDiagnosis)

dtest["ad"] = te_ad.texts_to_sequences(dtest.AdmitDiagnosis)



dtrain.head(3)

dtest.head(3)
# Similarly for column: AdmitProcedure

te_ap = Tokenizer(oov_token='<unk>')

te_ap.fit_on_texts(mimic_df.AdmitProcedure.values)

dtrain["ap"] = te_ap.texts_to_sequences(dtrain.AdmitProcedure)

dtest["ap"] = te_ap.texts_to_sequences(dtest.AdmitProcedure)



dtrain.head(3)

dtest.head(3)
# Standardize numerical data

se = StandardScaler()

# Train om dtrain

se.fit(dtrain.loc[:,num])

# Then transform both dtrain and dtest

dtrain[num] = se.transform(dtrain[num])

dtest[num] = se.transform(dtest[num])

dtest.loc[:,num].head(3)
#  Get max length of the sequences

#    in dtrain["ad"], dtest["ad"]

maxlen_ad = 0

for i in dtrain["ad"]:

    if maxlen_ad < len(i):

        maxlen_ad = len(i)



for i in dtest["ad"]:

    if maxlen_ad < len(i):

        maxlen_ad = len(i)



maxlen_ad 
#  Get max length of the sequences

#    in dtrain["ap"], dtest["ap"]



maxlen_ap = 0

for i in dtrain["ap"]:

    if maxlen_ap < len(i):

        maxlen_ap = len(i)



maxlen_ap      



for i in dtest["ap"]:

    if maxlen_ap < len(i):

        maxlen_ap = len(i)



maxlen_ap  
#  Get max vocabulary size ie value of highest

#    integer in dtrain["ad"] and in dtest["ad"]



one = np.max([np.max(i) for i in dtrain["ad"].tolist() ])

two = np.max([np.max(i) for i in dtest["ad"].tolist() ])

MAX_VOCAB_AD = np.max([one,two])
# Get max vocabulary size ie value of highest

#     integer in dtrain["ap"] and in dtest["ap"]

one = np.max([np.max(i) for i in dtrain["ap"].tolist() ])

two = np.max([np.max(i) for i in dtest["ap"].tolist() ])

MAX_VOCAB_AP = np.max([one,two])
MAX_VOCAB_RE = len(dtrain.religion.value_counts())

MAX_VOCAB_ETH = len(dtrain.ethnicity.value_counts())
#  Let us put our data in a dictionary form

#     Required when we have multiple inputs

#     to Deep Neural network. Each Input layer

#     should also have the corresponding 'key'

#     name



#  Training data

Xtr = {

    "num" : dtrain[num].values,          # Note the name 'num'

    "ohe" : dtrain_ohe.toarray(),        # Note the name 'ohe'

    "re"  : dtrain["re"].values,

    "eth" : dtrain["eth"].values,

    "ad"  : pad_sequences(dtrain.ad, maxlen=maxlen_ad),

    "ap"  : pad_sequences(dtrain.ap, maxlen=maxlen_ap )

      }
# Test data

Xte = {

    "num" : dtest[num].values,

    "ohe" : dtest_ohe.toarray(),

    "re"  : dtest["re"].values,

    "eth" : dtest["eth"].values,

    "ad"  : pad_sequences(dtest.ad, maxlen=maxlen_ad ),

    "ap"  : pad_sequences(dtest.ap, maxlen=maxlen_ap )

      }
#Design a simple model now



dr_level = 0.1



num = Input(

                      shape= (Xtr["num"].shape[1], ),

                      name = "num"            # Name 'num' should be a key in the dictionary for numpy array input

                                              #    That is, this name should be the same as that of key in the dictionary

                      )





ohe =   Input(

                      shape= (Xtr["ohe"].shape[1], ),

                      name = "ohe"

                      )





re =   Input(

                      shape= [1],  # 1D shape or one feature

                      name = "re"

                      )



eth =   Input(

                      shape= [1],  # 1D shape or one feature

                      name = "eth"

                      )



ad =   Input(

                      shape= (Xtr["ad"].shape[1], ),

                      name = "ad"

                      )



ap =   Input(

                      shape= (Xtr["ap"].shape[1],),

                      name = "ap"

                      )
#  Embedding layers for each of the two of the columns with sequence data

#     Why add 1 to vocabulary?

#     See: https://stackoverflow.com/questions/52968865/invalidargumenterror-indices127-7-43-is-not-in-0-43-in-keras-r



emb_ad  =      Embedding(MAX_VOCAB_AD+ 1 ,      32  )(ad )

emb_ap  =      Embedding(MAX_VOCAB_AP+ 1 ,      32  )(ap)

# Embedding layers for the two categorical variables

emb_re  =      Embedding(MAX_VOCAB_RE+ 1 ,      32  )(re)

emb_eth =      Embedding(MAX_VOCAB_ETH+ 1 ,      32  )(eth)



#  RNN layers for sequences

rnn_ad = GRU(16) (emb_ad)          # Output of GRU is a vector of size 8

rnn_ap = GRU(16) (emb_ap)
rnn_re = GRU(16) (emb_re) 

rnn_eth = GRU(16) (emb_eth)
#Interim model summary.

#      For 'output' we have all the existing (unterminated) outputs

model = Model([num, ohe, re, eth, ad,ap], [rnn_ad, rnn_ap, emb_re, emb_eth, num, ohe])

model.summary()
#  Concatenate all outputs

class_l = concatenate([

                      rnn_ad,        # GRU output is already 1D

                      rnn_ap,

                      

                      num,                # 1D output. No need to flatten. See model summary

                      ohe,           # 1D output

                      Flatten()(emb_re),   # Why flatten? See model summary above

                      Flatten()(emb_eth)

                      ]

                     )





#  Add classification layer

class_l = Dense(64) (class_l)

class_l = Dropout(0.1)(class_l)

class_l = Dense(32) (class_l)

class_l = Dropout(0.1) (class_l)



#  Output neuron. Activation is linear

#      as our output is continous

output = Dense(1, activation="linear") (class_l)



#  Formulate Model now

model = Model(

              inputs= [num, ohe, re, eth, ad, ap],

              outputs= output

             )



model.summary()



#  Model plot uisng keras plot_model()

plt.figure(figsize = (14,14))

plot_model(model, to_file = "model.png")

io.imshow("model.png")
#Compile model

model.compile(loss="mse",

              optimizer="adam",

              metrics=["mae"]

              )



BATCH_SIZE = 5000

epochs = 20



######



start = time.time()

history= model.fit(Xtr,

                   dtrain.LOSdays,

                   epochs=epochs,

                   batch_size=BATCH_SIZE,

                   validation_data=(Xte, dtest.LOSdays),

                   verbose = 1

                  )

end = time.time()

print((end-start)/60)