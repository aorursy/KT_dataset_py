# Importing Data manipulation and plotting modules



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pydot

from skimage import io
# Import Keras Libraries

from keras.layers import Input, Dense

from keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten,Embedding, GRU

from keras.layers.merge import concatenate

from keras.models import Model

from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences
# Import library for keras plotting model

from keras.utils import  plot_model
# Import sklearn libraries

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
# Importing Miscelaneous libraries

import os

import time
pathToData = '../input/'

os.chdir(pathToData)

# 2.2

data = pd.read_csv('../input/mimic3d/mimic3d.csv',

	               compression='infer',

                   encoding="ISO-8859-1"      # 'utf-8' gives error, hence the choice

                  )





# 2.3 Inspect the data

data.shape            # (58976, 28)
# Drop first column: 'hadm_id', being id column

data.drop(['hadm_id'], axis = 'columns' , inplace = True)





# Check for missing values

data.isnull().values.sum()        # 10611



# Check which columns have missing values

data.columns[data.isnull().sum()  > 0]    # Three: Index(['AdmitDiagnosis', 'religion', 'marital_status'], dtype='object')



# Let us follow a conservative and safe approach to fill missing values

data.AdmitDiagnosis = data.AdmitDiagnosis.fillna("missing")

data.religion = data.religion.fillna("missing")

data.marital_status = data.marital_status.fillna("missing")

data.isnull().values.sum()  
plt.figure(figsize=(15,10))

sns.distplot(data['age'],hist=False)

plt.title('Patient Age Distribution',fontsize=16)
plt.figure(figsize=(15,10))

sns.distplot(data['LOSdays'],hist=False)

plt.title('Patient Length Of stay Distribution',fontsize=16)
# Divide data into train/test

dtrain,  dtest = train_test_split(data, test_size=0.33)





# Check which columns are 'object'

obj_columns = data.select_dtypes(include = ['object']).columns.values

obj_columns

# Check which columns have numeric data

num = data.select_dtypes(include = ['int64', 'float64']).columns.values

num



"""

array(['age', 'LOSdays', 'NumCallouts', 'NumDiagnosis', 'NumProcs',

       'NumCPTevents', 'NumInput', 'NumLabs', 'NumMicroLabs', 'NumNotes',

       'NumOutput', 'NumRx', 'NumProcEvents', 'NumTransfers',

       'NumChartEvents', 'ExpiredHospital', 'TotalNumInteract',

       'LOSgroupNum'], dtype=object)



IInd column is target

And columns 'ExpiredHospital', 'LOSgroupNum' are categorical. See below



"""



# Levels in columns: 'ExpiredHospital', 'LOSgroupNum'

data.LOSgroupNum.value_counts()          # 4 levels

data.ExpiredHospital.value_counts()      # 2 levels
for i in obj_columns:

    print(i,len(data[i].value_counts()))

    if(len(data[i].value_counts())<25):

        data.groupby(data[i]).size().plot.bar()

        plt.show()
plt.figure(figsize=(12,10))

sns.barplot(x=data.admit_type, y=data.age,data=data, hue=data.gender,palette='spring')

plt.xlabel('Age',fontsize=16)

plt.ylabel('Admit Type',fontsize=16)
# 4.4 Final seven obj_columns for One Hot Encoding

obj_cols = ["gender", "admit_type", "admit_location", "insurance" ,"marital_status", 'ExpiredHospital', 'LOSgroupNum']

ohe = OneHotEncoder()

# 4.4.1 Traing on dtrain

ohe = ohe.fit(dtrain[obj_cols])

# 4.4.2 Transform train (dtrain) and test (dtest) data

dtrain_ohe = ohe.transform(dtrain[obj_cols])

dtest_ohe = ohe.transform(dtest[obj_cols])

# 4.4.3

dtrain_ohe.shape       # (39513, 34)

dtest_ohe.shape        # (19463, 34)


# 5.0 Label encode relegion and ethnicity

# 5.1 First 'religion'



le = LabelEncoder()

le.fit(data["religion"])                           # Train on full data else some labels may be absent in test data

dtrain["re"] = le.transform(dtrain['religion'])    # Create new column in dtrain

dtest["re"] = le.transform(dtest['religion'])      #   and in dtest



# 5.2 Now 'ethnicity'

le = LabelEncoder()

le.fit(data["ethnicity"])                          # train on full data

dtrain["eth"]= le.transform(dtrain['ethnicity'])   # Create new column in dtrain

dtest["eth"]= le.transform(dtest['ethnicity'])     #   and in dtest





# 6. Finally transform two obj_columns for tokenization

te_ad = Tokenizer()

# 6.1 Train tokenizer on train data ie 'dtrain'

te_ad.fit_on_texts(data.AdmitDiagnosis.values)

# 6.2 Transform both dtrain and dtest and create new columns

dtrain["ad"] = te_ad.texts_to_sequences(dtrain.AdmitDiagnosis)

dtest["ad"] = te_ad.texts_to_sequences(dtest.AdmitDiagnosis)



dtrain.shape

dtest.shape



# 6.3 Similarly for column: AdmitProcedure

te_ap = Tokenizer(oov_token='<unk>')

te_ap.fit_on_texts(data.AdmitProcedure.values)

dtrain["ap"] = te_ap.texts_to_sequences(dtrain.AdmitProcedure)

dtest["ap"] = te_ap.texts_to_sequences(dtest.AdmitProcedure)



dtrain.shape

dtest.shape



# dtrain["ad"], dtest["ad"]



maxlen_ad = 0

for i in dtrain["ad"]:

	if maxlen_ad < len(i):

		maxlen_ad = len(i)



for i in dtest["ad"]:

	if maxlen_ad < len(i):

		maxlen_ad = len(i)



maxlen_ad



# dtrain["ap"], dtest["ap"]



maxlen_ap = 0

for i in dtrain["ap"]:

	if maxlen_ap < len(i):

		maxlen_ap = len(i)



for i in dtest["ap"]:

	if maxlen_ap < len(i):

		maxlen_ap = len(i)



maxlen_ap

# in dtrain["ad"] and in dtest["ad"]



one = np.max([np.max(i) for i in dtrain["ad"].tolist() ])

two = np.max([np.max(i) for i in dtest["ad"].tolist() ])

MAX_VOCAB_AD = np.max([one,two])



# in dtrain["ap"] and in dtest["ap"]



one = np.max([np.max(i) for i in dtrain["ap"].tolist() ])

two = np.max([np.max(i) for i in dtest["ap"].tolist() ])

MAX_VOCAB_AP = np.max([one,two])



# 

MAX_VOCAB_RE = len(dtrain.religion.value_counts())

MAX_VOCAB_ETH = len(dtrain.ethnicity.value_counts())

num = ['age', 'NumCallouts', 'NumDiagnosis', 'NumProcs',

       'NumCPTevents', 'NumInput', 'NumLabs', 'NumMicroLabs', 'NumNotes',

       'NumOutput', 'NumRx', 'NumProcEvents', 'NumTransfers',

       'NumChartEvents', 'TotalNumInteract']
# Standardize numerical data

se = StandardScaler()

# Train on dtrain

se.fit(dtrain.loc[:,num])

# Then transform both dtrain and dtest

dtrain[num] = se.transform(dtrain[num])

dtest[num] = se.transform(dtest[num])

dtest.loc[:,num].head(3)
# Reshape train num data

dtrain[num].values.shape

dtr_reshape = dtrain[num].values.reshape(39513,15,1)
# Reshape test num data

dtest[num].values.shape

dts_reshape = dtest[num].values.reshape(19463,15,1)
# Training data

Xtr = {

	"num" : dtr_reshape,          # Note the name 'num'

	"ohe" : dtrain_ohe.toarray(),        # Note the name 'ohe'

	"re"  : dtrain["re"].values,

	"eth" : dtrain["eth"].values,

	"ad"  : pad_sequences(dtrain.ad, maxlen=maxlen_ad),

	"ap"  : pad_sequences(dtrain.ap, maxlen=maxlen_ap )

      }
# Test data

Xte = {

	"num" : dts_reshape,

	"ohe" : dtest_ohe.toarray(),

	"re"  : dtest["re"].values,

	"eth" : dtest["eth"].values,

	"ad"  : pad_sequences(dtest.ad, maxlen=maxlen_ad ),

	"ap"  : pad_sequences(dtest.ap, maxlen=maxlen_ap )

      }
# Just check shapes.

# Total data features are now: 15 + 34 + 24 + 7 + 1 +1 = 82

# Embeddings have thus generated new features.

Xtr["num"].shape         # (39513, 15)

Xtr["ohe"].shape         # (39513, 34)

Xtr["ad"].shape          # (39513, 24)

Xtr["ap"].shape          # (39513, 7)

Xtr["re"].shape          # (39513,)  1D

Xtr["eth"].shape         # (39513,)  1D
# Design a simple model now



dr_level = 0.1



# 11.1

num = Input(

                      shape= (Xtr["num"].shape[1], 1 ),

					  name = "num"            # Name 'num' should be a key in the dictionary for numpy array input

					                          #    That is, this name should be the same as that of key in the dictionary

					  )



# 11.2

ohe =   Input(

                      shape= (Xtr["ohe"].shape[1], ),

					  name = "ohe"

					  )



# 11.3

re =   Input(

                      shape= [1],  # 1D shape or one feature

					  name = "re"

					  )

# 11.4

eth =   Input(

                      shape= [1],  # 1D shape or one feature

					  name = "eth"

					  )

# 11.5

ad =   Input(

                      shape= (Xtr["ad"].shape[1], ),

					  name = "ad"

					  )

# 11.6

ap =   Input(

                      shape= (Xtr["ap"].shape[1],),

					  name = "ap"

					  )

# Embedding layers for each of the two of the columns with sequence data

emb_ad  =      Embedding(MAX_VOCAB_AD+ 1 ,      32  )(ad )

emb_ap  =      Embedding(MAX_VOCAB_AP+ 1 ,      32  )(ap)



# Embedding layers for the two categorical variables

emb_re  =      Embedding(MAX_VOCAB_RE+ 1 ,      32  )(re)

emb_eth =      Embedding(MAX_VOCAB_ETH+ 1 ,      32  )(eth)



# GRU layers for sequences

gru_ad = GRU(16) (emb_ad)          # Output of GRU is a vector of size 8

gru_ap = GRU(16) (emb_ap)
conv_out = Conv1D(32, kernel_size=2, activation='relu')(num)

mp_num = MaxPooling1D(pool_size=2)(conv_out)

num_x = Flatten()(mp_num)

num_in = Flatten()(num)

num_final = concatenate([num_in,num_x])
model = Model([num, ohe, re, eth, ad,ap], [gru_ad, gru_ap, emb_re, emb_eth, num_final, ohe])

model.summary()
# Concatenate all outputs

class_l = concatenate([

                      gru_ad,              # GRU output is already 1D

                      gru_ap,

                      num_final,           # 1D output. No need to flatten. Observe model summary

                      ohe,                 # 1D output

                      Flatten()(emb_re),   # Need to flatten. Observe model summary above

                      Flatten()(emb_eth)

                      ]

                     )
# Add classification layer

class_l = Dense(64) (class_l)

class_l = Dropout(0.1)(class_l)

class_l = Dense(32) (class_l)

class_l = Dropout(0.1) (class_l)
# Output neuron. Activation is linear as our output is continous

output = Dense(1, activation="linear") (class_l)

# Formulate Model now

model = Model(

              inputs= [num, ohe, re, eth, ad, ap],

              outputs= output

             )

# 

model.summary()
# Model plot uisng keras plot_model()

plt.figure(figsize = (14,14))

plot_model(model, to_file = "model.png")

io.imshow("model.png")
# Compile model

model.compile(loss="mse",

              optimizer="adam",

              metrics=["mae"]

			  )
# 13.1

BATCH_SIZE = 5000

epochs = 20


# 

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