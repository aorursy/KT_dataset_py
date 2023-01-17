# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load both test and train data, here not loading test since it's only for competision

# df_full_test = pd.read_csv('/kaggle/input/test.csv')

df_full = pd.read_csv('/kaggle/input/train.csv')
## Visualize both the data types

# df_full.shape  

# df_full.dtypes
# Check for null or miussing values

# df_full.isnull().sum().sum() # There are no null values

# df_full_test.isnull().sum().sum() # THere are no null values
print("Counts: \n"+str(df_full.event.value_counts())+"\n")

print("Unique values: \n"+str(df_full.event.nunique()))
# ds_o_y = ds_o_train['event']

# ds_o_y.head()

# del ds_o_train['event']
from sklearn.model_selection import train_test_split 

# Split data into train and validation set

df_train , df_val = train_test_split(df_full,test_size=0.25,shuffle='true')

# Split train again for Train and Test data

df_train, df_test = train_test_split(df_train,test_size=0.10)



print("Shape:")

print("======================")

print("Train: "+str(df_train.shape))

print("Validation: "+str(df_val.shape))

print("Test: "+str(df_test.shape))
# Copy Y values from the training data set seperately 

df_tra_y = df_train['event']

df_val_y = df_val['event']

df_tes_y = df_test['event']



# Delete Y values from the data set, so that its not part of X values

del df_train['event']

del df_val['event']

del df_test['event']



# Printing all the values to keep track of shape

print("TRAIN     : "+str(df_train.shape)+" Y: "+str(df_tra_y.shape)+"\n")

print("VALIDATION: "+str(df_val.shape)+" Y: "+str(df_val_y.shape)+"\n")

print("TEST      : "+str(df_test.shape)+" Y: "+str(df_tes_y.shape)+"\n")
# Copy the Categorical data seperately, because that will be fed into Embedding layers seperately 

df_tra_experiment = df_train['experiment']

df_val_experiment = df_val['experiment']

df_tes_experiment = df_test['experiment']



# delete the value

del df_train['experiment']

del df_val['experiment']

del df_test['experiment']



print(df_tra_experiment.shape)

print(df_train.shape)

print(df_val_experiment.shape)

print(df_val.shape)

print(df_tes_experiment.shape)

print(df_test.shape)
# Empty array

categorical_array = [] 

# Array of all columns

all_columns = df_train.columns

# Define just the categorical column name

categorical_array.append('experiment')

# Substract categorical coumn from other columns

other_columns = [i for i in all_columns if i not in categorical_array]



print(categorical_array)

print(other_columns)
## This is for sequential model

# t_model = Sequential()

# t_model.add(Dense(100,activation='relu',input_shape(27,)))

# t_model.add(Dense(50,activation='relu'))

# t_model.add(Dense(4))

# t_model.compile(loss="mean_squared_error",

#                 optimizer=Adam(lr=0.001),

#                 metrics=[metrics.mae])

# This will convert categorical data to integer data, only then It can be converted into One-hot encoding

from sklearn.preprocessing import LabelEncoder

df_tra_experiment = LabelEncoder().fit_transform(df_tra_experiment)

df_val_experiment = LabelEncoder().fit_transform(df_val_experiment)

df_tes_experiment = LabelEncoder().fit_transform(df_tes_experiment)
# This will convert Label (Y values) to integer data, only then It can be converted into One-hot encoding

df_tra_y = LabelEncoder().fit_transform(df_tra_y)

df_val_y = LabelEncoder().fit_transform(df_val_y)

df_tes_y = LabelEncoder().fit_transform(df_tes_y)
# Convert labels to categorical one-hot encoding, This is optional , depending on the LOSS_FUNCTION

from keras.utils import to_categorical

one_hot_tra_y = to_categorical(df_tra_y, num_classes=4)

one_hot_val_y = to_categorical(df_val_y, num_classes=4)

one_hot_tes_y = to_categorical(df_tes_y, num_classes=4)
df_train.head(5)
# MEAN Normalize here

df_train = (df_train-df_train.mean())/df_train.std()

df_val = (df_val-df_val.mean())/df_val.std()

df_test = (df_test-df_test.mean())/df_test.std()

# df_tra_experiment_n = (df_tra_experiment-df_tra_experiment.mean())/df_tra_experiment.std()

# df_val_experiment_n = (df_val_experiment-df_val_experiment.mean())/df_val_experiment.std()

# df_tes_experiment_n = (df_tes_experiment-df_tes_experiment.mean())/df_tes_experiment.std()

for i in range(5):

    print(df_tes_experiment[i])
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# # sns.scatterplot(x="crew",y="crew",data=df_train)



# ax1 = df_train.plot(kind='scatter', x='eeg_fp1', y='eeg_f7', color='r')    

# ax2 = df_train.plot(kind='scatter', x='eeg_f8', y='eeg_t4', color='g', ax=ax1)    

ax3 = df_train.plot(kind='scatter', x='eeg_t6', y='eeg_t5', color='b')

# # ax3 = df_train.plot(kind='scatter', x='eeg_t3', y='eeg_fp2', color='b', ax=ax1)

ax3 = df_train.plot(kind='scatter', x='eeg_o1', y='eeg_p3', color='g', ax=ax3)

# # ax3 = df_train.plot(kind='scatter', x='eeg_pz', y='eeg_f3', color='b', ax=ax1)

# # ax3 = df_train.plot(kind='scatter', x='eeg_fz', y='eeg_f4', color='b', ax=ax1)

# # ax3 = df_train.plot(kind='scatter', x='eeg_c4', y='eeg_p4', color='b', ax=ax1)

# # ax3 = df_train.plot(kind='scatter', x='eeg_poz', y='eeg_c3', color='b', ax=ax1)

# # ax3 = df_train.plot(kind='scatter', x='eeg_cz', y='eeg_o2', color='b', ax=ax1)

# ax3 = df_train_n.plot(kind='scatter', x='ecg', y='eeg_p4', color='b', ax=ax1)

# Get MAX and MIN value of a column , this is the perfect place to check for values range

# After this we can verify and apply standardization

df_train.max()

df_train.min()
# MIN MAX normalization



df_train = (df_train-df_train.min())/(df_train.max()-df_train.min())

df_val = (df_val-df_val.min())/(df_val.max()-df_val.min())

df_test = (df_test-df_test.min())/(df_test.max()-df_test.min())
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# # sns.scatterplot(x="crew",y="crew",data=df_train)



# ax1 = df_train.plot(kind='scatter', x='eeg_fp1', y='eeg_f7', color='r')    

# ax2 = df_train.plot(kind='scatter', x='eeg_f8', y='eeg_t4', color='g', ax=ax1)    

ax3 = df_train.plot(kind='scatter', x='eeg_t6', y='eeg_t5', color='b')

# # ax3 = df_train.plot(kind='scatter', x='eeg_t3', y='eeg_fp2', color='b', ax=ax1)

ax3 = df_train.plot(kind='scatter', x='eeg_o1', y='eeg_p3', color='g', ax=ax3)

# # ax3 = df_train.plot(kind='scatter', x='eeg_pz', y='eeg_f3', color='b', ax=ax1)

# # ax3 = df_train.plot(kind='scatter', x='eeg_fz', y='eeg_f4', color='b', ax=ax1)

# # ax3 = df_train.plot(kind='scatter', x='eeg_c4', y='eeg_p4', color='b', ax=ax1)

# # ax3 = df_train.plot(kind='scatter', x='eeg_poz', y='eeg_c3', color='b', ax=ax1)

# # ax3 = df_train.plot(kind='scatter', x='eeg_cz', y='eeg_o2', color='b', ax=ax1)

# ax3 = df_train_n.plot(kind='scatter', x='ecg', y='eeg_p4', color='b', ax=ax1)

# # This FUNCTIONAL MODEL produces wrong output file 



# from keras.models import Sequential,Model

# from keras.layers import Dense,Activation,Dropout,Embedding,Reshape,concatenate,Input,Flatten

# from keras.optimizers import Adam,RMSprop

# from keras import metrics

# from keras.utils import plot_model



# input_of_experiment = Input(shape=(1,),dtype='int32',name='input_of_experiment')

# x = Embedding(output_dim=4,input_dim=100,input_length=1)(input_of_experiment)

# out_of_experiment = Flatten()(x)

# output_of_experiment = Dense(4,activation='relu',name='output_of_experiment')(out_of_experiment)



# input_of_other = Input(shape=(df_train.shape[1],), name='input_of_other')

# lyr = concatenate([input_of_other,output_of_experiment])

# lyr = Dense(50, activation='relu')(lyr)

# lyr = Dense(50, activation='relu')(lyr)

# lyr = Dense(50, activation='relu')(lyr)



# ## For one hot encoding this is 4

# output_of_other = Dense(4,activation='softmax',name='output_of_other')(lyr)





# t_model = Model(inputs = [input_of_experiment,input_of_other], 

#                 outputs = [output_of_other,output_of_experiment])





# ## THese to compile functions work when the Y value is in INTEGER form

# # t_model.compile(loss = "mean_squared_error",

# #                optimizer = Adam(lr=0.01),

# #                metrics = [metrics.mae],

# #                loss_weights = [1.0,0.5])



# # t_model.compile(loss = "mean_absolute_percentage_error",

# #                optimizer = Adam(lr=0.01),

# #                metrics = [metrics.mae],

# #                loss_weights = [1.0,0.5])



# # t_model.compile(optimizer = Adam(lr=0.01),

# #              loss='categorical_crossentropy',

# #              metrics=['accuracy'])



# t_model.compile(optimizer = RMSprop(lr=0.01),

#              loss='categorical_crossentropy',

#              metrics=['accuracy'])

# # val_output_of_other_acc: 0.7926 - val_output_of_experiment_acc: 0.8770 One Hot encoding of the values

# # val_output_of_other_acc: 0.9727 - val_output_of_experiment_acc: 0.8039 Normalization 

# # val_output_of_other_acc: 0.9761 - val_output_of_experiment_acc: 0.8770 (lr=0.01 , 200 input, 6 epoch )

# # val_output_of_other_acc: 0.9745 - val_output_of_experiment_acc: 0.8770 (lr=0.001 , 200 input, 10 epoch )

# # val_output_of_other_acc: 0.9804 - val_output_of_experiment_acc: 0.8772 (lr=0.001 , 200 input, 10 epoch , added layer)

# # val_output_of_other_acc: 0.9887 - val_output_of_experiment_acc: 0.8772 (lr=0.001 , 200 input, 10 epoch , added another layer)

# # val_output_of_other_acc: 0.9685 - val_output_of_experiment_acc: 0.8775 (added Dropout , but not helping)

# # val_output_of_other_acc: 0.9567 - val_output_of_experiment_acc: 0.9236 (optimizer = 'rmsProp' , loss='categorical_crossentropy' ,batch=500 , epoch=5)
# THIS IS SECOND MODEL USING FUNCTIONAL LAYER IN KERAS WITH PROPER OUTPUT FORMAT



from keras.models import Sequential,Model

from keras.layers import Dense,Activation,Dropout,Embedding,Reshape,concatenate,Input,Flatten

from keras.optimizers import Adam,RMSprop

from keras import metrics

from keras.utils import plot_model



inputExperiment = Input(shape=(1,),dtype='int32', name='inputExperiment')

x1 = Embedding(output_dim=4,input_dim=50,input_length=1)(inputExperiment)

x1 = Flatten()(x1)

x1 = Dense(50,activation='relu',name='outputExperiment')(x1)

x1 = Dense(50,activation='relu')(x1)

x1 = Dense(50,activation='relu')(x1)

x1 = Model(inputs=inputExperiment,outputs=x1)



inputOther = Input(shape=(df_train.shape[1],), name='inputOther')

x2 = Dense(50, activation='relu')(inputOther)

x2 = Dense(50, activation='relu')(x2)

x2 = Dense(50, activation='relu')(x2)

x2 = Dense(50, activation='relu')(x2)

x2 = Model(inputs=inputOther, outputs=x2)



combined = concatenate([x1.output,x2.output])



y = Dense(25,activation='relu', name='outputCombined')(combined)

y = Dense(4, activation='softmax')(y)



## For one hot encoding this is 4





t_model = Model(inputs = [inputExperiment,inputOther], 

                outputs = y)







t_model.compile(optimizer = RMSprop(lr=0.001),

             loss='categorical_crossentropy',

             metrics=['accuracy'])



# loss: 0.1503 - acc: 0.9653 - val_loss: 0.1560 - val_acc: 0.9668

## This is using one hot encoded labels and NORMALIZED values

history = t_model.fit([df_tra_experiment,df_train],[one_hot_tra_y],batch_size = 1000,epochs = 100,shuffle=True,verbose=1, 

                      validation_data = ([df_val_experiment,df_val],[one_hot_val_y]))



score = t_model.evaluate([df_tes_experiment,df_test],[one_hot_tes_y],verbose=1)
score
# Releases most of the memory taken by dataframe

import gc

del df_full

del df_train

del df_val

del df_test

gc.collect()
df_full_test = pd.read_csv('/kaggle/input/test.csv')
# This will convert categorical data to integer data, only then It can be converted into One-hot encoding

from sklearn.preprocessing import LabelEncoder

df_full_test[categorical_array] = LabelEncoder().fit_transform(df_full_test[categorical_array])
# Prediction for SINGLE VALUE, contains 2 arrays since our model has 2 inputs(categorical and numerical) we need to pass is seperately

# Here we are passing 2 value for each input, but OUTPUT is 1

single_input_array = [ 

                        [df_full_test[categorical_array].iloc[1111111]] , 

                        [df_full_test[other_columns].iloc[1111111]]

                    ]



predicted_value = t_model.predict(single_input_array, verbose=1)

print(predicted_value)
np.shape(single_input_array)
# # This DOES'NT WORK for 2 inputs values



# # Prediction for TWO VALUE, contains 2 arrays since our model has 2 inputs(categorical and numerical) we need to pass is seperately

# # Here we are passing 2 value for each input, but OUTPUT is 1

# two_input_array = [ 

#                         [[df_full_test[categorical_array].iloc[1111111]] , 

#                         [df_full_test[other_columns].iloc[1111111]],],

#                         [[df_full_test[categorical_array].iloc[1111111]] , 

#                         [df_full_test[other_columns].iloc[1111111]],]

#                 ]

# np.shape(two_input_array)

# predicted_value = t_model.predict(two_input_array, verbose=1)

# # print(predicted_value)
# Prediction for MULTIPLE VALUE, contains 2 arrays since our model has 2 inputs(categorical and numerical) we need to pass is seperately



# multiple_input_array_categorical = pd.concat([df_full_test[categorical_array]], axis = 1)

# multiple_input_array_integer = pd.concat([df_full_test[other_columns]], axis = 1)



# multiple_input_array = [ 

#                         multiple_input_array_categorical , 

#                         multiple_input_array_integer 

#                     ]



predicted_value = t_model.predict([df_full_test[categorical_array],df_full_test[other_columns]], verbose=1)

print(predicted_value)
np.shape(predicted_value)
# Copy ID so the df_full_test can be released from momory

ID = df_full_test.id

ID.shape
event = predicted_value

event.shape
del df_full_test

gc.collect()
sub = pd.DataFrame(ID)

sub.head(5)
sub1 = pd.DataFrame(event)

sub1.head(5)
# Create DATAFRAME for SUBMISSION 

sub2 = pd.concat([sub, sub1],axis=1, ignore_index=True, sort =False)
sub2.shape
# Rename column names to 

sub2.columns = ['id', 'A','B','C','D']
# Convert FLOAT to INT (Pretty easy and Usefull actually)

sub2 = sub2.astype('int32')
sub2.head(2)
# you could use any filename. We choose submission here

sub2.to_csv('submission.csv', index=False)
# ## This is using one hot encoded labels

# history = t_model.fit([df_tra_experiment,df_train],[one_hot_tra_y,one_hot_tra_y],batch_size = 50,epochs = 5,shuffle=True,verbose=1, 

#                       validation_data = ([df_val_experiment,df_val],[one_hot_val_y,one_hot_val_y]))



# score = t_model.evaluate([df_tes_experiment,df_test],[one_hot_tes_y,one_hot_tes_y],verbose=1)
# t_model.summary()

# plot_model(t_model,to_file='/plot.png',show_shapes=True,show_layer_names=True)
# t_model.get_session()
# ## Not using this, since there is not test data

# history = t_model.fit([df_tra_experiment,df_train],[df_tra_y,df_tra_y],batch_size = 128,epochs = 5,shuffle=True,verbose=1, 

#                       validation_data = ([df_val_experiment,df_val],[df_val_y,df_val_y]))



# score = t_model.evaluate([df_tes_experiment,df_test],[df_tes_y,df_tes_y],verbose=0)

# def preproc(X_train):

#     input_list_train = []

#     for c in categorical_array:

#         vals = np.asarray(X_train[c].tolist())

#         vals = pd.factorize(vals)[0]

#         input_list_train.append(np.asarray(vals))

        

#     input_list_train.append(X_train[other_columns].values)

#     return input_list_train



# df_tr_modified = preproc(ds_o_X)

# df_val_modified = preproc(ds_o_val)
# from keras.models import Sequential

# from keras.layers import Dense,Activation,Dropout,Embedding,Reshape,Concatenate



# for categorical_val in categorical_array:

#     print ("for categorical column ",categorical_val)

#     model = Sequential()

#     no_of_unique_cat = ds_o_train[categorical_val].nunique()

# #     no_of_unique_cat = 3

#     print ("number of unique cat", no_of_unique_cat)

    

#     embedding_size = min(np.ceil(no_of_unique_cat/2),50)

#     embedding_size = int(embedding_size)

    

#     print ("embedding_size set as ", embedding_size)

    

#     model.add(Embedding(no_of_unique_cat+1,embedding_size, input_length = 1))

#     model.add(Reshape(target_shape=([embedding_size])))

#     print (model.summary())

#     model_array.append(model)

     
# So the model contains N_CAT + 1 models (N_CAT models for each of the categorical

# columns and one for all other columns)

# model_array[0].summary()
# from keras.layers import Concatenate,BatchNormalization,PReLU

# from keras.models import Model

# full_model = Concatenate()([model_array])



# full_model.compile(optimizer=ada_grad,

#                    loss='binary_crossentropy',

#                    metrics=['accuracy'])



# x = BatchNormalization()(full_model)

# x = Dense(80, activation='relu')(full_model)

# x = Dropout(.35)(x)

# x = Dense(20, activation='relu')(x)

# x = Dropout(.15)(x)

# x = Dense(10, activation='relu')(x)

# x = Dropout(.15)(x)

# output = Dense(1, activation='sigmoid')(x)

# # x = PReLU()(x)

# # x = Dropout(0.2)(x)

# # x = Dense(4)(x)

# # x = BatchNormalization()(x)

# out = Activation('sigmoid')(x)



# merged_model = Model(model_array,out)

# merged_model.compile(optimizer = 'rmsprop',

#                  loss='categorical_crossentropy',

#                  metrics=['accuracy'])



# # merged_model.compile(loss = 'binary_crossentropy',

# #                      optimizer = 'adam', 

# #                      metrics = ['accuracy'])

## Delete categorical data(temporary)

# ds_o_val = ds_o_val.drop(ds_o_val.index[[1]])

# del ds_o_X['experiment']

# del ds_o_val['experiment']



# input_list_train =[]

# vals = np.asarray(ds_o_train)



# # Aattempt to embed categorical data

# def embedding_cat():

#     mode = Sequential()

#     no_of_uniqe_cat = ds_o_X['experiment'].nunique()

#     emb_size = min(np.ceil(no_of_uniqe_cat/2),50)

#     emb_size = int(emb_size)

#     vocab = no_of_uniqe_cat+1

#     model.add(Embedding(vocab,emb_size,input_length=4))

# #     model.add(Reshape(target_shape=(emb_size,)))

#     models_array.append(model)