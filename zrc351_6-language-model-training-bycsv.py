import tensorflow as tf

import os

import csv

import random

import pathlib

import numpy as np



print(os.listdir("../input/pfb-countor-train-data/"))

print(os.listdir("../input/pfb-countor-train-data/CSV"))

print(os.listdir("../input/pfb-countor-train-data/ACOUSTIC_H5"))
#"amtf", "gsyps", "gyps", "nmamtf", "nmgsyps"



fohao = "gyps"

MODEL_SAVE_PATH = r"./"



print("=============================Preparation training：{}====================================".format(fohao))

#=================================================================



def LoadCSV(path):    

    with open(path) as csvfile: 

        reader = csv.reader(csvfile)

        lx = []

        ly = []

        for row in reader:

            lx.append(row[0:-1])

            ly.append(row[-1])   

        lx = np.reshape(lx, (-1, 5, 39)).astype("float32")        

        ly = np.array(ly).astype("float32")        

        print("load csv successed!:{0} ".format(path))

    return lx, ly



lstm_seq_length = 5

dim = lstm_seq_length * 32



def getData(path):

    features, labels = LoadCSV(path)

    features_expand = np.expand_dims(features, axis=-1)

    rps = model_rec.predict(features_expand)    

    

    data_x, data_y = [], []

    data_len = len(labels) - lstm_seq_length

    for i in range(0, len(features) - lstm_seq_length):

        data_x.append(np.reshape(rps[i:i+lstm_seq_length], (dim,)))  

    data_x = np.expand_dims(data_x, axis=0)

    data_y = np.reshape(labels[0:data_len], (1,data_len,1))   

    print("create data shape x:{}, y:{}".format(data_x.shape, data_y.shape))       

    return data_x, data_y





CSV_FILE_DIR = r"../input/pfb-countor-train-data/CSV/{}/".format(fohao)

MODEL_REC_SAVEFILE_PATH = "{}ACOUSTIC_32_MODEL_{}.h5".format(r"../input/pfb-countor-train-data/ACOUSTIC_H5/", fohao)

model_rec = tf.keras.models.load_model(MODEL_REC_SAVEFILE_PATH)



print("=============================Create model====================================")

#=================================================================



lstm_seq_length = 5

dim = lstm_seq_length * 32



model_ctor = tf.keras.Sequential()

model_ctor.add(tf.keras.layers.LSTM(256, batch_input_shape=(1, None,dim), return_sequences=True, stateful=True))

model_ctor.add(tf.keras.layers.LSTM(256, return_sequences=True, stateful=True))

model_ctor.add(tf.keras.layers.Dense(1024, activation='relu'))

model_ctor.add(tf.keras.layers.Dense(512, activation='relu'))

model_ctor.add(tf.keras.layers.Dense(128, activation='relu'))

model_ctor.add(tf.keras.layers.Dense(1, activation='sigmoid'))

print(model_ctor.summary())

model_ctor.compile(optimizer=tf.keras.optimizers.Adam(0.00001), loss='binary_crossentropy', metrics=['acc'])


out_epochs = 1

epochs = 1000



csv_root = pathlib.Path(CSV_FILE_DIR)

filelist = list(csv_root.glob("*"))

total_epochs = 1

for oe in range(out_epochs):

    random.shuffle(filelist)

    for file in filelist:    

        path = pathlib.Path(file)             

        print("out_epoch:{}/{}".format(total_epochs, len(filelist) * out_epochs))    

        train_x, train_y  = getData(path)

        print((train_x.shape, train_y.shape))

        model_ctor.reset_states()

        model_ctor.fit(train_x, train_y, epochs=epochs) 

        total_epochs += 1

        print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx({})".format(file.name))    

print("finished!")
MODEL_SAVEFILE_PATH = "{}LANG_MODEL_{}.h5".format(MODEL_SAVE_PATH, fohao)

model_ctor.save(MODEL_SAVEFILE_PATH)

print("saved:{}".format(MODEL_SAVEFILE_PATH))

print(os.listdir("./"))