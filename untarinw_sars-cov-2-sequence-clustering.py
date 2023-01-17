import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
ii=1

with open('/kaggle/input/coronavirusgenometxt/Complete sequences.txt','r') as readfile:

    with open('hasilSekuensComplete.txt','w') as writefile:

        for line in readfile:

            if (line[0]!='>'):

                writefile.write(line[0:len(line)-1])

            else:

                writefile.write('\n')

                ii=ii+1
baris=0

kolom=0

with open('/kaggle/input/coronavirusgenometxt/hasilSekuensComplete.txt','r') as readfile:

    for line in readfile:

        baris=baris+1

        if (len(line)>kolom):

            kolom=len(line)

print('Input matrix dimension= ',baris,' x ',kolom)

InputData=np.zeros((baris, kolom))



np.save('InputData.npy', InputData)


InputData = np.load('InputData.npy')

with open('/kaggle/input/coronavirusgenometxt/hasilSekuensComplete.txt','r') as readfile:

    ii=0;

    for line in readfile:

        for jj in range (0,len(line)-1):

            if line[jj]=='T':

                InputData[ii,jj]=1

            elif line[jj]=='C':

                InputData[ii,jj]=2

            elif line[jj]=='A':

                InputData[ii,jj]=3

            elif line[jj]=='G':

                InputData[ii,jj]=4

            else:

                InputData[ii,jj]=0

        ii=ii+1

print('Mapping result:')

print(InputData)

np.save('InputDataInteger.npy', InputData)

import keras

from matplotlib import pyplot as plt

import numpy as np

#import gzip

%matplotlib inline

from keras.layers import Input, Dense

from keras.models import Model

from tensorflow.keras.utils import plot_model

#from keras.optimizers import RMSprop
InputData = np.load('/kaggle/input/coronavirusgenometxt/InputDataInteger.npy')
# Model AutoEncoder 1 (500, 250, 100, 50, 10)

InputDim = InputData.shape[1]

# Encoder

input_encoder = Input(shape=(InputDim,))

encoded1 = Dense(500, activation='relu')(input_encoder)

encoded2 = Dense(250, activation='relu')(encoded1)

encoded3 = Dense(100, activation='relu')(encoded2)

encoded4 = Dense(50, activation='relu')(encoded3)

output_encoder = Dense(10, activation='relu')(encoded4)

encoder = Model(input_encoder, output_encoder, name='encoder')

encoder.summary()



# Decoder

input_decoder = Input(shape=(10,))

decoded1 = Dense(50, activation='relu')(input_decoder)

decoded2 = Dense(100, activation='relu')(decoded1)

decoded3 = Dense(250, activation='relu')(decoded2)

decoded4 = Dense(500, activation='relu')(decoded3)

output_decoder = Dense(InputDim, activation='relu')(decoded4)

decoder = Model(input_decoder, output_decoder, name='decoder')

decoder.summary()



# Autoencoder

autoencoder = Model(input_encoder,

                    decoder(encoder(input_encoder)),

                    name='autoencoder')

autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.summary()
# Model AutoEncoder 2 (500, 250, 100, 50)

InputDim = InputData.shape[1]

# Encoder

input_encoder = Input(shape=(InputDim,))

encoded1 = Dense(500, activation='relu')(input_encoder)

encoded2 = Dense(250, activation='relu')(encoded1)

encoded3 = Dense(100, activation='relu')(encoded2)

output_encoder = Dense(50, activation='relu')(encoded3)

encoder = Model(input_encoder, output_encoder, name='encoder')

encoder.summary()



# Decoder

input_decoder = Input(shape=(50,))

decoded1 = Dense(100, activation='relu')(input_decoder)

decoded2 = Dense(250, activation='relu')(decoded1)

decoded3 = Dense(500, activation='relu')(decoded2)

output_decoder = Dense(InputDim, activation='relu')(decoded3)

decoder = Model(input_decoder, output_decoder, name='decoder')

decoder.summary()



# Autoencoder

autoencoder = Model(input_encoder,

                    decoder(encoder(input_encoder)),

                    name='autoencoder')

autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.summary()
# Model AutoEncoder 3 (500, 250, 100)

InputDim = InputData.shape[1]

# Encoder

input_encoder = Input(shape=(InputDim,))

encoded1 = Dense(500, activation='relu')(input_encoder)

encoded2 = Dense(250, activation='relu')(encoded1)

output_encoder = Dense(100, activation='relu')(encoded2)

encoder = Model(input_encoder, output_encoder, name='encoder')

encoder.summary()



# Decoder

input_decoder = Input(shape=(100,))

decoded1 = Dense(250, activation='relu')(input_decoder)

decoded2 = Dense(500, activation='relu')(decoded1)

output_decoder = Dense(InputDim, activation='relu')(decoded2)

decoder = Model(input_decoder, output_decoder, name='decoder')

decoder.summary()



# Autoencoder

autoencoder = Model(input_encoder,

                    decoder(encoder(input_encoder)),

                    name='autoencoder')

autoencoder.compile(loss='mse', optimizer='adam')

autoencoder.summary()
from sklearn.model_selection import KFold

from numpy import savetxt

 

n_split=5

ii=1

kf=KFold(n_split)

#autoencoder.load_weights('ae_weights.npy')



for train_index, val_index in kf.split(InputData):

    x_train,x_test=InputData[train_index],InputData[val_index]

    y_train,y_test=InputData[train_index],InputData[val_index]

    autoencoder.fit(x_train, y_train,epochs=10)

    evalModel =  autoencoder.evaluate(x_test,y_test)

    print('Model evaluation ',evalModel)

autoencoder.save_weights('ae_weights.npy')
autoencoder.load_weights('ae_weights.npy')

sekuens_reduced = encoder.predict(InputData)

print('size before reduction:', InputData.shape)

print('size after reduction :', sekuens_reduced.shape)

print(sekuens_reduced)

np.save('sekuens_reduced.npy',sekuens_reduced)
from sklearn.cluster import KMeans

import numpy as np

from sklearn.metrics import silhouette_samples, silhouette_score
sse = {}

label_sekuens={}

silAvg=-1

nClustOpt=2

for nCluster in range(2, 51):

    # Train K-means

    kmeans = KMeans(n_clusters=nCluster, max_iter=100).fit(sekuens_reduced)

    cluster_labels = kmeans.fit_predict(sekuens_reduced)

    print("Number of iteration: ", kmeans.n_iter_)

    #print(cluster_labels)

    

    # Evaluation using Elbow Method:

    sse[nCluster] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center

    

    # Evaluation using Silhouette Analysis

    silhouette_avg = silhouette_score(sekuens_reduced, cluster_labels)

    print("For n_clusters =", nCluster,

          "The average silhouette_score is :", silhouette_avg)

    if (silhouette_avg>=silAvg):

        nClustOpt=nCluster



print("Number of cluster optimum =", nClustOpt)

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()
kmeans = KMeans(n_clusters=nClustOpt, max_iter=100).fit(sekuens_reduced)

cluster_labels = kmeans.fit_predict(sekuens_reduced)



import numpy as np

np.savetxt('ClusterLabel.csv', cluster_labels, delimiter=',')