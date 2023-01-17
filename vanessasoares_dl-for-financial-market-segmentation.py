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
from time import time

import keras.backend as K

from keras.engine.topology import Layer, InputSpec

from keras.layers import Dense, Input

from keras.models import Model

from keras.optimizers import SGD

from keras import callbacks

from keras.initializers import VarianceScaling

from sklearn.cluster import KMeans

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score

import numpy as np

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA
# Data preparation

data = pd.read_csv('/kaggle/input/ccdata/CC GENERAL.csv')

data.head()
# Selecting features

# O campo cust_id é unico e não serve para clusterização, por isso é descartado

data_x = data.drop(['CUST_ID'], axis=1)
# Re-escala tudo para o intervalo entre 0 e 1

# Pois o kmeans é sensitivo para a escala de valores das features, já que usa distância euclidiana como métrica de similaridade

from sklearn.preprocessing import MinMaxScaler

numeric_columns = data_x.columns.values.tolist()

scaler = MinMaxScaler()

data_x[numeric_columns] = scaler.fit_transform(data_x[numeric_columns])

data_x.head()
# Lidando com dados faltando

data_x.isnull().sum()
# Completando os dados que faltam com zero

data_x.fillna(0, inplace=True)
# Creating and training autoencoder

def autoencoder(dims, act='relu', init='glorot_uniform'):

    """

    Fully connected symmetric auto-encoder model.

  

    dims: list of the sizes of layers of encoder like [500, 500, 2000, 10]. 

          dims[0] is input dim, dims[-1] is size of the latent hidden layer.



    act: activation function

    

    return:

        (autoencoder_model, encoder_model): Model of autoencoder and model of encoder

    """

    n_stacks = len(dims) - 1

    

    input_data = Input(shape=(dims[0],), name='input')

    x = input_data

    

    # internal layers of encoder

    for i in range(n_stacks-1):

        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)



    # latent hidden layer

    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)



    x = encoded

    # internal layers of decoder

    for i in range(n_stacks-1, 0, -1):

        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)



    # decoder output

    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)

    

    decoded = x

    

    autoencoder_model = Model(inputs=input_data, outputs=decoded, name='autoencoder')

    encoder_model     = Model(inputs=input_data, outputs=encoded, name='encoder')

    

    return autoencoder_model, encoder_model

data_x.dtypes
x = data_x.values

x.shape
# Estimating the number of clusters

# Para treinar o kmeans é necessário ter o numero de clusters

# O número de clusters é estimado explorando os valores de silhouette de diferentes execuções de k-means



# um valor de silhouette mede o quão similar um dado é dentro do seu cluster, comparado com os outros clusters

# o valor vai de -1  a +1 

# onde um valor alto indica que o dado bate com seu próprio cluster, caso seja baixo, é pq bate mais com os clusters vizinhos 



for num_clusters in range(2,10):

    clusterer = KMeans(n_clusters=num_clusters, n_jobs=4)

    preds = clusterer.fit_predict(x)

    score = silhouette_score(x, preds, metric='euclidean')

    print('For n_clusters = {}, Kmeans silhouette score is {}'.format(num_clusters, score))

    
n_clusters = 3 

n_epochs   = 100

batch_size = 128
# Creating and Training K-means model

kmeans = KMeans(n_clusters=n_clusters, n_jobs=4)

y_pred_kmeans = kmeans.fit_predict(x) #x == valores do csv


# Tamanho das camadas

# configuração generica do autoencoder da rede neural de qualquer dataset

dims = [x.shape[-1], 500, 500, 2000, 10]

init = VarianceScaling(scale=1./3., mode='fan_in', distribution='uniform')

pretrain_optimizer = SGD(lr=1, momentum=0.9)

pretrain_epochs = n_epochs

batch_size = batch_size

save_dir='/kaggle/output'
dims
init
# Criação do modelo de autoencoder



autoencoder, encoder = autoencoder(dims, init=init)
from keras.utils import plot_model

from IPython.display import Image
plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True)

Image(filename='autoencoder.png')
plot_model(autoencoder, to_file='encoder.png', show_shapes=True)

Image(filename='encoder.png')
# Treinamento do autoencoder

autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')

autoencoder.fit(x,x, batch_size=batch_size, epochs=pretrain_epochs)

autoencoder.save_weights('ae_weights.h5')
autoencoder.load_weights('ae_weights.h5')
'''

Um dos componentes chaves de DEC é o soft labeling, que é a atribuição a uma classe estimada para cada dado, 

de forma que possa ser refinado iterativamente

'''

class ClusteringLayer(Layer):

    '''

    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the

    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    '''



    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):

        if 'input_shape' not in kwargs and 'input_dim' in kwargs:

            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(ClusteringLayer, self).__init__(**kwargs)

        self.n_clusters = n_clusters

        self.alpha = alpha

        self.initial_weights = weights

        self.input_spec = InputSpec(ndim=2)



    def build(self, input_shape):

        assert len(input_shape) == 2

        input_dim = input_shape[1]

        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        self.clusters = self.add_weight(name='clusters', shape=(self.n_clusters, input_dim), initializer='glorot_uniform') 

        

        if self.initial_weights is not None:

            self.set_weights(self.initial_weights)

            del self.initial_weights

        self.built = True



    def call(self, inputs, **kwargs):

        ''' 

        student t-distribution, as used in t-SNE algorithm.

        It measures the similarity between embedded point z_i and centroid µ_j.

                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.

                 q_ij can be interpreted as the probability of assigning sample i to cluster j.

                 (i.e., a soft assignment)

       

        inputs: the variable containing data, shape=(n_samples, n_features)

        

        Return: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)

        '''

        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))

        q **= (self.alpha + 1.0) / 2.0

        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure all of the values of each sample sum up to 1.

        

        return q



    def compute_output_shape(self, input_shape):

        assert input_shape and len(input_shape) == 2

        return input_shape[0], self.n_clusters



    def get_config(self):

        config = {'n_clusters': self.n_clusters}

        base_config = super(ClusteringLayer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)

model = Model(inputs=encoder.input, outputs=clustering_layer)
plot_model(model, to_file='model.png', show_shapes=True)

Image(filename='model.png')
model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

kmeans = KMeans(n_clusters=n_clusters, n_init=20)

y_pred = kmeans.fit_predict(encoder.predict(x))
y_pred_last = np.copy(y_pred)
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
#Training the new DEC model

# computing an auxiliary target distribution

def target_distribution(q):

    weight = q ** 2 / q.sum(0)

    return (weight.T / weight.sum(1)).T



loss = 0

index = 0

maxiter = 1000

update_interval = 100

index_array = np.arange(x.shape[0])
tol = 0.001
for ite in range(int(maxiter)):

    if ite % update_interval == 0:

        q = model.predict(x, verbose=0)

        p = target_distribution(q)

    

    idx = index_array[index*batch_size : min((index+1)*batch_size, x.shape[0])]

    loss = model.train_on_batch(x=x[idx], y=p[idx])

    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    print ('ite: {}'.format(str(ite)))

    

model.save_weights('DEC_model_final.h5')
model.load_weights('DEC_model_final.h5')
#  Using Trained DEC Model for Predicting Clustering Classes

q = model.predict(x, verbose=0)

p = target_distribution(q)



y_pred = q.argmax(1)
data_all = data_x.copy()
data_all
data_all['cluster'] = y_pred
data_all.head()
data_all['cluster'].value_counts()
x_embedded = TSNE(n_components=2).fit_transform(x)

x_embedded.shape
vis_x = x_embedded[:, 0]

vis_y = x_embedded[:, 1]

plt.scatter(vis_x, vis_y, c=y_pred, cmap=plt.cm.get_cmap("jet", 256))

plt.colorbar(ticks=range(256))

plt.clim(-0.5, 9.5)

plt.show()
y_pred_kmeans.shape
plt.scatter(vis_x, vis_y, c=y_pred_kmeans, cmap=plt.cm.get_cmap('jet', 256))

plt.colorbar(ticks=range(100))

plt.clim(-0.5, 9.5)

plt.show()
score = silhouette_score(x, y_pred_kmeans, metric='euclidean')

print ("For n_clusters = {}, Kmeans silhouette score is {})".format(n_clusters, score))
score = silhouette_score (x, y_pred, metric='euclidean')

print ("For n_clusters = {}, Deep clustering silhouette score is {})".format(n_clusters, score))
for num_clusters in range(2,10):

    clusterer = KMeans(n_clusters=num_clusters, n_jobs=4)

    preds = clusterer.fit_predict(x)

    # centers = clusterer.cluster_centers_

    score = silhouette_score (x, preds, metric='euclidean')

    print ("For n_clusters = {}, Kmeans silhouette score is {})".format(num_clusters, score))
# Need to re-run autoencoder function declaration!!!

def autoencoder(dims, act='relu', init='glorot_uniform'):

    """

    Fully connected auto-encoder model, symmetric.

    Arguments:

        dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.

            The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1

        act: activation, not applied to Input, Hidden and Output layers

    return:

        (ae_model, encoder_model), Model of autoencoder and model of encoder

    """

    n_stacks = len(dims) - 1

    # input

    input_data = Input(shape=(dims[0],), name='input')

    x = input_data

    

    # internal layers in encoder

    for i in range(n_stacks-1):

        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)



    # hidden layer

    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here



    x = encoded

    # internal layers in decoder

    for i in range(n_stacks-1, 0, -1):

        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)



    # output

    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)

    decoded = x

    return Model(inputs=input_data, outputs=decoded, name='AE'), Model(inputs=input_data, outputs=encoded, name='encoder')
# Jointly Refining DEC Model

'''

 ideia principal é aprender simultaneamente a representação da feature e 

 fazer as atribuições do cluster usando DNN. 

 

 Esse código usa o autoencoder pre-treinado e o modelo de kmeans para definir um novo 

 modelo que pega o dataset pre-processado como input e dá como output as classes de clsuterização da predição 

'''

autoencoder.load_weights('ae_weights.h5')

clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)

model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])


plot_model(model, to_file='model.png', show_shapes=True)

Image(filename='model.png')



kmeans = KMeans(n_clusters=n_clusters, n_init=20)

y_pred = kmeans.fit_predict(encoder.predict(x))

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

y_pred_last = np.copy(y_pred)
model.compile(loss=['kld', 'mse'], loss_weights=[0.1, 1], optimizer=pretrain_optimizer)
for ite in range(int(maxiter)):

    if ite % update_interval == 0:

        q, _  = model.predict(x, verbose=0)

        p = target_distribution(q)  # update the auxiliary target distribution p



        # evaluate the clustering performance

        y_pred = q.argmax(1)



        # check stop criterion

        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]

        y_pred_last = np.copy(y_pred)

        if ite > 0 and delta_label < tol:

            print('delta_label ', delta_label, '< tol ', tol)

            print('Reached tolerance threshold. Stopping training.')

            break

    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]

    loss = model.train_on_batch(x=x[idx], y=[p[idx], x[idx]])

    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0



model.save_weights('b_DEC_model_final.h5')
model.load_weights('b_DEC_model_final.h5')
# evaluation of model prediction

q, _ = model.predict(x, verbose=0)

p = target_distribution(q)



# evaluate the clustering performance

y_pred = q.argmax(1)
score = silhouette_score(x, y_pred, metric='euclidean')

print ("For n_clusters = {}, Deep clustering silhouette score is {})".format(n_clusters, score))
plt.scatter(vis_x, vis_y, c=y_pred, cmap=plt.cm.get_cmap("jet", 256))

plt.colorbar(ticks=range(256))

plt.clim(-0.5, 9.5)

plt.show()




plt.scatter(vis_x, vis_y, c=y_pred_kmeans, cmap=plt.cm.get_cmap("jet", 256))

plt.colorbar(ticks=range(256))

plt.clim(-0.5, 9.5)

plt.show()



data_all['cluster'] = y_pred
data_all['cluster'].value_counts()
data_cluster_0 = data_all[data_all['cluster'] == 0]
data_cluster_0.describe()




data_cluster_1 = data_all[data_all['cluster'] == 1]

data_cluster_1.describe()



data_cluster_2 = data_all[data_all['cluster'] == 2]

data_cluster_2.describe()
pca = PCA(n_components=2)

x_pca = pca.fit_transform(x)



x_pca.shape




vis_x = x_pca[:, 0]

vis_y = x_pca[:, 1]

plt.scatter(vis_x, vis_y, c=y_pred, cmap=plt.cm.get_cmap("jet", 256))

plt.colorbar(ticks=range(256))

plt.clim(-0.5, 9.5)

plt.show()


