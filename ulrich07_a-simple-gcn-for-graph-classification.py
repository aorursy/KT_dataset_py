import os

import sys

import networkx as nx

import numpy as np

import matplotlib.pyplot as plt

from math import ceil



import warnings

warnings.filterwarnings('ignore')
def get_graph_features(G, all_embds, nmax=15):



    n = len(G.nodes())

    

    node2id = {node:i for i, node in enumerate(G.nodes())}

    id2node = {i:node for node,i in node2id.items()}



    adj = np.zeros((nmax,nmax))

    embds = np.zeros((nmax, all_embds.shape[1]))



    for i in G.nodes():

        embds[node2id[i]] = all_embds[i]

        for j in G.neighbors(i):

            adj[node2id[j],node2id[i]] = 1

    

    return adj, embds
def simulate_graph(nb_samples=50, vocab_size=15, graph_length=10, 

                   word_embed_dim=3, nb_classes=5, probas = None, 

                   centers=None, random_state=None):

    

    np.random.seed(random_state)

    

    graphs = []

    

    y = np.zeros(nb_samples)

    

    for i in range(nb_samples):

        

        p = np.random.uniform()

        

        cat = int(p*nb_classes)

        

        y[i] = cat

        

        if probas is None:

        

            G = nx.binomial_graph(graph_length, p)

        else:

            

            G = nx.binomial_graph(graph_length, probas[cat])

        

        new_nodes = np.random.randint(vocab_size*cat,vocab_size*(cat+1), graph_length).tolist()

        

        mapping  = dict(zip(G.nodes(), new_nodes))

        

        G = nx.relabel_nodes(G,mapping)

        

        graphs.append(G)



    try:

        embds = np.vstack((c + np.random.normal(size = (vocab_size, word_embed_dim))  for c in centers))

    except:

        embds = np.random.normal(size = (nb_classes *  vocab_size, word_embed_dim))

    

    return embds, graphs, y
def one_hot_encode(y):

    mods = len(np.unique(y))

    y_enc = np.zeros((y.shape[0], mods))

    

    for i in range(y.shape[0]):

        y_enc[i, y[i]] = 1

    return y_enc
import tensorflow as tf
class GCN():

    

    def __init__(self, node_dim=2, graph_dim=2, nb_classes=2, nmax=15, alpha=0.025):

        """

        Parameters of the model architecture

        

        """

        self.node_dim = node_dim

        self.graph_dim = graph_dim

        self.nb_classes = nb_classes

        self.nmax = nmax

        self.alpha = alpha

        

        self.build_model()

        

    def build_model(self):

        self.adjs = tf.placeholder(tf.float32, shape=[None, self.nmax, self.nmax])

        self.embeddings = tf.placeholder(tf.float32, shape=[None, self.nmax, self.node_dim])

        self.targets = tf.placeholder(tf.float32, shape=[None, self.nb_classes])

        

        A1 = tf.Variable(tf.random_normal([self.graph_dim, self.node_dim], seed=None))

        B1 = tf.Variable(tf.random_normal([self.graph_dim, self.node_dim]))

        W  = tf.Variable(tf.random_normal([self.graph_dim, self.nb_classes]))

        

        M1 = tf.einsum('adc,adb->abc', self.embeddings, self.adjs)

        H1 = tf.nn.relu(tf.tensordot(M1, A1, (2, 1)) + tf.tensordot(self.embeddings, B1, (2, 1)))

        G1 = tf.reduce_mean(H1, 1)

        

        Y_OUT = tf.matmul(G1,W)

        cost = tf.losses.softmax_cross_entropy(self.targets, Y_OUT)

        

        self.predictions = tf.argmax(Y_OUT, 1)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)

        self.train = optimizer.minimize(cost)

        

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        

    def fit(self, adj, embds, y, epochs=20, batch_size=10, shuffle=True):

        self.scores = []

        y_enc = one_hot_encode(y)

        minibatches = ceil(len(adj) / batch_size)

        

        j = 0

        for i in range(epochs):

            INDS = np.array(range(len(adj)))

            

            if shuffle:

                idx = np.random.permutation(y.shape[0]) 

                INDS = INDS[idx]

                

            mini = np.array_split(INDS, minibatches)

            

            for inds in mini:

                j+=1

                sys.stderr.write('\rEpoch: %d/%d' % (j, epochs*minibatches))

                sys.stderr.flush()

                self.sess.run(self.train, feed_dict={self.adjs:adj[inds], self.embeddings:embds[inds], 

                                                self.targets:y_enc[inds]})

                #

            self.scores.append(self.score(adj, embds, y))

            #

    def predict(self, adj, embds):

        return self.sess.run(self.predictions, feed_dict={self.adjs:adj, self.embeddings:embds})

    

    def score(self, adj, embds,y):

        y_ = self.predict(adj, embds)

        return 100*(y==y_).mean()

        

        
WORD_EMBED_DIM = 5

NB_SAMPLES = 10000

VOCAB_SIZE = 15

MAX_LENGTH = 10

NB_CLASSES = 3

PROBAS = [0.3, 0.4, 0.55]

CENTERS  =[0.1, 0.15, 0.2]

SHARE = .75

GRAPH_DIM = 10
embds, graphs, y = simulate_graph(NB_SAMPLES, VOCAB_SIZE, MAX_LENGTH,

                                                    WORD_EMBED_DIM, NB_CLASSES, 

                                                    PROBAS, CENTERS, random_state=123)

y = y.astype(int)



Adjs, Ids = [], []



for graph in graphs:

    adj, embds_g = get_graph_features(graph, embds, nmax=VOCAB_SIZE)

    Adjs.append(adj)

    Ids.append(embds_g)
ADJ = np.array(Adjs)

ID = np.array(Ids)



CUT = int(NB_SAMPLES * SHARE)

ADJ_train, y_train, ADJ_test, y_test = ADJ[:CUT], y[:CUT], ADJ[CUT:], y[CUT:]

ID_train, ID_test = ID[:CUT], ID[CUT:]
gcn_tf = GCN(node_dim=WORD_EMBED_DIM, graph_dim=GRAPH_DIM, nb_classes=NB_CLASSES, 

             nmax=VOCAB_SIZE, alpha=0.025)
gcn_tf.fit(ADJ_train, ID_train, y_train, epochs=15, batch_size=32)
gcn_tf.score(ADJ_train, ID_train, y_train)
gcn_tf.score(ADJ_test, ID_test, y_test)
class MLGCN():

    

    def __init__(self, node_dim=2, graph_dim=[3,3], nb_classes=2, nmax=15, alpha=0.025):

        """

        Parameters of the model architecture

        

        """

        self.graph_dims = [node_dim] + graph_dim

        self.n_layers = len(graph_dim)

        self.nb_classes = nb_classes

        self.nmax = nmax

        self.alpha = alpha

        

        self.build_model()

        

    def build_model(self):

        self.adjs = tf.placeholder(tf.float32, shape=[None, self.nmax, self.nmax])

        self.targets = tf.placeholder(tf.float32, shape=[None, self.nb_classes])

        

        self.A = {i+1: tf.Variable(tf.random_normal([self.graph_dims[i+1], self.graph_dims[i]])) \

             for i in range(self.n_layers)}

        self.B = {i+1: tf.Variable(tf.random_normal([self.graph_dims[i+1], self.graph_dims[i]])) \

             for i in range(self.n_layers)}

        self.W  = tf.Variable(tf.random_normal([self.graph_dims[-1], self.nb_classes]))

        

        

        self.M, self.H, self.G = {}, {}, {}

        

        self.H[0] = tf.placeholder(tf.float32, shape=[None, self.nmax, self.graph_dims[0]])

        

        for i in range(1, self.n_layers+1):

        

            self.M[i] = tf.einsum('adc,adb->abc', self.H[i-1], self.adjs)

            self.H[i] = tf.nn.relu(tf.tensordot(self.M[i], self.A[i], (2, 1)) 

                                   + tf.tensordot(self.H[i-1], self.B[i], (2, 1)))

            self.G[i] = tf.reduce_mean(self.H[i], 1)

        

        Y_OUT = tf.matmul(self.G[self.n_layers], self.W)

        cost = tf.losses.softmax_cross_entropy(self.targets, Y_OUT)

        

        self.predictions = tf.argmax(Y_OUT, 1)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.alpha)

        self.train = optimizer.minimize(cost)

        

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        

    def fit(self, adj, embds, y, epochs=20, batch_size=10, shuffle=True):

        self.scores = []

        minibatches = ceil(len(adj) / batch_size)

        

        y_enc = one_hot_encode(y)

        

        j = 0

        for i in range(epochs):

            INDS = np.array(range(len(adj)))

            

            if shuffle:

                idx = np.random.permutation(y.shape[0]) 

                INDS = INDS[idx]

                

            mini = np.array_split(INDS, minibatches)

            

            for inds in mini:

                j+=1

                sys.stderr.write('\rEpoch: %d/%d' % (j, epochs*minibatches))

                sys.stderr.flush()

                self.sess.run(self.train, feed_dict={self.adjs:adj[inds], self.H[0]:embds[inds], 

                                                self.targets:y_enc[inds]})

                

            self.scores.append(self.score(adj, embds, y))

            

        

        

    def predict(self, adj, embds):

        return self.sess.run(self.predictions, feed_dict={self.adjs:adj, self.H[0]:embds})

    

    def score(self, adj, embds,y):

        y_ = self.predict(adj, embds)

        return 100*(y==y_).mean()

        

        
mlgcn_tf = MLGCN(node_dim=WORD_EMBED_DIM, graph_dim=[10,10], nb_classes=NB_CLASSES, 

             nmax=VOCAB_SIZE, alpha=0.025)
mlgcn_tf.fit(ADJ_train, ID_train, y_train, epochs=15, batch_size=32)
mlgcn_tf.score(ADJ_train, ID_train, y_train)
mlgcn_tf.score(ADJ_test, ID_test, y_test)
plt.plot(gcn_tf.scores, label = 'one-layer')

plt.plot(mlgcn_tf.scores, label = 'multi-layer')

plt.title('accuracy')

plt.xlabel('epochs')

plt.legend(loc='best')

plt.show()