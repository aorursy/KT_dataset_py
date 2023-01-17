
denoise = True


## copy pretrain model to working dir
import shutil
import glob

    
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc
import os
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
#%matplotlib inline

strategy = tf.distribute.get_strategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
root_dir = '../input/stanford-covid-vaccine/'
import json
import glob
from tqdm import tqdm

train = pd.read_json(root_dir + "train.json",lines=True)
if denoise:
    train = train[train.signal_to_noise > 1].reset_index(drop = True)
test  = pd.read_json(root_dir + "test.json",lines=True)
sub = pd.read_csv(root_dir + "sample_submission.csv")
aug_df = pd.read_csv('../input/covid-aug/'+ 'aug_data.csv')




def aug_data(df):
    target_df = df.copy()
    new_df = aug_df[aug_df['id'].isin(target_df['id'])]                     
    del target_df['structure']
    del target_df['predicted_loop_type']
    new_df = new_df.merge(target_df, on=['id','sequence'], how='left')

    df['cnt'] = df['id'].map(new_df[['id','cnt']].set_index('id').to_dict()['cnt'])
    df['log_gamma'] = 100
    df['score'] = 1.0
    df = df.append(new_df[df.columns])
    return df
print(train.shape, test.shape)

org_train = train.copy()
train = aug_data(train)
test = aug_data(test)
print('aug_reulst:',train.shape, test.shape)



test_pub = test[test["seq_length"] == 107]
test_pri = test[test["seq_length"] == 130]


As = []
for id in tqdm(train["id"]):
    a = np.load(root_dir + f"bpps/{id}.npy")
    As.append(a)
As = np.array(As)

As_org=[]
for id in tqdm(org_train["id"]):
    a = np.load(root_dir + f"bpps/{id}.npy")
    As_org.append(a)
As_org = np.array(As_org)

As_pub = []
for id in tqdm(test_pub["id"]):
    a = np.load(root_dir + f"bpps/{id}.npy")
    As_pub.append(a)
As_pub = np.array(As_pub)
As_pri = []
for id in tqdm(test_pri["id"]):
    a = np.load(root_dir + f"bpps/{id}.npy")
    As_pri.append(a)
As_pri = np.array(As_pri)
def read_bpps_sum(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(root_dir + f"bpps/{mol_id}.npy").sum(axis=1))
    bpps_arr = np.expand_dims(np.array(bpps_arr), -1)
    return bpps_arr

def read_bpps_max(df):
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps_arr.append(np.load(root_dir + f"/bpps/{mol_id}.npy").max(axis=1))
    
    bpps_arr = np.expand_dims(np.array(bpps_arr), -1)
    return bpps_arr

def read_bpps_nb(df):
    # normalized non-zero number
    # from https://www.kaggle.com/symyksr/openvaccine-deepergcn 
    bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data
    bpps_nb_std = 0.08914   # std of bpps_nb across all training data
    bpps_arr = []
    for mol_id in df.id.to_list():
        bpps = np.load(root_dir + f"bpps/{mol_id}.npy")
        bpps_nb = (bpps > 0).sum(axis=0) / bpps.shape[0]
        bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std
        bpps_arr.append(bpps_nb)
    bpps_arr = np.expand_dims(np.array(bpps_arr), -1)
    return bpps_arr 

train_sum = read_bpps_sum(train)
test_sum_pub = read_bpps_sum(test_pub)
test_sum_pri = read_bpps_sum(test_pri)
train_max = read_bpps_max(train)
test_max_pub = read_bpps_max(test_pub)
test_max_pri = read_bpps_max(test_pri)
train_nb = read_bpps_nb(train)
test_nb_pub = read_bpps_nb(test_pub)
test_nb_pri = read_bpps_nb(test_pri)



org_train_sum = read_bpps_sum(org_train)
org_train_max = read_bpps_max(org_train)
org_train_nb = read_bpps_nb(org_train)

targets = list(sub.columns[1:])
print(targets)

y_train = []
seq_len = train["seq_length"].iloc[0]
seq_len_target = train["seq_scored"].iloc[0]
ignore = -10000
ignore_length = seq_len - seq_len_target
for target in targets:
    y = np.vstack(train[target])
    dummy = np.zeros([y.shape[0], ignore_length]) + ignore
    y = np.hstack([y, dummy])
    y_train.append(y)
y = np.stack(y_train, axis = 2)
y.shape
def get_structure_adj(train):
    Ss = []
    for i in tqdm(range(len(train))):
        seq_length = train["seq_length"].iloc[i]
        structure = train["structure"].iloc[i]
        sequence = train["sequence"].iloc[i]

        cue = []
        a_structures = {
            ("A", "U") : np.zeros([seq_length, seq_length]),
            ("C", "G") : np.zeros([seq_length, seq_length]),
            ("U", "G") : np.zeros([seq_length, seq_length]),
            ("U", "A") : np.zeros([seq_length, seq_length]),
            ("G", "C") : np.zeros([seq_length, seq_length]),
            ("G", "U") : np.zeros([seq_length, seq_length]),
        }
        a_structure = np.zeros([seq_length, seq_length])
        for i in range(seq_length):
            if structure[i] == "(":
                cue.append(i)
            elif structure[i] == ")":
                start = cue.pop()
#                 a_structure[start, i] = 1
#                 a_structure[i, start] = 1
                a_structures[(sequence[start], sequence[i])][start, i] = 1
                a_structures[(sequence[i], sequence[start])][i, start] = 1
        
        a_strc = np.stack([a for a in a_structures.values()], axis = 2)
        a_strc = np.sum(a_strc, axis = 2, keepdims = True)
        Ss.append(a_strc)
    
    Ss = np.array(Ss)
    print(Ss.shape)
    return Ss
Ss = get_structure_adj(train)
Ss_pub = get_structure_adj(test_pub)
Ss_pri = get_structure_adj(test_pri)

Ss_org = get_structure_adj(org_train)

def get_distance_matrix(As):
    idx = np.arange(As.shape[1])
    Ds = []
    for i in range(len(idx)):
        d = np.abs(idx[i] - idx)
        Ds.append(d)

    Ds = np.array(Ds) + 1
    Ds = 1/Ds
    Ds = Ds[None, :,:]
    Ds = np.repeat(Ds, len(As), axis = 0)
    
    Dss = []
    for i in [1, 2, 4]:
        Dss.append(Ds ** i)
    Ds = np.stack(Dss, axis = 3)
    print(Ds.shape)
    return Ds

Ds = get_distance_matrix(As)
Ds_pub = get_distance_matrix(As_pub)
Ds_pri = get_distance_matrix(As_pri)

Ds_org = get_distance_matrix(As_org)
## concat adjecent
As = np.concatenate([As[:,:,:,None], Ss, Ds], axis = 3).astype(np.float32)
As_pub = np.concatenate([As_pub[:,:,:,None], Ss_pub, Ds_pub], axis = 3).astype(np.float32)
As_pri = np.concatenate([As_pri[:,:,:,None], Ss_pri, Ds_pri], axis = 3).astype(np.float32)
As_org = np.concatenate([As_org[:,:,:,None], Ss_org, Ds_org], axis = 3).astype(np.float32)

aslist=[]
for i in range(len(As)):
    aslist.append(As[i])
train['As']=aslist

aslist=[]
for i in range(len(As_pub)):
    aslist.append(As_pub[i])
test_pub['As']=aslist

aslist=[]
for i in range(len(As_pri)):
    aslist.append(As_pri[i])
test_pri['As']=aslist

aslist=[]
for i in range(len(As_org)):
    aslist.append(As_org[i])
org_train['As']=aslist


del Ss, Ds, Ss_pub, Ds_pub, Ss_pri, Ds_pri , Ds_org
As.shape, As_pub.shape, As_pri.shape

import gc
gc.collect()
## sequence
def return_ohe(n, i):
    tmp = [0] * n
    tmp[i] = 1
    return tmp

def get_input(train , ts, tm, tn):
    mapping = {}
    vocab = ["A", "G", "C", "U"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_node = np.stack(train["sequence"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

    mapping = {}
    vocab = ["S", "M", "I", "B", "H", "E", "X"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_loop = np.stack(train["predicted_loop_type"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    
    mapping = {}
    vocab = [".", "(", ")"]
    for i, s in enumerate(vocab):
        mapping[s] = return_ohe(len(vocab), i)
    X_structure = np.stack(train["structure"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))
    
    
    X_node = np.concatenate([X_node, X_loop], axis = 2)
    
    ## interaction
    a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis = 2)
    vocab = sorted(set(a.flatten()))
    print(vocab)
    ohes = []
    for v in vocab:
        ohes.append(a == v)
    ohes = np.stack(ohes, axis = 2)
    X_node = np.concatenate([X_node, ohes, ts, tm, tn], axis = 2).astype(np.float32)
    
    xnodelist =[]
    for i in range(len(X_node)):
        xnodelist.append(X_node[i])
    train['X_node']= xnodelist
    return train, X_node

train, X_node = get_input(train, train_sum, train_max, train_nb)
test_pub, _ = get_input(test_pub, test_sum_pub, test_max_pub, test_nb_pub)
test_pri, _ = get_input(test_pri, test_sum_pri, test_max_pri, test_nb_pri)

org_train, X_node_org = get_input(org_train, org_train_sum, org_train_max, org_train_nb)

import tensorflow as tf
from tensorflow.keras import layers as L
import tensorflow_addons as tfa
from tensorflow.keras import backend as K

def mcrmse_5loss(t, y, seq_len_target = seq_len_target ):
    t = t[:, : seq_len_target]
    y = y[:, :seq_len_target]
    loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean((t - y) ** 2, axis = 2)))
    return loss

def mcrmse_3loss(true, pred):
    t0 = true[:, :68, 0]
    y0 = pred[:, :68, 0]
    loss1 = tf.reduce_mean(tf.sqrt((t0 - y0) ** 2))
    t1 = true[:, :68, 1]
    y1 = pred[:, :68, 1]
    loss2 = tf.reduce_mean(tf.sqrt((t1 - y1) ** 2))
    t3 = true[:, :68, 3]
    y3 = pred[:, :68, 3]
    loss3 = tf.reduce_mean(tf.sqrt((t3 - y3) ** 2))
    return (loss1+loss2+loss3)/3

def attention(x_inner, x_outer, n_factor, dropout):
    x_Q =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_inner)
    x_K =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_V =  L.Conv1D(n_factor, 1, activation='linear', 
                  kernel_initializer='glorot_uniform',
                  bias_initializer='glorot_uniform',
                 )(x_outer)
    x_KT = L.Permute((2, 1))(x_K)
    res = L.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])
#     res = tf.expand_dims(res, axis = 3)
#     res = L.Conv2D(16, 3, 1, padding = "same", activation = "relu")(res)
#     res = L.Conv2D(1, 3, 1, padding = "same", activation = "relu")(res)
#     res = tf.squeeze(res, axis = 3)
    att = L.Lambda(lambda c: K.softmax(c, axis=-1))(res)
    att = L.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])
    return att

def multi_head_attention(x, y, n_factor, n_head, dropout):
    if n_head == 1:
        att = attention(x, y, n_factor, dropout)
    else:
        n_factor_head = n_factor // n_head
        heads = [attention(x, y, n_factor_head, dropout) for i in range(n_head)]
        att = L.Concatenate()(heads)
        att = L.Dense(n_factor, 
                      kernel_initializer='glorot_uniform',
                      bias_initializer='glorot_uniform',
                     )(att)
    x = L.Add()([x, att])
    x = L.LayerNormalization()(x)
    if dropout > 0:
        x = L.Dropout(dropout)(x)
    return x

def res(x, unit, kernel = 3, rate = 0.1, use_seper=False):
    if use_seper==False:
        h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    else:
        h = L.SeparableConv1D(unit, kernel, 1, padding = "same", activation = None)(x)        
    h = L.LayerNormalization()(h)
#     h = L.BatchNormalization()(h)
    h = L.LeakyReLU()(h)
    h = L.Dropout(rate)(h)
    return L.Add()([x, h])

def forward(x, unit, kernel = 3, rate = 0.1, use_seper=False):
#     h = L.Dense(unit, None)(x)
    if use_seper==False:
        h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)
    else:
        h = L.SeparableConv1D(unit, kernel, 1, padding = "same", activation = None)(x)        
    h = L.LayerNormalization()(h)
#     h = L.BatchNormalization()(h)
    h = L.Dropout(rate)(h)
#         h = tf.keras.activations.swish(h)
    h = L.LeakyReLU()(h)
    h = res(h, unit, kernel, rate, use_seper=use_seper)
    return h

def adj_attn(x, adj, unit, n = 2, rate = 0.1):
    x_a = x
    x_as = []
    for i in range(n):
        x_a = forward(x_a, unit)
        x_a = tf.matmul(adj, x_a)
        x_as.append(x_a)
    if n == 1:
        x_a = x_as[0]
    else:
        x_a = L.Concatenate()(x_as)
    x_a = forward(x_a, unit)
    return x_a

def adj_conv_learn(adj, kernel_size = 15):
    adj_id = L.Conv2D(4, kernel_size, 1, padding = "same", activation = 'relu')(adj) 
    adj_learned = L.Conv2D(4, kernel_size, 1, padding = "same", activation = 'relu')(adj_id)     
    adj_learned = L.Add()([adj_id, adj_learned])
    adj_learned = L.BatchNormalization()(adj_learned)
    return adj_learned
    
    
def get_base(config):
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")
#     adj_learned = adj_conv_learn(adj)
    
    adj_learned = L.Dense(1, "relu")(adj)
    adj_all = L.Concatenate(axis = 3)([adj, adj_learned])
        
    xs = []
    xs.append(node)
    
    x1 = forward(node, 128, kernel = 3, rate = config['inner_dropout'],use_seper = config['use_seper'])
    x2 = forward(x1, 64, kernel = 6, rate = config['inner_dropout'],use_seper = config['use_seper'])
    x3 = forward(x2, 32, kernel = 15, rate = config['inner_dropout'],use_seper = config['use_seper'])
    x4 = forward(x3, 16, kernel = 30, rate = config['inner_dropout'],use_seper = config['use_seper'])

    x = L.Concatenate()([x1, x2, x3, x4])

    
    for unit in [64, 32]:
        x_as = []
        for i in range(adj_all.shape[3]):
            x_a = adj_attn(x, adj_all[:, :, :, i], unit, rate = 0.0)
            x_as.append(x_a)
        x_c = forward(x, unit, kernel = 30)
        
        x = L.Concatenate()(x_as + [x_c])
        x = forward(x, unit)
#         x = multi_head_attention(x, x, unit, 4, 0.0)
        x = multi_head_attention(x, x, unit, 4, 0.0)
        xs.append(x)
        
    x = L.Concatenate()(xs)

    model = tf.keras.Model(inputs = [node, adj], outputs = [x])
    return model



def get_ae_model(base, config):
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")

    x = base([L.SpatialDropout1D(0.3)(node), adj])
    x = forward(x, 64, rate = 0.3)
    p = L.Dense(X_node.shape[2], "linear")(x)
    
    print(node.shape, p.shape)
    loss = tf.reduce_mean( (node - p)**2)
    
    model = tf.keras.Model(inputs = [node, adj], outputs = [loss])

    opt = tf.optimizers.Adam()


    model.compile(optimizer = opt, loss = lambda t, y : y)
    return model


def get_model(base, config):
    node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")
    adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")
    x = base([node, adj])
#     x = L.GlobalAvgPool1D()(x)
    for i in range(config['last_layer_n']):
        x = forward(x, config['last_forward_n'], rate = config['last_dropout'])
    x = L.Dense(5, None)(x)

    model = tf.keras.Model(inputs = [node, adj], outputs = [x])
    

    return model
config = dict(use_seper=True)
configs=[]
configs.append(config)
test_profile='gnn_attn_cnn_debug_lastremove'
import os
save_dir = f'./weights/{test_profile}'
try:
    os.mkdir(save_dir)
except:
    print('already exist dir')
    
debug =False
if debug ==True:
    ae_epochs = 1
    ae_epochs_each = 1
    ae_batch_size = 32

    epochs_list = [1]
    batch_size_list = [32]
else:
    ae_epochs = 20
    ae_epochs_each = 5
    ae_batch_size = 32

    batch_size_list=[8, 16, 32, 64, 128, 256]
    epochs_list = [10, 10, 10, 20, 30, 30]

def train_base_ae(save_dir, config):
    print('train_base_ae', save_dir, config)
    base = get_base(config)
    with strategy.scope():
        ae_model = get_ae_model(base, config)
    ## TODO : simultaneous train
    for i in range(ae_epochs//ae_epochs_each):
        print(f"------ {i} ------")
        print("--- train ---")
        ae_model.fit([np.array(train['X_node'].tolist()), np.array(train['As'].tolist())], [np.array(train['X_node'].tolist())[:,0]],
                  epochs = ae_epochs_each,
                  batch_size = ae_batch_size)
        
        print("--- public ---")
        ae_model.fit([np.array(test_pub['X_node'].tolist()), np.array(test_pub['As'].tolist())], [np.array(test_pub['X_node'].tolist())[:,0]],
                  epochs = ae_epochs_each,
                  batch_size = ae_batch_size)
        print("--- private ---")
        ae_model.fit([np.array(test_pri['X_node'].tolist()), np.array(test_pri['As'].tolist())], [np.array(test_pri['X_node'].tolist())[:,0]],
                  epochs = ae_epochs_each,
                  batch_size = ae_batch_size)
        gc.collect()
    print("****** save ae model ******")
    base.save_weights(f"{save_dir}/base_ae.h5")
    
# debug=True
if debug==True:
    config['inner_dropout']=0.2
    config['last_dropout']=0.0
    train_base_ae(save_dir, config)
from sklearn.model_selection import KFold,GroupKFold
from sklearn.cluster import KMeans
# kfold = KFold(5, shuffle = True, random_state = 42)
n_folds=5
seed=42
# clustering for  GroupKFold
kmeans_model = KMeans(n_clusters=200, random_state=111).fit((X_node_org)[:,:,0])
org_train['cluster_id'] = kmeans_model.labels_
gkf = GroupKFold(n_splits=n_folds)
if 'cluster_id' not in train.columns:
    train = train.merge(org_train[['id', 'cluster_id']], how='left', on='id')
from tensorflow.keras.utils import Sequence
class CovidGenerator(Sequence):
    def __init__(self, Node, Adj, y, sn_tr, batch_size, shuffle=True, crop_length=107, min_crop=68
                 , crop_ratio=0.5, left_padd=0):
        self.Node = Node
        self.Adj = Adj
        self.y = y
        self.w_trn = np.abs(np.array(np.log(train.signal_to_noise+1.104)/np.log(train.signal_to_noise+1.104).mean()))
        self._batch_size = batch_size
        self._list_idx = [i for i in range(len(Node))]
        self._shuffle = shuffle
        self.crop_length = crop_length
        self.min_crop = min_crop
        self.crop_ratio = crop_ratio
        self.left_padd=left_padd
        self.on_epoch_end()  
        
    def __len__(self):
        return int(np.ceil(len(self.Node)/self._batch_size))
    
    def __getitem__(self, index):
        batch_idx = self._indices[index*self._batch_size:(index+1)*self._batch_size]
        X_nodes = []
        Adjs =[]
        ys =[]
        ws = []

        cur_crop_len = self.crop_length
        
                
        for i in batch_idx:
            X_nodes.append(np.pad(self.Node[i],((self.left_padd,0),(0,0)))[:cur_crop_len])
            Adjs.append(np.pad(self.Adj[i],((self.left_padd,0),(self.left_padd,0),(0,0)))[:cur_crop_len,:cur_crop_len])
            ys.append(np.pad(self.y[i],((self.left_padd,0),(0,0)))[:cur_crop_len])
            ws.append(self.w_trn[i])            
        return ([np.array(X_nodes), np.array(Adjs)], np.array(ys), np.array(ws))
    
    
    def on_epoch_end(self):
        self._indices = np.concatenate([np.arange(len(self._list_idx)),np.arange(len(self._list_idx))])
        if self._shuffle:
            np.random.shuffle(self._indices)
# train_gen = CovidGenerator(X_node, As, y,train.signal_to_noise, 64, shuffle=True, min_crop=68, crop_length=91, crop_ratio=1.0, left_padd=0)
# kk = train_gen.__getitem__(0)
# kk[0][0].shape
def mcrmse(t, y, seq_len_target = seq_len_target ):
    t = t[:, :seq_len_target]
    y = y[:, :seq_len_target]
    loss = np.mean(np.sqrt(np.mean((t - y) ** 2, axis = 2)))
    print(loss)
    return np.mean(loss)
cv_scores=[]

def train_main_cv_model(save_dir, config, foldid=None, trainskip=False):
#for i, (tr_idx, va_idx) in enumerate(kfold.split(X_node, As)):
    print('train_main_model', save_dir, config)
    scores = []
    preds = np.zeros([len(X_node), X_node.shape[1], 5])
    for i, (tr_idx, va_idx) in enumerate(gkf.split(train, train['reactivity'], train['cluster_id'])):
        print(f"------ fold {i} start -----")
#         if i<= config['skip_fold']:
#             print('config[skip_fold]',config['skip_fold'])
#             continue
        
        print(i, foldid)
        if foldid is not None:
            if i!= foldid:
                continue
        
        X_node_tr = np.array(train.X_node.tolist())[tr_idx]#X_node[tr_idx]
        X_node_va = np.array(train.X_node.tolist())[va_idx]#X_node[va_idx]
        As_tr = np.array(train.As.tolist())[tr_idx]##As[tr_idx]
        As_va = np.array(train.As.tolist())[va_idx]#As[va_idx]
        sn_tr = train.loc[tr_idx].signal_to_noise
        y_tr = y[tr_idx]
        y_va = y[va_idx]
        

        base = get_base(config)
        if ae_epochs > 0:
            print("****** load ae model ******")
            base.load_weights(f"{save_dir}/base_ae.h5")
        with strategy.scope():
            model = get_model(base, config)
        w_path = f'{save_dir}/model{i}.h5'
        if trainskip==True:
            model.load_weights(w_path)
            val_pred = model.predict([X_node_va, As_va], verbose=1) 
            scores.append(mcrmse(y_va, val_pred))
            continue
        

        checkpoint = tf.keras.callbacks.ModelCheckpoint(w_path,monitor='val_mcrmse_3loss', verbose=1, save_best_only=True, mode='min')


        val=train.loc[va_idx]
    #     print('val=train[va_idx]', val.shape)        
        val=val[val.SN_filter == 1]
        X_node_va = X_node[val.index]
        As_va =As[val.index]
        y_va = y[val.index]

        for bidx, (epochs, batch_size) in enumerate(zip(epochs_list, batch_size_list)):
            print(f"epochs : {epochs}, batch_size : {batch_size}")
            if bidx != 0:
                model.load_weights(w_path)

            lr = 0.001
            lrs = np.arange(lr, lr*0.01, -(lr)/epochs)
    
           
            for ep in range(epochs):
                cur_lr = lrs[ep]
                left_padd = np.random.randint(24)
                cur_crop_len = 107
                if np.random.rand() < config['crop_ratio']:
                    cur_crop_len = np.random.randint(config['min_crop'] + left_padd, 108)
                print('left_padd', left_padd, 'cur_crop_len',cur_crop_len)
                train_gen = CovidGenerator( X_node_tr, As_tr, y_tr, sn_tr, batch_size
                                           , shuffle=True,  crop_length=cur_crop_len
#                                            , crop_ratio=config['crop_ratio']min_crop=config['min_crop'],
                                          ,left_padd=left_padd)
                
                print('current lr', cur_lr)
                opt = tf.optimizers.Adam(learning_rate=  cur_lr)         
                def mcrmse_loss(t, y, seq_len_target = seq_len_target , left_padd=left_padd):
                    t = t[:, left_padd:left_padd+ seq_len_target]
                    y = y[:, left_padd:left_padd+ seq_len_target]
                    loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean((t - y) ** 2, axis = 2)))
                    return loss
                model.compile(optimizer = opt, loss = mcrmse_loss, metrics=[mcrmse_5loss, mcrmse_3loss] )
                
                history = model.fit( train_gen, # x = [X_node_tr, As_tr], y= [y_tr], batch_size = batch_size,
                          validation_data=([X_node_va, As_va], [y_va]),
                          epochs = ep+1,callbacks=[checkpoint],
                          verbose=1,max_queue_size=1, initial_epoch=ep)


        scores.append(min(history.history['val_mcrmse_5loss']))

    print(scores)
    return model, np.mean(scores)


if debug==True:
    config={'use_seper': True, 'inner_dropout':0.2, 'last_dropout':0.0}
    model, cv_score = train_main_cv_model(save_dir, config)
    print(np.mean(cv_score))
def predict(save_dir, model):
    p_pub = 0
    p_pri = 0
    
    X_node_pub = np.array(test_pub['X_node'].tolist())
    X_node_pri = np.array(test_pri['X_node'].tolist())
    As_pub = np.array(test_pub['As'].tolist())
    As_pri = np.array(test_pri['As'].tolist())
    for i in range(n_folds):
        w_path = f'{save_dir}/model{i}.h5'        
        model.load_weights(w_path) 
        p_pub += model.predict([X_node_pub, As_pub], verbose=1) / n_folds
        p_pri += model.predict([X_node_pri, As_pri], verbose=1) / n_folds

    for i, target in enumerate(targets):
        test_pub[target] = [list(p_pub[k, :, i]) for k in range(p_pub.shape[0])]
        test_pri[target] = [list(p_pri[k, :, i]) for k in range(p_pri.shape[0])]

    preds_ls = []
    for df, preds in [(test_pub, p_pub), (test_pri, p_pri)]:
        for i, uid in enumerate(df.id):
            single_pred = preds[i]
            single_df = pd.DataFrame(single_pred, columns=targets)
            single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]
            preds_ls.append(single_df)
            
    preds_df = pd.concat(preds_ls).groupby('id_seqpos').mean().reset_index()
    preds_df.to_csv(f"{save_dir}/submission.csv", index = False)
    return preds_df


if debug==True:            
    preds_df = predict(save_dir,model)

def train_and_predict(save_dir, config, skip_ae=False, foldid=None, trainskip=False):
    print('train_and_predict', save_dir, config)
    try:
        os.mkdir(save_dir)
    except:
        print('already exist dir')
    if skip_ae==False:
        train_base_ae(save_dir, config)
        
    model, cv_score = train_main_cv_model(save_dir, config, foldid=foldid, trainskip=trainskip)
    
    if foldid is not None:
        print('individual fold train end')
        return None
    
    sub_df = predict(save_dir, model)
    with open(f'{save_dir}/scores.txt', mode='at') as score_memo:
        logstr = f'{config},{cv_score}'
        score_memo.write(logstr+'\r')
    return sub_df

# if debug == True:
#     train_and_predict(save_dir, config)
def dict_to_path(tt):
    path_str =''
    for key, value in tt.items():
        path_str+= str(key)+str(value)
    return path_str
configs=[]
profiles=[]
configs.append(dict(use_seper=False, inner_dropout = 0.2, last_dropout=0.0
                    , crop_ratio=0.3, last_forward_n=512, min_crop=68, last_layer_n =1))
profiles.append('gnncnnpaddcrop_'+ dict_to_path(configs[-1]))
# configs.append(dict(use_seper=True, inner_dropout = 0.2, last_dropout=0.0
#                     , crop_ratio=0.3, last_forward_n=512, min_crop=68, last_layer_n =1))
# profiles.append('gnncnnpaddcrop_'+ dict_to_path(configs[-1]))

print(len(profiles))
for profile, config in zip(profiles, configs):
    save_dir = f'./weights/{profile}'
    sub_df = train_and_predict(save_dir, config)
#     sub_df = train_and_predict(save_dir, config, skip_ae=True, foldid=None, trainskip=True)
    

