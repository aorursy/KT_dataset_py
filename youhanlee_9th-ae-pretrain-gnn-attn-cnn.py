# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

# os.environ["CUDA_VISIBLE_DEVICES"]="0"





# pretrain_dir = None#"/kaggle/input/covid-v9-no-consis/"



# one_fold = False

# # one_fold = True#False

# # with_ae = False#True

# run_test = False

# # run_test = True

# denoise = True



# ae_epochs = 28

# ae_epochs_each = 7

# ae_batch_size = 32



# epochs_list = [32, 16, 16, 16, 8, 8]

# batch_size_list = [8, 16, 32, 64, 128, 256]



# ## copy pretrain model to working dir

# import shutil

# import glob

# if pretrain_dir is not None:

#     for d in glob.glob(pretrain_dir + "*"):

#         shutil.copy(d, ".")

    

# import numpy as np # linear algebra

# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# import gc

# import os

# import matplotlib.pyplot as plt

# %matplotlib inline



# NAME = 'covid_ae_pretrain_gnn_attn_cnn_addAdj_more_deep_gaussDist_struct_gru_feats_Se'



# ## load



# import json

# import glob

# from tqdm.notebook import tqdm



# train = pd.read_json("../input//train.json",lines=True)

# mask = train.signal_to_noise > 1

# if denoise:

#     train = train[train.signal_to_noise > 1].reset_index(drop = True)

# test  = pd.read_json("../input//test.json",lines=True)

# test_pub = test[test["seq_length"] == 107]

# test_pri = test[test["seq_length"] == 130]

# sub = pd.read_csv("../input//sample_submission.csv")



# if run_test:

#     train = train[:30]

#     test_pub = test_pub[:30]

#     test_pri = test_pri[:30]



# As = []

# for id in tqdm(train["id"]):

#     a = np.load(f"../input//bpps/{id}.npy")

#     As.append(a)

# As = np.array(As)

# As_pub = []

# for id in tqdm(test_pub["id"]):

#     a = np.load(f"../input//bpps/{id}.npy")

#     As_pub.append(a)

# As_pub = np.array(As_pub)

# As_pri = []

# for id in tqdm(test_pri["id"]):

#     a = np.load(f"../input//bpps/{id}.npy")

#     As_pri.append(a)

# As_pri = np.array(As_pri)



# As1 = []

# for id in tqdm(train["id"]):

#     a = np.load(f"../input_bpps/bpps_contrafold/{id}.npy")

#     As1.append(a)

# As1 = np.array(As1)

# As1_pub = []

# for id in tqdm(test_pub["id"]):

#     a = np.load(f"../input_bpps/bpps_contrafold/{id}.npy")

#     As1_pub.append(a)

# As1_pub = np.array(As1_pub)

# As1_pri = []

# for id in tqdm(test_pri["id"]):

#     a = np.load(f"../input_bpps/bpps_contrafold/{id}.npy")

#     As1_pri.append(a)

# As1_pri = np.array(As1_pri)



# As2 = []

# for id in tqdm(train["id"]):

#     a = np.load(f"../input_bpps/bpps_eternafold//{id}.npy")

#     As2.append(a)

# As2 = np.array(As2)

# As2_pub = []

# for id in tqdm(test_pub["id"]):

#     a = np.load(f"../input_bpps/bpps_eternafold//{id}.npy")

#     As2_pub.append(a)

# As2_pub = np.array(As2_pub)

# As2_pri = []

# for id in tqdm(test_pri["id"]):

#     a = np.load(f"../input_bpps/bpps_eternafold//{id}.npy")

#     As2_pri.append(a)

# As2_pri = np.array(As2_pri)



# As3 = []

# for id in tqdm(train["id"]):

#     a = np.load(f"../input_bpps/bpps_nupack///{id}.npy")

#     As3.append(a)

# As3 = np.array(As3)

# As3_pub = []

# for id in tqdm(test_pub["id"]):

#     a = np.load(f"../input_bpps/bpps_nupack//{id}.npy")

#     As3_pub.append(a)

# As3_pub = np.array(As3_pub)

# As3_pri = []

# for id in tqdm(test_pri["id"]):

#     a = np.load(f"../input_bpps/bpps_nupack//{id}.npy")

#     As3_pri.append(a)

# As3_pri = np.array(As3_pri)



# As4 = []

# for id in tqdm(train["id"]):

#     a = np.load(f"../input_bpps/bpps_contrafold_linear//{id}.npy")

#     As4.append(a)

# As4 = np.array(As4)

# As4_pub = []

# for id in tqdm(test_pub["id"]):

#     a = np.load(f"../input_bpps/bpps_contrafold_linear//{id}.npy")

#     As4_pub.append(a)

# As4_pub = np.array(As4_pub)

# As4_pri = []

# for id in tqdm(test_pri["id"]):

#     a = np.load(f"../input_bpps/bpps_contrafold_linear//{id}.npy")

#     As4_pri.append(a)

# As4_pri = np.array(As4_pri)



# As5 = []

# for id in tqdm(train["id"]):

#     a = np.load(f"../input_bpps/bpps_vienna_linear///{id}.npy")

#     As5.append(a)

# As5 = np.array(As5)

# As5_pub = []

# for id in tqdm(test_pub["id"]):

#     a = np.load(f"../input_bpps/bpps_vienna_linear//{id}.npy")

#     As5_pub.append(a)

# As5_pub = np.array(As5_pub)

# As5_pri = []

# for id in tqdm(test_pri["id"]):

#     a = np.load(f"../input_bpps/bpps_vienna_linear//{id}.npy")

#     As5_pri.append(a)

# As5_pri = np.array(As5_pri)



# print(train.shape)

# train.head()



# print(test.shape)

# test.head()



# print(sub.shape)

# sub.head()



# ## target



# targets = list(sub.columns[1:])

# print(targets)



# y_train = []

# seq_len = train["seq_length"].iloc[0]

# seq_len_target = train["seq_scored"].iloc[0]

# ignore = -10000

# ignore_length = seq_len - seq_len_target

# for target in targets:

#     y = np.vstack(train[target])

#     dummy = np.zeros([y.shape[0], ignore_length]) + ignore

#     y = np.hstack([y, dummy])

#     y_train.append(y)

# y = np.stack(y_train, axis = 2)

# y.shape



# ## structure adj



# def get_structure_adj(train):

#     Ss = []

#     for i in tqdm(range(len(train))):

#         seq_length = train["seq_length"].iloc[i]

#         structure = train["structure"].iloc[i]

#         sequence = train["sequence"].iloc[i]



#         cue = []

#         a_structures = {

#             ("A", "U") : np.zeros([seq_length, seq_length]),

#             ("C", "G") : np.zeros([seq_length, seq_length]),

#             ("U", "G") : np.zeros([seq_length, seq_length]),

#             ("U", "A") : np.zeros([seq_length, seq_length]),

#             ("G", "C") : np.zeros([seq_length, seq_length]),

#             ("G", "U") : np.zeros([seq_length, seq_length]),

#         }

#         a_structure = np.zeros([seq_length, seq_length])

#         for i in range(seq_length):

#             if structure[i] == "(":

#                 cue.append(i)

#             elif structure[i] == ")":

#                 start = cue.pop()

# #                 a_structure[start, i] = 1

# #                 a_structure[i, start] = 1

#                 a_structures[(sequence[start], sequence[i])][start, i] = 1

#                 a_structures[(sequence[i], sequence[start])][i, start] = 1

        

#         a_strc = np.stack([a for a in a_structures.values()], axis = 2)

#         a_strc = np.sum(a_strc, axis = 2, keepdims = True)

#         Ss.append(a_strc)

    

#     Ss = np.array(Ss)

#     print(Ss.shape)

#     return Ss

# Ss = get_structure_adj(train)

# Ss_pub = get_structure_adj(test_pub)

# Ss_pri = get_structure_adj(test_pri)



# ## distance adj



# def get_distance_matrix(As):

#     idx = np.arange(As.shape[1])

#     Ds = []

#     for i in range(len(idx)):

#         d = np.abs(idx[i] - idx)

#         Ds.append(d)



#     Ds = np.array(Ds) + 1

#     Ds = 1/Ds

#     Ds = Ds[None, :,:]

#     Ds = np.repeat(Ds, len(As), axis = 0)

    

#     Dss = []

#     for i in [1, 2, 4]:

#         Dss.append(Ds ** i)

#     #Ds = np.stack(Dss, axis = 3)

#     for gamma in np.arange(10):

#         Dss.append(np.exp(-np.power(Ds, 2) * gamma))

#     Ds = np.stack(Dss, axis = 3)



#     print(Ds.shape)

#     return Ds



# Ds = get_distance_matrix(As)

# Ds_pub = get_distance_matrix(As_pub)

# Ds_pri = get_distance_matrix(As_pri)



# ## concat adjecent

# As = np.concatenate([As[:,:,:,None], As1[:,:,:,None], As2[:,:,:,None], As3[:,:,:,None], As4[:,:,:,None], As5[:,:,:,None], Ss, Ds], axis = 3).astype(np.float32)

# As_pub = np.concatenate([As_pub[:,:,:,None], As1_pub[:,:,:,None], As2_pub[:,:,:,None], As3_pub[:,:,:,None], As4_pub[:,:,:,None], As5_pub[:,:,:,None],

#                          Ss_pub, Ds_pub], axis = 3).astype(np.float32)

# As_pri = np.concatenate([As_pri[:,:,:,None], As1_pri[:,:,:,None], As2_pri[:,:,:,None], As3_pri[:,:,:,None], As4_pri[:,:,:,None], As5_pri[:,:,:,None],

#                          Ss_pri, Ds_pri], axis = 3).astype(np.float32)

# del Ss, Ds, Ss_pub, Ds_pub, Ss_pri, Ds_pri

# As.shape, As_pub.shape, As_pri.shape



# np.save('../input/train_As.npy', As)



# np.save('../input/public_As.npy', As_pub)



# np.save('../input/private_As.npy', As_pri)







# ## node



# ## sequence

# def return_ohe(n, i):

#     tmp = [0] * n

#     tmp[i] = 1

#     return tmp



# def get_input(train):

#     mapping = {}

#     vocab = ["A", "G", "C", "U"]

#     for i, s in enumerate(vocab):

#         mapping[s] = return_ohe(len(vocab), i)

#     X_node = np.stack(train["sequence"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))



#     mapping = {}

#     vocab = ["S", "M", "I", "B", "H", "E", "X"]

#     for i, s in enumerate(vocab):

#         mapping[s] = return_ohe(len(vocab), i)

#     X_loop = np.stack(train["predicted_loop_type"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

    

#     mapping = {}

#     vocab = [".", "(", ")"]

#     for i, s in enumerate(vocab):

#         mapping[s] = return_ohe(len(vocab), i)

#     X_structure = np.stack(train["structure"].apply(lambda x : list(map(lambda y : mapping[y], list(x)))))

    

    

#     X_node = np.concatenate([X_node, X_loop, X_structure], axis = 2)

    

#     ## interaction

#     a = np.sum(X_node * (2 ** np.arange(X_node.shape[2])[None, None, :]), axis = 2)

#     vocab = sorted(set(a.flatten()))

#     print(vocab)

#     ohes = []

#     for v in vocab:

#         ohes.append(a == v)

#     ohes = np.stack(ohes, axis = 2)

#     X_node = np.concatenate([X_node, ohes], axis = 2).astype(np.float32)

    

    

#     print(X_node.shape)

#     return X_node



# X_node = get_input(train)

# X_node_pub = get_input(test_pub)

# X_node_pri = get_input(test_pri)







# ## model



# import tensorflow as tf

# from tensorflow.keras import layers as L

# import tensorflow_addons as tfa

# from tensorflow.keras import backend as K



# def mcrmse(t, p, seq_len_target = seq_len_target):

#     score = np.mean(np.sqrt(np.mean((p - y_va) ** 2, axis = 2))[:, :seq_len_target])

#     return score



# def mcrmse_loss(t, y, seq_len_target = seq_len_target):

#     t = t[:, :seq_len_target]

#     y = y[:, :seq_len_target]

    

#     loss = tf.reduce_mean(tf.sqrt(tf.reduce_mean((t - y) ** 2, axis = 2)))

#     return loss



# def se_block(x_in, layer_n):

#     x = L.GlobalAveragePooling1D()(x_in)

#     x = L.Dense(layer_n//8, activation="relu")(x)

#     x = L.Dense(layer_n, activation="sigmoid")(x)

#     x_out= L.Multiply()([x_in, x])

#     return x_out



# def attention(x_inner, x_outer, n_factor, dropout):

#     x_Q =  L.Conv1D(n_factor, 1, activation='linear', 

#                   kernel_initializer='glorot_uniform',

#                   bias_initializer='glorot_uniform',

#                  )(x_inner)

#     x_K =  L.Conv1D(n_factor, 1, activation='linear', 

#                   kernel_initializer='glorot_uniform',

#                   bias_initializer='glorot_uniform',

#                  )(x_outer)

#     x_V =  L.Conv1D(n_factor, 1, activation='linear', 

#                   kernel_initializer='glorot_uniform',

#                   bias_initializer='glorot_uniform',

#                  )(x_outer)

#     x_KT = L.Permute((2, 1))(x_K)

#     res = L.Lambda(lambda c: K.batch_dot(c[0], c[1]) / np.sqrt(n_factor))([x_Q, x_KT])

# #     res = tf.expand_dims(res, axis = 3)

# #     res = L.Conv2D(16, 3, 1, padding = "same", activation = "relu")(res)

# #     res = L.Conv2D(1, 3, 1, padding = "same", activation = "relu")(res)

# #     res = tf.squeeze(res, axis = 3)

#     att = L.Lambda(lambda c: K.softmax(c, axis=-1))(res)

#     att = L.Lambda(lambda c: K.batch_dot(c[0], c[1]))([att, x_V])

#     return att



# def multi_head_attention(x, y, n_factor, n_head, dropout):

#     if n_head == 1:

#         att = attention(x, y, n_factor, dropout)

#     else:

#         n_factor_head = n_factor // n_head

#         heads = [attention(x, y, n_factor_head, dropout) for i in range(n_head)]

#         att = L.Concatenate()(heads)

#         att = L.Dense(n_factor, 

#                       kernel_initializer='glorot_uniform',

#                       bias_initializer='glorot_uniform',

#                      )(att)

#     x = L.Add()([x, att])

#     x = L.LayerNormalization()(x)

#     if dropout > 0:

#         x = L.Dropout(dropout)(x)

#     return x



# def res(x, unit, kernel = 3, rate = 0.1):

#     h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)

#     h = se_block(h, unit)

#     h = L.LayerNormalization()(h)

#     h = L.LeakyReLU()(h)

#     h = L.Dropout(rate)(h)

#     return L.Add()([x, h])



# def forward(x, unit, kernel = 3, rate = 0.1):

# #     h = L.Dense(unit, None)(x)

#     h = L.Conv1D(unit, kernel, 1, padding = "same", activation = None)(x)

#     h = L.LayerNormalization()(h)

#     h = L.Dropout(rate)(h)

# #         h = tf.keras.activations.swish(h)

#     h = L.LeakyReLU()(h)

#     h = res(h, unit, kernel, rate)

#     return h



# def adj_attn(x, adj, unit, n = 2, rate = 0.1):

#     x_a = x

#     x_as = []

#     for i in range(n):

#         x_a = forward(x_a, unit)

#         x_a = tf.matmul(adj, x_a)

#         x_as.append(x_a)

#     if n == 1:

#         x_a = x_as[0]

#     else:

#         x_a = L.Concatenate()(x_as)

#     x_a = forward(x_a, unit)

#     return x_a





# def get_base(config):

#     node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")

#     adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")

    

#     adj_learned = L.Dense(64, "relu")(adj)

#     adj_learned = L.Dense(1, "relu")(adj_learned)

    

    

#     adj_all = L.Concatenate(axis = 3)([adj, adj_learned])

        

#     xs = []

#     xs.append(node)

#     x1 = forward(node, 128, kernel = 3, rate = 0.0)

#     x2 = forward(x1, 64, kernel = 6, rate = 0.0)

#     x3 = forward(x2, 32, kernel = 12, rate = 0.0)

#     x4 = forward(x3, 16, kernel = 24, rate = 0.0)

#     x5 = forward(x4, 8, kernel = 48, rate = 0.0)

#     x = L.Concatenate()([x1, x2, x3, x4, x5])

    

#     for unit in [64, 48, 32]:

#         x_as = []

#         for i in range(adj_all.shape[3]):

#             x_a = adj_attn(x, adj_all[:, :, :, i], unit, rate = 0.0)

#             x_as.append(x_a)

#         x_c = forward(x, unit, kernel = 32)

        

#         x = L.Concatenate()(x_as + [x_c])

#         x = forward(x, unit)

#         x = multi_head_attention(x, x, unit, 4, 0.0)

#         xs.append(x)

        

#     x = L.Concatenate()(xs)



#     model = tf.keras.Model(inputs = [node, adj], outputs = [x])

#     return model





# def get_ae_model(base, config):

#     node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")

#     adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")



#     x = base([L.SpatialDropout1D(0.3)(node), adj])

#     x = forward(x, 64, rate = 0.3)

#     p = L.Dense(X_node.shape[2], "sigmoid")(x)

    

#     loss = - tf.reduce_mean(20 * node * tf.math.log(p + 1e-4) + (1 - node) * tf.math.log(1 - p + 1e-4))

#     model = tf.keras.Model(inputs = [node, adj], outputs = [loss])

    

#     opt = get_optimizer()

#     model.compile(optimizer = opt, loss = lambda t, y : y)

#     return model



# def gru_layer(hidden_dim, dropout):

#     return tf.keras.layers.Bidirectional(

#                                 tf.keras.layers.GRU(hidden_dim,

#                                 dropout=dropout,

#                                 return_sequences=True,

#                                 kernel_initializer = 'orthogonal'))



# def get_model(base, config):

#     node = tf.keras.Input(shape = (None, X_node.shape[2]), name = "node")

#     adj = tf.keras.Input(shape = (None, None, As.shape[3]), name = "adj")

#     inputs_nums = tf.keras.layers.Input(shape=(seq_len, 18), name='input_nums')

#     x = base([node, adj])

#     nums = tf.keras.layers.Dense(128, activation='relu')(inputs_nums)



#     x = L.Concatenate()([x, nums])

    

#     dropout = 0.1

#     x = gru_layer(128, dropout)(x)

#     x = forward(x, 128, rate = 0.4)

#     x = gru_layer(128, dropout)(x)

#     x = forward(x, 128, rate = 0.4)

    

#     x = L.Dense(5, None)(x)



#     model = tf.keras.Model(inputs = [node, adj, inputs_nums], outputs = [x])

    

#     opt = get_optimizer()

#     model.compile(optimizer = opt, loss = mcrmse_loss)

#     return model



# def get_optimizer():

# #     sgd = tf.keras.optimizers.SGD(0.05, momentum = 0.9, nesterov=True)

#     adam = tf.optimizers.Adam()

# #     radam = tfa.optimizers.RectifiedAdam()

# #     lookahead = tfa.optimizers.Lookahead(adam, sync_period=6)

# #     swa = tfa.optimizers.SWA(adam)

#     return adam



# train_nums = np.load('../input/train_nums.npy')[mask]

# public_nums = np.load('../input/public_nums.npy')

# private_nums = np.load('../input/private_nums.npy')



# ## pretrain



# config = {}



# if ae_epochs > 0:

#     base = get_base(config)

#     ae_model = get_ae_model(base, config)

#     ## TODO : simultaneous train

#     for i in range(ae_epochs//ae_epochs_each):

#         print(f"------ {i} ------")

#         print("--- train ---")

#         ae_model.fit([X_node, As], [X_node[:,0]],

#                   epochs = ae_epochs_each,

#                   batch_size = ae_batch_size,

#                   callbacks=[tf.keras.callbacks.ModelCheckpoint(f"./base_ae_{NAME}", save_weights_only=True)])

#         print("--- public ---")

#         ae_model.fit([X_node_pub, As_pub], [X_node_pub[:,0]],

#                   epochs = ae_epochs_each,

#                   batch_size = ae_batch_size,

#                      callbacks=[tf.keras.callbacks.ModelCheckpoint(f"./base_ae_{NAME}", save_weights_only=True)])

#         print("--- private ---")

#         ae_model.fit([X_node_pri, As_pri], [X_node_pri[:,0]],

#                   epochs = ae_epochs_each,

#                   batch_size = ae_batch_size,

#                     callbacks=[tf.keras.callbacks.ModelCheckpoint(f"./base_ae_{NAME}", save_weights_only=True)])

#         gc.collect()

#     print("****** save ae model ******")

#     base.save_weights(f"./base_ae_{NAME}")



# ## train



# from sklearn.model_selection import KFold

# kfold = KFold(5, shuffle = True, random_state = 42)



# scores = []

# preds = np.zeros([len(X_node), X_node.shape[1], 5])

# for i, (tr_idx, va_idx) in enumerate(kfold.split(X_node, As)):

#     print(f"------ fold {i} start -----")

#     print(f"------ fold {i} start -----")

#     print(f"------ fold {i} start -----")

#     X_node_tr = X_node[tr_idx]

#     X_node_va = X_node[va_idx]

#     X_nums_tr = train_nums[tr_idx]

#     X_nums_va = train_nums[va_idx]

#     As_tr = As[tr_idx]

#     As_va = As[va_idx]

#     y_tr = y[tr_idx]

#     y_va = y[va_idx]

    

#     base = get_base(config)

#     if ae_epochs > 0:

#         print("****** load ae model ******")

#         base.load_weights(f"./base_ae_{NAME}")

#     model = get_model(base, config)

#     if pretrain_dir is not None:

#         d = f"./model_addmore{i}_{NAME}"

#         print(f"--- load from {d} ---")

#         model.load_weights(d)

#     for epochs, batch_size in zip(epochs_list, batch_size_list):

#         print(f"epochs : {epochs}, batch_size : {batch_size}")

#         model.fit([X_node_tr, As_tr, X_nums_tr], [y_tr],

#                   validation_data=([X_node_va, As_va, X_nums_va], [y_va]),

#                   epochs = epochs,

#                   batch_size = batch_size, validation_freq = 1,

#                   callbacks=[tf.keras.callbacks.ModelCheckpoint(f"./model_addmore{i}_{NAME}", save_weights_only=True)])

        

#     model.load_weights(f"./model_addmore{i}_{NAME}")

#     p = model.predict([X_node_va, As_va, X_nums_va])

#     scores.append(mcrmse(y_va, p))

#     print(f"fold {i}: mcrmse {scores[-1]}")

#     preds[va_idx] = p

#     if one_fold:

#         break

        

# pd.to_pickle(preds, f"oof_addmore_{NAME}.pkl")



# print(scores)



# ## predict



# p_pub = 0

# p_pri = 0

# for i in range(5):

#     model.load_weights(f"./model_addmore{i}_{NAME}")

#     p_pub += model.predict([X_node_pub, As_pub, public_nums]) / 5

#     p_pri += model.predict([X_node_pri, As_pri, private_nums]) / 5

#     if one_fold:

#         p_pub *= 5

#         p_pri *= 5

#         break



# for i, target in enumerate(targets):

#     test_pub[target] = [list(p_pub[k, :, i]) for k in range(p_pub.shape[0])]

#     test_pri[target] = [list(p_pri[k, :, i]) for k in range(p_pri.shape[0])]



# ## sub



# preds_ls = []

# for df, preds in [(test_pub, p_pub), (test_pri, p_pri)]:

#     for i, uid in enumerate(df.id):

#         single_pred = preds[i]



#         single_df = pd.DataFrame(single_pred, columns=targets)

#         single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



#         preds_ls.append(single_df)



# preds_df = pd.concat(preds_ls)

# preds_df.to_csv(f"{NAME}.csv", index = False)

# preds_df.head()



# print(scores)

# print(np.mean(scores))


