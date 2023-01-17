# import pandas as pd

# import numpy as np

# import json

# import tensorflow.keras.layers as L

# import tensorflow as tf

# from tensorflow.keras import backend as K



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



# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

# os.environ["CUDA_VISIBLE_DEVICES"]="0"







# NAME = 'tf_reverseTrain_transformer_convattention_morebpps_feat_convforward_adjAtt_nowave'



# pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']



# def gru_layer(hidden_dim, dropout):

#     return tf.keras.layers.Bidirectional(

#                                 tf.keras.layers.GRU(hidden_dim,

#                                 dropout=dropout,

#                                 return_sequences=True,

#                                 kernel_initializer = 'orthogonal'))



# def lstm_layer(hidden_dim, dropout):

#     return tf.keras.layers.Bidirectional(

#                                 tf.keras.layers.LSTM(hidden_dim,

#                                 dropout=dropout,

#                                 return_sequences=True,

#                                 kernel_initializer = 'orthogonal'))



# def build_model(gru=False,seq_len=107, pred_len=68, dropout=0.25,

#                 embed_dim=128, hidden_dim=128, reverse=False):

    

#     def wave_block(x, filters, kernel_size, n):

#         dilation_rates = [2 ** i for i in range(n)]

#         x = tf.keras.layers.Conv1D(filters = filters, 

#                                    kernel_size = 1,

#                                    padding = 'same')(x)

#         res_x = x

#         for dilation_rate in dilation_rates:

#             tanh_out = tf.keras.layers.Conv1D(filters = filters,

#                               kernel_size = kernel_size,

#                               padding = 'same', 

#                               activation = 'tanh', 

#                               dilation_rate = dilation_rate)(x)

#             sigm_out = tf.keras.layers.Conv1D(filters = filters,

#                               kernel_size = kernel_size,

#                               padding = 'same',

#                               activation = 'sigmoid', 

#                               dilation_rate = dilation_rate)(x)

#             x = tf.keras.layers.Multiply()([tanh_out, sigm_out])

#             x = tf.keras.layers.Conv1D(filters = filters,

#                        kernel_size = 1,

#                        padding = 'same')(x)

#             res_x = tf.keras.layers.Add()([res_x, x])

#         return res_x

    

#     inputs = tf.keras.layers.Input(shape=(seq_len, 3))

#     inputs_bpps = tf.keras.layers.Input(shape=(seq_len, 6), name='input_bpps')

#     inputs_nums = tf.keras.layers.Input(shape=(seq_len, 18), name='input_nums')

#     adj = tf.keras.Input(shape = (None, None, train_As.shape[3]), name = "adj")

    

#     adj_learned = L.Dense(64, "relu")(adj)

#     adj_learned = L.Dense(1, "relu")(adj_learned)

    

#     adj_all = L.Concatenate(axis = 3)([adj, adj_learned])

    



#     embed0 = tf.keras.layers.Embedding(input_dim=len(token2int0), output_dim=embed_dim)(inputs[:, :, 0])

#     embed1 = tf.keras.layers.Embedding(input_dim=len(token2int1), output_dim=embed_dim)(inputs[:, :, 1])

#     embed2 = tf.keras.layers.Embedding(input_dim=len(token2int2), output_dim=embed_dim)(inputs[:, :, 2])

    

    

#     embed0 = tf.keras.layers.SpatialDropout1D(.2)(embed0)

#     embed1 = tf.keras.layers.SpatialDropout1D(.2)(embed1)

#     embed2 = tf.keras.layers.SpatialDropout1D(.2)(embed2)

    

#     embed = tf.concat([embed0, embed1, embed2], axis=2)

    

#     #reshaped = tf.reshape(

#     #    embed, shape=(-1, embed.shape[1],  embed.shape[2] * embed.shape[3]))

    

#     embed = tf.keras.layers.SpatialDropout1D(.2)(embed)

    

#     bpps = tf.keras.layers.Dense(embed_dim, activation='relu', name='dense_bpps')(inputs_bpps)

#     nums = tf.keras.layers.Dense(embed_dim, activation='relu', name='dense_nums')(inputs_nums)

    

#     print(nums.shape)

#     embed = tf.concat([embed, bpps, nums], axis=2)

#     embed = tf.keras.layers.Conv1D(filters = 256,

#                               kernel_size = 1,

#                               padding = 'same', 

#                               activation = 'relu')(embed)

    

#     transformer_block = TransformerBlock(256, 4, 256)

#     embed = transformer_block(embed)

    

#     xs = []

#     xs.append(embed)

#     x2 = forward(embed, 64, kernel = 6, rate = 0.0)

#     x3 = forward(x2, 32, kernel = 12, rate = 0.0)

#     x4 = forward(x3, 16, kernel = 24, rate = 0.0)

#     x5 = forward(x4, 8, kernel = 48, rate = 0.0)

    

#     x = L.Concatenate()([embed, x2, x3, x4, x5])

    

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

    

    

#     hidden = gru_layer(hidden_dim, dropout)(x)

#     hidden = gru_layer(hidden_dim, dropout)(hidden)

#     x = gru_layer(hidden_dim, dropout)(hidden)

    

#     x = forward(x, 256, kernel = 3, rate = 0.1)

#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.Dropout(dropout)(x)

    



#     x = forward(x, 128, kernel = 3, rate = 0.1)

#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.Dropout(dropout)(x)

    

#     x = forward(x, 128, kernel = 3, rate = 0.1)

#     x = tf.keras.layers.BatchNormalization()(x)

#     x = tf.keras.layers.Dropout(dropout)(x)

    

#     #only making predictions on the first part of each sequence

#     if reverse:

#         truncated = x[:, -pred_len:]

#     else:

#         truncated = x[:, :pred_len]

    

#     out1 = tf.keras.layers.Dense(5, activation='linear', name='out1')(truncated)

#     out2 = tf.keras.layers.Dense(5, activation='linear', name='out2')(truncated)



#     model = tf.keras.Model(inputs=[inputs, inputs_bpps, inputs_nums, adj], outputs=[out1, out2])



#     #some optimizers

#     adam = tf.optimizers.Adam()

#     def MCRMSE(y_true, y_pred):

#         colwise_mse = tf.reduce_mean(tf.square(y_true - y_pred), axis=1)

#         return tf.reduce_mean(tf.sqrt(colwise_mse), axis=1)

    

#     model.compile(optimizer = adam, loss={'out1': MCRMSE, 'out2': 'mae'}, loss_weights={'out1': 0.7, 'out2': 0.3})

    

#     return model



# from tqdm import tqdm_notebook



# def process_inputs_2(df):

#     df1 = df.copy()

#     df2 = df.copy()

#     df3 = df.copy()

#     df4 = df.copy()

#     df5 = df.copy()

#     from collections import Counter as count

#     bases = []

#     for j in range(len(df1)):

#         counts = dict(count(df1.iloc[j]['sequence']))

#         bases.append((

#             counts['A'] / df1.iloc[j]['seq_length'],

#             counts['G'] / df1.iloc[j]['seq_length'],

#             counts['C'] / df1.iloc[j]['seq_length'],

#             counts['U'] / df1.iloc[j]['seq_length']

#         ))



#     bases = pd.DataFrame(bases, columns=['A_percent', 'G_percent', 'C_percent', 'U_percent'])

#     del df1

    

#     pairs = []

#     all_partners = []

#     for j in tqdm_notebook(range(len(df2))):

#         partners = [-1 for i in range(130)]

#         pairs_dict = {}

#         queue = []

#         for i in range(0, len(df2.iloc[j]['structure'])):

#             if df2.iloc[j]['structure'][i] == '(':

#                 queue.append(i)

#             if df2.iloc[j]['structure'][i] == ')':

#                 first = queue.pop()

#                 try:

#                     pairs_dict[(df2.iloc[j]['sequence'][first], df2.iloc[j]['sequence'][i])] += 1

#                 except:

#                     pairs_dict[(df2.iloc[j]['sequence'][first], df2.iloc[j]['sequence'][i])] = 1



#                 partners[first] = i

#                 partners[i] = first



#         all_partners.append(partners)



#         pairs_num = 0

#         pairs_unique = [('U', 'G'), ('C', 'G'), ('U', 'A'), ('G', 'C'), ('A', 'U'), ('G', 'U')]

#         for item in pairs_dict:

#             pairs_num += pairs_dict[item]

#         add_tuple = list()

#         for item in pairs_unique:

#             try:

#                 add_tuple.append(pairs_dict[item]/pairs_num)

#             except:

#                 add_tuple.append(0)

#         pairs.append(add_tuple)



#     pairs = pd.DataFrame(pairs, columns=['U-G', 'C-G', 'U-A', 'G-C', 'A-U', 'G-U'])

#     del df2

    

#     pairs_rate = []

#     for j in range(len(df3)):

#         res = dict(count(df3.iloc[j]['structure']))

#         pairs_rate.append(res['('] / (df3.iloc[j]['seq_length']/2))



#     pairs_rate = pd.DataFrame(pairs_rate, columns=['pairs_rate'])

#     del df3

    

#     loops = []

#     for j in range(len(df4)):

#         counts = dict(count(df4.iloc[j]['predicted_loop_type']))

#         available = ['E', 'S', 'H', 'B', 'X', 'I', 'M']

#         row = []

#         for item in available:

#             try:

#                 row.append(counts[item] / df4.iloc[j]['seq_length'])

#             except:

#                 row.append(0)

#         loops.append(row)



#     loops = pd.DataFrame(loops, columns=available)

#     del df4

    

#     return pd.concat([df5, bases, pairs, loops, pairs_rate], axis=1)



# token2int = {x:i for i, x in enumerate('().ACGUBEHIMSX')}



# def preprocess_inputs(df, cols=['sequence', 'structure', 'predicted_loop_type']):

#     return np.transpose(

#         np.array(

#             df[cols]

#             .applymap(lambda seq: [token2int[x] for x in seq])

#             .values

#             .tolist()

#         ),

#         (0, 2, 1)

#     )



# train = pd.read_json('../input//train.json', lines=True)

# test = pd.read_json('../input//test.json', lines=True)

# sample_df = pd.read_csv('../input//sample_submission.csv')



# #train = process_inputs_2(train)

# #test = process_inputs_2(test)



# target_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']



# token2int0 = {'G': 0, 'A': 1, 'C': 2, 'U': 3}

# token2int1 = {'.': 0,  '(': 1, ')': 2}

# token2int2 = {'E': 0, 'S': 1, 'H': 2, 'B': 3, 'X': 4, 'I': 5, 'M': 6}



# def convert_seq(x, tmp_dict):

#     return [tmp_dict[ele] for ele in x]



# train['sequence'] = train['sequence'].apply(lambda x: [token2int0[ele] for ele in x])

# train['structure'] = train['structure'].apply(lambda x: [token2int1[ele] for ele in x])

# train['predicted_loop_type'] = train['predicted_loop_type'].apply(lambda x: [token2int2[ele] for ele in x])

# train_inputs = np.transpose(np.array(train[['sequence', 'structure', 'predicted_loop_type']].values.tolist()), (0, 2, 1))



# train_inputs = train_inputs[train.signal_to_noise > 1]

# train_labels = np.array(train[train.signal_to_noise > 1][target_cols].values.tolist()).transpose((0, 2, 1))





# train_bpps1 = np.stack([1 - np.load(f'../input/bpps/{ele}.npy').sum(1) for ele in train['id']])

# train_bpps1 = train_bpps1[train.signal_to_noise > 1][:, :, np.newaxis]



# train_bpps2 = np.stack([1 - np.load(f'../input/bpps_nupack/{ele}.npy').sum(1) for ele in train['id']])

# train_bpps2 = train_bpps2[train.signal_to_noise > 1][:, :, np.newaxis]



# train_bpps3 = np.stack([1 - np.load(f'../input/bpps_contrafold//{ele}.npy').sum(1) for ele in train['id']])

# train_bpps3 = train_bpps3[train.signal_to_noise > 1][:, :, np.newaxis]



# train_bpps4 = np.stack([1 - np.load(f'../input/bpps_contrafold_linear//{ele}.npy').sum(1) for ele in train['id']])

# train_bpps4 = train_bpps4[train.signal_to_noise > 1][:, :, np.newaxis]



# train_bpps5 = np.stack([1 - np.load(f'../input/bpps_eternafold/{ele}.npy').sum(1) for ele in train['id']])

# train_bpps5 = train_bpps5[train.signal_to_noise > 1][:, :, np.newaxis]



# train_bpps6 = np.stack([1 - np.load(f'../input/bpps_vienna_linear/{ele}.npy').sum(1) for ele in train['id']])

# train_bpps6 = train_bpps6[train.signal_to_noise > 1][:, :, np.newaxis]



# train_bpps = np.concatenate([train_bpps1, train_bpps2, train_bpps3, train_bpps4, train_bpps5, train_bpps6], axis=-1)



# def preprocess_ns(df, pred_len = 68):

#     ns_columns = ['A_percent',

#        'G_percent', 'C_percent', 'U_percent', 'U-G', 'C-G', 'U-A', 'G-C',

#        'A-U', 'G-U', 'E', 'S', 'H', 'B', 'X', 'I', 'M', 'pairs_rate']

#     z = np.array(df[ns_columns])

#     b = np.repeat(z[:, np.newaxis,:], pred_len, axis=1)

#     return b



# #train_nums = preprocess_ns(train, 107)

# #np.save('../input/train_nums.npy', train_nums)



# train_nums = np.load('../input/train_nums.npy')[train.signal_to_noise > 1]



# #[train.signal_to_noise > 1]



# #test_nums = preprocess_ns(test, 108)



# # public_df = test.query("seq_length == 107").copy()

# # private_df = test.query("seq_length == 130").copy()



# # public_nums = preprocess_ns(public_df, 107)

# # private_nums = preprocess_ns(private_df, 130)

# # np.save('../input/public_nums.npy', public_nums)

# # np.save('../input/private_nums.npy', private_nums)



# train_As = np.load('../input/train_As.npy')



# from sklearn.model_selection import KFold





# # In[28]:





# import tensorflow as tf

# from tensorflow import keras

# from tensorflow.keras import layers



# class MultiHeadSelfAttention(layers.Layer):

#     def __init__(self, embed_dim, num_heads=8):

#         super(MultiHeadSelfAttention, self).__init__()

#         self.embed_dim = embed_dim

#         self.num_heads = num_heads

#         if embed_dim % num_heads != 0:

#             raise ValueError(

#                 f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"

#             )

#         self.projection_dim = embed_dim // num_heads

#         self.query_dense = layers.Conv1D(embed_dim, 1)

#         self.key_dense = layers.Conv1D(embed_dim, 1)

#         self.value_dense = layers.Conv1D(embed_dim, 1)

#         self.combine_heads = layers.Conv1D(embed_dim, 1)



#     def attention(self, query, key, value):

#         score = tf.matmul(query, key, transpose_b=True)

#         dim_key = tf.cast(tf.shape(key)[-1], tf.float32)

#         scaled_score = score / tf.math.sqrt(dim_key)

#         weights = tf.nn.softmax(scaled_score, axis=-1)

#         output = tf.matmul(weights, value)

#         return output, weights



#     def separate_heads(self, x, batch_size):

#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))

#         return tf.transpose(x, perm=[0, 2, 1, 3])



#     def call(self, inputs):

#         # x.shape = [batch_size, seq_len, embedding_dim]

#         batch_size = tf.shape(inputs)[0]

#         query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)

#         key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)

#         value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)

#         query = self.separate_heads(

#             query, batch_size

#         )  # (batch_size, num_heads, seq_len, projection_dim)

#         key = self.separate_heads(

#             key, batch_size

#         )  # (batch_size, num_heads, seq_len, projection_dim)

#         value = self.separate_heads(

#             value, batch_size

#         )  # (batch_size, num_heads, seq_len, projection_dim)

#         attention, weights = self.attention(query, key, value)

#         attention = tf.transpose(

#             attention, perm=[0, 2, 1, 3]

#         )  # (batch_size, seq_len, num_heads, projection_dim)

#         concat_attention = tf.reshape(

#             attention, (batch_size, -1, self.embed_dim)

#         )  # (batch_size, seq_len, embed_dim)

#         output = self.combine_heads(

#             concat_attention

#         )  # (batch_size, seq_len, embed_dim)

#         return output

    

# class TransformerBlock(layers.Layer):

#     def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):

#         super(TransformerBlock, self).__init__()

#         self.att = MultiHeadSelfAttention(embed_dim, num_heads)

#         self.ffn = keras.Sequential(

#             [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]

#         )

#         self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)

#         self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

#         self.dropout1 = layers.Dropout(rate)

#         self.dropout2 = layers.Dropout(rate)



#     def call(self, inputs, training):

#         attn_output = self.att(inputs)

#         attn_output = self.dropout1(attn_output, training=training)

#         out1 = self.layernorm1(inputs + attn_output)

#         ffn_output = self.ffn(out1)

#         ffn_output = self.dropout2(ffn_output, training=training)

#         return self.layernorm2(out1 + ffn_output)

    

#     def get_config(self):

#         config = super().get_config().copy()

#         return config

    

# class TokenAndPositionEmbedding(layers.Layer):

#     def __init__(self, maxlen, vocab_size, embed_dim):

#         super(TokenAndPositionEmbedding, self).__init__()

#         self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)

#         self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)



#     def call(self, x):

#         maxlen = tf.shape(x)[-1]

#         positions = tf.range(start=0, limit=maxlen, delta=1)

#         positions = self.pos_emb(positions)

#         x = self.token_emb(x)

#         return x + positions





# FOLDS = KFold(n_splits=5, random_state=42, shuffle=True)



# oofs_pred = np.zeros_like(train_labels)

# public_preds_array = []

# private_preds_array = []

# result_array1 = []

# result_array2 = []



# for i, (trn_idx, vld_idx) in enumerate(FOLDS.split(train_inputs)):

#     trn_inputs = train_inputs[trn_idx]

#     vld_inputs = train_inputs[vld_idx]

    

#     trn_inputs_bpps = train_bpps[trn_idx]

#     vld_inputs_bpps = train_bpps[vld_idx]

    

#     trn_inputs_nums = train_nums[trn_idx]

#     vld_inputs_nums = train_nums[vld_idx]

    

#     trn_inputs_adjs = train_As[trn_idx]

#     vld_inputs_adjs = train_As[vld_idx]



#     trn_labels = train_labels[trn_idx]

#     vld_labels = train_labels[vld_idx]

    

#     vld_labels_orig = vld_labels.copy()

    

#     model = build_model()

#     model.summary()



#     history = model.fit(

#         [trn_inputs, trn_inputs_bpps, trn_inputs_nums, trn_inputs_adjs], [trn_labels, trn_labels], 

#         validation_data=([vld_inputs, vld_inputs_bpps, vld_inputs_nums, vld_inputs_adjs], [vld_labels, vld_labels]),

#         batch_size=32,

#         epochs=120,

#         callbacks=[

#             tf.keras.callbacks.ReduceLROnPlateau(),

#             tf.keras.callbacks.ModelCheckpoint(f'{NAME}_1.h5')

#         ],

#         verbose=2,

#     )

    

#     model.load_weights(f'{NAME}_1.h5')

#     outputs_1, outputs_ = model.predict([vld_inputs, vld_inputs_bpps, vld_inputs_nums, vld_inputs_adjs])

    

# #     trn_inputs = trn_inputs[:, ::-1, :]

# #     trn_inputs_bpps = trn_inputs_bpps[:, ::-1, :]

# #     trn_labels = trn_labels[:, ::-1, :]

    

# #     vld_inputs = vld_inputs[:, ::-1, :]

# #     vld_inputs_bpps = vld_inputs_bpps[:, ::-1, :]

# #     vld_labels = vld_labels[:, ::-1, :]

    

    

# #     model2 = build_model(reverse=True)

# #     model2.summary()



# #     history = model2.fit(

# #         [trn_inputs, trn_inputs_bpps], trn_labels, 

# #         validation_data=([vld_inputs, vld_inputs_bpps], vld_labels),

# #         batch_size=32,

# #         epochs=120,

# #         callbacks=[

# #             tf.keras.callbacks.ReduceLROnPlateau(),

# #             tf.keras.callbacks.ModelCheckpoint(f'{NAME}_2.h5')

# #         ],

# #         verbose=2,

# #     )

    

# #     model.load_weights(f'{NAME}_2.h5')

# #     outputs_2, outputs_ =  model2.predict([vld_inputs, vld_inputs_bpps])

# #     #trn_inputs = np.concatenate([trn_inputs, trn_inputs[:, ::-1, :]], axis=0)

#     outputs = outputs_1#0.5 * (outputs_1 + outputs_2[:, ::-1, :])

#     oofs_pred[vld_idx] = outputs

    

#     from sklearn.metrics import mean_squared_error

#     errors = []

#     for idx in range(5):

#          errors.append(np.sqrt(mean_squared_error(vld_labels_orig[:, idx], outputs[:, idx])))

#     final_error = np.mean(errors)

#     print('#'*20, final_error)

    

#     result_array1.append(final_error)

    

#     oofs_ls = []

#     for idx in vld_idx:

#         single_pred = oofs_pred[idx]

#         single_df = pd.DataFrame(single_pred, columns=pred_cols)

#         oofs_ls.append(single_df)

#     oofs_df = pd.concat(oofs_ls).values

    

#     target_ls = []

#     for idx in range(len(vld_labels)):

#         single_pred = vld_labels_orig[idx]

#         single_df = pd.DataFrame(single_pred, columns=pred_cols)

#         target_ls.append(single_df)

#     target_df = pd.concat(target_ls).values

    

#     from sklearn.metrics import mean_squared_error

#     errors = []

#     for idx in range(5):

#          errors.append(np.sqrt(mean_squared_error(target_df[:, idx], oofs_df[:, idx])))

#     final_error = np.mean(errors)

#     print('#'*20, final_error)

    

#     result_array2.append(final_error)

    

    



#     public_df = test.query("seq_length == 107").copy()

#     private_df = test.query("seq_length == 130").copy()

    

#     public_df['sequence'] = public_df['sequence'].apply(lambda x: [token2int0[ele] for ele in x])

#     public_df['structure'] = public_df['structure'].apply(lambda x: [token2int1[ele] for ele in x])

#     public_df['predicted_loop_type'] = public_df['predicted_loop_type'].apply(lambda x: [token2int2[ele] for ele in x])

#     public_inputs = np.transpose(np.array(public_df[['sequence', 'structure', 'predicted_loop_type']].values.tolist()), (0, 2, 1))



#     private_df['sequence'] = private_df['sequence'].apply(lambda x: [token2int0[ele] for ele in x])

#     private_df['structure'] = private_df['structure'].apply(lambda x: [token2int1[ele] for ele in x])

#     private_df['predicted_loop_type'] = private_df['predicted_loop_type'].apply(lambda x: [token2int2[ele] for ele in x])

#     private_inputs = np.transpose(np.array(private_df[['sequence', 'structure', 'predicted_loop_type']].values.tolist()), (0, 2, 1))



#     public_bpps1 = np.stack([1 - np.load(f'../input/bpps/{ele}.npy').sum(1) for ele in public_df['id']])

#     public_bpps1 = public_bpps1[:, :, np.newaxis]



#     public_bpps2 = np.stack([1 - np.load(f'../input/bpps_nupack/{ele}.npy').sum(1) for ele in public_df['id']])

#     public_bpps2 = public_bpps2[:, :, np.newaxis]



#     public_bpps3 = np.stack([1 - np.load(f'../input/bpps_contrafold//{ele}.npy').sum(1) for ele in public_df['id']])

#     public_bpps3 = public_bpps3[:, :, np.newaxis]



#     public_bpps4 = np.stack([1 - np.load(f'../input/bpps_contrafold_linear//{ele}.npy').sum(1) for ele in public_df['id']])

#     public_bpps4 = public_bpps4[:, :, np.newaxis]



#     public_bpps5 = np.stack([1 - np.load(f'../input/bpps_eternafold/{ele}.npy').sum(1) for ele in public_df['id']])

#     public_bpps5 = public_bpps5[:, :, np.newaxis]



#     public_bpps6 = np.stack([1 - np.load(f'../input/bpps_vienna_linear/{ele}.npy').sum(1) for ele in public_df['id']])

#     public_bpps6 = public_bpps6[:, :, np.newaxis]



#     public_bpps = np.concatenate([public_bpps1, public_bpps2, public_bpps3, public_bpps4, public_bpps5, public_bpps6], axis=-1)

    

#     private_bpps1 = np.stack([1 - np.load(f'../input/bpps/{ele}.npy').sum(1) for ele in private_df['id']])

#     private_bpps1 = private_bpps1[:, :, np.newaxis]



#     private_bpps2 = np.stack([1 - np.load(f'../input/bpps_nupack/{ele}.npy').sum(1) for ele in private_df['id']])

#     private_bpps2 = private_bpps2[:, :, np.newaxis]



#     private_bpps3 = np.stack([1 - np.load(f'../input/bpps_contrafold//{ele}.npy').sum(1) for ele in private_df['id']])

#     private_bpps3 = private_bpps3[:, :, np.newaxis]



#     private_bpps4 = np.stack([1 - np.load(f'../input/bpps_contrafold_linear//{ele}.npy').sum(1) for ele in private_df['id']])

#     private_bpps4 = private_bpps4[:, :, np.newaxis]



#     private_bpps5 = np.stack([1 - np.load(f'../input/bpps_eternafold/{ele}.npy').sum(1) for ele in private_df['id']])

#     private_bpps5 = private_bpps5[:, :, np.newaxis]



#     private_bpps6 = np.stack([1 - np.load(f'../input/bpps_vienna_linear/{ele}.npy').sum(1) for ele in private_df['id']])

#     private_bpps6 = private_bpps6[:, :, np.newaxis]



#     private_bpps = np.concatenate([private_bpps1, private_bpps2, private_bpps3, private_bpps4, private_bpps5, private_bpps6], axis=-1)



    

#     public_nums = np.load('../input/public_nums.npy')

#     private_nums = np.load('../input/private_nums.npy')

    

#     public_adjs = np.load('../input/public_As.npy')

#     private_adjs = np.load('../input/private_As.npy')

    

    

#     # Caveat: The prediction format requires the output to be the same length as the input,

#     # although it's not the case for the training data.

#     model_short = build_model(seq_len=107, pred_len=107)

#     model_long = build_model(seq_len=130, pred_len=130)

    

#     model_short2 = build_model(seq_len=107, pred_len=107, reverse=True)

#     model_long2 = build_model(seq_len=130, pred_len=130, reverse=True)



#     model_short.load_weights(f'{NAME}_1.h5')

#     model_long.load_weights(f'{NAME}_1.h5')



#     public_preds_1, outputs2 = model_short.predict([public_inputs, public_bpps, public_nums, public_adjs])

#     private_preds_1, outputs2 = model_long.predict([private_inputs, private_bpps, private_nums, private_adjs])

    

    

# #     model_short2.load_weights(f'{NAME}_2.h5')

# #     model_long2.load_weights(f'{NAME}_2.h5')



# #     public_inputs = public_inputs[:, ::-1, :]

# #     public_bpps = public_bpps[:, ::-1, :]

    

# #     private_inputs = private_inputs[:, ::-1, :]

# #     private_bpps = private_bpps[:, ::-1, :]

    

# #     public_preds_2, outputs2 = model_short2.predict([public_inputs, public_bpps])

# #     private_preds_2, outputs2 = model_long2.predict([private_inputs, private_bpps])

    

    

#     public_preds = public_preds_1# 0.5 * (public_preds_1 + public_preds_2[:, ::-1, :])

#     private_preds = private_preds_1#0.5 * (private_preds_1 + private_preds_2[:, ::-1, :])

    

#     public_preds_array.append(public_preds)

#     private_preds_array.append(private_preds)



#     print(public_preds.shape, private_preds.shape)



#     preds_ls = []



#     for df, preds in [(public_df, public_preds), (private_df, private_preds)]:

#         for idx, uid in enumerate(df.id):

#             single_pred = preds[idx]



#             single_df = pd.DataFrame(single_pred, columns=pred_cols)

#             single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]



#             preds_ls.append(single_df)



#     preds_df = pd.concat(preds_ls)



#     submission = sample_df[['id_seqpos']].merge(preds_df, on=['id_seqpos'])

#     submission.to_csv(f'{NAME}_{i}.csv', index=False)

    

# print(result_array1)

# print(result_array2)















# trn_inputs_bpps.shape



# trn_inputs_nums.shape



# # - [0.3643929474422003, 0.35387031483746306, 0.35421918628471893, 0.3625872161452601, 0.34353836209736854]

# # - [0.22639929831042677, 0.2168258249933515, 0.23131894944245507, 0.2370713470214479, 0.2271213424723008]

# # - 0.2278776791567095





# oofs_ls = []

# for idx in range(len(oofs_pred)):

#     single_pred = oofs_pred[idx]

#     single_df = pd.DataFrame(single_pred, columns=pred_cols)

#     oofs_ls.append(single_df)

# oofs_df = pd.concat(oofs_ls).values



# target_ls = []

# for idx in range(len(train_labels)):

#     single_pred = train_labels[idx]

#     single_df = pd.DataFrame(single_pred, columns=pred_cols)

#     target_ls.append(single_df)

# target_df = pd.concat(target_ls).values



# from sklearn.metrics import mean_squared_error

# errors = []

# for idx in range(5):

#      errors.append(np.sqrt(mean_squared_error(target_df[:, idx], oofs_df[:, idx])))

# final_error = np.mean(errors)

# print('#'*20, final_error)



# train_clean = train[train.signal_to_noise > 1].reset_index(drop=True)





# preds_ls = []

# for idx, uid in enumerate(train_clean.id):

#     single_pred = oofs_pred[idx]



#     single_df = pd.DataFrame(single_pred, columns=pred_cols)

#     single_df['id_seqpos'] = [f'{uid}_{x}' for x in range(single_df.shape[0])]

#     preds_ls.append(single_df)

# preds_df = pd.concat(preds_ls)



# preds_df.to_csv(f'../src_oofs/oofs.{NAME}.csv', index=False)



# sub1 = pd.read_csv(f'./{NAME}_0.csv')

# sub2 = pd.read_csv(f'./{NAME}_1.csv')

# sub3 = pd.read_csv(f'./{NAME}_2.csv')

# sub4 = pd.read_csv(f'./{NAME}_3.csv')

# sub5 = pd.read_csv(f'./{NAME}_4.csv')



# new_sub = sub1.copy()



# new_sub[pred_cols] = (1/5) * (sub1[pred_cols].values + sub2[pred_cols].values + sub3[pred_cols].values + sub4[pred_cols].values + sub5[pred_cols].values)



# new_sub.to_csv(f'./{NAME}_5fold.csv', index=False)



# NAME


