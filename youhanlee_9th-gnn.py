# #!/usr/bin/env python

# # coding: utf-8



# # In[1]:





# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152

# os.environ["CUDA_VISIBLE_DEVICES"]="0"

# import math

# import torch.nn as nn

# # fix seed ----------------------------------------------------------------------

# import torch

# import numpy as np

# import random

# import copy

# from tqdm.notebook import tqdm



# NAME = 'final_gnn_genconv_815'



# seed = 815

# def seed_torch(seed):

#     torch.manual_seed(seed)

#     torch.cuda.manual_seed_all(seed)

#     return seed



# def seed_py(seed):

#     random.seed(seed)

#     np.random.seed(seed)

#     return seed



# seed_torch(seed)

# seed_py(seed)

# torch.backends.cudnn.deterministic = True

# torch.backends.cudnn.benchmark = False



# print('!!!!! seed=%d'%seed)

# #---------------------------------------------------------------------------------

# ## https://www.kaggle.com/symyksr/openvaccine-deepergcn #########################





# from torch.nn import Linear, LayerNorm, ReLU, Dropout

# from torch_geometric.nn import ChebConv, NNConv, DeepGCNLayer, ARMAConv, ClusterGCNConv, GENConv

# from torch_geometric.data import Data, DataLoader

# from sklearn.model_selection import StratifiedKFold



# import os

# import pandas as pd

# #---------------------------------------------------------------------------------



# data_dir = '/kaggle/input/stanford-covid-vaccine/'

# train_file = data_dir+'/train.json'

# test_file  = data_dir+'/test.json'

# bpps_top   = data_dir+'/bpps'





# # settings



# train_with_noisy_data     = True

# add_edge_for_paired_nodes = True

# add_codon_nodes           = True



# bpps_nb_mean = 0.077522 # mean of bpps_nb across all training data

# bpps_nb_std  = 0.08914   # std of bpps_nb across all training data

# error_mean_limit = 0.5



# nb_fold    = 5

# device     = 'cuda'

# batch_size = 16

# epochs     = 200

# lr         = 0.001

# T = 5

# node_hidden_channels = 144

# edge_hidden_channels = 144

# hidden_channels3 = 144

# num_layers = 10

# dropout1 = 0.1

# dropout2 = 0.1

# dropout3 = 0.1



# ##################### all data preparation ####################################



# def match_pair(structure):

#     pair = [-1] * len(structure)

#     pair_no = -1



#     pair_no_stack = []

#     for i, c in enumerate(structure):

#         if c == '(':

#             pair_no += 1

#             pair[i] = pair_no

#             pair_no_stack.append(pair_no)

#         elif c == ')':

#             pair[i] = pair_no_stack.pop()

#     return pair



# class MyData(Data):

#     def __init__(self, x=None, edge_index=None, edge_attr=None, y=None,

#                  pos=None, norm=None, face=None, weight=None, **kwargs):

#         super(MyData, self).__init__(x=x, edge_index=edge_index,

#                                      edge_attr=edge_attr, y=y, pos=pos,

#                                      norm=norm, face=face, **kwargs)

#         self.weight = weight





# def calc_error_mean(row):

#     reactivity_error = row['reactivity_error']

#     deg_error_Mg_pH10 = row['deg_error_Mg_pH10']

#     deg_error_Mg_50C = row['deg_error_Mg_50C']



#     return np.mean(np.abs(reactivity_error) +

#                    np.abs(deg_error_Mg_pH10) + \

#                    np.abs(deg_error_Mg_50C)) / 3





# def calc_sample_weight(row):

#     if sample_is_clean(row):

#         return 1.

#     else:

#         error_mean = calc_error_mean(row)

#         if error_mean >= error_mean_limit:

#             return 0.



#         return 1. - error_mean / error_mean_limit





# # add directed edge for node1 -> node2 and for node2 -> node1

# def add_edges(edge_index, edge_features, node1, node2, feature1, feature2):

#     edge_index.append([node1, node2])

#     edge_features.append(feature1)

#     edge_index.append([node2, node1])

#     edge_features.append(feature2)





# def add_edges_between_base_nodes(edge_index, edge_features, node1, node2):

#     edge_feature1 = [

#         0, # is edge for paired nodes

#         0, # is edge between codon node and base node

#         0, # is edge between coden nodes

#         1, # forward edge: 1, backward edge: -1

#         1, # bpps if edge is for paired nodes

#     ]

#     edge_feature2 = [

#         0, # is edge for paired nodes

#         0, # is edge between codon node and base node

#         0, # is edge between coden nodes

#         -1, # forward edge: 1, backward edge: -1

#         1, # bpps if edge is for paired nodes

#     ]

#     add_edges(edge_index, edge_features, node1, node2,

#               edge_feature1, edge_feature2)



# def add_edges_between_paired_nodes(edge_index, edge_features, node1, node2,

#                                    bpps_value):

#     edge_feature1 = [

#         1, # is edge for paired nodes

#         0, # is edge between codon node and base node

#         0, # is edge between coden nodes

#         0, # forward edge: 1, backward edge: -1

#         bpps_value, # bpps if edge is for paired nodes

#     ]

#     edge_feature2 = [

#         1, # is edge for paired nodes

#         0, # is edge between codon node and base node

#         0, # is edge between coden nodes

#         0, # forward edge: 1, backward edge: -1

#         bpps_value, # bpps if edge is for paired nodes

#     ]

#     add_edges(edge_index, edge_features, node1, node2,

#               edge_feature1, edge_feature2)



# def add_edges_between_codon_nodes(edge_index, edge_features, node1, node2):

#     edge_feature1 = [

#         0, # is edge for paired nodes

#         0, # is edge between codon node and base node

#         1, # is edge between coden nodes

#         1, # forward edge: 1, backward edge: -1

#         0, # bpps if edge is for paired nodes

#     ]

#     edge_feature2 = [

#         0, # is edge for paired nodes

#         0, # is edge between codon node and base node

#         1, # is edge between coden nodes

#         -1, # forward edge: 1, backward edge: -1

#         0, # bpps if edge is for paired nodes

#     ]

#     add_edges(edge_index, edge_features, node1, node2,

#               edge_feature1, edge_feature2)



# def add_edges_between_codon_and_base_node(edge_index, edge_features,

#                                           node1, node2):

#     edge_feature1 = [

#         0, # is edge for paired nodes

#         1, # is edge between codon node and base node

#         0, # is edge between coden nodes

#         0, # forward edge: 1, backward edge: -1

#         0, # bpps if edge is for paired nodes

#     ]

#     edge_feature2 = [

#         0, # is edge for paired nodes

#         1, # is edge between codon node and base node

#         0, # is edge between coden nodes

#         0, # forward edge: 1, backward edge: -1

#         0, # bpps if edge is for paired nodes

#     ]

#     add_edges(edge_index, edge_features, node1, node2,

#               edge_feature1, edge_feature2)



# def add_node(node_features, feature):

#     node_features.append(feature)





# def add_base_node(node_features, sequence, predicted_loop_type,

#                   bpps_sum, bpps_nb):

#     feature = [

#         0, # is codon node

#         sequence == 'A',

#         sequence == 'C',

#         sequence == 'G',

#         sequence == 'U',

#         predicted_loop_type == 'S',

#         predicted_loop_type == 'M',

#         predicted_loop_type == 'I',

#         predicted_loop_type == 'B',

#         predicted_loop_type == 'H',

#         predicted_loop_type == 'E',

#         predicted_loop_type == 'X',

#         bpps_sum,

#         bpps_nb,

#     ]

#     add_node(node_features, feature)



# def add_codon_node(node_features):

#     feature = [

#         1, # is codon node

#         0, # sequence == 'A',

#         0, # sequence == 'C',

#         0, # sequence == 'G',

#         0, # sequence == 'U',

#         0, # predicted_loop_type == 'S',

#         0, # predicted_loop_type == 'M',

#         0, # predicted_loop_type == 'I',

#         0, # predicted_loop_type == 'B',

#         0, # predicted_loop_type == 'H',

#         0, # predicted_loop_type == 'E',

#         0, # predicted_loop_type == 'X',

#         0, # bpps_sum

#         0, # bpps_nb

#     ]

#     add_node(node_features, feature)



# def build_data(df, is_train):

#     data = []

#     for i in range(len(df)):

#         targets = []

#         node_features = []

#         edge_features = []

#         edge_index = []

#         train_mask = []

#         test_mask = []

#         weights = []



#         id = df.loc[i, 'id']

#         path = os.path.join(bpps_top, id + '.npy')

#         bpps = np.load(path)

#         bpps_sum = bpps.sum(axis=0)

#         sequence = df.loc[i, 'sequence']

#         structure = df.loc[i, 'structure']

#         pair_info = match_pair(structure)

#         predicted_loop_type = df.loc[i, 'predicted_loop_type']

#         seq_length = df.loc[i, 'seq_length']

#         seq_scored = df.loc[i, 'seq_scored']

#         bpps_nb = (bpps > 0).sum(axis=0) / seq_length

#         bpps_nb = (bpps_nb - bpps_nb_mean) / bpps_nb_std

#         if is_train:

#             sample_weight = calc_sample_weight(df.loc[i])



#             reactivity = df.loc[i, 'reactivity']

#             deg_Mg_pH10 = df.loc[i, 'deg_Mg_pH10']

#             deg_Mg_50C = df.loc[i, 'deg_Mg_50C']



#             for j in range(seq_length):

#                 if j < seq_scored:

#                     targets.append([

#                         reactivity[j],

#                         deg_Mg_pH10[j],

#                         deg_Mg_50C[j],

#                         ])

#                 else:

#                     targets.append([0, 0, 0])



#         paired_nodes = {}

#         for j in range(seq_length):

#             add_base_node(node_features, sequence[j], predicted_loop_type[j],

#                           bpps_sum[j], bpps_nb[j])



#             if j + 1 < seq_length: # edge between current node and next node

#                 add_edges_between_base_nodes(edge_index, edge_features,

#                                              j, j + 1)



#             if pair_info[j] != -1:

#                 if pair_info[j] not in paired_nodes:

#                     paired_nodes[pair_info[j]] = [j]

#                 else:

#                     paired_nodes[pair_info[j]].append(j)



#             train_mask.append(j < seq_scored)

#             test_mask.append(True)

#             if is_train:

#                 weights.append(sample_weight)





#         if add_edge_for_paired_nodes:

#             for pair in paired_nodes.values():

#                 bpps_value = bpps[pair[0], pair[1]]

#                 add_edges_between_paired_nodes(edge_index, edge_features,

#                                                pair[0], pair[1], bpps_value)







#         if add_codon_nodes:

#             codon_node_idx = seq_length - 1

#             for j in range(seq_length):

#                 if j % 3 == 0:

#                     # add codon node

#                     add_codon_node(node_features)

#                     codon_node_idx += 1

#                     train_mask.append(False)

#                     test_mask.append(False)

#                     if is_train:

#                         weights.append(0)

#                         targets.append([0, 0, 0])



#                     if codon_node_idx > seq_length:

#                         # add edges between adjacent codon nodes

#                         add_edges_between_codon_nodes(edge_index, edge_features,

#                                                       codon_node_idx - 1,

#                                                       codon_node_idx)



#                 # add edges between codon node and base node

#                 add_edges_between_codon_and_base_node(edge_index, edge_features,

#                                                       j, codon_node_idx)



#         node_features = torch.tensor(node_features, dtype=torch.float)

#         edge_index    = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

#         edge_features = torch.tensor(edge_features, dtype=torch.float)



#         if is_train:

#             data.append(MyData(x=node_features, edge_index=edge_index,

#                                edge_attr=edge_features,

#                                train_mask=torch.tensor(train_mask),

#                                weight=torch.tensor(weights, dtype=torch.float),

#                                y=torch.tensor(targets, dtype=torch.float)))

#         else:

#             data.append(MyData(x=node_features, edge_index=edge_index,

#                                edge_attr=edge_features,

#                                test_mask=torch.tensor(test_mask)))



#     return data



# #---------------------------------------------------------------------------------

# def build_id_seqpos(df):

#     id_seqpos = []

#     for i in range(len(df)):

#         id = df.loc[i, 'id']

#         seq_length = df.loc[i, 'seq_length']

#         for seqpos in range(seq_length):

#             id_seqpos.append(id + '_' + str(seqpos))

#     return id_seqpos



# def sample_is_clean(row):

#     return row['SN_filter'] == 1

#     #return row['signal_to_noise'] > 1 and \

#     #       min((min(row['reactivity']),

#     #            min(row['deg_Mg_pH10']),

#     #            min(row['deg_pH10']),

#     #            min(row['deg_Mg_50C']),

#     #            min(row['deg_50C']))) > -0.5



# # categorical value for target (used for stratified kfold)

# def add_y_cat(df):

#     target_mean = df['reactivity'].apply(np.mean) +                   df['deg_Mg_pH10'].apply(np.mean) +                   df['deg_Mg_50C'].apply(np.mean)

#     df['y_cat'] = pd.qcut(np.array(target_mean), q=20).codes



# ##################### all model preparation ####################################



# # originally copied from

# # https://github.com/rusty1s/pytorch_geometric/blob/master/examples/ogbn_proteins_deepgcn.py

# #

# class MapE2NxN(torch.nn.Module):

#     def __init__(self, in_channels, out_channels, hidden_channels):

#         super(MapE2NxN, self).__init__()

#         self.linear1 = Linear(in_channels, hidden_channels)

#         self.linear2 = Linear(hidden_channels, out_channels)

#         self.dropout = Dropout(dropout3)



#     def forward(self, x):

#         x = self.linear1(x)

#         x = self.dropout(x)

#         x = self.linear2(x)

#         return x





# class PositionEncode(nn.Module):

#     def __init__(self, dim, length=174):

#         super(PositionEncode, self).__init__()

#         position = torch.zeros(length,dim)

#         p = torch.arange(0, length, dtype=torch.float).unsqueeze(1)

#         div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

#         position[:,0::2] = torch.sin(p * div)

#         position[:,1::2] = torch.cos(p * div)

#         #position = position.transpose(0, 1).reshape(1,dim,length) #.contiguous()

#         position = position.reshape(length, 1, dim) #.contiguous()

#         self.register_buffer('position', position)



#         #self.position = nn.Parameter( torch.randn(1, dim, length) ) #random



#     def forward(self, x):

#         length, batch_size, _ = x.shape

#         position = self.position.repeat(1, batch_size, 1)

#         position = position[:length, :, :, ].contiguous()

#         return position





# class MyDeeperGCN(torch.nn.Module):

#     def __init__(self, num_node_features, num_edge_features,

#                  node_hidden_channels,

#                  edge_hidden_channels,

#                  num_layers, num_classes, seq_length = 143):

#         super(MyDeeperGCN, self).__init__()



#         self.node_encoder = GENConv(num_node_features, node_hidden_channels)

#         self.edge_encoder = Linear(num_edge_features, edge_hidden_channels)



#         self.layers = torch.nn.ModuleList()

#         for i in range(1, num_layers + 1):

#             conv = NNConv(node_hidden_channels, node_hidden_channels,

#                           MapE2NxN(edge_hidden_channels,

#                                    node_hidden_channels * node_hidden_channels,

#                                    hidden_channels3))

#             norm = LayerNorm(node_hidden_channels, elementwise_affine=True)

#             act = ReLU(inplace=True)



#             layer = DeepGCNLayer(conv, norm, act, block='res+',

#                                  dropout=dropout1, ckpt_grad=i % 3)

#             self.layers.append(layer)



#         self.postition = PositionEncode(node_hidden_channels)

#         self.transformer = torch.nn.TransformerEncoder(

#             torch.nn.TransformerEncoderLayer(node_hidden_channels, 8, 256, dropout=0.2, activation='relu'),

#             2

#         )



#         self.lin = Linear(node_hidden_channels, num_classes)

#         self.dropout = Dropout(dropout2)

#         self.seq_length = seq_length



#     def forward(self, data):

#         batch_size = data.x.shape[0]//self.seq_length ##hard code





#         x = data.x

#         edge_index = data.edge_index

#         edge_attr = data.edge_attr



#         # edge for paired nodes are excluded for encoding node

#         seq_edge_index = edge_index[:, edge_attr[:,0] == 0]

#         x = self.node_encoder(x, seq_edge_index)



#         edge_attr = self.edge_encoder(edge_attr)



#         x = self.layers[0].conv(x, edge_index, edge_attr)



#         for layer in self.layers[1:]:

#             x = layer(x, edge_index, edge_attr)



#         x = self.layers[0].act(self.layers[0].norm(x))

#         x = self.dropout(x)



#         #----

#         #hard code

#         x = x.reshape(batch_size, -1, node_hidden_channels)

#         x = x.permute(1,0,2).contiguous() #length, batch_size, 128

        

#         if 1:

#             pos = self.postition(x)

#             x = self.transformer(x+pos)



#         #----

#         predict = self.lin(x)

#         predict = predict.permute(1,0,2).contiguous()

#         predict = predict.reshape(-1, 3)  #to keep compatible with rest of code

#         return predict



# def weighted_mse_loss(prds, tgts, weight):

#     return torch.mean(weight * (prds - tgts)**2)



# def criterion(prds, tgts, weight=None):

#     if weight is None:

#         return (torch.sqrt(torch.nn.MSELoss()(prds[:,0], tgts[:,0])) +

#                 torch.sqrt(torch.nn.MSELoss()(prds[:,1], tgts[:,1])) +

#                 torch.sqrt(torch.nn.MSELoss()(prds[:,2], tgts[:,2]))) / 3

#     else:

#         return (torch.sqrt(weighted_mse_loss(prds[:,0], tgts[:,0], weight)) +

#                 torch.sqrt(weighted_mse_loss(prds[:,1], tgts[:,1], weight)) +

#                 torch.sqrt(weighted_mse_loss(prds[:,2], tgts[:,2], weight))) / 3





# # In[ ]:











# # In[2]:





# print('Reading', train_file)

# df_tr = pd.read_json(train_file, lines=True)

# add_y_cat(df_tr)



# is_clean = df_tr.apply(sample_is_clean, axis=1)

# df_clean = df_tr[is_clean].reset_index(drop=True)

# df_noisy = df_tr[is_clean==False].reset_index(drop=True)

# del df_tr



# #------------------------------------------------------------

# print('Training')

# all_ys = torch.zeros((0, 3)).to(device).detach()

# all_outs = torch.zeros((0, 3)).to(device).detach()

# best_model_states = []

# kf = StratifiedKFold(nb_fold, shuffle=True, random_state=seed)

# for fold, ((clean_train_idx, clean_valid_idx),

#            (noisy_train_idx, noisy_valid_idx)) \

#                in enumerate(zip(kf.split(df_clean, df_clean['y_cat']),

#                                 kf.split(df_noisy, df_noisy['y_cat']))):

#     print('Fold', fold)

#     #epochs = 5

#     #start_timer = timer()



#     out_dir = f'./{NAME}/sn1-fold-{fold}'#%fold

#     initial_checkpoint =         None #out_dir + '/checkpoint/00007200_model.pth' #





#     ## setup  ----------------------------------------

#     for f in ['checkpoint',] : os.makedirs(out_dir +'/'+f, exist_ok=True)

# #     log = Logger()

# #     log.open(out_dir+'/log.train.txt',mode='a')

# #     log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))

# #     log.write('\t%s\n' % COMMON_STRING)

# #     log.write('\t__file__ = %s\n' % __file__)

# #     log.write('\tout_dir  = %s\n' % out_dir)

# #     log.write('\n')







#     #----------------------------------------------------------------------------

#     # build train data

#     df_train = df_clean.loc[clean_train_idx]

#     if train_with_noisy_data:

#         df_train_noisy = df_noisy.loc[noisy_train_idx]

#         df_train_noisy =            df_train_noisy[df_train_noisy.apply(calc_error_mean, axis=1) <=                           error_mean_limit]

#         df_train = df_train.append(df_train_noisy)

#     data_train = build_data(df_train.reset_index(drop=True), True)

#     del df_train

#     loader_train = DataLoader(data_train, batch_size=batch_size,

#                               shuffle=True)



#     # build validation data

#     df_valid_clean = df_clean.loc[clean_valid_idx].reset_index(drop=True)

#     data_valid_clean = build_data(df_valid_clean, True)

#     del df_valid_clean

#     loader_valid_clean = DataLoader(data_valid_clean, batch_size=batch_size,

#                                     shuffle=False)



#     model = MyDeeperGCN(data_train[0].num_node_features,

#                         data_train[0].num_edge_features,

#                         node_hidden_channels=node_hidden_channels,

#                         edge_hidden_channels=edge_hidden_channels,

#                         num_layers=num_layers,

#                         num_classes=3, seq_length=143).to(device)







#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

#     best_mcrmse = np.inf

#     for epoch in tqdm(range(epochs)):

#         #print('Epoch', epoch)

#         model.train()

#         train_loss = 0.0

#         nb = 0

#         for data in (loader_train):

#             data = data.to(device)

#             mask = data.train_mask

#             weight = data.weight[mask]



#             optimizer.zero_grad()

#             out = model(data)[mask]

#             y = data.y[mask]

#             loss = criterion(out, y, weight)

#             loss.backward()

#             optimizer.step()

#             train_loss += loss.item() * y.size(0)

#             nb += y.size(0)



#             del data

#             del out

#             del y

#             del loss

#             #gc.collect()

#             #torch.cuda.empty_cache()

#         train_loss /= nb



#         model.eval()

#         valid_loss = 0.0

#         nb = 0

#         ys = torch.zeros((0, 3)).to(device).detach()

#         outs = torch.zeros((0, 3)).to(device).detach()

#         for data in (loader_valid_clean):

#             data = data.to(device)

#             mask = data.train_mask



#             out = model(data)[mask].detach()

#             y = data.y[mask].detach()

#             loss = criterion(out, y).detach()

#             valid_loss += loss.item() * y.size(0)

#             nb += y.size(0)



#             outs = torch.cat((outs, out), dim=0)

#             ys = torch.cat((ys, y), dim=0)



#             del data

#             del out

#             del y

#             del loss

#             #gc.collect()

#             #torch.cuda.empty_cache()

#         valid_loss /= nb



#         mcrmse = criterion(outs, ys).item()

#         print('Epoch %3d |  T Loss: %0.5f  |  V Loss: %0.5f  |  V MCRMSE: %0.5f | %s \n'%(

#             epoch, train_loss, valid_loss, mcrmse, 'min'))

        

#         scheduler.step(mcrmse)

        

#         if mcrmse < best_mcrmse:

#             #print('Best valid MCRMSE updated to', mcrmse)

#             best_mcrmse = mcrmse

#             best_model_state = copy.deepcopy(model.state_dict())



#         # --------------------------------------------------

#         if int(epoch)%5 ==0:

#             torch.save({'state_dict': model.state_dict(),}, out_dir + '/checkpoint/%08d_model.pth' % (int(epoch)))

#     print('#'*20, best_mcrmse)

#     del data_train

#     del data_valid_clean

#     #gc.collect()

#     #torch.cuda.empty_cache()







#     #--------------------------------------------------

#     best_model_states.append(best_model_state)



#     # predict for CV

#     model.load_state_dict(best_model_state)

#     model.eval()

#     for data in loader_valid_clean:

#         data = data.to(device)

#         mask = data.train_mask



#         out = model(data)[mask].detach()

#         y = data.y[mask].detach()



#         all_ys = torch.cat((all_ys, y), dim=0)

#         all_outs = torch.cat((all_outs, out), dim=0)



#         del data

#         del out

#         del y

#         #gc.collect()

#         #torch.cuda.empty_cache()



# # calculate MCRMSE by all training data

# print('CV MCRMSE ', criterion(all_outs, all_ys).item())

# del all_outs

# del all_ys

# #gc.collect()

# #torch.cuda.empty_cache()





# # In[3]:





# # predict for test data

# print('Predicting test data')

# print('Reading', test_file)

# df_te = pd.read_json(test_file, lines=True)



# df_te = df_te.query('seq_length==107').reset_index(drop=True)



# data_test = build_data(df_te, False)

# loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

# id_seqpos = build_id_seqpos(df_te)



# model = MyDeeperGCN(data_test[0].num_node_features,

#                     data_test[0].num_edge_features,

#                     node_hidden_channels=node_hidden_channels,

#                     edge_hidden_channels=edge_hidden_channels,

#                     num_layers=num_layers,

#                     num_classes=3, seq_length=143).to(device)



# preds = torch.zeros((len(id_seqpos), 3)).to(device).detach()

# for best_model_state in best_model_states:

#     model.load_state_dict(best_model_state)

#     model.eval()



#     outs = torch.zeros((0, 3)).to(device).detach()

#     for data in loader_test:

#         data = data.to(device)

#         mask = data.test_mask



#         out = model(data)[mask].detach()

#         outs = torch.cat((outs, out), dim=0)



#         del data

#         del out

#         #gc.collect()

#         #torch.cuda.empty_cache()

#     preds += outs

# preds /= len(best_model_states)

# preds = preds.cpu().numpy()



# df_sub_pub = pd.DataFrame({'id_seqpos': id_seqpos,

#                        'reactivity': preds[:,0],

#                        'deg_Mg_pH10': preds[:,1],

#                        'deg_pH10': 0,

#                        'deg_Mg_50C': preds[:,2],

#                        'deg_50C': 0})

# print('Writing submission.csv')





# # In[4]:





# # predict for test data

# print('Predicting test data')

# print('Reading', test_file)

# df_te = pd.read_json(test_file, lines=True)



# df_te = df_te.query('seq_length==130').reset_index(drop=True)



# data_test = build_data(df_te, False)

# loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=False)

# id_seqpos = build_id_seqpos(df_te)



# model = MyDeeperGCN(data_test[0].num_node_features,

#                     data_test[0].num_edge_features,

#                     node_hidden_channels=node_hidden_channels,

#                     edge_hidden_channels=edge_hidden_channels,

#                     num_layers=num_layers,

#                     num_classes=3, seq_length=174).to(device)



# preds = torch.zeros((len(id_seqpos), 3)).to(device).detach()

# for best_model_state in best_model_states:

#     model.load_state_dict(best_model_state)

#     model.eval()



#     outs = torch.zeros((0, 3)).to(device).detach()

#     for data in loader_test:

#         data = data.to(device)

#         mask = data.test_mask



#         out = model(data)[mask].detach()

#         outs = torch.cat((outs, out), dim=0)



#         del data

#         del out

#         #gc.collect()

#         #torch.cuda.empty_cache()

#     preds += outs

# preds /= len(best_model_states)

# preds = preds.cpu().numpy()



# df_sub_pri = pd.DataFrame({'id_seqpos': id_seqpos,

#                        'reactivity': preds[:,0],

#                        'deg_Mg_pH10': preds[:,1],

#                        'deg_pH10': 0,

#                        'deg_Mg_50C': preds[:,2],

#                        'deg_50C': 0})

# print('Writing submission.csv')





# # In[5]:





# df_sub = pd.concat([df_sub_pub, df_sub_pri]).reset_index(drop=True).sort_values('id_seqpos').reset_index(drop=True) 





# # In[6]:





# best = pd.read_csv('ensemble_v37.csv').reset_index(drop=True).sort_values('id_seqpos').reset_index(drop=True) 



# pred_cols = ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']



# print(np.corrcoef(df_sub[pred_cols[0]], best[pred_cols[0]]))

# print(np.corrcoef(df_sub[pred_cols[1]], best[pred_cols[1]]))

# print(np.corrcoef(df_sub[pred_cols[3]], best[pred_cols[3]]))



# df_sub.to_csv(f'{NAME}.csv', index=False)