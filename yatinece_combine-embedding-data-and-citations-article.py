

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

import scipy as sp
#title_data=pd.read_csv('/kaggle/input/covid-pickle-csv/title.csv')

title_data_pkl=pd.read_pickle('/kaggle/input/covid-pickle-csv/title.pkl')

import scipy.spatial

val_k=100

base=pd.DataFrame()

for k in range(title_data_pkl.shape[0]//val_k):

    distances=scipy.spatial.distance.cdist(title_data_pkl[val_k*k:val_k*(k+1)], title_data_pkl, "cosine")

    base=base.append(pd.DataFrame(np.argwhere(1-distances > 0.9)))

    print("----" + str(k) + "----")

base.to_pickle('result.pkl')
#title_data=pd.read_csv('/kaggle/input/covid-pickle-csv/title.csv')

title_data_pkl=pd.read_pickle('/kaggle/input/covid-pickle-csv/title_abstract.pkl')

import scipy.spatial

val_k=100

base=pd.DataFrame()

for k in range(title_data_pkl.shape[0]//val_k):

    distances=scipy.spatial.distance.cdist(title_data_pkl[val_k*k:val_k*(k+1)], title_data_pkl, "cosine")

    base=base.append(pd.DataFrame(np.argwhere(1-distances > 0.9)))

    print("----" + str(k) + "----")

base.to_pickle('result_title_abstract.pkl')
# document_mapping={}

# for i in range(len(title_data_pkl)):

#     distances = sp.spatial.distance.cdist([title_data_pkl[i]], title_data_pkl, "cosine")

#     for j in range(distances.shape[1]):

#         if i!=j:

#             val=distances[0][j]

#             val=1-val

#             if val>=0.7:

#                 if j % 1000==0:

#                     print(j)

#                 if i in document_mapping:

#                     document_mapping[i].append((j,val))

#                 else:

#                     document_mapping[i]=[]

#                     document_mapping[i].append((j,val))
# np.array(title_data_pkl).shape

# title_data_pkl = pd.DataFrame(title_data_pkl)







# #also can output sparse matrices

# similarities_sparse = cosine_similarity(A_sparse,dense_output=False)

#print('pairwise sparse output:\n {}\n'.format(similarities_sparse))


# from sklearn.metrics.pairwise import cosine_similarity

# title_data_pkl= cosine_similarity(title_data_pkl, title_data_pkl)

# print(title_data_pkl.shape)

# idx = np.argwhere(title_data_pkl > 0.01) 
# idx.shape
# from sklearn.metrics.pairwise import cosine_similarity

# from scipy import sparse

# title_data_pkl = sparse.csr_matrix(title_data_pkl)
# import scipy.sparse as sp

# #del(A_sparse)

# B = title_data_pkl > 0.8 #the condition

# indexes_list = sp.find(B)
# similarities_sparse
# del(similarities_sparse)

# import gc

# gc.collect()
# np.where(df_new>0.8,1,0).sum()
os.listdir('/kaggle/input/data-json/')

def _eval_f(x):

    try:

        return ast.literal_eval(x)

    except:

        return x
location_for_ref_data='/kaggle/input/data-json/'

all_ref=pd.read_pickle(location_for_ref_data+'get_ref_text_df.pkl')

all_ref['title']=all_ref['title'].str.lower().apply(lambda x: _eval_f(x))

all_ref['ref_title']=all_ref['ref_title'].str.lower().apply(lambda x: _eval_f(x)).fillna('na')

all_ref=all_ref[all_ref['title'].apply(lambda x:len(x))>5]

all_ref=all_ref[all_ref['ref_title'].apply(lambda x:len(str(x)))>5]
all_ref_1=all_ref[['title','ref_title','paper_id']]

all_ref_1['ref_title']=all_ref_1['title']

all_ref_all=all_ref.append(all_ref_1.drop_duplicates(),sort=False)
int_lt=list(set(all_ref['title'].str.lower().unique().tolist()) & set(all_ref['ref_title'].str.lower().unique().tolist()))

print("Citations present in the data " + str(len(int_lt)))
all_ref_all_new=all_ref_all[['ref_title','paper_id','title']]

all_ref_all_new=all_ref_all_new[all_ref_all_new['ref_title'].isin(int_lt)]

print("Paper id missing for all citations the data =" + str(all_ref_all_new.paper_id.isna().sum()))
# title_abstract_data=pd.read_csv('/kaggle/input/covid-pickle-csv/title_abstract.csv')

# title_abstract_data_pkl=pd.read_pickle('/kaggle/input/covid-pickle-csv/title_abstract.pkl')

# title_abstract_data
# title_abstract_data_pkl.shape
self_join=all_ref_all_new.merge(all_ref_all_new,on='paper_id')

self_join=self_join

self_join.columns

print(self_join.ref_title_y.unique().shape[0])

print(self_join.ref_title_x.unique().shape[0])

all_=self_join.ref_title_x.unique()

# creating dict 

node_lookup={}

for k in range(all_.shape[0]):

    node_lookup[all_[k]]=k

node_lookup_rev={}

for k in range(all_.shape[0]):

    node_lookup_rev[k]=all_[k]

#check

self_join[self_join['ref_title_y']=='molecular advances in severe acute respiratory syndrome-associated coronavirus (sars-cov)']
self_join['node_1']=self_join['ref_title_x'].apply(lambda x: node_lookup[x])

self_join['node_2']=self_join['ref_title_y'].apply(lambda x: node_lookup[x])

self_join
self_join.to_pickle('res_ref.pkl')