!pip uninstall tensorflow==2.1.0 --yes

!pip install bert-serving-server

!pip install bert-serving-client

!pip install --upgrade ipykernel

!pip install tensorflow==1.13.1
!wget https://github.com/naver/biobert-pretrained/releases/download/v1.1-pubmed/biobert_v1.1_pubmed.tar.gz
!tar xvzf biobert_v1.1_pubmed.tar.gz

%cd biobert_v1.1_pubmed

!rename 's/model.ckpt-1000000.data-00000-of-00001/bert_model.ckpt.data-00000-of-00001/' *

!rename 's/model.ckpt-1000000.meta/bert_model.ckpt.meta/' *

!rename 's/model.ckpt-1000000.index/bert_model.ckpt.index/' *

!ls #/kaggle/working/biobert_v1.1_pubmed

#!port_num=5555
import tensorflow as tf

print(tf.__version__)
import pandas as pd

from bert_serving.client import BertClient

import numpy as np

from bert_serving.server.helper import get_args_parser

from bert_serving.server import BertServer
a = get_args_parser().parse_args(['-model_dir', '/kaggle/working/biobert_v1.1_pubmed',

                                     '-port', '5555',

                                     '-port_out', '5556',

                                     '-max_seq_len', 'NONE',

                                     '-mask_cls_sep',

                                     '-cpu',

                                     '-num_worker','4'])

server = BertServer(a)

server.start()
bc = BertClient(port=5555, port_out=5556)
biorx_df = pd.read_csv('../../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv')

biorx_lst = biorx_df['title'].astype(str).to_list()

print(biorx_lst)
doc_vecs = bc.encode(biorx_lst)

print(doc_vecs.shape)
def find_similar_articles(query,topk):

    query_vec = bc.encode([query])[0]

    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)

    topk_idx = np.argsort(score)[::-1][:topk]

    for idx in topk_idx:

        print('> %s\t%s' % (score[idx], biorx_lst[idx]))

find_similar_articles("smoking or pre-existing pulmonary disease increase risk of COVID-19",5)
find_similar_articles("Socio-economic and behavioral factors to understand the economic impact of the coronavirus and whether there were differences.",5)
find_similar_articles("Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities",5)
find_similar_articles("Risk of of COVID-19 for neonates and pregnant women",5)
find_similar_articles(" Potential risk factors of Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors",5)
find_similar_articles("What is the severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups?",5)
find_similar_articles("Susceptibility of populations",5)
find_similar_articles("Public health mitigation measures that could be effective for control",5)
!bert-serving-terminate -port 5555