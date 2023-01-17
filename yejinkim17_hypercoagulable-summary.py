!git clone https://github.com/yejinjkim/auto-review.git
import pandas as pd
import pickle
import numpy as np
import os
#path information
task='task1'# or 'task2' # specify task
root_path='auto-review/'
data_path=root_path+'data/'+task+'/'
literature_file_name='metadata_hypercoagulable.tsv'
hypercoagulable=pd.read_csv(data_path+literature_file_name, sep='\t').drop(columns='Unnamed: 0')
hypercoagulable.head()
from IPython.display import IFrame
IFrame(src=root_path+'etc/corpus_6_16.html', width = 1200, height=700)
keyword_path=data_path+'keywords/'#keyword list
pseudo_label_file_name='pseudo_label.pkl'
save_path = save_path=root_path+'results/'+task+'/'
def load_keyword(keyword_file_name):
    with open(keyword_path+keyword_file_name, "r") as f:
        keylist=f.read().split(',')
    return keylist

keywordslist=load_keyword('keywords.txt')
viruslist=load_keyword('viruslist.txt')
triallist=load_keyword('triallist.txt')
mustlist=['hypercoagulable']
keywordslist
ranking=pickle.load(open(save_path+pseudo_label_file_name, 'rb'))
ranking.head()
ranking['prob'].hist()
ranking_file_name='pseudo_label.pkl'
question_file_name='questions_structured.csv'
literature_file_name='metadata_hypercoagulable.tsv'
answer_confidence_threshold=1
top_k=300 # top_k articles
col='abstract'
#ranking=pickle.load(open(save_path+pseudo_label_file_name, 'rb'))
ranking=ranking.loc[ranking['label']==1]
ranking_top_k=ranking.iloc[:top_k,:]
ranking_top_k
questions=pd.read_csv(data_path+question_file_name, header=None, names=['type','question'], )
questions
summary=pd.read_csv(save_path+'summary.csv',header=[0,1], index_col=0)
summary
summary.to_csv('summary.csv')
