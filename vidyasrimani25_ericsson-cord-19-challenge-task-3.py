from IPython.display import Image

Image("/kaggle/input/task3-data/Images/Images/Cover2.PNG")

!pip install bert-extractive-summarizer
#!pip install nxviz
!pip install -U sentence-transformers
import pandas as pd
import os
import numpy as np
from tqdm.notebook import tqdm
import scipy


import textwrap
import json
import logging
import pickle
import warnings
warnings.simplefilter('ignore')

import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.translate.bleu_score import sentence_bleu

import json
import glob
import string


from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from IPython.core.display import display, HTML
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
#from sentence_transformers import models, SentenceTransformer
import shutil

import torch
from transformers import BertTokenizer, BertModel

import pandas as pd

from nltk.tokenize import word_tokenize
import numpy as np

# Load the library with the CountVectorizer method
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

from summarizer import Summarizer
import math

import os, stat
import ntpath
from string import ascii_uppercase
#import nxviz

%matplotlib inline
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import os, stat
import ntpath
from string import ascii_uppercase
import spacy
from spacy.matcher import PhraseMatcher #import PhraseMatcher class

import matplotlib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
# Specify the Kaggle Username and Key to use the Kaggle Api

# os.environ['KAGGLE_USERNAME'] = '*************'
# os.environ['KAGGLE_KEY'] = '****************'
# from kaggle.api.kaggle_api_extended import KaggleApi

# api = KaggleApi()
# api.authenticate()

# api.dataset_download_files(dataset="allen-institute-for-ai/CORD-19-research-challenge", path=DATA_PATH, unzip=True)

# HTML('''<script>
# code_show=true; 
# function code_toggle() {
#  if (code_show){
#  $('div.input').hide();
#  } else {
#  $('div.input').show();
#  }
#  code_show = !code_show
# } 
# $( document ).ready(code_toggle);
# </script>
# The raw code for this IPython notebook is by default hidden for easier reading.
# To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')
DATA_PATH = os.getcwd()+'/kaggle/input/CORD-19-research-challenge/'
#Get data from path '/data1/cov19/kaggle_data_0331/'
'''
bio_path = '/kaggle/input/CORD-19-research-challenge//biorxiv_medrxiv/'
comm_path = '/kaggle/input/CORD-19-research-challenge//comm_use_subset/'
non_comm_path = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/'
custom_path = '/kaggle/input/CORD-19-research-challenge/custom_license/'
journals = {"BIORXIV_MEDRXIV": bio_path,
             "COMMON_USE_SUB" : comm_path,
             "NON_COMMON_USE_SUB" : non_comm_path,
             "CUSTOM_LICENSE" : custom_path}
'''
'''
def parse_each_json_file(file_path,journal):
    inp = None
    with open(file_path) as f:
        inp = json.load(f)
    rec = {}
    rec['sha'] = inp['paper_id'] or None
    rec['title'] = inp['metadata']['title'] or None
    abstract = "\n ".join([inp['abstract'][_]['text'] for _ in range(len(inp['abstract']) - 1)]) 
    rec['abstract'] = abstract or None
    full_text = []
    for _ in range(len(inp['body_text'])):
        try:
            full_text.append(inp['body_text'][_]['text'])
        except:
            pass

    rec['full_text'] = "\n ".join(full_text) or None
    rec['source'] =  journal     or None    
    return rec

def parse_json_and_create_csv(journals):
    journal_dfs = []
    for journal, path in journals.items():
        parsed_rcds = []  
        json_files = glob.glob('{}/**/*.json'.format(path), recursive=True)
        for file_name in json_files:
            rec = parse_each_json_file(file_name,journal)
            parsed_rcds.append(rec)
        df = pd.DataFrame(parsed_rcds)
        journal_dfs.append(df)
    return pd.concat(journal_dfs)


fin_df = parse_json_and_create_csv(journals=journals)
fin_df.head()

'''
fin_df = pd.read_csv('/kaggle/input/task3-data/fin_df.csv')
fin_df.head()
Image("/kaggle/input/task3-data/Images/Images/SequenceConversion.PNG")
###-------------------------------------------------------------
### READING AND ASSEMBLING GENOME SEQUENCE FILE
###-------------------------------------------------------------
#File 1 to compare
fname = open("/kaggle/input/task3-data/gnome_data_countries/gnome_data_countries/CHINA_SHENZHEN_MN938384.1.fasta","r")
A = fname.read().replace('\n','')
fname.close()
sequence1 = (list(A))

#File2 to compare
fname2 = open("/kaggle/input/task3-data/gnome_data_countries/gnome_data_countries/italy_MT066156.1.fasta","r")
B = fname2.read().replace('\n','')
fname.close()
sequence2 = (list(B))
Image("/kaggle/input/task3-data/Images/Images/SignalSegmentation.PNG")
Image("/kaggle/input/task3-data/Images/Images/SC1.PNG")
Image("/kaggle/input/task3-data/Images/Images/SC2.PNG")
Image("/kaggle/input/task3-data/Images/Images/SC3.PNG")
###-------------------------------------------------------------
### corrERSION OF SEQUENCE INTO A TIME DOMAIN RANDOM COMPLEX SIGNAL
###-------------------------------------------------------------
from string import ascii_uppercase

seq_len0 = len(sequence1)
seq_len1 = len(sequence2)
    
signal0 = np.zeros(seq_len0, dtype=np.complex_)
signal1 = np.zeros(seq_len1, dtype=np.complex_)
    
    
max_seq_len=max(seq_len0,seq_len1);
### 26 is total number of alphabets
L = 26


RANDOM_VEC = (1/np.sqrt(2))*((np.random.randn(1,L))+1j*(np.random.randn(1,L)))

prob0 = np.zeros(L)
prob1 = np.zeros(L)

for i in ascii_uppercase:
    prob0[ord(i)-65] =  sequence1.count(i)
    prob1[ord(i)-65] =  sequence2.count(i)
  
       
for i in range (seq_len0):
    signal0[i] = RANDOM_VEC[0][ord(sequence1[i])-65]
    #print('signal0[', i, ']=', signal0[i], '\n')
        
for i in range (seq_len1):
    signal1[i] = RANDOM_VEC[0][ord(sequence2[i])-65]
###------------------------------------------------
###TIME DOMAIN SIGNAL SEGMENTATION - FFT SIZE
###------------------------------------------------
fft_size = 1024;
import math
rows = math.ceil(max_seq_len/fft_size)
cols = fft_size

###print("seq_len=", seq_len, " rows=", rows, " columns=", cols)

seq_arr0=np.zeros((rows,cols), dtype=np.complex_)
seq_arr1=np.zeros((rows,cols), dtype=np.complex_)


for row_pointer in range (rows):
    start_index = fft_size*(row_pointer)
    end_index = fft_size*(row_pointer+1)
    if(end_index <= len(sequence1)):
        seq_arr0[row_pointer,:] = signal0[start_index:end_index]
    else:
        if (start_index >= len(sequence1)):
            ###fill all 0
            seq_arr0[row_pointer,:] = np.zeros(1,fft_size);
                
        else:
            ###copy till seq_len and then all 0s
            ###print('row pointer:', row_pointer, ' start:',start_index, ' end:',end_index, ' seq_len:', seq_len0)
            seq_arr0[row_pointer,0:(seq_len0-start_index)] = signal0[start_index:seq_len0]
            seq_arr0[row_pointer,(seq_len0-start_index):] = np.zeros((1,end_index-seq_len0), dtype=np.complex_)
                
for row_pointer in range (rows):
    start_index = fft_size*(row_pointer)
    end_index = fft_size*(row_pointer+1)
    if(end_index <= len(sequence2)):
        seq_arr1[row_pointer,:] = signal1[start_index:end_index]
    else:
        if (start_index >= len(sequence2)):
            ###fill all 0
            seq_arr1[row_pointer,:] = np.zeros(1,fft_size);
                
        else:
            ###copy till seq_len and then all 0s
            seq_arr1[row_pointer,0:(seq_len1-start_index)] = signal1[start_index:seq_len1]
            seq_arr1[row_pointer,(seq_len1-start_index):] = np.zeros((1,end_index-seq_len1), dtype=np.complex_)

###------------------------------------------------
### SIGNAL IN FREQ DOMAIN  -- CHECK
###------------------------------------------------

SEQ_ARR0=np.zeros((rows,cols), dtype=np.complex_)
SEQ_ARR1=np.zeros((rows,cols), dtype=np.complex_)
for j in range (rows):
    SEQ_ARR0[j,:]=np.fft.fft(seq_arr0[j,:])
    SEQ_ARR1[j,:]=np.fft.fft(seq_arr1[j,:])

###------------------------------------------------
###SIGNAL CORRELATION IN FREQ DOMAIN
###------------------------------------------------
nrof_segments=rows
    
CORR_CELL=np.zeros((nrof_segments,2,2,cols), dtype=np.complex_)
    
for m in range (nrof_segments):
    CORR_CELL[m, 0, 0, :]= np.multiply( (SEQ_ARR0[m,:]),(np.conj(SEQ_ARR0[m,:])) )
    CORR_CELL[m, 0, 1, :]= np.multiply( (SEQ_ARR0[m,:]),(np.conj(SEQ_ARR1[m,:])) )
    CORR_CELL[m, 1, 0, :]= np.multiply( (SEQ_ARR1[m,:]),(np.conj(SEQ_ARR0[m,:])) )
    CORR_CELL[m, 1, 1, :]= np.multiply( (SEQ_ARR1[m,:]),(np.conj(SEQ_ARR1[m,:])) )
###------------------------------------------------
### BACK TO TIME DOMAIN - correlation of piece wise time domain signal
###------------------------------------------------

corr_cell=np.zeros((nrof_segments,2,2,cols), dtype=np.complex_)
coef0 = np.zeros(nrof_segments);
coef1 = np.zeros(nrof_segments);
for m in range (nrof_segments):
    for i in range (2):
        for j in range(2):
            corr_cell[m, i, j, :]= np.fft.ifft(CORR_CELL[m,i,j])
    coef0[m]=np.absolute(corr_cell[m, 0, 0, 0])
    coef1[m]=np.absolute(corr_cell[m, 1, 0, 0])
###------------------------------------------------
###PLOT DATA
###------------------------------------------------
import matplotlib.pyplot as plt
metric1_arr = np.divide(coef0,coef1)
metric1 = np.sum(np.absolute(metric1_arr) - 1)
plt.rcParams['figure.figsize'] = [15,10]
fig, axs = plt.subplots(2,1, constrained_layout=True)

fig = plt.figure()
axs[0].plot(range(nrof_segments), metric1_arr, 'cs')
axs[0].set_title('metric1 : coef1/coef2 - 1 : '+str(metric1));
axs[0].set_xlabel('number of segments');
axs[0].set_ylabel('C1/C2-1');
axs[0].grid()
fig.suptitle('COVID19 GENOME Variation Plot', fontsize=16)
    
metric2_arr = np.subtract(coef0,coef1)
metric2 = np.sum(np.absolute(metric2_arr))
axs[1].plot(range(nrof_segments), metric2_arr, 'gs')
axs[1].set_title('metric2 : coef1-coef2 : '+str(metric2));
axs[1].set_xlabel('number of segments');
axs[1].set_ylabel('C1-C2');
axs[1].grid()

files = []

#currdir = os.getcwd()
#path = currdir + '/' + 'gnome_data'

path = '/kaggle/input/task3-data/gnome_data_countries/gnome_data_countries'
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file or '.fasta' in file:
            files.append(os.path.join(r, file))

g_fname = files    
for f in g_fname:
    print(f)
out_file_path = os.getcwd()
def run_corr_2files(num1, num2):
    
    
    ###-------------------------------------------------------------
    ### corrERSION OF SEQUENCE INTO A TIME DOMAIN RANDOM COMPLEX SIGNAL
    ###-------------------------------------------------------------
    
    seq_len0 = len(sequence[num1])
    seq_len1 = len(sequence[num2])
    
    
    ###max_seq_len=max(seq_len0,seq_len1);
    min_seq_len = min(seq_len0,seq_len1)
    ### 26 is total number of alphabets
    L = 26
    
    B=100
    metric_samples=np.zeros(B)
    for b in range(B):
        
        signal0 = np.zeros(seq_len0, dtype=np.complex_)
        signal1 = np.zeros(seq_len1, dtype=np.complex_)
    
        RANDOM_VEC = (1/np.sqrt(2))*((np.random.randn(1,L))+1j*(np.random.randn(1,L)))
        
        prob0 = np.zeros(L)
        prob1 = np.zeros(L)
        
        for i in ascii_uppercase:
            prob0[ord(i)-65] =  sequence[num1].count(i)
            prob1[ord(i)-65] =  sequence[num2].count(i)
         
          
        for i in range (seq_len0):
            signal0[i] = RANDOM_VEC[0][ord(sequence[num1][i])-65]
            #print('signal0[', i, ']=', signal0[i], '\n')
            
            
        for i in range (seq_len1):
            signal1[i] = RANDOM_VEC[0][ord(sequence[num2][i])-65]
        
        ###------------------------------------------------
        ###TIME DOMAIN SIGNAL SEGMENTATION - FFT SIZE
        ###------------------------------------------------
        fft_size = 1024
    
        rows = math.ceil(min_seq_len/fft_size)
        cols = fft_size
        ###print("seq_len=", seq_len, " rows=", rows, " columns=", cols)
        
        seq_arr0=np.zeros((rows,cols), dtype=np.complex_)
        seq_arr1=np.zeros((rows,cols), dtype=np.complex_)
        
        for row_pointer in range (rows):
            start_index = fft_size*(row_pointer)
            end_index = fft_size*(row_pointer+1)
            if(end_index <= len(sequence[num1])):
                seq_arr0[row_pointer,:] = signal0[start_index:end_index]
            else:
                if (start_index >= len(sequence[num1])):
                    ###fill all 0
                    seq_arr0[row_pointer,:] = np.zeros((1,fft_size), dtype=np.complex_);
                    
                else:
                    ###copy till seq_len and then all 0s
                    ###print('row pointer:', row_pointer, ' start:',start_index, ' end:',end_index, ' seq_len:', seq_len0)
                    seq_arr0[row_pointer,0:(seq_len0-start_index)] = signal0[start_index:seq_len0]
                    seq_arr0[row_pointer,(seq_len0-start_index):] = np.zeros((1,end_index-seq_len0), dtype=np.complex_)
                    
        for row_pointer in range (rows):
            start_index = fft_size*(row_pointer)
            end_index = fft_size*(row_pointer+1)
            if(end_index <= len(sequence[num2])):
                seq_arr1[row_pointer,:] = signal1[start_index:end_index]
            else:
                if (start_index >= len(sequence[num2])):
                    ###fill all 0
                    seq_arr1[row_pointer,:] = np.zeros((1,fft_size), dtype=np.complex_);
                    
                else:
                    ###copy till seq_len and then all 0s
                    seq_arr1[row_pointer,0:(seq_len1-start_index)] = signal1[start_index:seq_len1]
                    seq_arr1[row_pointer,(seq_len1-start_index):] = np.zeros((1,end_index-seq_len1), dtype=np.complex_)
                              
    
        ###------------------------------------------------
        ###SIGNAL IN FREQ DOMAIN  -- CHECK
        ###------------------------------------------------
        
        SEQ_ARR0=np.zeros((rows,cols), dtype=np.complex_)
        SEQ_ARR1=np.zeros((rows,cols), dtype=np.complex_)
        for j in range (rows):
            SEQ_ARR0[j,:]=np.fft.fft(seq_arr0[j,:])
            SEQ_ARR1[j,:]=np.fft.fft(seq_arr1[j,:])
        
        ###------------------------------------------------
        ###SIGNAL CORRELATION IN FREQ DOMAIN
        ###------------------------------------------------
        nrof_segments=rows
        
        CORR_CELL=np.zeros((nrof_segments,1,2,cols), dtype=np.complex_)
        
        for m in range (nrof_segments):
            CORR_CELL[m, 0, 0, :]= np.multiply( (SEQ_ARR0[m,:]),(np.conj(SEQ_ARR0[m,:])) )
            CORR_CELL[m, 0, 1, :]= np.multiply( (SEQ_ARR0[m,:]),(np.conj(SEQ_ARR1[m,:])) )
                   
        
        ###------------------------------------------------
        ###BACK TO TIME DOMAIN - correlation of piece wise time domain signal
        ###------------------------------------------------
        
        corr_cell=np.zeros((nrof_segments,1,2,cols), dtype=np.complex_)
        coef0 = np.zeros(nrof_segments);
        coef1 = np.zeros(nrof_segments);
        for m in range (nrof_segments):
            corr_cell[m, 0, 0, :]= np.fft.ifft(CORR_CELL[m,0,0])
            corr_cell[m, 0, 1, :]= np.fft.ifft(CORR_CELL[m,0,1])
            coef0[m]=max(np.absolute(corr_cell[m, 0, 0]))
            coef1[m]=max(np.absolute(corr_cell[m, 0, 1]))
            
        coef = np.divide(coef0,coef1) - 1
        
        metric_samples[b] = np.sum(np.absolute(coef))
        
              
    np.sort(metric_samples)
    #print('sorted: ', metric_samples,'\n')
    return(metric_samples[89])
#print(g_fname)
    
sequence = [] 
check_var1 = 0


i = 0

for j in range (len(g_fname)):
    A=''
    with open(g_fname[j]) as fhandler:
        first_line = fhandler.readline()
        if (first_line.find('genome') != -1):
            ###print(first_line)
            print("Discarding first line of:", g_fname[j],'\n')
            ##messagebox.showwarning('Warning', 'Discarding first line: '+first_line)
        else :
            A = first_line

        for line in fhandler:
            A += line   

        A = A.replace('\n','') 

        sequence.append([])
        #####-----------------------------
        #####DNA to Amino
        #####-----------------------------          
        if(check_var1 == 1):
            #####-----------------------------
            #####DNA to RNA
            #####-----------------------------
            A = A.replace('T','U') 

            ####------------------------------
            #### RNA to amino Acid
            ####------------------------------                
            key = ''
            for k in range(0,3*math.floor(len(A)/3),3):
                key = A[k]+A[k+1]+A[k+2]
                ##print("here:",k, A[k],A[k+1],A[k+2],'\n')
                if(rna_to_amino.get(key) == None):
                    print('Data Corrupted for file...abanding:\n', g_fname[j])
                    messagebox.showerror('Error', 'Data corrupted for file: '+g_fname[j]+' => expected only A,C,G and T in the DNA sequence')


                if(rna_to_amino[key] != '*'):
                    sequence[i].append(rna_to_amino[key])
        else :
            sequence[i] = list(A)

    i = i+1
global out_file_path
nrof_samples = len(g_fname)
nrof_calc_metrics = int((nrof_samples*(nrof_samples-1))/2)

metric_arr = np.zeros(nrof_calc_metrics)
sample_loc_arr = np.chararray((nrof_calc_metrics,1,2), itemsize=100, unicode=True)
sample_number_arr = np.zeros((nrof_calc_metrics,3))
record_arr = np.chararray((nrof_calc_metrics), itemsize=100, unicode=True)
loc_arr = np.chararray((nrof_calc_metrics), itemsize=100, unicode=True)

k=0
for i in range(nrof_samples):
    for j in range((i+1),nrof_samples):
        if(len(sequence[i]) > len(sequence[j])):
            metric_arr[k] = run_corr_2files(i, j)
        else :
            metric_arr[k] = run_corr_2files(j, i)

        sample_loc_arr[k,0,0] = ntpath.basename(g_fname[i]).replace('.fasta','')
        sample_loc_arr[k,0,1] = ntpath.basename(g_fname[j]).replace('.fasta','')
        record_arr[k] = sample_loc_arr[k,0,0]+', '+sample_loc_arr[k,0,1]+' = '+str(metric_arr[k]) 
        loc_arr[k] = sample_loc_arr[k,0,0]+', '+sample_loc_arr[k,0,1]
        print(record_arr[k]+'\n')
        sample_number_arr[k,0] = i;
        sample_number_arr[k,1] = j;
        sample_number_arr[k,2] = metric_arr[k];
        k = k+1

#-----------
#OUtput file
#-----------
sort_metric_arr = np.sort(metric_arr, axis=0)
index = np.argsort(metric_arr, axis=0)
sort_record_arr = []
sort_loc_arr = []
for i in index:
    sort_record_arr.append(record_arr[i])
    sort_loc_arr.append(loc_arr[i])


#remove previous created output folder
#!rmdir 'Outpur'
!mkdir 'Output'
filename_output= 'Output/output_metrics1.csv'
filename_output_raw= 'Output/output_raw_data1.csv'
filename_output_loc= 'Output/output_loc1.txt'
filename_output_sample_number= 'Output//sample_number1.txt'

fileID = open(filename_output,'w');
fileID2 = open(filename_output_raw,'w');
fileID3 = open(filename_output_loc,'w');
fileID4 = open(filename_output_sample_number,'w');
for k in range(nrof_calc_metrics):    
    fileID.write(str(sort_metric_arr[k])+'\n')  
    fileID2.write(sort_record_arr[k]+'\n')   
    fileID3.write(sort_loc_arr[k]+'\n')
    fileID4.write(str(sample_number_arr[k,:])+'\n')

fileID.close()
fileID2.close()
fileID3.close()
fileID4.close()
filename_output
filename_output_raw
filename_output_loc
filename_output_sample_number
files = []

path = 'Output/'

for r, d, f in os.walk(out_file_path):
    for file in f:
        print(file)
files = []

#currdir = os.getcwd()
#path = currdir + '/' + 'gnome_data'

path = './' + 'gnome_data_time'
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for file in f:
        if '.txt' in file or '.fasta' in file:
            files.append(os.path.join(r, file))

g_fname = files    
for f in g_fname:
    print(f)
# Output File Location
out_file_path = 'Output/'
def run_corr_2files(num1, num2):
    
    
    ###-------------------------------------------------------------
    ### corrERSION OF SEQUENCE INTO A TIME DOMAIN RANDOM COMPLEX SIGNAL
    ###-------------------------------------------------------------
    
    seq_len0 = len(sequence[num1])
    seq_len1 = len(sequence[num2])
    
    
    ###max_seq_len=max(seq_len0,seq_len1);
    min_seq_len = min(seq_len0,seq_len1)
    ### 26 is total number of alphabets
    L = 26
    
    B=100
    metric_samples=np.zeros(B)
    for b in range(B):
        
        signal0 = np.zeros(seq_len0, dtype=np.complex_)
        signal1 = np.zeros(seq_len1, dtype=np.complex_)
    
        RANDOM_VEC = (1/np.sqrt(2))*((np.random.randn(1,L))+1j*(np.random.randn(1,L)))
        
        prob0 = np.zeros(L)
        prob1 = np.zeros(L)
        
        for i in ascii_uppercase:
            prob0[ord(i)-65] =  sequence[num1].count(i)
            prob1[ord(i)-65] =  sequence[num2].count(i)
         
          
        for i in range (seq_len0):
            signal0[i] = RANDOM_VEC[0][ord(sequence[num1][i])-65]
            #print('signal0[', i, ']=', signal0[i], '\n')
            
            
        for i in range (seq_len1):
            signal1[i] = RANDOM_VEC[0][ord(sequence[num2][i])-65]
        
        ###------------------------------------------------
        ###TIME DOMAIN SIGNAL SEGMENTATION - FFT SIZE
        ###------------------------------------------------
        fft_size = 1024
    
        rows = math.ceil(min_seq_len/fft_size)
        cols = fft_size
        ###print("seq_len=", seq_len, " rows=", rows, " columns=", cols)
        
        seq_arr0=np.zeros((rows,cols), dtype=np.complex_)
        seq_arr1=np.zeros((rows,cols), dtype=np.complex_)
        
        for row_pointer in range (rows):
            start_index = fft_size*(row_pointer)
            end_index = fft_size*(row_pointer+1)
            if(end_index <= len(sequence[num1])):
                seq_arr0[row_pointer,:] = signal0[start_index:end_index]
            else:
                if (start_index >= len(sequence[num1])):
                    ###fill all 0
                    seq_arr0[row_pointer,:] = np.zeros((1,fft_size), dtype=np.complex_);
                    
                else:
                    ###copy till seq_len and then all 0s
                    ###print('row pointer:', row_pointer, ' start:',start_index, ' end:',end_index, ' seq_len:', seq_len0)
                    seq_arr0[row_pointer,0:(seq_len0-start_index)] = signal0[start_index:seq_len0]
                    seq_arr0[row_pointer,(seq_len0-start_index):] = np.zeros((1,end_index-seq_len0), dtype=np.complex_)
                    
        for row_pointer in range (rows):
            start_index = fft_size*(row_pointer)
            end_index = fft_size*(row_pointer+1)
            if(end_index <= len(sequence[num2])):
                seq_arr1[row_pointer,:] = signal1[start_index:end_index]
            else:
                if (start_index >= len(sequence[num2])):
                    ###fill all 0
                    seq_arr1[row_pointer,:] = np.zeros((1,fft_size), dtype=np.complex_);
                    
                else:
                    ###copy till seq_len and then all 0s
                    seq_arr1[row_pointer,0:(seq_len1-start_index)] = signal1[start_index:seq_len1]
                    seq_arr1[row_pointer,(seq_len1-start_index):] = np.zeros((1,end_index-seq_len1), dtype=np.complex_)
                              
    
        ###------------------------------------------------
        ###SIGNAL IN FREQ DOMAIN  -- CHECK
        ###------------------------------------------------
        
        SEQ_ARR0=np.zeros((rows,cols), dtype=np.complex_)
        SEQ_ARR1=np.zeros((rows,cols), dtype=np.complex_)
        for j in range (rows):
            SEQ_ARR0[j,:]=np.fft.fft(seq_arr0[j,:])
            SEQ_ARR1[j,:]=np.fft.fft(seq_arr1[j,:])
        
        ###------------------------------------------------
        ###SIGNAL CORRELATION IN FREQ DOMAIN
        ###------------------------------------------------
        nrof_segments=rows
        
        CORR_CELL=np.zeros((nrof_segments,1,2,cols), dtype=np.complex_)
        
        for m in range (nrof_segments):
            CORR_CELL[m, 0, 0, :]= np.multiply( (SEQ_ARR0[m,:]),(np.conj(SEQ_ARR0[m,:])) )
            CORR_CELL[m, 0, 1, :]= np.multiply( (SEQ_ARR0[m,:]),(np.conj(SEQ_ARR1[m,:])) )
                   
        
        ###------------------------------------------------
        ###BACK TO TIME DOMAIN - correlation of piece wise time domain signal
        ###------------------------------------------------
        
        corr_cell=np.zeros((nrof_segments,1,2,cols), dtype=np.complex_)
        coef0 = np.zeros(nrof_segments);
        coef1 = np.zeros(nrof_segments);
        for m in range (nrof_segments):
            corr_cell[m, 0, 0, :]= np.fft.ifft(CORR_CELL[m,0,0])
            corr_cell[m, 0, 1, :]= np.fft.ifft(CORR_CELL[m,0,1])
            coef0[m]=max(np.absolute(corr_cell[m, 0, 0]))
            coef1[m]=max(np.absolute(corr_cell[m, 0, 1]))
            
        coef = np.divide(coef0,coef1) - 1
        
        metric_samples[b] = np.sum(np.absolute(coef))
        
              
    np.sort(metric_samples)
    #print('sorted: ', metric_samples,'\n')
    return(metric_samples[89])
sequence = [] 
check_var1 = 0
i = 0

for j in range (len(g_fname)):
    A=''
    with open(g_fname[j]) as fhandler:
        first_line = fhandler.readline()
        if (first_line.find('genome') != -1):
            ###print(first_line)
            print("Discarding first line of:", g_fname[j],'\n')
            messagebox.showwarning('Warning', 'Discarding first line: '+first_line)
        else :
            A = first_line

        for line in fhandler:
            A += line   

        A = A.replace('\n','') 

        if ((A.count('A')+A.count('T')+A.count('C')+A.count('G')) != len(A)):
            print ('Error :Data corrupted for file: '+g_fname[j]+' => expected only A,C,G and T in the DNA sequence')
            ##root.destroy()
            ##return()

        sequence.append([])
        #####-----------------------------
        #####DNA to Amino
        #####-----------------------------          
        if(check_var1 == 1):
            #####-----------------------------
            #####DNA to RNA
            #####-----------------------------
            A = A.replace('T','U') 

            ####------------------------------
            #### RNA to amino Acid
            ####------------------------------                
            key = ''
            for k in range(0,3*math.floor(len(A)/3),3):
                key = A[k]+A[k+1]+A[k+2]
                ##print("here:",k, A[k],A[k+1],A[k+2],'\n')
                if(rna_to_amino.get(key) == None):
                    print('Data Corrupted for file...abanding:\n', g_fname[j])
                    #messagebox.showerror('Error', 'Data corrupted for file: '+g_fname[j]+' => expected only A,C,G and T in the DNA sequence')
                    ##root.destroy()
                    ##return()

                if(rna_to_amino[key] != '*'):
                    sequence[i].append(rna_to_amino[key])
        else :
            sequence[i] = list(A)

    i = i+1
global out_file_path
nrof_samples = len(g_fname)
nrof_calc_metrics = int((nrof_samples*(nrof_samples-1))/2)

metric_arr = np.zeros(nrof_calc_metrics)
sample_loc_arr = np.chararray((nrof_calc_metrics,1,2), itemsize=100, unicode=True)
sample_number_arr = np.zeros((nrof_calc_metrics,3))
record_arr = np.chararray((nrof_calc_metrics), itemsize=100, unicode=True)
loc_arr = np.chararray((nrof_calc_metrics), itemsize=100, unicode=True)

k=0
for i in range(nrof_samples):
    for j in range((i+1),nrof_samples):
        if(len(sequence[i]) > len(sequence[j])):
            metric_arr[k] = run_corr_2files(i, j)
        else :
            metric_arr[k] = run_corr_2files(j, i)

        sample_loc_arr[k,0,0] = ntpath.basename(g_fname[i]).replace('.fasta','')
        sample_loc_arr[k,0,1] = ntpath.basename(g_fname[j]).replace('.fasta','')
        record_arr[k] = sample_loc_arr[k,0,0]+', '+sample_loc_arr[k,0,1]+' = '+str(metric_arr[k]) 
        loc_arr[k] = sample_loc_arr[k,0,0]+', '+sample_loc_arr[k,0,1]
        print(record_arr[k]+'\n')
        sample_number_arr[k,0] = i;
        sample_number_arr[k,1] = j;
        sample_number_arr[k,2] = metric_arr[k];
        k = k+1

#-----------
#OUtput file
#-----------
sort_metric_arr = np.sort(metric_arr, axis=0)
index = np.argsort(metric_arr, axis=0)
sort_record_arr = []
sort_loc_arr = []
for i in index:
    sort_record_arr.append(record_arr[i])
    sort_loc_arr.append(loc_arr[i])

filename_output= out_file_path+'/output_metrics1.csv'
filename_output_raw= out_file_path+'/output_raw_data1.csv'
filename_output_loc= out_file_path+'/output_loc1.txt'
filename_output_sample_number= out_file_path+'/sample_number1.txt'

filename_output
fileID = open(filename_output,'w');
fileID2 = open(filename_output_raw,'w');
fileID3 = open(filename_output_loc,'w');
fileID4 = open(filename_output_sample_number,'w');
for k in range(nrof_calc_metrics):    
    fileID.write(str(sort_metric_arr[k])+'\n')  
    fileID2.write(sort_record_arr[k]+'\n')   
    fileID3.write(sort_loc_arr[k]+'\n')
    fileID4.write(str(sample_number_arr[k,:])+'\n')

fileID.close()
fileID2.close()
fileID3.close()
fileID4.close()
Image("/kaggle/input/task3-data/Images/Images/Res1.PNG")
Image("/kaggle/input/task3-data/Images/Images/Res2.PNG")
Image("/kaggle/input/task3-data/Images/Images/Res3.PNG")
Image("/kaggle/input/task3-data/Images/Images/Res4.PNG")
Image("/kaggle/input/task3-data/Images/Images/Res5.PNG")
def wordcloud_draw(text, color = 'white'):
    """
    Plots wordcloud of string text after removing stopwords
    """
    cleaned_word = " ".join([word for word in text.split()])
    wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color=color,
                      width=1000,
                      height=1000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(15, 15))
    plt.imshow(wordcloud)
    plt.axis('off')
    display(plt.show())
animals = []
animalList = []
with open('/kaggle/input/task3-data/animals.txt', "r") as f:
    animals = f.readlines()
animalList = [s.replace('\n', '') for s in animals]
animalList.append('pangolin')
animalList.append('mice')
animalList.append('animal')
animalList = [string for string in animalList if string != ""]
animalList = list(map(lambda x:x.lower(), animalList))
animalList.remove('human')
animalList.remove('discus')
pluralList = ['{0}s'.format(elem) for elem in animalList]
animalList = animalList + pluralList
#animalList = [' {0} '.format(elem) for elem in animalList]
animalList[0:5]
covid19_list = []
with open('/kaggle/input/task3-data/covid19.txt', "r") as f:
    words = f.readlines()
covid19_list = [s.replace('\n', '') for s in words]
covid19_list[0:5]
#Load Meta data
metadata = pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
metadata.head()
#### Merging Articles with Meta data. Loading data from already merged csv

Left_join = pd.merge(fin_df,  
                     metadata,  
                     on ='sha',  
                     how ='left') 
Left_join = Left_join.drop(columns=['cord_uid', 'doi', 'title_y', 'pmcid', 'abstract_y', 'Microsoft Academic Paper ID', 'WHO #Covidence',
                       'full_text_file', 'url', 'pubmed_id'])
Left_join.head()
Left_join['title_x'].fillna("NoTitle", inplace = True)
Left_join['abstract_x'].fillna("NoAbstract", inplace = True)
Left_join['full_text'].fillna("NoText", inplace = True)
Left_join['combined'] = Left_join['title_x'] + ' ' + Left_join['abstract_x'] + ' ' + Left_join['full_text']

Left_join.head()
cond2 = Left_join['abstract_x'].str.contains('livestock')
print(sum(cond2))
abstract_livestock = Left_join[cond2]
abstract_livestock.shape
abstract_livestock.head(5)
cond3 = abstract_livestock['abstract_x'].str.contains('farmer')
abstract_livestock_farmer = abstract_livestock[cond3]
abstract_livestock_farmer.head(5)
def clean_text(article):
    clean1 = re.sub(r'['+string.punctuation + '’—”'+']', "", article.lower())
    return re.sub(r'\W+', ' ', clean1)
import re

#Left_join['tokenized'] = Left_join['combined'].map(lambda x: clean_text(x))

import nltk
'''
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
Left_join['tokenized'] = Left_join['tokenized'].apply(lemmatize_text)
'''
## If we need space
#del Left_join

def find_spillover_wds(content, wds = ['transfer','spillover','pass on','transmit','contract',
                                       'distribute','progress','incubate','spread','disseminate','zoonosis','zoonotic']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found
def find_human_wds(content, wds = ['human','people','man','child','kid']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found
def find_covid_wds(content, wds = ['covid-19','covid','cov','coronavirus','corona']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found
def find_evidence_wds(content, wds = ['domestic animal', 'backyard*livestock', 'wet markets', 'meat markets','seafood markets', 'bites', 'bitten', 
                                      'laboratory*accident', 'fairs', 'petting*zoo',  'trading*wild animal', 
                                      'destruction*habitat', 'wild animal*food', 'lifestock pathogens','genetic mutations',
                                      'animal testing', ' hunting ', 'industrial*farming', ' pet ', 'butcher', ' eat ', ' meat ']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ','
    return found
'''
Left_join['evidence_wds'] = Left_join['combined'].apply(find_evidence_wds)
Left_join.evidence_wds.unique()
'''
'''
Left_join['evidence_wds'].replace('', np.nan, inplace=True)


Left_join['animal_wds'] = Left_join['tokenized'].apply(find_animal_wds)
Left_join['spillover_wds'] = Left_join['tokenized'].apply(find_spillover_wds)
Left_join['human_wds'] = Left_join['tokenized'].apply(find_human_wds)
Left_join['virus_wds'] = Left_join['tokenized'].apply(find_covid_wds)
Left_join['evidence_wds'] = Left_join['combined'].apply(find_evidence_wds)

Left_join['animal_wds'] = Left_join['animal_wds'].str.replace('discus', '')
Left_join['animal_wds'].replace('', np.nan, inplace=True)
Left_join['spillover_wds'].replace('', np.nan, inplace=True)
Left_join['human_wds'].replace('', np.nan, inplace=True)
Left_join['virus_wds'].replace('', np.nan, inplace=True)
Left_join['evidence_wds'].replace('', np.nan, inplace=True)

articlesAnimal = Left_join[Left_join[['animal_wds', 'spillover_wds', 'human_wds', 'virus_wds','evidence_wds']].notnull().all(1)]
articlesAnimal['animal_wds'] = articlesAnimal['animal_wds'].map(lambda x: clean_text(x))
articlesAnimal['animal_wds'].replace(' ', np.nan, inplace=True)
articlesAnimal = articlesAnimal.dropna(subset=['animal_wds'])
articlesAnimal
'''
'''
print('Total articles containing animal keywords: ' + str(len(articlesAnimal)))
print('Total articles available: ' + str(len(fin_df)))
print(str(float(len(articlesAnimal)/len(fin_df)*100)) + '% of the articles available contains animal keywords')
'''
'''
def find_country_wds(content, wds = ['transfer','spillover','pass on','transmit','contract',
                                       'distribute','progress','incubate','spread','disseminate','zoonosis']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found
'''
#articlesAnimal.to_csv('animal_articles.csv', sep=',', encoding='utf-8')
del Left_join
articlesAnalysis = pd.read_csv("/kaggle/input/task3-data/animal_articles.csv")
#articlesAnalysis['combined'] = articlesAnalysis['combined'].str.replace(r'\b(\w{1,2})\b', '')
articlesAnalysis.head()
# Join the different processed titles together.
animal_string = ''.join(list(articlesAnalysis['animal_wds'].values))
# Create a WordCloud object
wordcloud = WordCloud(width = 600, height = 400, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', collocations=False)
# Generate a word cloud
wordcloud.generate(animal_string)
# Visualize the word cloud
wordcloud.to_image()
def find_animal_wds(content, wds = [
 ' dog ',
 ' cat ',
 'pig',
 'mouse',
 'bird',
 ' bat ',
 'horse',
 ' rat ',
 'sheep',
 'chicken',
 'rabbit',
 'insect',
 'goat',
 'monkey',
 'fox',
 'fish',
 'cow',
 'ferret',
 'deer',
 'fly',
 'raccoon',
 'camel',
 'hamster',
 'bear',
'pangolin',
'tiger']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ','
            if found.count(',') == 5:
                break
    return found
articlesAnalysis['animal_wds'] = articlesAnalysis['combined'].apply(find_animal_wds)
articlesAnalysis['animal_wds'] = articlesAnalysis['animal_wds'].str.replace(',', ' ')
articlesAnalysis['animal_wds'].replace('', np.nan, inplace=True)
articlesAnalysis = articlesAnalysis.dropna(subset=['animal_wds'])
articlesAnalysis
# Join the different processed titles together.
animal_string = ''.join(list(articlesAnalysis['animal_wds'].values))
# Create a WordCloud object
wordcloud = WordCloud(width = 600, height = 400, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', collocations=False)
# Generate a word cloud
wordcloud.generate(animal_string)
# Visualize the word cloud
wordcloud.to_image()
def plot_50_most_common_words(count_data, count_vectorizer):
    import matplotlib.pyplot as plt
    words = count_vectorizer.get_feature_names()
    total_counts = np.zeros(len(words))
    for t in count_data:
        total_counts+=t.toarray()[0]
    
    count_dict = (zip(words, total_counts))
    count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:20]
    words = [w[0] for w in count_dict]
    counts = [w[1] for w in count_dict]
    x_pos = np.arange(len(words)) 
    
    plt.figure(2, figsize=(10, 8/1.6180))
    plt.subplot(title='20 most common words')
    sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
    sns.barplot(x_pos, counts, palette='husl')
    plt.xticks(x_pos, words, rotation=90) 
    plt.xlabel('words')
    plt.ylabel('counts')
    plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data2 = count_vectorizer.fit_transform(articlesAnalysis['animal_wds'])
# Visualise the 50 most common words
plot_50_most_common_words(count_data2, count_vectorizer)
import re
def listToString(s):  
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(s)) 

articlesAnalysisTiger= articlesAnalysis[articlesAnalysis['animal_wds'].str.contains("tiger")]

articlesAnalysisTiger['animal_sentence'] = ''
for i in range(0,len(articlesAnalysisTiger)):
    articlesAnalysisTiger['animal_sentence'].iloc[i] = listToString(re.findall(r"([^.]*? tiger[^.]*\.)",'.' + articlesAnalysisTiger['combined'].iloc[i])).lower()

articlesAnalysisTiger
import re
def listToString(s):  
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(s)) 

articlesAnalysisBat= articlesAnalysis[articlesAnalysis['animal_wds'].str.contains("bat")]

articlesAnalysisBat['animal_sentence'] = ''
for i in range(0,len(articlesAnalysisBat)):
    articlesAnalysisBat['animal_sentence'].iloc[i] = listToString(re.findall(r"([^.]*? bats [^.]*\.)",'.' + articlesAnalysisBat['combined'].iloc[i])).lower()

articlesAnalysisBat
import re
def listToString(s):  
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(s)) 

articlesAnalysisPangolin= articlesAnalysis[articlesAnalysis['combined'].str.contains("pangolin")]

articlesAnalysisPangolin['animal_sentence'] = ''
for i in range(0,len(articlesAnalysisPangolin)):
    articlesAnalysisPangolin['animal_sentence'].iloc[i] = listToString(re.findall(r"([^.]*? pangolin[^.]*\.)",'.' + articlesAnalysisPangolin['combined'].iloc[i])).lower()

articlesAnalysisPangolin
import re
def listToString(s):  
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(s)) 

articlesAnalysisCat= articlesAnalysis[articlesAnalysis['animal_wds'].str.contains("cat")]

articlesAnalysisCat['animal_sentence'] = ''
for i in range(0,len(articlesAnalysisCat)):
    articlesAnalysisCat['animal_sentence'].iloc[i] = listToString(re.findall(r"([^.]*? cat [^.]*\.)",'.' + articlesAnalysisCat['combined'].iloc[i])).lower()

articlesAnalysisCat
articlesAnalysisTigerCov = articlesAnalysisTiger[articlesAnalysisTiger['animal_sentence'].str.contains("cov" or "corona")]
articlesAnalysisTigerCov
# Join the different processed titles together.
animal_string = ''.join(list(articlesAnalysisBat['animal_sentence'].values))
# Create a WordCloud object
wordcloud = WordCloud(width = 600, height = 400, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', collocations=False)
# Generate a word cloud
wordcloud.generate(animal_string)
# Visualize the word cloud
wordcloud.to_image()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(articlesAnalysisBat['animal_sentence'])
# Visualise the 50 most common words
plot_50_most_common_words(count_data, count_vectorizer)
from sklearn.feature_extraction.text import TfidfTransformer
 
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(count_data)

docs_test=articlesAnalysisBat['animal_sentence'].tolist()

def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
feature_names=count_vectorizer.get_feature_names()
 
# get the document that we want to extract keywords from
doc=listToString(docs_test)
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(count_vectorizer.transform([doc]))
 
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
 
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
# now print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])

from collections import Counter

evidence = ['backyard*livestock', 'wet markets', 'meat markets','seafood markets', 'bit*', 
                                      'laboratory*accident', 'fairs', 'petting*zoo',  'trading*wild animal', 
                                      'destruction*habitat', 'wild animal*food', 'lifestock pathogens','genetic mutations',
                                      'animal testing', ' hunting ', 'industrial*farming', ' pet ', 'butcher', ' eat ', ' meat ']
from operator import itemgetter
articlesAnalysis['evidence_wds'] = articlesAnalysis['evidence_wds'].str.replace('domestic animal','')
flat_list = [item for sublist in articlesAnalysis['evidence_wds'].str.split(',') for item in sublist]
flat_list.remove('')
test_dict = Counter(flat_list)
del test_dict['']
test_dict
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')


#explsion
explode = (0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05)
 
plt.pie(test_dict.values(), labels=test_dict, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = explode)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Spillover Source')
plt.tight_layout()
plt.show()

articlesAnalysis['animal_wds'] = articlesAnalysis['animal_wds'].str.replace('animal','')
#articlesAnalysisAnimal = articlesAnalysis[articlesAnalysis['animal_wds'] == "bat "]
#articlesAnalysisAnimal = articlesAnalysis[articlesAnalysis['animal_wds'].str.contains("dog")]
articlesAnalysisAnimal = articlesAnalysis[articlesAnalysis['evidence_wds'].str.contains("meat")]
articlesAnalysisAnimal = articlesAnalysisAnimal[articlesAnalysisAnimal['evidence_wds'].str.contains(" eat ")]
articlesAnalysisAnimal = articlesAnalysisAnimal[articlesAnalysisAnimal['spillover_wds'].str.contains("zoono")]
articlesAnalysisAnimal
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data2 = count_vectorizer.fit_transform(articlesAnalysisAnimal['animal_wds'])
# Visualise the 50 most common words
plot_50_most_common_words(count_data2, count_vectorizer)
import matplotlib as mpl
from matplotlib.pyplot import figure
mpl.rcParams['font.size'] = 9.0
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

flat_list = [item for sublist in articlesAnalysisAnimal['evidence_wds'].str.split(',') for item in sublist]
flat_list.remove('')
test_dict = Counter(flat_list)
del test_dict['']

#explsion
explode = (0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05)
 
plt.pie(test_dict.values(), labels=test_dict, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = None, textprops={'fontsize': 8})
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Spillover Source')
plt.tight_layout()
plt.show()
import datetime as dt
articlesAnalysis['publish_time'] = pd.to_datetime(articlesAnalysis['publish_time'])
articlesAnalysis.head()
import datetime as dt

articlesAnalysis3 = articlesAnalysis[articlesAnalysis['evidence_wds'].str.contains(" eat ")]
articlesAnalysis3 = articlesAnalysis3[articlesAnalysis3['publish_time'].dt.year > 2018]
articlesAnalysis3 = articlesAnalysis3[~articlesAnalysis3['evidence_wds'].str.contains("bit")]
articlesAnalysis3 = articlesAnalysis3[articlesAnalysis3['spillover_wds'].str.contains("zoono")]

articlesAnalysis3

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data4 = count_vectorizer.fit_transform(articlesAnalysis3['animal_wds'])
# Visualise the 50 most common words
plot_50_most_common_words(count_data4, count_vectorizer)
articlesAnalysis2 = articlesAnalysis[articlesAnalysis['combined'].str.contains("zoono")]
articlesAnalysis2

from sklearn.decomposition import LatentDirichletAllocation as LDA

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(articlesAnalysis2['combined']) 
# Helper function
def print_topics(model, count_vectorizer, n_top_words):
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(model.components_):
        print("\nTopic #%d:" % topic_idx)
        print(" ".join([words[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        
# Tweak the two parameters below
number_topics = 10
number_words = 10
# Create and fit the LDA model
lda = LDA(n_components=number_topics, n_jobs=-1)
lda.fit(count_data)
# Print the topics found by the LDA model
print("Topics found via LDA:")
print_topics(lda, count_vectorizer, number_words)

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data = count_vectorizer.fit_transform(articlesAnalysis2['combined'])
# Visualise the 50 most common words
plot_50_most_common_words(count_data, count_vectorizer)
from sklearn.feature_extraction.text import TfidfTransformer
 
tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
tfidf_transformer.fit(count_data)
docs_test=articlesAnalysis2['combined'].tolist()
def sort_coo(coo_matrix):
    tuples = zip(coo_matrix.col, coo_matrix.data)
    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
 
def extract_topn_from_vector(feature_names, sorted_items, topn=10):
    """get the feature names and tf-idf score of top n items"""
    
    #use only topn items from vector
    sorted_items = sorted_items[:topn]
 
    score_vals = []
    feature_vals = []
    
    # word index and corresponding tf-idf score
    for idx, score in sorted_items:
        
        #keep track of feature name and its corresponding score
        score_vals.append(round(score, 3))
        feature_vals.append(feature_names[idx])
 
    #create a tuples of feature,score
    #results = zip(feature_vals,score_vals)
    results= {}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    
    return results
feature_names=count_vectorizer.get_feature_names()
 
# get the document that we want to extract keywords from
doc=docs_test[0]
 
#generate tf-idf for the given document
tf_idf_vector=tfidf_transformer.transform(count_vectorizer.transform([doc]))
 
#sort the tf-idf vectors by descending order of scores
sorted_items=sort_coo(tf_idf_vector.tocoo())
 
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,10)
 
# now print the results
print("\n=====Doc=====")
print(doc)
print("\n===Keywords===")
for k in keywords:
    print(k,keywords[k])
 
def find_africa_wds(content, wds = ['Africa','Burundi', 'Comoros', 'Djibouti', 'Eritrea', 'Ethiopia', 'Kenya', 'Madagascar', 'Malawi', 'Mauritius', 'Mayotte’,  ‘Mozambique', 'Reunion', 'Rwanda', 'Seychelles', 'Somalia', 'Tanzania', 'United Republic of Uganda', 'Zambia', 'Zimbabwe', 'Angola', 'Cameroon', 'Chad', 'Congo', 'Algeria', 'Egypt', 'Libyan Arab Jamahiriya', 'Morroco', 'South Sudan', 'Sudan', 'Tunisia', 'Western Sahara', 'Botswana', 'Eswatini’, ’Swaziland', 'Lesotho', 'Namibia', 'South Africa', 'Benin', 'Burkina Faso', 'Cape Verde', 'Ivory Coast', 'Gambia', 'Ghana', 'Guinea', 'Guinea-Bissau', 'Liberia', 'Mali', 'Mauritania', 'Niger', 'Nigeria', 'Saint Helena', 'Senegal', 'Sierra Leone', 'Togo']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found
def find_asia_wds(content, wds = ['Asia','Afganistan', 'Armenia', 'Azerbaijan', 'Bangladesh', 'Bhutan', 'Brunei Darussalam', 'Cambodia', 'China', 'Georgia', 'Hong Kong', 'India', 'Indonesia', 'Japan', 'Kazakhstan', 'North Korea’, “South Korea', 'Kyrgyzstan', 'Laos', 'Macao', 'Malaysia', 'Maldives', 'Mongolia', 'Myanmar', 'Nepal', 'Pakistan', 'Phillipines', 'Singapore', 'Sri Lanka', 'Taiwan', 'Tajikistan', 'Thailand', 'Turkmenistan', 'Uzbekistan', 'Vietnam']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found
def find_america_wds(content, wds = ['Bermuda', 'Canada', 'Greenland', 'United States', 'U.S.A.', 'USA', 'US', 'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela', 'Anguilla', 'Antigua', 'Barbuda', 'Aruba', 'Bahamas', 'Barbados', 'Bonaire', 'British Virgin Islands', 'Cayman Islands', 'Cuba', 'Curaçao', 'Dominican Republic', 'Grenada', 'Guadeloupe', 'Haiti', 'Jamaica', 'Martinique', 'Monserrat', 'Puerto Rico', 'Saint Lucia', 'Saint Martin', 'Saint Vincent and the Grenadines', 'Sint Maarten', 'Trinidad and Tobago', 'Turks and Caicos Islands', 'Virgin Islands (US)', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Mexico', 'Nicaragua', 'Panama']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found

def find_europe_wds(content, wds = ['Albania', 'Andorra', 'Belarus', 'Bosnia', 'Croatia', 'European Union', 'Faroe Islands', 'Gibraltar’,  ‘Iceland', 'Jersey', 'Kosovo', 'Liechtenstein', 'Moldova', 'Monaco', 'Montenegro', 'North Macedonia', 'Norway', 'Russia', 'San Marino', 'Serbia', 'Switzerland', 'Turkey', 'Ukraine']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found

def find_middleeast_wds(content, wds = ['Bahrain', 'Iraq', 'Iran', 'Israel', 'Jordan', 'Kuwait', 'Lebanon', 'Oman', 'Palestine', 'Qatar', 'Saudi Arabia', 'Syria', 'United Arab Emirates', 'Yemen']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found

def find_oceania_wds(content, wds = ['Australia', 'Fiji', 'French Polynesia', 'Guam', 'Kiribati', 'Marshall Islands', 'Micronesia', 'New Caledonia', 'New Zealand', 'Papua New Guinea', 'Samoa', 'Samoa, American', 'Solomon, Islands', 'Tonga', 'Vanuatu']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ' '
    return found
articlesAnalysis['africa'] = articlesAnalysis['combined'].apply(find_africa_wds)
articlesAnalysis['africa'].replace('', np.nan, inplace=True)
articlesAnalysis['asia'] = articlesAnalysis['combined'].apply(find_asia_wds)
articlesAnalysis['asia'].replace('', np.nan, inplace=True)
articlesAnalysis['america'] = articlesAnalysis['combined'].apply(find_america_wds)
articlesAnalysis['america'].replace('', np.nan, inplace=True)
articlesAnalysis['europe'] = articlesAnalysis['combined'].apply(find_europe_wds)
articlesAnalysis['europe'].replace('', np.nan, inplace=True)
articlesAnalysis['middleeast'] = articlesAnalysis['combined'].apply(find_middleeast_wds)
articlesAnalysis['middleeast'].replace('', np.nan, inplace=True)
articlesAnalysis['oceania'] = articlesAnalysis['combined'].apply(find_oceania_wds)
articlesAnalysis['oceania'].replace('', np.nan, inplace=True)
articlesAnalysis

articlesAnalysisAmerica = articlesAnalysis.dropna(subset=['america'])
articlesAnalysisAmerica

articlesAnalysisUSA = articlesAnalysisAmerica[articlesAnalysisAmerica['america'].str.contains('USA' or 'United States')]
articlesAnalysisUSA = articlesAnalysisUSA[articlesAnalysisUSA[['africa', 'asia', 'europe', 'middleeast','oceania']].isnull().all(1)]
articlesAnalysisUSA
flat_list = [item for sublist in articlesAnalysisUSA['evidence_wds'].str.split(',') for item in sublist]
flat_list.remove('')
test_dict = Counter(flat_list)
del test_dict['']
test_dict
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
#explsion
explode = (0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05)
 
plt.pie(test_dict.values(), labels=test_dict, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = None)
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Spillover Source')
plt.tight_layout()
plt.show()
# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data4 = count_vectorizer.fit_transform(articlesAnalysisUSA['animal_wds'])
# Visualise the 50 most common words
plot_50_most_common_words(count_data4, count_vectorizer)
### Filtering Dataset

articlesAnalysisRiskReduction = articlesAnalysis[articlesAnalysis['combined'].str.contains('risk reduction')]
articlesAnalysisRiskReduction['risk_reduction_sentence'] = ''
for i in range(0,len(articlesAnalysisRiskReduction)):
    articlesAnalysisRiskReduction['risk_reduction_sentence'].iloc[i] = listToString(re.findall(r"([^.]*?[^.]*?risk reduction[^.]*\.[^.]*\.)",'.' + articlesAnalysisRiskReduction['combined'].iloc[i])).lower()
articlesAnalysisRiskReduction['risk_reduction_sentence'].replace('', np.nan, inplace=True)
articlesAnalysisRiskReduction = articlesAnalysisRiskReduction.dropna(subset=['risk_reduction_sentence'])
articlesAnalysisRiskReduction
### Keywords Visualization

# Initialise the count vectorizer with the English stop words
count_vectorizer = CountVectorizer(stop_words='english')
# Fit and transform the processed titles
count_data10 = count_vectorizer.fit_transform(articlesAnalysisRiskReduction['risk_reduction_sentence'])
# Visualise the 50 most common words
plot_50_most_common_words(count_data10, count_vectorizer)

# Join the different processed titles together.
animal_string = ''.join(list(articlesAnalysisRiskReduction['risk_reduction_sentence'].values))
# Create a WordCloud object
wordcloud = WordCloud(width = 600, height = 400, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', collocations=False)
# Generate a word cloud
wordcloud.generate(animal_string)
# Visualize the word cloud
wordcloud.to_image()
### Filtering Dataset

def find_socioeconomic_wds(content, wds = ['socioeconomic','economy', 'behavioral risk', 'unemployment', 'population density', 'school closure', 'daily wage', 'discrimination', 'racism', 'religion', 'financial status', 'education']):
    found = ''
    for w in wds:
        if w in content:
            found += w + ', '
    return found

articlesAnalysis['social_wds'] = articlesAnalysis['combined'].apply(find_socioeconomic_wds)
articlesAnalysis['social_wds'].replace('', np.nan, inplace=True)
articlesAnalysisSocial = articlesAnalysis.dropna(subset=['social_wds'])
articlesAnalysisSocio = articlesAnalysisSocial[articlesAnalysisSocial['social_wds'].str.contains('socioeconomic')]
articlesAnalysisSocio
### Keywords Visualization

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

from operator import itemgetter
flat_list = [item for sublist in articlesAnalysisSocial['social_wds'].str.split(', ') for item in sublist]
test_dict = Counter(flat_list)
del test_dict['']

#explsion
explode = (0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05)
 
plt.pie(test_dict.values(), labels=test_dict, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode = None, textprops={'fontsize': 8})
#draw circle
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
plt.title('Socioeconomic Keywords')
plt.tight_layout()
plt.show()
test_dict
import re
def listToString(s):  
    # initialize an empty string 
    str1 = " " 
    # return string   
    return (str1.join(s)) 

articlesAnalysisSocio['socioeconomic_sentence'] = ''
for i in range(0,len(articlesAnalysisSocio)):
    articlesAnalysisSocio['socioeconomic_sentence'].iloc[i] = listToString(re.findall(r"([^.]*?socioeconomic[^.]*\.[^.]*\.[^.]*\.[^.]*\.)",'.' + articlesAnalysisSocio['combined'].iloc[i])).lower()
articlesAnalysisSocio['socioeconomic_sentence'].replace('', np.nan, inplace=True)
articlesAnalysisSocialEco = articlesAnalysisSocio.dropna(subset=['socioeconomic_sentence'])
articlesAnalysisSocialEco

articlesAnalysisSocio['socioeconomic_sentence'].iloc[0]

# Join the different processed titles together.
animal_string = ''.join(list(articlesAnalysisSocio['socioeconomic_sentence'].values))
# Create a WordCloud object
wordcloud = WordCloud(width = 600, height = 400, background_color="white", max_words=5000, contour_width=3, contour_color='steelblue', collocations=False)
# Generate a word cloud
wordcloud.generate(animal_string)
# Visualize the word cloud
wordcloud.to_image()

fin_df = pd.read_csv('/kaggle/input/task3-data/fin_df.csv')
fin_df.head()
fin_df['title'].fillna("NoTitle", inplace = True)
fin_df['abstract'].fillna("NoAbstract", inplace = True)
fin_df['full_text'].fillna("NoText", inplace = True)
fin_df['combined'] = 'Title : '+fin_df['title'] + '; Abstract ' + fin_df['abstract'] + '; Full Text ' + fin_df['full_text']


# Load the BERT model. Various models trained on Natural Language Inference (NLI) https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md and 
# Semantic Textual Similarity are available https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md

model = SentenceTransformer('bert-base-nli-mean-tokens')
#Uncomment line below if you want to embed
#content = [re.sub(' \n\n\n ','',x) for x in fin_df['full_text'].to_list()]

### Embed the article contents
'''
embedding = model.encode(content, show_progress_bar=True)
'''
#Save embeddings
'''

with open('full_text_embeddings.pkl', 'wb') as embed:
    pickle.dump(embedding, embed)
'''
with open('/kaggle/input/task3-data/full_text_embeddings.pkl','rb') as f:
    embedding = pickle.load(f)
queries = ['How is livestock affected due to Corona virus?',
          'How are farmers afftected due to Coronoa vurus?',
          'How is the spread of corona virus?',
          'Has corona virus infected animals?',
          'How does corona virus transfer?']
query_embeddings = model.encode(queries)
fin_df.head()
df = pd.DataFrame(columns=['Query','Cosine Similarity','Summary','Article Full Text'])

import scipy as sc
top_n_selects = 1
for query, query_embedding in zip(queries, query_embeddings):
    distances = sc.spatial.distance.cdist([query_embedding], embedding, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])
    
    
    
    print('Query : ',query)
    print('###########################################')
    
    for idx, distance in results[0:top_n_selects]:
        print("\nCosine Similarity (Score: %.4f)" % (1-distance),"\n")
        
        body = fin_df['combined'][idx].strip() 
        summary_model = Summarizer()
        result = summary_model(body, min_length=60)
        summary = ''.join(result)
        print('Summary:',summary)
        print('\n')
        print('Article:',body)
        
        similarity = (1-distance)
        to_append = [query, similarity, summary, body]
        a_series = pd.Series(to_append, index = df.columns)
        df = df.append(a_series, ignore_index=True)

        
        print('_________________________________________')
df.head()
'''
queries = []
query = input('Enter you query:')
queries.append(query)
queries
'''
# Load the BERT model. Various models trained on Natural Language Inference (NLI) https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/nli-models.md and 
# Semantic Textual Similarity are available https://github.com/UKPLab/sentence-transformers/blob/master/docs/pretrained-models/sts-models.md

#model = SentenceTransformer('bert-base-nli-mean-tokens')
#query_embeddings = model.encode(queries)
'''
with open('/kaggle/input/task3-data/full_text_embeddings.pkl','rb') as f:
    embedding = pickle.load(f)
'''
#summary_model = Summarizer()
#df = pd.DataFrame(columns=['Query','Cosine Similarity','Summary','Article'])
'''
top_n_selects = 2
import scipy as sc

for query, query_embedding in zip(queries, query_embeddings):
    distances = sc.spatial.distance.cdist([query_embedding], embedding, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print('Query : ',query)
    print('###########################################')
    
    for idx, distance in results[0:top_n_selects]:
        print("\nCosine Similarity (Score: %.4f)" % (1-distance),"\n")
        
        body = fin_df['combined'][idx].strip() 
        summary_model = Summarizer()
        result = summary_model(body, min_length=60)
        summary = ''.join(result)
        similarity = (1-distance)
        to_append = [query, similarity, summary, body]
        a_series = pd.Series(to_append, index = df.columns)
        df = df.append(a_series, ignore_index=True
'''
#df.head()