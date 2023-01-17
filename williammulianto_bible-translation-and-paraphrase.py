save_loc = "/kaggle/tmp/model"
!mkdir /kaggle/tmp
!mkdir /kaggle/tmp/model
!ls /kaggle/tmp
!pip install git+https://github.com/Maluuba/nlg-eval.git@master
!pip install wandb
# !wandb login 

import pandas as pd
import numpy as np
df_tb = pd.read_csv('../input/indonesia-bible-tb/tb.csv')
df_vmd = pd.read_csv('../input/indonesia-bible-tb/vmd.csv')

df_tb.head(5)
df_tb.rename({'firman': 'input_text'}, axis=1, inplace=True)
df_vmd.rename({'firman': 'target_text'}, axis=1, inplace=True)
df_tb.drop('id',axis=1,inplace=True)
df_vmd.drop('id',axis=1,inplace=True)

df_tb.head(5)
df_train = df_tb.merge(df_vmd,how='outer',left_on=['kitab','pasal','ayat'],right_on=['kitab','pasal','ayat'])
df_tb['prefix'] = 'simplify'
! pip uninstall torch torchvision -y
! pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
!pip install -U transformers
!pip install -U simpletransformers  
train_df=df_train.sample(frac=0.5,random_state=200) #random state is a seed value
test_df=df_train.drop(train_df.index)


df_train.drop(['kitab','pasal','ayat'],inplace=True,axis=1)
df_train.head(5)
!pip install torchsummary
from torchsummary import summary
# from summary import summary

import logging

import transformers
from simpletransformers.t5 import T5Model

from simpletransformers.seq2seq import Seq2SeqModel
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)



model_args = {
    "reprocess_input_data": True,
#     "no_save": True,
    "max_seq_length": 50,
    "train_batch_size": 100,
    "output_dir" : save_loc,
    "num_train_epochs": 10,
    "wandb_project" : "Bible Paraphrase",
    "save_model_every_epoch": False,
    "overwrite_output_dir" :True,
    "max_length": 50,
#     "do_sample" : True,
#     "num_beams" : None,
#     "num_return_sequences" : 3,
#     "top_k" : 50,
#     "top_p" : 0.95,
}

# Initialize model
# model = Seq2SeqModel(
#     encoder_decoder_type="marian",
#     encoder_decoder_name="sshleifer/student_marian_en_ro_6_4",
#     args=model_args,
# )
model = T5Model("t5-base",None, args=model_args)
# Train the model

# Evaluate the model
# results = model.eval_model(eval_df)
train_df['prefix'] = 'paraphrase'

model.train_model(train_df,use_cuda=True)

# Use the model for prediction
print(model.predict(['Dan Allah menamai terang itu siang, dan gelap itu malam. Jadilah petang dan jadilah pagi, itulah hari pertama.']))


test_df.reset_index(inplace=True)
df_train.head(5)
def print_df(df,index):
    print(df.loc[index]['input_text'])
    print(df.loc[index]['target_text'])
    print(model.predict([df.loc[index]['input_text']]))
    
print_df(test_df,6)
from tqdm import tqdm
len(test_df)
# results = model.eval_model(test_df,output_dir='./')
res = model.predict(test_df.loc[:5000]['input_text'].tolist())
df_eval = test_df.head(5001)
df_eval.head(5)
df_eval['pred'] = res
df_eval.head(5)
!du -h /kaggle/tmp/model
df_eval.to_csv('result.csv',index=False)
len(df_eval)
!pip install sacrebleu
import sacrebleu
refs = [df_eval['target_text'].tolist()]
sys = df_eval['pred'].tolist()
bleu = sacrebleu.corpus_bleu(sys, refs)
print(bleu.score)
from nlgeval import NLGEval
nlgeval = NLGEval()  # loads the models
metrics_dict = nlgeval.compute_metrics(df_eval['target_text'], df_eval['pred'])
len(sys)
