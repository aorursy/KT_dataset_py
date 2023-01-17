# Please switch on the TPU before running these lines.



!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py

!python pytorch-xla-env-setup.py --apt-packages libomp5 libopenblas-dev
# Imports required to use TPUs with Pytorch.

# https://pytorch.org/xla/release/1.5/index.html



import torch_xla

import torch_xla.core.xla_model as xm
import pandas as pd

import numpy as np

import os

import gc



import random



import torch

import torch.nn as nn

import torch.optim as optim

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



# set a seed value

torch.manual_seed(555)



from sklearn.utils import shuffle

from sklearn.metrics import roc_auc_score, accuracy_score



import transformers

from transformers import BertTokenizer, BertForSequenceClassification 

from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

from transformers import AdamW



import warnings

warnings.filterwarnings("ignore")





print(torch.__version__)
from transformers import BertTokenizer



# Instantiate the Bert tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
# How many words (tokens) does Bert have in it's vocab?



len(tokenizer.vocab)
# The vocab is an ordered dictionary - key/value pairs.

# This is how to see which tokens are associated with a particular word.



bert_vocab = tokenizer.vocab



print(bert_vocab['[CLS]'])

print(bert_vocab['[SEP]'])

print(bert_vocab['[PAD]'])



print(bert_vocab['hello'])

print(bert_vocab['world'])
# Given a token, this is how to see what word is associated with that token.



bert_keys = []



for token in tokenizer.vocab.keys():

    

    bert_keys.append(token)

    

    

print(bert_keys[101])

print(bert_keys[102])

print(bert_keys[0])



print(bert_keys[7592])

print(bert_keys[2088])
# Instantiate a Bert model



# model = BertForSequenceClassification.from_pretrained(

#               'bert-base-multilingual-uncased', 

#               num_labels = 3, # The number of output labels. 2 for binary classification.  

#               output_attentions = False,

#               output_hidden_states = False

#               )







# outputs = model(input_ids=b_input_ids, 

#               token_type_ids=b_token_type_ids, 

#               attention_mask=b_input_mask,

#               labels=b_labels)







# These are the model inputs:



#   input_ids (type: torch tensor)

#   token_type_ids (type: torch tensor)

#   attention_mask (type: torch tensor)

#   labels (type: torch tensor)

# 1. input_ids

# -------------



# The input_ids are the sentence or sentences represented as tokens. 

# There are a few BERT special tokens that one needs to take note of:



# [CLS] - Classifier token, value: 101

# [SEP] - Separator token, value: 102

# [PAD] - Padding token, value: 0



# Bert expects every row in the input_ids to have the special tokens included as follows:



# For one sentence as input:

# [CLS] ...word tokens... [SEP]



# For two sentences as input:

# [CLS] ...sentence1 tokens... [SEP]..sentence2 tokens... [SEP]





# This is an example of an encoded sentence with padding (token value: 0) added. 

# We add padding (or truncate sentences) because each row in an input batch needs 

# to have the same length. The max allowed length is 512.



# [101, 7592, 2045, 1012,  102,    0,    0,    0,    0,    0]







# 2. token_type_ids

# ------------------



# token_type_ids are used when there are two sentences that need to be part of the input. 

# The token type ids indicate which tokens are part of sentence1 and which are part of sentence2.



# This is an example:



# [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]



# The first set of zeros identify all tokens that are part of the first sentence. 

# The ones identify all tokens that are part of the second sentence. 

# The zeros on the right are the padding.







# 3. attention_mask

# ------------------



# The attention mask has the same length as the input_ids. 

# It tells the model which tokens in the input_ids are words and which are padding. 

# 1 indicates a word (or special token) and 0 indicates padding.



# For example:

# Tokens: [101, 7592, 2045, 1012,  102,    0,    0,    0,    0,    0]

# Attention mask: [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]







# 4. labels

# ----------



# The label (target) for each row in the input_ids. 

# The labels are integers representing each target class e.g. 1, 2, 3 etc.



# For example if we have three target classes (0, 1 and 2) then

# the labels could look like this for a batch size of 8:



# [0, 2, 0, 1, 2, 0, 3, 1]

# Batch size is 8.





# outputs = model(input_ids=b_input_ids, 

#               token_type_ids=b_token_type_ids, 

#               attention_mask=b_input_mask,

#               labels=b_labels)





# outputs

# ........



# The output is a tuple: (loss, preds)

# Type: torch tensor





# (tensor(1.1095, device='xla:1', grad_fn=<NllLossBackward>),

#  tensor([[ 0.4005, -0.0222,  0.1946],

#          [ 0.1117, -0.1652,  0.0208],

#          [ 0.3866, -0.0635,  0.1842],

#          [ 0.0423, -0.1887,  0.0691],

#          [ 0.2817, -0.1092,  0.1111],

#          [ 0.2353, -0.1156,  0.0977],

#          [ 0.1253, -0.1821,  0.0402],

#          [ 0.0879, -0.1970,  0.0718]], device='xla:1', grad_fn=<AddmmBackward>))







# outputs[0]

# ..........



# tensor(1.1095, device='xla:1', grad_fn=<NllLossBackward>)





# outputs[0].item()

# ..................



# 1.109534740447998





# outputs[1]

# ..........



# tensor([[ 0.4005, -0.0222,  0.1946],

#         [ 0.1117, -0.1652,  0.0208],

#         [ 0.3866, -0.0635,  0.1842],

#         [ 0.0423, -0.1887,  0.0691],

#         [ 0.2817, -0.1092,  0.1111],

#         [ 0.2353, -0.1156,  0.0977],

#         [ 0.1253, -0.1821,  0.0402],

#         [ 0.0879, -0.1970,  0.0718]], device='xla:1', grad_fn=<AddmmBackward>)

# Batch size is 8.





# preds = model(input_ids=b_input_ids, 

#               token_type_ids=b_token_type_ids, 

#               attention_mask=b_input_mask,

#               )





# preds

# ------



# The output is a tuple with only one value: (preds,)

# Type: torch tensor



# (tensor([[ 0.4005, -0.0222,  0.1946],

#          [ 0.1117, -0.1652,  0.0208],

#          [ 0.3866, -0.0635,  0.1842],

#          [ 0.0423, -0.1887,  0.0691],

#          [ 0.2817, -0.1092,  0.1111],

#          [ 0.2353, -0.1156,  0.0977],

#          [ 0.1253, -0.1821,  0.0402],

#          [ 0.0879, -0.1970,  0.0718]], device='xla:1', grad_fn=<AddmmBackward>),)







# preds[0]

# --------



# tensor([[ 0.4005, -0.0222,  0.1946],

#         [ 0.1117, -0.1652,  0.0208],

#         [ 0.3866, -0.0635,  0.1842],

#         [ 0.0423, -0.1887,  0.0691],

#         [ 0.2817, -0.1092,  0.1111],

#         [ 0.2353, -0.1156,  0.0977],

#         [ 0.1253, -0.1821,  0.0402],

#         [ 0.0879, -0.1970,  0.0718]], device='xla:1', grad_fn=<AddmmBackward>)

MAX_LEN = 10 # This value could be set as 256, 512 etc.



sentence1 = 'Hello there.'



encoded_dict = tokenizer.encode_plus(

            sentence1,                      # Sentence to encode.

            add_special_tokens = True,      # Add '[CLS]' and '[SEP]'

            max_length = MAX_LEN,           # Pad or truncate.

            pad_to_max_length = True,

            return_attention_mask = True,   # Construct attn. masks.

            return_tensors = 'pt',          # Return pytorch tensors.

           )





encoded_dict
# These have already been converted to torch tensors.

input_ids = encoded_dict['input_ids'][0]

token_type_ids = encoded_dict['token_type_ids'][0]

att_mask = encoded_dict['attention_mask'][0]



print(input_ids)

print(token_type_ids)

print(att_mask)
MAX_LEN = 15



sentence1 = 'Hello there.'

sentence2 = 'How are you?'



encoded_dict = tokenizer.encode_plus(

            sentence1, sentence2,           # Sentences to encode.

            add_special_tokens = True,      # Add '[CLS]' and '[SEP]'

            max_length = MAX_LEN,           # Pad or truncate.

            pad_to_max_length = True,

            return_attention_mask = True,   # Construct attn. masks.

            return_tensors = 'pt',          # Return pytorch tensors.

           )





encoded_dict
input_ids = encoded_dict['input_ids'][0]

token_type_ids = encoded_dict['token_type_ids'][0]

att_mask = encoded_dict['attention_mask'][0]



# These are torch tensors.

print(input_ids)

print(token_type_ids)

print(att_mask)
# https://huggingface.co/transformers/main_classes/tokenizer.html

# skip_special_tokens – if this is set to True, then special tokens will be replaced.



# Note that do_lower_case=True in the tokenizer.

# This is why all text is lower case.



a = tokenizer.decode(input_ids,

                skip_special_tokens=False)



b = tokenizer.decode(input_ids,

                skip_special_tokens=True)



print(a)

print(b)
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification



MODEL_TYPE = 'xlm-roberta-base'



tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
# Check the vocab size



tokenizer.vocab_size
# What are the special tokens



tokenizer.special_tokens_map
print('bos_token_id <s>:', tokenizer.bos_token_id)

print('eos_token_id </s>:', tokenizer.eos_token_id)

print('sep_token_id </s>:', tokenizer.sep_token_id)

print('pad_token_id <pad>:', tokenizer.pad_token_id)
# from transformers import XLMRobertaForSequenceClassification



# MODEL_TYPE = 'xlm-roberta-base'



# model = XLMRobertaForSequenceClassification.from_pretrained(

#                  MODEL_TYPE, 

#                  num_labels = 3 # The number of output labels. 2 for binary classification.

#               )





# outputs = model(input_ids=b_input_ids, 

#                 attention_mask=b_input_mask, 

#                 labels=b_labels)







# These are the model inputs:

#   input_ids (type: torch tensor)

#   attention_mask (type: torch tensor)

#   labels (type: torch tensor)
# 1. input_ids

# -------------



# The input_ids are the sentence or sentences represented as tokens. 

# These are special tokens:



# bos_token_id <s>: 0

# eos_token_id </s>: 2

# sep_token_id </s>: 2

# pad_token_id <pad>: 1

    

    

# XLM-RoBERTa expects every row in the input_ids to have the special tokens included as follows:



# For one sentence as input:

# <s> ...word tokens... </s>



# For two sentences as input:<br>

# <s> ...sentence1 tokens... </s></s>..sentence2 tokens... </s>





# This is an example of an encoded sentence with padding (pad token value: 1). 



# [0, 35378, 2685, 5, 2, 1, 1, 1, 1, 1]









# 2. token_type_ids

# ------------------



# XLM-RoBERTa does not use token_type_ids like BERT does.

# Therefore, there's no need to create token_type_ids.







# 3. attention_mask

# ------------------



# The attention mask has the same length as the input_ids. 

# It tells the model which tokens in the input_ids are works and which are padding. 

# 1 indicates a word (or special token) and 0 indicates padding.



# For example, the attention mask for the above input_ids is as follows:

# [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]





# 3. labels

# ----------



# The label (target) for each row in the input_ids. 

# The labels are integers representing each target class e.g. 1, 2, 3 etc.



# For example if we have three target classes (0, 1 and 2) then

# the labels could look like this for a batch size of 8:



# [0, 2, 0, 1, 2, 0, 3, 1]
MAX_LEN = 10 # This value could be set as 256, 512 etc.



sentence1 = 'Hello there.'



encoded_dict = tokenizer.encode_plus(

            sentence1,                

            add_special_tokens = True,

            max_length = MAX_LEN,     

            pad_to_max_length = True,

            return_attention_mask = True,  

            return_tensors = 'pt' # return pytorch tensors

       )





encoded_dict
# These have already been converted to torch tensors.

input_ids = encoded_dict['input_ids'][0]

att_mask = encoded_dict['attention_mask'][0]



print(input_ids)

print(att_mask)
MAX_LEN = 15



sentence1 = 'Hello there.'

sentence2 = 'How are you?'



encoded_dict = tokenizer.encode_plus(

            sentence1, sentence2,      

            add_special_tokens = True,

            max_length = MAX_LEN,     

            pad_to_max_length = True,

            return_attention_mask = True,   

            return_tensors = 'pt' # return pytorch tensors

       )





encoded_dict
input_ids = encoded_dict['input_ids'][0]

att_mask = encoded_dict['attention_mask'][0]



# These are torch tensors.

print(input_ids)

print(att_mask)
# input_ids from above



input_ids = encoded_dict['input_ids'][0]



print(input_ids)
# https://huggingface.co/transformers/main_classes/tokenizer.html

# skip_special_tokens – if set to True, will replace special tokens.



a = tokenizer.decode(input_ids,

                skip_special_tokens=False)



b = tokenizer.decode(input_ids,

                skip_special_tokens=True)







print(a)

print(b)
MAX_LEN = 15 # This value could be set as 256, 512 etc.



sentence1 = 'Hello there. How are you? Have a nice day. This is a test?'





encoded_dict = tokenizer.encode_plus(

            sentence1,                

            max_length = MAX_LEN,

            stride=0,

            pad_to_max_length = True,

            return_overflowing_tokens=True,

       )





encoded_dict
MAX_LEN = 15 # This value could be set as 256, 512 etc.



sentence1 = 'Hello there. How are you? Have a nice day. This is a test?'





encoded_dict = tokenizer.encode_plus(

            sentence1,                

            max_length = MAX_LEN,

            stride=3,

            pad_to_max_length = True,

            return_overflowing_tokens=True,

       )





encoded_dict
# Here you can see the overlap.



print(encoded_dict['input_ids'])

print(encoded_dict['overflowing_tokens'])
os.listdir('../input/contradictory-my-dear-watson')
# Load the training data.



path = '../input/contradictory-my-dear-watson/train.csv'

df_train = pd.read_csv(path)



print(df_train.shape)



df_train.head()
# Load the test data.



path = '../input/contradictory-my-dear-watson/test.csv'

df_test = pd.read_csv(path)



print(df_test.shape)



df_test.head()
from sklearn.model_selection import KFold, StratifiedKFold



# shuffle

df = shuffle(df_train)



# initialize kfold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1024)



# for stratification

y = df['label']



# Note:

# Each fold is a tuple ([train_index_values], [val_index_values])

# fold_0, fold_1, fold_2, fold_3, fold_5 = kf.split(df, y)



# Put the folds into a list. This is a list of tuples.

fold_list = list(kf.split(df, y))



train_df_list = []

val_df_list = []



for i, fold in enumerate(fold_list):



    # map the train and val index values to dataframe rows

    df_train = df[df.index.isin(fold[0])]

    df_val = df[df.index.isin(fold[1])]

    

    train_df_list.append(df_train)

    val_df_list.append(df_val)

    

    



print(len(train_df_list))

print(len(val_df_list))
# Display one train fold



df_train = train_df_list[0]



df_train.head()
# Display one val fold



df_val = val_df_list[0]



df_val.head()
MODEL_TYPE = 'bert-base-multilingual-uncased'



NUM_FOLDS = 5



# Saving 5 TPU models will exceed the 4.9GB disk space.

# Therefore, will will only train on 3 folds.

NUM_FOLDS_TO_TRAIN = 3 



L_RATE = 1e-5

MAX_LEN = 256

NUM_EPOCHS = 3

BATCH_SIZE = 32

NUM_CORES = os.cpu_count()



NUM_CORES
# For GPU



#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#print(device)
# For TPU



device = xm.xla_device()



print(device)
from transformers import BertTokenizer



# Load the BERT tokenizer.

print('Loading BERT tokenizer...')

tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE, do_lower_case=True)


class CompDataset(Dataset):



    def __init__(self, df):

        self.df_data = df







    def __getitem__(self, index):



        # get the sentence from the dataframe

        sentence1 = self.df_data.loc[index, 'premise']

        sentence2 = self.df_data.loc[index, 'hypothesis']



        # Process the sentence

        # ---------------------



        encoded_dict = tokenizer.encode_plus(

                    sentence1, sentence2,           # Sentences to encode.

                    add_special_tokens = True,      # Add '[CLS]' and '[SEP]'

                    max_length = MAX_LEN,           # Pad or truncate all sentences.

                    pad_to_max_length = True,

                    return_attention_mask = True,   # Construct attn. masks.

                    return_tensors = 'pt',          # Return pytorch tensors.

               )  

        

        # These are torch tensors already.

        padded_token_list = encoded_dict['input_ids'][0]

        att_mask = encoded_dict['attention_mask'][0]

        token_type_ids = encoded_dict['token_type_ids'][0]

        

        # Convert the target to a torch tensor

        target = torch.tensor(self.df_data.loc[index, 'label'])



        sample = (padded_token_list, att_mask, token_type_ids, target)





        return sample





    def __len__(self):

        return len(self.df_data)

    

    

    

    

    



class TestDataset(Dataset):



    def __init__(self, df):

        self.df_data = df







    def __getitem__(self, index):



        # get the sentence from the dataframe

        sentence1 = self.df_data.loc[index, 'premise']

        sentence2 = self.df_data.loc[index, 'hypothesis']



        # Process the sentence

        # ---------------------



        encoded_dict = tokenizer.encode_plus(

                    sentence1, sentence2,           # Sentence to encode.

                    add_special_tokens = True,      # Add '[CLS]' and '[SEP]'

                    max_length = MAX_LEN,           # Pad or truncate all sentences.

                    pad_to_max_length = True,

                    return_attention_mask = True,   # Construct attn. masks.

                    return_tensors = 'pt',          # Return pytorch tensors.

               )

        

        # These are torch tensors already.

        padded_token_list = encoded_dict['input_ids'][0]

        att_mask = encoded_dict['attention_mask'][0]

        token_type_ids = encoded_dict['token_type_ids'][0]

               



        sample = (padded_token_list, att_mask, token_type_ids)





        return sample





    def __len__(self):

        return len(self.df_data)



df_train = df_train.reset_index(drop=True)

df_val = df_val.reset_index(drop=True)
train_data = CompDataset(df_train)

val_data = CompDataset(df_val)

test_data = TestDataset(df_test)







train_dataloader = torch.utils.data.DataLoader(train_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=True,

                                       num_workers=NUM_CORES)



val_dataloader = torch.utils.data.DataLoader(val_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=True,

                                       num_workers=NUM_CORES)



test_dataloader = torch.utils.data.DataLoader(test_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=False,

                                       num_workers=NUM_CORES)







print(len(train_dataloader))

print(len(val_dataloader))

print(len(test_dataloader))
# Get one train batch



padded_token_list, att_mask, token_type_ids, target = next(iter(train_dataloader))



print(padded_token_list.shape)

print(att_mask.shape)

print(token_type_ids.shape)

print(target.shape)
# Get one val batch



padded_token_list, att_mask, token_type_ids, target = next(iter(val_dataloader))



print(padded_token_list.shape)

print(att_mask.shape)

print(token_type_ids.shape)

print(target.shape)
# Get one test batch



padded_token_list, att_mask, token_type_ids = next(iter(test_dataloader))



print(padded_token_list.shape)

print(att_mask.shape)

print(token_type_ids.shape)
# Load BertForSequenceClassification, the pretrained BERT model with a single 

# linear classification layer on top. 

model = BertForSequenceClassification.from_pretrained(

    MODEL_TYPE, 

    num_labels = 3, 

    output_attentions = False,

    output_hidden_states = False)



# Send the model to the device.

model.to(device)
# Get one train batch



train_dataloader = torch.utils.data.DataLoader(train_data,

                                        batch_size=8,

                                        shuffle=True,

                                       num_workers=NUM_CORES)



batch = next(iter(train_dataloader))



b_input_ids = batch[0].to(device)

b_input_mask = batch[1].to(device)

b_token_type_ids = batch[2].to(device)

b_labels = batch[3].to(device)
outputs = model(b_input_ids, 

                token_type_ids=b_token_type_ids, 

                attention_mask=b_input_mask,

                labels=b_labels)
outputs
# The output is a tuple: (loss, preds)



len(outputs)
# This is the loss.



outputs[0]
# These are the predictions.



outputs[1]
preds = outputs[1].detach().cpu().numpy()



y_true = b_labels.detach().cpu().numpy()

y_pred = np.argmax(preds, axis=1)



y_pred
# This is the accuracy without any fine tuning.



val_acc = accuracy_score(y_true, y_pred)



val_acc
# The loss and preds are Torch tensors



print(type(outputs[0]))

print(type(outputs[1]))
# For info: 

# Think in terms of fold models.

# Fold model 0, for example, is only training on fold 0 in each epoch.

# The same applies to the other fold models.
%%time





# Set a seed value.

seed_val = 1024



random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)







# Store the accuracy scores for each fold model in this list.

# [[model_0 scores], [model_1 scores], [model_2 scores], [model_3 scores], [model_4 scores]]

# [[ecpoch 1, epoch 2, ...], [ecpoch 1, epoch 2, ...], [ecpoch 1, epoch 2, ...], [ecpoch 1, epoch 2, ...], [ecpoch 1, epoch 2, ...]]



# Create a list of lists to store the val acc results.

# The number of items in this list will correspond to

# the number of folds that the model is being trained on.

fold_val_acc_list = []

for i in range(0, NUM_FOLDS):

    

    # append an empty list

    fold_val_acc_list.append([])

    

    

    

    



# For each epoch...

for epoch in range(0, NUM_EPOCHS):

    

    print("\nNum folds used for training:", NUM_FOLDS_TO_TRAIN)

    print('======== Epoch {:} / {:} ========'.format(epoch + 1, NUM_EPOCHS))

    

    # Get the number of folds

    num_folds = len(train_df_list)



    # For this epoch, store the val acc scores for each fold in this list.

    # We will use this list to calculate the cv at the end of the epoch.

    epoch_acc_scores_list = []

    

    # For each fold...

    for fold_index in range(0, NUM_FOLDS_TO_TRAIN):

        

        print('\n== Fold Model', fold_index)

        

        

        # .........................

        # Load the fold model

        # .........................

        

        if epoch == 0:

            

            # define the model

            model = BertForSequenceClassification.from_pretrained(

            MODEL_TYPE, 

            num_labels = 3,       

            output_attentions = False, 

            output_hidden_states = False,

            )

            

            # Send the model to the device.

            model.to(device)

            

            optimizer = AdamW(model.parameters(),

              lr = L_RATE, 

              eps = 1e-8

            )

            

        else:

        

            # Get the fold model

            path_model = 'model_' + str(fold_index) + '.bin'

            model.load_state_dict(torch.load(path_model))



            # Send the model to the device.

            model.to(device)

        

        

        

        # .....................................

        # Set up the train and val dataloaders

        # .....................................

        

        

        # Intialize the fold dataframes

        df_train = train_df_list[fold_index]

        df_val = val_df_list[fold_index]

        

        # Reset the indices or the dataloader won't work.

        df_train = df_train.reset_index(drop=True)

        df_val = df_val.reset_index(drop=True)

    

        # Create the dataloaders

        train_data = CompDataset(df_train)

        val_data = CompDataset(df_val)



        train_dataloader = torch.utils.data.DataLoader(train_data,

                                                batch_size=BATCH_SIZE,

                                                shuffle=True,

                                               num_workers=NUM_CORES)



        val_dataloader = torch.utils.data.DataLoader(val_data,

                                                batch_size=BATCH_SIZE,

                                                shuffle=True,

                                               num_workers=NUM_CORES)

    

    

    



       



        # ========================================

        #               Training

        # ========================================

        

        stacked_val_labels = []

        targets_list = []



        print('Training...')



        # put the model into train mode

        model.train()



        # This turns gradient calculations on and off.

        torch.set_grad_enabled(True)





        # Reset the total loss for this epoch.

        total_train_loss = 0



        for i, batch in enumerate(train_dataloader):



            train_status = 'Batch ' + str(i+1) + ' of ' + str(len(train_dataloader))



            print(train_status, end='\r')





            b_input_ids = batch[0].to(device)

            b_input_mask = batch[1].to(device)

            b_token_type_ids = batch[2].to(device)

            b_labels = batch[3].to(device)



            model.zero_grad()        





            outputs = model(b_input_ids, 

                        token_type_ids=b_token_type_ids, 

                        attention_mask=b_input_mask,

                        labels=b_labels)



            # Get the loss from the outputs tuple: (loss, logits)

            loss = outputs[0]



            # Convert the loss from a torch tensor to a number.

            # Calculate the total loss.

            total_train_loss = total_train_loss + loss.item()



            # Zero the gradients

            optimizer.zero_grad()



            # Perform a backward pass to calculate the gradients.

            loss.backward()

            

            # Clip the norm of the gradients to 1.0.

            # This is to help prevent the "exploding gradients" problem.

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)



            # Use the optimizer to update Weights

            

            # Optimizer for GPU

            # optimizer.step() 

            

            # Optimizer for TPU

            # https://pytorch.org/xla/

            xm.optimizer_step(optimizer, barrier=True)

            

           





        print('Train loss:' ,total_train_loss)





        # ========================================

        #               Validation

        # ========================================



        print('\nValidation...')



        # Put the model in evaluation mode.

        model.eval()



        # Turn off the gradient calculations.

        # This tells the model not to compute or store gradients.

        # This step saves memory and speeds up validation.

        torch.set_grad_enabled(False)





        # Reset the total loss for this epoch.

        total_val_loss = 0





        for j, val_batch in enumerate(val_dataloader):



            val_status = 'Batch ' + str(j+1) + ' of ' + str(len(val_dataloader))



            print(val_status, end='\r')



            b_input_ids = val_batch[0].to(device)

            b_input_mask = val_batch[1].to(device)

            b_token_type_ids = val_batch[2].to(device)

            b_labels = val_batch[3].to(device)      





            outputs = model(b_input_ids, 

                    token_type_ids=b_token_type_ids, 

                    attention_mask=b_input_mask, 

                    labels=b_labels)



            # Get the loss from the outputs tuple: (loss, logits)

            loss = outputs[0]



            # Convert the loss from a torch tensor to a number.

            # Calculate the total loss.

            total_val_loss = total_val_loss + loss.item()



            # Get the preds

            preds = outputs[1]





            # Move preds to the CPU

            val_preds = preds.detach().cpu().numpy()



            # Move the labels to the cpu

            targets_np = b_labels.to('cpu').numpy()



            # Append the labels to a numpy list

            targets_list.extend(targets_np)



            if j == 0:  # first batch

                stacked_val_preds = val_preds



            else:

                stacked_val_preds = np.vstack((stacked_val_preds, val_preds))

                

                

                

        # .........................................

        # Calculate the val accuracy for this fold

        # .........................................      





        # Calculate the validation accuracy

        y_true = targets_list

        y_pred = np.argmax(stacked_val_preds, axis=1)



        val_acc = accuracy_score(y_true, y_pred)

        

        

        epoch_acc_scores_list.append(val_acc)





        print('Val loss:' ,total_val_loss)

        print('Val acc: ', val_acc)

        

        

        # .........................

        # Save the best model

        # .........................

        

        if epoch == 0:

            

            # Save the Model

            model_name = 'model_' + str(fold_index) + '.bin'

            torch.save(model.state_dict(), model_name)

            print('Saved model as ', model_name)

            

        if epoch != 0:

        

            val_acc_list = fold_val_acc_list[fold_index]

            best_val_acc = max(val_acc_list)

            

            if val_acc > best_val_acc:

                # save the model

                model_name = 'model_' + str(fold_index) + '.bin'

                torch.save(model.state_dict(), model_name)

                print('Val acc improved. Saved model as ', model_name)

                

                

                

        # .....................................

        # Save the val_acc for this fold model

        # .....................................

        

        # Note: Don't do this before the above 'Save Model' code or 

        # the save model code won't work. This is because the best_val_acc will

        # become current val accuracy.

                

        # fold_val_acc_list is a list of lists.

        # Each fold model has it's own list corresponding to the fold index.

        # Here we choose a list corresponding to the fold number and append the acc score to that list.

        fold_val_acc_list[fold_index].append(val_acc)

        

            



        # Use the garbage collector to save memory.

        gc.collect()

        

        

    # .............................................................

    # Calculate the CV accuracy score over all folds in this epoch

    # .............................................................   

        

        

    # Print the average val accuracy for all 5 folds

    cv_acc = sum(epoch_acc_scores_list)/NUM_FOLDS_TO_TRAIN

    print("\nCV Acc:", cv_acc)

    

# Check that the models have been saved



!ls
# Display the accuracy scores for each fold model.

# For info: 

# Fold model 0 is only training on fold 0 in each epoch.

# The same applies to the other fold models.



fold_val_acc_list
# Create the dataloader



test_data = TestDataset(df_test)





test_dataloader = torch.utils.data.DataLoader(test_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=False,

                                       num_workers=NUM_CORES)



print(len(test_dataloader))
# ========================================

#               Test Set

# ========================================



print('\nTest Set...')



model_preds_list = []



print('Total batches:', len(test_dataloader))



for fold_index in range(0, NUM_FOLDS_TO_TRAIN):

    

    print('\nFold Model', fold_index)



    # Load the fold model

    path_model = 'model_' + str(fold_index) + '.bin'

    model.load_state_dict(torch.load(path_model))



    # Send the model to the device.

    model.to(device)





    stacked_val_labels = []

    



    # Put the model in evaluation mode.

    model.eval()



    # Turn off the gradient calculations.

    # This tells the model not to compute or store gradients.

    # This step saves memory and speeds up validation.

    torch.set_grad_enabled(False)





    # Reset the total loss for this epoch.

    total_val_loss = 0



    for j, h_batch in enumerate(test_dataloader):



        inference_status = 'Batch ' + str(j + 1)



        print(inference_status, end='\r')



        b_input_ids = h_batch[0].to(device)

        b_input_mask = h_batch[1].to(device)

        b_token_type_ids = h_batch[2].to(device)     





        outputs = model(b_input_ids, 

                token_type_ids=b_token_type_ids, 

                attention_mask=b_input_mask)





        # Get the preds

        preds = outputs[0]





        # Move preds to the CPU

        val_preds = preds.detach().cpu().numpy()

        

        

        # Stack the predictions.



        if j == 0:  # first batch

            stacked_val_preds = val_preds



        else:

            stacked_val_preds = np.vstack((stacked_val_preds, val_preds))



        

    model_preds_list.append(stacked_val_preds)

    

            

print('\nPrediction complete.')        
model_preds_list
# Sum the predictions of all fold models

for i, item in enumerate(model_preds_list):

    

    if i == 0:

        

        preds = item

        

    else:

    

        # Sum the matrices

        preds = item + preds



        

# Average the predictions

avg_preds = preds/(len(model_preds_list))





test_preds = np.argmax(avg_preds, axis=1)
test_preds
# Load the sample submission.

# The row order in the test set and the sample submission is the same.



path = '../input/contradictory-my-dear-watson/sample_submission.csv'



df_sample = pd.read_csv(path)



print(df_sample.shape)



df_sample.head()
# Assign the preds to the prediction column



df_sample['prediction'] = test_preds



df_sample.head()
# Create a submission csv file

df_sample.to_csv('submission.csv', index=False)
# Check that the fold models have been saved.



!ls
# Check the distribution of the predicted classes.



df_sample['prediction'].value_counts()
MODEL_TYPE = 'xlm-roberta-base'





L_RATE = 1e-5

MAX_LEN = 256



NUM_EPOCHS = 3

BATCH_SIZE = 32

NUM_CORES = os.cpu_count()



NUM_CORES
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#print(device)
# Tell PyTorch to use the TPU.    

device = xm.xla_device()



print(device)
df_train = train_df_list[0]



df_train.head()
df_val = val_df_list[0]



df_val.head()
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification



# xlm-roberta-large

print('Loading XLMRoberta tokenizer...')

tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
df_train = df_train.reset_index(drop=True)

df_val = df_val.reset_index(drop=True)
class CompDataset(Dataset):



    def __init__(self, df):

        self.df_data = df







    def __getitem__(self, index):



        # get the sentence from the dataframe

        sentence1 = self.df_data.loc[index, 'premise']

        sentence2 = self.df_data.loc[index, 'hypothesis']



        # Process the sentence

        # ---------------------



        encoded_dict = tokenizer.encode_plus(

                    sentence1, sentence2,           # Sentences to encode.

                    add_special_tokens = True,      # Add the special tokens.

                    max_length = MAX_LEN,           # Pad & truncate all sentences.

                    pad_to_max_length = True,

                    return_attention_mask = True,   # Construct attn. masks.

                    return_tensors = 'pt',          # Return pytorch tensors.

               )

        

        # These are torch tensors.

        padded_token_list = encoded_dict['input_ids'][0]

        att_mask = encoded_dict['attention_mask'][0]

        

        # Convert the target to a torch tensor

        target = torch.tensor(self.df_data.loc[index, 'label'])



        sample = (padded_token_list, att_mask, target)





        return sample





    def __len__(self):

        return len(self.df_data)

    

    

    

    

    



class TestDataset(Dataset):



    def __init__(self, df):

        self.df_data = df







    def __getitem__(self, index):



        # get the sentence from the dataframe

        sentence1 = self.df_data.loc[index, 'premise']

        sentence2 = self.df_data.loc[index, 'hypothesis']



        # Process the sentence

        # ---------------------



        encoded_dict = tokenizer.encode_plus(

                    sentence1, sentence2,           # Sentence to encode.

                    add_special_tokens = True,      # Add the special tokens.

                    max_length = MAX_LEN,           # Pad & truncate all sentences.

                    pad_to_max_length = True,

                    return_attention_mask = True,   # Construct attn. masks.

                    return_tensors = 'pt',          # Return pytorch tensors.

               )

        

        # These are torch tensors.

        padded_token_list = encoded_dict['input_ids'][0]

        att_mask = encoded_dict['attention_mask'][0]

        

               



        sample = (padded_token_list, att_mask)





        return sample





    def __len__(self):

        return len(self.df_data)
train_data = CompDataset(df_train)

val_data = CompDataset(df_val)

test_data = TestDataset(df_test)



train_dataloader = torch.utils.data.DataLoader(train_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=True,

                                       num_workers=NUM_CORES)



val_dataloader = torch.utils.data.DataLoader(val_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=True,

                                       num_workers=NUM_CORES)



test_dataloader = torch.utils.data.DataLoader(test_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=False,

                                       num_workers=NUM_CORES)







print(len(train_dataloader))

print(len(val_dataloader))

print(len(test_dataloader))
# Get one train batch



padded_token_list, att_mask, target = next(iter(train_dataloader))



print(padded_token_list.shape)

print(att_mask.shape)

print(target.shape)
# Get one val batch



padded_token_list, att_mask, target = next(iter(val_dataloader))



print(padded_token_list.shape)

print(att_mask.shape)

print(target.shape)
# Get one test batch



padded_token_list, att_mask = next(iter(test_dataloader))



print(padded_token_list.shape)

print(att_mask.shape)
from transformers import XLMRobertaForSequenceClassification



model = XLMRobertaForSequenceClassification.from_pretrained(

    MODEL_TYPE, 

    num_labels = 3, # The number of output labels. 2 for binary classification.

)



# Send the model to the device.

model.to(device)
# Create a batch of train samples

# We will set a small batch size of 8 so that the model's output can be easily displayed.



train_dataloader = torch.utils.data.DataLoader(train_data,

                                        batch_size=8,

                                        shuffle=True,

                                       num_workers=NUM_CORES)



b_input_ids, b_input_mask, b_labels = next(iter(train_dataloader))



print(b_input_ids.shape)

print(b_input_mask.shape)

print(b_labels.shape)
# Pass a batch of train samples to the model.



batch = next(iter(train_dataloader))



# Send the data to the device

b_input_ids = batch[0].to(device)

b_input_mask = batch[1].to(device)

b_labels = batch[2].to(device)



# Run the model

outputs = model(b_input_ids, 

                        attention_mask=b_input_mask, 

                        labels=b_labels)



# The ouput is a tuple (loss, preds).

outputs
outputs
# The output is a tuple: (loss, preds)



len(outputs)
# This is the loss.



outputs[0]
# These are the predictions.



outputs[1]
preds = outputs[1].detach().cpu().numpy()



y_true = b_labels.detach().cpu().numpy()

y_pred = np.argmax(preds, axis=1)



y_pred
# This is the accuracy without fine tuning.



val_acc = accuracy_score(y_true, y_pred)



val_acc
# The loss and preds are Torch tensors



print(type(outputs[0]))

print(type(outputs[1]))
# Define the optimizer

optimizer = AdamW(model.parameters(),

              lr = L_RATE, 

              eps = 1e-8 

            )
# Create the dataloaders.



train_data = CompDataset(df_train)

val_data = CompDataset(df_val)

test_data = TestDataset(df_test)



train_dataloader = torch.utils.data.DataLoader(train_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=True,

                                       num_workers=NUM_CORES)



val_dataloader = torch.utils.data.DataLoader(val_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=True,

                                       num_workers=NUM_CORES)



test_dataloader = torch.utils.data.DataLoader(test_data,

                                        batch_size=BATCH_SIZE,

                                        shuffle=False,

                                       num_workers=NUM_CORES)







print(len(train_dataloader))

print(len(val_dataloader))

print(len(test_dataloader))
%%time





# Set the seed.

seed_val = 101



random.seed(seed_val)

np.random.seed(seed_val)

torch.manual_seed(seed_val)

torch.cuda.manual_seed_all(seed_val)



# Store the average loss after each epoch so we can plot them.

loss_values = []





# For each epoch...

for epoch in range(0, NUM_EPOCHS):

    

    print("")

    print('======== Epoch {:} / {:} ========'.format(epoch + 1, NUM_EPOCHS))

    



    stacked_val_labels = []

    targets_list = []



    # ========================================

    #               Training

    # ========================================

    

    print('Training...')

    

    # put the model into train mode

    model.train()

    

    # This turns gradient calculations on and off.

    torch.set_grad_enabled(True)





    # Reset the total loss for this epoch.

    total_train_loss = 0



    for i, batch in enumerate(train_dataloader):

        

        train_status = 'Batch ' + str(i) + ' of ' + str(len(train_dataloader))

        

        print(train_status, end='\r')





        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)



        model.zero_grad()        





        outputs = model(b_input_ids, 

                    attention_mask=b_input_mask,

                    labels=b_labels)

        

        # Get the loss from the outputs tuple: (loss, logits)

        loss = outputs[0]

        

        # Convert the loss from a torch tensor to a number.

        # Calculate the total loss.

        total_train_loss = total_train_loss + loss.item()

        

        # Zero the gradients

        optimizer.zero_grad()

        

        # Perform a backward pass to calculate the gradients.

        loss.backward()

        

        

        # Clip the norm of the gradients to 1.0.

        # This is to help prevent the "exploding gradients" problem.

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        

        

        

        # Use the optimizer to update the weights.

        

        # Optimizer for GPU

        # optimizer.step() 

        

        # Optimizer for TPU

        # https://pytorch.org/xla/

        xm.optimizer_step(optimizer, barrier=True)



    

    print('Train loss:' ,total_train_loss)





    # ========================================

    #               Validation

    # ========================================

    

    print('\nValidation...')



    # Put the model in evaluation mode.

    model.eval()



    # Turn off the gradient calculations.

    # This tells the model not to compute or store gradients.

    # This step saves memory and speeds up validation.

    torch.set_grad_enabled(False)

    

    

    # Reset the total loss for this epoch.

    total_val_loss = 0

    



    for j, batch in enumerate(val_dataloader):

        

        val_status = 'Batch ' + str(j) + ' of ' + str(len(val_dataloader))

        

        print(val_status, end='\r')



        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)

        b_labels = batch[2].to(device)      





        outputs = model(b_input_ids, 

                attention_mask=b_input_mask, 

                labels=b_labels)

        

        # Get the loss from the outputs tuple: (loss, logits)

        loss = outputs[0]

        

        # Convert the loss from a torch tensor to a number.

        # Calculate the total loss.

        total_val_loss = total_val_loss + loss.item()

        



        # Get the preds

        preds = outputs[1]





        # Move preds to the CPU

        val_preds = preds.detach().cpu().numpy()

        

        # Move the labels to the cpu

        targets_np = b_labels.to('cpu').numpy()



        # Append the labels to a numpy list

        targets_list.extend(targets_np)



        if j == 0:  # first batch

            stacked_val_preds = val_preds



        else:

            stacked_val_preds = np.vstack((stacked_val_preds, val_preds))



    

    # Calculate the validation accuracy

    y_true = targets_list

    y_pred = np.argmax(stacked_val_preds, axis=1)

    

    val_acc = accuracy_score(y_true, y_pred)

    

    

    print('Val loss:' ,total_val_loss)

    print('Val acc: ', val_acc)





    # Save the Model

    torch.save(model.state_dict(), 'model.pt')

    

    # Use the garbage collector to save memory.

    gc.collect()
for j, batch in enumerate(test_dataloader):

        

        inference_status = 'Batch ' + str(j+1) + ' of ' + str(len(test_dataloader))

        

        print(inference_status, end='\r')



        b_input_ids = batch[0].to(device)

        b_input_mask = batch[1].to(device)





        outputs = model(b_input_ids, 

                attention_mask=b_input_mask)

        

        

        # Get the preds

        preds = outputs[0]





        # Move preds to the CPU

        preds = preds.detach().cpu().numpy()

        

        # Move the labels to the cpu

        targets_np = b_labels.to('cpu').numpy()



        # Append the labels to a numpy list

        targets_list.extend(targets_np)

        

        # Stack the predictions.



        if j == 0:  # first batch

            stacked_preds = preds



        else:

            stacked_preds = np.vstack((stacked_preds, preds))
stacked_preds
# Take the argmax. This returns the column index of the max value in each row.



preds = np.argmax(stacked_preds, axis=1)



preds
# Load the sample submission.

# The row order in the test set and the sample submission is the same.



path = '../input/contradictory-my-dear-watson/sample_submission.csv'



df_sample = pd.read_csv(path)



print(df_sample.shape)



df_sample.head()
# Assign the preds to the prediction column



df_sample['prediction'] = preds



df_sample.head()
# Create a submission csv file

# Note that for this competition the submission file must be named submission.csv.

# Therefore, it won't be possible to submit this csv file for leaderboard scoring.

df_sample.to_csv('xlmroberta_submission.csv', index=False)
# Check that the model has been saved.



!ls
# Check the distribution of the predicted classes.



df_sample['prediction'].value_counts()