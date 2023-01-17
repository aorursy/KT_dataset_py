

import torch

import numpy as np

import transformers

import spacy

nlp = spacy.load('en_core_web_sm')
if torch.cuda.is_available():

    device = torch.device('cuda')
device.type
import pandas as pd

import os

print(os.listdir("../input/training-dataset"))
# Load dataset

sentences = pd.read_csv('../input/training-dataset/cleaned_train_fina.csv')['user_review'].values

labels = pd.read_csv('../input/training-dataset/train.csv')['user_suggestion'].values
#! wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip
#!unzip uncased_L-12_H-768_A-12.zip
#!  'uncased_L-12_H-768_A-12.zip' 'bert-base-uncased'
from transformers import RobertaConfig,RobertaForSequenceClassification,RobertaModel,RobertaTokenizer

# Load tokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=True)
config = RobertaConfig('roberta-base')
# For an example try for single sentence

input_ids = tokenizer.encode(sentences[0])

# these ids come out from pretrained bert embeddings model which we have just loaded and this represent index of unique tokens or in 

# other words it's just a index of each token to represent that particular token. NOTHING MORE THAN THAT.



# But we don't only need input_ids instead we also need attention mask ids where attention mask ids are those ids which differentiate

# padded or truncated ids with original input ids of given sentences.Let's see how attention mask ids look like but how can we get 

# those ids because encode don't contain those ids? so for that we will use it's next version encode_plus BAM!



ids_dictionary = tokenizer.encode_plus(sentences[0],return_attention_mask=True,return_token_type_ids=True,max_length=100,pad_to_max_length=True) # for now we are discarding padding,later i will explain you this with padding also

print('input_ids',ids_dictionary['input_ids'],'\nattention_mask',ids_dictionary['attention_mask']) # for now we didn't do padding

                                                                                                   # but if we do padding then we can differentiate it from input_ids



# So attention mask ids basically differentiating padded sequences with original input sequence by binary representation where

# 1 means true indexes of input tokens where 0 represents padded tokens
import torch
# Now do this for whole training data

input_ids = [] # list of input_ids of all sentences

attention_mask = [] # list of attention_mask of all sentences

for sent in sentences:

    # create a dictionary which will return input_ids and attention_mask of sentences

    # find maximum length of sentence from all sentences

    ids_dictionary = tokenizer.encode_plus(str(sent),add_special_tokens=True,return_tensors='pt',return_attention_mask=True,return_token_type_ids=True,max_length =300 ,pad_to_max_length = True)

    input_ids.append(ids_dictionary['input_ids'])

    attention_mask.append(ids_dictionary['attention_mask'])

    



    

    
from transformers import BertForSequenceClassification,AdamW,BertConfig

classifier = RobertaForSequenceClassification.from_pretrained('roberta-base',num_labels=2,output_attentions=False,output_hidden_states=False)
input_ids
input_ids = torch.cat(input_ids,dim=0)

attention_mask = torch.cat(attention_mask,dim=0)

labels = torch.tensor(labels)
input_ids[:4]
from torch.utils.data import TensorDataset,random_split

dataset = TensorDataset(input_ids,attention_mask,labels)

print(len(dataset))

train_size =int(0.9 * len(dataset))

val_size = len(dataset) - train_size

print(train_size)

train_dataset,val_dataset = random_split(dataset,[int(train_size),int(val_size)])

# print model parameters and transfer model to cuda

params = classifier.named_parameters()

for param in params:

    print('layers',param[0],'parameters_size',param[1].shape)

    
classifier.to(device)
from sklearn.metrics import f1_score

# convert given dataset into tensordataset which consist of input sente

from torch.utils.data import DataLoader,random_split,RandomSampler,SequentialSampler

train_loader = DataLoader(train_dataset,batch_size = 32 , sampler=RandomSampler(train_dataset))

val_loader = DataLoader(val_dataset,batch_size=32,

                       sampler= SequentialSampler(val_dataset)) # in validation order doesn't bother



from torch.utils.data import DataLoader,RandomSampler,SequentialSampler
# define optimizer

from transformers import AdamW

# here ADAMW is a optimizer with weight decaying

optimizer = AdamW(classifier.parameters(),lr=2e-5,eps=1e-8)
# set scheduler for learning rate, for that calculate total steps

from transformers import get_linear_schedule_with_warmup

epochs = 4 # researchers suggest to take epochs should be in range of 4-7 for fine tuning pretrained model as we have concern 

# just for last layer which is untrained classification layer.

total_steps = len(train_loader) * epochs

# scheduler take care of linear schedule of learning rate 

scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=0,num_training_steps=total_steps)

len(train_loader)
# Start training 

epochs = 2

import random





# total loss calculate for each epochs stores in this list

training_stats = []

for epoch in range(epochs):

    

    total_train_loss = 0

    classifier.train()

    

    for step,batch in enumerate(train_loader):

        

        # print out batch[0],batch[1] and label to get input_ids,attention_mask and labels

        if step%40==0 and not step==0: # this is just to show elapsed time with number of steps

            print('steps',step) # so here find steps for every 40 batches

        b_input_ids = batch[0].to(device)

        b_attention_mask = batch[1].to(device)

        b_labels = batch[2].to(device)

        # now before calculating loss, first zero all previously calculated gradients

        classifier.zero_grad()

        

        # now calculate loss and logits and add loss as a loss.item

        loss,logits = classifier(token_type_ids=None,input_ids=b_input_ids,attention_mask = b_attention_mask,labels = b_labels)

        total_train_loss+=loss.item()

        

        # now backward pass your loss and then update parameters using optimizer

        loss.backward()

        # to prevent gradient exploding use grad_norm

        torch.nn.utils.clip_grad_norm_(classifier.parameters(),1.0)

        # now to optimize loss for next batch of size 32(this size is recommended by researchers)

        optimizer.step() # here step is a function which updates model parameters which we have just passed in optimizer function

        

        scheduler.step()

    average_loss_batch = total_train_loss / len(train_loader) # here train_loader helps to arrange data randomly or sequentially depends

                                                              # on sampler and batch size.

    

    print('average_loss_batch:',average_loss_batch)

    

    # now evaluate model for validation dataset

    classifier.eval()

    # so side by side calculate validation loss, here don't need to forward pass for the inputs

    total_val_loss = 0 

    total_accuracy = 0

    for step,batch in enumerate(val_loader):

        # remember train_loader or val_loader is used to represent data in form of batches

        bt_input_ids = batch[0].to(device)

        bt_attention_mask = batch[1].to(device)

        bt_labels = batch[2].to(device)

        

        # tell torch not to bother forward pass instead of just backprop

        with torch.no_grad():

            (loss,logits) = classifier(token_type_ids=None,input_ids = bt_input_ids,attention_mask = bt_attention_mask,labels = bt_labels)

        total_val_loss += loss.item()

        # calculate accuracy as well for val_batch examples and true labels correspondingly.

        # detach true labels and logits and pass to cpu

        label = bt_labels.detach().cpu().numpy()

        predicted_labels = logits.detach().cpu().numpy()

        

        accuracy = f1_score(np.argmax(predicted_labels,axis=1).flatten(),label.flatten())

        total_accuracy+= accuracy

    # now find average loss per val_loader

    average_val_loss_batch = total_val_loss/len(val_loader)

    # now find average accuracy as well 

    average_val_accuracy = total_accuracy/len(val_loader)

    training_stats.append({'average_loss_batch':average_loss_batch,

                          'average_val_loss_batch':average_val_loss_batch,

                          'average_val_accuracy':average_val_accuracy})

    

    

print('Training Complete')       
#import torch, gc

#gc.collect()



#torch.cuda.empty_cache()
print(predicted_labels)
#! nvidia-smi clear
training_stats
# now finally we have trained model and now we just want to predict 
# now save weights of a trained model

torch.save(classifier,'fine-tuned-robert-SA.pth')
# now evaluate for testing dataset

# read testing dataset

test_user_review = pd.read_csv('../input/testingdataset/cleaned_test_fina.csv')['user_review'].values

classifier
type(test_user_review[0])
# now prepare testing dataset i.e. in form of tensors and then load using Data Loader

# First encode testing data as we did in case of training data

test_input_ids =[]

test_attention_mask = []



for sent in test_user_review:

    encode_dictionary = tokenizer.encode_plus(str(sent),max_length=300,pad_to_max_length = True,return_tensors='pt', return_token_type_ids=True,return_attention_mask=True,add_special_tokens=True)

    test_ids = encode_dictionary['input_ids']

    attention_mask = encode_dictionary['attention_mask']

    test_input_ids.append(test_ids)

    test_attention_mask.append(attention_mask)

    

    
# now integrate all testing dataset using TensorDataset

test_dataset = TensorDataset(torch.cat(test_input_ids),torch.cat(test_attention_mask)) # torch.cat function is used to concatenate all tensors 

                                                                                        # generated into one dim=0 like(4,5) -> (8,5) if there is two tensors
len(test_dataset)
# now load dataloader

test_loader = DataLoader(test_dataset,batch_size = 32,sampler= SequentialSampler(test_dataset))
# start evaluating model named classifier

classifier.eval()

epochs = 1

total_accuracy = 0

for epoch in range(epochs):

    

    final_predictions = []

    classifier.eval()

    for step,batch in enumerate(test_loader):

        if step%40==0:

            print(step)

        test_input_ids = batch[0].to(device)

        test_attention_mask = batch[1].to(device)

        with torch.no_grad():

            output = classifier(input_ids= test_input_ids ,token_type_ids=None, attention_mask = test_attention_mask)

        logits = output[0].detach().cpu().numpy()

        final_predictions.append(logits)

        
# then we will compare predictions with actual test labels using f1score or mcc score
final_predictions
predict = []

for predictions in final_predictions:

    for pred in predictions:

        predict.append(np.argmax(pred,axis=0).flatten())

len(predict)
predict = pd.DataFrame({'predictions':predict}).to_csv('predictions.csv')
torch.save(classifier.state_dict(),'weights-fine-tuned-robert.ckpt')
os.listdir('../output1')
classifier.save_pretrained('../output1')
model = torch.load('fine-tuned-robert-SA.pth')