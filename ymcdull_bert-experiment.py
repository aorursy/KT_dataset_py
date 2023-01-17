# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.tsv", sep='\t')

test = pd.read_csv("../input/test.tsv", sep='\t')

#train = train[0:1000]

train['Sentiment'] = train['Sentiment'].apply(str)

train.head()

test.head()
train.Sentiment.value_counts()
train.shape
import tensorflow as tf



device_name = tf.test.gpu_device_name()

if device_name != '/device:GPU:0':

  raise SystemError('GPU device not found')

print('Found GPU at: {}'.format(device_name))
# test = pd.read_csv("../input/test.tsv", sep='\t')

# test_id = test['PhraseId']
test['Phrase'][0]
train['Sentiment'].unique()
!pip install pytorch-pretrained-bert pytorch-nlp
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split

from pytorch_pretrained_bert import BertTokenizer, BertConfig

from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

from tqdm import tqdm, trange

import pandas as pd

import io

import numpy as np

import matplotlib.pyplot as plt

% matplotlib inline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_gpu = torch.cuda.device_count()

torch.cuda.get_device_name(0)
train['Sentiment'] = train.Sentiment.astype(int)

test['Sentiment'] = test.Sentiment.astype(int)
# Create sentence and label lists

sentences = train.Phrase.values



# We need to add special tokens at the beginning and end of each sentence for BERT to work properly

sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

labels = train.Sentiment.values
np.unique(labels)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)



tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]

print ("Tokenize the first sentence:")

print (tokenized_texts[0])
import seaborn as sns
sns.distplot(train.Phrase.apply(lambda x: len(x.split())))
# Set the maximum sequence length. The longest sequence in our training set is 47, but we'll leave room on the end anyway. 

# In the original paper, the authors used a length of 512.

MAX_LEN = 32
# Pad our input tokens

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],

                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
# Create attention masks

attention_masks = []



# Create a mask of 1s for each token followed by 0s for padding

for seq in input_ids:

  seq_mask = [float(i>0) for i in seq]

  attention_masks.append(seq_mask)
# Use train_test_split to split our data into train and validation sets for training



train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 

                                                            random_state=2018, test_size=0.1)

train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,

                                             random_state=2018, test_size=0.1)
train_labels
# Convert all of our data into torch tensors, the required datatype for our model



train_inputs = torch.tensor(train_inputs)

validation_inputs = torch.tensor(validation_inputs)

train_labels = torch.tensor(train_labels)

validation_labels = torch.tensor(validation_labels)

train_masks = torch.tensor(train_masks)

validation_masks = torch.tensor(validation_masks)
# Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of 16 or 32

batch_size = 32



# Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a for loop, 

# with an iterator the entire dataset does not need to be loaded into memory



train_data = TensorDataset(train_inputs, train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)

validation_sampler = SequentialSampler(validation_data)

validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

# Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top. 



model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

model.cuda()
param_optimizer = list(model.named_parameters())

no_decay = ['bias', 'gamma', 'beta']

optimizer_grouped_parameters = [

    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.01},

    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

     'weight_decay_rate': 0.0}

]

# This variable contains all of the hyperparemeter information our training loop needs

optimizer = BertAdam(optimizer_grouped_parameters,

                     lr=2e-5,

                     warmup=.1)
# Function to calculate the accuracy of our predictions vs labels

def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=1).flatten()

    labels_flat = labels.flatten()

    return np.sum(pred_flat == labels_flat) / len(labels_flat)
device
train_loss_set = []



# Number of training epochs (authors recommend between 2 and 4)

epochs = 4



# trange is a tqdm wrapper around the normal python range

for _ in trange(epochs, desc="Epoch"):

  

  

  # Training

  

  # Set our model to training mode (as opposed to evaluation mode)

  model.train()

  

  # Tracking variables

  tr_loss = 0

  nb_tr_examples, nb_tr_steps = 0, 0

  

  # Train the data for one epoch

  for step, batch in enumerate(train_dataloader):

    # Add batch to GPU

    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader

    b_input_ids, b_input_mask, b_labels = batch

    # Clear out the gradients (by default they accumulate)

    optimizer.zero_grad()

    # Forward pass

    loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)

    train_loss_set.append(loss.item())    

    # Backward pass

    loss.backward()

    # Update parameters and take a step using the computed gradient

    optimizer.step()

    

    

    # Update tracking variables

    tr_loss += loss.item()

    nb_tr_examples += b_input_ids.size(0)

    nb_tr_steps += 1



  print("Train loss: {}".format(tr_loss/nb_tr_steps))

    

    

  # Validation



  # Put model in evaluation mode to evaluate loss on the validation set

  model.eval()



  # Tracking variables 

  eval_loss, eval_accuracy = 0, 0

  nb_eval_steps, nb_eval_examples = 0, 0



  # Evaluate data for one epoch

  for batch in validation_dataloader:

    # Add batch to GPU

    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader

    b_input_ids, b_input_mask, b_labels = batch

    # Telling the model not to compute or store gradients, saving memory and speeding up validation

    with torch.no_grad():

      # Forward pass, calculate logit predictions

      logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

    

    # Move logits and labels to CPU

    logits = logits.detach().cpu().numpy()

    label_ids = b_labels.to('cpu').numpy()



    tmp_eval_accuracy = flat_accuracy(logits, label_ids)

    

    eval_accuracy += tmp_eval_accuracy

    nb_eval_steps += 1



  print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
plt.figure(figsize=(15,8))

plt.title("Training loss")

plt.xlabel("Batch")

plt.ylabel("Loss")

plt.plot(train_loss_set)

plt.show()
test = pd.read_csv("../input/test.tsv", sep='\t')

test_id = test['PhraseId']
# Create sentence and label lists

sentences = test.Phrase.values



# We need to add special tokens at the beginning and end of each sentence for BERT to work properly

sentences = ["[CLS] " + sentence + " [SEP]" for sentence in sentences]

# labels = test.Sentiment.values



tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
MAX_LEN = 32

# Pad our input tokens

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],

                          maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary

input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

# Create attention masks

attention_masks = []



# Create a mask of 1s for each token followed by 0s for padding

for seq in input_ids:

  seq_mask = [float(i>0) for i in seq]

  attention_masks.append(seq_mask) 
test['Sentiment'] = 0

labels = test.Sentiment.values
prediction_inputs = torch.tensor(input_ids)

prediction_masks = torch.tensor(attention_masks)

prediction_labels = torch.tensor(labels)

  

batch_size = 32  
prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)

prediction_sampler = SequentialSampler(prediction_data)

prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
# Prediction on test set



# Put model in evaluation mode

model.eval()



# Tracking variables 

predictions , true_labels = [], []



# Predict 

for batch in prediction_dataloader:

  # Add batch to GPU

  batch = tuple(t.to(device) for t in batch)

  # Unpack the inputs from our dataloader

  b_input_ids, b_input_mask, b_labels = batch

  # Telling the model not to compute or store gradients, saving memory and speeding up prediction

  with torch.no_grad():

    # Forward pass, calculate logit predictions

    logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)



  # Move logits and labels to CPU

  logits = logits.detach().cpu().numpy()

  label_ids = b_labels.to('cpu').numpy()

  

  # Store predictions and true labels

  predictions.append(logits)

#   true_labels.append(label_ids)
preds = np.concatenate(predictions)
# preds, target = learn_classifier.get_preds(DatasetType.Test, ordered=True)

labels = np.argmax(preds, axis =1)

submission = pd.DataFrame({'PhraseId': test_id, 'Sentiment': labels})

submission.to_csv('submission.csv', index=False)

submission.head()