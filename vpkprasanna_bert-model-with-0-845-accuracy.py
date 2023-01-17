# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import re

from tqdm import tqdm

import transformers

%matplotlib inline



from sklearn.model_selection import train_test_split

import nltk

from nltk import word_tokenize

from nltk.corpus import stopwords

words = stopwords.words("english")

lemma = nltk.stem.WordNetLemmatizer()





import torch

from transformers import BertTokenizer

from transformers import RobertaTokenizer

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn

import torch.nn.functional as F

from transformers import BertModel,RobertaModel

from transformers import AdamW, get_linear_schedule_with_warmup



import random

import time

train = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')

test = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')

sub = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv')
train.columns
targets = ['ComputerScience', 'Physics', 'Mathematics','Statistics', 'QuantitativeBiology', 'QuantitativeFinance']
train["text"] = train["TITLE"]+""+train["ABSTRACT"]

test["text"] = test["TITLE"]+""+test["ABSTRACT"]
X = train.text.values

y = train[['Computer Science', 'Physics', 'Mathematics',

       'Statistics', 'Quantitative Biology', 'Quantitative Finance']].values



X_train, X_val, y_train, y_val =train_test_split(X, y, test_size=0.1, random_state=2020)
if torch.cuda.is_available():       

    device = torch.device("cuda")

    print(f'There are {torch.cuda.device_count()} GPU(s) available.')

    print('Device name:', torch.cuda.get_device_name(0))



else:

    print('No GPU available, using the CPU instead.')

    device = torch.device("cpu")

# def text_preprocessing(text):

#     """

#     - Remove entity mentions (eg. '@united')

#     - Correct errors (eg. '&amp;' to '&')

#     @param    text (str): a string to be processed.

#     @return   text (Str): the processed string.

#     """

#     text = text.lower()

#     text = re.sub(r"what's", "what is ", text)

#     text = re.sub(r"won't", "will not ", text)

#     text = re.sub(r"\'s", " ", text)

#     text = re.sub(r"\'ve", " have ", text)

#     text = re.sub(r"can't", "can not ", text)

#     text = re.sub(r"n't", " not ", text)

#     text = re.sub(r"i'm", "i am ", text)

#     text = re.sub(r"\'re", " are ", text)

#     text = re.sub(r"\'d", " would ", text)

#     text = re.sub(r"\'ll", " will ", text)

#     text = re.sub(r"\'scuse", " excuse ", text)

#     text = re.sub(r"\'\n", " ", text)

#     text = re.sub(r"-", " ", text)

#     text = re.sub(r"\'\xa0", " ", text)

#     text = re.sub('\s+', ' ', text)

#     text = ''.join(c for c in text if not c.isnumeric())

    

#     # Remove '@name'

#     text = re.sub(r'(@.*?)[\s]', ' ', text)

    

      # Remove Latex tags in the test

#     text = re.sub(r'(\$+)(?:(?!\1)[\s\S])*\1',' ',text)

    

      # Remove words with in brackets

#     text = re.sub(r'\([^)]*\)', '', text)



#     # Replace '&amp;' with '&'

#     text = re.sub(r'&amp;', '&', text)



#     # Remove trailing whitespace

#     text = re.sub(r'\s+', ' ', text).strip()

     

# #     tokens = word_tokenize(text)

# #     remove_words = [word for word in tokens if not word in words]

# #     text = [lemma.lemmatize(word) for word in remove_words]

# #     joined_words = " ".join(text)



#     return text
def text_preprocessing(text):

    """

    - Remove entity mentions (eg. '@united')

    - Correct errors (eg. '&amp;' to '&')

    @param    text (str): a string to be processed.

    @return   text (Str): the processed string.

    """

    

    text = text.lower()

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"won't", "will not ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "can not ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    text = re.sub(r"\'\n", " ", text)

    text = re.sub(r"-", " ", text)

    text = re.sub(r"\'\xa0", " ", text)

    text = re.sub('\s+', ' ', text)

    text = ''.join(c for c in text if not c.isnumeric())

    

    # Remove '@name'

    text = re.sub(r'(@.*?)[\s]', ' ', text)



    # Replace '&amp;' with '&'

    text = re.sub(r'&amp;', '&', text)



    # Remove trailing whitespace

    text = re.sub(r'\s+', ' ', text).strip()



    return text
temp = train["text"][1000]

print(temp)

val = text_preprocessing(text=temp)

print(val)
#Load the Bert tokenizer

# tokenizer = RobertaTokenizer.from_pretrained('roberta-base',do_lower_case=True)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",do_lower_case=True)

# Create a funcition to tokenize a set of text



def preprocessing_for_bert(data):

    """Perform required preprocessing steps for pretrained BERT.

    @param    data (np.array): Array of texts to be processed.

    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.

    @return   attention_masks (torch.Tensor): Tensor of indices specifying which

                  tokens should be attended to by the model.

    """

    # create empty lists to store outputs

    input_ids = []

    attention_masks = []

    

    #for every sentence...

    

    for sent in data:

        # 'encode_plus will':

        # (1) Tokenize the sentence

        # (2) Add the `[CLS]` and `[SEP]` token to the start and end

        # (3) Truncate/Pad sentence to max length

        # (4) Map tokens to their IDs

        # (5) Create attention mask

        # (6) Return a dictionary of outputs

        encoded_sent = tokenizer.encode_plus(

            text = text_preprocessing(sent),   #preprocess sentence

            add_special_tokens = True,         #Add `[CLS]` and `[SEP]`

            max_length= MAX_LEN  ,             #Max length to truncate/pad

            pad_to_max_length = True,          #pad sentence to max length 

            return_attention_mask= True        #Return attention mask 

        )

        # Add the outputs to the lists

        input_ids.append(encoded_sent.get('input_ids'))

        attention_masks.append(encoded_sent.get('attention_mask'))

        

    #convert lists to tensors

    input_ids = torch.tensor(input_ids)

    attention_masks = torch.tensor(attention_masks)

    

    return input_ids,attention_masks
# Before tokenizing we need to specify the maximum length of our sentences



#concat the train data and test data



all_text = np.concatenate([train.text.values,test.text.values])



#Encode the concatenated data

len_sent = [len(text_preprocessing(sent)) for sent in all_text]



# Find the maximum length

avg_len = np.mean(len_sent)

print('Avg length: ',avg_len)
## Now Tokenizing the data



MAX_LEN = 500



# Print sentece 0 and its encoded token ids

token_ids = list(preprocessing_for_bert([X[0]])[0].squeeze().numpy())

print('Original: ',X[0])

print('Token IDs: ',token_ids)
# Run function 'preprocessing_for_bert' on the train set and validation set

print('Tokenizing data...')

train_inputs, train_masks = preprocessing_for_bert(X_train)

val_inputs, val_masks = preprocessing_for_bert(X_val)
# Convert other data types to torch.Tensor

train_labels = torch.tensor(y_train)

val_labels = torch.tensor(y_val)



## For fine-tuning Bert, the authors recommmend a batch size of 16 or 32

batch_size = 16



# Create the DataLoader for our training set

train_data = TensorDataset(train_inputs,train_masks, train_labels)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)



# Create the DataLoader for our validation set

val_data = TensorDataset(val_inputs, val_masks, val_labels)

val_sampler = SequentialSampler(val_data)

val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
%%time

# Create the BertClassifier class



class BertClassifier(nn.Module):

    """

        Bert Model for classification Tasks.

    """

    def __init__(self, freeze_bert=False):

        """

        @param   bert: a BertModel object

        @param   classifier: a torch.nn.Module classifier

        @param   freeze_bert (bool): Set `False` to fine_tune the Bert model

        """

        super(BertClassifier,self).__init__()

        # Specify hidden size of Bert, hidden size of our classifier, and number of labels

        D_in, H,D_out = 768,30,6

        

#         self.bert = RobertaModel.from_pretrained('roberta-base')

        self.bert = BertModel.from_pretrained("bert-base-uncased")

        

        self.classifier = nn.Sequential(

                            nn.Linear(D_in, H),

                            nn.ReLU(),

                            nn.Linear(H, D_out))

        self.sigmoid = nn.Sigmoid()

        # Freeze the Bert Model

        if freeze_bert:

            for param in self.bert.parameters():

                param.requires_grad = False

    

    def forward(self,input_ids,attention_mask):

        """

        Feed input to BERT and the classifier to compute logits.

        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,

                      max_length)

        @param    attention_mask (torch.Tensor): a tensor that hold attention mask

                      information with shape (batch_size, max_length)

        @return   logits (torch.Tensor): an output tensor with shape (batch_size,

                      num_labels)

        """

        outputs = self.bert(input_ids=input_ids,

                           attention_mask = attention_mask)

        

        # Extract the last hidden state of the token `[CLS]` for classification task

        last_hidden_state_cls = outputs[0][:,0,:]

        

        # Feed input to classifier to compute logits

        logit = self.classifier(last_hidden_state_cls)

        

#         logits = self.sigmoid(logit)

        

        return logit
def initialize_model(epochs=4):

    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.

    """

    

    # Instantiate Bert Classifier

    bert_classifier = BertClassifier(freeze_bert=False)

    

    bert_classifier.to(device)

    

    # Create the optimizer

    optimizer = AdamW(bert_classifier.parameters(),

                     lr=5e-5, #Default learning rate

                     eps=1e-8 #Default epsilon value

                     )

    

    # Total number of training steps

    total_steps = len(train_dataloader) * epochs

    

    # Set up the learning rate scheduler

    scheduler = get_linear_schedule_with_warmup(optimizer, 

                                              num_warmup_steps=0, # Default value

                                              num_training_steps=total_steps)

    return bert_classifier, optimizer, scheduler
# Specify loss function

#loss_fn = nn.CrossEntropyLoss()

loss_fn = nn.BCEWithLogitsLoss()



def set_seed(seed_value=42):

    """Set seed for reproducibility.

    """

    random.seed(seed_value)

    np.random.seed(seed_value)

    torch.manual_seed(seed_value)

    torch.cuda.manual_seed_all(seed_value)



def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):

    """Train the BertClassifier model.

    """

    # Start training loop

    print("Start training...\n")

    for epoch_i in range(epochs):

        # =======================================

        #               Training

        # =======================================

        # Print the header of the result table

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")

        print("-"*70)



        # Measure the elapsed time of each epoch

        t0_epoch, t0_batch = time.time(), time.time()



        # Reset tracking variables at the beginning of each epoch

        total_loss, batch_loss, batch_counts = 0, 0, 0



        # Put the model into the training mode

        model.train()



        # For each batch of training data...

        for step, batch in enumerate(train_dataloader):

            batch_counts +=1

            # Load batch to GPU

            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)



            # Zero out any previously calculated gradients

            model.zero_grad()



            # Perform a forward pass. This will return logits.

            logits = model(b_input_ids, b_attn_mask)



            # Compute loss and accumulate the loss values

            loss = loss_fn(logits, b_labels.float())

            batch_loss += loss.item()

            total_loss += loss.item()



            # Perform a backward pass to calculate gradients

            loss.backward()



            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)



            # Update parameters and the learning rate

            optimizer.step()

            scheduler.step()



            # Print the loss values and time elapsed for every 20--50000 batches

            if (step % 50000 == 0 and step != 0) or (step == len(train_dataloader) - 1):

                # Calculate time elapsed for 20 batches

                time_elapsed = time.time() - t0_batch



                # Print training results

                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")



                # Reset batch tracking variables

                batch_loss, batch_counts = 0, 0

                t0_batch = time.time()



        # Calculate the average loss over the entire training data

        avg_train_loss = total_loss / len(train_dataloader)



        print("-"*70)

        # =======================================

        #               Evaluation

        # =======================================

        if evaluation == True:

            # After the completion of each training epoch, measure the model's performance

            # on our validation set.

            val_loss, val_accuracy = evaluate(model, val_dataloader)



            # Print performance over the entire training data

            time_elapsed = time.time() - t0_epoch

            

            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")

            print("-"*70)

        print("\n")

    

    print("Training complete!")





def evaluate(model, val_dataloader):

    """After the completion of each training epoch, measure the model's performance

    on our validation set.

    """

    # Put the model into the evaluation mode. The dropout layers are disabled during

    # the test time.

    model.eval()



    # Tracking variables

    val_accuracy = []

    val_loss = []



    # For each batch in our validation set...

    for batch in val_dataloader:

        # Load batch to GPU

        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)



        # Compute logits

        with torch.no_grad():

            logits = model(b_input_ids, b_attn_mask)



        # Compute loss

        loss = loss_fn(logits, b_labels.float())

        val_loss.append(loss.item())



        # Get the predictions

        #preds = torch.argmax(logits, dim=1).flatten()

        

        # Calculate the accuracy rate

        #accuracy = (preds == b_labels).cpu().numpy().mean() * 100

        accuracy = accuracy_thresh(logits.view(-1,6),b_labels.view(-1,6))

        

        val_accuracy.append(accuracy)



    # Compute the average accuracy and loss over the validation set.

    val_loss = np.mean(val_loss)

    val_accuracy = np.mean(val_accuracy)



    return val_loss, val_accuracy



def accuracy_thresh(y_pred, y_true, thresh:float=0.5, sigmoid:bool=True):

    "Compute accuracy when `y_pred` and `y_true` are the same size."

    if sigmoid: 

        y_pred = y_pred.sigmoid()

    return ((y_pred>thresh)==y_true.byte()).float().mean().item()

    #return np.mean(((y_pred>thresh).float()==y_true.float()).float().cpu().numpy(), axis=1).sum()
set_seed(42)    # Set seed for reproducibility

bert_classifier, optimizer, scheduler = initialize_model(epochs=1)

train(bert_classifier, train_dataloader, val_dataloader, epochs=1, evaluation=True)


def bert_predict(model, test_dataloader):

    """Perform a forward pass on the trained BERT model to predict probabilities

    on the test set.

    """

    # Put the model into the evaluation mode. The dropout layers are disabled during

    # the test time.

    model.eval()



    all_logits = []



    # For each batch in our test set...

    for batch in test_dataloader:

        # Load batch to GPU

        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]



        # Compute logits

        with torch.no_grad():

            logits = model(b_input_ids, b_attn_mask)

        all_logits.append(logits)

    

    # Concatenate logits from each batch

    all_logits = torch.cat(all_logits, dim=0)



    # Apply softmax to calculate probabilities

    #probs = F.softmax(all_logits, dim=1).cpu().numpy()

    probs = all_logits.sigmoid().cpu().numpy()

    



    return probs



#probs = all_logits.sigmoid().cpu().numpy()

## Compute predicted probabilities on the test set



probs = bert_predict(bert_classifier,val_dataloader)



# Evalueate the bert classifier



# evaluate_roc(probs, y_val)
# Concatenate the train set and the validation set



full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])

full_train_sampler = RandomSampler(full_train_data)

full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=batch_size)



# Train the Bert Classifier on the entire training data

set_seed(42)

bert_classifier, optimizer, scheduler = initialize_model(epochs=4)

train(bert_classifier, full_train_dataloader, epochs=4)
test["text"] = test["text"].apply(text_preprocessing)



## Run preprocessing_for_bert on the test set

print('Tokenizing data...')

test_inputs, test_masks = preprocessing_for_bert(test.text)



# Create the DataLoader for our test set

test_dataset = TensorDataset(test_inputs, test_masks)

#test_sampler = SequentialSampler(test_dataset)

#test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=32)

test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=16)



# Compute predicted probabilities on the test set

probs = bert_predict(bert_classifier, test_dataloader)



# Get predictions from the probabilities

#threshold = 0.5. ## Change depending on the accuracy you need

#preds = np.where(probs[:, 1] > threshold, 1, 0)
submission = pd.DataFrame(probs,columns=['Computer Science', 'Physics', 'Mathematics',

       'Statistics', 'Quantitative Biology', 'Quantitative Finance'])

test[['Computer Science', 'Physics', 'Mathematics',

       'Statistics', 'Quantitative Biology', 'Quantitative Finance']]=submission

final_sub = test[['Computer Science', 'Physics', 'Mathematics',

       'Statistics', 'Quantitative Biology', 'Quantitative Finance']]

final_sub.head()

final_sub.to_csv("BertModel_proba.csv",index=False)
def check_thresold(thresold):

    sub["Computer Science"] = [1 if n >= thresold else 0 for n in final_sub["Computer Science"]]

    sub["Physics"] = [1 if n >= thresold else 0 for n in final_sub["Physics"]]

    sub["Mathematics"] = [1 if n >= thresold else 0 for n in final_sub["Mathematics"]]

    sub["Statistics"] = [1 if n >= thresold else 0 for n in final_sub["Statistics"]]

    sub["Quantitative Biology"] = [1 if n >= thresold else 0 for n in final_sub["Quantitative Biology"]]

    sub["Quantitative Finance"] = [1 if n >= thresold else 0 for n in final_sub["Quantitative Finance"]]

    

    sub.to_csv("_bert_{}_hdn30_padd462_split8_text_cleaned1.csv".format(thresold),index=False)
check_thresold(0.36)