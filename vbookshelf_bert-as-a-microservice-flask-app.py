

import pandas as pd
import numpy as np
import os

import torch

from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score

import transformers

import warnings
warnings.filterwarnings("ignore")
MODEL_TYPE = "bert-base-multilingual-uncased"
CASE_BOOL = True # do_lower_case=CASE_BOOL

MAX_LEN = 256
NUM_EPOCHS = 2
BATCH_SIZE = 16
LRATE = 2e-5

NUM_CORES = os.cpu_count()

NUM_CORES
os.listdir('../input/')
device = torch.device("cuda")
path = '../input/jigsaw-multilingual-toxic-comment-classification/' + 'jigsaw-toxic-comment-train.csv'
df_train_toxic = pd.read_csv(path, usecols=['comment_text', 'toxic'])

path = '../input/jigsaw-multilingual-toxic-comment-classification/' + 'jigsaw-unintended-bias-train.csv'
df_train_bias = pd.read_csv(path, usecols=['comment_text', 'toxic'])

path = '../input/jigsaw-multilingual-toxic-comment-classification/' + 'validation.csv'
df_val = pd.read_csv(path)

path = '../input/jigsaw-multilingual-toxic-comment-classification/' + 'test.csv'
df_test = pd.read_csv(path)


# Rename the 'content' column
df_test = df_test.rename(columns={'content': 'comment_text'})

print(df_train_toxic.shape)
print(df_train_bias.shape)
print(df_val.shape)
print(df_test.shape)
# Filter out only the toxic comments from df_train_bias
df_1 = df_train_bias[df_train_bias['toxic'] >= 0.5]

df_1.head()
# Combine df_1 and df_train_toxic
df_train = pd.concat([df_1, df_train_toxic], axis=0).reset_index(drop=True)

df_train.shape
# Take a sample of 10000 rows
df_train = df_train.sample(n=10000, random_state=1024)

# Reset the indices
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# Include the val set in df_train.
# We will be using the val set for training.

df_train = pd.concat([df_train, df_val], axis=0)

df_train = shuffle(df_train)

df_train = df_train.reset_index(drop=True)

df_train.shape

def preprocess_for_bert(sentences, MAX_LEN):
    
    """
    Preprocesses sentences to suit BERT.
    Input:
    sentences: numpy array
    
    Output:
    Tokenized sentences, padded and truncated.
    
    """

    
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []

    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            #max_length = 128,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)
        
    
    # Pad the token matrix
    
    # **** Issue: If the length is greater than max_len then 
    # this part will cut off the [SEP] token (102), which is
    # at the end of the long sentence.

    from keras.preprocessing.sequence import pad_sequences

    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    padded_input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                              value=0, truncating="post", padding="post")
    
    
    # *** This fixes the issue above.
    # Check if the SEP token was cut off and if so put it back in.
    # Check if the last index is 102. 102 is the SEP token.
    # Correct the last token if needed.
    for sent in padded_input_ids: # go row by row through the numpy 2D array.
        length = len(sent)
        
        if (sent[length-1] != 0) and (sent[length-1] != 102): # 102 is the SEP token
            sent[length-1] = 102 # set the last value to be the SEP token i.e. 102
    
    
    # Create attention masks
    attention_masks = []

    # For each sentence...
    for sent in padded_input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
        
        

    return padded_input_ids, attention_masks
from transformers import BertTokenizer

# Load the BERT tokenizer.
tokenizer = BertTokenizer.from_pretrained(MODEL_TYPE, do_lower_case=CASE_BOOL)
sentences = df_test['comment_text'].values
X_test, X_test_att_masks = preprocess_for_bert(sentences, MAX_LEN)


sentences = df_train['comment_text'].values
X_train, X_train_att_masks = preprocess_for_bert(sentences, MAX_LEN)
y_train = df_train['toxic'].values

sentences = df_val['comment_text'].values
X_val, X_val_att_masks = preprocess_for_bert(sentences, MAX_LEN)
y_val = df_val['toxic'].values



print(X_test.shape)
print(len(X_test_att_masks))

print('---')

print(X_train.shape)
print(len(X_train_att_masks))
print(y_train.shape)

print('---')

print(X_val.shape)
print(len(X_val_att_masks))
print(y_val.shape)



import torch

y_train = y_train.astype('long')
y_val = y_val.astype('long')

# Convert inputs and labels into torch tensors

train_inputs = torch.tensor(X_train)
validation_inputs = torch.tensor(X_val)

train_labels = torch.tensor(y_train)
validation_labels = torch.tensor(y_val)

train_masks = torch.tensor(X_train_att_masks)
validation_masks = torch.tensor(X_val_att_masks)
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler


# Create the DataLoader for our training set.
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = \
DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=NUM_CORES)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = \
DataLoader(validation_data, sampler=validation_sampler, batch_size=BATCH_SIZE, num_workers=NUM_CORES)
from transformers import BertForSequenceClassification, AdamW, BertConfig

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    MODEL_TYPE, 
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.to(device)
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = LRATE, 
                  eps = 1e-8 
                )
from transformers import get_linear_schedule_with_warmup

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * NUM_EPOCHS

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
import numpy as np

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

import random

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128


# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Store the average loss after each epoch so we can plot them.
loss_values = []

# For each epoch...
for epoch_i in range(0, NUM_EPOCHS):


    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, NUM_EPOCHS))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_loss = 0

    # Put the model into training mode. 
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 100 batches.
        if step % 100 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Clear any previously calculated gradients
        model.zero_grad()        

        outputs = model(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_labels)
        
        # The call to `model` always returns a tuple, so we need to pull the 
        # loss value out of the tuple.
        loss = outputs[0]

        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)            
    
    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(format_time(time.time() - t0)))
        
    
    
    # save the model
    torch.save(model.state_dict(), 'model.pt')
    
    

print("")
print("Training complete!")

# Convert all inputs and labels into torch tensors.
test_inputs = torch.tensor(X_test)
test_masks = torch.tensor(X_test_att_masks)

# Create the DataLoader for our validation set.
test_data = TensorDataset(test_inputs, test_masks)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=NUM_CORES)
# Evaluate data for one epoch
for j, batch in enumerate(test_dataloader):

    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)

    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask = batch

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up validation
    with torch.no_grad():        

    
        outputs = model(b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
        
        preds = outputs[0]
        
        if j == 0:
            stacked_preds = preds
        else:
            stacked_preds = torch.cat((stacked_preds, preds), dim=0)
            
            
# Apply the sigmoid function to the raw preds
stacked_preds = torch.sigmoid(stacked_preds)

orig_np_preds = stacked_preds.cpu().numpy()

# Select the second column which is class 1 i.e. toxic
preds = orig_np_preds[:, 1]

# Create a dataframe

df_results = pd.DataFrame({'id': df_test.id,
                             'toxic': preds
                         }).set_index('id')


# Create a submission csv file

df_results.to_csv('submission.csv',
                  columns=['toxic'])

df_results.head()
!ls

