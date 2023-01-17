!pip install spacy textvec
!python -m spacy download en_core_web_sm
# Data
import pandas as pd
import numpy as np

# Preprocessing
import nltk
import re
import spacy

# ML
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from textvec.vectorizers import TforVectorizer

# NNs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences

#Utils
from tqdm import tqdm
# BERT Model
!pip install transformers
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, AdamW, BertConfig
# Define datafolder
# If you are using google colab you can put data in /content/drive/My Drive/Colab/Real-or-Not/data/
try:
    from google.colab import drive
    is_in_colab = True
    
except:
    is_in_colab = False

if is_in_colab:
    drive.mount('/content/drive')
    data_folder = r'/content/drive/My Drive/Colab/Real-or-Not/data/'
else:
    data_folder = r'../input/nlp-getting-started'
data = pd.read_csv(data_folder + '/train.csv')
data.head(30)
# Look at the class ratio
data.target.hist()
data.target.describe()
def clean(text):
  text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',    # Substitute different urls with "url" token.
                  'url', text)
  text = re.sub('#', '', text)              # Delete hashtag signs.
  text = re.sub('\d+[.,]?\d*', 'num', text)  # Substitute different numbers with "num" token.                                                 
  text = re.sub('@\w+_?\w*', 'username', text)  # Substitute different usernames with 'username' token.                                                       
  return text

def lemmatize(text):
  preprocessed_tokens = []    # preprocessed_tokens will be lemmatized, stopwords removed and lowercased
  nlp = spacy.load("en_core_web_sm", disable=['ner', 'parser'])   # We will not use NER and syntactic parser.
  doc = nlp(text)
  for token in doc:
    if not token.is_stop:
      preprocessed_tokens.append(token.lemma_.lower())
  return ' '.join(preprocessed_tokens)
data['text'] = data.text.apply(clean)    # Clean texts.
data['lemmatized_text'] = data.text.apply(lemmatize)  # Add new column with lemmatized and preprocessed texts. This task requires a lot of time.
data.drop(['keyword', 'location'], axis=1, inplace=True) # We don't need 'keyword', 'location' columns.
# data.to_pickle(data_folder+'/cleaned_lemmatized_train.pkl')
# data = pd.read_pickle('../input/realornottrainsplitted/cleaned_lemmatized_train.pkl')
# Split data on train and test.
X_train, X_val, y_train, y_val = train_test_split(data, data.target, train_size = 0.8, random_state=42)   
# We will use TF-IDF and TFOR as features describing texts.
# We will try word level features using lemmas and char level features. 

word_vectorizer = TfidfVectorizer(    # TF-IDF for word level features.
    analyzer='word',                  # Word level tokenization.
    ngram_range=(1, 1),               # We don't use ngrams. But you can try.
    max_features=10000,      # Number of max features (it will be shape of our training matrix: shape=(num_documents,max_features))
    max_df=0.7,                        # We will not use tokens that are found in more than 70 percent of documents
    min_df=1)                          # We will not use tokens that are found only in 1 document

char_vectorizer = TfidfVectorizer(    # TF-IDF for char level features.
    analyzer='char',                  # Char level tokenization.                            
    ngram_range=(1, 4),               # We use [1..4]grams. 
    max_features=30000,                                   
    max_df=0.7,                                            
    min_df=1
    )

# Get char ngrams features for the train set and the validation set.
word_vectorizer.fit(data.lemmatized_text)

train_w_features = word_vectorizer.fit_transform(X_train.lemmatized_text)
val_w_features = word_vectorizer.transform(X_val.lemmatized_text)

# Get lemma features for the train set and the validation set.
char_vectorizer.fit(data.text)

train_c_features = char_vectorizer.transform(X_train.text)
val_c_features = char_vectorizer.transform(X_val.text)

# We can use information about classes in features with help of supervised vectorizer. 
# For example, TFOR (for binary classification) from textvec lib. Let's fit TFOR on char ngrams.
tfor = TforVectorizer()

tfor_train = tfor.fit_transform(char_vectorizer.transform(X_train.text), y_train)
tfor_val = tfor.transform(char_vectorizer.transform(X_val.text))

# Logreg on tf-idf char ngrams from cleaned texts

# You should try different values of C [0.01, 0.1, 1, 10], solver ['liblinear', 'sag'], penalty ['l1', 'l2']

logreg = LogisticRegression(C=1, solver='sag')             
logreg.fit(train_c_features, y_train)
preds = logreg.predict(val_c_features)
print(classification_report(y_val, preds))
# Logreg on tf-idf lowercased lemmas from cleaned, lemmatized, stop words filtered texts.

# You should try different values of C [0.01, 0.1, 1, 10], solver ['liblinear', 'sag'], penalty ['l1', 'l2']

logreg = LogisticRegression(C=1, solver='sag')               
logreg.fit(train_w_features, y_train)
preds = logreg.predict(val_w_features)
print(classification_report(y_val, preds))
# Logreg on tfor char ngrams from cleaned texts

# You should try different values of C [0.01, 0.1, 1, 10], solver ['liblinear', 'sag'], penalty ['l1', 'l2']

logreg = LogisticRegression(C=1, solver='sag')            
logreg.fit(tfor_train, y_train)
preds = logreg.predict(tfor_val)
print(classification_report(y_val, preds))
# Resolve what Tensors we will use - GPU or CPU. 
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device('cuda:0')
    from torch.cuda import FloatTensor, LongTensor            # Import GPU Tensors.
else:
    device = torch.device('cpu')
    from torch import FloatTensor, LongTensor                 # Import CPU Tensors.
is_cuda
# This function for training NN models.

def fit(model, loss_function, train_data=None, val_data=None, optimizer=None,
        epoch_count=1, batch_size=1, scheduler=None, alpha=1, type_nn=None):
  
    train_history = []
    val_history = []

    for epoch in range(epoch_count):
            name_prefix = '[{} / {}] '.format(epoch + 1, epoch_count)
            epoch_train_score = 0
            epoch_val_score = 0
            
            if train_data:
                epoch_train_score = do_epoch(model, loss_function, train_data, batch_size, 
                                              optimizer, name_prefix + 'Train:', alpha=alpha, type_nn=type_nn,
                                             scheduler=scheduler)
                train_history.append(epoch_train_score)

            if val_data:
                name = '  Val:'
                if not train_data:
                    name = ' Test:'
                epoch_val_score = do_epoch(model, loss_function, val_data, batch_size, 
                                             optimizer=None, name=name_prefix + name, alpha=alpha, type_nn=type_nn,
                                           scheduler=scheduler)
                
                val_history.append(epoch_val_score)

    return train_history, val_history
    
# This is a function for generating one epoch for each NN model (BERT, FNN, CNN).

def do_epoch(model, loss_function, data, batch_size, optimizer=None, name=None, alpha=1, type_nn=None, scheduler=None):
    """
       One epoch generation
    """
    accuracy = 0
    epoch_loss = 0
   
    batch_count = len(data)
   
    is_train = not optimizer is None
    name = name or ''
    model.train(is_train)
    
    with torch.autograd.set_grad_enabled(is_train):
        with tqdm(total=batch_count) as progress_bar:               
            for ind, batch in enumerate(data):
                if type_nn == 'BERT':
                  X_batch, X_mask, y_batch =  batch[0].to(device), batch[1].to(device), batch[2].to(device)
                  loss, prediction = model(X_batch, token_type_ids=None, attention_mask=X_mask, labels=y_batch)
                if type_nn == 'CNN':
                  X_batch, y_batch = batch
                  prediction = model(X_batch)
                  loss = loss_function(prediction, y_batch)
                if type_nn == 'FNN':
                  X_batch, y_batch = batch[0].to(device), batch[1].to(device)
                  prediction = model(X_batch)
                  loss = loss_function(prediction, y_batch)

                  for param in model.children():
                    if type(param) == nn.Linear:
                        loss += alpha * torch.abs(param.weight).sum()

                epoch_loss += loss.item()

                true_indices = torch.argmax(prediction, dim=1)
                correct_samples = torch.sum(true_indices == y_batch).cpu().numpy()
                accuracy += correct_samples / y_batch.shape[0]

                if is_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler: scheduler.step(accuracy)
              
                progress_bar.update()
                progress_bar.set_description('Epoch {} - accuracy: {:.2f}, loss {:.2f}'.format(
                    name, (accuracy / (ind+1)), epoch_loss / (ind+1))
                )
            
            accuracy /= (ind + 1)
            epoch_loss /= (ind + 1) 
            progress_bar.set_description(f'Epoch {name} - accuracy: {accuracy:.2f}, loss: {epoch_loss:.2f}')

    return accuracy
# Preparing data
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
def prepare_data_for_bert(texts):
  MAX_LEN = 0
  input_ids = []
  attention_masks = []
  for tweet in texts:
    # `encode` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    encoded_tweet = bert_tokenizer.encode(
                        tweet,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
 
                        # This function also supports truncation and conversion
                        # to pytorch tensors, but we need to do padding, so we
                        # can't use these features :( .
                        #max_length = 128,          # Truncate all sentences.
                        #return_tensors = 'pt',     # Return pytorch tensors.
                   )
    # Add the encoded sentence to the list.
    input_ids.append(encoded_tweet)
 
    if len(encoded_tweet) > MAX_LEN:
      MAX_LEN = len(encoded_tweet)
 
  input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", 
                        value=0, truncating="post", padding="post")
  
  # Make attention masks token -> 1, [PAD] -> 0
  for tweet in input_ids:
    att_mask = [int(token_id > 0) for token_id in tweet]
    attention_masks.append(att_mask)
    
  return input_ids, attention_masks
# Prepare data for BERT
input_ids, attention_masks = prepare_data_for_bert(data.text)
labels = data.target.values

train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels, 
                                                            random_state=2018, test_size=0.25)
train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                             random_state=2018, test_size=0.25)


batch_size = 16

# Create the DataLoader for our training set.
train_data = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_masks), torch.tensor(train_labels))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.
validation_data = TensorDataset(torch.tensor(validation_inputs), torch.tensor(validation_masks), torch.tensor(validation_labels))
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
# Load model
bert = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)

bert.cuda()   #To GPU
import random

seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


# Number of training epochs (between 2 and 4 recommended)
epochs = 3

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# AdamW is a class from the huggingface library. Read docs for get an idea of parameters.
optimizer = AdamW(bert.parameters(),
                  lr = 5e-5, # args.learning_rate - default is 5e-5, you can try [1e-5 .. 5e-5]
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

train_history_bert, val_history_bert = fit(bert, loss_function=None, train_data=train_dataloader, val_data=validation_dataloader, optimizer=optimizer, epoch_count=epochs, batch_size=batch_size, type_nn='BERT', scheduler=scheduler, alpha=1, )
# Prepare data for NN

# Create the DataLoader for our training set. We will use TF-IDF matrix
train_data = TensorDataset(torch.FloatTensor(train_c_features.toarray()), torch.tensor(np.array(y_train)))
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Create the DataLoader for our validation set.  We will use TF-IDF matrix
validation_data = TensorDataset(torch.FloatTensor(val_c_features.toarray()), torch.tensor(np.array(y_val)))
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)
# fit settings
batch_size = 100
epoch_count = 4

# optim settings. You should try different values.
learning_rate = 1e-4
weight_decay = 0.1
alpha = 0.005

# model settings. 
linear1_out = int(val_c_features.shape[1]**0.5)            # You should try different values.
output = 2                                                 # Equals to num classes.
dropout = 0.3                                              # You should try different values.


model = nn.Sequential(nn.Linear(train_c_features.shape[1], linear1_out),
                      nn.BatchNorm1d(linear1_out),
#                       nn.Dropout(p=dropout, inplace=True),
                      nn.ReLU(inplace=True),
                      nn.Linear(linear1_out, output),
                      nn.ReLU(inplace=True)
                     ).to(device)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
                        model.parameters(),
                        lr=learning_rate, 
                        weight_decay=weight_decay
                    )

#
train_history, val_history = fit(model, loss_function, train_dataloader, validation_dataloader, optimizer, epoch_count, batch_size, scheduler=None, alpha=alpha, type_nn='FNN')
# Prepare data for torch dataset in Google colab or local
#train, val = train_test_split(data[['text', 'target']], train_size = 0.8, random_state=42)
#train.to_csv(data_folder + 'train_cnn.csv')
#val.to_csv(data_folder + 'val_cnn.csv')

#In Kaggle Kernel I uploaded splitted train data
cnn_data = '../input/realornottrainsplitted/'
# Prepare torch dataset

import torchtext

MAX_TEXT_LEN = max(data.text.apply(lambda x: len(x)))

train, val = train_test_split(data[['text', 'target']], train_size = 0.8, random_state=42)

text_field = torchtext.data.Field(lower=True, include_lengths=False, fix_length=1000, batch_first=True)
target_field = torchtext.data.Field(sequential=False, is_target=True, use_vocab=False)

train_dataset = torchtext.data.TabularDataset(cnn_data + 'train_cnn.csv', format='csv', fields={'text': ('text', text_field), 'target': ('target', target_field)})
val_dataset = torchtext.data.TabularDataset(cnn_data + 'val_cnn.csv', format='csv', fields={'text': ('text', text_field), 'target': ('target', target_field)})

text_field.build_vocab(train_dataset, min_freq=2)
target_field.build_vocab(train_dataset)
vocab = text_field.vocab


print('Vocab size: ', len(vocab))
print(train_dataset[0].text)
print(train_dataset[0].target)
# Define architecture of our CNN.

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_classes,
                 kernel_sizes_cnn, filters_cnn: int, dense_size: int,
                 dropout_rate: float = 0.,):
        super().__init__()

        self._n_classes = n_classes
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size
        self._kernel_sizes_cnn = kernel_sizes_cnn
        self._filters_cnn = filters_cnn
        self._dense_size = dense_size
        self._dropout_rate = dropout_rate

        self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)

        self.cnns = []
        for i in range(len(kernel_sizes_cnn)):
            in_channels = embedding_size

            cnn = nn.Sequential(
                nn.Conv1d(in_channels, filters_cnn, kernel_sizes_cnn[i]),
                nn.BatchNorm1d(filters_cnn),
                nn.ReLU()
            )
            cnn.apply(self.init_weights)

            self.add_module(f'cnn_{i}', cnn)
            self.cnns.append(cnn)
        
        # concatenated to hidden to classes
        self.projection = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(filters_cnn * len(kernel_sizes_cnn), dense_size),
            nn.BatchNorm1d(dense_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dense_size, n_classes)
        )

    @staticmethod
    def init_weights(module):
        if type(module) == nn.Linear or type(module) == nn.Conv1d:
            nn.init.kaiming_normal_(module.weight)

    def forward(self, x):
        x0 = self.embedding(x)
        x0 = torch.transpose(x0, 1, 2)

        outputs0 = []
        outputs1 = []

        for i in range(len(self.cnns)):
            cnn = getattr(self, f'cnn_{i}')
            # apply cnn and global max pooling
            pooled, _ = cnn(x0).max(dim=2)
            outputs0.append(pooled)

        x0 = torch.cat(outputs0, dim=1) if len(outputs0) > 1 else outputs0[0]
        return self.projection(x0)
# Prepare data loaders.
train_loader, val_loader = torchtext.data.Iterator.splits((train_dataset, val_dataset),
                                                           batch_sizes=(64, 64),
                                                           sort=False,
                                                           device='cuda')

batch = next(iter(train_loader))
print(batch)
# Init CNN model and parameters.

epochs = 8                        # you can different number of epochs. 
batch_size = 32                   # you can different number of batch size. 

vocab_size = len(vocab)
embedding_size = 300
n_classes = 2
kernel_sizes = (1, 2, 3, 5)       # You can try different values.
dense_size = 256                  # You can try different values.
dropout = 0.5                     # You can try different values.
filters_size = 512                # You can try different values.

model = CNN(vocab_size, embedding_size, n_classes, kernel_sizes,
            filters_size, dense_size, dropout)

model.cuda()  # move model to GPU

loss_function = nn.CrossEntropyLoss()
total_steps = len(train_loader) * epochs

# AdamW is a class from the huggingface library

# You can try different values of learning rate, i.e. [1e-2, 1e-3, 1e-4, 5e-3, 5e-4] and others. Check learning process to get the best value.

optimizer = AdamW(model.parameters(), 
                  lr=1e-3)                                          

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 10,  # You can try different values.
                                            num_training_steps = total_steps)
train_history, val_history = fit(model, loss_function, train_loader, val_loader, optimizer, epochs, batch_size, scheduler, type_nn='CNN')