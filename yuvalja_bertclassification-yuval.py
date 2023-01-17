# If working in google colab need to run to make the data accessible 

# !pip install pytorch_pretrained_bert
# from google.colab import drive
# drive.mount('/content/gdrive')
!pip install pytorch_pretrained_bert # If running in kaggle kernel make sure internet is enabled (under settings/Internet)
import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam

import pandas as pd
import numpy as np
# init_checkpoint_pt = '../pretrained_models/bert/cased_L-12_H-768_A-12/'
# DATA_PATH = "/content/gdrive/My Drive/Colab Notebooks/"
DATA_PATH = "../input/"
MAX_LEN = 350
bs = 8
import pandas as pd
df = pd.read_csv(DATA_PATH + 'cp_challenge_train.csv')
df = df[:32] # TODO: Comment out - just to make sure everything runs....
sentences = df.Plot
classes = list(set(df['label'].values))
clas2idx = {t : i for i,t in enumerate(classes)}
y = [clas2idx.get(l) for l in df.label.values]
# tokenize the words
tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
def trp(l, n):
    return l[:n] + [0]*(n-len(l))

# bert get's id's and not words, so convert to that and padd (and trim) the sentences to by in MAX_LEN
input_ids = [trp(tokenizer.convert_tokens_to_ids(txt), MAX_LEN) for txt in tokenized_texts]
# We can tell bery where we added words just for padding, it will help him embed better
attention_masks = [[float(i>0) for i in ii] for ii in input_ids]
# split train test, use random_state so it will be the same split :)
tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, y, 
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)
# transform the vectors into something pytorch can read
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device == 'cude':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)
model = BertForSequenceClassification.from_pretrained("bert-base-cased", num_labels=len(clas2idx))

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'gamma', 'beta']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)

from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm, trange
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 3
max_grad_norm = 1.0
for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
#         print("forward pass")
        loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
#         print("backward pass")
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy, eval_f1 = 0, 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            predictions.extend([np.argmax(logits, axis=1)])
            true_labels.append(label_ids)
            tmp_eval_accuracy = accuracy_score(np.argmax(logits, axis=1), label_ids)
            tmp_eval_f1 = f1_score(np.argmax(logits, axis=1), label_ids, average='weighted')
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy
            eval_f1 += tmp_eval_f1
            nb_eval_examples += b_input_ids.size(0)
            nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    print("Validation f1: {}".format(eval_f1/nb_eval_steps))
    torch.save(model, 'bert_classifier')
df_test = pd.read_csv(DATA_PATH + 'challenge_testset.csv')

test_tokenized_texts = [tokenizer.tokenize(sent) for sent in df_test.Plot]
test_input_ids = [trp(tokenizer.convert_tokens_to_ids(txt), MAX_LEN) for txt in test_tokenized_texts]
test_attention_masks = [[float(i>0) for i in ii] for ii in test_input_ids]
te_inputs = torch.tensor(test_input_ids)
te_mask = torch.tensor(test_attention_masks)
test_data = TensorDataset(te_inputs, te_mask)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=2)
predictions = []
for batch in test_dataloader:
    batch = tuple(t.to(device) for t in batch)
    model_input, mask = batch
    y_hat_proba = model(model_input, token_type_ids=None, attention_mask=mask)
    y_hat_proba = y_hat_proba.detach().cpu().numpy()
    predictions.extend(y_hat_proba)
#     print(predictions)
idx2class = {v:k for k,v in clas2idx.items()}
idx2class[-1] = 'unknown'

y_hat = list(np.argmax(predictions, axis=1))
df_test['predictions'] = [idx2class[i] for i in y_hat]
df_test[['imdbID', 'predictions']].to_csv(DATA_PATH + 'predictions.csv', index=False)
