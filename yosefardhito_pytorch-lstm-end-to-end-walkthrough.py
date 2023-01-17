import string

import re

import operator

import pickle

from collections import Counter



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from tqdm._tqdm_notebook import tqdm_notebook as tqdm; tqdm.pandas()



from nltk.tokenize.treebank import TreebankWordTokenizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

from torchtext.data import Example, Field, Dataset, LabelField, Iterator, BucketIterator

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim



# dataframe options to display the whole comments

pd.set_option('display.max_colwidth', -1)



INPUT = "../input/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load only necessary fields to reduce memory footprint

fields = ['id', 'comment_text', 'toxicity_annotator_count', 'insult', 'target']

# load training set

train_df = pd.read_csv(f"{INPUT}jigsaw-unintended-bias-in-toxicity-classification/train.csv", usecols=fields).set_index('id')

X_train = train_df['comment_text']

y_train = train_df['target'] >= 0.5  # create binary target column

train_df['is_toxic'] = y_train

train_df['is_insult'] = train_df['insult'] >= 0.5

# split training set to train/dev/test set with stratified strategy

X_train, X_dev_test, y_train, y_dev_test = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2, random_state=1337)

X_dev, X_test, y_dev, y_test = train_test_split(X_dev_test, y_dev_test, stratify=y_dev_test, test_size=0.5, random_state=1338)

# load submission set

test_df = pd.read_csv(f"{INPUT}jigsaw-unintended-bias-in-toxicity-classification/test.csv").set_index('id')

X_submission = test_df['comment_text']
print("Length train: {:,}; dev: {:,}; test: {:,}; submission: {:,}".format(

    len(X_train), len(X_dev), len(X_test), len(X_submission)

))
def load_embeddings(path):

    with open(path,'rb') as f:

        emb_arr = pickle.load(f)

    return emb_arr



EMBEDDING_PATH = f"{INPUT}pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl"

embeddings = load_embeddings(EMBEDDING_PATH)
# take all comments with toxicity annotator count >= 10 ** 1.5

# train_trust_df = train_df[(train_df.index.isin(X_train.index)) & (train_df['toxicity_annotator_count'] >= 10 ** 1.5)]

# filter comments with toxicity annotator count < 10 ** 1.5 based on the target value

# train_chosen_df = train_df[train_df.index.isin(X_train.index) & (train_df['toxicity_annotator_count'] < 10 ** 1.5)]

train_chosen_df = train_df[train_df.index.isin(X_train.index)]

# for safe comments, take only those with both target and insult value < 0.2 ("very safe")

train_chosen_very_safe_df = train_chosen_df[(train_chosen_df['target'] < 0.2) & (train_chosen_df['insult'] < 0.2)]

# for toxic comments, ignore those between 0.4 and 0.5

train_chosen_toxic_df = train_chosen_df[train_chosen_df['target'] >= 0.6]

train_chosen_others_df = train_chosen_df[(train_chosen_df['target'] >= 0.2) & (train_chosen_df['target'] < 0.4)]

# take maximum 1mil to fit processing within 2-hour time limit for submission and the given memory size

# n_very_safe = min(len(train_chosen_very_safe_df), 

#                   1100032 - len(train_trust_df) - len(train_chosen_others_df) - len(train_chosen_toxic_df))

n_very_safe = min(len(train_chosen_very_safe_df), 

                  1100032 - len(train_chosen_others_df) - len(train_chosen_toxic_df))

train_chosen_very_safe_sample_df = train_chosen_very_safe_df.sample(n=n_very_safe, random_state=1337)

# get all valid comment indices

train_indices = [

    #*train_trust_df.index.values,

    *train_chosen_very_safe_sample_df.index.values, 

    *train_chosen_others_df.index.values, 

    *train_chosen_toxic_df.index.values

]

X_train = X_train[X_train.index.isin(train_indices)]

X_dev = X_dev.sample(frac=1, random_state=1337)

X_test = X_test.sample(frac=1, random_state=1337)
print("Length train: {:,}; dev: {:,}; test: {:,}; submission: {:,}".format(

    len(X_train), len(X_dev), len(X_test), len(X_submission)

))
def build_character_set(sentences, verbose=True):

    characters_in_dataset = {}

    for sentence in tqdm(sentences, disable=(not verbose)):

        for character in sentence:

            try:

                characters_in_dataset[character] += 1

            except KeyError:

                characters_in_dataset[character] = 1

    return characters_in_dataset



train_charset = build_character_set(X_train)

# build all accepted characters

latin_similar = " ’'‘ÆÐƎƏƐƔĲŊŒẞÞǷȜæðǝəɛɣĳŋœĸſßþƿȝĄƁÇĐƊĘĦĮƘŁØƠŞȘŢȚŦŲƯY̨Ƴąɓçđɗęħįƙłøơşșţțŧųưy̨ƴÁÀÂÄǍĂĀÃÅǺĄÆǼǢƁĆĊĈČÇĎḌĐƊÐÉÈĖÊËĚĔĒĘẸƎƏƐĠĜǦĞĢƔáàâäǎăāãåǻąæǽǣɓćċĉčçďḍđɗðéèėêëěĕēęẹǝəɛġĝǧğģɣĤḤĦIÍÌİÎÏǏĬĪĨĮỊĲĴĶƘĹĻŁĽĿʼNŃN̈ŇÑŅŊÓÒÔÖǑŎŌÕŐỌØǾƠŒĥḥħıíìiîïǐĭīĩįịĳĵķƙĸĺļłľŀŉńn̈ňñņŋóòôöǒŏōõőọøǿơœŔŘŖŚŜŠŞȘṢẞŤŢṬŦÞÚÙÛÜǓŬŪŨŰŮŲỤƯẂẀŴẄǷÝỲŶŸȲỸƳŹŻŽẒŕřŗſśŝšşșṣßťţṭŧþúùûüǔŭūũűůųụưẃẁŵẅƿýỳŷÿȳỹƴźżžẓ"

white_list = string.ascii_letters + string.digits + latin_similar

# check all symbols that have embeddings to keep it

glove_symbols = [symbol for symbol in embeddings if len(symbol) == 1 and symbol not in white_list]

char_translate_map = {}

# isolate symbols that have embeddings vector in glove

for symbol in glove_symbols:

    char_translate_map[ord(symbol)] = ' ' + symbol + ' '

# remove all unknown symbols

unknown_symbols = [c for c in train_charset if c not in white_list and c not in glove_symbols]

for symbol in unknown_symbols:

    char_translate_map[ord(symbol)] = ''

# replace newline characters with normal separator (blank space)

char_translate_map[ord('\n')] = ' ' 
def find_in_embedding(word, embedding):

    '''

    find the word's vector in the embedding. 

    If there is no exact word found, try also with lower case and title case.

    In case there is no entry found in the embedding, replace with the unknown token (<unk>)

    '''

    if word in embedding:

        return word, embedding[word]

    elif word.lower() in embedding:

        return word.lower(), embedding[word.lower()]

    elif word.title() in embedding:

        return word.title(), embedding[word.title()]

    else:

        return '<unk>', embedding['<unk>']



def preprocess(comment, translate_map, tokenizer, emb):

    # translate symbols

    preprocessed_comment = comment.translate(translate_map)

    # handle quote at the beginning

    # because we accepted quotes as characters, sometimes we encounter token such as "'we".

    # with this step, we transform "'we" into "' we" and such that it can be tokenized as 2 tokens.

    preprocessed_comment = [("' " + token[1:]) if len(token) > 0 and token[0] == "'" else token for token in preprocessed_comment.split(' ')]

    # tokenize with Treebank tokenizer

    tokenized_comment = tokenizer.tokenize(' '.join(preprocessed_comment))

    # check and find each token in embeddings

    tokenized_comment = [find_in_embedding(token, emb)[0] for token in tokenized_comment]

    return ' '.join(tokenized_comment)



tokenizer = TreebankWordTokenizer()

X_train = X_train.progress_apply(lambda comment: preprocess(comment, char_translate_map, tokenizer, embeddings))

X_dev = X_dev.progress_apply(lambda comment: preprocess(comment, char_translate_map, tokenizer, embeddings))

X_test = X_test.progress_apply(lambda comment: preprocess(comment, char_translate_map, tokenizer, embeddings))

X_submission = X_submission.progress_apply(lambda comment: preprocess(comment, char_translate_map, tokenizer, embeddings))
X_train.head()
def prepare_torch_dataset(comments, target, fields):

    '''

    transform a series `comments` to torchtext `Dataset`

    '''

    df = pd.DataFrame(comments).join(target[['is_toxic', 'is_insult']], how='left')

    train_examples = [Example.fromlist(i, fields) for i in tqdm(df.values.tolist())]

    return Dataset(train_examples, fields)



# define the Fields

ID = Field(sequential=False, use_vocab=False, dtype=torch.long)

TEXT = Field(sequential=True, tokenize='spacy', pad_first=True)

LABEL = LabelField(sequential=False, use_vocab=False, dtype=torch.float32)

fields = [("comment_text", TEXT), ("target", LABEL), ("insult", LABEL)]

# prepare train/dev/test datasets as torch `Dataset`

train_dataset = prepare_torch_dataset(X_train, train_df, fields)

dev_dataset = prepare_torch_dataset(X_dev, train_df, fields)

test_dataset = prepare_torch_dataset(X_test, train_df, fields)

# prepare submission dataset as torch `Dataset`

test_df['comment_text'] = X_submission

subm_fields = [("id", ID), ("comment_text", TEXT)]  # notice that there is no label field for the submission dataset

subm_examples = [Example.fromlist(i, subm_fields) for i in tqdm(test_df.reset_index().values.tolist())]

subm_dataset = Dataset(subm_examples, subm_fields)
def build_matrix(vocab_itos, emb):

    embedding_matrix = np.zeros((len(vocab_itos), 300))

    for i, word in enumerate(vocab_itos):

        embedding_matrix[i] = find_in_embedding(word, emb)[1]

    return embedding_matrix



TEXT.build_vocab(train_dataset, dev_dataset, max_size=60000)

embedding_matrix = build_matrix(TEXT.vocab.itos, embeddings)
for i in range(20):

    word = TEXT.vocab.itos[i]

    print("id={}, token='{}', freq={}".format(i, word, TEXT.vocab.freqs[word]))
train_iter, dev_iter, test_iter, subm_iter = BucketIterator.splits(

    (train_dataset, dev_dataset, test_dataset, subm_dataset),

    batch_sizes=(64, 64, 64, 64),

    device=0,

    sort_key=lambda comment: len(comment.comment_text),

    sort_within_batch=False,

    repeat=False  # do not repeat any inputs

)



class BatchWrapper:

    '''

    helper wrapper to iterate over `BucketIterator` during training and return (x, y).

    from: https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/

    '''

    

    def __init__(self, dl, x_var, y_vars):

        self.dl, self.x_var, self.y_vars = dl, x_var, y_vars



    def __iter__(self):

        for batch in self.dl:

            # return x_var attribute of the data loader (dl) as x

            x = getattr(batch, self.x_var)

            # return label attributes of the data loader (dl) as y

            y = torch.cat([getattr(batch, feat).unsqueeze(1) for feat in self.y_vars], dim=1)

            yield (x, y)



    def __len__(self):

        return len(self.dl)



train_dl = BatchWrapper(train_iter, "comment_text", ["target", "insult"])

dev_dl = BatchWrapper(dev_iter, "comment_text", ["target", "insult"])

test_dl = BatchWrapper(test_iter, "comment_text", ["target", "insult"])
class SpatialDropout(nn.Dropout2d):

    def forward(self, x):

        x = x.unsqueeze(2)    # (N, T, 1, K)

        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)

        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked

        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)

        x = x.squeeze(2)  # (N, T, K)

        return x



class LSTM(nn.Module):

    def __init__(self, emb_matrix, lstm_units=150):

        super().__init__()

        # LAYER 1: EMBEDDING, non-trainable

        self.embedding = nn.Embedding(*emb_matrix.shape)

        # use glove embeddings

        self.embedding.weight = nn.Parameter(torch.tensor(emb_matrix, dtype=torch.float32), requires_grad=False)

        # LAYER 2: SPATIAL DROPOUT

        self.embedding_dropout = SpatialDropout(0.3)

        # LAYER 3: LSTM, bi-directional, output = 2 * lstm_units

        self.lstm_units = lstm_units

        self.lstm1 = nn.LSTM(embedding_matrix.shape[1], lstm_units, bidirectional=True)

        # LAYER 4: LSTM, bi-directional, output = 2 * lstm_units

        self.lstm2 = nn.LSTM(2 * lstm_units, lstm_units, bidirectional=True)

        # LAYER 5: CONCAT, no object created

        # LAYER 6: FC1

        self.linear1 = nn.Linear(3 * 2 * lstm_units, 3 * 2 * lstm_units)

        # LAYER 7: FC2

        self.linear2 = nn.Linear(3 * 2 * lstm_units, 3 * 2 * lstm_units)

        # LAYER 8: ADDITION of layer outputs, no object created

        # LAYER 9: DROPOUT

        self.dropout = nn.Dropout(0.1)

        # LAYER 10: OUTPUT LAYER for target

        self.target_out = nn.Linear(3 * 2 * lstm_units, 1)

        # LAYER 11: OUTPUT LAYER for insult

        self.aux_out = nn.Linear(3 * 2 * lstm_units, 1)



    def forward(self, seq):

        # get embedding vector of every word

        h_emb = self.embedding_dropout(self.embedding(seq))

        # walk through the first bi-directional LSTM

        h_lstm1, _ = self.lstm1(h_emb)

        # use output of previous LSTM as input for the second bi-directional LSTM

        h_lstm2, _ = self.lstm2(h_lstm1)

        # get the latest state of LSTM 2

        # because it is bi-directional, get the first half output of the last state (LSTM direction forward)

        # and the second half output of the first state (LSTM direction backward)

        h_lstm_last = h_lstm2[-1, :, :self.lstm_units]

        h_lstm_first = h_lstm2[0, :, self.lstm_units:]

        # concat both half

        h_bi_lstm = torch.cat((h_lstm_first, h_lstm_last), -1)

        # get average and max pool of all hidden state from LSTM 2

        avg_pool = torch.mean(h_lstm2, 0)

        max_pool, _ = torch.max(h_lstm2, 0)

        # concat last state with pooling

        h_conc = torch.cat((h_bi_lstm, max_pool, avg_pool), 1)

        # fully-connected layer with last state and the pooling

        h_lin1 = F.relu(self.linear1(h_conc))

        h_lin2 = F.relu(self.linear2(h_conc))

        # add concat with output of fully-connected layer

        h_conc_add = h_conc + h_lin1 + h_lin2

        # another dropout to force model check last state, average pool, and max pool values

        h_conc_do = self.dropout(h_conc_add)

        # get linear prediction of target and insult

        # we do not use sigmoid here because pyTorch provides a better loss calculation

        # by combining sigmoid with cross-entropy

        target_out = self.target_out(h_conc_do)

        aux_out = self.aux_out(h_conc_do)

        out = torch.cat([target_out, aux_out], 1)

        return out

    

model = LSTM(embedding_matrix).to(device)

model
def sigmoid(x):

    return 1 / (1 + np.exp(-x))



def evaluate(model, dl, loss_func):

    '''

    evaluation steps for dev and test set

    '''

    running_loss = 0.0

    model.eval()  # turn on evaluation mode

    all_preds = {'y_pred' : [], 'y_true' : []}

    for x, y in dl: # iterate over mini-batches

        x, y = x.to(device), y.to(device)  # transfer x and y to GPU when it is available

        preds = model(x)  # get the prediction from the model

        loss = loss_func(preds, y)  # calculate loss values

        # add all predictions and y_true to a dictionary for calculating the evaluation metrics

        all_preds['y_pred'] = [*all_preds['y_pred'], *sigmoid(preds.detach().cpu().numpy()[:,0].ravel())]

        all_preds['y_true'] = [*all_preds['y_true'], *y.detach().cpu().numpy()[:,0].ravel()]

        # add current mini-batch loss to the total loss

        running_loss += loss.item() / len(dl)

    # create a binary prediction to calculate accuracy

    all_preds['y_pred_bin'] = [1 if pred >= 0.5 else 0 for pred in all_preds['y_pred']]

    print('\tEval Loss: {:.4f}, Eval AUC: {:.4f}, Eval Accuracy: {:.4f}'.format(

        loss,

        roc_auc_score(all_preds['y_true'], all_preds['y_pred']),

        accuracy_score(all_preds['y_true'], all_preds['y_pred_bin'])

    ))

    # confusion analysis, especially helpful for imbalanced dataset

    tn, fp, fn, tp = confusion_matrix(all_preds['y_true'], all_preds['y_pred_bin']).ravel()

    print('\tEval CM: tn={:,}({:.2f}%), tp={:,}({:.2f}%), fn={:,}({:.2f}%), fp={:,}({:.2f}%)'.format(

        tn, tn * 100 / len(all_preds['y_pred_bin']),

        tp, tp * 100 / len(all_preds['y_pred_bin']),

        fn, fn * 100 / len(all_preds['y_pred_bin']),

        fp, fp * 100 / len(all_preds['y_pred_bin'])

    ))

    

    

def train(model, train, dev, n_epochs, loss_func, lr):

    '''

    model training steps for the training set. Output dev metrics at each epoch.

    PyTorch requires implementation of training procedure.

    '''

    opt = optim.Adam(model.parameters(), lr=lr)  # optimizer

    # degrade learning rate at every epoch with lr scheduler

    scheduler = optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.3)

    for epoch in range(1, n_epochs + 1):

        print('Epoch: {}'.format(epoch))

        model.train()  # during training, dropout is activated

        running_loss = 0.0

        for x, y in tqdm(train):

            opt.zero_grad()  # necessary at the beginning of each mini-batch for pytorch

            x, y = x.to(device), y.to(device)  # transfer to GPU when available

            preds = model(x)  # get prediction

            loss = loss_func(preds, y)  # calculate loss

            loss.backward()  # backpropagation

            opt.step()  # calculate optimizer with adam

            running_loss += loss.item() / len(train_dl)  # add current mini-batch loss to total loss

        print('\tTraining Loss: {:.4f}'.format(running_loss))

        evaluate(model, dev, loss_func)  # check dev scores

        scheduler.step()  # reduce learning rate



# Binary Cross-Entropy With Logits Loss: optimized loss function for binary classification

# when using this loss function, do not use sigmoid activation function in the output layer of the model.

bce_loss = nn.BCEWithLogitsLoss(

    pos_weight=torch.FloatTensor([1]).cuda(),   # I tried to penalize mis-classified positive instance more

                                                # but it did not give better LB score

    reduction='mean'  # because we are predicting 2 outputs, take the mean as loss value

)

train(model, train_dl, dev_dl, 2, bce_loss, lr=8e-4)  # run training
evaluate(model, test_dl, bce_loss)  # evaluate on test set
# fine-tuning embedding layer

model.embedding.weight.requires_grad = True

train(model, train_dl, dev_dl, 1, bce_loss, lr=8e-5)
evaluate(model, test_dl, bce_loss)  # evaluate on test set
model.eval()  # turn on evaluation mode to ignore dropouts

subm_id = []

subm_preds = []

for example in tqdm(subm_iter):

    x_id = example.id

    x = example.comment_text.to(device)

    preds = model(x)

    subm_id = [*subm_id, *x_id.detach().cpu().numpy().ravel()]

    subm_preds = [*subm_preds, *sigmoid(preds.detach().cpu().numpy()[:,0].ravel())]

submission = pd.DataFrame.from_dict({

    'id': subm_id,

    'prediction': subm_preds

})

submission.to_csv('submission.csv', index=False)