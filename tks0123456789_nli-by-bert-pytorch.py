!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q
!pip install pytorch-pretrained-bert
import time

import os

import random



import numpy as np

import pandas as pd

import re

import spacy



import torch

from torch.optim import Adam

from torch.nn import Module, Linear, Dropout

from torch.nn import BCEWithLogitsLoss

from torch.utils.data import Dataset, DataLoader

from torch.nn.utils.rnn import pad_sequence



import torch.nn.functional as F



from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.modeling import BertModel, BertLayer

from pytorch_pretrained_bert.optimization import BertAdam



from sklearn.metrics import log_loss

from sklearn.preprocessing import normalize
device = torch.device("cuda")



seed = 42



random.seed(seed)

os.environ["PYTHONHASHSEED"] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
# Preprocessing

max_len = 300

do_lower_case = True

N = "nobody"



# Model

bert_model = "bert-base-uncased"

n_bertlayers = 12

dropout = 0.1

no_pooler = True



# Training

optim = "bertadam"

train_batch_size = 20

num_train_epochs = 3

gradient_accumulation_steps = 1

learning_rate = 2e-5

warmup_proportion = 0.1

weight_decay = False



# Evaluation

eval_batch_size = 32
nlp = spacy.load("en")

def get_sentence(text, offset, token_after="[PRONOUN]"):

    """

    Extract a sentence containing a word at position offset by character and

    replace the word with token_after.

    output: Transformed sentence

            token_before

            a pos tag of the word.

    """

    doc = nlp(text)

    # idx: Character offset

    idx_begin = 0

    for token in doc:

        if token.sent_start:

            idx_begin = token.idx

        if token.idx == offset:

            sent = token.sent.string

            pos_tag = token.pos_

            idx_token = offset - idx_begin

            break

    token_before = token.string.strip()

    subtxt_transformed = re.sub("^" + token_before, token_after, sent[idx_token:])

    sent_transformed = sent[:idx_token] + subtxt_transformed

    return sent_transformed, token_before, pos_tag





def generate_choices(text, offset, A, B, N=None):

    """

    Extract a sentence contain a pronoun at a offset position.

    Then replace the pronoun with A, B or N.

        3 choices.

        [Pronoun] likes something. ==>

          A likes something.

          B likes something.

          neigher A nor B likes something. (If N is None.)

          N likes something. (If N is not None.)

    text:  str

    offset: int

    A, B: Person's names. str

    N: nobody or something. str

    """

    sents = []

    text_pronoun, pronoun, pos_tag = get_sentence(text, offset)

    if pos_tag == "ADJ" or pronoun == "hers":

        _post = "'s"

    elif pronoun == "his":

        _post = "'s"

    else:

        _post = ""

    who_s = [A + _post, B + _post]

    if N is None:

        who_s += ["neither " + A + " nor " + B]

    else:

        who_s += [N + _post]

    sents.extend([re.sub("\[PRONOUN\]", who, text_pronoun) for who in who_s])



    return sents
# https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_swag.py#L216

def _truncate_seq_pair(tokens_a, tokens_b, max_length):

    """Truncates a sequence pair in place to the maximum length."""



    # This is a simple heuristic which will always truncate the longer sequence

    # one token at a time. This makes more sense than truncating an equal percent

    # of tokens from each, since if one sequence is very short then each token

    # that's truncated likely contains more information than a longer sequence.

    while True:

        total_length = len(tokens_a) + len(tokens_b)

        if total_length <= max_length:

            break

        if len(tokens_a) > len(tokens_b):

            tokens_a.pop()

        else:

            tokens_b.pop()





class NLIDataset(Dataset):

    """

    NLI Dataset

    p_texts: Premise texts

    h_texts: Hypothesis texts

    tokenizer: BertTokenizer

    y      : Target sequence

    y_values: Class labels

    """

    def __init__(self, p_texts, h_texts, tokenizer,

                 y=None, y_values=None, max_len=100):

        if y is None:

            self.labels = None

        else:

            mapper = {label: i for i, label in enumerate(y_values)}

            self.labels = torch.LongTensor([mapper[v] for v in y])



        self.max_tokens = 0

        self.inputs = []

        for e, (p_text, h_text) in enumerate(zip(p_texts, h_texts)):

            p_tokens = tokenizer.tokenize(p_text)

            h_tokens = tokenizer.tokenize(h_text)

            _truncate_seq_pair(p_tokens, h_tokens, max_len - 3)

            p_len = len(p_tokens)

            h_len = len(h_tokens)



            tokens = ["[CLS]"] + p_tokens + ["[SEP]"] + h_tokens + ["[SEP]"]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            segment_ids = [0] * (p_len + 2) + [1] * (h_len + 1)

            input_mask = [1] * len(input_ids)

            self.inputs.append([torch.LongTensor(input_ids),

                                torch.LongTensor(segment_ids),

                                torch.LongTensor(input_mask)])

            self.max_tokens = max(p_len + h_len + 3, self.max_tokens)

            if e < 1:

                print("tokens:", p_tokens)



        print("max_len:", self.max_tokens)



    def __getitem__(self, index):

        return self.inputs[index], self.labels[index]



    def __len__(self):

        return len(self.inputs)





def get_gap_nli_dataset(df, tokenizer, max_len, labeled=True, N=""):

    p_texts = df["Text"].repeat(3)

    h_texts = df.apply(lambda x:

                       generate_choices(x["Text"], x["Pronoun-offset"], x["A"], x["B"], N=N),                                     

                       axis=1)

    h_texts = sum(h_texts, [])

    if labeled:

        y_A = df["A-coref"].astype(int)

        y_B = df["B-coref"].astype(int)

        y_Neither = 1 - y_A - y_B

        labels = np.column_stack((y_A, y_B, y_Neither)).reshape(-1)

    return NLIDataset(p_texts, h_texts, tokenizer,

                      y=labels, y_values=(0, 1), max_len=max_len)





def collate_fn(batch):

    """

    Pad the inputs sequence.

    """

    x_lst, y_lst = list(zip(*batch))

    xy_batch = [pad_sequence(x, batch_first=True) for x in zip(*x_lst)]

    xy_batch.append(torch.stack(y_lst, dim=0))

    return xy_batch
def get_pretrained_bert(modelname, n_bertlayers=None):

    bert = BertModel.from_pretrained(modelname)

    if n_bertlayers is None:

        return bert

    if n_bertlayers < bert.config.num_hidden_layers:

        # Only use the bottom n layers

        del bert.encoder.layer[n_bertlayers:]

        bert.config.num_hidden_layers = n_bertlayers

    return bert





class BertCl(Module):

    def __init__(self, modelname, n_bertlayers, dropout, num_labels,

                 no_pooler=False):

        super(BertCl, self).__init__()

        self.bert = get_pretrained_bert(modelname, n_bertlayers)

        self.dropout = Dropout(dropout)

        self.classifier = Linear(self.bert.config.hidden_size, num_labels)

        self.no_pooler = no_pooler



    def forward(self, input_ids, segment_ids, input_mask):

        encoded_layer, pooled_output = self.bert(input_ids, segment_ids, input_mask,

                                                 output_all_encoded_layers=False)

        if self.no_pooler:

            x = self.classifier(self.dropout(encoded_layer[:, 0]))

        else:

            x = self.classifier(self.dropout(pooled_output))

        return x





def predict(model, data_loader, device, proba=True, to_numpy=True):

    model.eval()

    preds = []

    for step, batch in enumerate(data_loader):

        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():

            logits = model(*batch[:-1])

            preds.append(logits.detach().cpu())

    preds = torch.cat(preds) if len(preds) > 1 else preds[0]

    if proba:

        if preds.size(-1) > 1:

            preds = F.softmax(preds, dim=1)

        else:

            preds = torch.sigmoid(preds)

    if to_numpy:

        preds = preds.numpy()

    return preds





def get_param_size(model):

    trainable_psize = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

    total_psize = np.sum([np.prod(p.size()) for p in model.parameters()])

    return total_psize, trainable_psize
def get_loader(train_df, val_df, test_df):

    tokenizer = BertTokenizer.from_pretrained(bert_model,

                                              do_lower_case=do_lower_case)



    train_ds = get_gap_nli_dataset(train_df, tokenizer, max_len, labeled=True, N=N)                       

    val_ds = get_gap_nli_dataset(val_df, tokenizer, max_len, labeled=True, N=N)

    test_ds = get_gap_nli_dataset(test_df, tokenizer, max_len, labeled=True, N=N)

    

    train_loader = DataLoader(

        train_ds,

        batch_size=train_batch_size,

        collate_fn=collate_fn,

        shuffle=True,

        drop_last=True)

    val_loader = DataLoader(

        val_ds,

        batch_size=eval_batch_size,

        collate_fn=collate_fn,

        shuffle=False)

    test_loader = DataLoader(

        test_ds,

        batch_size=eval_batch_size,

        collate_fn=collate_fn,

        shuffle=False)



    return train_loader, val_loader, test_loader





def get_gap_cl_model(device, steps_per_epoch, bert_model, n_bertlayers, dropout,

                     num_labels=1, no_pooler=False):

    model = BertCl(bert_model, n_bertlayers, dropout,

                   num_labels=num_labels, no_pooler=no_pooler)

    model.to(device)



    param_optimizer = list(model.named_parameters())



    if weight_decay:

        no_decay = ["bias", "gamma", "beta", "classifier"]

        optimizer_grouped_parameters = [

            {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],

             "weight_decay": 0.01},

            {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],

             "weight_decay": 0.0}

        ]

    else:

        optimizer_grouped_parameters = [

            {"params": [p for n, p in param_optimizer], "weight_decay": 0.0}



        ]



    t_total = int(

        steps_per_epoch / gradient_accumulation_steps * num_train_epochs)

    if optim == "bertadam":

        optimizer = BertAdam(optimizer_grouped_parameters,

                             lr=learning_rate,

                             warmup=warmup_proportion,

                             t_total=t_total)

    elif optim == "adam":

        optimizer = Adam(optimizer_grouped_parameters,

                         lr=learning_rate)



    return model, optimizer







def run_epoch(model, dataloader, optimizer, criterion, device, verbose_step=10000):

    model.train()

    t1 = time.time()

    tr_loss = 0

    for step, batch in enumerate(dataloader):

        batch = tuple(t.to(device) for t in batch)

        label_ids = batch[-1]

        outputs = model(*batch[:-1])

        if criterion._get_name() == "BCEWithLogitsLoss":

            outputs = outputs[:, 0]

            label_ids = label_ids.float()

        loss = criterion(outputs, label_ids)

        if gradient_accumulation_steps > 1:

            loss = loss / gradient_accumulation_steps

        loss.backward()

        tr_loss += loss.item()

        if (step + 1) % verbose_step == 0:

            loss_now = gradient_accumulation_steps * tr_loss / (step + 1)

            print(f"step:{step+1} loss:{loss_now:.7f} time:{time.time() - t1:.1f}s")

        if (step + 1) % gradient_accumulation_steps == 0:

            optimizer.step()

            model.zero_grad()

    return gradient_accumulation_steps * tr_loss / (step + 1)





def eval_model(model, dataloader, y, device):

    pr_sgmd = predict(model, dataloader, device, proba=True, to_numpy=True)[:, 0].reshape((-1, 3))

    loss_s = [log_loss(y[i::3], pr_sgmd[:, i]) for i in range(3)]

    pr_ABN = normalize(pr_sgmd, norm="l1")

    ABN_loss = log_loss(y.reshape((-1, 3)), pr_ABN)

    return {"A_loss": loss_s[0], "B_loss": loss_s[1], "N_loss": loss_s[2],

            "ABN_loss": ABN_loss}
train_df = pd.read_csv("gap-test.tsv", delimiter="\t")

val_df = pd.read_csv("gap-validation.tsv", delimiter="\t")

test_df = pd.read_csv("gap-development.tsv", delimiter="\t")



val_y_AB = val_df[["A-coref", "B-coref"]].astype(int)

val_y_N = 1 - val_y_AB.sum(1)

val_y = np.column_stack((val_y_AB, val_y_N)).reshape(-1)



test_y_AB = test_df[["A-coref", "B-coref"]].astype(int)

test_y_N = 1 - test_y_AB.sum(1)

test_y = np.column_stack((test_y_AB, test_y_N)).reshape(-1)
train_loader, val_loader, test_loader = get_loader(train_df, val_df, test_df)
model, optimizer = get_gap_cl_model(device, len(train_loader),

                                    bert_model, n_bertlayers, dropout, no_pooler=no_pooler)

total_psize, trainalbe_psize = get_param_size(model)

print(f"Total params: {total_psize}\nTrainable params: {trainalbe_psize}")
criterion = BCEWithLogitsLoss()

for e in range(num_train_epochs):

    t1 = time.time()

    tr_loss = run_epoch(model, train_loader, optimizer, criterion, device)

    val_obj = eval_model(model, val_loader, val_y, device)

    print(f"Epoch:{e + 1} tr_loss:{tr_loss:.4f}"

          f"\n val  A_loss:{val_obj['A_loss']:.4f}"

          f" B_loss:{val_obj['B_loss']:.4f}"

          f" N_loss:{val_obj['N_loss']:.4f}"

          f" ABN_loss: {val_obj['ABN_loss']:.4f}",

          f" time:{time.time() - t1:.1f}s")
test_obj = eval_model(model, test_loader, test_y, device)

print(f" \ntest   ABN_loss: {test_obj['ABN_loss']:.4f}")