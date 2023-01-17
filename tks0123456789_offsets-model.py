!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-development.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-test.tsv -q

!wget https://github.com/google-research-datasets/gap-coreference/raw/master/gap-validation.tsv -q
!ls ../input
!pip install pytorch-pretrained-bert
import time

import os

import random



import numpy as np

import pandas as pd



import torch

from torch.optim import Adam

from torch.utils.data import Dataset

from torch.nn import Module, Linear, Dropout

import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel, BertLayer

from pytorch_pretrained_bert import BertTokenizer

from pytorch_pretrained_bert.optimization import BertAdam

from pytorch_pretrained_bert.optimization import warmup_linear

from torch.utils.data import DataLoader

from torch.utils.data import RandomSampler



from sklearn.metrics import log_loss

import matplotlib.pyplot as plt
seed = 9876



random.seed(seed)

os.environ['PYTHONHASHSEED'] = str(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
# Model

bert_model = "bert-large-cased"

n_bertlayers = 22

dropout = 0.1



# Preprocessing

do_lower_case = False



# Training

train_batch_size = 4

gradient_accumulation_steps = 5

lr = 1e-5

num_train_epochs = 2

warmup_proportion = 0.1

optim = "bertadam"

weight_decay = False





# Others

n_models = 10

eval_batch_size = 32



device = torch.device("cuda")

data_dir = ""
def insert_tag(row):

    """Insert custom tags to help us find the position of A, B, and the pronoun after tokenization."""

    to_be_inserted = sorted([

        (row["A-offset"], " [A] "),

        (row["B-offset"], " [B] "),

        (row["Pronoun-offset"], " [P] ")

    ], key=lambda x: x[0], reverse=True)

    text = row["Text"]

    for offset, tag in to_be_inserted:

        text = text[:offset] + tag + text[offset:]

    return text





def tokenize(text, tokenizer):

    """Returns a list of tokens and the positions of A, B, and the pronoun."""

    entries = {}

    final_tokens = []

    for token in tokenizer.tokenize(text):

        if token in ("[A]", "[B]", "[P]"):

            entries[token] = len(final_tokens)

            continue

        final_tokens.append(token)

    return final_tokens, (entries["[A]"], entries["[B]"], entries["[P]"])





class GAPDataset(Dataset):

    """Custom GAP Dataset class"""

    def __init__(self, df, tokenizer, labeled=True):

        self.labeled = labeled

        if labeled:

            tmp = df[["A-coref", "B-coref"]].copy()

            tmp["Neither"] = ~(df["A-coref"] | df["B-coref"])

            self.y = tmp.values.astype("bool")

        # Extracts the tokens and offsets(positions of A, B, and P)

        self.offsets, self.tokens = [], []

        self.seq_len = []

        for _, row in df.iterrows():

            text = insert_tag(row)

            tokens, offsets = tokenize(text, tokenizer)

            self.offsets.append(offsets)

            self.tokens.append(tokenizer.convert_tokens_to_ids(

                ["[CLS]"] + tokens + ["[SEP]"]))

            self.seq_len.append(len(self.tokens[-1]))



    def __len__(self):

        return len(self.tokens)



    def __getitem__(self, idx):

        if self.labeled:

            return self.tokens[idx], self.offsets[idx], self.y[idx]

        return self.tokens[idx], self.offsets[idx]



    def get_seq_len(self):

        return self.seq_len





def collate_examples(batch, truncate_len=500):

    """Batch preparation.



    1. Pad the sequences

    2. Transform the target.

    """

    transposed = list(zip(*batch))

    max_len = min(

        max((len(x) for x in transposed[0])),

        truncate_len

    )

    tokens = np.zeros((len(batch), max_len), dtype=np.int64)

    for i, row in enumerate(transposed[0]):

        row = np.array(row[:truncate_len])

        tokens[i, :len(row)] = row

    token_tensor = torch.from_numpy(tokens)

    # Offsets

    offsets = torch.stack([

        torch.LongTensor(x) for x in transposed[1]

    ], dim=0) + 1 # Account for the [CLS] token

    # Labels

    if len(transposed) == 2:

        return token_tensor, offsets, None

    one_hot_labels = torch.stack([

        torch.from_numpy(x.astype("uint8")) for x in transposed[2]

    ], dim=0)

    _, labels = one_hot_labels.max(dim=1)

    return token_tensor, offsets, labels
def get_pretrained_bert(modelname, num_hidden_layers=None):

    bert = BertModel.from_pretrained(modelname)

    if num_hidden_layers is None:

        return bert

    old_num_hidden_layers = bert.config.num_hidden_layers

    if num_hidden_layers < old_num_hidden_layers:

        # Only use the bottom n layers

        del bert.encoder.layer[num_hidden_layers:]

    elif num_hidden_layers > old_num_hidden_layers:

        # Add BertLayer(s)

        for i in range(old_num_hidden_layers, num_hidden_layers):

            bert.encoder.layer.add_module(str(i), BertLayer(bert.config))

    if num_hidden_layers != old_num_hidden_layers:

        bert.config.num_hidden_layers = num_hidden_layers

        bert.init_bert_weights(bert.pooler.dense)

    return bert





class BertCl_GAP(Module):

    """The main model."""

    def __init__(self, bert, dropout, n_offsets=3):

        super().__init__()

        self.bert = bert

        self.bert_hidden_size = self.bert.config.hidden_size

        self.dropout = Dropout(dropout)

        self.classifier = Linear(self.bert.config.hidden_size * n_offsets, n_offsets)



    def forward(self, token_tensor, offsets, label_id=None):

        bert_outputs, _ = self.bert(

            token_tensor, attention_mask=(token_tensor > 0).long(),

            token_type_ids=None, output_all_encoded_layers=False)

        extracted_outputs = bert_outputs.gather(

            1, offsets.unsqueeze(2).expand(-1, -1, bert_outputs.size(2))

        ).view(bert_outputs.size(0), -1)

        outputs = self.classifier(self.dropout(extracted_outputs))

        return outputs





def get_param_size(model):

    trainable_psize = np.sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])

    total_psize = np.sum([np.prod(p.size()) for p in model.parameters()])

    return total_psize, trainable_psize





def run_epoch(model, dataloader, optimizer, criterion, device,

              verbose_step=10000):

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

            print(f'step:{step+1} loss:{loss_now:.7f} time:{time.time() - t1:.1f}s')

        if (step + 1) % gradient_accumulation_steps == 0:

            optimizer.step()

            model.zero_grad()

    return gradient_accumulation_steps * tr_loss / (step + 1)





def predict(model, data_loader, device, proba=True, to_numpy=True):

    model.eval()

    preds = []

    for step, batch in enumerate(data_loader):

        batch = batch[:2]

        batch = tuple(t.to(device) for t in batch)

        # input_ids, offsets, label_ids = batch

        # label_ids = batch[-1]

        with torch.no_grad():

            logits = model(*batch)

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
def get_gap_model(bert_model, n_bertlayers, dropout,

                  steps_per_epoch, device):

    bert = get_pretrained_bert(bert_model, n_bertlayers)

    model = BertCl_GAP(bert, dropout)



    model.to(device)



    param_optimizer = list(model.named_parameters())



    if weight_decay:

        no_decay = ["bias", "gamma", "beta", "head"]

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

    if optim == 'bertadam':

        optimizer = BertAdam(optimizer_grouped_parameters,

                             lr=lr,

                             warmup=warmup_proportion,

                             t_total=t_total)

    elif optim == 'adam':

        optimizer = Adam(optimizer_grouped_parameters, lr=lr)

    return model, optimizer





def get_loader(train_df, val_df, test_df):

    tokenizer = BertTokenizer.from_pretrained(

        bert_model,

        do_lower_case=do_lower_case,

        never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]", "[A]", "[B]", "[P]")

    )

    # These tokens are not actually used, so we can assign arbitrary values.

    tokenizer.vocab['[A]'] = -1

    tokenizer.vocab['[B]'] = -1

    tokenizer.vocab['[P]'] = -1



    train_ds = GAPDataset(train_df, tokenizer)

    val_ds = GAPDataset(val_df, tokenizer)

    test_ds = GAPDataset(test_df, tokenizer, labeled=False)

        

    train_loader = DataLoader(

        train_ds,

        collate_fn=collate_examples,

        batch_size=train_batch_size,

        shuffle=True,

        drop_last=True

    )

    val_loader = DataLoader(

        val_ds,

        collate_fn=collate_examples,

        batch_size=eval_batch_size,

        shuffle=False

    )

    test_loader = DataLoader(

        test_ds,

        collate_fn=collate_examples,

        batch_size=eval_batch_size,

        shuffle=False

    )

    return train_loader, val_loader, test_loader
train_df = pd.concat([pd.read_csv(data_dir + "gap-test.tsv", delimiter="\t"),

                      pd.read_csv(data_dir + "gap-development.tsv", delimiter="\t")])

val_df = pd.read_csv(data_dir + "gap-validation.tsv", delimiter="\t")

val_y = val_df[['A-coref', 'B-coref']].astype(int)

val_y['None'] = 1 - val_y.sum(1)



test_df = pd.read_csv("../input/test_stage_2.tsv", delimiter="\t")



print(f"Train:{train_df.shape[0]}, Valid:{val_df.shape[0]}, Test:{test_df.shape[0]}")
train_loader, val_loader, test_loader = get_loader(train_df, val_df, test_df)
steps_per_epoch = len(train_loader)

steps_per_epoch
scores = []



criterion = torch.nn.CrossEntropyLoss()

val_pr_avg = [np.zeros(val_y.shape) for _ in range(num_train_epochs)]

test_pr_avg = np.zeros((test_df.shape[0], 3))
for model_id in range(n_models):

    model, optimizer = get_gap_model(bert_model, n_bertlayers, dropout,

                                     steps_per_epoch, device)

    total_psize, trainalbe_psize = get_param_size(model)

    print(f"Total params: {total_psize}\nTrainable params: {trainalbe_psize}")

    for e in range(num_train_epochs):

        t1 = time.time()

        tr_loss = run_epoch(model, train_loader, optimizer, criterion, device)

        val_pr = predict(model, val_loader, device)

        val_pr_avg[e] += val_pr

        val_loss = log_loss(val_y, val_pr)

        val_avg_loss = log_loss(val_y, val_pr_avg[e] / (model_id + 1))

        elapsed = time.time() - t1

        print(f"Epoch:{e + 1} tr_loss:{tr_loss:.4f} val_loss:{val_loss:.4f}"

              f" val_avg_loss: {val_avg_loss:.4f} time:{elapsed:.1f}s")

        scores.append({"model_id": model_id, "epoch": e + 1, "time": elapsed,

                       "tr_loss": tr_loss, "val_loss": val_loss, "val_avg_loss": val_avg_loss})

    test_pr = predict(model, test_loader, device)

    test_pr_avg += test_pr

    del model, optimizer

    torch.cuda.empty_cache()
df = pd.DataFrame(scores)



pd.set_option("precision", 5)

print("\nSingle model")

print(df.groupby("epoch")[['tr_loss', 'val_loss']].mean())



if n_models > 1:

    print(f"\nAvg of {n_models} models")

    print(df[df.model_id == n_models - 1][['epoch', 'val_avg_loss']].set_index('epoch'))
test_pr_avg /= n_models

df_sub = pd.DataFrame(test_pr_avg, columns=["A", "B", "NEITHER"])

df_sub["ID"] = test_df.ID

df_sub.to_csv("submission.csv", index=False)
