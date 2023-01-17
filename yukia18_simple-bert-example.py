DEGUB = False
import os

from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm



import torch

import torch.nn as nn

from torch.optim import Adam

from torch.utils.data import Dataset, DataLoader

from transformers import BertModel, BertTokenizer



%matplotlib inline

tqdm.pandas()
def get_logger(

    filename="log",

    disable_stream_handler=False,

    disable_file_handler=False

):

    logger = getLogger(__name__)

    logger.setLevel(INFO)



    if not disable_stream_handler:

        handler1 = StreamHandler()

        handler1.setFormatter(Formatter("%(message)s"))

        logger.addHandler(handler1)



    if not disable_file_handler:

        handler2 = FileHandler(filename=f"{filename}.log")

        handler2.setFormatter(Formatter("%(message)s"))

        logger.addHandler(handler2)



    return logger
logger = get_logger("log")



INPUT_DIR = "../input/ailab-ml-training-2/"



train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))

test_df = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))



pretrained_weights = "bert-base-uncased"

tokenizer = BertTokenizer.from_pretrained(pretrained_weights)

backbone = BertModel.from_pretrained(pretrained_weights)



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train_df.head()
train_df = train_df.fillna({"item_description": "No description."})

test_df = test_df.fillna({"item_description": "No description."})
train_df["item_description_length"] = train_df["item_description"].progress_apply(lambda x: len(x.split()))

test_df["item_description_length"] = test_df["item_description"].progress_apply(lambda x: len(x.split()))

print(max(train_df["item_description_length"].max(), test_df["item_description_length"].max()))

sns.distplot(train_df["item_description_length"], kde=False, label="train")

sns.distplot(test_df["item_description_length"], kde=False, label="test")

plt.legend()

plt.show()
train_df["price_log1p"] = np.log1p(train_df["price"])
class MercariDataset(Dataset):

    def __init__(self, names, descriptions, prices, tokenizer, maxlen=512):

        super().__init__()

        self.names = names

        self.descriptions = descriptions

        self.prices = prices

        self.tokenizer = tokenizer

        self.maxlen = maxlen

    

    def __len__(self):

        return len(self.names)

    

    def __getitem__(self, idx):

        name = self.names[idx]

        description = self.descriptions[idx]

        

        tokens = tokenizer.encode(name, description, max_length=self.maxlen, pad_to_max_length=True)

        tokens = torch.tensor(tokens)

        attn_mask = (tokens != 0).long()

        price = torch.tensor(self.prices[idx])

        

        return tokens, attn_mask, price
class MercariBert(nn.Module):

    def __init__(self, backbone):

        super().__init__()

        self.backbone = backbone

        self.reg_fc = nn.Linear(768, 1, bias=True)

    

    def forward(self, x, mask):

        x, _ = self.backbone(x, attention_mask=mask)

        x = x[:, 0]

        x = self.reg_fc(x)

        return x
dev_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)

dev_df = dev_df.reset_index(drop=True)

val_df = val_df.reset_index(drop=True)

logger.info(f"dev: {len(dev_df)}, val: {len(val_df)}")
dev_datasets = MercariDataset(dev_df["name"], dev_df["item_description"], dev_df["price_log1p"],

                              tokenizer=tokenizer, maxlen=128)

val_datasets = MercariDataset(val_df["name"], val_df["item_description"], val_df["price_log1p"],

                              tokenizer=tokenizer, maxlen=128)



dev_dataloader = DataLoader(dev_datasets, batch_size=32, shuffle=True)

val_dataloader = DataLoader(val_datasets, batch_size=32, shuffle=False)
model = MercariBert(backbone).to(DEVICE)

optimizer = Adam(model.parameters(), lr=2e-5)

criterion = nn.MSELoss()
for epoch in tqdm(range(1)):

    dev_losses = []

    model.train()

    for i, (x, mask, y) in enumerate(tqdm(dev_dataloader, leave=False)):

        optimizer.zero_grad()

        

        x = x.to(dtype=torch.long, device=DEVICE)

        mask = mask.to(dtype=torch.long, device=DEVICE)

        y = y.to(dtype=torch.float32, device=DEVICE)

        

        y_pred = model(x, mask).view(-1)

        loss = criterion(y_pred, y)

        loss.backward()

        optimizer.step()

        

        dev_losses.append(loss.item())

        

        if (i + 1) % 200 == 0:

            logger.info("iter: {}/{}, moving loss - {:.5f}".format(

                i + 1, len(dev_dataloader), np.mean(dev_losses[-200:])

            ))

        

        if DEGUB and (i + 1) >= 400:

            break

    

    val_losses = []

    model.eval()

    for i, (x, mask, y) in enumerate(val_dataloader):

        x = x.to(dtype=torch.long, device=DEVICE)

        mask = mask.to(dtype=torch.long, device=DEVICE)

        y = y.to(dtype=torch.float32, device=DEVICE)

        

        with torch.no_grad():

            y_pred = model(x, mask).view(-1)

            loss = criterion(y_pred, y)

        

        val_losses.append(loss.item())

        

        if DEGUB and (i + 1) >= 400:

            break

    

    logger.info("epoch: {}, loss - {:.5f}, val_loss - {:.5f}".format(

        epoch + 1, np.mean(dev_losses), np.mean(val_losses)

    ))
test_datasets = MercariDataset(test_df["name"], test_df["item_description"], test_df["shipping"],

                               tokenizer=tokenizer, maxlen=128)

test_dataloader = DataLoader(test_datasets, batch_size=32, shuffle=False)



test_preds = []

model.eval()

for i, (x, mask, _) in enumerate(tqdm(test_dataloader)):

    x = x.to(dtype=torch.long, device=DEVICE)

    mask = mask.to(dtype=torch.long, device=DEVICE)

    with torch.no_grad():

        y_pred = model(x, mask).view(-1)

    test_preds.append(y_pred.cpu().numpy())

    if DEGUB and (i + 1) >= 400:

        break

test_preds = np.concatenate(test_preds, axis=0)



test_preds = np.where(test_preds < 0.0, 0.0, test_preds)

test_preds = np.exp(test_preds) - 1
sub_df = pd.DataFrame()

sub_df["test_id"] = test_df["test_id"]

sub_df["price"] = test_preds

sub_df.to_csv("submission.csv", index=False)