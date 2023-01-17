#import modules
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler as Sampler
from torch import nn
import torch.nn.functional as F
import torchtext
from gensim.models import FastText
!pip install fse
from fse.models import Average
from fse import IndexedList
from torch.utils.data import Dataset
from torchtext.data import Field
from tqdm.notebook import tqdm
import pandas as pd
import re
df = pd.read_csv("/kaggle/input/mbti-type/mbti_1.csv")

#split all sentences
new_features, new_labels = [], []
for index in tqdm(range(len(df["posts"].values))):
    person = df["posts"].iloc[index]
    for sent in person.split("|||"):
        new_features.append(sent)
        new_labels.append(df["type"].iloc[index])
        
df1 = pd.DataFrame({"feature": new_features, "label": new_labels})

#lowercase
df1.feature = df1.feature.str.lower()

#delete urls
url_pattern = re.compile("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+")
new_features_url = re.sub(url_pattern, "", "|||".join(df1.feature))

df1.feature = new_features_url.split("|||")

new_features_url2, new_labels_url2 = [], []
for index in tqdm(range(len(df1.feature))):
    if len(df1.feature.iloc[index]) > 1:
        new_features_url2.append(df1.feature.iloc[index])
        new_labels_url2.append(df1.label.iloc[index])
        
df2 = pd.DataFrame({"feature": new_features_url2, "label": new_labels_url2})

#map label values to independent binary values
def map_label(df, charlist = ["E", "N", "T", "P"]):
    labels = []
    map_char = lambda label: [(charlist[i] == label[i])*1 for i in range(len(charlist))]
    for label in df["label"].values: 
        labels.append(map_char(label))
    dfx = pd.DataFrame(labels, columns = charlist)
    return pd.concat([df.drop("label", 1), dfx], axis = 1) 

df3 = map_label(df2)

#tokenize and vectorize sentences
text_field = Field(
    tokenize='basic_english', 
    lower=True
)
label_field = Field(sequential=False, use_vocab=False)
preprocessed_text = df3['feature'].apply(lambda x: text_field.preprocess(x))

sentences = preprocessed_text.tolist()
ft = FastText(sentences, min_count=1, size=100)
vectorizer = Average(ft)
vectorizer.train(IndexedList(sentences))

tmps = [(df3["feature"].iloc[i].split(" "), i) for i in tqdm(range(len(df3.index)))]

x = vectorizer.infer(tmps)
y = df3.drop("feature", 1).values

#x,y -> dataset
class CustomDataset(Dataset):
    def __init__(self, x, y, shuffle=True):
        super().__init__()
        if(shuffle): n = np.random.permutation(len(x))
        else: n=range(len(x))
        x = torch.from_numpy(x[n])
        y = torch.from_numpy(y[n])
        self.data = list(zip(x, y))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

#whole dataset -> dataset in batches
class CustomDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)
    def __len__(self):
        return len(self.dl)
    def __getitem__(self, idx):
        return self.dl[idx]

#persist data/model in gpu memory
def get_default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def to_device(data, device):
    if(isinstance(data, (list, tuple))):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

#data -> train, validation
def split_indices(n, val_pct):
    every = np.random.permutation(n)
    return every[int(n*val_pct):], every[:int(n*val_pct)]    

#initial hyperparameters
TEST_PCT = 0.2
VAL_PCT = 0.1
DEVICE = get_default_device()
BATCH_SIZE = 20

#datasets
dataset = CustomDataset(x, y)
test_dataset = dataset[:int(len(dataset) * TEST_PCT)]
dataset = dataset[int(len(dataset) * TEST_PCT):]

#dataloaders
train_indices, val_indices = split_indices(len(dataset), VAL_PCT)
train_dl = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=Sampler(train_indices))
val_dl = DataLoader(dataset, batch_size = BATCH_SIZE, sampler = Sampler(val_indices))
train_dl = CustomDataLoader(dl=train_dl, device=DEVICE)
val_dl = CustomDataLoader(dl=val_dl, device=DEVICE)

#detector model
class Detector(nn.Module):
    def __init__(self, act_fn, end_act_fn, *layers):
        super().__init__()
        self.module_list = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.module_list.append(nn.Linear(layers[i], layers[i+1]))
        self.act_fn = act_fn
        self.end_act_fn = end_act_fn
        
    def forward(self, xb):
        for i in range(len(self.module_list)):
            xb = self.module_list[i](xb)
            if(i==len(self.module_list)-1):
                xb = self.end_act_fn(xb)
            else:
                xb = self.act_fn(xb)
        return xb
    
def loss_batch(model, xb, yb, loss_fn, opt_fn=None, metric_fn=None, verbose=True, n_class=4):
    pb = list(model(xb).reshape(n_class, -1))
    yb = list(yb.reshape(n_class, -1)*1.0)
    loss = 0
    for i in range(len(pb)):
        p, y = pb[i], yb[i]
        loss += LOSS_FN(p, y)
    if opt_fn is not None:
        opt_fn.zero_grad()
        loss.backward()
        opt_fn.step()
    num = len(xb)
    metric=None
    metric_string = ""
    if metric_fn is not None:
        metrics = []
        for i in range(len(pb)):
            p, y = pb[i], yb[i]
            p = predict(p)
            metrics.append(metric_fn(p, y))
        metric = sum(metrics)/n_class
        metric_string = f", Metric: {metric:.4f}"
    string_to_format = f"Loss: {loss:.4f}"+metric_string
    if(verbose):
        print(string_to_format)
    return loss.item(), num, metric
    
def evaluate(model, val_dl, loss_fn, metric_fn=None, verbose=True):
    with torch.no_grad():
        losses, nums, metrics = zip(*[loss_batch(model=model, xb=xb, yb=yb, loss_fn=loss_fn, metric_fn=metric_fn, verbose=False) for xb, yb in val_dl])
        total = np.sum(nums)
        loss = np.sum(np.multiply(losses, nums))/total
        metric=None
        metric_string = ""
        if metric_fn is not None:
            metric = np.sum(np.multiply(metrics, nums))/total
            metric_string = f", Metric: {metric:.4f}"
        string_to_format = f"Loss: {loss:.4f}" + metric_string
        if(verbose):
            print(string_to_format)
        return loss, total, metric
    
def fit(model, epochs, train_dl, val_dl, loss_fn, opt_fn=None, metric_fn=None, verbose=True):
    losses, metrics = [], []
    for epoch in tqdm(range(epochs)):
        for i, (xb, yb) in enumerate(tqdm(train_dl)):
            loss_batch(model=model, xb=xb, yb=yb, loss_fn=loss_fn, opt_fn=opt_fn, metric_fn=metric_fn, verbose=False)
        loss, _, metric = evaluate(model=model, val_dl=val_dl, loss_fn=loss_fn, metric_fn=metric_fn, verbose=False)
        metric_string = ""
        if metric is not None:
            metric_string = f", Metric: {metric:.4f}"
        string_to_format = f"Epoch: [{epoch+1}/{epochs}], Loss: {loss:.4f}"+metric_string
        if(verbose):
            print(string_to_format)
        losses.append(loss)
        metrics.append(metric)
    return losses, metrics

#metric=accuracy
predict = lambda pb: torch.round(pb)
def accuracy(pb, yb):
    return torch.sum(pb==yb).item()/len(pb)

#build model
LAYERS = [100, 1000, 400, 4]
ACT_FN = torch.tanh
END_ACT_FN = torch.sigmoid
MODEL = to_device(Detector(ACT_FN, END_ACT_FN, *LAYERS), device=DEVICE)

#last hyperparameters
LR = 5e-2
OPT_FN = torch.optim.SGD(list(MODEL.parameters()), lr=LR)
LOSS_FN = nn.BCELoss()
METRIC_FN = accuracy
EPOCHS = 50
VERBOSE = True
fit(model=MODEL, epochs=10, train_dl=train_dl, val_dl=val_dl, loss_fn=LOSS_FN, metric_fn=METRIC_FN, opt_fn=OPT_FN)
