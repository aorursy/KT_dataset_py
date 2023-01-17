import joblib
import torch
import torch.nn as nn
import transformers
import os
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn import model_selection

from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
! git clone https://github.com/pranav-ust/BERT-keyphrase-extraction.git
class config:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    EPOCHS = 3
    BASE_MODEL_PATH = "bert-base-uncased"
    MODEL_PATH = "model.bin"
    TOKENIZER = transformers.BertTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        do_lower_case=True
    )
class EntityDataset:
    def __init__(self, texts, tags):
        self.texts = texts
        self.tags = tags
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]

        ids = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            # abhishek: ab ##hi ##sh ##ek
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:config.MAX_LEN - 2]
        target_tag = target_tag[:config.MAX_LEN - 2]

        ids = [101] + ids + [102]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.MAX_LEN - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
        }
def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():
            data[k] = v.to(device)
        _,loss = model(**data)
        final_loss += loss.item()
    return final_loss / len(data_loader)
def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss
class EntityModel(nn.Module):
    def __init__(self, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.bert = transformers.BertModel.from_pretrained(
            config.BASE_MODEL_PATH
        )
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
    
    def forward(
        self, 
        ids, 
        mask, 
        token_type_ids,  
        target_tag
    ):
        o1, _ = self.bert(
            ids, 
            attention_mask=mask, 
            token_type_ids=token_type_ids
        )

        bo_tag = self.bert_drop_1(o1)

        tag = self.out_tag(bo_tag)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)

        loss = loss_tag
        return tag, loss
! pip install pytorch_pretrained_bert

from pytorch_pretrained_bert import BertTokenizer
bert_model_dir='bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_dir, do_lower_case=True)

def data_preprocess(data_path,data_type):
    sentences=[]
    sentence_word=[]
    tags=[]
    
    sentences_file=os.path.join(data_path,data_type,'sentences.txt')
    tag_file=os.path.join(data_path,data_type,'tags.txt')
    
    with open(sentences_file, 'r') as file:
        for line in file:
            # replace each token by its index
            
            tokens = line.strip().split()
#             print(len(tokens))
            input_len=len(tokens)
            sentences.extend([line]*input_len)
            sentence_word.extend(tokens)
            
    with open(tag_file, 'r') as file:
        for line in file:
            
            tag=line.strip().split()
            tags.extend(tag)
    final_data=pd.DataFrame({'Sentence':sentences,'Sentence_words':sentence_word,'tag':tags,})
    
    enc_tag = preprocessing.LabelEncoder()

    final_data.loc[:, "tag"] = enc_tag.fit_transform(final_data["tag"])
    sentences = final_data.groupby("Sentence")["Sentence_words"].apply(list).values
    
    tag = final_data.groupby("Sentence")["tag"].apply(list).values
    return sentences,enc_tag,tag

sentences_file='./BERT-keyphrase-extraction/data/task1/train/sentences.txt'
sentences = []
tags = []

with open(sentences_file, 'r') as file:
    for line in file:
        # replace each token by its index
        tokens = line.strip().split()
        print(tokens)
        s=config.TOKENIZER.convert_tokens_to_ids(tokens)
#         print(s)
        sentences.append(s)
#         print(tokens)

print(len(sentences))

print(sentences[0])

data_path='./BERT-keyphrase-extraction/data/task1'
data_type='train'
sentences,enc_tag,tag=data_preprocess(data_path,data_type)

# sentences=data_preprocess(data_path,data_type)
print(len(sentences))
print(len(tag))
meta_data = {
    "enc_tag": enc_tag
}


num_tag = len(list(enc_tag.classes_))


(
    train_sentences,
    test_sentences,
    train_tag,
    test_tag
) = model_selection.train_test_split(
    sentences, 
    tag, 
    random_state=42, 
    test_size=0.1
)

train_dataset = EntityDataset(
    texts=train_sentences, tags=train_tag
)

train_data_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
)

valid_dataset = EntityDataset(
    texts=test_sentences, tags=test_tag
)

valid_data_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
)
device = torch.device("cuda")
model = EntityModel(num_tag=num_tag)
model.to(device)

param_optimizer = list(model.named_parameters())
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
optimizer_parameters = [
    {
        "params": [
            p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay
            )
        ],
        "weight_decay": 0.001,
    },
    {
        "params": [
            p for n, p in param_optimizer if any(
                nd in n for nd in no_decay
            )
        ],
        "weight_decay": 0.0,
    },
]

num_train_steps = int(
    len(train_sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS
)
optimizer = AdamW(optimizer_parameters, lr=3e-5)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=0, 
    num_training_steps=num_train_steps
)

best_loss = np.inf
for epoch in range(config.EPOCHS):
    train_loss = train_fn(
        train_data_loader, 
        model, 
        optimizer, 
        device, 
        scheduler
    )
    test_loss = eval_fn(
        valid_data_loader,
        model,
        device
    )
    print(f"Train Loss = {train_loss} Valid Loss = {test_loss}")
    if test_loss < best_loss:
        torch.save(model.state_dict(), config.MODEL_PATH)
        best_loss = test_loss
# enc_tag = meta_data["enc_tag"]
# enc_tag
### Testing model on a sample_txt

# meta_data = joblib.load("meta.bin")
enc_tag = meta_data["enc_tag"]


num_tag = len(list(enc_tag.classes_))

sentence = """

complex lange ##vin ( cl ) [ 1 , 2 ] sign problem numerical simulations of lattice field theories weight , sampling . nonzero chemical potential , lower and four - dimensional field theories sign problem in the thermodynamic limit [ 3 – 8 ] ( for reviews , e . g . refs . [ 9 , 10 ] ) . however , inc ##epti ##on , [ 11 – 16 ] . improved understanding , relying on the combination of analytical and numerical insight . past , probability distribution complex ##ified configuration space , lange ##vin process , [ 17 , 18 ] . distribution local ##ised cl results . importantly , non ##abel ##ian gauge theories , sl ( n , c ) gauge cooling [ 8 , 10 ] .
nuclear theory thermal ##ization nuclear reactions , semi - classical methods [ 13 , 14 , 10 ] , quantum liquids [ 15 , 16 ] . improved molecular dynamics methods combining quantum features semi classical treatment of dynamical correlations [ 17 , 18 ] . still , clear - cut quantum approach yet , [ 19 , 20 , 10 ] . field of clusters and nano structures lasers imaging techniques . semic ##lass ##ical [ 21 , 22 ] qualitatively describe dynamical processes . simple metals with sufficiently del ##ocal ##ized wave functions , justify ##ing semic ##lass ##ical approximations . organic systems , celebr ##ated c ##60 [ 4 , 23 ] , way . classical , approaches , very intense laser pulses [ 2 ] . blow ##n quantum mechanical features anym ##ore . scenarios , quantum shell effects ignored .
dirac equation . the cre ##utz model [ 32 ] treatment , objects hopping on a lattice instead of particles moving in a space - time continuum . 
"""
tokenized_sentence = config.TOKENIZER.encode(sentence)

sentence = sentence.split()
print(sentence)
print(tokenized_sentence)

test_dataset = EntityDataset(
    texts=[sentence], 
    tags=[[0] * len(sentence)]
)

device = torch.device("cuda")
model = EntityModel(num_tag=num_tag)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)

with torch.no_grad():
    data = test_dataset[0]
    for k, v in data.items():
        data[k] = v.to(device).unsqueeze(0)
    tag, _ = model(**data)

    print(
        enc_tag.inverse_transform(
            tag.argmax(2).cpu().numpy().reshape(-1)
        )[:len(tokenized_sentence)]
    )