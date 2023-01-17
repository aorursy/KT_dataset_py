# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
! pip install transformers -q fastai
import csv
import pandas as pd
from pathlib import Path
import matplotlib.cm as cm
from fastai import *
from fastai.text import *
from fastai.callbacks import *
from fastai.metrics import *
import numpy as np
import pandas as pd

from pathlib import Path
from typing import *

import torch
import torch.optim as optim
seed = 42

import random
random.seed(seed)

import torch
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

import numpy as np
np.random.seed(seed)
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


config = Config(
    testing=False,
    model_name="bert-base-cased",
    max_lr=3e-5,
    epochs=4,
    use_fp16=True,
    bs=64,
    discriminative=False,
    max_seq_len=64,
)
from transformers import BertTokenizer
tok = BertTokenizer.from_pretrained(config.model_name,)
class FastAiTokenizer(BaseTokenizer):
    """Wrapper around BertTokenizer to be compatible with fast.ai"""
    def __init__(self, tokenizer: BertTokenizer, max_seq_len: int=128, **kwargs):
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, *args, **kwargs):
        return self

    def tokenizer(self, t:str) -> List[str]:
        """Limits the maximum sequence length"""
        return ["[CLS]"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]
fastai_tokenizer = Tokenizer(tok_func=FastAiTokenizer(tok, max_seq_len=config.max_seq_len), pre_rules=[], post_rules=[])
fastai_vocab = Vocab(list(tok.vocab.keys()))
train = pd.read_csv("/kaggle/input/traincsv/train.csv")
train.head()
test = pd.read_csv("/kaggle/input/testcsv/test.csv")
test.head()
from nltk.tokenize import TweetTokenizer
twt = TweetTokenizer(strip_handles=True)
def tweets(r):
    s = ' '.join(twt.tokenize(r['text']))
    s = re.sub(r'http\S+', '', s)
    s = re.sub(r'https\S+', '', s)    
    return s
train['ptext'] = train.apply(tweets, axis=1)
test['ptext'] = test.apply(tweets, axis=1)
label_cols = list(set(train['target']))
class AutoTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)

class AutoNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=Vocab(list(bert_tok.vocab.keys())), **kwargs)

def get_auto_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    """
    Constructing preprocessors for BERT
    We remove sos/eos tokens since we add that ourselves in the tokenizer.
    We also use a custom vocabulary to match the numericalization with the original BERT model.
    """
    return [AutoTokenizeProcessor(tokenizer=tokenizer),
            NumericalizeProcessor(vocab=vocab)]
class AutoDataBunch(TextDataBunch):
    @classmethod
    def from_df(cls, path:PathOrStr, train_df:DataFrame, valid_df:DataFrame, test_df:Optional[DataFrame]=None,
                tokenizer:Tokenizer=None, vocab:Vocab=None, classes:Collection[str]=None, text_cols:IntsOrStrs=1,
                label_cols:IntsOrStrs=0, label_delim:str=None, **kwargs) -> DataBunch:
        "Create a `TextDataBunch` from DataFrames."
        p_kwargs, kwargs = split_kwargs_by_func(kwargs, get_auto_processor)
        # use our custom processors while taking tokenizer and vocab as kwargs
        processor = get_auto_processor(tokenizer=tokenizer, vocab=vocab, **p_kwargs)
        if classes is None and is_listy(label_cols) and len(label_cols) > 1: 
          classes = label_cols
        src = ItemLists(path, TextList.from_df(train_df, path, cols=text_cols, processor=processor), TextList.from_df(valid_df, path, cols=text_cols, processor=processor))

        src = src.label_for_lm() if cls==TextLMDataBunch else src.label_from_df(cols=label_cols, classes=classes)

        if test_df is not None: 
          src.add_test(TextList.from_df(test_df, path, cols=text_cols))
        return src.databunch(**kwargs)
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(train,test_size = 0.3)
databunch = AutoDataBunch.from_df(".", 
                    train_df = X_train, 
                    valid_df = X_val,
                    test_df = test,
                  tokenizer=fastai_tokenizer,
                  vocab=fastai_vocab,
                  text_cols="ptext",
                  label_cols='target',
                  bs=config.bs,
                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),
             )

import torch
def softmax(v):
  e = torch.exp(v)
  s = torch.sum(e, dim=0)
  return e/s
from transformers import BertPreTrainedModel, BertModel
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return softmax(outputs[0])  # (loss), logits, (hidden_states), (attentions)
auto_model = BertForSequenceClassification.from_pretrained(config.model_name, num_labels=len(set(train['target'])))
learner = Learner(
    databunch, auto_model,
    metrics=[accuracy,error_rate]
)
learner.callbacks.append(ShowGraph(learner))

learner.lr_find()
learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(2, slice(1e-4))
learner.save('bert')
learner.unfreeze()
learner.lr_find()
learner.recorder.plot(suggestion=True)
learner.fit_one_cycle(3, slice(1e-6))
def get_preds_as_nparray(ds_type) -> np.ndarray:
    """
    the get_preds method does not yield the elements in order by default
    we borrow the code from the RNNLearner to resort the elements into their correct order
    """
    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()
    sampler = [i for i in databunch.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    return preds[reverse_sampler, :]
val_preds = get_preds_as_nparray(DatasetType.Valid)
test_preds = get_preds_as_nparray(DatasetType.Test)
preds = []

vpreds = []
for i in val_preds:
    vpreds.append(np.argmax(i))
for i in test_preds:
    preds.append(np.argmax(i))
from sklearn.metrics import classification_report, f1_score , recall_score, precision_score
print(classification_report(X_val['target'], vpreds))