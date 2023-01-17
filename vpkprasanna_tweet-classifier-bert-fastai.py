# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

train.head()

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
from fastai import *

from fastai.text import *

from fastai.callbacks import *



# transformers

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig



from transformers import BertForSequenceClassification, BertTokenizer, BertConfig

from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig

from transformers import XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig

from transformers import XLMForSequenceClassification, XLMTokenizer, XLMConfig

from transformers import DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig
import fastai

import transformers

print('fastai version :', fastai.__version__)

print('transformers version :', transformers.__version__)
MODEL_CLASSES = {

    'bert': (BertForSequenceClassification, BertTokenizer, BertConfig),

    'xlnet': (XLNetForSequenceClassification, XLNetTokenizer, XLNetConfig),

    'xlm': (XLMForSequenceClassification, XLMTokenizer, XLMConfig),

    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),

    'distilbert': (DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertConfig)

}
# Parameters

seed = 2020

use_fp16 = False

bs = 16



model_type = 'roberta'

pretrained_model_name = 'roberta-base'
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]

print(model_class)

print(tokenizer_class)

print(config_class)
model_class.pretrained_model_archive_map.keys()
def seed_all(seed_value):

    random.seed(seed_value) # Python

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False

seed_all(seed)
class TransformersBaseTokenizer(BaseTokenizer):

    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):

        self._pretrained_tokenizer = pretrained_tokenizer

        self.max_seq_len = pretrained_tokenizer.max_len

        self.model_type = model_type



    def __call__(self, *args, **kwargs): 

        return self



    def tokenizer(self, t:str) -> List[str]:

        """Limits the maximum sequence length and add the spesial tokens"""

        CLS = self._pretrained_tokenizer.cls_token

        SEP = self._pretrained_tokenizer.sep_token

        if self.model_type in ['roberta']:

            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]

            tokens = [CLS] + tokens + [SEP]

        else:

            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]

            if self.model_type in ['xlnet']:

                tokens = tokens + [SEP] +  [CLS]

            else:

                tokens = [CLS] + tokens + [SEP]

        return tokens
transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)

transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)

fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])
class TransformersVocab(Vocab):

    def __init__(self, tokenizer: PreTrainedTokenizer):

        super(TransformersVocab, self).__init__(itos = [])

        self.tokenizer = tokenizer

    

    def numericalize(self, t:Collection[str]) -> List[int]:

        "Convert a list of tokens `t` to their ids."

        return self.tokenizer.convert_tokens_to_ids(t)

        #return self.tokenizer.encode(t)



    def textify(self, nums:Collection[int], sep=' ') -> List[str]:

        "Convert a list of `nums` to their tokens."

        nums = np.array(nums).tolist()

        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

    

    def __getstate__(self):

        return {'itos':self.itos, 'tokenizer':self.tokenizer}



    def __setstate__(self, state:dict):

        self.itos = state['itos']

        self.tokenizer = state['tokenizer']

        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)

numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)



tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)



transformer_processor = [tokenize_processor, numericalize_processor]
pad_first = bool(model_type in ['xlnet'])

pad_idx = transformer_tokenizer.pad_token_id
databunch = (TextList.from_df(train, cols='text', processor=transformer_processor)

             .split_by_rand_pct(0.1,seed=seed)

             .label_from_df(cols= 'target')

             .add_test(test)

             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))
print('[CLS] token :', transformer_tokenizer.cls_token)

print('[SEP] token :', transformer_tokenizer.sep_token)

print('[PAD] token :', transformer_tokenizer.pad_token)

#databunch.show_batch()
print('[CLS] id :', transformer_tokenizer.cls_token_id)

print('[SEP] id :', transformer_tokenizer.sep_token_id)

print('[PAD] id :', pad_idx)

test_one_batch = databunch.one_batch()[0]

print('Batch shape : ',test_one_batch.shape)

print(test_one_batch)
class CustomTransformerModel(nn.Module):

    def __init__(self, transformer_model: PreTrainedModel):

        super(CustomTransformerModel,self).__init__()

        self.transformer = transformer_model

        

    def forward(self, input_ids, attention_mask=None):

        

        attention_mask = (input_ids!=pad_idx).type(input_ids.type()) 

        

        logits = self.transformer(input_ids,

                                  attention_mask = attention_mask)[0]   

        return logits

config = config_class.from_pretrained(pretrained_model_name)

config.num_labels = 2

#config.attention_probs_dropout_prob = .05

#config.use_bfloat16 = use_fp16

#config.max_seq_len=256

#config.hidden_dropout_prob=.05



print(config)

    
transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)

custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)
from fastai.callbacks import *

from transformers import AdamW

from functools import partial



CustomAdamW = partial(AdamW, correct_bias=False)



learner = Learner(databunch, 

                  custom_transformer_model, 

                  opt_func = CustomAdamW, 

                  metrics=[accuracy, error_rate])



# Show graph of learner stats and metrics after each epoch.

learner.callbacks.append(ShowGraph(learner))



# Put learn in FP16 precision mode. --> Seems to not working

if use_fp16: learner = learner.to_fp16()
print(learner.model)
list_layers = [learner.model.transformer.roberta.embeddings,

              learner.model.transformer.roberta.encoder.layer[0],

              learner.model.transformer.roberta.encoder.layer[1],

              learner.model.transformer.roberta.encoder.layer[2],

              learner.model.transformer.roberta.encoder.layer[3],

              learner.model.transformer.roberta.encoder.layer[4],

              learner.model.transformer.roberta.encoder.layer[5],

              learner.model.transformer.roberta.encoder.layer[6],

              learner.model.transformer.roberta.encoder.layer[7],

              learner.model.transformer.roberta.encoder.layer[8],

              learner.model.transformer.roberta.encoder.layer[9],

              learner.model.transformer.roberta.encoder.layer[10],

              learner.model.transformer.roberta.encoder.layer[11],

              learner.model.transformer.roberta.pooler]
learner.split(list_layers)

num_groups = len(learner.layer_groups)

print('Learner split in',num_groups,'groups')

print(learner.layer_groups)
learner.save('untrain')
seed_all(seed)

learner.load('untrain');
learner.freeze_to(-1)
learner.summary()
learner.lr_find()
learner.recorder.plot(skip_end=10,suggestion=True)
learner.fit_one_cycle(1,max_lr=2e-03,moms=(0.8,0.7))
learner.save('first_cycle')
seed_all(seed)

learner.load('first_cycle');
learner.predict('This is the worst game of 2020 I wont recommened')
torch.cuda.empty_cache()
def get_preds_as_nparray(ds_type) -> np.ndarray:

    """

    the get_preds method does not yield the elements in order by default

    we borrow the code from the RNNLearner to resort the elements into their correct order

    """

    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in databunch.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    return preds[reverse_sampler, :]
test_preds = get_preds_as_nparray(DatasetType.Test)
test_preds
sub = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sub['target'] = np.argmax(test_preds,axis=1)
sub.to_csv("predictions_with_v3.csv", index=False)
sub.head()