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
from fastai.text import *
#!pip install pytorch-transformers
from pytorch_transformers import BertTokenizer,BertForSequenceClassification
bert_model = "bert-base-chinese"

max_seq_len = 128

batch_size = 32
bert_tokenizer = BertTokenizer.from_pretrained(bert_model)
#len(list(bert_tokenizer.vocab.items()))
bert_vocab = Vocab(list(bert_tokenizer.vocab.keys()))
#为fastai创建使用Bert的Tokenizer

class BertFastaiTokenizer(BaseTokenizer):

    def __init__(self, tokenizer, max_seq_len=128, **kwargs):

        self.pretrained_tokenizer = tokenizer

        self.max_seq_len = max_seq_len

    

    def __call__(self, *args, **kwargs):

        return self

    

    def tokenizer(self, t):

        return ["[CLS]"] + self.pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["[SEP]"]
tok_func = BertFastaiTokenizer(bert_tokenizer, max_seq_len=max_seq_len)
bert_fastai_tokenizer = Tokenizer(

    tok_func=tok_func,

    pre_rules = [],

    post_rules = []

)
#使用BertForSequenceClassification模型，只取结果tuple的第一个值

class MyNoTupleModel(BertForSequenceClassification):

    def forward(self, *args, **kwargs):

        return super().forward(*args, **kwargs)[0]
#BertForSequenceClassification预训练模型

bert_pretrained_model = MyNoTupleModel.from_pretrained(bert_model,num_labels=80)
!mkdir dataset

!cp -r /kaggle/input/sentiment-analysis-simplified/simplified_train.csv /kaggle/working/dataset/simplified_train.csv

!cp -r /kaggle/input/sentiment-analysis-simplified/simplified_valid.csv /kaggle/working/dataset/simplified_valid.csv
train = pd.read_csv("/kaggle/working/dataset/simplified_train.csv")

valid = pd.read_csv("/kaggle/working/dataset/simplified_valid.csv")

#test = pd.read_csv("/kaggle/input/sentiment-analysis/dataset/ai_challenger_sentiment_analysis_testa_20180816/sentiment_analysis_testa.csv")
path = Path(".")
label_cols = ['location_traffic_convenience_1',

 'location_traffic_convenience_0',

 'location_traffic_convenience_-1',

 'location_traffic_convenience_-2',

 'location_distance_from_business_district_1',

 'location_distance_from_business_district_0',

 'location_distance_from_business_district_-1',

 'location_distance_from_business_district_-2',

 'location_easy_to_find_1',

 'location_easy_to_find_0',

 'location_easy_to_find_-1',

 'location_easy_to_find_-2',

 'service_wait_time_1',

 'service_wait_time_0',

 'service_wait_time_-1',

 'service_wait_time_-2',

 'service_waiters_attitude_1',

 'service_waiters_attitude_0',

 'service_waiters_attitude_-1',

 'service_waiters_attitude_-2',

 'service_parking_convenience_1',

 'service_parking_convenience_0',

 'service_parking_convenience_-1',

 'service_parking_convenience_-2',

 'service_serving_speed_1',

 'service_serving_speed_0',

 'service_serving_speed_-1',

 'service_serving_speed_-2',

 'price_level_1',

 'price_level_0',

 'price_level_-1',

 'price_level_-2',

 'price_cost_effective_1',

 'price_cost_effective_0',

 'price_cost_effective_-1',

 'price_cost_effective_-2',

 'price_discount_1',

 'price_discount_0',

 'price_discount_-1',

 'price_discount_-2',

 'environment_decoration_1',

 'environment_decoration_0',

 'environment_decoration_-1',

 'environment_decoration_-2',

 'environment_noise_1',

 'environment_noise_0',

 'environment_noise_-1',

 'environment_noise_-2',

 'environment_space_1',

 'environment_space_0',

 'environment_space_-1',

 'environment_space_-2',

 'environment_cleaness_1',

 'environment_cleaness_0',

 'environment_cleaness_-1',

 'environment_cleaness_-2',

 'dish_portion_1',

 'dish_portion_0',

 'dish_portion_-1',

 'dish_portion_-2',

 'dish_taste_1',

 'dish_taste_0',

 'dish_taste_-1',

 'dish_taste_-2',

 'dish_look_1',

 'dish_look_0',

 'dish_look_-1',

 'dish_look_-2',

 'dish_recommendation_1',

 'dish_recommendation_0',

 'dish_recommendation_-1',

 'dish_recommendation_-2',

 'others_overall_experience_1',

 'others_overall_experience_0',

 'others_overall_experience_-1',

 'others_overall_experience_-2',

 'others_willing_to_consume_again_1',

 'others_willing_to_consume_again_0',

 'others_willing_to_consume_again_-1',

 'others_willing_to_consume_again_-2']
train.head()

#batch_size = 2
databunch = TextClasDataBunch.from_df(path, train, valid,

                                     tokenizer=bert_fastai_tokenizer,

                                     vocab=bert_vocab,

                                     include_bos=False,

                                     include_eos=False,

                                     text_cols="simplified_content",

                                     label_cols=label_cols,

                                     bs=batch_size,

                                     collate_fn=partial(pad_collate,pad_first=False,pad_idx=0)

                                     )
#databunch.save()
databunch.show_batch()
#自定义损失函数，每4个激活值作为一组（一个粒度）计算CrossEntropyLoss

class FineGritLoss(nn.CrossEntropyLoss):

    __constants__ = ['weight', 'ignore_index', 'reduction']

    

    def __init__(self, weight=None, size_average=None, ignore_index=-100,

                 reduce=None, reduction='mean'):

        super(FineGritLoss, self).__init__(weight, size_average, reduce, reduction)

        self.ignore_index = ignore_index

    

    def forward(self, input, target):

        batch_size = input.shape[0]

        grouped = input.resize(batch_size*20,4)

        grouped_target = target.resize(batch_size*20,4)

        target_argmax = grouped_target.argmax(-1)

        loss = F.cross_entropy(grouped, target_argmax, weight=self.weight,

ignore_index=self.ignore_index, reduction=self.reduction)

        return loss
#自定义accuracy，每4个激活值作为一组取argmax作为输出

def accuracy_f(input:Tensor, targs:Tensor)->Rank0Tensor:

    "Computes accuracy with `targs` when `input` is bs * n_classes."

    n = targs.shape[0]

    input = input.view(n,20,-1).argmax(dim=-1).view(n,-1)

    targs = targs.view(n,20,-1).argmax(dim=-1).view(n,-1)

    return (input==targs).float().mean()
#自定义f1 score，每4个激活值作为一组取argmax作为输出

def f1(y_pred:Tensor, y_true:Tensor, thresh:float=0.2, beta:float=1, eps:float=1e-9, sigmoid:bool=True)->Rank0Tensor:

    "Computes the f_beta between `preds` and `targets`"

    beta2 = beta ** 2

    if sigmoid: y_pred = y_pred.sigmoid()

    n = y_true.shape[0]

    max_to_zero = y_pred.view(n,20,-1).max(dim=-1).values.view(n,20,1).repeat(1,1,4)-y_pred.view(n,20,-1)

    y_pred = torch.eq(max_to_zero.cuda(), torch.zeros(n,20,4).cuda()).cuda()

    #y_pred = torch.eq(max_to_zero, torch.zeros(n,20,4))

    y_pred = y_pred.float().view(n,-1)

    

    y_true = y_true.float()

    TP = (y_pred*y_true).sum(dim=1)

    prec = TP/(y_pred.sum(dim=1)+eps)

    rec = TP/(y_true.sum(dim=1)+eps)

    res = (prec*rec)/(prec*beta2+rec+eps)*(1+beta2)

    return res.mean()
learn = Learner(databunch,

               bert_pretrained_model,

               loss_func=FineGritLoss(),

               metrics=[accuracy_f,f1])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2, 1e-3)
learn.save('first')
learn.freeze_to(-2)
learn.fit_one_cycle(1, 1e-4)
learn.freeze_to(-3)
learn.fit_one_cycle(1, 1e-4)
learn.predict('出乎意料地惊艳，椰子鸡清热降火，美容养颜，大大满足了爱吃火锅怕上火星人。椰子冻是帅帅的老板原创，不加吉利丁片，而是在专业机器发酵，只加淡奶油和椰汁，还是每天限量；鸡肉很嫩，是不吃饲料的嫩鸡；蟹柳太好吃了，日本进口，当刺身直接蘸酱油吃比在锅里煮了还好吃，可见食材的新鲜，再也不会想吃普通火锅店冰镇的蟹肉棒了；蔬菜很新鲜，老板说是在大兴有蔬菜园；煲仔饭也是高于一般水平，老板说遗憾的是商场只能用电炉，不能用传统做法。最后，安利一下老板，原来是西餐厨师，看得出是对食物有要求的人，一万个赞！！！')