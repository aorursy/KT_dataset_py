from transformers import BertTokenizer, BertModel

from transformers import XLNetModel, XLNetTokenizer

from sklearn import preprocessing

from sklearn.model_selection import GroupKFold

import re

import pandas as pd
sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")

test = pd.read_csv('../input/google-quest-qa-add-on/test_v3.csv')

train = pd.read_csv("../input/google-quest-qa-add-on/train_v3.csv")
l_cols = ['question_asker_intent_understanding', 'question_body_critical',

       'question_conversational', 'question_expect_short_answer',

       'question_fact_seeking', 'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible',

       'answer_relevance', 'answer_satisfaction',

       'answer_type_instructions', 'answer_type_procedure',

       'answer_type_reason_explanation', 'answer_well_written']
test.columns.values
columns = ['question_title', 'question_cleaned', 'answer_cleaned', 'category',

       'question_asker_intent_understanding', 'question_body_critical',

       'question_conversational', 'question_expect_short_answer',

       'question_fact_seeking', 'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible',

       'answer_relevance', 'answer_satisfaction',

       'answer_type_instructions', 'answer_type_procedure',

       'answer_type_reason_explanation', 'answer_well_written']
train[train.isnull().T.any()]
def txt_re(content):

    res_list = []

    for txt in content:

        if pd.isnull(txt):

            res_list.append('code')

        else:

            txt = txt.strip()

            txt = re.sub('https?.*$', '', txt)

            txt = re.sub('https?.*\s', '', txt)

            txt = re.sub('\n+', '', txt)

            txt = re.sub('\r+', '', txt)

            res_list.append(txt)

    return res_list
content = []

labels = []

for col in zip(train[columns].values.tolist()):

    cont = col[0][:4]

    cont = txt_re(cont)

    label = col[0][4:]

    content.append(cont)

    labels.append(label)
len(labels)
test.columns.values
test_content = []

for col in zip(test[['question_title', 'question_body', 'answer', 'category']].values.tolist()):

    cont = col[0][:4]

    cont = txt_re(cont)

    test_content.append(cont)
ls
import os

os.mkdir('./t7')
tokenizer = BertTokenizer.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-tokenizer')
q_inputs = []

q_input_masks = []

q_segment_masks = []



a_inputs = []

a_input_masks = []

a_segment_masks = []

for cont in content:    

    q_input = tokenizer.encode_plus(cont[0], text_pair=cont[1],  add_special_tokens=True,max_length=512, pad_to_max_length='right')

    assert len(q_input['input_ids']) == len(q_input['token_type_ids']) == len(q_input['attention_mask'])

    q_inputs.append(q_input['input_ids'])

    q_segment_masks.append(q_input['token_type_ids'])

    q_input_masks.append(q_input['attention_mask'])

    

    a_input = tokenizer.encode_plus(cont[0], text_pair=cont[2],  add_special_tokens=True,max_length=512, pad_to_max_length='right')

    assert len(a_input['input_ids']) == len(a_input['token_type_ids']) == len(a_input['attention_mask'])

    a_inputs.append(a_input['input_ids'])

    a_segment_masks.append(a_input['token_type_ids'])

    a_input_masks.append(a_input['attention_mask'])
len(q_inputs)
t_q_inputs = []

t_q_input_masks = []

t_q_segment_masks = []



t_a_inputs = []

t_a_input_masks = []

t_a_segment_masks = []

for cont in test_content:

    t_q_input = tokenizer.encode_plus(cont[0], text_pair=cont[1],  add_special_tokens=True,max_length=512, pad_to_max_length='right')

    assert len(t_q_input['input_ids']) == len(t_q_input['token_type_ids']) == len(t_q_input['attention_mask'])

    t_q_inputs.append(t_q_input['input_ids'])

    t_q_segment_masks.append(t_q_input['token_type_ids'])

    t_q_input_masks.append(t_q_input['attention_mask'])

    

    t_a_input = tokenizer.encode_plus(cont[0], text_pair=cont[2],  add_special_tokens=True,max_length=512, pad_to_max_length='right')

    assert len(t_a_input['input_ids']) == len(t_a_input['token_type_ids']) == len(t_a_input['attention_mask'])

    t_a_inputs.append(t_a_input['input_ids'])

    t_a_segment_masks.append(t_a_input['token_type_ids'])

    t_a_input_masks.append(t_a_input['attention_mask'])
len(t_a_input_masks)
text = [[q_input, q_input_mask, q_segment_mask, a_input, a_input_mask, a_segment_mask]\

        for q_input, q_input_mask, q_segment_mask, a_input, a_input_mask, a_segment_mask in \

        zip(q_inputs, q_input_masks, q_segment_masks, a_inputs, a_input_masks, a_segment_masks)]
data = {'txt': text, 'label': labels}
t_text = [[t_q_input, t_q_input_mask, t_q_segment_mask, t_a_input, t_a_input_mask, t_a_segment_mask] \

          for t_q_input, t_q_input_mask, t_q_segment_mask, t_a_input, t_a_input_mask, t_a_segment_mask in \

          zip(t_q_inputs, t_q_input_masks, t_q_segment_masks, t_a_inputs, t_a_input_masks, t_a_segment_masks)]
t_data = {'txt': t_text}
import torch

torch.save(data, './t7/data.t7')
torch.save(t_data, './t7/test.t7')
import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from transformers import BertTokenizer, BertModel

from scipy.stats import spearmanr
class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.bert = BertModel.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-pytorch-model')

        self.dropout = nn.Dropout(0.2)

        self.pool = nn.AvgPool2d((512, 1))

        self.output = nn.Linear(768 * 2, 30) 

        

    def forward(self, q_inputs, q_input_masks, q_segment_masks, a_inputs, a_input_masks, a_segment_masks):

        q_outputs = self.bert(q_inputs, attention_mask=q_input_masks, token_type_ids=q_segment_masks)

        q_x = q_outputs[0]

        q_x = self.dropout(q_x)

        q_x = q_x.unsqueeze(1)

        q_x = self.pool(q_x)

        q_x = q_x.squeeze()

        

        a_outputs = self.bert(a_inputs, attention_mask=a_input_masks, token_type_ids=a_segment_masks)

        a_x = a_outputs[0]

        a_x = self.dropout(a_x)

        a_x = a_x.unsqueeze(1)

        a_x = self.pool(a_x)

        a_x = a_x.squeeze()

        

        x = torch.cat((q_x, a_x), -1)



        x = torch.sigmoid(self.output(x))

        return x
import numpy as np

import torch

import torch.nn as nn

from torch.autograd import Variable

from torch.utils.data import DataLoader,Dataset, TensorDataset

from sklearn.model_selection import train_test_split

from transformers import AdamW, get_linear_schedule_with_warmup
data = torch.load('./t7/data.t7')
test_data = torch.load('./t7/test.t7')
test_set = TensorDataset(torch.LongTensor(np.array(test_data['txt'])))
test_loader = DataLoader(

        test_set,

        batch_size=1,

        shuffle=False)
# optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=600, num_training_steps=6000)

# criterion = nn.BCEWithLogitsLoss()

criterion = nn.BCELoss()
def compute_spearmanr_ignore_nan(trues, preds):

    rhos = []

    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):

        rhos.append(spearmanr(tcol, pcol).correlation)

    return np.nanmean(rhos)
gkf = GroupKFold(n_splits=5).split(X=train.question_body, groups=train.question_body)

final_predicts = []

for fold, (train_idx, valid_idx) in enumerate(gkf):   

    if fold in [0, 1, 2]:

        model = Model()

        model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

        train_set = TensorDataset(torch.LongTensor(np.array(data['txt'])[train_idx]),\

                                  torch.FloatTensor(np.array(data['label'])[train_idx])) 

        dev_set = TensorDataset(torch.LongTensor(np.array(data['txt'])[valid_idx]),\

                                  torch.FloatTensor(np.array(data['label'])[valid_idx]))

        train_loader = DataLoader(

            train_set,

            batch_size=6,

            shuffle=True, drop_last=True)

        dev_loader = DataLoader(

            dev_set,

            batch_size=min(len(dev_set), 1),

            shuffle=False)

        for epoch_idx in range(3):

            for batch_idx, (model_in, labels) in enumerate(train_loader):

                model.train()

                optimizer.zero_grad()

                model_in = model_in.cuda()

                labels = labels.cuda()

                model_in = Variable(model_in, requires_grad=False)

                scores = model(model_in[:, 0], model_in[:, 1], model_in[:, 2], model_in[:, 3], model_in[:, 4], model_in[:, 5])

                labels = Variable(labels, requires_grad=False)

                labels = labels.transpose(0, 1)

                scores = scores.transpose(0, 1)

                losses = [criterion(score, label) for score, label in zip(scores, labels)]

                loss = sum(losses)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()

                # scheduler.step()

            print("train epoch: {} loss: {}".format(epoch_idx, loss.item()/30))

        # torch.save(model.state_dict(), './t7/model_{}.t7'.format(epoch_idx))

        

        torch.cuda.empty_cache()

        model.eval()

        pre_list = []

        tru_list = []

        with torch.no_grad():

            for model_in, labels in dev_loader:

                model_in = model_in.cuda()

                model_in = Variable(model_in, requires_grad=False)

                scores = model(model_in[:, 0], model_in[:, 1], model_in[:, 2], model_in[:, 3], model_in[:, 4], model_in[:, 5])

                pre_list.append(scores)

                tru_list.append(labels)

        dev_predicts = [[p.item() for p in pre] for pre in pre_list]

        truthes = [t.squeeze(0).numpy().tolist() for t in tru_list]

        dev_rho = compute_spearmanr_ignore_nan(dev_predicts, truthes)

        print("dev score: ", dev_rho)

        

        torch.cuda.empty_cache()

        model.eval()

        test_predicts = []

        with torch.no_grad():

            for model_in in test_loader:

                # model_in = model_in[0]

                model_in = model_in[0].cuda()

                model_in = Variable(model_in, requires_grad=False)

                scores = model(model_in[:, 0], model_in[:, 1], model_in[:, 2], model_in[:, 3], model_in[:, 4], model_in[:, 5])

                test_predicts.append(scores)

        final_predicts.append(test_predicts)

        torch.cuda.empty_cache()
pres = np.average(final_predicts, axis=0)
len(pres)
test_output = [[p.item() for p in pre] for pre in pres]
output_cols = ['question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible',

       'answer_relevance', 'answer_satisfaction',

       'answer_type_instructions', 'answer_type_procedure',

       'answer_type_reason_explanation', 'answer_well_written']
output_values = np.transpose(test_output).tolist()
output_dict = {k: v for k, v in zip(output_cols, output_values)}
output_dict['qa_id'] = sample_submission['qa_id'].values.tolist()
output = pd.DataFrame.from_dict(output_dict)
order = ['qa_id', 'question_asker_intent_understanding',

       'question_body_critical', 'question_conversational',

       'question_expect_short_answer', 'question_fact_seeking',

       'question_has_commonly_accepted_answer',

       'question_interestingness_others', 'question_interestingness_self',

       'question_multi_intent', 'question_not_really_a_question',

       'question_opinion_seeking', 'question_type_choice',

       'question_type_compare', 'question_type_consequence',

       'question_type_definition', 'question_type_entity',

       'question_type_instructions', 'question_type_procedure',

       'question_type_reason_explanation', 'question_type_spelling',

       'question_well_written', 'answer_helpful',

       'answer_level_of_information', 'answer_plausible',

       'answer_relevance', 'answer_satisfaction',

       'answer_type_instructions', 'answer_type_procedure',

       'answer_type_reason_explanation', 'answer_well_written']
output = output[order]
output.head()
output.to_csv('submission.csv', index=False)
import pandas as pd

test_v3 = pd.read_csv("../input/google-quest-qa-add-on/test_v3.csv")

train_v3 = pd.read_csv("../input/google-quest-qa-add-on/train_v3.csv")