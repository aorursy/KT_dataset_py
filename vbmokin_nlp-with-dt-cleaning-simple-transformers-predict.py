!pip install --upgrade transformers

!pip install simpletransformers
import numpy as np

import pandas as pd

import sklearn



import torch

from simpletransformers.classification import ClassificationModel



import warnings

warnings.simplefilter('ignore')
train_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/train_data_cleaning.csv')[['text', 'target']]

test_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/test_data_cleaning.csv')[['text']]
train_data
test_data
model_type = 'distilbert'

model_name = 'distilbert-base-uncased'

seed = 100

model_args =  {'fp16': False,

               'train_batch_size': 4,

               'gradient_accumulation_steps': 2,

               'do_lower_case': True,

               'learning_rate': 1e-05,

               'overwrite_output_dir': True,

               'manual_seed': seed,

               'num_train_epochs': 2}
%%time

# Training model and prediction

model = ClassificationModel(model_type, model_name, args=model_args) 

model.train_model(train_data)

result, model_outputs, wrong_predictions = model.eval_model(train_data, acc=sklearn.metrics.accuracy_score)

y_preds, _, = model.predict(test_data['text'])
# Training model accuracy

print('accuracy =',result['acc'])
# Predicted data

y_preds[:20]
# Submission predicted data

test_data["target"] = y_preds

test_data.to_csv("submission.csv", index=False)