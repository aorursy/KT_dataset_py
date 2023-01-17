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
# Set your own project id here

PROJECT_ID = 'your-google-cloud-project'

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)

from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)

from google.cloud import automl_v1beta1 as automl

automl_client = automl.AutoMlClient()
import torch

from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "What is the greatest"
model = GPT2LMHeadModel.from_pretrained('gpt2')
def token_tensor(text, model):

    

    indexed_tokens = tokenizer.encode(text)

    tokens_tensor = torch.tensor([indexed_tokens])

    model.eval()

    

    tokens_tensor = tokens_tensor.to('cuda')

    model.to('cuda')

    

    with torch.no_grad():

        outputs = model(tokens_tensor)

        predictions = outputs[0]

        

    predicted_index = torch.argmax(predictions[0, -1, :]).item()

    predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])

     

    return predicted_text
a = []



for i in range(10):

    

    a.append(token_tensor(text, model))

    

    text = a[i]

    

    
print(a[9].strip())