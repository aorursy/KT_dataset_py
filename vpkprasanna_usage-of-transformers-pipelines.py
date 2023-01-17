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
# from transformers import pipeline



# # Using default model and tokenizer for the task

# pipeline("<task-name>")



# # Using a user-specified model

# pipeline("<task-name>", model="<model_name>")



# # Using custom model/tokenizer as str

# pipeline('<task-name>', model='<model name>', tokenizer='<tokenizer_name>')
tasks = ['feature-extraction', 'sentiment-analysis', 'ner', 'question-answering', 'fill-mask', 'summarization', 'translation_en_to_fr', 'translation_en_to_de', 'translation_en_to_ro', 'text-generation']

print(tasks)
import transformers

from transformers import pipeline

nlp_sentence_classif = pipeline('sentiment-analysis')

nlp_sentence_classif('Such a nice weather outside !')




nlp_token_class = pipeline('ner',tokenizer=transformers.PreTrainedTokenizer('bert-based-uncased'))

nlp_token_class('Hugging Face is a French company based in New-York.')



nlp_qa = pipeline('question-answering')

nlp_qa(context='Hugging Face is a French company based in New-York.', question='Where is based Hugging Face ?')
nlp_fill = pipeline('fill-mask')

nlp_fill('Hugging Face is a French company based in <mask>')
nlp_features = pipeline('feature-extraction')

output = nlp_features('Hugging Face is a French company based in Paris')

np.array(output).shape   # (Samples, Tokens, Vector Size)
# use bart in pytorch

summarizer1 = pipeline("summarization")

summarizer1("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)



# use t5 in tf

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base", framework="tf")

summarizer("Sam Shleifer writes the best docstring examples in the whole world.", min_length=5, max_length=20)