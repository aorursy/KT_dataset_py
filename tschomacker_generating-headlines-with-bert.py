import numpy as np
import pandas as pd
df = pd.read_csv('/kaggle/input/news-headlines-summary-from-select-12-sources/full_data.csv')
columns = ['source','author', 'description', 'url', 'requested_date', 'publishedAt',]
df.drop(columns, inplace=True, axis=1)
df.head()
# drop nan values
df = df.dropna()
# resetting the index
df = df.reset_index(drop=True)
df.head()
# remove noise
for i in range(0,len(df['content'])):
    if type(df['content'][i]) == str:
        df['content'][i] = df['content'][i].replace('\n','').replace('\r','').replace('/','')
        df['title'][i] = df['title'][i].replace('\n','').replace('\r','').replace('/','')
    else:
        print(str(df['content'][i]))
df.head()
#train_data_length = int(len(df['content'])*0.95)
import sklearn.model_selection as model_selection
train_df, test_df = model_selection.train_test_split(df, train_size = 0.997)
train_data = []
for i in range(0,len(train_df['content'])):
    train_data.append((df['content'][i], df['title'][i]))
train_data[2]
!pip install headliner
!pip install tensorflow_datasets
from headliner.preprocessing.bert_preprocessor import BertPreprocessor
from spacy.lang.en import English

# use BERT-specific start and end token
preprocessor = BertPreprocessor(nlp=English())
train_prep = [preprocessor(t) for t in train_data]
targets_prep = [t[1] for t in train_prep]
from tensorflow_datasets.core.features.text import SubwordTextEncoder
from transformers import BertTokenizer
from headliner.model.bert_summarizer import BertSummarizer
from headliner.preprocessing.bert_vectorizer import BertVectorizer
from headliner.trainer import Trainer

# Use a pre-trained BERT embedding and BERT tokenizer for the encoder 
tokenizer_input = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_target = SubwordTextEncoder.build_from_corpus(
    targets_prep, target_vocab_size=2**13,  reserved_tokens=[preprocessor.start_token, preprocessor.end_token])

vectorizer = BertVectorizer(tokenizer_input, tokenizer_target)
summarizer = BertSummarizer(num_heads=2,
                            feed_forward_dim=512,
                            num_layers_encoder=0,
                            num_layers_decoder=4,
                            bert_embedding_encoder='bert-base-uncased',
                            embedding_size_encoder=768,
                            embedding_size_decoder=768,
                            dropout_rate=0.1,
                            max_prediction_len=50)
summarizer.init_model(preprocessor, vectorizer)

trainer = Trainer(batch_size=2)
trainer.train(summarizer, train_data, num_epochs=10)
test_df = test_df.reset_index(drop=True)
for i in range(0, len(test_df['content'])):
    print('t:',test_df['title'][i])
    prediction = summarizer.predict(test_df['content'][i])
    print('p:',prediction)