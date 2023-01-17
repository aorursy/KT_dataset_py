!pip install transformers datasets --upgrade
import math

import os

import pickle

import re

from dataclasses import dataclass



import datasets

import matplotlib.pyplot as plt

import nltk

import numpy as np

import pandas as pd

import seaborn as sns

import torch

import torch.nn.functional as F

from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (accuracy_score, classification_report,

                             confusion_matrix, precision_recall_fscore_support)

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from tqdm.notebook import tqdm

from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,

                          DataCollatorWithPadding,

                          get_linear_schedule_with_warmup)



datasets.logging.set_verbosity_error()

def load_data():

    no_theme = pd.read_csv(

        '/kaggle/input/portuguese-tweets-for-sentiment-analysis/NoThemeTweets.csv', 

        index_col=0)

    # the `type` column will be important in the future to stratify the splits

    no_theme['type'] = 'no_theme-'



    with_theme = pd.read_csv(

        '/kaggle/input/portuguese-tweets-for-sentiment-analysis/TweetsWithTheme.csv', 

        index_col=0)

    with_theme['type'] = 'with_theme-'



    data = pd.concat([no_theme, with_theme])

    data['type'] = data['type'] + data['sentiment']

    # Remove duplicate tweets

    data = data[~data.index.duplicated(keep='first')]

    

    return data



data = load_data()

data
data.info()
sentiments = data.sentiment.value_counts()

print('Class ratio:', sentiments['Positivo']/sentiments['Negativo'])

sentiments
def create_splits(data):

    test_validation_size = int(0.01*data.shape[0])

    train_validation, test = train_test_split(data, test_size=test_validation_size, random_state=42, stratify=data['type'])

    train, validation = train_test_split(train_validation, test_size=test_validation_size, random_state=42, stratify=train_validation['type'])

    return train, validation, test

train, validation, test = create_splits(data)

print('Training samples:  ', train.shape[0])

print('Validation samples:', validation.shape[0])

print('Test samples:      ', test.shape[0])
def build_dataset(tokenizer, splits):

    train, validation, test = splits

    # I could create the dataset directly from pandas, but I will save and load from disk so Datasets com cache it

    # on disk. This is specially useful when you have a very large dataset that does not fit in memory, which is not

    # the case, but I will leave here this way as a demonstration. 

    train.to_csv('train_split.csv')

    validation.to_csv('validation_split.csv')

    test.to_csv('test_split.csv')

    dataset = datasets.load_dataset('csv', data_files={'train': 'train_split.csv',

                                                       'validation':'validation_split.csv',

                                                       'test': 'test_split.csv'})

    dataset = dataset.map(lambda example: {'unbiased_text': re.sub(r':[\)\(]+', '', example['tweet_text'])}, batched=False)

    dataset = dataset.map(lambda examples: tokenizer(examples['unbiased_text']), batched=True)

    dataset = dataset.map(lambda example: {'labels': 1 if example['sentiment'] == 'Positivo' else 0}, batched=False)

    dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])

    

    return dataset
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')

dataset = build_dataset(tokenizer, (train, validation, test))
def compute_metrics(preds, labels):

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    acc = accuracy_score(labels, preds)

    return {

        'accuracy': acc,

        'f1': f1,

        'precision': precision,

        'recall': recall

    }



def send_inputs_to_device(inputs, device):

    return {key:tensor.to(device) for key, tensor in inputs.items()}
train_loader = torch.utils.data.DataLoader(dataset['train'], batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))

validation_loader = torch.utils.data.DataLoader(dataset['validation'], batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))

test_loader = torch.utils.data.DataLoader(dataset['test'], batch_size=32, collate_fn=DataCollatorWithPadding(tokenizer))

num_epochs = 1

num_warmup_steps = 5000



model = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model.train().to(device)



optimizer = AdamW(model.parameters(), lr=5e-6)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_epochs*len(train_loader))
def predict(model, validation_loader, device):

    with torch.no_grad():

        model.eval()

        preds = []

        labels = []

        validation_losses = []

        for inputs in validation_loader:

            labels.append(inputs['labels'].numpy())

            

            inputs = send_inputs_to_device(inputs, device)

            loss, scores = model(**inputs)[:2]

            validation_losses.append(loss.cpu().item())



            _, classifications = torch.max(scores, 1)

            preds.append(classifications.cpu().numpy())

        model.train()

    return np.concatenate(preds), np.concatenate(labels)

        
epoch_bar = tqdm(range(num_epochs))

loss_acc = 0

alpha = 0.95

for epoch in epoch_bar:

    batch_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', total=len(train_loader))

    for idx, inputs in batch_bar:

        inputs = send_inputs_to_device(inputs, device)

        optimizer.zero_grad()

        loss, logits = model(**inputs)[:2]

        

        loss.backward()

        optimizer.step()

        

        # calculate a simplified ewma to the loss

        if epoch == 0 and idx == 0:

            loss_acc = loss.cpu().item()

        else:

            loss_acc = loss_acc * alpha + (1-alpha) * loss.cpu().item()

        

        batch_bar.set_postfix(loss=loss_acc)

        

        if idx%5000 == 0:

            preds, labels = predict(model, validation_loader, device)

            metrics = compute_metrics(preds, labels)

            print(metrics)

            



        scheduler.step()

    os.makedirs('/kaggle/working/checkpoints/epoch'+str(epoch))

    model.save_pretrained('/kaggle/working/checkpoints/epoch'+str(epoch))  
preds, labels = predict(model, test_loader, device)

metrics = compute_metrics(preds, labels)

print(metrics)
stemmer = nltk.stem.snowball.PortugueseStemmer()

analyzer = TfidfVectorizer().build_analyzer()



def stemmed_words(doc):

    return (stemmer.stem(w) for w in analyzer(doc) if w[0]!='@')



vectorizer = TfidfVectorizer(

    stop_words=nltk.corpus.stopwords.words('portuguese'), 

    analyzer=stemmed_words,

    min_df=0.0001, 

    max_features=100000, 

    max_df=0.8)



X_train = vectorizer.fit_transform(train['tweet_text'].apply(lambda s: re.sub(r':[\)\(]+', '', s)))

X_validation = vectorizer.transform(validation['tweet_text'].apply(lambda s: re.sub(r':[\)\(]+', '', s)))

X_test = vectorizer.transform(test['tweet_text'].apply(lambda s: re.sub(r':[\)\(]+', '', s)))



y_train = (train['sentiment']=='Positivo').astype(int).values

y_validation = (validation['sentiment']=='Positivo').astype(int).values

y_test = (test['sentiment']=='Positivo').astype(int).values
' | '.join(vectorizer.get_feature_names()[:5000])
lr = LogisticRegression(random_state=0, class_weight='balanced', max_iter=500, verbose=True)

lr.fit(X_train, y_train)

y_pred = lr.predict(X_validation)



print(classification_report(y_validation, y_pred))
nb = MultinomialNB()

nb.fit(X_train, y_train)

y_pred = nb.predict(X_validation)

print(classification_report(y_validation, y_pred))
imdb_dataset = datasets.load_dataset('csv', data_files={'test': '/kaggle/input/imdb-ptbr/imdb-reviews-pt-br.csv'})

imdb_dataset = imdb_dataset.map(lambda examples: tokenizer(examples['text_pt']), batched=True)

imdb_dataset = imdb_dataset.filter(lambda example: len(example['input_ids']) <= 512)

imdb_dataset = imdb_dataset.map(lambda example: {'labels': 1 if example['sentiment'] == 'pos' else 0}, batched=False)



imdb_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'])



imdb_loader = torch.utils.data.DataLoader(imdb_dataset['test'], batch_size=16, collate_fn=DataCollatorWithPadding(tokenizer))
preds, labels = predict(model, imdb_loader, device)

metrics = compute_metrics(preds, labels)

print(metrics)
ax = sns.heatmap(confusion_matrix(labels, preds), cmap='Greens_r', annot=True, fmt='d')

_ = ax.set(xlabel='Predicted', ylabel='Truth', title='Confusion Matrix')