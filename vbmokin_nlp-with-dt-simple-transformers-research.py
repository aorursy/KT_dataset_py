model_type = 'distilbert'

model_name = 'distilbert-base-uncased'

with_kfold = False

weight = [0.43, 0.57]

dataset = 'DATA2'  # or 'DATA1'

n_splits = 1   # if with_kfold then must be n_splits > 1

seed = 42

model_args =  {'fp16': False,

               'train_batch_size': 4,

               'gradient_accumulation_steps': 2,

               'do_lower_case': True,

               'learning_rate': 1e-05,

               'overwrite_output_dir': True,

               'manual_seed': seed,

               'num_train_epochs': 2}
!pip install --upgrade transformers

!pip install simpletransformers
import os, re, string

import random



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns



from wordcloud import WordCloud, STOPWORDS

from collections import Counter



import numpy as np

import pandas as pd

import sklearn

from sklearn.decomposition import PCA, TruncatedSVD



import torch



from simpletransformers.classification import ClassificationModel

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split, KFold



import warnings

warnings.simplefilter('ignore')



pd.set_option('max_rows', 100)

pd.set_option('max_colwidth', 2000)
random.seed(seed)

np.random.seed(seed)

torch.manual_seed(seed)

torch.cuda.manual_seed(seed)

torch.backends.cudnn.deterministic = True
if dataset == 'DATA1':

    # Original dataset of the competition

    train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')[['text', 'target']]

    test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')[['text']]

    

elif (dataset == 'DATA2') or (dataset == 'DATA2b'):

    # Cleaned dataset from https://www.kaggle.com/vbmokin/nlp-with-disaster-tweets-cleaning-data

    train_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/train_data_cleaning.csv')[['text', 'target']]

    test_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/test_data_cleaning.csv')[['text']]



# Original dataset of the competition

sample_submission = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_data['target'].hist()
print("Weights which I offer for 0 and 1:", weight)
train_data
train_data.info()
test_data['text']
test_data.info()
def subtext_repeation_in_df(df, col, subtext, num):

    # Calc statistics as table for character repetition (1...num times) from subtext list in the df[col]

    

    text = "".join(df[col])

    result = pd.DataFrame(columns = ['subtext', 'count'])

    i = 0

    if (len(df) > 0) and (len(subtext) > 0):

        for c in subtext:

            for j in range(num):

                cs = c*(j+1)

                result.loc[i,'count'] = text.count(cs)

                if c == ' ':

                    cs = cs.replace(' ','<space>')

                result.loc[i,'subtext'] = cs                

                i += 1

    print('Number of all data is', len(df))

    result = result[result['count'] > 0].reset_index(drop=True)

    display(result.sort_values(by='subtext'))

    

    print('Text examples')

    problem_examples = pd.DataFrame(columns = ['problem_examples'])

    problem_examples['problem_examples'] = ''

    for i in range(len(result)):

        problem_examples.loc[i,'problem_examples'] = df[df[col].str.find(result.loc[i,'subtext'])>-1].reset_index(drop=True).loc[0, col]

    problem_examples = problem_examples.drop_duplicates()

    display(problem_examples)
# Analysis of punctuation marks repetition in training data

print('Statistics for punctuation marks repetition in training data')

subtext_repeation_in_df(train_data, 'text', list(string.punctuation), 10)
# Analysis of punctuation marks repetition in test data

print('Statistics for punctuation marks repetition in test data')

subtext_repeation_in_df(test_data, 'text', list(string.punctuation), 10)
# Model training without KFold

if not with_kfold:

    model = ClassificationModel(model_type, model_name, args=model_args, weight=weight) 

    model.train_model(train_data)

    result, model_outputs, wrong_predictions = model.eval_model(train_data, acc=sklearn.metrics.accuracy_score)

    y_preds, _, = model.predict(test_data['text'])

    pred_train, _ = model.predict(train_data['text'])
if not with_kfold:

    acc = result['acc']

    print('acc =',acc)
# Model training with KFold

if with_kfold:

    kf = KFold(n_splits=n_splits, random_state=seed, shuffle=True)



    results = []

    wrong_predictions = []

    y_preds = np.zeros(test_data.shape[0])

    pred_train = np.zeros(train_data.shape[0])

    

    first_fold = True

    for train_index, val_index in kf.split(train_data):

        train_df = train_data.iloc[train_index]

        val_df = train_data.iloc[val_index]



        # Model training

        model = ClassificationModel(model_type, model_name, args=model_args)

        model.train_model(train_df)



        # Validation data prediction

        result, model_outputs_fold, wrong_predictions_fold = model.eval_model(val_df, acc=sklearn.metrics.accuracy_score)

        pred_train[val_index], _ = model.predict(val_df['text'].reset_index(drop=True))

        

        # Save fold results

        if first_fold:

            model_outputs = model_outputs_fold

            first_fold = False

        else: model_outputs = np.vstack((model_outputs,model_outputs_fold))

        

        wrong_predictions += wrong_predictions_fold

        results.append(result['acc'])



        # Test data prediction

        y_pred, _ = model.predict(test_data['text'])

        y_preds += y_pred / n_splits
# Thanks to https://www.kaggle.com/szelee/simpletransformers-hyperparam-tuning-k-fold-cv

# CV accuracy result output

if with_kfold:

    for i, result in enumerate(results, 1):

        print(f"Fold-{i}: {result}")

    

    acc = np.mean(results)



    print(f"{n_splits}-fold CV accuracy result: Mean: {acc} Standard deviation:{np.std(results)}")
y_preds[:] = y_preds[:]>=0.5

y_preds = y_preds.astype(int)

np.mean(y_preds)
# Data prediction and submission

sample_submission["target"] = y_preds

sample_submission.to_csv("submission.csv", index=False)

y_preds[:20]
# Visualization of model outputs for each rows of training data

def plot_data_lavel(data, labels):

    colors = ['orange','blue']

    plt.scatter(data[:,0], data[:,1], s=8, alpha=.8, c=labels, cmap=matplotlib.colors.ListedColormap(colors))

    orange_patch = mpatches.Patch(color='orange', label='Not')

    blue_patch = mpatches.Patch(color='blue', label='Real')

    plt.legend(handles=[orange_patch, blue_patch], prop={'size': 30})



fig = plt.figure(figsize=(16, 16))          

plot_data_lavel(model_outputs, train_data['target'].values)

plt.show()
stop_words = list(STOPWORDS) + list('0123456789') + ['rt', 'amp', 'us', 'will', 'via', 'dont', 'cant', 'u', 'work', 'im',

                               'got', 'back', 'first', 'one', 'two', 'know', 'going', 'time', 'go', 'may', 'youtube', 'say', 'day', 'love', 

                               'still', 'see', 'watch', 'make', 'think', 'even', 'right', 'left', 'take', 'want', 'http', 'https', 'co']
def plot_word_cloud(x, col, num_common_words, stop_words):

    # Building the WordCloud for the num_common_words most common data in x[col] without words from list stop_words

    

    corpus = " ".join(x[col].str.lower())

    corpus = corpus.translate(str.maketrans('', '', string.punctuation))

    corpus_without_stopwords = [word for word in corpus.split() if word not in stop_words]

    common_words = Counter(corpus_without_stopwords).most_common(num_common_words)

    

    plt.figure(figsize=(12,8))

    word_cloud = WordCloud(stopwords = stop_words,

                           background_color='black',

                           max_font_size = 80

                           ).generate(" ".join(corpus_without_stopwords))

    plt.imshow(word_cloud)

    plt.axis('off')

    plt.show()

    return common_words
# Training data visualization as WordCloud

print('Word Cloud for training data without stopwords and apostrophes')

plot_word_cloud(train_data, 'text', 50, stop_words)
# Test data visualization as WordCloud

print('Word Cloud for test data without stopwords and apostrophes')

plot_word_cloud(test_data, 'text', 50, stop_words)
# Form DataFrame with outliers

outliers = pd.DataFrame(columns = ['text', 'label'])

for i in range(len(wrong_predictions)):

    outliers.loc[i, 'text'] = wrong_predictions[i].text_a

    outliers.loc[i, 'label'] = wrong_predictions[i].label
outliers
# Outliers visualization as WordCloud

print('Word Cloud for outliers without stopwords and apostrophes in the training data predictions')

outliers_top50 = plot_word_cloud(outliers, 'text', 50, stop_words)
outliers_top50
# Analysis of punctuation marks repetition in outliers

print('Statistics for punctuation marks repetition in outliers')

subtext_repeation_in_df(outliers, 'text', list(string.punctuation), 10)
# Thanks to https://www.kaggle.com/marcovasquez/basic-nlp-with-tensorflow-and-wordcloud and https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert

# Showing Confusion Matrix

def plot_cm(y_true, y_pred, title, figsize=(5,5)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    plt.title(title)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)
# Showing Confusion Matrix for ST Bert model

plot_cm(pred_train, train_data['target'].values, 'Confusion matrix for ST Bert model', figsize=(7,7))
num_outliers_per_cent = round(len(outliers)*100/len(test_data), 1)
acc_round = round(acc,3)

print('acc =', acc, '=', acc_round)
if n_splits == 1:

    n_splits_res = ""

else: n_splits_res = f"n_splits = {n_splits}, "
print(f"Model - {model_type}, {model_name}")

print(f"* {dataset} - Commit __ (LB = 0._____): lr = {model_args['learning_rate']}, {n_splits_res}num_epochs = {model_args['num_train_epochs']}, seed = {seed}, acc = {acc_round}, num_outliers = {len(outliers)}({num_outliers_per_cent}%), weight = {weight}")