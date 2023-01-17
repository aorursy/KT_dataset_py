!pip install --upgrade transformers simpletransformers

import pandas as pd, torch, warnings; warnings.simplefilter('ignore'); from simpletransformers.classification import ClassificationModel

train_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/train_data_cleaning.csv')[['text', 'target']]

test_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/test_data_cleaning.csv')[['id','text']]

model = ClassificationModel('distilbert', 'distilbert-base-uncased', args={'fp16': False,'train_batch_size': 4, 'gradient_accumulation_steps': 2,

        'learning_rate': 1e-05, 'do_lower_case': True, 'overwrite_output_dir': True, 'manual_seed': 100, 'num_train_epochs': 2}, weight = [0.44, 0.56])

model.train_model(train_data)

test_data["target"], _ = model.predict(test_data['text'])

test_data.drop(columns=['text']).to_csv("submission.csv", index=False)
# In addition - accuracy evaluation

import sklearn

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report



# Accuracy

result, model_outputs, wrong_predictions = model.eval_model(train_data, acc=sklearn.metrics.accuracy_score)

print('Accuracy = ',round(result['acc'],2),'%', sep = "")



# Showing Confusion Matrix

def plot_cm(y_true, y_pred, title, figsize=(5,5)):

    # From https://www.kaggle.com/vbmokin/nlp-eda-bag-of-words-tf-idf-glove-bert

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

    

predictions, _ = model.predict(train_data['text'])

plot_cm(predictions, train_data['target'], 'Confusion matrix for model', figsize=(7,7))



# Classification report

report = classification_report(train_data['target'],predictions)

print('Classification report:',report)