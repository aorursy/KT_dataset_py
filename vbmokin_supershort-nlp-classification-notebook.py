!pip install --upgrade transformers simpletransformers

import pandas as pd, torch, warnings; warnings.simplefilter('ignore'); from simpletransformers.classification import ClassificationModel

train_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/train_data_cleaning.csv')[['text', 'target']]

test_data = pd.read_csv('../input/nlp-with-disaster-tweets-cleaning-data/test_data_cleaning.csv')[['id', 'text']]

model = ClassificationModel('distilbert', 'distilbert-base-uncased', args={'fp16': False,'train_batch_size': 4, 'gradient_accumulation_steps': 2,

        'learning_rate': 4e-05, 'do_lower_case': True, 'overwrite_output_dir': True, 'manual_seed': 42, 'num_train_epochs': 1}, weight = [0.44, 0.56])

model.train_model(train_data)

test_data["target"], _ = model.predict(test_data['text'])

test_data.drop(columns=['text']).to_csv("submission.csv", index=False)
# In addition - accuracy evaluation

# Accuracy

import sklearn

result, model_outputs, wrong_predictions = model.eval_model(train_data, acc=sklearn.metrics.accuracy_score)

print('Accuracy = ',round(result['acc'],2),'%', sep = "")



# Confusion_matrix, Accuracy_score, Classification_report

from sklearn.metrics import confusion_matrix,accuracy_score,classification_report

predictions, _ = model.predict(train_data['text'])

matrix = confusion_matrix(train_data["target"],predictions)

print(matrix)



score = accuracy_score(train_data["target"],predictions)

print(score)



report = classification_report(train_data['target'],predictions)

print(report)