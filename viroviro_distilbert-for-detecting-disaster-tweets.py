import numpy as np 

import pandas as pd 



train_data = pd.read_csv('../input/nlp-getting-started/train.csv')

print(train_data.shape)

train_data.head(3)
# load test dataset

test_data = pd.read_csv('../input/nlp-getting-started/test.csv')

print(test_data.shape)

test_data.head(3)
for tweet_index in range(1,30,5):

    print(f'Text of the tweet: {train_data["text"][tweet_index]}')

    print(f'Target: {"Real disaster" if train_data["target"][tweet_index]==1 else "Not real disaster"}\n')
import seaborn as sns



sns.countplot(train_data["target"])
X_train = train_data["text"]

y_train = train_data["target"]

X_test = test_data["text"]
from transformers import AutoConfig
pretrained_model = 'distilbert-base-uncased'

num_labels = len(set(y_train))

print(num_labels)

config = AutoConfig.from_pretrained(pretrained_model, num_labels=num_labels)
config
from transformers import AutoTokenizer



tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
tokenizer.tokenize("A long trip to Mordor")
tokenizer.encode("A long trip to Mordor")
for token_id in tokenizer.encode("A long trip to Mordor"):

    print(f'{token_id} -> {tokenizer.decode([token_id])}')

texts = [

    "A long trip to Mordor", 

    "Our mind a sea",

    "Mabuka is the end of light"

]



tokenizer.batch_encode_plus(texts, max_length=10, pad_to_max_length=True)
tokenizer.batch_encode_plus(texts, max_length=10, pad_to_max_length=True, return_tensors="pt")
encoded = tokenizer.batch_encode_plus(X_train)

lenghts = [len(x) for x in encoded["input_ids"]]
maxlength = int(np.quantile(lenghts, 0.9))

print(maxlength)
from transformers import AutoModel



distilbert = AutoModel.from_pretrained('distilbert-base-uncased')
sample = tokenizer.batch_encode_plus(X_train[0:1], max_length=40, pad_to_max_length=True, return_tensors="pt")

sample
outputs = distilbert(**sample)
embeddings = outputs[0]

print(f"Input tensor shape: {sample['input_ids'].shape}")

print(f"Input tensor values: {sample['input_ids']}")

print(f"DistilBERT embeddings shape: {embeddings.shape}")

print(f"DistilBERT embeddings values: {embeddings}")
from transformers import AutoModelForSequenceClassification

distilbert_classification = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device
# We split the training dataset for training and validation.

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=1)
train_dataset = list(zip(X_train, y_train))

eval_dataset = list(zip(X_val, y_val))

test_dataset = list(zip(X_test, np.zeros(shape=(X_test.shape[0],1))))
#from transformers import DataCollator



class TextClassificationCollator():#DataCollator 

    """Data collator for a text classification problem"""

    

    def __init__(self, tokenizer, max_length):

        """Initializes the collator with a tokenizer and a maximum document length (in tokens)"""

        self.tokenizer = tokenizer

        self.max_length = max_length

    

    def encode_texts(self, texts):

        """Transforms an iterable of texts into a dictionary of model input tensors, stored in the GPU"""

        # Tokenize and encode texts as tensors, with maximum length

        tensors = self.tokenizer.batch_encode_plus(

            texts, 

            max_length=self.max_length, 

            pad_to_max_length=True, 

            return_tensors="pt"

        )

        # Move tensors to GPU

        for key in tensors:

            tensors[key] = tensors[key].to(device)

        return tensors

    

    def collate_batch(self, patterns): #__call__

        """Collate a batch of patterns

        

        Arguments:

            - patterns: iterable of tuples in the form (text, class)

            

        Output: dictionary of torch tensors ready for model input

        """

        # Split texts and classes from the input list of tuples

        train_idx, targets = zip(*patterns)

        # Encode inputs

        input_tensors = self.encode_texts(train_idx)

        # Transform class labels to a tensor in GPU

        Y = torch.tensor(targets).long().to(device)

        # Return batch as a dictionary wikth all the inputs tensors and the labels

        batch = {**input_tensors, "labels": Y}

        return batch

    

#https://github.com/huggingface/transformers/issues/5049
collator = TextClassificationCollator(tokenizer, maxlength)
from transformers import TrainingArguments



training_args = TrainingArguments(

    output_dir="./models/model",   # Folder in which to save the trained model

    overwrite_output_dir=True,     # Whether to overwrite previous models found in the output folder

    per_gpu_train_batch_size=64,   # batch size during training

    per_gpu_eval_batch_size=128,   # batch size during evaluation (prediction)

    num_train_epochs=1,            # Model training epochs

    logging_steps=50,              # After how many training steps (batches) a log message showing progress will be printed

    save_steps=1000                # After how many training steps (batches) the model will be checkpointed to disk

)


from transformers import Trainer



trainer = Trainer(

    model=distilbert_classification,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=eval_dataset,

    data_collator=collator

)
%%time

trainer.train()
preds_val = trainer.predict(eval_dataset)

print(preds_val.predictions[0])
from scipy.special import softmax

probs_val = softmax(preds_val.predictions, axis=1)

print(probs_val[0])
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, f1_score



print("AUC score", roc_auc_score(y_val, probs_val[:,1:2], multi_class='ovr'))

y_pred_val = np.argmax(probs_val, axis=1).flatten()

y_pred_val[0:10]
print('The score of prediction: ', f1_score(y_val, y_pred_val, average = 'micro'))
print(classification_report(y_val, y_pred_val))
import matplotlib.pyplot as plt



confusion_matrix_val = confusion_matrix(y_val, y_pred_val, labels=[1,0])

# plot the confusion matrix

ax = plt.axes()

sns.heatmap(confusion_matrix_val, annot=True, fmt="d")

ax.set_title('Confusion matrix Validation set')
preds_test = trainer.predict(test_dataset)

probs_test = softmax(preds_test.predictions, axis=1)

y_pred_test = np.argmax(probs_test, axis=1).flatten()
# Copy the results to a pandas dataframe with an "id" column and a "target" column

final_submission = pd.DataFrame( data={"id":test_data["id"], "target":y_pred_test} )
final_submission.head()
# Use pandas to write the submission file

final_submission.to_csv("submissionTweets.csv", index=False)