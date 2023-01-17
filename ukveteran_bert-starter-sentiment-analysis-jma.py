import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
data_train= pd.read_csv("../input/yelp-review-full-csv/train.csv", header=None)
data_test= pd.read_csv("../input/yelp-review-full-csv/test.csv", header=None)
data_train.head(5)
data_train[0].unique()
data_train[0] = (data_train[0] -1)
data_test[0] = (data_test[0] -1)
data_train[0].unique()
def get_labels(self):
 return ["0", "1", "2", "3", "4"]
df_bert = pd.DataFrame({
 'id':range(len(data_train)),
 'label':data_train[0],
 'alpha':['a']*data_train.shape[0],
 'text': data_train[1].replace(r'\n', ' ', regex=True)
})
df_bert.head()
df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.01)
df_bert_test = pd.DataFrame({
 'id':range(len(data_test)),
 'text': data_test[1].replace(r'\n', '', regex=True)
})
df_bert_test.tail()