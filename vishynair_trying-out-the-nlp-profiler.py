#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import time
#To ignore warning messages
import warnings
warnings.filterwarnings('ignore')
#Pulling the dataset
df = pd.read_csv("/kaggle/input/spooky-author-identification/train.zip")
#To display the full text column instead of truncating one
pd.set_option('display.max_colwidth', -1)
#To display a maximum of 100 columns
pd.set_option('display.max_columns',100)
df.head()
df.shape
#Retaining just first 100 records
df = df[:100]
#Python provides a constant called string.punctuation that provides a great list of punctuation characters. 
print(string.punctuation)
def remove_punctuations(input_col):
    """To remove all the punctuations present in the text.Input the text column"""
    table = str.maketrans('','',string.punctuation)
    return input_col.translate(table)
#Applying the remove_punctuation function
df['text'] = df['text'].apply(remove_punctuations)
!pip install git+https://github.com/neomatrix369/nlp_profiler.git@master
#Importing the apply_text_profiling
from nlp_profiler.core import apply_text_profiling
#Applying on the text column of the dataframe
#Official git mentions Pandas dataframe series as input param to be passed
start = time.time()
profiled_df = apply_text_profiling(df,'text')
end = time.time()
total_time = end - start / 60*60
print("Time taken(in secs) for the apply_text_profiling to run on 100 records: ",total_time)
profiled_df.head(2)
profiled_df.columns
#Hist plot for the sentiment polarity for the first 100 sentences
profiled_df['sentiment_polarity'].hist()
plt.title("Sentiment Polarity")
plt.show()
#Subjective or Objective sentence
profiled_df['sentiment_subjectivity_summarised'].hist()
plt.title("Sentiment Subjectivity")
plt.show()
#Histogram on the words_count
profiled_df['words_count'].hist()
plt.title("Word Count Distribution with NLP_Profiler")
plt.show()
#Average stop word count with the sentences
profiled_df['stop_words_count'].mean()
sns.heatmap(profiled_df[['sentiment_polarity_score','sentiment_subjectivity_score']].corr(),annot=True,cmap='Blues')
plt.title("Correlation Between Sentiment Polarity and Sentiment Subjectivity")
plt.xticks(rotation=45)
plt.show()