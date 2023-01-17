import sys
import pandas as pd

### import the utility function (see https://www.kaggle.com/neomatrix369/nlp-profiler-class)
# from nlp_profiler_class import NLPProfiler  

### import NLP Profiler after installing from the GitHub repo (master branch)
!pip uninstall -qy typing
!pip install git+https://github.com/neomatrix369/nlp_profiler.git@master
import nlp_profiler.core as NLPProfiler  
#To ignore warning messages
import warnings
warnings.filterwarnings('ignore')
#Pulling the dataset
df = pd.read_csv("../input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv")
df.head()
df.shape
#Importing the apply_text_profiling
from nlp_profiler.core import apply_text_profiling
profiled_df = apply_text_profiling(df,'Review')
profiled_df.head(2)
profiled_df.columns
import matplotlib.pyplot as plt
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
import seaborn as sns
sns.heatmap(profiled_df[['sentiment_polarity_score','sentiment_subjectivity_score']].corr(),annot=True,cmap='Blues')
plt.title("Correlation Between Sentiment Polarity and Sentiment Subjectivity")
plt.xticks(rotation=45)
plt.show()