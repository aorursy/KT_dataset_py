from IPython.display import Image

Image("../input/nlpimages/NlpProfiler.png")
# Installing NLP Profiler



print("Instalation of NLP Profiler initiated.\n")



# !pip install git+https://github.com/neomatrix369/nlp_profiler.git@master # Version - 1



# Version 2 of NLP Profiler for better performance with large dataset.

!pip uninstall -qy typing    # some new issue on Kaggle

!pip install -U git+https://github.com/neomatrix369/nlp_profiler@scale-when-applied-to-larger-datasets

    

print("\nInstalation completed.")
# Importing libraries



import pandas as pd

from nlp_profiler.core import apply_text_profiling
# Loading dataset



df = pd.read_csv('../input/nlpprofiler/Tweets.csv')

print("Number of records: %s" % len(df))

df.head()
# Preparing datasest for profiling



df100rows=df.head(500) # Fetching out 100 rows for profiling

text_nlp = pd.DataFrame(df100rows, columns=['text']) # Preparing dataset for profiling with column 'text'

print(text_nlp.shape) # printing shape of the profiling dataset

text_nlp.head() # printing few records
# Text profiling using NLP Profiler



print("NLP profiler initiated..")

profile_data = apply_text_profiling(text_nlp, 'text')

print("NLP profiler completed!")



profile_data.head() # verifying few records of the profiled output
# Analysis of profiled data



profile_data.describe()
# Sentiment Polarity Score Analysis



print("Sentiment Polarity Score Analysis")

profile_data['sentiment_polarity_score'].hist()
# Sentiment Polarity Analysis



print("Sentiment Polarity Analysis")

profile_data['sentiment_polarity'].hist()
# Sentiment Subjectivity Summarised Analysis



print("Sentiment Subjectivity Summarised Analysis")

profile_data['sentiment_subjectivity_summarised'].hist()
# Spelling Quality Analysis



print("Spelling Quality Analysis")

profile_data['spelling_quality'].hist()
# Emoji Count Analysis



print("Emoji Count Analysis")

profile_data['emoji_count'].hist()
print("Notebook completed!")