# Followining:

# http://mailchi.mp/5f0a34899a89/data-challenge-day-1-read-in-and-summarize-a-csv-file-2576429
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/anonymous-survey-responses.csv')
dataset.head()
frequency_table = dataset['Just for fun, do you prefer dogs or cat?'].value_counts()
plt.bar(frequency_table.index, frequency_table.values)
sns.barplot(frequency_table.index, frequency_table.values)
sns.countplot(dataset['Just for fun, do you prefer dogs or cat?'])