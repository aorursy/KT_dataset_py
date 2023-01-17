import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# read in our data
dataframe = pd.read_csv('../input/anonymous-survey-responses.csv')
# look at the first few rows
dataframe.head()
# Matplotlib is a little bit involved, so we need to do quite a bit 
# data manipulation before we can make our bar chart

# data preparation
# rename the columns in pet_preference
dataframe["Just for fun, do you prefer dogs or cat?"]=dataframe["Just for fun, do you prefer dogs or cat?"].replace({'Dogs ?¶':'Dogs','Both ?±?¶':'Both','Cats ?±':'Cats','Neither ?…':'Neither'})
#count how often each pet preference is observed
petFreqTable = dataframe["Just for fun, do you prefer dogs or cat?"].value_counts()
# this will get us a list of the counts
# petFreqTable.values
# this will get us a list of the names
labels = petFreqTable.index
# generate a list of numbers as long as our number of labels
postionsForBars = list(range(len(labels)))

# plot
# pass the names and counts to the bar function
plt.bar(postionsForBars,petFreqTable.values)
plt.xticks(postionsForBars,labels)
plt.title("Pet preference")
# make a barplot 
g = sns.countplot(dataframe['Just for fun, do you prefer dogs or cat?'],palette='muted')
g = g.set_title("Dogs vs Cats")