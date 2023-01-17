import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



p = sns.color_palette()



# Import Document files for exploration

documents_categories = pd.read_csv('../input/documents_categories.csv')

documents_entities = pd.read_csv('../input/documents_entities.csv')

documents_meta = pd.read_csv('../input/documents_meta.csv')

documents_topics = pd.read_csv('../input/documents_topics.csv')



documents_categories.head()

documents_categories.head()
