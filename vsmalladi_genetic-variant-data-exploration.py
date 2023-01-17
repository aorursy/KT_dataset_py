import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()

%matplotlib inline



# Read in files

training_variants = pd.read_csv('../input/training_variants')

training_text = pd.read_csv('../input/training_text',sep="\|\|",engine='python',header=None, skiprows=1, names=["ID","Text"])

test_variants = pd.read_csv('../input/test_variants')

test_text = pd.read_csv('../input/test_text',sep="\|\|",engine='python',header=None, skiprows=1, names=["ID","Text"])

submissionFile = pd.read_csv('../input/submissionFile')
# Plot the frequncy of variant

training_variants_class = training_variants.groupby(['Class']).size()

plt.figure(figsize = (12,8))

variants_by_class = training_variants_class.plot(kind='bar',title="Total Variants by Class")

variants_by_class.set_xlabel("Class")

variants_by_class.set_ylabel("No. Variations")

plt.show()