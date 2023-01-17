# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline
pd.options.display.max_columns = 999
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Read input datasets:
kiva_loans = pd.read_csv("../input/kiva_loans.csv")
kiva_mpi_reg_locs = pd.read_csv("../input/kiva_mpi_region_locations.csv")
loan_theme_ids = pd.read_csv("../input/loan_theme_ids.csv")
loan_themes_by_reg = pd.read_csv("../input/loan_themes_by_region.csv")
# Lets explore each dataset.
kiva_loans.head()
kiva_loans = kiva_loans.fillna('')
#Lets see how is the loan fund distributed across Countries.
Kiva_loans_group_by_country=kiva_loans.groupby('country')['loan_amount'].sum().reset_index()
Kiva_loans_group_by_country=Kiva_loans_group_by_country.sort_values('loan_amount').reset_index()
Kiva_loans_group_by_country.head()
plt.figure(figsize=(12,20))
ax = sns.barplot(y="country", x="loan_amount", data=Kiva_loans_group_by_country)
#Lets chekc on Sector and activity wise.
kiva_loans['sector'].unique()
kiva_loans['activity'].unique()
#Lets see how is the loan fund distributed across Sectors.
Kiva_loans_group_by_sector=kiva_loans.groupby('sector')['loan_amount'].sum().reset_index()
Kiva_loans_group_by_sector=Kiva_loans_group_by_sector.sort_values('loan_amount').reset_index()
plt.figure(figsize=(12,12))
ax = sns.barplot(y="sector", x="loan_amount", data=Kiva_loans_group_by_sector)
#Lets see gender distributions.
kiva_loans['borrower_genders'].unique()
#Wordcloud for Use Text
from wordcloud import WordCloud, STOPWORDS
Alltext= (' '.join(kiva_loans['use']))
wc = WordCloud(width = 1000, height = 500,stopwords=STOPWORDS).generate(Alltext)

plt.figure(figsize=(15,5));
plt.imshow(wc);
plt.axis('off');
plt.title('Word Cloud for Use text');
#Wordcloud for Use Text
from wordcloud import WordCloud, STOPWORDS
Alltext= (' '.join(kiva_loans['borrower_genders']))
wc = WordCloud(width = 1000, height = 500,stopwords=STOPWORDS).generate(Alltext)

plt.figure(figsize=(15,5));
plt.imshow(wc);
plt.axis('off');
plt.title('Word Cloud for Use text');

















