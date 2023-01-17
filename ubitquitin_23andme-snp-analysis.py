# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
genome_desc = pd.read_csv('../input/genome_file_description.csv')

genome = pd.read_csv('../input/genome_zeeshan_usmani.csv')
genome.head()
genome["chromosome"].value_counts()
chromosomes = {k: v for k, v in genome.groupby('chromosome')}
chromosomes.keys()
for i in chromosomes:

    chromosomes[i].sort_values(by='position')
print(chromosomes[1][0:10])

print(chromosomes['X'][0:10])
chromosomes.keys()
genome.loc[genome['chromosome'] == 'Y'].head()
genome.loc[genome['# rsid'].isin(['rs4778241','rs12913832','rs7495174', 'rs8028689', 'rs7183877', 'rs1800401'])] #genoset 237 for eye color - SNPedia. 

genome.loc[genome['# rsid'] == 'rs1799971']
genome.loc[genome['# rsid'] == 'rs1333049']
genome.loc[genome['# rsid'] == 'rs7412']

genome.loc[genome['# rsid'] == 'rs429358']