# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from Bio.Seq import Seq

from Bio.Alphabet import IUPAC
with open ("/kaggle/input/covid19-genomes/MN908947.txt", "r") as myfile:

    MN908947=myfile.read().replace('\n', '').replace('\ufeff', '')
MN908947
my_seq = MN908947

my_seq.count("A")
my_seq = MN908947

my_seq.count("C")
my_seq = MN908947

my_seq.count("T")
from Bio.SeqUtils import GC

GC(my_seq)
from Bio.SeqUtils import molecular_weight

molecular_weight(my_seq)
from Bio import SeqIO

import pylab
len(my_seq)
max(my_seq) 
min(my_seq)