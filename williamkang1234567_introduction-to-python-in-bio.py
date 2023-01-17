# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

A = np.array([3, 6, 7, 8])

B = np.array([1, 2, 3, 4])

print(A)

print(B)

C = np.zeros(6)

print(C)

D = np.concatenate((A, B), axis=0)

print(D)

print(A[3])

A[2]=3

print(A)

print(np.add(A, B))

print(np.multiply(A, B))

print(np.mean(A))
import matplotlib.pylab as plt

import numpy as np

%matplotlib inline

from sklearn.linear_model import LinearRegression

from sklearn import datasets

diabetes = datasets.load_diabetes() # load data

print(diabetes.data.shape) # feature matrix shape

print(diabetes.target.shape) # target vector shape

diabetes.feature_names # column names

# Sperate train and test data

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)

# There are three steps to model something with sklearn

# 1. Set up the model

model = LinearRegression()

# 2. Use fit

model.fit(X_train, y_train)

# 3. Check the score

model.score(X_test, y_test)

model.predict(X_test)

# plot prediction and actual data

y_pred = model.predict(X_test) 

plt.plot(y_test, y_pred, '.')



# plot a line, a perfit predict would all fall on this line

x = np.linspace(0, 330, 100)

y = x

plt.plot(x, y)

plt.show()
# BIOPYTHON EXAMPLES



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import Bio

from Bio.Seq import Seq

from Bio.Alphabet import IUPAC

from Bio.SeqUtils import GC

DNA = Seq("AGTACACTGGTT")

print(DNA)

print(DNA + " - Sequence")

print(DNA.complement() + " - Complement")

print(DNA.reverse_complement() + " - Reverse Complement")

print(len(DNA))

print("First Letter: " + DNA[0])

print("Third Letter: " + DNA[2])

print("Last Letter: " + DNA[-1])

print(DNA.count("AC"))

DNAcount = (DNA.count("A")/len(DNA))

print(DNAcount*100)

print("GC% Percent:\t" + str(GC(DNA)))

mRNA = DNA.transcribe()

print(mRNA)

protein = mRNA.translate()

print(protein)
dictionary={'UUU':'PHE','UUC':'PHE','UUA':'PHE','UUG':'PHE','UCU':'SER','UCC':

'SER','UCA':'SER','UCG':'SER','UAU':'TYR','UAC':'TYR','UAA':'STOP','UAG':'STOP',

'UGU':'CYS','UGC':'CYS','UGA':'STOP','UGG':'TRP','CUU':'LEU','CUC':'LEU','CUA':

'LEU','CUG':'LEU','CCU':'PRO','CCC':'PRO','CCA':'PRO','CCG':'PRO','CAU':'HIS',

'CAC':'HIS','CAA':'GLN','CAG':'GLN','CGU':'ARG','CGC':'ARG','CGA':'ARG','CGG':

'ARG','AUU':'ILE','AUC':'ILE','AUA':'ILE','AUG':'MET','ACU':'THR','ACC':'THR'

,'ACA':'THR','ACG':'THR','AAU':'ASN','AAC':'ASN','AAA':'LYS','AAG':'LYS','AGU'

:'SER','AGC':'SER','AGA':'ARG','AGG':'ARG','GUU':'VAL','GUC':'VAL','GUA':'VAL',

'GUG':'VAL','GCU':'ALA','GCC':'ALA','GCA':'ALA','GCG':'ALA','GAU':'ASP','GAC':

'ASP','GAA':'GLU','GAG':'GLU','GGU':'GLY','GGC':'GLY','GGA':'GLY','GGG':'GLY'}

            

print(dictionary)
def countAlphabet(strDNA): 

    dictionary={'A':0,'C':0,'T':0,'G':0}

    for x in strDNA:

        for key,value in dictionary.items():

            if x==key:

                dictionary[key]=dictionary[key]+1

    dictionary2={'A':0,'C':0,'T':0,'G':0}

    for x in strDNA:

        for key,value in dictionary.items():

            if x==key:

                dictionary2[key]=(dictionary2[key]+1/.99) # adjust .99 to however many base pairs

    return dictionary, dictionary2

strDNA = "3’-TACTCTCGTTCTTGCAGCTTGTCAGTACTTTCAGAATCATGGTGTGCATGGTAGAATGACTCTTATAACGAACTTCGACATGGCAATAACCCCCCGATT-5’"

dictionary, dictionary2 = countAlphabet(strDNA)

print(dictionary)

print(dictionary2)

strDNA = "3’-TACTCTCGTTCTTGCAGCTTGTCAGTACTTTCAGAATCATGGTGTGCATGGTAGAATGACTCTTATAACGAACTTCGACATGGCAATAACCCCCCGATT-5’"

result=""

for x in strDNA:

    if x=="T":

        x="A"

    elif x=="A":

        x="T"

    elif x=="C":

        x="G"

    elif x=="G":

        x="C"

    elif x=="3":

        x="5"

    elif x=="5":

        x="3"

    result=result+x

print(result)
strDNA = "3’-TACTCTCGTTCTTGCAGCTTGTCAGTACTTTCAGAATCATGGTGTGCATGGTAGAATGACTCTTATAACGAACTTCGACATGGCAATAACCCCCCGATT-5’"

result=""

for x in strDNA:

    if x=="T":

        x="U"

    result=result+x

print(result)
dictionary={'UUU':'PHE','UUC':'PHE','UUA':'PHE','UUG':'PHE','UCU':'SER','UCC':

'SER','UCA':'SER','UCG':'SER','UAU':'TYR','UAC':'TYR','UAA':'STOP','UAG':'STOP',

'UGU':'CYS','UGC':'CYS','UGA':'STOP','UGG':'TRP','CUU':'LEU','CUC':'LEU','CUA':

'LEU','CUG':'LEU','CCU':'PRO','CCC':'PRO','CCA':'PRO','CCG':'PRO','CAU':'HIS',

'CAC':'HIS','CAA':'GLN','CAG':'GLN','CGU':'ARG','CGC':'ARG','CGA':'ARG','CGG':

'ARG','AUU':'ILE','AUC':'ILE','AUA':'ILE','AUG':'MET','ACU':'THR','ACC':'THR'

,'ACA':'THR','ACG':'THR','AAU':'ASN','AAC':'ASN','AAA':'LYS','AAG':'LYS','AGU'

:'SER','AGC':'SER','AGA':'ARG','AGG':'ARG','GUU':'VAL','GUC':'VAL','GUA':'VAL',

'GUG':'VAL','GCU':'ALA','GCC':'ALA','GCA':'ALA','GCG':'ALA','GAU':'ASP','GAC':

'ASP','GAA':'GLU','GAG':'GLU','GGU':'GLY','GGC':'GLY','GGA':'GLY','GGG':'GLY'}

strRNA="AUGAGAGCAAGAACGUCGAACAGUCAUGAAAGUCUUAGUACCACACGUACCAUCUUACUGAGAAUAUUGCUUGAAGCUGUACCGUUAUUGGGGGGCUAA"

result=""

i = 0

while i < 97: # change length sequence - 2

    s=strRNA[i: i+3]

    for key,value in dictionary.items():

        if s == key:

            s=value

    result=result+s

    i=i+3

print(result)
# 1. write a paragraph that describes what sequence analysis is (5+ sentences)

# - what SA is used for, what kind of sequences are used, some of the things you do in SA



# 2. pick a DNA sequence perform 2 of the actions



# 3. Biopython



# 4. write another paragraph- why you use libraries and python tool (5+ sentences)