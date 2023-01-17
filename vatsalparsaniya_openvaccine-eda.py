%%capture

!pip install forgi

!yes Y |conda install -c bioconda viennarna
import os, math, random

from collections import Counter



import RNA

import subprocess

from forgi.graph import bulge_graph

import forgi.visual.mplotlib as fvm

from IPython.display import Image, SVG



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from colorama import Fore, Back, Style
path = '/kaggle/input/stanford-covid-vaccine'

train_df = pd.read_json(f'{path}/train.json',lines=True)

test_df = pd.read_json(f'{path}/test.json', lines=True)

sub_df = pd.read_csv(f'{path}/sample_submission.csv')



print('Train set sequences: ', train_df.shape)

print('Test set sequences: ', test_df.shape)
train_df.head()
def data_info(_id, df):

    print(Fore.YELLOW)

    print("ID = ",_id)

    sample_data = df.loc[df['id'] == _id]

    print(Fore.MAGENTA)

    print("Secquence : \n\n",sample_data['sequence'].values[0])

    print(dict(Counter(sample_data['sequence'].values[0])))

    print("Secquence Length : ",len(sample_data['sequence'].values[0]))

    print(Fore.CYAN)

    print("Structure : \n",sample_data['structure'].values[0])

    print(dict(Counter(sample_data['structure'].values[0])))

    print("Structure Length : ",len(sample_data['structure'].values[0]))

    print(Fore.WHITE)

    print("predicted_loop_type : \n\n",sample_data['predicted_loop_type'].values[0])

    print(dict(Counter(sample_data['predicted_loop_type'].values[0])))

    print("predicted_loop_type Length : ",len(sample_data['predicted_loop_type'].values[0]))

    print(Fore.GREEN)

    print("seq_length :",sample_data['seq_length'].values)

    print("seq_scored :",sample_data['seq_scored'].values)

    print(Style.RESET_ALL)
data_info("id_001f94081",train_df)
plt.figure(figsize=(12,5))

n, bins, patches = plt.hist(x=train_df['signal_to_noise'], bins='auto', color='#0504aa', alpha=1, rwidth=0.80)

plt.grid(axis='y', alpha=0.75)

plt.xlabel('signal_to_noise')

plt.ylabel('Frequency')

plt.title('signal_to_noise Histogram')

plt.text(12, 110, f"(SN_filter == 1)  : {train_df['SN_filter'].value_counts()[0]}", fontsize=15)

plt.text(12, 85, f"(SN_filter == 0) : {train_df['SN_filter'].value_counts()[1]}", fontsize=15)

plt.show()
train_df.seq_length.value_counts()
train_df.seq_scored.value_counts()
test_df.head()
data_info("id_000ae4237",test_df)
test_df.seq_length.value_counts()
test_df.seq_scored.value_counts()
def character_count(row):

    _dictionary = {'G': 0,'A': 0, 'C': 0, 'U': 0, '.': 0, '(': 0, ')': 0, 'E': 0, 'S': 0, 'H': 0, 'B': 0, 'X': 0,'I': 0,'M':0}

    _dictionary = {**_dictionary, **dict(Counter(row['sequence']))}

    _dictionary = {**_dictionary, **dict(Counter(row['structure']))}

    _dictionary = {**_dictionary, **dict(Counter(row['predicted_loop_type']))}

    return list(_dictionary.values())
## Train-Data

feature_columns = ['G','A', 'C', 'U', '.', '(', ')', 'E', 'S', 'H', 'B', 'X','I','M']

train_df[feature_columns] = train_df.apply(character_count,axis=1,result_type="expand")
fig, _ax = plt.subplots(nrows=4,ncols=4,figsize=(20,20))

fig.suptitle("Train Data New Features Histograms", fontsize=20,)

for i,_ax in enumerate(_ax.ravel()[:14]):

    mean_value = train_df[feature_columns[i]].mean()

    max_value_index,max_value = Counter(train_df[feature_columns[i]]).most_common(1)[0]

    

    _ax.hist(x=train_df[feature_columns[i]],bins='auto', color='#0504aa', alpha=1, rwidth=1)

    _ax.set(ylabel=f"'{feature_columns[i]}' Frequency", title= f"'{feature_columns[i]}' Histogram")

    _ax.axvline(x=mean_value, color='r', label= 'Average',linewidth=2)

    _ax.axvline(x=max_value_index, color='y', label= 'Max',linewidth=2)

    _ax.legend([f"Average : {mean_value:0.2f}",f"Max Frequency : {max_value}", "Hist"], loc ="upper right")

plt.show()
# Train Data New Features correlation

corr = train_df[feature_columns].corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

plt.title("Train Data New Features correlation : ")

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
## Test-Data

feature_columns = ['G','A', 'C', 'U', '.', '(', ')', 'E', 'S', 'H', 'B', 'X','I','M']

test_df[feature_columns] = test_df.apply(character_count,axis=1,result_type="expand")
fig, _ax = plt.subplots(nrows=4,ncols=4,figsize=(20,20))

fig.suptitle("Test Data New Features Histograms", fontsize=20,)

for i,_ax in enumerate(_ax.ravel()[:14]):

    mean_value = test_df[feature_columns[i]].mean()

    max_value_index,max_value = Counter(test_df[feature_columns[i]]).most_common(1)[0]

    

    _ax.hist(x=test_df[feature_columns[i]],bins='auto', color='#0504aa', alpha=1, rwidth=1)

    _ax.set(ylabel=f"'{feature_columns[i]}' Frequency", title= f"'{feature_columns[i]}' Histogram")

    _ax.axvline(x=mean_value, color='r', label= 'Average',linewidth=2)

    _ax.axvline(x=max_value_index, color='y', label= 'Max',linewidth=2)

    _ax.legend([f"Average : {mean_value:0.2f}",f"Max Frequency : {max_value}", "Hist"], loc ="upper right")

plt.show()
# Test Data New Features correlation

corr = test_df[feature_columns].corr()

sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})

plt.figure(figsize=(13,7))

plt.title("Test Data New Features correlation : ")

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask, 1)] = True

a = sns.heatmap(corr,mask=mask, annot=True, fmt='.2f')

rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)

roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
Select_id = "id_001f94081"
Sequence = train_df[train_df['id'] == Select_id]["sequence"].values[0]

structure = train_df[train_df['id'] == Select_id]["structure"].values[0]

predicted_loop_type = train_df[train_df['id'] == Select_id]["predicted_loop_type"].values[0]

print("Sequence :",Sequence)

print("structure :",structure)

print("predicted_loop_type :",predicted_loop_type)
bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + structure)



plt.figure(figsize=(10,10))

fvm.plot_rna(bg, text_kwargs={"fontweight":"black"}, lighten=0.7,

             backbone_kwargs={"linewidth":3})

plt.show()
def render_neato(s, format='png', dpi=100):

    p = subprocess.Popen(['neato', '-T', format, '-o', '/dev/stdout', '-Gdpi={}'.format(dpi)], 

                         stdout=subprocess.PIPE, stdin=subprocess.PIPE)

    image, _ = p.communicate(bytes(s, encoding='utf-8'))

    return image
Image(render_neato(bg.to_neato_string(), dpi=60), format='png')
def get_couples(structure):

    """

    For each closing parenthesis, I find the matching opening one and store their index in the couples list.

    The assigned list is used to keep track of the assigned opening parenthesis

    """

    opened = [idx for idx, i in enumerate(structure) if i == '(']

    closed = [idx for idx, i in enumerate(structure) if i == ')']



    assert len(opened) == len(closed)





    assigned = []

    couples = []



    for close_idx in closed:

        for open_idx in opened:

            if open_idx < close_idx:

                if open_idx not in assigned:

                    candidate = open_idx

            else:

                break

        assigned.append(candidate)

        couples.append([candidate, close_idx])

        

    assert len(couples) == len(opened)

    

    return couples
def build_matrix(couples, size):

    mat = np.zeros((size, size))

    

    for i in range(size):  # neigbouring bases are linked as well

        if i < size - 1:

            mat[i, i + 1] = 1

        if i > 0:

            mat[i, i - 1] = 1

    

    for i, j in couples:

        mat[i, j] = 1

        mat[j, i] = 1

        

    return mat
couples = get_couples(structure)

mat = build_matrix(couples, len(structure))



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.35, 5))



im = axes[0].imshow(mat, interpolation='none',cmap='gray')

axes[0].set_title('Graph of the structure')

# axes[0].pcolormesh(adj_mat, )



bpp = np.load(path +f"/bpps/{Select_id}.npy")



im = axes[1].imshow(bpp, interpolation='none',cmap='gray')

axes[1].set_title('BPP Matrix')



plt.show()
Select_id = "id_000ae4237"
Sequence = test_df[test_df['id'] == Select_id]["sequence"].values[0]

structure = test_df[test_df['id'] == Select_id]["structure"].values[0]

predicted_loop_type = test_df[test_df['id'] == Select_id]["predicted_loop_type"].values[0]

print("Sequence :",Sequence)

print("structure :",structure)

print("predicted_loop_type :",predicted_loop_type)
bg, = bulge_graph.BulgeGraph.from_fasta_text('>seq\n' + Sequence + '\n' + structure)



plt.figure(figsize=(20,10))

fvm.plot_rna(bg, text_kwargs={"fontweight":"black"}, lighten=0.8,

             backbone_kwargs={"linewidth":3})

plt.show()
def render_neato(s, format='png', dpi=100):

    p = subprocess.Popen(['neato', '-T', format, '-o', '/dev/stdout', '-Gdpi={}'.format(dpi)], 

                         stdout=subprocess.PIPE, stdin=subprocess.PIPE)

    image, _ = p.communicate(bytes(s, encoding='utf-8'))

    return image
Image(render_neato(bg.to_neato_string(), dpi=60), format='png')
couples = get_couples(structure)

mat = build_matrix(couples, len(structure))



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(11.35, 5))



im = axes[0].imshow(mat, interpolation='none',cmap='gray')

axes[0].set_title('Graph of the structure')



bpp = np.load(path +f"/bpps/{Select_id}.npy")



im = axes[1].imshow(bpp, interpolation='none',cmap='gray')

axes[1].set_title('BPP Matrix')



plt.show()