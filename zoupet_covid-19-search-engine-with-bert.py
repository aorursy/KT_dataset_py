!pip install -U sentence-transformers
import numpy as np

import pandas as pd

import scipy as sc



import os

import json

import warnings



from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer



warnings.filterwarnings("ignore")

model = SentenceTransformer('bert-base-nli-mean-tokens')
question_embedding = model.encode(['What do we know about virus genetics, origin, and evolution?'])



queries = ['What is known about transmission, incubation, and environmental stability?', 'What do we know about COVID-19 risk factors?', 

           'What do we know about virus genetics, origin, and evolution?', 'What do we know about vaccines and therapeutics?',

           'Are there geographic variations in the rate of COVID-19 spread?', 'Are there geographic variations in the mortality rate of COVID-19?',

           'Is there any evidence to suggest geographic based virus mutations?','What do we know about diagnostics and surveillance?',

           'What do we know about non-pharmaceutical interventions?','What has been published about medical care?',

           'What has been published about ethical and social science considerations?', 'What has been published about information sharing and inter-sectoral collaboration?']



query_embeddings = model.encode(queries)
count = 0

file_exts = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        count += 1

        file_ext = filename.split(".")[-1]

        file_exts.append(file_ext)



file_ext_set = set(file_exts)

file_ext_list = list(file_ext_set)



count = 0

for root, folders, filenames in os.walk('/kaggle/input'):

    print(root, folders)

    

json_folder_path = "/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset"

json_file_name = os.listdir(json_folder_path)[0]

json_path = os.path.join(json_folder_path, json_file_name)



with open(json_path) as json_file:

    json_data = json.load(json_file)

    

json_data_df = pd.io.json.json_normalize(json_data)
from tqdm import tqdm



# to process all files, uncomment the next line and comment the line below

#list_of_files = list(os.listdir(json_folder_path))

list_of_files = list(os.listdir(json_folder_path))[0:1000]

comm_use_subset_df = pd.DataFrame()



for file in tqdm(list_of_files):

    json_path = os.path.join(json_folder_path, file)

    with open(json_path) as json_file:

        json_data = json.load(json_file)

    json_data_df = pd.io.json.json_normalize(json_data)

    comm_use_subset_df = comm_use_subset_df.append(json_data_df)
comm_use_subset_df['abstract_text'] = comm_use_subset_df['abstract'].apply(lambda x: x[0]['text'] if x else "")

comm_use_subset_df['abstract_cleaned'] = comm_use_subset_df['abstract_text'].str.replace('\d+', '')

comm_use_subset_df.reset_index(drop = True, inplace = True)
abstract_embeddings = model.encode(comm_use_subset_df['abstract_cleaned'])
# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity

closest_n = 5

for query, query_embedding in zip(queries, query_embeddings):

    distances = sc.spatial.distance.cdist([query_embedding], abstract_embeddings, "cosine")[0]



    results = zip(range(len(distances)), distances)

    results = sorted(results, key=lambda x: x[1])



    print("\n\n======================\n\n")

    print("Query:", query)

    print("\nTop 5 most similar sentences in corpus:")



    for idx, distance in results[0:closest_n]:

        print(comm_use_subset_df['abstract_cleaned'][idx].strip(), "\n(Score: %.4f)" % (1-distance),"\n")
question_abstract = []

bert_scores = []



for abstract_embedding, abstract_text in zip(abstract_embeddings, comm_use_subset_df['abstract_text']):

    bert_score = cosine_similarity([question_embedding[0], abstract_embedding])[1][0]

    question_abstract.append(bert_score)

print("Index of the document: ", question_abstract.index(max(question_abstract)), "\nAbstract of the document: ", comm_use_subset_df['abstract_text'].ix[question_abstract.index(max(question_abstract))],

     "\nBert Similarity between the question and the document: ", max(question_abstract))