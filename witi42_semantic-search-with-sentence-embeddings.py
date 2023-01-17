!pip install faiss-cpu
import json

import glob

import numpy as np

import tensorflow_hub as hub

import faiss

import nltk

from nltk.tokenize import sent_tokenize

nltk.download('punkt')

from IPython.display import clear_output



#own files

from index_file_map import IndexFileMap
# get all papers

def get_cord_papers():

    folders = ['../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json',

               '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pdf_json',

               '../input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/pmc_json',

               '../input/CORD-19-research-challenge/custom_license/custom_license/pdf_json',

               '../input/CORD-19-research-challenge/custom_license/custom_license/pmc_json',

               '../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pdf_json',

               '../input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/pmc_json']

    papers = []

    for folder in folders:

        path = folder + '/*.json'

        papers.extend(glob.glob(path))

    return papers



papers = get_cord_papers()
# extract text from a paper

def text_from_cord_json(paper):

    dic = json.load(open(paper, "r"))

    doc = []

    if "body_text" not in dic:

        print("empty paper")

        return []

    for x in dic["body_text"]:

        doc.append(x["text"])

    return doc
# split text into sentences

def tokenize(text):

    return sent_tokenize(text)
# download universal-sentence-encoder

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# What has been published about medical care?

queries = ["alternative methods to advise on disease management",

           "clinical characterization and management of the virus",

           

           "Resources to support skilled nursing facilities and long term care facilities", 

           "Mobilization of surge medical staff to address shortages in overwhelmed communities",

           "Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure",

           "Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients",

           "Outcomes data for COVID-19 after mechanical ventilation adjusted for age.",

           "Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including possible cardiomyopathy and cardiac arrest.",

           "Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.",

           "Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.",

           "Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.",

           "Guidance on the simple things people can do at home to take care of sick people and manage disease.",

           "Oral medications that might potentially work.",

           "Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.",

           "Best practices and critical challenges and innovative solutions and technologies.",

           "Efforts to define the natural history of disease",

           "Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials",

           "Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients"

          ]
# search files for similar sentences to query sentences

def search(files, queries, num_results):

    # map the index of a results to its papers

    index_map = IndexFileMap()

    # map the index of a results to the section within a paper

    section_map = IndexFileMap()

    count = 0

    sentences = []

    for i in range(len(files)):

        doc = text_from_cord_json(files[i])

        if len(doc) is 0:

            continue

        for j in range(len(doc)):

            new_sentences = tokenize(doc[j])

            sentences.extend(new_sentences)

            section_map.add(count, j)

            index_map.add(count, files[i])

            count += len(new_sentences)

    

    # map sentences to vectors

    vectors = embed(np.array(sentences)).numpy()

    search_vectors = embed(np.array(queries)).numpy()

    

    # add vectors to search index

    search_index = faiss.IndexFlatL2(512)

    search_index.add(vectors)

    

    D, I = search_index.search(search_vectors, num_results)  

    results = []

    for i in range(len(D)):

        result = []

        for j in range(num_results):

            dist = D[i][j]

            index = I[i][j]

            start_index, path, list_index = index_map.get(index)

            _, section, _ = section_map.get(index)

            # append distance, sentence, path, and section of a query to the results

            result.append( (dist, sentences[index], path, section) )    

        results.append(result)

    del vectors, sentences, search_index

    

    return results
results = []



batch_size = 16

num_results = 5



results = search(papers[0:batch_size], queries, num_results)



for i in range(batch_size, len(papers), batch_size): #use len(papers) instead

    end = min(i+batch_size, len(papers))

    

    clear_output(wait=True)

    batch_number = i // batch_size

    print(batch_number, "/", len(papers) // batch_size)

    

    new_result = search(papers[i:end], queries, 5)

    

    for all_res, new_res in zip(results, new_result):

        all_res.extend(new_res)

        

    for j in range(len(results)):

        results[j].sort()

        results[j] = results[j][0:num_results]

print("done")
# write results to file "results.txt"

output = open("results.txt", "w")

for quer, res in zip(queries, results):

    output.write("QUERY: " + str(quer) + "\n\n")

    for sent in res:

        dist = sent[0]

        sentence = sent[1]

        file = sent[2]

        section = sent[3]

        paper_json = json.load(open(file, "r"))

        title = paper_json["metadata"]["title"]

        section_name = paper_json["body_text"][section]["section"]

        

        output.write("distance: " + str(dist) + " , title: " + title  + "\n")

        output.write("section_name: " + section_name + " , section_number: " +  str(section) + "\n")

        output.write("file: " + file + "\n")

        output.write("SENTENCE: " + str(sentence) + "\n\n")

    output.write("\n\n")