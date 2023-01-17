%%capture

!pip install sentence-transformers

!pip install transformers --upgrade

!pip install langdetect

!pip install rank_bm25

!pip install networkx
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path, PurePath #Easy path directory

import json # Reads json

import os

import multiprocessing as mp



import transformers # NLP task pipeline

from transformers import pipeline, AutoModelWithLMHead, AutoModelForQuestionAnswering, AutoTokenizer, AutoModel, AutoTokenizer, AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer # For downloading pretrained models

from sentence_transformers import SentenceTransformer, models # For sentence embeddings trained on semantic similarity



from langdetect import detect

import re



import scipy

import statistics 

from rank_bm25 import BM25Okapi

import nltk

from nltk.corpus import stopwords



# For network analysis on citations

import networkx as nx

from itertools import chain

from tqdm.auto import tqdm

tqdm.pandas()



nltk.download("punkt")

nltk.download('stopwords')

pd.set_option('display.max_columns', None)  
keywords = [

    'remdesivir',

    'azithromycin',

    'ciprofloxacin',

    'lopinavir',

    'ritonavir',

    'interferon',

    'chloroquine',

    'hydroxychloroquine',

    'darunavir',

    'cobicistat',

    'emtricitabine',

    'nelfinavir',

    'tenofovir',

    'saquinavir',

    'azuvudine',

    'favipiravir',

    'umifenovir',

    'oseltamivir',

    'baloxavir',

    'methylprednisolone',

    'ribvarin',

    'sofosbuvir',

    'beclabuvir',

    'galidesivir',

    'simeprevir',

    'nitazoxanide',

    'niclosamide',

    'naproxen',

    'clarithromycin',

    'minocyclinethat',

    'human monoclonal antibody',

    'tocilizumab',

    'sarilumab',

    'leronlimab',

    'foralumab',

    'camrelizumab',

    'ifx-1',

    'ifx',

    'arbidol',

    'fingolimod',

    'brilacidin',

    'sirolimus',

    'danoprevir',

    'rintatolimod',

    'cynk-001',

    'cynk',

    'tmprss2',

    'jak',

    'zinc',

    'quercetin',

    'convalescent plasma',

    'nanoviricide',

    'corticosteroids',

    'bevacizumab',

    'bxt-25',

    'bxt',

    'angiotension',

    'rhace2',

    'pirfenidone',

    'thalidomide',

    'brohexine hydrochloride',

    'dehydroandrographolide succinate',

    'antibody dependent enhancement',

    'antibody-dependent enhancement',

    ' ade ',

    'prophylaxis',

    'prophylactic',

    'vaccine',

    'assay',

    'elisa',

    'th1',

    'th2',

    'elispot',

    'cytometry',

    'ctc'

]

# Functions



# This is a list of English stopwords that do not contribute to useful information and helps to generate useful tokens for searching

english_stopwords = list(set(stopwords.words('english')))



# This function performs the BM25 algorithm on the full text against a query of tokenized words and returns the relevant sections of text that meets a threshold criteria

def get_deeper_context(row):

    # Grab variables from the input data row

    sha = row['sha']

    results = row['result']

    

    # Sometimes outputs will contain multiple results, that is separated by the "\n\n" delimiter. So we make sure to split the results and process them separately.

    results = results.split("\n\n")

    

    # We take each results and preprocess them so that useful words are tokenized. These will help perform our BM25 search

    tokenized_results = [preprocess(result) for result in results]

    

    # Create variables to be stored later

    paragraphs = []

    tokenized_paragraphs = []

    candidate_paragraphs = []

    candidates = ''

    

    # Scanning through data to read the relevant file based on the Sha and acquire the full text

    for path in Path(directory).rglob('*.json'):

        if sha in path.name:

            data = json.loads(open(path.absolute()).read())

            # Grabs the full body text (which is a list of paragraphs)

            body_text = data['body_text']



            # Loops through each paragraph and appends them into a list

            for i, paragraph in enumerate(body_text):

                text = " ".join(paragraph['text'].split()).replace(" ,",",")

                paragraphs.append(text)

                tokenized_paragraphs.append(preprocess(text))

            break



    try: 

        # Feed the paragraphs into the BM25 API

        bm25 = BM25Okapi(tokenized_paragraphs)

        # Loop through the tokenized results and get BM25 scores to see which paragraphs were most relevant to the query.

        for tokenized_result in tokenized_results:

            doc_scores = bm25.get_scores(tokenized_result) # BM25 scores

            candidate_paragraphs.append(get_sorted_indices(doc_scores)) # Saving the indices of relevant paragraphs into a list



        # Deduping and sorting the list by index number

        candidate_paragraphs = [item for sublist in candidate_paragraphs for item in sublist]

        candidate_paragraphs = list(set(candidate_paragraphs))

        candidate_paragraphs.sort()



        # Combine the relevant paragraphs into a single string.

        for index in candidate_paragraphs:

            candidates = candidates+"Paragraph: "+str(index)+"\n"

            candidates = candidates+paragraphs[index]+"\n \n"

    except:

        candidates = "NA"

        

    # Saving the results into their own column

    row['context'] = candidates



    return row



# This function takes the BM25 scores, produces the mean and standard deviation of the score, and outputs the relevant scores only if it is 1.5 standard deviations way from the mean

def get_sorted_indices(l):

    std = statistics.stdev(l) # Standard deviation

    mean = statistics.mean(l) # Mean

    threshold = mean+(std*1.5) # 1.5 standard deviation treshold

    max_score = max(l) # Max scire

    

    indices = []



    # Looping through the scores and applying the threshold

    for index, score in enumerate(l):

        if score >= threshold:

            indices.append(index)

    

    indices.sort()

    

    return indices



# This function strips characters such as apostrophes, etc.

def strip_characters(text):

    t = re.sub('\(|\)|:|,|;|\.|’|”|“|\?|%|>|<', '', text)

    t = re.sub('/', ' ', t)

    t = t.replace("'",'')

    return t



# This function calls the strip_characters() function and also lowercases all text

def clean(text):

    t = text.lower()

    t = strip_characters(t)

    return t



# This function takes a text and cleans and tokenizes it, while removing the stopwords.

def tokenize(text):

    words = nltk.word_tokenize(text)

    return list(set([word for word in words 

                     if len(word) > 1

                     and not word in english_stopwords

                     and not (word.isnumeric() and len(word) is not 4)

                     and (not word.isnumeric() or word.isalpha())] ))



# This is the wrapper function that incorporates the previous functions to clean and tokenize text

def preprocess(text):

    t = clean(text)

    tokens = tokenize(t)

    return tokens





# This function takes in a piece of text and extracts out the pre-specified keywords found in the text.

def get_keywords(row):

    found_keywords = []

        

    # Looping through each row and column of the data

    for col in row.iteritems():

        # Checking if the column is the title or abstract

        if ("title" in col[0]) | ("abstract" in col[0]):

            text = col[1].lower() # lowercass

            text = " ".join(text.split()) # removes useless whitespace

            

            # Loops through the known keywords and detects if it is found in the text

            for keyword in keywords:

                if keyword in text:

                    found_keywords.append(keyword)

            

    # De-duplicates the keywords found

    found_keywords = set(found_keywords)

    

    # If no keywords found, return NA

    if len(found_keywords) == 0:

        row['keywords'] = 'NA'

    else:

        row['keywords'] = "; ".join(found_keywords)

        

    return row

# Read the metadata

all_data = []

directory = '/kaggle/input/CORD-19-research-challenge/'

metadata = pd.read_csv(directory+"metadata.csv")

print("Total number of articles")

print(metadata.shape)



# Filter for articles newer than 2019-11-01

date = '2019-11-01'

metadata = metadata[metadata['publish_time'] >= date]

print("Filter for articles after date: "+date)

print(metadata.shape)



# Filter articles that don't have full text

metadata = metadata[metadata['has_pdf_parse']]

print("Filter for full text")

print(metadata.shape)



# Filter articles that don't have title or abstract

metadata = metadata[metadata['title'].str.len() > 0]

metadata = metadata[metadata['abstract'].str.len() > 0]

print("Filter for non-empty title and abstract")

print(metadata.shape)



# Filter certain articles that seem to be of bad quality and messes up export formating

metadata = metadata[metadata['sha'] != 'a5293bb4f17ad25a72133cdd9eee8748dd6a4b8d']

metadata = metadata[metadata['sha'] != 'b30770ae30b35cdfaf0a173863e74e93edbb0329']



# Clean text data

metadata['title'] = metadata['title'].apply(lambda x: " ".join(x.split()))

metadata['abstract'] = metadata['abstract'].apply(lambda x: " ".join(x.split()))



# Filter for titles and abstracts that have mention of one of the keywords

keyword_query = "|".join(keywords)

metadata = metadata[metadata['title'].str.contains(keyword_query, flags=re.IGNORECASE, regex=True) | 

                    metadata['abstract'].str.contains(keyword_query, flags=re.IGNORECASE, regex=True)]



# Finds the keywords found in each article and makes a column out of it

metadata = metadata.apply(get_keywords, axis=1)

print("Filter for terms relating to treatments")

print(metadata.shape)



# Drop non-english articles

for index, row in metadata.iterrows():

    title = row['title']

    lang = detect(title)

    if lang != 'en':

        metadata.drop(index, inplace=True)



print("Filter for English articles")

print(metadata.shape)



# Resets the index

metadata = metadata.reset_index(drop=True)
def open_json(file_path):

    """

    Helper function to open json file

    """

    with open(file_path, 'r') as f:

        json_dict = json.load(f)

    return json_dict





def json_path_generator(data_path=os.path.abspath("/kaggle/input"), limit=None):

    """

    Helper function to get all the paths of json files in the input directory.

    """

    return_files = []

    for dirname, _, filenames in os.walk(data_path):

        for filename in filenames:

            if filename[-5:] == ".json":

                return_files.append(os.path.join(dirname, filename))

                if limit is not None and type(limit) == int:

                    if len(return_files) >= limit:

                        return return_files

    return return_files





def get_json_dicts(paths, progress_bar=None):

    """

    Helper function to open a list of paths as json dicts. Careful about memory usage here.

    Optionally takes a tqdm bar as input to show progress of loading.

    """

    json_dicts = []

    # (I) Max of 2 or number of cpus minus 1, then min of (I) or the number of paths. If limit is used and its small,

    # avoid excessive pool sizes.

    process_num = min(max(2, os.cpu_count()), len(paths))

    with mp.Pool(process_num) as pool:

        for result in pool.imap_unordered(open_json, paths):

            json_dicts.append(result)

            if progress_bar is not None:

                progress_bar.update(1)



    return json_dicts





# Get articles cited by our articles

def get_article_citations(article_dict, select_articles=None):

    """

    Function to extract set of (from, to) citation edges from articles.

    :select_articles: A set of articles to check so that only citations recorded are those

                      where the 'to' node is in the set. Used in this block to limit our citation graph

                      to only those citations of the articles we've deemed relevant.

    """

    article_title = article_dict.get("metadata", {}).get("title", "")

    article_citations = set()

    # Get citations and their ids

    bib_entries = article_dict.get("bib_entries", {})

    for entry in bib_entries:

        entry_dict = bib_entries.get(entry, {})

        entry_title = entry_dict.get("title", "")

        if select_articles is not None and type(select_articles) == set:

            # Check that article being cited is in our list of articles to look for

            if entry_title in select_articles:

                article_citations.add((article_title, entry_title))

        else:

            article_citations.add((article_title, entry_title))

    return list(article_citations)





def get_article_citations_meta(arg_list):

    """

    Meta function of get_article_citations so that we can parallelize with multiple arguments.

    First argument is a single arguments dictionary.

    Second argument is the optional set of article titles to limit citations.

    """

    return get_article_citations(arg_list[0], arg_list[1])





def divide_chunks(l, n): 

    """

    Function shameless taken from stackoverflow to split a list (l) into sublists of size n.

    """

    # looping till length l 

    for i in range(0, len(l), n):  

        yield l[i:i + n]





def process_all_articles(tasks: list, function, progress_bar=None):

    """

    A wrapper function to call a single function over a set of tasks with a multiprocessing pool.

    """

    results = []

    process_num = min(max(2, os.cpu_count()), len(tasks))

    with mp.Pool(process_num) as pool:

        for result in pool.imap_unordered(function, tasks):

            results.append(result)

            if progress_bar is not None:

                progress_bar.update(1)

    return results





def add_citation_edges(graph: nx.DiGraph, edges: list):

    """

    Function to record edges in the networkx DiGraph object.

    """

    all_citation_edges = list(set(chain.from_iterable(edges)))

    graph.add_edges_from(all_citation_edges)





def build_citation_graph(filter_articles=None, paths=None, limit=None, chunk_size=5000):

    """

    Function to build citation graph from beginning to end.

    :filter_articles: Set of article titles to limit citations to. 

                      Citations will only be recorded if the 'to' article's title is in the set.

    :paths: Subset of paths to operate over. If None operate over all.

    :limit: Can limit number of paths to this given number.

    :chunk_size: Number of article dicts to hold in memory at once to process. 5000 seems like a decent choice.

    """

    # Get paths of articles

    if paths is None:

        paths = json_path_generator(limit=limit)

    # Split the paths into chunks for memory efficiency

    chunked_paths = list(divide_chunks(paths, chunk_size))

    # Build the citation graph

    graph = nx.DiGraph()

    functions = [

        get_article_citations_meta

    ]

    function_progress_bar = tqdm(total=len(functions), leave=False, position=1, desc="Function progress on chunk")

    if limit is None:

        task_num = chunk_size

    else:

        task_num = min(chunk_size, limit)

    task_progress_bar = tqdm(total=task_num, leave=False, position=2, desc="Task progress bar")

    all_results = list()

    for paths in tqdm(chunked_paths, leave=False, position=0, desc="Chunk progress"):

        task_progress_bar.reset(total=len(paths))

        path_dicts = get_json_dicts(paths)

        tasks = [[x, filter_articles] for x in path_dicts]

        function_progress_bar.reset()

        for func in functions:

            func_name = func.__name__

            function_progress_bar.set_description("Calling " + func_name)

            results = process_all_articles(tasks, func, task_progress_bar)

            # Combine list of sets

            result_edges = list(set(chain.from_iterable(results)))

            all_results.append(result_edges)

            function_progress_bar.update(1)

    # Update the graph object

    add_citation_edges(graph, all_results)

    function_progress_bar.close()

    task_progress_bar.close()

    print("Done")

    return graph





relevant_article_citation_graph = build_citation_graph(set(metadata.title.tolist()), chunk_size=5000)
def get_number_citations(article_title: str, citation_graph: nx.DiGraph) -> int:

    """

    Function to get the number of citations received by an article.

    :article_title: Title of article to check for numbre of citations received.

    :citation_graph: nx.DiGraph instance with edges denoting the number of citations

                     built with the orientation (citing article, cited article).

    """

    num_citations = citation_graph.in_degree(article_title)

    if type(num_citations) is not int:

        return 0

    return num_citations



# Calculating the number of times an article was cited in each of the articles we filtered for

metadata['number_citations'] = metadata.title.apply(lambda title: get_number_citations(title, relevant_article_citation_graph))



# Saving the initial filtering in a file

metadata.to_pickle('/kaggle/working/metadata.pkl')

metadata.to_csv('/kaggle/working/metadata.csv', index=False)



# Printing out most popular citations in our filter search

metadata.sort_values('number_citations', ascending=False).head(10)[['title', 'number_citations']]
# Create folder to store out BioBERT model

if not os.path.exists('/kaggle/working/model'):

    os.makedirs('/kaggle/working/model')

    

# Make dataframe of sentences from abstract

sent_dict = {'sha':[],'sentence':[]}



# Loop through our filtered list from the metadata

for index, row in metadata.iterrows():

    sha = row['sha']

    abstract = row['abstract']

    

    # Take the abstract and tokenize them on the sentence level

    sentences = nltk.tokenize.sent_tokenize(abstract)

    

    # Loop through the abstract sentences

    for sentence in sentences:

        # Make sure sentence is at least 6 words (to filter out useless labelings or headings)

        sentence_split = sentence.split()

        if len(sentence_split) > 5:

            sent_dict['sha'].append(sha)

            sent_dict['sentence'].append(sentence)



# Convert our list of abstract sentences to a dataframe

df_sentences = pd.DataFrame(sent_dict)

df_sentences.head()



# Download and setup the model

tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli")

model = AutoModelWithLMHead.from_pretrained("gsarti/biobert-nli")



# Initialize and save the model

model.save_pretrained("/kaggle/working/model")

tokenizer.save_pretrained("/kaggle/working/model")

embedding = models.BERT("/kaggle/working/model",max_seq_length=128,do_lower_case=True)

pooling_model = models.Pooling(embedding.get_word_embedding_dimension(),

                               pooling_mode_mean_tokens=True,

                               pooling_mode_cls_token=False,

                               pooling_mode_max_tokens=False)



model = SentenceTransformer(modules=[embedding, pooling_model])

model.save("/kaggle/working/model")

encoder = SentenceTransformer("/kaggle/working/model")



# Perform the sentence embedding conversion

sentences = df_sentences['sentence'].tolist()

sentence_embeddings = encoder.encode(sentences)

df_sentences['embeddings'] = sentence_embeddings



# Save the sentence embeddings dataframe.

df_sentences.to_pickle('/kaggle/working/sentence_embeddings.pkl')

# Some analysis

found_keywords = []

for keyword in metadata['keywords']:

    found_keywords.append(keyword.split("; "))



found_keywords = set([item for sublist in found_keywords for item in sublist])



print(found_keywords)

print("Number of keywords found: "+str(len(found_keywords)))
# Make querys against the abstract sentence embeddings to identify candidates



def execute_query(query, metadata, sha = '', similarity_threshold = 0.65, print_output=True):

    

    # 1. Take the query and produce a sentence embeddings from it

    query = [query]

    query_embedding = encoder.encode(query)

    

    # See if the query contains any of our keywords

    query_keywords = list(set(query[0].split()) & set(keywords))

    

    # 2. If there is no query and a sha is provided, then set the similarity threshold to 0, as it means we want to output the full text of a single article

    similarity_threshold = 0 if ((query[0] == "") and len(sha)>0) else similarity_threshold



    # 3. Calculate the cosine distance between the query and the abstract sentences to produce simlarity scores

    distances = scipy.spatial.distance.cdist(query_embedding, sentence_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances) # Pair the indices with the cosine distance

    results = sorted(results, key=lambda x: x[0]) # Sort them by index (this is needed to match the cosine scores with the results)



    # 4. Grab all the similarity results above the specified similarity threshold  

    result_dict = {'sha':[],'result':[]}

    

    # Loop through the results of the cosine distance calculations

    for idx, distance in results:

        # The similarity score is 1-distance (so that higher score = better)

        similarity_score = 1-distance

        

        # Get the Sha and the sentence from our sentence dataframe

        sentence = df_sentences['sentence'].iloc[idx].strip()

        sha_id = df_sentences['sha'].iloc[idx].strip()



        # If the similarity score of the sentence is below the threshold, then ignore it

        if similarity_score < similarity_threshold:

            continue

            

        # If a single sha id was provided, make sure to skip all the other articles that don't match that id

        if len(sha) > 0 and sha_id !=sha:

            continue



        # If known keywords were found in the query, then make sure that the abstract contains that keyword

        if len(query_keywords) > 0:

                # Get the abstract from the sha id

                abstract = metadata[metadata['sha'] == sha_id]['abstract'].item().lower()

                # Determine if the keyword is in the abstract, if so then add that result

                if any(keyword in abstract for keyword in query_keywords):

                    result_dict['sha'].append(sha_id)

                    result_dict['result'].append(sentence)

        # If instead a single Sha id was provided, then then if a query was provivded or not and return the results

        elif len(sha) > 0 and sha_id == sha:

            result_dict['sha'].append(sha_id)

            # If a query was not provided, then the result is blank (and later on the full body text will be returned)

            if query[0] == "":

                result_dict['result'].append("")

            # Otherwise, return the relevant result

            else:

                result_dict['result'].append(sentence)

        # If no known keywords or single sha was identified, just return all available matches

        else:

            result_dict['sha'].append(sha_id)

            result_dict['result'].append(sentence)

    

    # Convert the stage 1 results to a dataframe

    temp_result = pd.DataFrame(result_dict)



    # 5. Use the keywords from the results, process them, and feed them to the BM25 algorithm in order to extract the relevant pieces of text from the full body to provide a richer and deeper context of the results (**Stage 2**)

    

    # If there are multiple results found, then merge them together in the same string using the "\n\n" delimeter (So that context from all results are obtained)

    temp_result = temp_result.groupby('sha')['result'].apply("\n\n".join).reset_index()

    # Apply the get_deeper_context() method to all the results in order to grab the deeper context

    temp_result = temp_result.apply(get_deeper_context, axis=1)

    # Save all the rsults in a dataframe

    report = pd.merge(temp_result, metadata[['sha','title','publish_time','abstract','keywords','journal','number_citations']], how='inner')

    report['query'] = query[0]



    # If the print_output paramter is specified (default = True), then print the results in the console

    if print_output:

        # Print results

        print("======= REPORT =========")



        for index, row in report.iterrows():

            title = row['title']

            sha = row['sha']

            result = row['result']

            num_citations = row['number_citations']

            abstract = row['abstract']

            context = row['context']

            keywords_found = row['keywords']

            publish_date = row['publish_time']



            print("Query: "+query[0])

            print("Sha: "+sha+"\n")

            print("Title: "+title+"\n")

            print("Published Date: "+publish_date+"\n")

            print("Number of times cited: "+str(num_citations)+"\n")

            print("Antiviral Related Terms: "+keywords_found+"\n")

            if (len(result.replace("\n","")) > 0):

                print("Results: \n"+result+"\n")

            print("Abstract: ")

            print(abstract+"\n")

            print("Context: ")

            print(context+"\n")

            print("----------------")

    

    return report

                
execute_query("antiviral treatments for COVID-19", metadata, similarity_threshold=0.70)
execute_query("", metadata, sha='23e7355b5e4e0209f64c9d8d5772092a53b72686')
queries = [

    'antiviral treatments for COVID-19',

    'hydroxychloroquine for treatment of COVID-19',

    'preventative clinical studies for COVID-19',

    'anti-viral prophylaxis studies for COVID-19',

    'prophylaxis studies for COVID-19',

    'diagnostic assay for COV response',

    'immunoassay for antibody or cell response',

    'ELISA or flow cytometry assay for cov',

    'mouse or ferret model for assay evaluation'

]



for query in queries:

    result = execute_query(query, metadata, print_output = False)

    result.to_csv("/kaggle/working/query - "+query+".csv", index=False)