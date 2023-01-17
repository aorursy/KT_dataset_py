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
!pip install wordcloud
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
from fastai import *
from fastai.text import *
#IPython.display.Image._ACCEPTABLE_EMBEDDINGS
def gen_freq(text):
    word_list=[]
    #loop over all thetext
    for tw_words in text.split():
        word_list.extend(tw_words)
    #creating frequency using word list
    word_freq=pd.Series(word_list).value_counts()
    #print top 25 Words
    word_freq[:25]
    
    return word_freq
df=pd.read_csv("/kaggle/input/cord-19-create-dataframe/cord19_df.csv",nrows=1000).dropna()#.reset_index(drop=False)
df.head()
df=df.drop(['paper_id','methods', 'results', 'cord_uid',
       'title', 'doi', 'pmcid', 'pubmed_id', 'license', 'abstract',
       'publish_time', 'authors', 'journal', 'full_text_file', 'url',
       'publish_year', 'is_covid19', 'text_language', 'study_abstract',
       'study_methods', 'study_results', 'study_design'], axis=1)
df.head()
df.isnull().sum()
#sorting dataset based on source
a=list(df.groupby('source'))
len(a)#df1.size, df2.size
from nltk.tokenize import word_tokenize
def identify_tokens(row):
    #text = row['body_text']
    tokens = word_tokenize(row)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

df['body_text'] = df['body_text'].apply(identify_tokens)
df.head()
#removing the stop words
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
stops =  set(stopwords.words('english')+['com'])
from tqdm import tqdm
#converting all text_body in o=lower case for our eas
def lower_case(input_str):
    if type(input_str)=='float':
        return input_str
    else:
        wordslist=[]
        for word in input_str:
            worD = word.lower()
            wordslist.append(worD)
        return wordslist
df['body_text'] = df['body_text'].apply(lambda x: lower_case(x))


# covid_df[0]['body_text'] = covid_df[0]['body_text'].apply(lambda x: lower_case(x))
# covid_df[1]['body_text'] = covid_df[1]['body_text'].apply(lambda x: lower_case(x))
# covid_df[2]['body_text'] = covid_df[2]['body_text'].apply(lambda x: lower_case(x))
# covid_df[3]['body_text'] = covid_df[3]['body_text'].apply(lambda x: lower_case(x))
# covid_df[4]['body_text'] = covid_df[4]['body_text'].apply(lambda x: lower_case(x))
df.head()
"for" in stopwords.words("english")
def clean_words(w):
    if w not in stopwords.words("english"):
        return w
    else:
        pass
words=df['body_text'].apply(lambda word:[item for item in word if item not in stopwords.words("english")])
words
word_list=[]
    #loop over all thetext
for tw_words in words:
        word_list.extend(tw_words)
word_freq=pd.Series(word_list).value_counts()
    #print top 25 Words
ww=word_freq[:25]

#generate word cloud
wc=WordCloud(width=400,height=330,background_color="white").generate_from_frequencies(ww)
plt.figure(figsize=(13,9))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()
import gensim
# following 4 cells from https://www.kaggle.com/luisblanche/cord-19-use-doc2vec-to-match-articles-to-tasks notebook by Luis
def read_corpus(df, column, tokens_only=False):
    """
    Arguments
    ---------
        df: pd.DataFrame
        column: str 
            text column name
        tokens_only: bool
            wether to add tags or not
    """
    for i, line in enumerate(df[column]):

        tokens = gensim.parsing.preprocess_string(line)
        if tokens_only:
            yield tokens
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
complete_df=pd.read_csv("/kaggle/input/cord-19-create-dataframe/cord19_df.csv",nrows=1000)
complete_df.head(2)
import random
frac_of_articles = .10
train_df  = complete_df.sample(frac=frac_of_articles, random_state=1)
train_corpus = (list(read_corpus(train_df, 'body_text')))
# using distributed memory model
model = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=60, min_count=5, epochs=20, seed=42, workers=6)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
task1="""What do we know about virus genetics, origin, and evolution? What do we know about the virus origin and management measures at the human-animal interface?

Specifically, we want to know what the literature reports about:

Real-time tracking of whole genomes and a mechanism for coordinating the rapid dissemination of that information to inform the development of diagnostics and therapeutics and to track variations of the virus over time.
Access to geographic and temporal diverse sample sets to understand geographic distribution and genomic differences, and determine whether there is more than one strain in circulation. Multi-lateral agreements such as the Nagoya Protocol could be leveraged.
Evidence that livestock could be infected (e.g., field surveillance, genetic sequencing, receptor binding) and serve as a reservoir after the epidemic appears to be over.
Evidence of whether farmers are infected, and whether farmers could have played a role in the origin.
Surveillance of mixed wildlife- livestock farms for SARS-CoV-2 and other coronaviruses in Southeast Asia.
Experimental infections to test host range for this pathogen.
Animal host(s) and any evidence of continued spill-over to humans
Socioeconomic and behavioral risk factors for this spill-over
Sustainable risk reduction strategies"""
task2="""What is known about transmission, incubation, and environmental stability? What do we know about natural history, transmission, and diagnostics for the virus? What have we learned about infection prevention and control?

Specifically, we want to know what the literature reports about:

Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.
Prevalence of asymptomatic shedding and transmission (e.g., particularly children).
Seasonality of transmission.
Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).
Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).
Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).
Natural history of the virus and shedding of it from an infected person
Implementation of diagnostics and products to improve clinical processes
Disease models, including animal models for infection, disease and transmission
Tools and studies to monitor phenotypic change and potential adaptation of the virus
Immune response and immunity
Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings
Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings
Role of the environment in transmission"""
task3="""What do we know about COVID-19 risk factors? What have we learned from epidemiological studies?

Specifically, we want to know what the literature reports about:

Data on potential risks factors
Smoking, pre-existing pulmonary disease
Co-infections (determine whether co-existing respiratory/viral infections make the virus more transmissible or virulent) and other co-morbidities
Neonates and pregnant women
Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.
Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors
Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups
Susceptibility of populations
Public health mitigation measures that could be effective for control"""
task4="""What do we know about vaccines and therapeutics? What has been published concerning research and development and evaluation efforts of vaccines and therapeutics?

Specifically, we want to know what the literature reports about:

Effectiveness of drugs being developed and tried to treat COVID-19 patients.
Clinical and bench trials to investigate less common viral inhibitors against COVID-19 such as naproxen, clarithromycin, and minocyclinethat that may exert effects on viral replication.
Methods evaluating potential complication of Antibody-Dependent Enhancement (ADE) in vaccine recipients.
Exploration of use of best animal models and their predictive value for a human vaccine.
Capabilities to discover a therapeutic (not vaccine) for the disease, and clinical effectiveness studies to discover therapeutics, to include antiviral agents.
Alternative models to aid decision makers in determining how to prioritize and distribute scarce, newly proven therapeutics as production ramps up. This could include identifying approaches for expanding production capacity to ensure equitable and timely distribution to populations in need.
Efforts targeted at a universal coronavirus vaccine.
Efforts to develop animal models and standardize challenge studies
Efforts to develop prophylaxis clinical studies and prioritize in healthcare workers
Approaches to evaluate risk for enhanced disease after vaccination
Assays to evaluate vaccine immune response and process development for vaccines, alongside suitable animal models [in conjunction with therapeutics]"""
task5="""What has been published about medical care? What has been published concerning surge capacity and nursing homes? What has been published concerning efforts to inform allocation of scarce resources? What do we know about personal protective equipment? What has been published concerning alternative methods to advise on disease management? What has been published concerning processes of care? What do we know about the clinical characterization and management of the virus?

Specifically, we want to know what the literature reports about:

Resources to support skilled nursing facilities and long term care facilities.
Mobilization of surge medical staff to address shortages in overwhelmed communities
Age-adjusted mortality data for Acute Respiratory Distress Syndrome (ARDS) with/without other organ failure – particularly for viral etiologies
Extracorporeal membrane oxygenation (ECMO) outcomes data of COVID-19 patients
Outcomes data for COVID-19 after mechanical ventilation adjusted for age.
Knowledge of the frequency, manifestations, and course of extrapulmonary manifestations of COVID-19, including, but not limited to, possible cardiomyopathy and cardiac arrest.
Application of regulatory standards (e.g., EUA, CLIA) and ability to adapt care to crisis standards of care level.
Approaches for encouraging and facilitating the production of elastomeric respirators, which can save thousands of N95 masks.
Best telemedicine practices, barriers and faciitators, and specific actions to remove/expand them within and across state boundaries.
Guidance on the simple things people can do at home to take care of sick people and manage disease.
Oral medications that might potentially work.
Use of AI in real-time health care delivery to evaluate interventions, risk factors, and outcomes in a way that could not be done manually.
Best practices and critical challenges and innovative solutions and technologies in hospital flow and organization, workforce protection, workforce allocation, community-based support resources, payment, and supply chain management to enhance capacity, efficiency, and outcomes.
Efforts to define the natural history of disease to inform clinical care, public health interventions, infection prevention control, transmission, and clinical trials
Efforts to develop a core clinical outcome set to maximize usability of data across a range of trials
Efforts to determine adjunctive and supportive interventions that can improve the clinical outcomes of infected patients (e.g. steroids, high flow oxygen)"""
task6="""What do we know about the effectiveness of non-pharmaceutical interventions? What is known about equity and barriers to compliance for non-pharmaceutical interventions?

Specifically, we want to know what the literature reports about:

Guidance on ways to scale up NPIs in a more coordinated way (e.g., establish funding, infrastructure and authorities to support real time, authoritative (qualified participants) collaboration with all states to gain consensus on consistent guidance and to mobilize resources to geographic areas where critical shortfalls are identified) to give us time to enhance our health care delivery system capacity to respond to an increase in cases.
Rapid design and execution of experiments to examine and compare NPIs currently being implemented. DHS Centers for Excellence could potentially be leveraged to conduct these experiments.
Rapid assessment of the likely efficacy of school closures, travel bans, bans on mass gatherings of various sizes, and other social distancing approaches.
Methods to control the spread in communities, barriers to compliance and how these vary among different populations..
Models of potential interventions to predict costs and benefits that take account of such factors as race, income, disability, age, geographic location, immigration status, housing status, employment status, and health insurance status.
Policy changes necessary to enable the compliance of individuals with limited resources and the underserved with NPIs.
Research on why people fail to comply with public health advice, even if they want to do so (e.g., social or financial costs may be too high).
Research on the economic impact of this or any pandemic. This would include identifying policy and programmatic alternatives that lessen/mitigate risks to critical government services, food distribution and supplies, access to critical household supplies, and access to health diagnoses, treatment, and needed care, regardless of ability to pay."""
task7="""What do we know about diagnostics and surveillance? What has been published concerning systematic, holistic approach to diagnostics (from the public health surveillance perspective to being able to predict clinical outcomes)?

Specifically, we want to know what the literature reports about:

How widespread current exposure is to be able to make immediate policy recommendations on mitigation measures. Denominators for testing and a mechanism for rapidly sharing that information, including demographics, to the extent possible. Sampling methods to determine asymptomatic disease (e.g., use of serosurveys (such as convalescent samples) and early detection of disease (e.g., use of screening of neutralizing antibodies such as ELISAs).
Efforts to increase capacity on existing diagnostic platforms and tap into existing surveillance platforms.
Recruitment, support, and coordination of local expertise and capacity (public, private—commercial, and non-profit, including academic), including legal, ethical, communications, and operational issues.
National guidance and guidelines about best practices to states (e.g., how states might leverage universities and private laboratories for testing purposes, communications to public health officials and the public).
Development of a point-of-care test (like a rapid influenza test) and rapid bed-side tests, recognizing the tradeoffs between speed, accessibility, and accuracy.
Rapid design and execution of targeted surveillance experiments calling for all potential testers using PCR in a defined area to start testing and report to a specific entity. These experiments could aid in collecting longitudinal samples, which are critical to understanding the impact of ad hoc local interventions (which also need to be recorded).
Separation of assay development issues from instruments, and the role of the private sector to help quickly migrate assays onto those devices.
Efforts to track the evolution of the virus (i.e., genetic drift or mutations) and avoid locking into specific reagents and surveillance/detection schemes.
Latency issues and when there is sufficient viral load to detect the pathogen, and understanding of what is needed in terms of biological and environmental sampling.
Use of diagnostics such as host response markers (e.g., cytokines) to detect early disease or predict severe disease progression, which would be important to understanding best clinical practice and efficacy of therapeutic interventions.
Policies and protocols for screening and testing.
Policies to mitigate the effects on supplies associated with mass testing, including swabs and reagents.
Technology roadmap for diagnostics.
Barriers to developing and scaling up new diagnostic tests (e.g., market forces), how future coalition and accelerator models (e.g., Coalition for Epidemic Preparedness Innovations) could provide critical funding for diagnostics, and opportunities for a streamlined regulatory environment.
New platforms and technology (e.g., CRISPR) to improve response times and employ more holistic approaches to COVID-19 and future diseases.
Coupling genomics and diagnostic testing on a large scale.
Enhance capabilities for rapid sequencing and bioinformatics to target regions of the genome that will allow specificity for a particular variant.
Enhance capacity (people, technology, data) for sequencing with advanced analytics for unknown pathogens, and explore capabilities for distinguishing naturally-occurring pathogens from intentional.
One Health surveillance of humans and potential sources of future spillover or ongoing exposure for this organism and future pathogens, including both evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily trafficked and farmed wildlife and domestic food and companion species), inclusive of environmental, demographic, and occupational risk factors."""
task8="""What has been published about information sharing and inter-sectoral collaboration? What has been published about data standards and nomenclature? What has been published about governmental public health? What do we know about risk communication? What has been published about communicating with high-risk populations? What has been published to clarify community measures? What has been published about equity considerations and problems of inequity?

Specifically, we want to know what the literature reports about:

Methods for coordinating data-gathering with standardized nomenclature.
Sharing response information among planners, providers, and others.
Understanding and mitigating barriers to information-sharing.
How to recruit, support, and coordinate local (non-Federal) expertise and capacity relevant to public health emergency response (public, private, commercial and non-profit, including academic).
Integration of federal/state/local public health surveillance systems.
Value of investments in baseline public health response infrastructure preparedness
Modes of communicating with target high-risk populations (elderly, health care workers).
Risk communication and guidelines that are easy to understand and follow (include targeting at risk populations’ families too).
Communication that indicates potential risk of disease to all population groups.
Misunderstanding around containment and mitigation.
Action plan to mitigate gaps and problems of inequity in the Nation’s public health capability, capacity, and funding to ensure all citizens in need are supported and can access information, surveillance, and treatment.
Measures to reach marginalized and disadvantaged populations.
Data systems and research priorities and agendas incorporate attention to the needs and circumstances of disadvantaged populations and underrepresented minorities.
Mitigating threats to incarcerated people from COVID-19, assuring access to information, prevention, diagnosis, and treatment.
Understanding coverage policies (barriers and opportunities) related to testing, treatment, and care
"""
task9="""What has been published concerning ethical considerations for research? What has been published concerning social sciences at the outbreak response?

Specifically, we want to know what the literature reports about:

Efforts to articulate and translate existing ethical principles and standards to salient issues in COVID-2019
Efforts to embed ethics across all thematic areas, engage with novel ethical issues that arise and coordinate to minimize duplication of oversight
Efforts to support sustained education, access, and capacity building in the area of ethics
Efforts to establish a team at WHO that will be integrated within multidisciplinary research and operational platforms and that will connect with existing and expanded global networks of social sciences.
Efforts to develop qualitative assessment frameworks to systematically collect information related to local barriers and enablers for the uptake and adherence to public health measures for prevention and control. This includes the rapid identification of the secondary impacts of these measures. (e.g. use of surgical masks, modification of health seeking behaviors for SRH, school closures)
Efforts to identify how the burden of responding to the outbreak and implementing public health measures affects the physical and psychological health of those providing care for Covid-19 patients and identify the immediate needs that must be addressed.
Efforts to identify the underlying drivers of fear, anxiety and stigma that fuel misinformation and rumor, particularly through social media."""
from itertools import chain
from nltk.corpus import wordnet
def findsyns(word):
    synonyms = wordnet.synsets(word)
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    return lemmas
#list(findsynonyms('car'))
#words=df['body_text'].apply(lambda word:[item for item in word if item not in stopwords.words("english")])
task1a=[item for item in task1.split() if findsyns(item)]
task1a
#converting token back to string
import re
taska1a=re.sub(r"[^a-zA-Z0-9]"," ",str(task1a))
taska1a
list_of_tasks=list()
list_of_tasks=[task1,str(taska1a)]
list_of_tasks
def get_doc_vector(doc):
    tokens = gensim.parsing.preprocess_string(doc)
    vector = model.infer_vector(tokens)
    return vector
#getting vector for df and task1
body_text_vectors = model.docvecs.vectors_docs
array_of_tasks = [get_doc_vector(task) for task in list_of_tasks]
train_df['body_text_vectors'] = [vec for vec in body_text_vectors]
train_df = train_df[train_df['body_text'].apply(lambda x: len(re.findall(r"(?i)\b[a-z]+\b", x))) > 30]
train_df.shape
#KNN search fitted for body_text vec
train_array = train_df['body_text_vectors'].values.tolist()#.reshape(-1,1)
train_array[:3]
from sklearn.neighbors import NearestNeighbors
ball_tree = NearestNeighbors(algorithm='ball_tree', leaf_size=20).fit(train_array)
distances, indices = ball_tree.kneighbors(array_of_tasks, n_neighbors=3)
for i, info in enumerate(list_of_tasks):
    print("="*80, f"\n\nTask = {info[:100]}\n", )
    df =  train_df.iloc[indices[i]]
    body_text = df['body_text']
    titles = df['title']
    dist = distances[i]
    for l in range(len(dist)):
        print(f" Text index = {indices[i][l]} \n Distance to bullet = {distances[i][l]} \n Title: {titles.iloc[l]} \n Abstract Extract: {body_text.iloc[l][:200]}\n\n")
!pip install transformers
!pip install sentence-transformers
%%time
import os
import tqdm
import textwrap
import json
import prettytable
import logging
import pickle
import warnings
warnings.simplefilter('ignore')

from  transformers import *
import pandas as pd
import scipy
from sentence_transformers import SentenceTransformer

COVID_BROWSER_ASCII = """
================================================================================
  _____           _     _      __  ___    ____                                  
 / ____|         (_)   | |    /_ |/ _ \  |  _ \                                 
| |     _____   ___  __| | ___ | | (_) | | |_) |_ __ _____      _____  ___ _ __ 
| |    / _ \ \ / / |/ _` ||___|| |\__, | |  _ <| '__/ _ \ \ /\ / / __|/ _ \ '__|
| |___| (_) \ V /| | (_| |     | |  / /  | |_) | | | (_) \ V  V /\__ \  __/ |   
 \_____\___/ \_/ |_|\__,_|     |_| /_/   |____/|_|  \___/ \_/\_/ |___/\___|_|   
=================================================================================
"""

COVID_BROWSER_INTRO = """
This demo uses a state-of-the-art language model trained on scientific papers to
search passages matching user-defined queries inside the COVID-19 Open Research
Dataset. Ask something like 'Is smoking a risk factor for Covid-19?' to retrieve
relevant abstracts.\n
"""

BIORXIV_PATH = '/kaggle/input/CORD-19-research-challenge//biorxiv_medrxiv/biorxiv_medrxiv/'
COMM_USE_PATH = '/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/'
NONCOMM_USE_PATH = '/kaggle/input/CORD-19-research-challenge/noncomm_use_subset/noncomm_use_subset/'
METADATA_PATH = '/kaggle/input/CORD-19-research-challenge/metadata.csv'

DATA_PATH = '/kaggle/input/CORD-19-research-challenge/'
MODELS_PATH = 'models'
MODEL_NAME = 'scibert-nli'
CORPUS_PATH = os.path.join(DATA_PATH, 'corpus.pkl')
MODEL_PATH = os.path.join(MODELS_PATH, MODEL_NAME)
EMBEDDINGS_PATH = os.path.join(DATA_PATH, f'{MODEL_NAME}-embeddings.pkl')


def load_json_files(dirname):
    filenames = [file for file in os.listdir(dirname) if file.endswith('.json')]
    raw_files = []

    for filename in tqdm(filenames):
        filename = dirname + filename
        file = json.load(open(filename, 'rb'))
        raw_files.append(file)
    print('Loaded', len(raw_files), 'files from', dirname)
    return raw_files


def create_corpus_from_json(files):
    corpus = []
    for file in tqdm(files):
        for item in file['abstract']:
            corpus.append(item['text'])
        for item in file['body_text']:
            corpus.append(item['text'])
    print('Corpus size', len(corpus))
    return corpus


def cache_corpus(mode='CSV'):
    corpus = []
    if mode == 'CSV':
        df = pd.read_csv(METADATA_PATH)
        corpus = [a for a in df['abstract'] if type(a) == str and a != "Unknown"]
        print('Corpus size', len(corpus))
    elif mode == 'JSON':
        biorxiv_files = load_json_files(BIORXIV_PATH)
        comm_use_files = load_json_files(COMM_USE_PATH)
        noncomm_use_files = load_json_files(NONCOMM_USE_PATH)
        corpus = create_corpus_from_json(biorxiv_files + comm_use_files + noncomm_use_files)
    else:
        raise AttributeError('Mode should be either CSV or JSON')
    '''with open(CORPUS_PATH, 'wb') as file:
        pickle.dump(corpus, file)'''
    return corpus


def ask_question(query, model, corpus, corpus_embed, top_k=5):
    """
    Adapted from https://www.kaggle.com/dattaraj/risks-of-covid-19-ai-driven-q-a
    """
    queries = [query]
    query_embeds = model.encode(queries, show_progress_bar=False)
    for query, query_embed in zip(queries, query_embeds):
        distances = scipy.spatial.distance.cdist([query_embed], corpus_embed, "cosine")[0]
        distances = zip(range(len(distances)), distances)
        distances = sorted(distances, key=lambda x: x[1])
        results = []
        for count, (idx, distance) in enumerate(distances[0:top_k]):
            results.append([count + 1, corpus[idx].strip(), round(1 - distance, 4)])
    return results


def show_answers(results):
    table = prettytable.PrettyTable(
        ['Rank', 'Abstract', 'Score']
    )
    for res in results:
        rank = res[0]
        text = res[1]
        text = textwrap.fill(text, width=75)
        text = text + '\n\n'
        score = res[2]
        table.add_row([
            rank,
            text,
            score
        ])
    print('\n')
    print(str(table))
    print('\n')

if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')
    print(COVID_BROWSER_ASCII)
    print(COVID_BROWSER_INTRO)
    if not os.path.exists(CORPUS_PATH):
        print("Caching the corpus for future use...")
        corpus = cache_corpus()
    else:
        print("Loading the corpus from", CORPUS_PATH, '...')
        with open(CORPUS_PATH, 'rb') as corpus_pt:
            corpus = pickle.load(corpus_pt)

    model =  SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    if not os.path.exists(EMBEDDINGS_PATH):
        print("Computing and caching model embeddings for future use...")
        embeddings = model.encode(corpus, show_progress_bar=True)
        '''with open(EMBEDDINGS_PATH, 'wb') as file:
            pickle.dump(embeddings, file)'''
    else:
        print("Loading model embeddings from", EMBEDDINGS_PATH, '...')
        with open(EMBEDDINGS_PATH, 'rb') as file:
            embeddings = pickle.load(file)
#list(findsynonyms('car'))
#words=df['body_text'].apply(lambda word:[item for item in word if item not in stopwords.words("english")])
task2a=[item for item in task1.split() if findsyns(item)]
#converting token back to string
import re
taska2a=re.sub(r"[^a-zA-Z0-9]"," ",str(task2a))

#task3 
task3a=[item for item in task1.split() if findsyns(item)]
#converting token back to string
import re
taska3a=re.sub(r"[^a-zA-Z0-9]"," ",str(task3a))

#task 4
task4a=[item for item in task1.split() if findsyns(item)]
#converting token back to string
import re
taska4a=re.sub(r"[^a-zA-Z0-9]"," ",str(task4a))

#task5 
task5a=[item for item in task1.split() if findsyns(item)]
#converting token back to string
import re
taska5a=re.sub(r"[^a-zA-Z0-9]"," ",str(task5a))

#task6
task6a=[item for item in task1.split() if findsyns(item)]
#converting token back to string
import re
taska6a=re.sub(r"[^a-zA-Z0-9]"," ",str(task6a))

#task7
task7a=[item for item in task1.split() if findsyns(item)]
#converting token back to string
import re
taska7a=re.sub(r"[^a-zA-Z0-9]"," ",str(task7a))

#task8
task8a=[item for item in task1.split() if findsyns(item)]
#converting token back to string
import re
taska8a=re.sub(r"[^a-zA-Z0-9]"," ",str(task8a))


#task9
task9a=[item for item in task1.split() if findsyns(item)]
#converting token back to string
import re
taska9a=re.sub(r"[^a-zA-Z0-9]"," ",str(task9a))
list_of_tasks=list()
list_of_tasks=[task1,str(taska1a),task2,str(taska2a),task3,str(taska3a),task4,str(taska4a),task5,str(taska5a),task6,str(taska6a),task7,str(taska7a),task8,str(taska8a),task9,str(taska9a)]
for i in range(len(list_of_tasks)):
        query = list_of_tasks[i]
        print(f'Query {i+1} : {query}\n\n')
        results = ask_question(query, model, corpus, embeddings)
        show_answers(results)
