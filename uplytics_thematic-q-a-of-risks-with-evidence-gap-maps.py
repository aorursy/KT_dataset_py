#Import the required packages

import pandas as pd

import numpy as np

from numpy.random import seed

from numpy.random import randint

import re

import gc

import glob

import json

import time

import matplotlib.pyplot as plt

import plotly.graph_objects as go

from tqdm import tqdm

print("Basic Packages Loaded")
rebuild = True # if True the df_covid dataframe which hosts all data is build from scratch , else the earlier generated dataframe is reloaded

trainDoc2Vec = True # if True Doc2Vec Model would be re-trained else stored model is loaded from previous run

filename = 'df_covid_Apr-16-2020.csv'

csv_path = filename
if rebuild :

    # Code adopted from https://www.kaggle.com/maksimeren/covid-19-literature-clustering

    #Load Metadata

    root_path = '/kaggle/input/CORD-19-research-challenge/'

    metadata_path = f'{root_path}/metadata.csv'

    meta_df = pd.read_csv(metadata_path, dtype={

        'pubmed_id': str,

        'Microsoft Academic Paper ID': str, 

        'doi': str})

    # Load all Json 

    root_path = '/kaggle/input/CORD-19-research-challenge/'

    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

    print(len(all_json))

    

    #A File Reader Class which loads the json and make data available

    class FileReader:

        def __init__(self, file_path):

            with open(file_path) as file:

                content = json.load(file)

                self.paper_id = content['paper_id']

                self.abstract = []

                self.body_text = []

                # Abstract

                try:

                    if content['abstract']:

                        for entry in content['abstract']:

                            self.abstract.append(entry['text'])  

                except KeyError:

                    #do nothing

                    pass 

                # Body text

                for entry in content['body_text']:

                    self.body_text.append(entry['text'])

                self.abstract = '\n'.join(self.abstract)

                self.body_text = '\n'.join(self.body_text)

        def __repr__(self):

            return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

        

    first_row = FileReader(all_json[0])

    print(first_row)

        

    #Utiliity to add line breaks so that titles and abstracts can be displayed on hoover

    def get_breaks(content, length):

         data = ""

         words = content.split(' ')

         total_chars = 0



         # add break every length characters

         for i in range(len(words)):

            total_chars += len(words[i])

            if total_chars > length:

                data = data + "<br>" + words[i]

                total_chars = 0

            else:

                data = data + " " + words[i]

         return data

    

    #Create a dictionary which is eventually copied into a dataframe 

    dict_ = None

    dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'title_summary': [],'abstract_summary': [], 'publish_year': [], 'publish_date': [],'doi': []}

    for idx, entry in enumerate(all_json):

        if idx % (len(all_json) // 10) == 0:

            print(f'Processing index: {idx} of {len(all_json)}')

        content = FileReader(entry)



        # get metadata information

        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

        # no metadata, skip this paper

        if len(meta_data) == 0:

            continue



        dict_['paper_id'].append(content.paper_id)

        dict_['abstract'].append(content.abstract)

        dict_['body_text'].append(content.body_text)



        # also create a column for the summary of abstract to be used in a plot

        if len(content.abstract) == 0: 

            dict_['abstract_summary'].append("Not Available")

        elif len(content.abstract.split(' ')) > 100:

            # abstract provided is too long for plot, take first 100 words append with ...

            info = content.abstract.split(' ')[:100]

            summary = get_breaks(' '.join(info), 40)

            dict_['abstract_summary'].append(summary + "...")

        else:

            # abstract is short enough

            summary = get_breaks(content.abstract, 40)

            dict_['abstract_summary'].append(summary)



        # get metadata information

        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]



        try:

            # if more than one author

            authors = meta_data['authors'].values[0].split(';')

            if len(authors) > 2:

                # more than 2 authors, may be problem when plotting, so take first 2 append with ...

                dict_['authors'].append(". ".join(authors[:2]) + "...")

            else:

                # authors will fit in plot

                dict_['authors'].append(". ".join(authors))

        except Exception as e:

            # if only one author - or Null valie

            dict_['authors'].append(meta_data['authors'].values[0])



        # add the title information, add breaks when needed

        try:

            title = get_breaks(meta_data['title'].values[0], 40)

            dict_['title_summary'].append(title)

            dict_['title'].append(meta_data['title'].values[0])

        # if title was not provided

        except Exception as e:

            dict_['title_summary'].append("Not Available")

            dict_['title'].append("Not Available")





        # add the journal information

        dict_['journal'].append(meta_data['journal'].values[0])



        #add the year where available from the meta data 

        dict_['publish_year'].append((pd.DatetimeIndex(meta_data['publish_time']).year).values[0])

        

        #add the date as a separate column 

        dict_['publish_date'].append((meta_data['publish_time']).values[0])



        #add the doi where available from the meta data 

        dict_['doi'].append(meta_data['doi'].values[0])



    #print(len(dict_['paper_id']), len(dict_['abstract']),len(dict_['body_text']),len(dict_['authors']),len(dict_['title']),len(dict_['journal']),len(dict_['title_summary']),len(dict_['abstract_summary']),len(dict_['publish_year']),len(dict_['doi']))

    df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'title_summary','abstract_summary', 'doi', 'publish_year','publish_date'])

    

    #Format the doi so that it can be used as a link

    def doi_url(d):

        if d=='':

            return "Not Available"

        if str(d).startswith('http://'):

            return str(d)

        elif str(d).startswith('doi.org'):

            return f'http://{str(d)}'

        else:

            return f'http://doi.org/{str(d)}'

    df_covid['doi'] = df_covid['doi'].apply(lambda x: doi_url(x))

    

    #Drop duplicates of where the body text is same

    df_covid.drop_duplicates(['body_text'], inplace=True)

    df_covid['body_text'].describe(include='all')

    

    #Mark cells where data is not available

    df_covid.isna().sum()

    

    #Mark missing data so that the same is available while printing

    df_covid['title'].replace(np.nan, 'Not Available', regex=True, inplace=True)

    df_covid['journal'].replace(np.nan, 'Not Available', regex=True, inplace=True)

    df_covid['authors'].replace(np.nan, 'Not Available', regex=True, inplace=True)

    df_covid['abstract'].replace(np.nan, 'Not Available', regex=True, inplace=True)

    df_covid['abstract'].replace("", 'Not Available', regex=True, inplace=True)

    

    #Provide a timestamp and store the csv file so that it can be directly loaded

    t = time.localtime()

    timestamp = time.strftime('%b-%d-%Y', t)

    filename = ("df_covid_" + timestamp +'.csv')

    df_covid.to_csv(filename , index = False) 

else :

    df_covid = pd.read_csv(csv_path)
# Helper function to draw word clouds to analyze content which is either to short or long to ensure that they are relevant

from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=1000,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(str(data))



    fig = plt.figure(1, figsize=(12, 12))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

bodylengths = []

for ind in df_covid.index: 

    bodylengths.append(len(df_covid['body_text'][ind]))

bodylengths = np.array(bodylengths)

print("Mean {}".format(bodylengths.mean()))

print("Min {}".format(bodylengths.min()))

print("Max {}".format(bodylengths.max()))

print("Bottom 1 Percentile {}".format(np.percentile(bodylengths, 1)))
plt.hist(bodylengths,range=[0, 5000]) 

plt.title('Word Count of Papers')

plt.xlabel('Word Count')

plt.ylabel('Number of Papers')

plt.show()
df_covid_short = df_covid[(df_covid['body_text'].apply(lambda x: len(x))<2200)]

print(df_covid_short.shape)

print(df_covid_short[(df_covid_short['publish_year'] == 2020)].shape)

df_covid_short.head()
show_wordcloud(df_covid_short['body_text'])
df_covid = df_covid[(df_covid['body_text'].apply(lambda x: len(x))>2200)]

df_covid.reset_index(drop=True, inplace = True)
plt.hist(bodylengths,range=[100000, 200000]) 

plt.title('Word Count of Papers')

plt.xlabel('Word Count')

plt.ylabel('Number of Papers')

plt.show()
df_covid_long = df_covid[(df_covid['body_text'].apply(lambda x: len(x))>160000)]

print(df_covid_long.shape)

print(df_covid_long[(df_covid_long['publish_year'] == 2020)].shape)

df_covid_long.head()
show_wordcloud(df_covid_long['body_text'])
import gensim

from gensim import models, similarities

from gensim.models.doc2vec import TaggedDocument 

from gensim.models.doc2vec import Doc2Vec

from sklearn.manifold import TSNE
if trainDoc2Vec :

    # Train on the complete Body Text of the papers. 

    df_covid['body_text_clean'] = df_covid['body_text'].apply(lambda x: gensim.parsing.preprocess_string(x))

    tagged_data  = [TaggedDocument(doc,[i]) for i, doc in enumerate(list(df_covid['body_text_clean']))]

    # Using distributed memory’ (PV-DM) algorithm

    doc2vecmodel = gensim.models.doc2vec.Doc2Vec(dm=1, vector_size=50, min_count=5, epochs=10, seed=42, workers=4)

    doc2vecmodel.build_vocab(tagged_data)

    doc2vecmodel.train(tagged_data, total_examples=doc2vecmodel.corpus_count, epochs=doc2vecmodel.epochs)

    doc2vecmodel.save('covid19doc2vec.model')

    print('Doc2Vec Model Build')

else :

    doc2vecmodel = gensim.models.Doc2Vec.load("covid19doc2vec.model")

    print('Doc2Vec Model Loaded') 

#perplexity of 5 and learning rate of 500 gives good results

tsne5 = TSNE(n_components=2, perplexity=5, learning_rate = 500)

tsne20 = TSNE(n_components=2, perplexity=20, learning_rate = 500)

doc2vec_tsne5 = tsne5.fit_transform(doc2vecmodel.docvecs.vectors_docs)

doc2vec_tsne20 = tsne20.fit_transform(doc2vecmodel.docvecs.vectors_docs)

print("tSNE Cordinates Ready for Doc2Vec vectors")
from ipywidgets import interact

per = [5,20]

@interact

def update_tSNE(perplexity = per):

    if perplexity == 5 :

        a = doc2vec_tsne5[:,0]

        b = doc2vec_tsne5[:,1]

    else : 

        a = doc2vec_tsne20[:,0]

        b = doc2vec_tsne20[:,1]

    

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=a, y=b,mode='markers'))

    fig.show()

doc2vec_tsne = doc2vec_tsne5 
#Helper functions for referencing data from document index

def get_docid(n):

    return df_covid['paper_id'][n]

def get_title(n):

    title = df_covid['title'][n]

    return title

def get_text_body(n):

    body = df_covid['body_text'][n]

    return body

def get_abstract(n):

    body = df_covid['abstract'][n]

    return body

def get_xy_cordinates(n) :

    return doc2vec_tsne[n]

def get_abstract_formatted(n):

    body = df_covid['abstract_summary'][n]

    return body

def get_title_formatted(n):

    title = df_covid['title_summary'][n]

    return title

word2vec_root_path = '/kaggle/input/covid-19-word2vec-model-and-vectors/'

word2vec_filename = 'covid19word2vec.model'

word2vecfile =  word2vec_root_path + word2vec_filename

w2vecmodel = gensim.models.Word2Vec.load(word2vecfile)
print(w2vecmodel.wv.most_similar('covid', topn=10))
print(w2vecmodel.wv.most_similar('death', topn=10))
print(w2vecmodel.wv.most_similar('hypertension', topn=10))
Outcome = ["Death", "ICU Admission", 'Mechanical Ventilation', "Organ Failure", "Sepsis", "Discharge"]

      



death_synonyms = ['demise',

                  'fatal',

                 'mortal',

                 'critically ill']   



icu_admissions = ['icu admission','intensive care','care unit', 'requiring icu', 'icu length']

           

ventilator_synonyms = ['mechanical ventilation',

                       'respiratory failure',

                       'intubation',

                       'oxygen therapy',

                       'endotracheal intubation'

                       'respiratory distress',

                       'arterial oxygen',

                       'ventilatory support',

                       'ventilatory'

                      ]



organ_failure_synonyms = ['organ failure',

                          'organ dysfunction',

                          'renal failure',

                          'multiple organ',

                          'multiorgan failure',

                          'kidney injury',

                          'renal replacement',

                          'renal dysfunction']



sepsis_synonyms = ['sepsis','septic shock', 'refractory septic']



discharge_synonym = ['full recovery','discharge', 'recovery', 'recover within','recovered']
Diseases = ["Coronaviruses","ARDS","SARS", "MERS","Covid-19"]

Comorbidities = ["Diabetes","Hypertension","Immunodeficiency", "Cancer", "Respiratory", "Immunity"]

OtherRisks = ["Age", "Gender", "Body Weight", "Smoking", "Climate", "Transmission"]


covid19_synonyms = ['covid',

                    'coronavirus disease 19',

                    'sars cov 2', # Note that search function replaces '-' with ' '

                    '2019 ncov',

                    '2019ncov',

                    r'2019 n cov\b',

                    r'2019n cov\b',

                    'ncov 2019',

                    r'\bn cov 2019',

                    'coronavirus 2019',

                    '2019 novel coronavirus',

                    'wuhan pneumonia',

                    'wuhan virus',

                    'wuhan coronavirus',

                    r'coronavirus 2\b']



sars_synonyms = [r'\bsars\b',

                 'severe acute respiratory syndrome']



mers_synonyms = [r'\bmers\b',

                 'middle east respiratory syndrome']



corona_synonyms = ['corona', r'\bcov\b']



ards_synonyms = ['acute respiratory distress syndrome',

                 r'\bards\b']



diabetes_synonyms = [

    'diabet', # picks up diabetes, diabetic, etc.

    'insulin', # any paper mentioning insulin likely to be relevant

    'blood sugar',

    'blood glucose',

    'ketoacidosis',

    'hyperglycemi', # picks up hyperglycemia and hyperglycemic

]



hypertension_synonyms = [

    'hypertension',

    'blood pressure',

    r'\bhbp\b', # HBP = high blood pressure

    r'\bhtn\b' # HTN = hypertension

]

immunodeficiency_synonyms = [

    'immunodeficiency',

    r'\bhiv\b',

    r'\baids\b'

    'granulocyte deficiency',

    'hypogammaglobulinemia',

    'asplenia',

    'dysfunction of the spleen',

    'spleen dysfunction',

    'complement deficiency',

    'neutropenia',

    'neutropaenia', # alternate spelling

    'cell deficiency' # e.g. T cell deficiency, B cell deficiency

]



cancer_synonyms = [

    'cancer',

    'malignant tumour',

    'malignant tumor',

    'melanoma',

    'leukemia',

    'leukaemia',

    'chemotherapy',

    'radiotherapy',

    'radiation therapy',

    'lymphoma',

    'sarcoma',

    'carcinoma',

    'blastoma',

    'oncolog'

]



chronicresp_synonyms = [

    'chronic respiratory disease',

    'asthma',

    'chronic obstructive pulmonary disease',

    r'\bcopd',

    'chronic bronchitis',

    'emphysema'

]



immunity_synonyms = [

    'immunity',

    r'\bvaccin',

    'innoculat'

]



age_synonyms = ['median age',

                'mean age',

                'average age',

                'elderly',

                r'\baged\b',

                r'\bold',

                'young',

                'teenager',

                'adult',

                'child'

               ]



sex_synonyms = ['sex',

                'gender',

                r'\bmale\b',

                r'\bfemale\b',

                r'\bmales\b',

                r'\bfemales\b',

                r'\bmen\b',

                r'\bwomen\b'

               ]



bodyweight_synonyms = [

    'overweight',

    'over weight',

    'obese',

    'obesity',

    'bodyweight',

    'body weight',

    r'\bbmi\b',

    'body mass',

    'body fat',

    'bodyfat',

    'kilograms',

    r'\bkg\b', # e.g. 70 kg

    r'\dkg\b'  # e.g. 70kg

]



smoking_synonyms = ['smoking',

                    'smoke',

                    'cigar', # this picks up cigar, cigarette, e-cigarette, etc.

                    'nicotine',

                    'cannabis',

                    'marijuana'

]





climate_synonyms = [

    'climate',

    'weather',

    'humid',

    'sunlight',

    'air temperature',

    'meteorolog', # picks up meteorology, meteorological, meteorologist

    'climatolog', # as above

    'dry environment',

    'damp environment',

    'moist environment',

    'wet environment',

    'hot environment',

    'cold environment',

    'cool environment',

    'latitiude',

    'tropical'

]



transmission_synonyms = [

    'transmiss', # Picks up 'transmission' and 'transmissibility'

    'transmitted',

    'incubation',

    'environmental stability',

    'airborne',

    'via contact',

    'human to human',

    'through droplets',

    'through secretions',

    r'\broute',

    'exportation'

]

# Help Function to count number of occurences of a pattern in text, Needed for Keyword based analysis

def count_number_of_occurences(pattern,text) :

  return re.subn(pattern, '', text)[1]



def check_thematic_diseases(disease, n):

    if disease=="Covid-19" :

        keywords=covid19_synonyms

    elif disease=="SARS" :

        keywords=sars_synonyms

    elif disease=="MERS" :    

        keywords=mers_synonyms

    elif disease=="Coronaviruses" : 

        keywords=corona_synonyms

    elif disease=="ARDS" :

        keywords=ards_synonyms 

    

    for dis in keywords:

        #Exact Keyword Check, may need to put a threshold of number of occurences

        status = (pd.Series(get_title(n).lower()).str.contains(dis , na=False) |

                     pd.Series(get_text_body(n).lower()).str.contains(dis, na=False))

        #Fuzzy Check TODO

        #Word2Vec Check TODO

        if status.bool() :

            break

    return status.bool()



def check_thematic_comorbidities(com, n):

    if com=="Diabetes" :

        keywords=diabetes_synonyms

    elif com=="Hypertension" :

        keywords=hypertension_synonyms

    elif com=="Immunodeficiency" :    

        keywords=immunodeficiency_synonyms

    elif com=="Cancer" : 

        keywords=cancer_synonyms

    elif com=="Respiratory" :

        keywords=chronicresp_synonyms 

    elif com=="Immunity" :

        keywords=immunity_synonyms    

    

    for como in keywords:

        #Exact Keyword Check, may need to put a threshold of number of occurences is results in a larger size

        status = (pd.Series(get_title(n).lower()).str.contains(como , na=False) |

                     pd.Series(get_text_body(n).lower()).str.contains(como, na=False))

        if status.bool() :

            break

    return status.bool()



def check_otherrisk_factors(com, n):

    if com=="Age" :

        keywords=age_synonyms

    elif com=="Gender" :

        keywords=sex_synonyms

    elif com=="Body Weight" :    

        keywords=bodyweight_synonyms

    elif com=="Smoking" : 

        keywords=smoking_synonyms

    elif com=="Climate" :

        keywords=climate_synonyms 

    elif com=="Transmission" :

        keywords=transmission_synonyms    

    

    for risk in keywords:

        #Exact Keyword Check, may need to put a threshold of number of occurences is results in a larger size

        status = (pd.Series(get_title(n).lower()).str.contains(risk , na=False) |

                     pd.Series(get_text_body(n).lower()).str.contains(risk, na=False))

        if status.bool() :

            break

    return status.bool()



def check_outcomes(out, n):

    if out=="Death" :

        keywords=death_synonyms

    elif out=="ICU Admission" :

        keywords=icu_admissions

    elif out=="Mechanical Ventilation" :    

        keywords=ventilator_synonyms

    elif out=="Organ Failure" : 

        keywords=organ_failure_synonyms

    elif out=="Sepsis" :

        keywords=sepsis_synonyms

    elif out=="Discharge" :

        keywords=discharge_synonym   

    

    for risk in keywords:

        #Exact Keyword Check, may need to put a threshold of number of occurences is results in a larger size

        status = (pd.Series(get_title(n).lower()).str.contains(risk , na=False) |

                     pd.Series(get_text_body(n).lower()).str.contains(risk, na=False))

        if status.bool() :

            break

    return status.bool()



print("Sub Keywords and Help Functions defined for Outcomes, Diseases , Comorbidities & Other Risk Factors ")
# To be extended as per top impacted areas in each geoghraphy

continental_regions = {

    'asia': 'asia|china|korea|japan|hubei|wuhan|malaysia|singapore|hong kong',

    'east_asia': 'east asia|china|korea|japan|hubei|wuhan|hong kong',

    'south_asia': 'south asia|india|pakistan|bangladesh|sri lanka',

    'se_asia': r'south east asia|\bse asia|malaysia|thailand|indonesia|vietnam|cambodia|viet nam',

    'europe': 'europe|italy|france|spain|germany|austria|switzerland|united kingdom|ireland',

    'africa': 'africa|kenya',

    'middle_east': 'middle east|gulf states|saudi arabia|\buae\b|iran|persian',

    'south_america': 'south america|latin america|brazil|argentina',

    'north_america': 'north america|usa|united states|canada|caribbean',

    'australasia': 'australia|new zealand|oceania|australasia|south pacific'

}



# Tag the Primary Geography for the study, based on number of occurences

def tag_primary_study_geography(text):

    score = []

    for cr, s in continental_regions.items():

        count=0

        splits = s.split('|')

        for reg in splits:

            count+= count_number_of_occurences(reg,text)

        score.append(count)

    if ((len(set(score)) == 1) & (score[0]==0)):

        tag = "unknown"

    else :

        tag = list(continental_regions.keys())[score.index(max(score))]

    return tag



df_covid['region'] = df_covid['body_text'].apply(lambda x: tag_primary_study_geography(x.lower())) 



print("Geographical Tagging completed on the Dataframe ")
df_covid['abstract'].replace(np.nan, 'Not Available', regex=True, inplace=True)

research_method = {

    'Systematic Review':'cohen\'s d|cohen\'s kappa|cochrane review|database search|databases searched|difference between means|d-pooled|difference in means|electronic search|heterogeneity|pooled relative risk|meta-analysis|pooled adjusted odds ratio|pooled aor|pooled odds ratio|pooled or|pooled risk ratio|pooled rr|prisma|search criteria|search strategy|search string|systematic review',

    'Randomized':'blind|consort|control arm|double-blind|placebo|randomisation|randomised|randomization method|randomized|randomized clinical trial|randomized controlled trial|rct|treatment arm|treatment effect',

    'Non-Randomized':'allocation method|blind|control arm|double-blind|non-randomised|non-randomized|non randomized|placebo|pseudo-randomised|pseudo-randomized|quasi-randomised|quasi-randomized|treatment arm|treatment effect',

    'Ecological Regression':'correlation|correlations|per capita|r-squared|adjusted hazard ratio|censoring|confounding|covariates|cox proportional hazards|demographics|enroll|enrolled|enrollment|eligibility criteria|etiology|gamma|hazard ratio|kaplan-meier|lognormal|longitudinal|median time to event|non-comparative study|potential confounders|recruit|recruited|recruitment|right-censored|survival analysis|time-to-event analysis|time series|time-series|time varying|time-varying|truncated|weibull',

    'Prospective Cohort': 'baseline|prospective|prospectively|prospective cohort|relative risk|risk ratio|rr|chart review|ehr|health records|medical records|etiology|exposure status|risk factor analysis|risk factors|cohort|followed|loss to follow-up|patients|subjects|adjusted odds ratio|aor|log odds|logistic regression|odds ratio',

    'Time Series Analysis': 'adjusted hazard ratio|censoring|confounding|covariates|cox proportional hazards|demographics|enroll|enrolled|enrollment|eligibility criteria|etiology|gamma|hazard ratio|kaplan-meier|lognormal|longitudinal|median time to event|non-comparative study|potential confounders|recruit|recruited|recruitment|right-censored|survival analysis|time-to-event analysis|time series|time-series|time varying|time-varying|truncated|weibull',

    'Retrospective Cohort' : 'cohen\'s kappa|data abstraction forms|data collection instrument|eligibility criteria|inter-rater reliability|potential confounders|retrospective|retrospective chart review|retrospective cohort|chart review|ehr|health records|medical records|etiology|exposure status|risk factor analysis|risk factors|cohort|followed|loss to follow-up|patients|subjects|adjusted odds ratio|aor|log odds|logistic regression|odds ratio',

    'Cross Sectional' :'cross sectional|cross-sectional|prevalence survey|case-control|data collection instrument|eligibility criteria|matching case|matched case|matching criteria|matched criteria|number of controls per case|non-response bias|potential confounders|psychometric evaluation of instrument|questionnaire development|response rate|survey instrument|chart review|ehr|health records|medical records|etiology|exposure status|risk factor analysis|risk factors|adjusted odds ratio|aor|log odds|logistic regression|odds ratio',

    'Case Control':'case-control|data collection instrument|eligibility criteria|matching case|matched case|matching criteria|matched criteria|number of controls per case|non-response bias|potential confounders|psychometric evaluation of instrument|questionnaire development|response rate|survey instrument|chart review|ehr|health records|medical records|etiology|exposure status|risk factor analysis|risk factors|adjusted odds ratio|aor|log odds|logistic regression|odds ratio',

    'Case Study': 'case report|case series|etiology|frequency|risk factors',

    'Simulation': 'bootstrap|computer model|computer modelling|forecast|forcasting|mathematical model|mathematical modelling|model simulation|monte carlo|simulate|simulation|simulated|synthetic data|synthetic dataset|cohort|followed|loss to follow-up|patients|subjects'

}



# Tag the Primary Research Method for the study, based on number of occurences

def tag_primary_research_method(text):

    score = []

    for cr, s in research_method.items():

        count=0

        splits = s.split('|')

        for reg in splits:

            count+= count_number_of_occurences(reg,text)

        score.append(count)

    if ((len(set(score)) == 1) & (score[0]==0)):

        tag = "unknown"

    else :

        tag = list(research_method.keys())[score.index(max(score))]

    return tag



df_covid['researchdesign']=""



df_covid['researchdesign'] = df_covid['title'].apply(lambda x: tag_primary_research_method(x))

df_covid['researchdesign'] = df_covid[df_covid['researchdesign']=='unknown']['abstract'].apply(lambda x: tag_primary_research_method(x))



df_covid['researchdesign'].replace(np.nan, 'unknown', regex=True, inplace=True)



print("Research Study Tags completed for all papers in the Dataframe")
import random

def evidencegapmap(dataset, x_column, y_column, xy_column=None, bubble_column=None, bubble_text=None, bubble_link=None, time_column=None, size_column=None, color_column=None,   

               xbin_list=None, ybin_list=None,xbin_size=100, ybin_size=100, x_title=None, y_title=None, title=None, colorbar_title=None,

               scale_bubble=10, colorscale=None, marker_opacity=None, marker_border_width=None,show_slider=True, show_button=True, show_colorbar=True, show_legend=None, 

               width=None, height=None):

    ''' Makes the animated and interactive bubble charts from a given dataset.'''

    

    # Initialize the number of bins 

    xbin_range = [0,(len(xbin_list)-1)]

    ybin_range = [0,(len(ybin_list)-1)]

    #Initialize Axes range                                  

    x_range=[0,0] 

    y_range=[0,0]

    # Set category_column as None and update it as color_column only in case

    # color_column is not None and categorical, in which case set color_column as None

    category_column = None

    if color_column: # Can be numerical or categorical

        if dataset[color_column].dtype.name in ['category', 'object', 'bool']:

            category_column = color_column

            color_column = None

    # Set the plotting mode for the plots inside a cell

    if xy_column :

        mode = 'nlpmode'

        xmax = max(map(lambda xy: xy[0], list(dataset[xy_column])))

        xmin = min(map(lambda xy: xy[0], list(dataset[xy_column])))

        ymax = max(map(lambda xy: xy[1], list(dataset[xy_column])))

        ymin = min(map(lambda xy: xy[1], list(dataset[xy_column])))

        xshift = (xmax + xmin)/2

        yshift = (ymax + ymin)/2

        xy_scale= max(xmax-xmin, ymax-ymin)

        #print("xmax {}, xmin {}, ymax {}, ymin {}, xshift {}, yshift {} xy_scale {}".format(xmax, xmin, ymax, ymin, xshift, yshift, xy_scale))

    else :

        mode = 'randommode'

        xy_scale = 1

        xshift=yshift =0

    

    # Set the variables for making the grid

    if time_column:

        years = dataset[time_column].unique()

    else:

        years = None

        show_slider = False

        show_button = False

        

    column_names = [x_column, y_column]

    

    column_names.append(bubble_column)

    if xy_column:

        column_names.append(xy_column)

    if bubble_text:

        column_names.append(bubble_text)

    if bubble_link:

        column_names.append(bubble_link)

    

    if size_column:

        column_names.append(size_column)

    

    if color_column:

        column_names.append(color_column)

        

        

    # Make the grid

    if category_column:

        categories = dataset[category_column].unique()

        col_name_template = '{}+{}+{}_grid'

        grid = make_grid_with_categories(dataset, column_names, time_column, category_column, years, categories)

        if show_legend is None:

            showlegend = True

        else: 

            showlegend = show_legend



        

    # Set the layout

    if show_slider:

        slider_scale = years

    else:

        slider_scale = None

                

    figure, sliders_dict = set_layout(x_title, y_title, title, show_slider, slider_scale, show_button, showlegend, width, height)

    

    if size_column:

        sizeref = 2.*max(dataset[size_column])/(scale_bubble**2) # Set the reference size for the bubbles

    else:

        sizeref = None



    # Add the frames

    if category_column:

        # Add the base frame

        for category in categories:

            if time_column:

                year = min(years) # The earliest year for the base frame

                col_name_template_year = col_name_template.format(year, {}, {})

            else:

                col_name_template_year = '{}+{}_grid'

            trace = get_trace(grid, col_name_template_year, x_column, y_column, xy_column, 

                              bubble_column,bubble_text, bubble_link, size_column, 

                              sizeref, scale_bubble, marker_opacity, marker_border_width, mode=mode,category=category, xsize=xbin_size, ysize=ybin_size,

                              xy_scale=xy_scale, xshift=xshift, yshift=yshift)

            figure['data'].append(trace)

           

        # Add time frames

        if time_column: # Only if time_column is not None

            for year in years:

                frame = {'data': [], 'name': str(year)}

                for category in categories:

                    col_name_template_year = col_name_template.format(year, {}, {})

                    trace = get_trace(grid, col_name_template_year, x_column, y_column, xy_column, 

                                      bubble_column, bubble_text, bubble_link, size_column, 

                                      sizeref, scale_bubble, marker_opacity, marker_border_width ,mode=mode, category=category, xsize=xbin_size, ysize=ybin_size,

                                      xy_scale=xy_scale, xshift=xshift, yshift=yshift)

                    

                    frame['data'].append(trace)



                    figure['frames'].append(frame) 



                if show_slider:

                    add_slider_steps(sliders_dict, year)

                

    else:

        # Add the base frame

        if time_column:

            year = min(years) # The earliest year for the base frame

            col_name_template_year = col_name_template.format(year, {})

        else:

            col_name_template_year = '{}_grid'

        trace = get_trace(grid, col_name_template_year, x_column, y_column, xy_column, 

                          bubble_column, bubble_text, bubble_link, size_column, 

                          sizeref, scale_bubble, marker_opacity, marker_border_width,

                          color_column, colorscale, show_colorbar, colorbar_title, mode=mode, xsize=xbin_size, ysize=ybin_size,

                          xy_scale=xy_scale, xshift=xshift, yshift=yshift)

       

        figure['data'].append(trace)

        

        # Add time frames

        if time_column: # Only if time_column is not None

            for year in years:

                col_name_template_year = col_name_template.format(year, {})

                frame = {'data': [], 'name': str(year)}

                trace = get_trace(grid, col_name_template_year, x_column, y_column, xy_column,

                                  bubble_column, bubble_text, bubble_link,size_column, 

                                  sizeref, scale_bubble, marker_opacity, marker_border_width,

                                  color_column, colorscale, show_colorbar, colorbar_title, mode=mode, xsize=xbin_size, ysize=ybin_size, 

                                  xy_scale=xy_scale, xshift=xshift, yshift=yshift)



                frame['data'].append(trace)

                figure['frames'].append(frame) 

                if show_slider:

                    add_slider_steps(sliders_dict, year) 

    # Set ranges for the axes

   

    x_range = set_range(dataset[x_column], xbin_size)

    y_range = set_range(dataset[y_column], ybin_size)

    

    figure['layout']['xaxis']['range'] = x_range

    figure['layout']['yaxis']['range'] = y_range

        

    if show_slider:

        figure['layout']['sliders'] = [sliders_dict]

    

    tracepoint = draw_evidence_gap_map_structure_horzero(xbin_list,ybin_list,xbin_size,ybin_size )

    figure['data'].append(tracepoint)

    for i in range(len(ybin_list)+1): 

        tracepoint = draw_evidence_gap_map_structure_hor(i, xbin_list,ybin_list,xbin_size,ybin_size )

        figure['data'].append(tracepoint)

    tracepoint = draw_evidence_gap_map_structure_verzero(xbin_list,ybin_list,xbin_size,ybin_size )

    figure['data'].append(tracepoint)

    for i in range(len(xbin_list)+1): 

        tracepoint = draw_evidence_gap_map_structure_ver(i, xbin_list,ybin_list,xbin_size,ybin_size )

        figure['data'].append(tracepoint)

    return figure



def draw_evidence_gap_map_structure_horzero(x_list=None, y_list=None,xbin=100, ybin=100):

    number_of_xcats = len(x_list)

    number_of_ycats = len(y_list)

    draw_horizontals_zero= {

        'x': [int((xbin/2)+i*(xbin)) for i in range(number_of_xcats)],

        'y': [0 for i in range(number_of_xcats)],

        'text': [x_list[line] for line in range(number_of_xcats)],

        'mode': 'lines+text',

        'textposition': 'bottom center',

        'showlegend': False

    }

    return draw_horizontals_zero

def draw_evidence_gap_map_structure_hor(linenum=1, x_list=None, y_list=None,xbin=100, ybin=100):

    number_of_xcats = len(x_list)

    number_of_ycats = len(y_list)

    draw_horizontals = {

        'x': [int(i*xbin) for i in range(number_of_xcats+1)],

        'y': [int(linenum*(ybin)) for i in range(number_of_xcats+1)],

        'text': "",

        'mode': 'lines',

        'showlegend': False

    }

    return draw_horizontals

def draw_evidence_gap_map_structure_verzero(x_list=None, y_list=None,xbin=100, ybin=100):

    number_of_xcats = len(x_list)

    number_of_ycats = len(y_list)

    draw_verticals_zero= {

        'x': [0 for i in range(number_of_ycats)],

        'y': [int((ybin/2)+i*(ybin)) for i in range(number_of_ycats)],

        'text': [y_list[line] for line in range(number_of_ycats)],

        'mode': 'lines+text',

        'textposition': 'middle left',

        'showlegend': False

    }

    return draw_verticals_zero

def draw_evidence_gap_map_structure_ver(linenum=1, x_list=None, y_list=None,xbin=100, ybin=100):

    number_of_xcats = len(x_list)

    number_of_ycats = len(y_list)

    draw_verticals = {

        'x': [int(linenum*(xbin)) for i in range(number_of_ycats+1)],

        'y': [int(i*ybin) for i in range(number_of_ycats+1)],

        'text': "",

        'mode': 'lines',

        'showlegend': False

    }

    return draw_verticals

    

def make_grid_with_categories(dataset, column_names, time_column, category_column, years=None, categories=None):

    '''Makes the grid for the plot as a pandas DataFrame.'''

    

    grid = pd.DataFrame()

    if categories is None:

        categories = dataset[category_column].unique()

    if time_column:

        col_name_template = '{}+{}+{}_grid'

        if years is None:

            years = dataset[time_column].unique()

            

        for year in years:

            for category in categories:

                dataset_by_year_and_cat = dataset[(dataset[time_column] == int(year)) & (dataset[category_column] == category)]

                for col_name in column_names:

                    # Each column name is unique

                    temp = col_name_template.format(year, col_name, category)

                    if dataset_by_year_and_cat[col_name].size != 0:

                        grid = grid.append({'value': list(dataset_by_year_and_cat[col_name]), 'key': temp}, ignore_index=True) 

    else:

        col_name_template = '{}+{}_grid'

        for category in categories:

            dataset_by_cat = dataset[(dataset[category_column] == category)]

            for col_name in column_names:

                # Each column name is unique

                temp = col_name_template.format(col_name, category)

                if dataset_by_cat[col_name].size != 0:

                        grid = grid.append({'value': list(dataset_by_cat[col_name]), 'key': temp}, ignore_index=True) 

    return grid



 

def set_layout(x_title=None, y_title=None, title=None, show_slider=True, slider_scale=None, show_button=True, show_legend=False,

            width=None, height=None):

    '''Sets the layout for the figure.'''

    

    # Define the figure object as a dictionary

    figure = {

        'data': [],

        'layout': {},

        'frames': []

    }

    

    # Start with filling the layout first

    

    figure = set_2Daxes(figure, x_title, y_title)

        

    figure['layout']['title'] = title    

    figure['layout']['hovermode'] = 'closest'

    figure['layout']['showlegend'] = show_legend

    figure['layout']['margin'] = dict(l=60, b=50, t=50, r=60, pad=10)

    

    

    if width:

        figure['layout']['width'] = width

    if height:

        figure['layout']['height'] = height

    

    # Add slider for the time scale

    if show_slider: 

        sliders_dict = add_slider(figure, slider_scale)

    else:

        sliders_dict = {}

    

    # Add a pause-play button

    if show_button:

        add_button(figure)

        

    # Return the figure object

    return figure, sliders_dict



def set_2Daxes(figure, x_title=None, y_title=None):

    '''Sets 2D axes'''

    

    figure['layout']['xaxis'] = {'title': x_title, 'autorange': False, 'showgrid': False, 'zeroline': False, 'showline': False, 'ticks': '',

    'showticklabels': False, 'automargin': True}

    figure['layout']['yaxis'] = {'title': y_title, 'autorange': False, 'showgrid': False, 'zeroline': False, 'showline': False, 'ticks': '',

    'showticklabels': False, 'automargin': True} 

        

    return figure

    

        

def add_slider(figure, slider_scale):

    '''Adds slider for animation'''

    

    figure['layout']['sliders'] = {

        'args': [

            'slider.value', {

                'duration': 400,

                'ease': 'cubic-in-out'

            }

        ],

        'initialValue': min(slider_scale),

        'plotlycommand': 'animate',

        'values': slider_scale,

        'visible': True

    }

    

    sliders_dict = {

        'active': 0,

        'yanchor': 'top',

        'xanchor': 'left',

        'currentvalue': {

            'font': {'size': 20},

            'prefix': 'Year:',

            'visible': True,

            'xanchor': 'right'

        },

        'transition': {'duration': 300, 'easing': 'cubic-in-out'},

        'pad': {'b': 10, 't': 50},

        'len': 0.9,

        'x': 0.1,

        'y': 0,

        'steps': []

    }

    

    return sliders_dict



def add_slider_steps(sliders_dict, year):

    '''Adds the slider steps.'''

    

    slider_step = {'args': [

        [year],

        {'frame': {'duration': 300, 'redraw': False},

         'mode': 'immediate',

       'transition': {'duration': 300}}

     ],

     'label': str(year),

     'method': 'animate'}

    sliders_dict['steps'].append(slider_step)

    

def add_button(figure):

    '''Adds the pause-play button for animation'''

    

    figure['layout']['updatemenus'] = [

        {

            'buttons': [

                {

                    'args': [None, {'frame': {'duration': 500, 'redraw': False},

                             'fromcurrent': True, 'transition': {'duration': 300, 'easing': 'quadratic-in-out'}}],

                    'label': 'Play',

                    'method': 'animate'

                },

                {

                    'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',

                    'transition': {'duration': 0}}],

                    'label': 'Pause',

                    'method': 'animate'

                }

            ],

            'direction': 'left',

            'pad': {'r': 10, 't': 87},

            'showactive': False,

            'type': 'buttons',

            'x': 0.1,

            'xanchor': 'right',

            'y': 0,

            'yanchor': 'top'

        }

    ]

    

def set_range(values, size): 

    ''' Finds the axis range for the figure.'''

    

    rmin = int(min([return_xbin_cords(x, size) for x in values]))-size/2

    rmax = int(max([return_xbin_cords(x, size) for x in values]))+size/2

    

        

    return [rmin, rmax] 



# To be used later when individual Risk Factos can be plotted

def return_xbin_cords(x_binnum, sizebin):

    # generate some random integers to fit in the research papers in a cell

    values = random.randint((-sizebin/2+5),(sizebin/2-5))

    #Plots start at (0, 0)

    xbin_cords = sizebin/2 + (x_binnum*sizebin) + values

    return int(xbin_cords)



# To be used later when individual Risk Factos can be plotted

def return_ybin_cords(y_binnum, sizebin):

    # generate some random integers to fit in the research papers in a cell

    values = random.randint((-sizebin/2+5),sizebin/2-5)

    #Plots start at (0, 0)

    ybin_cords = sizebin/2 + (y_binnum*sizebin) + values

    return int(ybin_cords)



# To be used later when individual Risk Factos can be plotted

def return_xy_cords_nlp(a, xy, sizebin, axes, scale, shift):

    if axes=='x':

        margin = 10

        # generate some random integers to fit in the research papers in a cell

        # remove a margin of 10 from the size of bin so effectively available size is 90 if bin is 100

        values = ((xy[0]-shift)/scale)*(sizebin - 10)

        #Plots start at (0, 0)

        x_cords = sizebin/2 + (a*sizebin) + values

        return int(x_cords)

    else :

        # generate some random integers to fit in the research papers in a cell

        # remove a margin of 10 from the size of bin so effectively available size is 90 if bin is 100

        values = ((xy[1]-shift)/scale)*(sizebin - 10)

        #Plots start at (0, 0)

        y_cords = sizebin/2 + (a*sizebin) + values

        return int(y_cords)

    

def return_text_by_category_in_bin(grid,category,xbinnum,ybinnum,template, xcol, ycol, column, bubbletext, link, size):

    indicesx=[]

    indicesy=[]

    for idx, row in grid[grid['key'].str.contains(category)].iterrows():

        if row['key']==template.format(xcol, category):

            for i, xx in enumerate(row['value']):

                if (xx==xbinnum):

                    indicesx.append(i)

        if row['key']==template.format(ycol, category):

            for i, yy in enumerate(row['value']):

                if (yy==ybinnum):

                    indicesy.append(i) 

    matchindex = list(set(indicesx) & set(indicesy))

    textoverall=[]

    textcol=[]

    texttext=[]

    textlink=[]

    textrelevance=[]

    for idx, row in grid[grid['key'].str.contains(category)].iterrows():

        for i, val in enumerate(matchindex):

            if row['key']==template.format(column, category):

                textcol.append('<b>Title:</b>'+ str(row['value'][val]))

            if bubbletext:

                if row['key']==template.format(bubbletext, category):

                    texttext.append('<br><b>Summary:</b>'+ str(row['value'][val]))

            if link:

                if row['key']==template.format(link, category):

                    textlink.append('<br><b>Link:</b>'+ str(row['value'][val]))   

            if size:

                if row['key']==template.format(size, category):

                    textrelevance.append('<br><b>Relevance:</b>'+ str(row['value'][val]))

    for idx, val in enumerate(textcol):

        # Display top 8 of relevant  and the highlighted 

        if idx==0:

            textall = ""

        else: 

            textall ='<br>----------------------------------------<br>'

        textall = textall + textcol[idx]

        if bubbletext:

            textall = textall + texttext[idx]

        if link:

            textall = textall + textlink[idx] 

        if size:

            textall = textall + textrelevance[idx]

        textoverall.append(textall)

        # Plotly only able to handle only upto 9 datapoints in hovertext

        # TODO ensure that the closest point being hovered is always included

        if idx==8 :

            break

    return "".join(textoverall)    



# The size is used to categorize in High (top 10% percentile), Medium ( to 50% ) and Rest as Low

def return_transformed_size(size, comparewith):

    if size > np.percentile(comparewith, 90):

        return size*1.25

    elif size > np.percentile(comparewith, 50):

        return size

    else :

        return size/1.25

    

def get_trace(grid, col_name_template, x_column, y_column,xy_column, bubble_column, bubble_text, bubble_link,size_column=None, 

            sizeref=1, scale_bubble=10, marker_opacity=None, marker_border_width=None,

            color_column=None, colorscale=None, show_colorbar=True, colorbar_title=None, mode=None, category=None, xsize=100, ysize=100, 

            xy_scale=1, xshift=0, yshift=0):

    ''' Makes the trace for the data as a dictionary object that can be added to the figure or time frames.'''

    try:

        if mode =='randommode':

            trace = {

                    'x': [return_xbin_cords(x, xsize) for x in grid.loc[grid['key']==col_name_template.format(x_column, category), 'value'].values[0]],

                    'y': [return_ybin_cords(y, ysize) for y in grid.loc[grid['key']==col_name_template.format(y_column, category), 'value'].values[0]],

                    'text': [i + '<br><b>Summary:</b>' + j + '<br><b>Link:</b>' + k for i, j, k in zip(grid.loc[grid['key']==col_name_template.format(bubble_column, category), 'value'].values[0], grid.loc[grid['key']==col_name_template.format(bubble_text, category), 'value'].values[0],grid.loc[grid['key']==col_name_template.format(bubble_link, category), 'value'].values[0])],

                    'hovertemplate': '<b>Title:</b>%{text}<extra></extra>',

                    'mode': 'markers'

            }

        else:

            trace = {

                    'x': [return_xy_cords_nlp(x,xy, xsize, 'x', xy_scale, xshift) for x, xy in zip(grid.loc[grid['key']==col_name_template.format(x_column, category), 'value'].values[0],grid.loc[grid['key']==col_name_template.format(xy_column, category), 'value'].values[0])],

                    'y': [return_xy_cords_nlp(y,xy, ysize, 'y', xy_scale, yshift) for y, xy in zip(grid.loc[grid['key']==col_name_template.format(y_column, category), 'value'].values[0],grid.loc[grid['key']==col_name_template.format(xy_column, category), 'value'].values[0])],

                    'text': [return_text_by_category_in_bin(grid,category,x,y,col_name_template,x_column,y_column,bubble_column,bubble_text,bubble_link,size_column) for x, y  in zip(grid.loc[grid['key']==col_name_template.format(x_column,category), 'value'].values[0],grid.loc[grid['key']==col_name_template.format(y_column, category), 'value'].values[0])],

                    'hovertemplate': '%{text}<extra></extra>',

                    'mode': 'markers'

            }

        if size_column:

                trace['marker'] = {

                    'sizemode': 'diameter',

                    'sizeref': sizeref,

                    'size': [return_transformed_size(size, grid.loc[grid['key']==col_name_template.format(size_column, category), 'value'].values[0]) 

                             for size in grid.loc[grid['key']==col_name_template.format(size_column, category), 'value'].values[0]],

                }

        else:

                trace['marker'] = {

                    'size': 10*scale_bubble,

                }



        if marker_opacity:

                trace['marker']['opacity'] = marker_opacity



        if marker_border_width:

                trace['marker']['line'] = {'width': marker_border_width}



        if color_column:

                    trace['marker']['color'] = grid.loc[grid['key']==col_name_template.format(color_column), 'value'].values[0]

                    trace['marker']['colorbar'] = {'title': colorbar_title}

                    trace['marker']['colorscale'] = colorscale



        if category:

                trace['name'] = category

    except:

        trace = {

            'x': [],

            'y': [],

            }



    return trace

task2 = ['Data on potential risks factors for COVID 19, Wuhan Coronaviruses, sars cov 2, ncov 2019, coronavirus 2019, wuhan pneumoni ',

'Smoking, pre-existing pulmonary disease',

'Co-infections (determine whether co-existing respiratory, viral infections make the virus more transmissible or virulent) and other co-morbidities like hypertension and diabetes',

'Neonates and pregnant women',

'Socio-economic and behavioral factors to understand the economic impact of the virus and whether there were differences.',

'Transmission dynamics of the virus, including the basic reproductive number, incubation period, serial interval, modes of transmission and environmental factors', 

'Severity of disease, including risk of fatality among symptomatic hospitalized patients, and high-risk patient groups',

'Susceptibility of populations',

'Public health mitigation measures that could be effective for control', 

'Studies that cover risk factor analysis,cross sectional case control,prospective case control,matched case control, medical records review, seroprevalence survey and syndromic surveillance',

 'health status (diabetes, hypertension, heart disease, pregnancy, neonates, cancer, smoking status, history of lung disease, local climate, elderly, small children, immune compromised groups, age deciles among adults between the ages of 15 and 65, race/ethnicity, insurance status, housing status).',

 'latitude, temperature, humidity. Covariates include: social distancing policies, population density, demographics (e.g., socioeconomic status, access to health services), access to testing.',

 'symptoms (cough, fever, sputum production, diarrhea, shortness of breath, sleep disruption, fatigue, etc.) and lab results (COVID-19 by PCR, chest CT scan, leucocyte counts, neutrophils, lymphocytes, hemoglobin, platelets, liver function abnormality, alanine aminotransferase (ALT) , aspartate aminotransferase (AST), lactate dehydrogenase, renal function damage, blood urea nitrogen, serum creatinine, procalcitonin (PCT), IL-6, C-reactive protein (CRP)',

 'social distancing directives, postponing nonessential medical services',

 'mandatory quarantine of exposed health workers',

 'COPD, COPD severity,GOLD score, FEV1/FVC, FEV1 % of normal']
task2processed = gensim.parsing.preprocess_string(' '.join(task2))

task2vector = doc2vecmodel.infer_vector(task2processed)

# Print the Doc2Vec vector for the Research Objectives query

print(task2vector)
shortlistdocs = 300
# Return the top matching documents for the risk task

similar_docs = doc2vecmodel.docvecs.most_similar([task2vector], topn=shortlistdocs)

# Find similar doc and convert back to doc_id from index and print top 10 relevant research

count=0

for i, score in similar_docs:

    print("Title: {0} \n Paper Id: {1} \n Score: {2} \n".format(get_title(i),get_docid(i),score))

    count+=1

    if count==5:

        break
#Build the dataframe for the plot

plot_data_dis = pd.DataFrame(columns=['doc_id','title','publish_year','doi','abstract','region','x','y','xy_column','relevance', 'researchdesign'])

# Load from CSV does not work in NLP mode and hence regenerate dataframe

#if generatefiles:

newi=0

for i, scoreofdoc in similar_docs:

    for dis in Diseases:

        for out in Outcome:

            if (check_outcomes(out, i) & check_thematic_diseases(dis, i)) :

                plot_data_dis.loc[newi] = ""

                plot_data_dis['doc_id'][newi] = df_covid['paper_id'][i]

                plot_data_dis['title'][newi] = df_covid['title_summary'][i]

                plot_data_dis['publish_year'][newi] = int(df_covid['publish_year'][i])

                plot_data_dis['doi'][newi] = df_covid['doi'][i]

                plot_data_dis['abstract'][newi] = df_covid['abstract_summary'][i]

                plot_data_dis['region'][newi] = df_covid['region'][i]

                plot_data_dis['x'][newi] = Outcome.index(out)

                plot_data_dis['y'][newi] = Diseases.index(dis)

                plot_data_dis['xy_column'][newi] = get_xy_cordinates(i)

                plot_data_dis['relevance'][newi]= int(scoreofdoc*100)

                plot_data_dis['researchdesign'][newi]=df_covid['researchdesign'][i]

                newi+=1

# Make format changes to data column for display 

convert_dict = {'relevance': int,'publish_year': int}

plot_data_dis = plot_data_dis.astype(convert_dict)

#plot_data_dis.to_csv('disease-outcome.csv',index=False)
from __future__ import division

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

# Time based plot for Disease vs Outcome

plot_data_dis.sort_values('publish_year', inplace=True)

plot_data_dis.reset_index(drop=True, inplace=True)



figure = evidencegapmap(dataset=plot_data_dis, x_column='x', y_column='y',xy_column='xy_column',

  bubble_column='title', bubble_link='doi', time_column='publish_year', size_column='relevance', color_column='region',xbin_list=Outcome, ybin_list = Diseases,

  xbin_size=100, ybin_size = 100, x_title="Outcome", y_title="Disease", title='Disease vs Outcome (Year wise, Mode: NLP))',scale_bubble=5, marker_opacity=0.5,height=600, width=900)

iplot(figure)
#Clean Up as plotting is RAM consuming

del plot_data_dis

gc.collect()
shortlistdocs = 600
similar_docs_otherrisks = doc2vecmodel.docvecs.most_similar([task2vector], topn=shortlistdocs)

# Plot the evidence gap map for Outcomes vs OtherRisk Factors for COVID 19

plot_data_otherrisks = pd.DataFrame(columns=['doc_id','title','publish_year','publish_date','journal','doi','abstract','region','x','y','xy_column','relevance', 'researchdesign', 'otherrisks', 'outcome'])

newi=0

for i, scoreofdoc in similar_docs_otherrisks:

    for com in OtherRisks:

        for out in Outcome:

            if (check_otherrisk_factors(com, i) & check_outcomes(out, i)):

                plot_data_otherrisks.loc[newi] = ""

                plot_data_otherrisks['doc_id'][newi] = df_covid['paper_id'][i]

                plot_data_otherrisks['publish_date'][newi] = df_covid['publish_date'][i]

                #plot_data_otherrisks['authors'][newi] = df_covid['authors'][i]

                plot_data_otherrisks['journal'][newi] = df_covid['journal'][i]

                plot_data_otherrisks['title'][newi] = df_covid['title_summary'][i]

                plot_data_otherrisks['publish_year'][newi] = int(df_covid['publish_year'][i])

                plot_data_otherrisks['doi'][newi] = df_covid['doi'][i]

                plot_data_otherrisks['abstract'][newi] = df_covid['abstract_summary'][i]

                plot_data_otherrisks['region'][newi] = df_covid['region'][i]

                plot_data_otherrisks['x'][newi] = OtherRisks.index(com)

                plot_data_otherrisks['y'][newi] = Outcome.index(out)

                plot_data_otherrisks['xy_column'][newi] = get_xy_cordinates(i)

                plot_data_otherrisks['relevance'][newi]= int(scoreofdoc*100)

                plot_data_otherrisks['researchdesign'][newi]=df_covid['researchdesign'][i]

                plot_data_otherrisks['otherrisks'][newi] = com

                plot_data_otherrisks['outcome'][newi]= out

                newi+=1

# Make format changes to data column for display 

convert_dict = {'relevance': int,'publish_year': int}

plot_data_otherrisks = plot_data_otherrisks.astype(convert_dict)

#plot_data_otherrisks.to_csv('outcome-otherrisks.csv',index=False)   
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()

figure = evidencegapmap(dataset=plot_data_otherrisks, x_column='x', y_column='y',

  bubble_column='title',bubble_text='abstract', bubble_link='doi', size_column='relevance', color_column='researchdesign',xbin_list=OtherRisks, ybin_list = Outcome,

  xbin_size=100, ybin_size = 100, x_title="Other Risks", y_title="Outcomes", title='Outcome vs Other Risk Factors (Mode:Random)',scale_bubble=4, marker_opacity=0.6,height=800)

iplot(figure)
resultsother = pd.DataFrame(columns=['Date','Title','URL','Journal','Severe','Severe Significant','Severe Adjusted','Severe Calculated','Fatality','Fatality Significant','Fatality Adjusted','Fatality Calculated','Multivariate adjustment', 'Design', 'Study population','Risk'])



#Filter out primarily the papers from 2020

plot_data_otherrisks = plot_data_otherrisks.loc[plot_data_otherrisks['publish_year'].astype({'publish_year': int}) == 2020]



for index, row in plot_data_otherrisks.iterrows(): 

    resultsother.loc[index] = ""

    resultsother['Date'][index]=plot_data_otherrisks['publish_date'][index]

    resultsother['Title'][index]=plot_data_otherrisks['title'][index]

    resultsother['URL'][index]=plot_data_otherrisks['doi'][index]

    resultsother['Journal'][index]=plot_data_otherrisks['journal'][index]

    resultsother['Design'][index]=plot_data_otherrisks['researchdesign'][index]

    resultsother['Risk'][index] = OtherRisks[plot_data_otherrisks['x'][index]]

    

resultsother.drop_duplicates(['Title'], inplace=True)



    

for iter,orisk in  enumerate(OtherRisks):

    filen = orisk + '.csv'

    numdocs = resultsother['Title'].loc[resultsother['Risk']==orisk].nunique()

    if numdocs:

        resultsother.loc[resultsother['Risk']==orisk].iloc[:,0:-1].to_csv(filen, index=False) 

        print('TOP RESULTS FROM ' + str(numdocs) + ' PAPERS ON OUTCOMES AND '+ orisk +' RELATED RISK FACTORS PUBLISHED IN 2020' + '\n')

        print('TOP RESULT TITLE: ' + resultsother['Title'].loc[resultsother['Risk']==orisk].iloc[0] + '\n')

        print('REFER OUTPUT FILE {} FOR DETAILED RESULTS \n'.format(filen))

        print('---------------------------------------------\n')

#Clean Up

del resultsother

del plot_data_otherrisks

gc.collect()
shortlistdocs = 750
similar_docs_comob = doc2vecmodel.docvecs.most_similar([task2vector], topn=shortlistdocs)

plot_data_comob = pd.DataFrame(columns=['doc_id','title','publish_year','publish_date','journal','doi','abstract','region','x','y','xy_column','relevance', 'researchdesign', 'comorbidity', 'outcome'])

newi=0 

for i, scoreofdoc in similar_docs_comob:

    for com in Comorbidities:

        for out in Outcome:

            if (check_thematic_comorbidities(com, i) & check_outcomes(out, i)):

                plot_data_comob.loc[newi] = ""

                plot_data_comob['doc_id'][newi] = df_covid['paper_id'][i]

                plot_data_comob['publish_date'][newi] = df_covid['publish_date'][i]

                #plot_data_comob['authors'][newi] = df_covid['authors'][i]

                plot_data_comob['journal'][newi] = df_covid['journal'][i]

                plot_data_comob['title'][newi] = df_covid['title_summary'][i]

                plot_data_comob['publish_year'][newi] = int(df_covid['publish_year'][i])

                plot_data_comob['doi'][newi] = df_covid['doi'][i]

                plot_data_comob['abstract'][newi] = df_covid['abstract_summary'][i]

                plot_data_comob['region'][newi] = df_covid['region'][i]

                plot_data_comob['x'][newi] = Comorbidities.index(com)

                plot_data_comob['y'][newi] = Outcome.index(out)

                plot_data_comob['xy_column'][newi] = get_xy_cordinates(i)

                plot_data_comob['relevance'][newi]= int(scoreofdoc*100)

                plot_data_comob['researchdesign'][newi]=df_covid['researchdesign'][i]

                plot_data_comob['comorbidity'][newi] = com

                plot_data_comob['outcome'][newi]= out

                newi+=1

# Make format changes to data column for display 

convert_dict = {'relevance': int}

plot_data_comob = plot_data_comob.astype(convert_dict) 

#plot_data_comob.to_csv('outcome-comorbidities.csv',index=False)
# Plot the evidence gap map for Outcomes vs Comorbidity

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()



figure = evidencegapmap(dataset=plot_data_comob, x_column='x', y_column='y',

  bubble_column='title',bubble_text='abstract', bubble_link='doi', size_column='relevance', color_column='researchdesign',xbin_list=Comorbidities, ybin_list = Outcome,

  xbin_size=100, ybin_size = 100, x_title="Comorbidities", y_title="Outcomes", title='Outcome vs Comorbidity (Mode Random)',scale_bubble=4, marker_opacity=0.6,height=800)

iplot(figure)
ComorbiditiesFilter = ["Diabetes","Hypertension","Immunodeficiency", "Cancer", "Respiratory", "Immunity"]

OutcomeFilter = ["Any","Death", "ICU Admission", 'Mechanical Ventilation', "Organ Failure", "Sepsis", "Discharge"]

DesignFilter = ["Any","Systematic Review","Randomized","Non-Randomized","Ecological Regression","Prospective Cohort","Time Series Analysis","Retrospective Cohort","Cross Sectional","Case Control","Case Study","Simulation", "Unknown"]

from ipywidgets import interact

@interact

def search_articles(xbin=ComorbiditiesFilter,

                    ybin=OutcomeFilter,

                    design=DesignFilter,

                    num_results=['All','Top10'],

                    relevance = ["Any",">75"],

                    publish_year=["2020", "Any"],

                    download=['No','Yes'],):

   

    

    select_cols = ['title', 'publish_year', 'abstract','relevance', 'researchdesign', 'x','y','doi'] 



    # Filter for Relevance 

    if relevance == '>75':

        relevanceval = 75

    elif relevance == 'Any':

        relevanceval = 0

    

    global results 

    

    #Filter for Relevance

    results = plot_data_comob[select_cols].loc[plot_data_comob['relevance'] > relevanceval]

    results = results.sort_values(by=['relevance'], ascending=False)



    # Filter for xbin

    results['x']= results['x'].apply(lambda x: Comorbidities[x])

    results['y']= results['y'].apply(lambda y: Outcome[y])    

    if (xbin=='Any') & (ybin=='Any') :

        pass

    elif (xbin=='Any') :

        results = results.loc[results['y'] == ybin]    

    elif (ybin=='Any') :

        results = results.loc[results['x'] == xbin]  

    else:

        results = results.loc[(results['x'] == xbin) & (results['y'] == ybin)]  



        

    

    # Filter for Design

    if (design == 'Any'):

        pass

    elif (design == 'Unknown'):

        results = results.loc[results['researchdesign'] == 'unknown']     

    else:

        results = results.loc[results['researchdesign'] == design] 

    

      # Publish Year 

    if publish_year == 'Any':

        pass

    else:

        results = results.loc[results['publish_year'].astype({'publish_year': int}) == int(publish_year)]

    

   

    # Number of Results 

    if num_results == 'Top10':

        numresults = min(10,len(results.index))

    elif num_results == 'All' :

        numresults = len(results.index)



        

    results = results.head(numresults)

    

    # Output Results Yes or No 

    if download=="Yes": 

        if (len(results.index) == 0):

            print('NO RESULTS')



            return None

        else:

            results.to_csv('egm_search_results_comorbidities.csv')

            numdocs = results['title'].nunique()

            print('TOP RESULTS FROM ' + str(numdocs) + ' PAPERS ON COMORBIDITIES AND OUTCOMES' + '\n')

            print('TITLE: ' + results.iloc[0]['title'] + '\n')

            print('REFER OUTPUT FILE {} FOR RESULTS WITH BIN AND OTHER DETAILS'.format("egm_search_results_comorbidities.csv"))

            return results[select_cols]
resultscomob = pd.DataFrame(columns=['Date','Title','URL','Journal','Severe','Severe Significant','Severe Adjusted','Severe Calculated','Fatality','Fatality Significant','Fatality Adjusted','Fatality Calculated','Multivariate adjustment', 'Design', 'Study population', 'Comorbidity'])



#Filter out primarily the papers from 2020

plot_data_comob = plot_data_comob.loc[plot_data_comob['publish_year'].astype({'publish_year': int}) == 2020]



for index, row in plot_data_comob.iterrows(): 

    resultscomob.loc[index] = ""

    resultscomob['Date'][index]=plot_data_comob['publish_date'][index]

    resultscomob['Title'][index]=plot_data_comob['title'][index]

    resultscomob['URL'][index]=plot_data_comob['doi'][index]

    resultscomob['Journal'][index]=plot_data_comob['journal'][index]

    resultscomob['Design'][index]=plot_data_comob['researchdesign'][index]

    resultscomob['Comorbidity'][index]= Comorbidities[plot_data_comob['x'][index]]



resultscomob.drop_duplicates(['Title'], inplace=True)



for iter,como in  enumerate(Comorbidities):

    filen = como + '.csv'

    numdocs = resultscomob['Title'].loc[resultscomob['Comorbidity']==como].nunique()

    if numdocs:

        resultscomob.loc[resultscomob['Comorbidity']==como].iloc[:,0:-1].to_csv(filen, index=False) 

        print('TOP RESULTS FROM ' + str(numdocs) + ' PAPERS ON OUTCOMES AND '+ como +' RELATED RISK PUBLISHED IN 2020' + '\n')

        print('TOP RESULT TITLE: ' + resultscomob['Title'].loc[resultscomob['Comorbidity']==como].iloc[0] + '\n')

        print('REFER OUTPUT FILE {} FOR DETAILED RESULTS'.format(filen))

        print('---------------------------------------------\n')

    
#Clean Up

del resultscomob

gc.collect()
!pip install transformers

!pip install bert-extractive-summarizer

import torch

from transformers import BertForQuestionAnswering

from transformers import BertTokenizer

from summarizer import Summarizer

summarizationmodel = Summarizer()
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
# no abstract provided, create a summary of the abstract . Return the summary 

def summarize_text (full_body_text) :

    if len(full_body_text) > 0:

        try:

            # ratio of words approximated as ratio of sentences for summarization and abstracts from first 2500 chars

            res = summarizationmodel(full_body_text[0:2500],ratio=0.4)

            full = ''.join(res).capitalize()

            return full

        except ValueError:

            return "Not Available"

    else :

        return "Not Available"
# Code adopted from https://mccormickml.com/2020/03/10/question-answering-with-a-fine-tuned-BERT/

def answer_question(question, answer_text):

    '''

    Returns the `answer_text` and scores . If the answer start is same as answer end or if answer start is greater than answer end, Not Found is returned    

    

    '''            

    # Apply the tokenizer to the input text, treating them as a text-pair. Trim to 512 max tokens 

    input_ids = tokenizer.encode(question, answer_text, max_length=512, pad_to_max_length=True)



    # Search the input_ids for the first instance of the `[SEP]` token.

    sep_index = input_ids.index(tokenizer.sep_token_id)



    # The number of segment A tokens includes the [SEP] token istelf.

    num_seg_a = sep_index + 1



    # The remainder are segment B.

    num_seg_b = len(input_ids) - num_seg_a



    # Construct the list of 0s and 1s.

    segment_ids = [0]*num_seg_a + [1]*num_seg_b



    # There should be a segment_id for every input token.

    assert len(segment_ids) == len(input_ids)



    # Run our example question through the model.

    start_scores, end_scores = model(torch.tensor([input_ids]), # The tokens representing our input text.

                                    token_type_ids=torch.tensor([segment_ids])) # The segment IDs to differentiate question from answer_text



    # Find the tokens with the highest `start` and `end` scores.

    answer_start = torch.argmax(start_scores).item()

    answer_end = torch.argmax(end_scores).item()

    start_score = torch.max(start_scores).item()

    end_score = torch.max(end_scores).item()

    

    # Get the string versions of the input tokens.

    tokens = tokenizer.convert_ids_to_tokens(input_ids)



    # Start with the first token.

    answer = tokens[answer_start]



    # Select the remaining answer tokens and join them with whitespace.

    for i in range(answer_start + 1, answer_end + 1):

        

        # If it's a subword token, then recombine it with the previous token.

        if tokens[i][0:2] == '##':

            answer += tokens[i][2:]

        

        # Otherwise, add a space then the token.

        else:

            answer += ' ' + tokens[i]

            

    # Tensor objects need to be deleted as they take a lot of overheads and can create RAM outage 

    del start_scores, end_scores, tokens

    if answer_start >=  answer_end :

        start_score = end_score = -10

        answer = "Not Found"

    return answer, start_score, end_score
from ipywidgets import widgets

from IPython.display import display

from ipywidgets import Button, Layout, VBox

import operator

out = widgets.Output()



searchpapers = results.drop_duplicates(subset="title")



def validatesearch():

    # print "validating"

    if len(search.value) < 500 :

        return True

    else:

        print("Please enter upto 500 characters only ")

        return False

    

def responsesearch(change):

    if validatesearch():

        pass

        

search = widgets.Textarea(

    value='Type your research question here. You can shortlist your search space by choosing the appropriate Bin(s) from above.',

    description='Question:',

    disabled=False,

    layout=Layout(width='50%', height='100px')

)



search.observe(responsesearch, names="value")



button = Button(description='Search',layout=Layout(width='80px', height='40px'))



box_layout = Layout(display='flex',

                    flex_flow='column',

                    width ='100%',

                    align_items = 'center')



searchbox = VBox(children=[search,button, out], layout=box_layout)



display(searchbox)



def responsebutton(b):

    answers=[]

    count=1

    searchpapers = results.drop_duplicates(subset="title")

    for index, row in searchpapers.iterrows():

        #print(search.value,row['abstract'])

        if row['abstract'] != 'Not Available' :

            answer, sstart, send = answer_question(search.value,row['abstract'])

            answers.append([index, answer, sstart+send])

        else :

            #print("Abstract missing , generating from full body text for Paper {}".format(count))

            #identify row from the covid dataframe , summarize from the full body text and 

            doi_match_index = df_covid.index[df_covid['doi'] == row['doi']].tolist()

            if doi_match_index :

                text_to_summary=df_covid['body_text'].iloc[doi_match_index[0]]

                sum= summarize_text(text_to_summary) 

                answer, sstart, send = answer_question(search.value,sum)

                answers.append([index, answer, sstart+send])

        print("Completed Analysis of Paper {} of {}".format(count, len(searchpapers.index)))

        count+=1

            

    answers = sorted(answers, key=operator.itemgetter(2), reverse = True) 

    print(answers)

    print('\n')

    print("Top Answer :-- {} ".format(str(answers[0][1])))

    print("2nd Best Answer :-- {} ".format(str(answers[1][1])))

    print("3rd Best Answer :-- {} ".format(str(answers[2][1])))



    

button.on_click(responsebutton)