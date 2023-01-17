!pip install sentence-transformers
import os
import re  # For preprocessing
import en_core_web_sm
from difflib import SequenceMatcher
import pandas as pd
import numpy as np
import pickle
from time import time  # To time our operations
import glob
import json
import zipfile
from tqdm import tqdm
import multiprocessing
import scipy

import torch
from transformers import BertTokenizer
from transformers import BertForQuestionAnswering
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity #for cosine similarity
from gensim.models.phrases import Phrases, Phraser #For create relevant phrases
from gensim.models import Word2Vec #Our model type
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
root_path = '/kaggle/input/CORD-19-research-challenge'

metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str,
    'doi': str
})

meta_df.head()
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            if 'abstract' in content:
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            else:
                self.abstract.append('Not provided.')
            # Body text
            if 'body_text' in content:
                for entry in content['body_text']:
                    self.body_text.append(entry['text'])
            else:
                self.body_text.append('Not provided.')
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)


    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

def get_date_dt(all_json, meta_df):
    dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [],
             'abstract_summary': []}
    for idx, entry in tqdm(enumerate(all_json), desc="Parsing the articles Json's content", total=len(all_json)):
        content = FileReader(entry)

        # get metadata information
        meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
        # no metadata, skip this paper
        if len(meta_data) == 0:
            continue

        dict_['paper_id'].append(content.paper_id)
        dict_['abstract'].append(content.abstract)
        dict_['body_text'].append(content.body_text)
        
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
        dict_['title'].append(meta_data['title'].values[0])

        # add the journal information
        dict_['journal'].append(meta_data['journal'].values[0])
    return pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal'])
def Initialization_Word2Vec_model():
    w2v_model = Word2Vec(min_count=50,
                     window=10,
                     size=200,
                     sample=6e-5,
                     alpha=0.02,
                     min_alpha=0.0003,
                     negative=20,
                     workers=multiprocessing.cpu_count() -1)
    return w2v_model

def create_sentences(df, col_name):
    """Build sentences to Word2Vec model """
    t = time()
    sent = [row.split() for row in df[col_name]]
    phrases = Phrases(sent, min_count=50, progress_per=100, max_vocab_size=1000000)
    bigram = Phraser(phrases)

    print('Time to create sentences: {} mins'.format(round((time() - t) / 60, 2)))
    return bigram, bigram[sent]

def Build_Word2Vec_vocab(w2v_model, sentences,update=True):
    t = time()
    w2v_model.build_vocab(sentences, progress_per=10000, update=update)
    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))
    return w2v_model

def Train_Word2Vec_model(w2v_model, sentences):
    t = time()
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=20, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))
    return w2v_model

def cleaning(spacy_doc):
    """Lemmatizes and removes stopwords
    doc needs to be a spacy Doc object """

    txt = [t.lemma_ for t in spacy_doc if
            t.dep_ not in ['prep', 'punct', 'det'] and
            len(t.text.strip()) > 2 and
            t.lemma_ != "-PRON-" and
            not t.is_stop]
    if len(txt) > 2:
        return ' '.join(txt)
    else:
        return "no text"

def word2vec_preprocessing(df, column, offline=True) -> pd.DataFrame:
    """ Prepare column from pd.DataFrame with text to Word2Vec model """
    url_pattern = r"(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})"
    data = df.drop_duplicates(column)
    data = data[
        data[column].apply(lambda x: len(x) > 0) &
        data[column].apply(lambda x: len(x.split()) > 3)
        ]
    data.reset_index(inplace=True, drop=True)
    data[column] = data[column].map(lambda x: re.sub(url_pattern, ' ', str(x)))
    brief_cleaning = (re.sub(r"[^a-zA-Z']+", ' ', str(row)).lower() for row in data[column])
    nlp = en_core_web_sm.load(disable=['ner', 'parser', 'tagger'])  # disabling Named Entity Recognition for speed
    t = time()
    txt = []
    if offline:
        paper_ids = data['paper_id']
        index = []
        for idx, doc in enumerate(nlp.pipe(brief_cleaning, batch_size=5000, n_threads=-1, n_process=5)):
            txt.append(cleaning(doc))
            index.append(paper_ids[idx]) #TODO verify its ok
        print('Time to clean up everything: {} mins'.format(round((time() - t) / 60, 2)))
    else:
        for doc in nlp.pipe(brief_cleaning):
            txt.append(cleaning(doc))

    df_clean = data
    print([t for t in txt if t is None])
    df_clean[f'clean_{column}'] = txt
    return df_clean

def preprocessing(corpus: List, is_offline: bool):
    dt_corpus = pd.DataFrame({'clean': corpus})
    procesed = word2vec_preprocessing(dt_corpus, 'clean', offline=is_offline)
    return procesed['clean_clean'].to_list()

def get_relevant_articles(query,  articles_abstract):
    queries: List[str] = preprocessing([query], is_offline=False)
    articles_abstract.score = articles_abstract.score.astype(float) # to make the scores as float in order to not lose precision
    for q in queries:
        query_key_words: List[str] = get_unk_kw(phraser, q)
        embedded_query = embed_sentence(w2v_model, query_key_words)
        for idx, abstract in tqdm(articles_abstract.iterrows(), desc="Iterate over all articles abstract", total=len(articles_abstract)):
            abs_score = get_abstract_score(w2v_model, embedded_query, abstract['clean_abstract'])
            articles_abstract.at[idx, 'score'] = abs_score
    scored_articles_abstract = articles_abstract.sort_values('score', ascending=False)
    return scored_articles_abstract
data_dt = get_date_dt(all_json, meta_df)
to_train = False
if to_train:
    w2v_model = Initialization_Word2Vec_model()
    df_clean = word2vec_preprocessing(data_dt, 'body_text')
    phraser, sentences = create_sentences(df_clean, 'clean_body_text')
    update_w2v_vocab = len(w2v_model.wv.vocab) != 0
    w2v_model = Build_Word2Vec_vocab(w2v_model, sentences, update_w2v_vocab)
    w2v_model = Train_Word2Vec_model(w2v_model, sentences)
    # w2v_model.save('saved_model/w2v_model_on_all_abstract_full_text.w2v')
    w2v_model.init_sims(replace=True)
else:
    with open("/kaggle/input/w2v-model/saved_model/cleaned_to_w2v_all_document.pkl", 'rb') as f:
        df_clean = pickle.load(f)
    with open("/kaggle/input/w2v-model/phraser.pkl", 'rb') as f:
        phraser = pickle.load(f)
    w2v_model = Word2Vec.load('/kaggle/input/w2v-model/saved_model/w2v_model_on_all_abstract_full_text.w2v')

X1 = df_clean['clean']
doc_X = TfidfVectorizer().fit_transform(X1)
cluster=MiniBatchKMeans(n_clusters = 20)
doc_assigned_clusters=cluster.fit_predict(doc_X)
doc_df = pd.DataFrame(doc_assigned_clusters, columns=['cluster'])
doc_df.head()
X_vocab = w2v_model[w2v_model.wv.vocab]
vocab_words = list(w2v_model.wv.vocab.keys())
from sklearn.cluster import MiniBatchKMeans
#cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
cluster=MiniBatchKMeans(n_clusters = 20)
vocab_assigned_clusters=cluster.fit_predict(X_vocab)
vocab_df = pd.DataFrame(vocab_assigned_clusters, index=vocab_words, columns=['cluster'])
vocab_df.head()



cluster = AgglomerativeClustering(n_clusters=130, affinity='euclidean', linkage='ward')
doc_assigned_clusters=cluster.fit_predict(X.toarray())
print("most similar words to kaletra:")
for s_w in w2v_model.wv.most_similar(positive=[ 'kaletra']):
    print(s_w)
print('#'*100)
print("most similar words to bat:")
for s_w in w2v_model.wv.most_similar(positive=['bat']):
    print(s_w)
print('#'*100)
print("most similar words to dead and people:")
for s_w in w2v_model.wv.most_similar(positive=['dead', 'people']):
    print(s_w)
ALPHA = 0.5

def embed_sentence(model, tokens: List[str]) -> List:
    res = []
    for t in tokens:
        try:
            vec = model.wv.word_vec(t, use_norm=False)
            res.append(vec)
        except KeyError:
            # logging.debug(f'Unidentified word while embedding:{t}')
            continue
    return res


def get_all_abstracts(phraser, cleaned_df):
    column = 'abstract'
    df = word2vec_preprocessing(cleaned_df, column, offline=True)
    df[f'clean_{column}'] = df[f'clean_{column}'].map(lambda x: get_unk_kw(phraser, x))
    df['score'] = [0] * len(df)
    return df


def get_unk_kw(phraser, query):
    if type(query) == str:
        return list(set(phraser[query.split()]))
    else:
        return []


def preprocessing(corpus: List, is_offline: bool):
    dt_corpus = pd.DataFrame({'clean': corpus})
    procesed = word2vec_preprocessing(dt_corpus, 'clean', offline=is_offline)
    return procesed['clean_clean'].to_list()


def get_abstract_score(model: Word2Vec, embedded_query:List, abstract: List[str]) -> float:
    f_score = 0
    valid_scores_sum = 1  # so we wouldn't divide by zero
    if type(abstract) != list or len(abstract) == 0:
        return f_score
    embedded_abstract = embed_sentence(model, abstract)
    for q_t in embedded_query:
        scores = model.wv.cosine_similarities(q_t, embedded_abstract)
        valid_scores = [s for s in scores if s > ALPHA]
        valid_scores_sum += len(valid_scores)
        f_score += np.sum(valid_scores)
    norm_score = f_score / valid_scores_sum # to normalize by the number of tokens that was counted
    return norm_score

# articles_abstract: pd.DataFrame = get_all_abstracts(phraser, all_data)
print(w2v_model.wv.most_similar(positive=['medicine', 'covid'], negative=['sars']))
print(w2v_model.wv.most_similar(positive=['dead', 'covid','israel']))
print(w2v_model.wv.similarity('kaletra', 'covid'))
print(w2v_model.wv.similarity('kaletra', 'sars'))


with open('/kaggle/input/w2v-model/parsed_abstract_ran_offline.pk', 'rb') as f: 
    articles_abstract = pickle.load(f)

encoder = SentenceTransformer("roberta-large-nli-stsb-mean-tokens")
encoded_articles_abstract = encoder.encode(articles_abstract['abstract'].tolist())
articles_abstract['encoded_articles_abstract'] = encoded_articles_abstract    
def query_questions(query, articles_abstract):    
    encoded_query = encoder.encode([query])
    articles_abstract['distances'] = scipy.spatial.distance.cdist(encoded_query, articles_abstract['encoded_articles_abstract'].tolist(), "cosine")[0]
    articles_abstract = articles_abstract.sort_values('distances').reset_index()[:70]
    
    articles_abstract['sentence_list'] = [body.split(". ") for body in articles_abstract['body_text'].to_list()] 
    paragraphs = []
    for index, ra in articles_abstract.iterrows():
        para_to_add = [". ".join(ra['sentence_list'][n:n+7]) for n in range(0, len(ra['sentence_list']), 7)]        
        para_to_add.append(ra['abstract'])
        paragraphs.append(para_to_add)
    articles_abstract['paragraphs'] = paragraphs
    answers = answer_question(query, articles_abstract)
    return answers
def get_QA_bert_model():
    """
    Download pre-trained QA model
    """
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    BERT_SQUAD = 'bert-large-uncased-whole-word-masking-finetuned-squad'
    model = BertForQuestionAnswering.from_pretrained(BERT_SQUAD)
    tokenizer = BertTokenizer.from_pretrained(BERT_SQUAD)
    model = model.to(torch_device)
    model.eval()
    return model, tokenizer
model, tokenizer = get_QA_bert_model()
from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def answer_question(question: str, context_list):
    # anser question given question and context
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    answers =[]
    all_para = [item for sublist in context_list['paragraphs'].to_list() for item in sublist] 
    print(f"paragraph to scan: {len(all_para)}")
    for _, article in tqdm(context_list.iterrows()):
        for context in article['paragraphs']:
            if len(context) < 10:
                continue
            encoded_dict = tokenizer.encode_plus(
                                question, context,
                                add_special_tokens = True,
                                max_length = 500,
                                pad_to_max_length = True,
                                return_tensors = 'pt'
                           )

            input_ids = encoded_dict['input_ids'].to(torch_device)
            token_type_ids = encoded_dict['token_type_ids'].to(torch_device)
            with torch.no_grad():  
                start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids)
            all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            start_index = torch.argmax(start_scores)
            end_index = torch.argmax(end_scores)

            answer = tokenizer.convert_tokens_to_string(all_tokens[start_index:end_index+1])
            answer = answer.replace('[CLS]', '').replace('[PAD]', '').replace('[SEP]', '')
            if len(answer.strip()) > 6 and similar(question.lower().strip(), answer.lower().strip()) < 0.8:
                answers.append({'answer': answer, 'paragraph': context, 'paper_id': article['paper_id']})
    return pd.DataFrame(answers)
q1 = "Which factors may affect the survial of viruses?"
q1_paper_id = "5925c03efa6ee06cf5f625cdc02d8d39b7ab1fa7" 
q1_paragraph = "The survival of viruses in the environment is influenced by a number of factors e.g. RH, temperature and suspending medium etc. [6, 7] . We have observed over the past several years that survival of different rotavirus isolates in air [7, 9-13, 16, 17] , and on surfaces [6] is influenced drastically by the level of RH used in experimental conditions. The experimental observations made in the present study on the survival of aerosolized (Tables 1 and 2) ."
q1_answer = query_questions(q1, articles_abstract)

q1_paper_id in list(q1_answer['paper_id'])
q2 = "What types of viruses can cause acute lower respiratory infections (ALRI)?"
q2_paper_id = "608318d1cbddf1a10ea1d6faca8b4cacf8467c29" 
q2_paragraph = "Acute respiratory infections are a leading cause of morbidity and mortality worldwide. 1 They represent around 2 million deaths per year, especially in infants. 2 The burden of these infections is particularly important in developing countries. 3 During the last decade, South East Asia received much attention from the international scientific community due to the emergence of respiratory viruses with pandemic potential (SARS-CoV, avian influenza A/ H5N1 virus). 4 Respiratory infections can be caused by numerous viruses, including influenza viruses, parainfluenza viruses, human respiratory syncytial virus (HRSV), human metapneumovirus (HMPV), human coronaviruses (HCoV), adenoviruses, human bocavirus, and human enteroviruses. Molecular techniques have become more and more popular to detect these viruses. Multiplex reverse transcription-polymerase chain reaction (RT-PCR) has been shown to be a sensitive tool and allows identification of a majority of respiratory viruses, as well as coinfections. [5] [6] [7] In Lao PDR, the etiology of respiratory infections is still poorly documented. To improve the clinical management of the patients, limit unnecessary antibiotic use, and prevent opportunistic secondary infections, it appears important to develop surveillance and tools to assess the etiology of acute respiratory infections in this country. 8, 9 The purpose of this study was to describe during a limited period of time the viral etiology of acute lower respiratory infections (ALRI) in patients hospitalized in two Lao hospitals by using a set of five multiplex RT-PCR/PCR targeting 18 common respiratory viruses."
q2_answer = query_questions(q2, articles_abstract)

q2_paper_id in list(q2_answer['paper_id'])
q3 = "which recptor agonists are possible candidates for treating autoimmune diseases?"
q3_paper_id = "711642ba1c1f4dc3cedb11f67f77d2c0425c9da3" 
q3_paragraph ="Sphingosine 1-phosphate type 1 (S1P 1 ) receptors are expressed on lymphocytes and regulate immune cells trafficking. Sphingosine 1-phosphate and its analogues cause internalization and degradation of S1P 1 receptors, preventing the auto reactivity of immune cells in the target tissues. It has been shown that S1P 1 receptor agonists such as fingolimod can be suitable candidates for treatment of autoimmune diseases. The current study aimed to generate GRIND-based 3D-QSAR predictive models for agonistic activities of 2-imino-thiazolidin-4-one derivatives on S1P 1 to be used in virtual screening of chemical libraries. The developed model for the S1P 1 receptor agonists showed appropriate power of predictivity in internal (r 2 acc 0.93 and SDEC 0.18) and external (r 2 0.75 and MAE (95% data), 0.28) validations. The generated model revealed the importance of variables DRY-N1 and DRY-O in the potency and selectivity of these compounds towards S1P 1 receptor. To propose potential chemical entities with S1P 1 agonistic activity, PubChem chemicals database was searched and the selected compounds were virtually tested for S1P 1 receptor agonistic activity using the generated models, which resulted in four potential compounds with high potency and selectivity towards S1P 1 receptor. Moreover, the affinities of the identified compounds towards S1P 1 receptor were evaluated using molecular dynamics simulations. The results indicated that the binding energies of the compounds were in the range of À39.31 to À46.18 and À3.20 to À9.75 kcal mol À1 , calculated by MM-GBSA and MM-PBSA algorithms, respectively. The findings in the current work may be useful for the identification of potent and selective S1P 1 receptor agonists with potential use in diseases such as multiple sclerosis." 
q3_answer = query_questions(q3, articles_abstract)

q3_paper_id in list(q3_answer['paper_id'])
q4 = "Which lessons are learned regarding the adoption of basic universal safety precautions (USPs) such as handwashing?"
q4_paper_id = "841e2b8df37c627db6013f7da95a3d9b7cff81f2" 
q4_paragraph ="Generally, educating the public with an emphasis on personal hygiene, such as washing hands, would take a lot of time and effort, especially in developing countries such as India. In a large cross-sectional comparative study conducted in Bangladesh from 2006 to 2011, including participants from 50 sub-districts inferred that there exists a significant gap between perceptions and practice of proper handwashing behaviours among their study participants. It also found that handwashing behaviour before eating food was lower, and unfortunately, only 8% of their study participants stated that they use soap for washing their hands at the baseline. It also noticed that handwashing knowledge and practices were relatively lower before cooking, serving and eating food [26] . Furthermore, socioeconomic status, including education, have shown a positive association with handwashing, which are similar to our current study findings. During outbreaks such as COVID-19, information regarding safety measures should be selflessly promoted by the news channels, print media, radio stations, and social media, as almost every individual relates to either of these platforms at some point in a day. Around three fourth of respondents in a descriptive cross-sectional study conducted in Nigeria have stated that they have acquired good handwashing measures by watching health education messages from social media, newspapers and radio channels [27] ." 
q4_answer = query_questions(q4, articles_abstract)

q4_paper_id in list(q4_answer['paper_id'])
q5 = "What can help reduce unnecessary usage of antibiotcs in cases of acute respiratory infections?"
q5_paper_id = "b58940f7756058fdb940f4fab7e4852313c09d20" 
q5_paragraph ="Acute respiratory tract infections are a major cause of morbidity and mortality and represent a significant burden on the health care system. Laboratory testing is required to definitively distinguish infecting influenza virus from other pathogens, resulting in prolonged emergency department (ED) visits and unnecessary antibiotic use. Recently available rapid point-of-care tests (POCT) may allow for appropriate use of antiviral and antibiotic treatments and decrease patient lengths of stay.We undertook a systematic review to assess the effect of POCT for influenza on three outcomes: (1) antiviral prescription, (2) antibiotic prescription, and (3) patient length of stay in the ED. The databases Medline and Embase were searched using MeSH terms and keywords for influenza, POCT, antivirals, antibiotics, and length of stay. Amongst 245 studies screened, 30 were included. The majority of papers reporting on antiviral prescription found that a positive POCT result significantly increased use of antivirals for influenza compared with negative POCT results and standard supportive care. A positive POCT result also led to decreased antibiotic use. The results of studies assessing the effect of POCT on ED length of stay were not definitive. The studies assessed in this systematic review support the use of POCT for diagnosis of influenza in patients suffering an acute respiratory infection. Diagnosis using POCT may lead to more appropriate prescription of treatments for infectious agents. Further studies are needed to assess the effect of POCT on the length of stay in ED. KEYWORDS influenza, point of care, systematic review" 
q5_answer = query_questions(q5, articles_abstract)

q5_paper_id in list(q5_answer['paper_id'])
q6 = "What were the initial actions for identifying immune system responses in patients?"
q6_paper_id = "17643a57b92abf0d6635d308949e78de2ecc7f66" 
q6_paragraph ="In order to detect anti-viral immune responses, we first constructed recombinant pET28-N-6XHis by linking 6 copies of His tag to the C-terminus of NP in the pET28-N vector (Biomed, Cat. number: BM2640). Escherichia coli transformed with pET28-N-6xHis was lysed and tested by Coomassie blue staining to confirm NP expression at 45.51 kDa. NP was further purified by Ni-NTA affinity chromatography and gel filtration. The purity of NP was approximately 90% ( Figure S1A ). The presence of NP was subsequently confirmed by anti-Flag antibody ( Figure S1B ). The RBD region of S protein (S-RBD) and main protease (doi: https://doi.org/10.1101/2020.02. 19 .956235) were produced by a baculovirus insect expression system and purified to reach the purity of 90% ( Figure S1A )." 
q6_answer = query_questions(q6, articles_abstract)
q6_paper_id in list(q6_answer['paper_id'])
q7 = "How is RAS related to blood pressure?"
q7_paper_id = "27fbfb693135818f95a81d461b54d2d3a359ab1e" 
q7_paragraph ="The RAS plays an important role in the regulation of arterial blood pressure. Renin is an enzyme that acts on angiotensinogen to catalyze the formation of Ang I. Ang I is then cleaved by ACE to yield Ang II. A representation of the biochemical pathways of RAS is shown in Fig. 1 ."
q7_answer = query_questions(q7, articles_abstract)
q7_paper_id in list(q7_answer['paper_id'])
q8 = "how to evaluate effect of different conditions on L. hongkongensis?"
q8_paper_id = "31b039f27dbcd96df12be89f281f576d26fe80e1" 
q8_paragraph ="To better understand how L. hongkongensis adapts to human body and freshwater habitat temperatures at the molecular level, the types and quantities of proteins expressed in L. hongkongensis HLHK9 cultured at 37uC and 20uC were compared. Since initial 2D gel electrophoresis analysis of L. hongkongensis HLHK9 proteins under a broad range of pI and molecular weight conditions revealed that the majority of the proteins reside on the weakly acidic to neutral portion, with a minority on the weak basic portion, consistent with the median pI value of 6.63 calculated for all putative proteins in the genome of L. hongkongensis HLHK9, we therefore focused on IPG strips of pH 4-7 and 7-10. Comparison of the 2D gel electrophoresis patterns from L. hongkongensis HLHK9 cells grown at 20uC and 37uC revealed 12 differentially expressed protein spots, with 7 being more highly expressed at 20uC than at 37uC and 5 being more highly expressed at 37uC than at 20uC (Table 2, Figure 3 ). The identified proteins were involved in various functions (Table 2 ). Of note, spot 8 [N-acetyl-L-glutamate kinase (NAGK)-37, encoded by argB-37] was up-regulated at 37uC, whereas spot 1 (NAGK-20, encoded by argB-20), was upregulated at 20uC (Figures 3, 4A and 4B ). These two homologous copies of argB encode two isoenzymes of NAGK [NAGK-20 (LHK_02829) and NAGK-37 (LHK_02337)], which catalyze the second step of the arginine biosynthesis pathway. note: this paper is not related to covid-19"
q8_answer = query_questions(q8, articles_abstract)
q8_paper_id in list(q8_answer['paper_id'])
q9 = "cost-effective approach for monitoring disease spread in poor areas"
q9_paper_id = "364b968c148ee72c7336bf89c06974a646683fd3" 
q9_paragraph ="Globally, regions at the highest risk for emerging infectious diseases are often the ones with the fewest resources. As a result, implementing sustainable infectious disease surveillance systems in these regions is challenging. The cost of these programs and difficulties associated with collecting, storing and transporting relevant samples have hindered them in the regions where they are most needed. Therefore, we tested the sensitivity and feasibility of a novel surveillance technique called xenosurveillance. This approach utilizes the host feeding preferences and behaviors of Anopheles gambiae, which are highly anthropophilic and rest indoors after feeding, to sample viruses in human beings. We hypothesized that mosquito bloodmeals could be used to detect vertebrate viral pathogens within realistic field collection timeframes and clinically relevant concentrations. PLOS Neglected Tropical Diseases |Note: here the paragraph is the abstract"
q9_answer = query_questions(q9, articles_abstract)
 
q9_paper_id in list(q9_answer['paper_id'])
q10 = "what causes the variation in viral envelope in mammals?"
q10_paper_id = "546e9ad603afa86e09d191388ec3a6a6b13febad" 
q10_paragraph ="In mammals, viral envelope variation in the surface protein subunit is likely driven to a large degree by positive selection in response to host adaptive immune systems (Caffrey 2011) . While innate immune responses in vertebrates, invertebrates, and plants have been shown to contribute to the evolution of virulence/effector proteins in pathogens that attenuate these responses (Finlay and McFadden 2006; Nishimura and Dangl 2010) , there is no evidence that antigenic variation is employed as a mechanism to escape innate immunity (Finlay and McFadden 2006) . Nor is there any evidence that envelope variants are responsible for suppression or evasion of silencing of viral gene expression by host siRNAs in plants or animals (Li and Ding 2006; Obbard et al. 2009 )."
q10_answer = query_questions(q10, articles_abstract)
 
q10_paper_id in list(q10_answer['paper_id'])
q11 = "What is the usage of crystal structure of proteolytically cleaved Ebola virus GP?" 
q11_paper_id = "3f704d2ac81e0dc4e9e74ffd7d25f790baa6ebe1"
q11_paragraph ="The filovirus surface glycoprotein (GP) mediates viral entry into host cells. Following viral internalization into endosomes, GP is cleaved by host cysteine proteases to expose a receptor-binding site (RBS) that is otherwise hidden from immune surveillance. Here, we present the crystal structure of proteolytically cleaved Ebola virus GP to a resolution of 3.3 Å. We use this structure in conjunction with functional analysis of a large panel of pseudotyped viruses bearing mutant GP proteins to map the Ebola virus GP endosomal RBS at molecular resolution. Our studies indicate that binding of GP to its endosomal receptor Niemann-Pick C1 occurs in two distinct stages: the initial electrostatic interactions are followed by specific interactions with a hydrophobic trough that is exposed on the endosomally cleaved GP 1 subunit. Finally, we demonstrate that monoclonal antibodies targeting the filovirus RBS neutralize all known filovirus GPs, making this conserved pocket a promising target for the development of panfilovirus therapeutics."
q11_answer = query_questions(q11, articles_abstract)

q11_paper_id in list(q11_answer['paper_id'])
q12 = "What are the issues with organisation and management models for psychological interventions?" 
q12_paper_id = "2af31bb92b4d8925973edce56eba44907c58ebec"
q12_paragraph ="The organisation and management models for psychological interventions in China must be improved. Several countries in the west (eg, the UK and USA) have established procedures for psychological crisis interventions to deal with public health emergencies. 3 Theoretical and practical research on psychological crisis interventions in China commenced relatively recently. In 2004, the Chinese Government issued guidelines on strengthening mental health initiatives, 4 and psychological crisis interventions have dealt with public health emergencies-eg, after the type A influenza outbreak and the Wenchuan earthquake-with good results. 5, 6 During the severe acute respiratory syndrome (SARS) epidemic, several psychological counselling telephone helplines were opened for the public, and quickly became important mechanisms in addressing psychological issues. However, the organisation and management of psychological intervention activities have several problems.First, little attention is paid to the practical implementation of interventions. Overall planning is not adequate. When an outbreak occurs, no authoritative organisation exists to deploy and plan psychological intervention activities in different regions and subordinate departments. Hence, most medical departments start psychological interventional activities independently without communicating with each other, thereby wasting mental health resources, and failing patients in terms of a lack of a timely diagnosis, and poor follow-up for treatments and evaluations. Second, the cooperation between community health services and mental-health-care institutions in some provinces and cites in China has been decoupled. After the assessment of the mental health states of individuals affected by the epidemic, patients cannot be assigned according to the severity of their condition and difficulty of treatment to the appropriate department or professionals for timely and reasonable diagnosis and treatment. And after remission of the viral infection, patients cannot be transferred quickly from a hospital to a community health service institution to receive continuous psychological treatment."
q12_answer = query_questions(q12, articles_abstract)

q12_paper_id in list(q12_answer['paper_id'])
q13 = "Why are treatments for MS useful?" 
q13_paper_id = "2a1c5b2541cb958a6280094aa3c1cd2e414fb468"
q13_paragraph ="A major hallmark of the autoimmune demyelinating disease multiple sclerosis (MS) is immune cell infiltration into the brain and spinal cord resulting in myelin destruction, which not only slows conduction of nerve impulses, but causes axonal injury resulting in motor and cognitive decline. Current treatments for MS focus on attenuating immune cell infiltration into the central nervous system (CNS). These treatments decrease the number of relapses, improving quality of life, but do not completely eliminate relapses so long-term disability is not improved. Therefore, therapeutic agents that protect the CNS are warranted. In both animal models as well as human patients with MS, T cell entry into the CNS is generally considered the initiating inflammatory event. In order to assess if a drug protects the CNS, any potential effects on immune cell infiltration or proliferation in the periphery must be ruled out. This protocol describes how to determine whether CNS protection observed after drug intervention is a consequence of attenuating CNS-infiltrating immune cells or blocking death of CNS cells during inflammatory insults. The ability to examine MS treatments that are protective to the CNS during inflammatory insults is highly critical for the advancement of therapeutic strategies since current treatments reduce, but do not completely eliminate, relapses (i.e., immune cell infiltration), leaving the CNS vulnerable to degeneration."
q13_answer = query_questions(q13, articles_abstract)

q13_paper_id in list(q13_answer['paper_id'])
q14 = "What are the potential causes of the covid-19 spread?" 
q14_paper_id = "27c6eda0f75c116059fc221888ee52e9ada306f6"
q14_paragraph ="As the world's annual event for Muslims in specified time (Dhu al-Hijjah 8-12) with a predetermined location (Mecca), the Hajj not only has a huge impact on the traditional international relations, but also posed health a major challenge in international health security. Millions of Muslims from different continents simultaneously gather in the holy city of Mecca, which has a tropical arid climate. This has made the Hajj a potential cause of the spread of bacteria and viruses, especially in face of ''SARS'' and ''Middle East respiratory syndrome'' and other global infectious diseases. As the world's largest gathering of Islamic pilgrimage activity and transnational movement of population, the Hajj poses a threat to public health to host countries such as Saudi Arabia. Unlike other international political, economic and even cultural activities, the Hajj is a religious assignment clearly requested in the Koran. Therefore, it is not possible to cancel the Hajj due to international health security challenges. The only choice is to strengthen the international health management cooperation and minimize Hajj's impact on international health security, on the basis of accumulated experience and the benefit of human beings."
q14_answer = query_questions(q14, articles_abstract)

q14_paper_id in list(q14_answer['paper_id'])
q15 = "What are the risk in exposure to polarizing conditions?" 
q15_paper_id = "0edacad0d29be4fa6ba0395f2fbdc8c0163b996e"
q15_paragraph ="The hypothesis that intrathymic exposure to type 1 polarizing conditions could lead to a loss of selftolerance implies that the activated T cells derived from BBDR rat ATOC exposed to type 1 cytokines should be self-reactive. Preliminary studies using adoptive transfer methods suggest that this is, in fact, the case (B.J.W., unpublished observations)."
q15_answer = query_questions(q15, articles_abstract)

q15_paper_id in list(q15_answer['paper_id'])
q16 = "What are the agents in the outbreak of diarrhea?" 
q16_paper_id = "fadf996ec85f71607f9d3a066fd7698143df0a0c"
q16_paragraph ="absract"
q16_answer = query_questions(q16, articles_abstract)

q16_paper_id in list(q16_answer['paper_id'])
q17 = "How could we reduced viral infection and replication of PEDV?" 
q17_paper_id = "f635dd0f13ee3859c7e4b653398c9b9dba9895d8"
q17_paragraph ="absract"
q17_answer = query_questions(q17, articles_abstract)
q17_paper_id in list(q17_answer['paper_id'])
q18 = "What can be the difficulties in detecting parallel adaptation using virus genomes?" 
q18_paper_id = "f62d64a50f26c6a45fc9f049fcaba788600bedbb."
q18_paragraph ="absract"
q18_answer = query_questions(q18, articles_abstract)
q18_paper_id in list(q18_answer['paper_id'])
q19 = "What could be a canidate for SARS DNA vaccination?" 
q19_paper_id = "ea5534297452fc523674a7fee46b42c7f61f31ec"
q19_paragraph ="absract"
q19_answer = query_questions(q19, articles_abstract)
q19_paper_id in list(q19_answer['paper_id'])
q20 = "How can we minimize the total expected cost of an emerging epidemic?" 
q20_paper_id = "d5794d9e687b1087383f2bca0c5beaf31ebc2955"
q20_paragraph ="absract"
q20_answer = query_questions(q20, articles_abstract)
q20_answer[13:19]
answer = query_questions("Are there geographic variations in the rate of COVID-19 spread?", articles_abstract)

answer['answer'][1]
answer2 = query_questions("What works have been done on infection spreading?", articles_abstract)

answer2['answer'].to_list()
answer2['paragraph'].to_list()
answer3 =  query_questions("Are there geographic variations in the mortality rate of COVID-19?", articles_abstract)
answer3.head()
answer4 =  query_questions("Is there any evidence to suggest geographic based virus mutations?", articles_abstract)
answer4.head()