### Data preprocess

import os

import pandas as pd

import warnings

warnings.simplefilter('ignore')

##############################################

root_path = '/kaggle/input/CORD-19-research-challenge/'

### Readin meta data

metadata_path = f'{root_path}/metadata.csv'

metadata = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str,

    'doi': str

})

metadata.set_index('sha', inplace=True);

metadata.head(2)
### Helper function in data preprocess



def file_scan(root_path, subfolder = None):

    '''

    Function used to scan all data files inside the the directory 

    @Param: 

        root_path: str

        subfolder: str

            Determine whether to scan all files under different folder or one specific folder

    return:

        res: list of all json files in file_name (no '.json' fix)

    '''

    if subfolder:

        root_path = root_path + os.sep + subfolder

    res = []

    for sub_path in os.listdir(root_path):

        tmp_path = root_path+os.sep + sub_path

        if os.path.isdir(tmp_path):

            res += file_scan(tmp_path)

        elif os.path.isfile(tmp_path) and tmp_path.endswith(".json"):

            res.append(tmp_path)

        else: 

            continue

    return res
### function file_scan use example:

non_comm_files = file_scan(root_path, 'noncomm_use_subset')

biorxiv_medrxiv = file_scan(root_path, 'biorxiv_medrxiv')

comm_use_subset = file_scan(root_path, 'comm_use_subset')

print(f'The number of files in non-comm folder: {len(comm_use_subset)}')

print(f'format example: \n {comm_use_subset[0]}')
### Helper function and class object for data parsing



import numpy as np

import string

import json

import nltk

nltk.download('punkt')

############################################



def remove_nonascii(string):

    '''

    @Param: 

        string: str

            original string, might contains Non-Ascii char

        

    return:

        un-named: str

            cleaned string withour non-ascii

    '''

    return "".join(i for i in string if ord(i) < 128)



'''

Move the definition of punctuation, stopwords, stemmer, lemmatizer outside of function 

to avoid redundent IO

'''

punctuation = string.punctuation

stopwords = nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.porter.PorterStemmer()

lemmatizer = nltk.WordNetLemmatizer()



def text_processor(text):

    '''

    Processing original text, remove punctuation, extract stem and furfill lemmatizer

    

    @Param: 

        text: str

            

    return:

        un-named: str

    '''

    text = remove_nonascii(text)

    if len(text) == 0:

        return text

    tok = nltk.word_tokenize(text)

    refined_words = []

    translator = str.maketrans('', '', punctuation)

    for word in tok:

        word = str(word).translate(translator)

        word = word.lower()

        if (2 <= len(word) <= 40) and (word not in stopwords) and (not word.isdigit()):

            refined_words.append(word)

    stemmed_words = []

    for word in refined_words:

        if word == "aeds" or word == "aed":

            continue

        word = stemmer.stem(word)

        word = lemmatizer.lemmatize(word)

        if (2 <= len(word) <= 40) and (word not in stopwords):

            stemmed_words.append(word)

    refined_text = ""

    for word in stemmed_words:

        refined_text += word

        refined_text += " "

    refined_text = refined_text.strip()

    refined_text += ". "

    return refined_text





######################

class File(object):

    '''

    The constuctor of File requires pre_readin of metadata

    '''

    def __init__(self, file_path):

        with open(file_path, 'r') as f:

            file_data = json.load(f)

        self.paper_id = file_data.get('paper_id') or "UK_id"

        self.title = file_data.get('metadata').get('title') if file_data.get('metadata') and file_data.get('metadata').get('title') else "UK_title"

        self.abstract = [i['text'] for i in file_data['abstract']] if file_data.get('abstract') else []

        self.body_text = [i['text'] for i in file_data['body_text']] if file_data.get('body_text') else []

        

        self.authors = [f'{i["first"]} {i["last"]}' for i in file_data['metadata']['authors']] if file_data.get('metadata') and file_data.get('metadata').get('authors') else []

        try: 

            meta = metadata.loc[self.paper_id]

            self.pubmed_id = meta['pubmed_id'] if not pd.isnull(meta['pubmed_id']) else ""

            self.publish_time = meta['publish_time'] if not pd.isnull(meta['publish_time']) else ""

            self.journal = meta['journal'] if not pd.isnull(meta['journal']) else ""

        except:

            self.pubmed_id = ""

            self.publish_time = ""

            self.journal = ""

            

    def __call__(self):

        return {

            'paper_id': self.paper_id,

            'title': self.title,

            'abstract': self.abstract,

            'body_text': self.body_text,

            'authors': self.authors,

            'pubmed_id':self.pubmed_id,

            'publish_time':self.publish_time,

            'journal': self.journal

            

        }





def file_parse(root_path, subfolder = None, save_to_file=True, show_progress=True):

    '''

    Generate cleaned papar abstract, title, body text dataframe for later use

    

    Params:

        root_path: string 

            Indicate the main folder of file

        subfolder: string

            Indicate whether to parse one folder or whole data

        save_to_file: bool

            Indicate whether to save it to local environment excel file

    

    Return:

        df_covid: pandas.DataFrame

            Return the parsed dataframe for quick use

    '''

    target_files = file_scan(root_path, subfolder)

    

    dict = {

        'paper_id': [],

        'abstract_raw': [], 

        'body_text': [], 

        'authors': [], 

        'title_raw': [], 

        'title_all':[],

        'journal': [], 

        'abstract_all': [],

        'pubmed_id': [],

        'publish_time': [],

    }

    for idx, path in enumerate(target_files):

        try:

            if idx % (int(len(target_files) // 10)) == 0 and show_progress:

                print(f'{round(idx / (len(target_files)) * 100)}% have finished');

            file = File(path)

            dict['paper_id'].append(file()['paper_id'])

            dict['journal'].append(file()['journal'])

            dict['pubmed_id'].append(file()['pubmed_id'])

            dict['publish_time'].append(file()['publish_time'])

            dict['authors'].append(','.join(file()['authors']))



            dict['title_raw'].append(remove_nonascii(file()['title']).replace("\"", "").replace("\'", "").replace("\\", "-").replace("\n", " ").strip())

            ### Original title, de-NonAscii, de-some bothering char



            dict['title_all'].append(text_processor(file()['title'].strip())) 

            ### Extract the stem and lemmetize the original title



            dict['body_text'].append('\n'.join(file()['body_text'])) 

            dict['abstract_raw'].append(remove_nonascii(".".join(file()['abstract'])).replace("\"", "").replace("\'", "").replace("\\", "-").replace("\n", " ").strip())

            dict['abstract_all'].append(text_processor(".".join(file()['abstract']).strip()))

        except:

            print(f"id: {idx}, path: {path} readin fails, please check")

            continue;

    print('All parsing finished!');

    df_covid = pd.DataFrame(dict, columns = ['paper_id',

        'abstract_raw', 

        'body_text', 

        'authors', 

        'title_raw', 

        'title_all',

        'journal', 

        'abstract_all',

        'pubmed_id',

        'publish_time',])

    if save_to_file:

        filename = f"df_covid_{subfolder}.xlsx"

        df_covid.to_excel(filename)

    return df_covid
### Sample usage of file parser with biorxiv_medrxiv sub directory

# Around one minute

biorxiv_medrxiv = file_parse(root_path, subfolder='biorxiv_medrxiv', show_progress=False)

biorxiv_medrxiv.head(2)
### Import Bert, Albert and configured for GPU accelerator if possible

from transformers import (BertForQuestionAnswering,BertTokenizer)

from transformers import (AlbertForQuestionAnswering, AlbertTokenizer)

import torch

import scipy

#################################################

### Load pretrained Bert large finetuned with SQuAD Dataset for Q&A

pretrained_bert_version = 'bert-large-uncased-whole-word-masking-finetuned-squad'

model_bert = BertForQuestionAnswering.from_pretrained(pretrained_bert_version)

model_bert_tokenizer = BertTokenizer.from_pretrained(pretrained_bert_version)



### Load pretrained AlBert xlarge finetuned with SQuAD Dataset for Q&A

model_name_path = "ktrapeznikov/albert-xlarge-v2-squad-v2"

model_albert = AlbertForQuestionAnswering.from_pretrained(model_name_path)

model_albert_tokenizer = AlbertTokenizer.from_pretrained(model_name_path)



# If there's a GPU available...

if torch.cuda.is_available():    



    # Tell PyTorch to use the GPU.    

    device = torch.device("cuda")



    print('There are %d GPU(s) available.' % torch.cuda.device_count())



    print('We will use the GPU:', torch.cuda.get_device_name(0))

    model_bert.cuda()

    model_albert.cuda()

# If not...

else:

    print('No GPU available, using the CPU instead.')

    device = torch.device("cpu")
# Test result of Bert and Albert with one task and paper abstract:

question = "which movement strategy can efficiently prevent secondary transmission in community settings?"

corpus_text = "Time variations in transmission potential have rarely been examined with regard to pandemic influenza. This paper reanalyzes the temporal distribution of pandemic influenza in Prussia, Germany, from 1918-19 using the daily numbers of deaths, which totaled 8911 from 29 September 1918 to 1 February 1919, and the distribution of the time delay from onset to death in order to estimate the effective reproduction number, Rt, defined as the actual average number of secondary cases per primary case at a given time..A discrete-time branching process was applied to back-calculated incidence data, assuming three different serial intervals (i.e. 1, 3 and 5 days). The estimated reproduction numbers exhibited a clear association between the estimates and choice of serial interval; i.e. the longer the assumed serial interval, the higher the reproduction number. Moreover, the estimated reproduction numbers did not decline monotonically with time, indicating that the patterns of secondary transmission varied with time. These tendencies are consistent with the differences in estimates of the reproduction number of pandemic influenza in recent studies; high estimates probably originate from a long serial interval and a model assumption about transmission rate that takes no account of time variation and is applied to the entire epidemic curve..The present findings suggest that in order to offer robust assessments it is critically important to clarify in detail the natural history of a disease (e.g. including the serial interval) as well as heterogeneous patterns of transmission. In addition, given that human contact behavior probably influences transmissibility, individual countermeasures (e.g. household quarantine and maskwearing) need to be explored to construct effective non-pharmaceutical interventions."





def quick_answer_test(question, corpus, model, tokenizer, device, sep="", show_tokens=False):

    if str(model).startswith('Albert'):

        model_name='Albert'

        sep = '▁'

    elif str(model).startswith('Bert'):

        model_name='Bert'

        sep = ' ##'

    else:

        model_name='Unknown'

        

    answer_text = corpus

    input_ids = tokenizer.encode(question, answer_text)

    print(f'The {model_name} tokenizer find the input has a total of {len(input_ids)} tokens.')



    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    if show_tokens: 

        for token, id in zip(tokens, input_ids):



            # If this is the [SEP] token, add some space around it to make it stand out.

            if id == tokenizer.sep_token_id:

                print('')



            # Print the token string and its ID in two columns.

            print('{:<12} {:>6,}'.format(token, id))



            if id == tokenizer.sep_token_id:

                print('')

    

    # Search the input_ids for the first instance of the `[SEP]` token.

    sep_index = input_ids.index(model_bert_tokenizer.sep_token_id)



    # The number of segment A tokens includes the [SEP] token istelf.

    num_seg_a = sep_index + 1



    # The remainder are segment B.

    num_seg_b = len(input_ids) - num_seg_a



    # Construct the list of 0s and 1s.

    segment_ids = [0]*num_seg_a + [1]*num_seg_b



    # There should be a segment_id for every input token.

    assert len(segment_ids) == len(input_ids)

    

    # device = torch.device("cpu")

    input_ids = torch.tensor(input_ids).to(device)

    segment_ids = torch.tensor(segment_ids).to(device)

    

    start_scores, end_scores = model(input_ids.reshape(1,-1), # The tokens representing our input text.

                                 token_type_ids=segment_ids.reshape(1,-1)) # The segment IDs to differentiate question from answer_text

    # Find the tokens with the highest `start` and `end` scores.

    answer_start = torch.argmax(start_scores)

    answer_end = torch.argmax(end_scores)



    # Combine the tokens in the answer and print it out.

    answer = ' '.join(tokens[answer_start:answer_end+1]).replace(sep, ' ').replace(' ,',',').replace(' .','.')

    

    return answer



print('Answer generated from Albert:', quick_answer_test(question, corpus_text, model_albert, model_albert_tokenizer, device=device))

print('-'*20)

print('Answer generated from Bert:', quick_answer_test(question, corpus_text, model_bert, model_bert_tokenizer, device=device))
### Function used to quickly compare the result of Bert vs. Albert 



def question_answering_method(task, corpus, abstract_or_not = True, model = model_albert, tokenizer=model_albert_tokenizer):

    '''

    Check whether one specific task could be handled with by specific corpus list, and give back the candidate answers for the specific task

    

    @Params:

        task: str

            String type variable used to describe the task or question

        

        corpus: pandas.DataFrame

            DataFrame contains each paper's informaton about Title, Abstract, Journal, Author, ID and so on

            

        abstract_or_not: bool

            Indicate whether to use abstract or just use title to find answer

            

        model: transformer model, by default is albert

        

        tokenizer: transformer tokenizer, by default is alberttokenizer

        

    return:

        corpus: pandas.DataFrame

            Updated corpus  dataframe has answer, score, can_handle_flag added for each paper

            

    '''

    corpus['start_score_max_index'] = np.zeros(corpus.shape[0])

    corpus['start_score_max'] = np.zeros(corpus.shape[0])

    corpus['can_handle_flag'] = np.zeros(corpus.shape[0])

    corpus['end_score_max_index'] = np.zeros(corpus.shape[0])

    corpus['end_score_max'] = np.zeros(corpus.shape[0])

    corpus['answer'] = np.array(['']*corpus.shape[0])

    corpus['start_score_prob'] = np.array(corpus.shape[0])

    

    if str(model).startswith('Albert'):

        sep = '▁'

    elif str(model).startswith('Bert'):

        sep = ' ##'

    else: 

        print('Model given is not supported right now')

        return 

    for row_nu in range(corpus.shape[0]):

        tmp = corpus.iloc[row_nu]

        answer_text = ""

        answer_text += tmp['title_raw'] if not pd.isnull(tmp['title_raw']) else ""

        answer_text += tmp['abstract_raw'] if abstract_or_not and not pd.isnull(tmp['abstract_raw']) else ""



        # Apply the tokenizer to the input text, treating them as a text-pair.

        input_ids = tokenizer.encode(task, answer_text, max_length=512)



        tokens = tokenizer.convert_ids_to_tokens(input_ids)



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



        # Move the target data to GPU

        input_ids = torch.tensor(input_ids).to(device)

        segment_ids = torch.tensor(segment_ids).to(device)



        # Run our example through the model.

        start_scores, end_scores = model(input_ids.reshape(1,-1), # The tokens representing our input text.

                                 token_type_ids=segment_ids.reshape(1,-1)) # The segment IDs to differentiate question from answer_text

        



        start_scores = start_scores.detach().to('cpu')

        end_scores = end_scores.detach().to('cpu')



        answer_start = torch.argmax(start_scores)

        answer_end = torch.argmax(end_scores)



        if answer_start.item() > sep_index+1 and answer_end.item() >= answer_start.item() and answer_end.item() < len(tokens) - 1:

            corpus.loc[row_nu, 'can_handle_flag'] = 1

            corpus.loc[row_nu, 'start_score_max_index'] = answer_start.item()

            corpus.loc[row_nu, 'end_score_max_index'] = answer_end.item()

            corpus.loc[row_nu, 'start_score_max'] = torch.max(start_scores).item()

            corpus.loc[row_nu, 'end_score_max'] = torch.max(end_scores).item()

            if str(model).startswith('Albert'):

                corpus.loc[row_nu, 'answer'] =  ''.join(tokens[answer_start:answer_end+1]).replace('▁', ' ').replace(' ,',',').replace(' .','.')

            else:

                corpus.loc[row_nu, 'answer'] =  ' '.join(tokens[answer_start:answer_end+1]).replace(sep, '').replace(' ,',',').replace(' .','.')

            corpus.loc[row_nu, 'start_score_prob'] = scipy.special.softmax(start_scores.reshape(-1).detach().numpy()).max()

        else:

            corpus.loc[row_nu, 'can_handle_flag'] = -1

            corpus.loc[row_nu, 'start_score_max_index'] = answer_start.item()

            corpus.loc[row_nu, 'end_score_max_index'] = answer_end.item()

            corpus.loc[row_nu, 'start_score_max'] = torch.max(start_scores).item()

            corpus.loc[row_nu, 'end_score_max'] = torch.max(end_scores).item()

            if str(model).startswith('Albert'):

                corpus.loc[row_nu, 'answer'] =  ''.join(tokens[answer_start:answer_end+1]).replace('▁', ' ').replace(' ,',',').replace(' .','.')

            else:

                corpus.loc[row_nu, 'answer'] =  ' '.join(tokens[answer_start:answer_end+1]).replace(sep, '').replace(' ,',',').replace(' .','.')

            corpus.loc[row_nu, 'start_score_prob'] = scipy.special.softmax(start_scores.reshape(-1).detach().numpy()).max()

    return corpus.copy()



def answer_check(ans, return_answer=False, show = True):

    ans = ans[ans['can_handle_flag'] == 1].sort_values('start_score_prob', ascending=False)

    answers = []

    

    for i in ans['answer']:

        if show:

            print(i.replace('<unk>',''))

        answers.append(i.replace('<unk>',''))

    if return_answer:

        return answers
albert_bio_answers = question_answering_method(task='which are movement control strategies can efficiently prevent secondary transmission in community settings?', corpus=biorxiv_medrxiv, abstract_or_not = True, model = model_albert, tokenizer=model_albert_tokenizer)

bert_bio_answers = question_answering_method(task='which are movement control strategies can efficiently prevent secondary transmission in community settings?', corpus=biorxiv_medrxiv, abstract_or_not = True, model = model_bert, tokenizer=model_bert_tokenizer)
print('Answer of Albert on Bioxriv dataset')

print('='*20)

answer_check(albert_bio_answers)
print('Answer of Bert on Bioxriv dataset:')

print('='*20)

answer_check(bert_bio_answers)
class EmbeddingSearch(object):

    

    def __init__(self, config={}):

        '''

        @Param: 

            config: dict that may contains following keys 

                model: default is AlBertForQuestionAnswering

                tokenizer: default is AlBertTokenizer 

                pretrained_version: default is 'bert-large-uncased-whole-word-masking-finetuned-squad'

                dataset: pandas.DataFrame

                gpu: bool,use or not, by default is false

        return:

            Embedding object which could be used to batchly calculate candidate answers for task lisk

        '''

        self.model = config.get('model') or AlbertForQuestionAnswering # from transformer

        self.tokenizer = config.get('tokenizer') or AlbertTokenizer # from transformer

        self.pretrained_version = config.get('pretrained_version') or 'ktrapeznikov/albert-xlarge-v2-squad-v2'

        try:

            self.model = self.model.from_pretrained(self.pretrained_version)

            self.tokenizer = self.tokenizer.from_pretrained(self.pretrained_version)

        except:

            return f'Pretrained-weights importing fail'

        

#         assert config.get('dataset')

        ## Tell whether dataset is qualified

        

        self.dataset = config.get('dataset') or None

        

        self.gpu_flag = config.get('gpu') or True

        

        if self.gpu_flag:

            if torch.cuda.is_available():    



                # Tell PyTorch to use the GPU.    

                self.device = torch.device("cuda")



                print('There are %d GPU(s) available.' % torch.cuda.device_count())



                print('We will use the GPU:', torch.cuda.get_device_name(0))

                self.model.cuda()

            # If not...

            else:

                print('No GPU available, using the CPU instead.')

                self.device = torch.device("cpu")

                

        self.trained_saved_inside = {}      

        if str(self.model).startswith('Albert'):

            self.model_name='albert'

        elif str(self.model).startswith('Bert'):

            self.modle_name='bert'

        else:

            print('Model unknown')

            return

    def qa_test(self, question, answer_text, show_tokens=False):

        '''

        

        '''

        # Apply the tokenizer to the input text, treating them as a text-pair.

        input_ids = self.tokenizer.encode(question, answer_text, max_length=512)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        

        print('The input has a total of {:} tokens.'.format(len(input_ids)))

        

        if show_tokens:

            for token, id in zip(tokens, input_ids):

                # If this is the [SEP] token, add some space around it to make it stand out.

                if id == self.tokenizer.sep_token_id:

                    print('')



                # Print the token string and its ID in two columns.

                print('{:<12} {:>6,}'.format(token, id))



                if id == self.tokenizer.sep_token_id:

                    print('')

        sep_index = input_ids.index(tokenizer.sep_token_id)



        # The number of segment A tokens includes the [SEP] token istelf.

        num_seg_a = sep_index + 1



        # The remainder are segment B.

        num_seg_b = len(input_ids) - num_seg_a



        # Construct the list of 0s and 1s.

        segment_ids = [0]*num_seg_a + [1]*num_seg_b



        # There should be a segment_id for every input token.

        assert len(segment_ids) == len(input_ids)

        

        # device = torch.device("cpu")

        input_ids = torch.tensor(input_ids).to(self.device)

        segment_ids = torch.tensor(segment_ids).to(self.device)

        

        start_scores, end_scores = self.model(input_ids.reshape(1,-1), # The tokens representing our input text.

                                 token_type_ids=segment_ids.reshape(1,-1))

        start_scores = start_scores.to('cpu')

        end_scores = end_scores.to('cpu')

        

        answer_start = torch.argmax(start_scores)

        answer_end = torch.argmax(end_scores)



        # Combine the tokens in the answer and print it out.

        answer = ' '.join(tokens[answer_start:answer_end+1])



        return 'Answer: "' + answer + '"'

        

    def find_task_answer_from_corpus(self, task, corpus, abstract_or_not = True, save_inside = True, save_to_file=True, dataset_name='undefined', task_index='undefined'):

        '''

        Check whether one specific task could be handled with one specific corpus

        '''

        corpus = corpus if corpus is not None else self.dataset

        corpus['start_score_max_index'] = np.zeros(corpus.shape[0])

        corpus['start_score_max'] = np.zeros(corpus.shape[0])

        corpus['can_handle_flag'] = np.zeros(corpus.shape[0])

        corpus['end_score_max_index'] = np.zeros(corpus.shape[0])

        corpus['end_score_max'] = np.zeros(corpus.shape[0])

        corpus['answer'] = np.array(['']*corpus.shape[0])

        corpus['start_score_prob'] = np.array(corpus.shape[0])

        

        if str(self.model).startswith('Albert'):

            sep = '▁'

        elif str(self.model).startswith('Bert'):

            sep = ' ##'

        else: 

            print('Model given is not supported right now')

            return 

        

        for row_nu in range(corpus.shape[0]):

            tmp = corpus.iloc[row_nu]

            answer_text = ""

            answer_text += tmp['title_raw'] if not pd.isnull(tmp['title_raw']) else ""

            answer_text += tmp['abstract_raw'] if abstract_or_not and not pd.isnull(tmp['abstract_raw']) else ""

            

            # Apply the tokenizer to the input text, treating them as a text-pair.

            input_ids = self.tokenizer.encode(task, answer_text, max_length=512)



            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)



            # Search the input_ids for the first instance of the `[SEP]` token.

            sep_index = input_ids.index(self.tokenizer.sep_token_id)



            # The number of segment A tokens includes the [SEP] token istelf.

            num_seg_a = sep_index + 1



            # The remainder are segment B.

            num_seg_b = len(input_ids) - num_seg_a



            # Construct the list of 0s and 1s.

            segment_ids = [0]*num_seg_a + [1]*num_seg_b



            # There should be a segment_id for every input token.

            assert len(segment_ids) == len(input_ids)



            # Move the target data to GPU

            input_ids = torch.tensor(input_ids).to(self.device)

            segment_ids = torch.tensor(segment_ids).to(self.device)



            # Run our example through the model.

            start_scores, end_scores = self.model(input_ids.reshape(1,-1), # The tokens representing our input text.

                                     token_type_ids=segment_ids.reshape(1,-1)) # The segment IDs to differentiate question from answer_text





            start_scores = start_scores.detach().to('cpu')

            end_scores = end_scores.detach().to('cpu')



            answer_start = torch.argmax(start_scores)

            answer_end = torch.argmax(end_scores)



            if answer_start.item() > sep_index+1 and answer_end.item() >= answer_start.item() and answer_end.item() < len(tokens) - 1:

                corpus.loc[row_nu, 'can_handle_flag'] = 1

                corpus.loc[row_nu, 'start_score_max_index'] = answer_start.item()

                corpus.loc[row_nu, 'end_score_max_index'] = answer_end.item()

                corpus.loc[row_nu, 'start_score_max'] = torch.max(start_scores).item()

                corpus.loc[row_nu, 'end_score_max'] = torch.max(end_scores).item()

                if str(self.model).startswith('Albert'):

                    corpus.loc[row_nu, 'answer'] =  ''.join(tokens[answer_start:answer_end+1]).replace('▁', ' ').replace(' ,',',').replace(' .','.')

                else:

                    corpus.loc[row_nu, 'answer'] =  ' '.join(tokens[answer_start:answer_end+1]).replace(sep, '').replace(' ,',',').replace(' .','.')

                corpus.loc[row_nu, 'start_score_prob'] = scipy.special.softmax(start_scores.reshape(-1).detach().numpy()).max()

            else:

                corpus.loc[row_nu, 'can_handle_flag'] = -1

                corpus.loc[row_nu, 'start_score_max_index'] = answer_start.item()

                corpus.loc[row_nu, 'end_score_max_index'] = answer_end.item()

                corpus.loc[row_nu, 'start_score_max'] = torch.max(start_scores).item()

                corpus.loc[row_nu, 'end_score_max'] = torch.max(end_scores).item()

                if str(self.model).startswith('Albert'):

                    corpus.loc[row_nu, 'answer'] =  ''.join(tokens[answer_start:answer_end+1]).replace('▁', ' ').replace(' ,',',').replace(' .','.')

                else:

                    corpus.loc[row_nu, 'answer'] =  ' '.join(tokens[answer_start:answer_end+1]).replace(sep, '').replace(' ,',',').replace(' .','.')

                corpus.loc[row_nu, 'start_score_prob'] = scipy.special.softmax(start_scores.reshape(-1).detach().numpy()).max()

                

        self.trained_saved_inside[task_index] = corpus.copy() if save_inside else None

        if save_to_file:

            corpus.to_excel(f'{task_index}_{dataset_name}_{self.model_name}.xlsx')

        return corpus

    

    def batch_find_task_answer_from_corpus(self,corpus, task_list, abstract_or_not = True, save_inside = True, show_output=True,answer_check_algo=answer_check,dataset_name='undefined'):

        '''

        Calculate the answer of task, based on task_list with the given model



        Param: 

            model:

                transformer-Bert/Albert

            tokenizer:



            corpus: pandas.DataFrame



            task_list: Dict 

                {'task_index': 'task_detail'}



            show_corpus: bool

                True for default



            output: bool

                True for default



            answer_check_algo: function

                function given to check the result after calulation



        Return:

            res: Dict

                {'task_index': pandas.DataFrame}

        '''

        start_function = time.time()

        res = {task: None for task in task_list.keys()}





        for task_index, task in task_list.items():

            tmp_res = self.find_task_answer_from_corpus(task, corpus, abstract_or_not = abstract_or_not, save_inside = save_inside, save_to_file=True, dataset_name=dataset_name, task_index=task_index)

            if not save_inside:

                res[task_index] = tmp_res.copy() #.copy is an absolute must

            print(f'===================={task} has finished====================')

            if show_output:

                answer_check(tmp_res)

        end_function = time.time()

        print(f'Total Time Consumed is {end_function-start_function}')

        if not save_inside:

            return res

        else: 

            return self.trained_saved_inside

    

    def trained_saved_inside_clear():

        self.trained_saved_inside = {}

        

    @staticmethod

    def question_answering_method(task, corpus, abstract_or_not = True, model = model_albert, tokenizer=model_albert_tokenizer):

        '''

        Check whether one specific task could be handled with one specific corpus

        '''

        corpus['start_score_max_index'] = np.zeros(corpus.shape[0])

        corpus['start_score_max'] = np.zeros(corpus.shape[0])

        corpus['can_handle_flag'] = np.zeros(corpus.shape[0])

        corpus['end_score_max_index'] = np.zeros(corpus.shape[0])

        corpus['end_score_max'] = np.zeros(corpus.shape[0])

        corpus['answer'] = np.array(['']*corpus.shape[0])

        corpus['start_score_prob'] = np.array(corpus.shape[0])



        if str(model).startswith('Albert'):

            sep = '▁'

        elif str(model).startswith('Bert'):

            sep = ' ##'

        else: 

            print('Model given is not supported right now')

            return 

        for row_nu in range(corpus.shape[0]):

            tmp = corpus.iloc[row_nu]

            answer_text = ""

            answer_text += tmp['title_raw'] if not pd.isnull(tmp['title_raw']) else ""

            answer_text += tmp['abstract_raw'] if abstract_or_not and not pd.isnull(tmp['abstract_raw']) else ""



            # Apply the tokenizer to the input text, treating them as a text-pair.

            input_ids = tokenizer.encode(task, answer_text, max_length=512)



            tokens = tokenizer.convert_ids_to_tokens(input_ids)



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



            # Move the target data to GPU

            input_ids = torch.tensor(input_ids).to(device)

            segment_ids = torch.tensor(segment_ids).to(device)



            # Run our example through the model.

            start_scores, end_scores = model(input_ids.reshape(1,-1), # The tokens representing our input text.

                                     token_type_ids=segment_ids.reshape(1,-1)) # The segment IDs to differentiate question from answer_text





            start_scores = start_scores.detach().to('cpu')

            end_scores = end_scores.detach().to('cpu')



            answer_start = torch.argmax(start_scores)

            answer_end = torch.argmax(end_scores)



            if answer_start.item() > sep_index+1 and answer_end.item() >= answer_start.item() and answer_end.item() < len(tokens) - 1:

                corpus.loc[row_nu, 'can_handle_flag'] = 1

                corpus.loc[row_nu, 'start_score_max_index'] = answer_start.item()

                corpus.loc[row_nu, 'end_score_max_index'] = answer_end.item()

                corpus.loc[row_nu, 'start_score_max'] = torch.max(start_scores).item()

                corpus.loc[row_nu, 'end_score_max'] = torch.max(end_scores).item()

                if str(model).startswith('Albert'):

                    corpus.loc[row_nu, 'answer'] =  ''.join(tokens[answer_start:answer_end+1]).replace('▁', ' ').replace(' ,',',').replace(' .','.')

                else:

                    corpus.loc[row_nu, 'answer'] =  ' '.join(tokens[answer_start:answer_end+1]).replace(sep, '').replace(' ,',',').replace(' .','.')

                corpus.loc[row_nu, 'start_score_prob'] = scipy.special.softmax(start_scores.reshape(-1).detach().numpy()).max()

            else:

                corpus.loc[row_nu, 'can_handle_flag'] = -1

                corpus.loc[row_nu, 'start_score_max_index'] = answer_start.item()

                corpus.loc[row_nu, 'end_score_max_index'] = answer_end.item()

                corpus.loc[row_nu, 'start_score_max'] = torch.max(start_scores).item()

                corpus.loc[row_nu, 'end_score_max'] = torch.max(end_scores).item()

                if str(model).startswith('Albert'):

                    corpus.loc[row_nu, 'answer'] =  ''.join(tokens[answer_start:answer_end+1]).replace('▁', ' ').replace(' ,',',').replace(' .','.')

                else:

                    corpus.loc[row_nu, 'answer'] =  ' '.join(tokens[answer_start:answer_end+1]).replace(sep, '').replace(' ,',',').replace(' .','.')

                corpus.loc[row_nu, 'start_score_prob'] = scipy.special.softmax(start_scores.reshape(-1).detach().numpy()).max()

        return corpus
###  EmbeddingSearch usage example 

task_list = {

    'task_1': "What is known about transmission, incubation, and environmental stability of corona virus?",

    'task_2': "Range of incubation periods for the disease in humans",

    'task_3': "Persistence of corona virus on surfaces of different materials",

    'task_4': "What are natural history of the virus and shedding of it from an infected person?",

    'task_5': "What is corona virus' seasonality of transmission?",

    'task_6': "What are the implementation of diagnostics and products to improve clinical processes?",

    'task_7': "What is corona virus' immune response and immunity?", 

    'task_8': "Which are movement control strategies can efficiently prevent secondary transmission in health care?",

    'task_9': "Which are movement control strategies can efficiently prevent secondary transmission in community settings",

    'task_10': "What is the role of the environment in transmission"

}

bio_embedding_search = EmbeddingSearch()

### Not run considering the time consuming(5~10 minutes)

# bio_embedding_search.batch_find_task_answer_from_corpus(corpus=biorxiv_medrxiv, task_list=task_list, abstract_or_not = True, save_inside = True, show_output=True,answer_check_algo=answer_check,dataset_name='biorxv')
### Load the file contains all the pretrained result

boosted_all = pd.read_excel('../input/keywords-addon-boosted/v2valid_all.xlsx')

### Fillin possible empty cell with 'UK'

boosted_all.fillna('UK', inplace=True)
for i in range(1,11):

    cur_task = 'task_'+str(i)

    boosted_all[cur_task+'_answer'] = boosted_all[cur_task+'_answer'].apply(lambda item:item.strip().capitalize())

    tmp_abs_length = boosted_all['abstract_raw'].apply(lambda x:len(x))

    tmp_title_length = boosted_all['title_raw'].apply(lambda x:len(x))

    tmp_length = tmp_abs_length + tmp_title_length

    

    tmp = tmp_length.apply(lambda x:max(np.sqrt(x), 1))

    tmp_mean = tmp.mean()

    tmp_mean_adjusted = tmp.apply(lambda x:min(x, tmp_mean))

    

    boosted_all[cur_task+'_score_normalized'] = boosted_all[cur_task+'_start_score_prob'] * (tmp_abs_length+tmp_title_length).apply(lambda x:max(np.sqrt(x), 1))



    

for i in range(1,11):

    cur_task = 'task_'+str(i)   

    boosted_all[cur_task+'_score_normalized'] = boosted_all[cur_task+'_start_score_prob'] * tmp

    boosted_all[cur_task+'_score_normalized_adjusted'] = boosted_all[cur_task+'_start_score_prob'] * tmp_mean_adjusted



### Round the score result for display

for i in range(1,11):

    cur_task = 'task_'+str(i)

    boosted_all[cur_task+'_start_score_prob'] = boosted_all[cur_task+'_start_score_prob'].apply(lambda item:round(item,2))

    boosted_all[cur_task+'_score_normalized'] = boosted_all[cur_task+'_score_normalized'].apply(lambda item:round(item,2))

    boosted_all[cur_task+'_score_normalized_adjusted'] = boosted_all[cur_task+'_score_normalized_adjusted'].apply(lambda item:round(item,2))
###

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, TableColumn, StringFormatter, Circle, Div, Paragraph, Select,DataTable

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.models.widgets import Slider,Dropdown

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure

from bokeh.layouts import column, row, gridplot

##############

boosted_all_dict = boosted_all.to_dict()

task_to_index = {value: key for key, value in task_list.items()}

output_notebook()

data = {

    'answer': list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['task_1_answer']),

    'score':list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['task_1_start_score_prob']),

    'index_from_original': list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1].index)

}



detail_data = ColumnDataSource({

    'title':list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['title_raw']),

    'answer': list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['task_1_answer']),

    'abstract':list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['abstract_raw']),

    'id': list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['paper_id'])

})





source = ColumnDataSource(data)



columns = [

    TableColumn(field='answer', title='Possible Answer', formatter=StringFormatter(font_style="bold")),

    TableColumn(field='score', title='Score', width=5)

]

data_table =  DataTable(source=source, columns=columns, selectable=True, index_header="", width=500,height=600, fit_columns=True, scroll_to_selection=True, height_policy='auto', editable=True)



select = Select(title='Task list', value=list(task_list.values())[0], options=list(task_list.values()), height=50, width=420)



scoretype_select = Select(title='Score type', value='standard', options=['standard', 'normalized', 'norm+adjust'], width=80, height=50)





scoretype_callback = CustomJS(args = dict(source=source, select = select, task_to_index=task_to_index, boosted_all_dict=boosted_all_dict), code="""

    

    Object.filter = function( obj, predicate) {

        var result = {};

        for (let key in obj) {

            if (obj.hasOwnProperty(key) && predicate(obj[key])) {

                result[key] = obj[key];

            }

        }

        return result;

    };

    

    Object.batch_select = function(obj, list){

        var res = [];

        for(let key of list){

            res.push(obj[key]);

        }

        return res;

    };

    

    var selected_type = cb_obj.value;

    var selected_task = select.value;

    var selected_index = task_to_index[selected_task];

    var target_indexs = Object.keys(Object.filter(boosted_all_dict[selected_index+'_can_handle_flag'], item => item == 1));

    var answer = Object.batch_select(boosted_all_dict[selected_index+'_answer'], target_indexs);

    var score = [];

    if (selected_type == 'standard'){

        score = Object.batch_select(boosted_all_dict[selected_index+'_start_score_prob'], target_indexs);

    }else if(selected_type == 'normalized'){

        score = Object.batch_select(boosted_all_dict[selected_index+'_score_normalized'], target_indexs);

    }else if(selected_type == 'norm+adjust'){

        score = Object.batch_select(boosted_all_dict[selected_index+'_score_normalized_adjusted'], target_indexs);

    }

    var title = Object.batch_select(boosted_all_dict['title_raw'], target_indexs);

    var abstract = Object.batch_select(boosted_all_dict['abstract_raw'], target_indexs);

    var id = Object.batch_select(boosted_all_dict['paper_id'], target_indexs);

    source.data = {

        'answer': answer,

        'score': score,

        'index_from_original': target_indexs

    };

    detail_data.data = {

       'title':title,

       'answer':answer,

       'abstract':abstract,

       'id': id

    };

    source.change.emit();

    detail_data.change.emit();

""")





task_select_callback = CustomJS(args=dict(source=source,detail_data=detail_data, task_to_index=task_to_index, scoretype_select=scoretype_select, boosted_all_dict=boosted_all_dict), code="""

    Object.filter = function( obj, predicate) {

        var result = {};

        for (let key in obj) {

            if (obj.hasOwnProperty(key) && predicate(obj[key])) {

                result[key] = obj[key];

            }

        }

        return result;

    };

    

    Object.batch_select = function(obj, list){

        var res = [];

        for(let key of list){

            res.push(obj[key]);

        }

        return res;

    };

    var selected_type = scoretype_select.value;

    var selected_task = cb_obj.value;

    var selected_index = task_to_index[selected_task];

    var target_indexs = Object.keys(Object.filter(boosted_all_dict[selected_index+'_can_handle_flag'], item => item == 1));

    var answer = Object.batch_select(boosted_all_dict[selected_index+'_answer'], target_indexs);

    var score = [];

    if (selected_type == 'standard'){

        score = Object.batch_select(boosted_all_dict[selected_index+'_start_score_prob'], target_indexs);

    }else if(selected_type == 'normalized'){

        score = Object.batch_select(boosted_all_dict[selected_index+'_score_normalized'], target_indexs);

    }else if(selected_type == 'norm+adjust'){

        score = Object.batch_select(boosted_all_dict[selected_index+'_score_normalized_adjusted'], target_indexs);

    }

    var title = Object.batch_select(boosted_all_dict['title_raw'], target_indexs);

    var abstract = Object.batch_select(boosted_all_dict['abstract_raw'], target_indexs);

    var id = Object.batch_select(boosted_all_dict['paper_id'], target_indexs);

    source.data = {

        'answer': answer,

        'score': score,

        'index_from_original': target_indexs

    };

    detail_data.data = {

       'title':title,

       'answer':answer,

       'abstract':abstract,

       'id':id

    };

    """) 

    

paper_detail = Div(text="Paper info shows here", margin=(50,0,10,0), style={'border':'1px solid black', 'width':'400px', 'height':'600px', 'padding':'20px', 'border-top-left-radius':'5px','border-top-right-radius':'5px','text-align':'center','overflow-y':'auto'})

select_callback=CustomJS(args=dict(source=source, div=paper_detail, detail_data=detail_data), code="""

    var selection_index=source.selected.indices[0];

    var answer = detail_data.data['answer'][selection_index];

    var title = detail_data.data['title'][selection_index];

    var abstract = detail_data.data['abstract'][selection_index] || "";

    var idlink = "";

    var id = detail_data.data['id'][selection_index];

    if(id.startsWith("PMC")){

        idlink = "https://www.ncbi.nlm.nih.gov/pmc/articles/" + id;

    }else{

        idlink = 'https://www.semanticscholar.org/paper/' + id;

    }

    var index = 0;

    if(title.toLowerCase().indexOf(answer.toLowerCase()) !== -1){

        index = title.toLowerCase().indexOf(answer.toLowerCase())

        title = title.slice(0,index) + `<a href=${idlink} target="_blank" style='text-decoration:none; background: linear-gradient(90deg, rgb(147, 222, 241), rgb(147, 222, 23)); background-image: linear-gradient(90deg, rgb(147, 222, 241), rgb(147, 222, 23)); background-position-x: initial; background-position-y: initial;background-size: initial; background-repeat-x: initial; background-repeat-y: initial; background-attachment: initial;background-origin: initial; background-clip: initial; background-color: initial; margin: 0 0.25em; line-height: 1.5; padding: 0px 3px !important; border-radius: 5rem !important;'><strong>${title.slice(index, index + answer.length)}</strong></a>` + title.slice(index + answer.length);

    }

    if(abstract.toLowerCase().indexOf(answer.toLowerCase()) !== -1){

        index = abstract.toLowerCase().indexOf(answer.toLowerCase());

        abstract = abstract.slice(0, index) + `<a style='text-decoration:none; background: linear-gradient(90deg, rgb(147, 222, 241), rgb(147, 222, 23)); background-image: linear-gradient(90deg, rgb(147, 222, 241), rgb(147, 222, 23)); background-position-x: initial; background-position-y: initial;background-size: initial; background-repeat-x: initial; background-repeat-y: initial; background-attachment: initial;background-origin: initial; background-clip: initial; background-color: initial; margin: 0 0.25em; line-height: 1.5; padding: 0px 3px !important; border-radius: 5rem !important;'><strong>${abstract.slice(index, index + answer.length)}</strong></a>` + abstract.slice(index + answer.length);

    }

    div.text = `<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">`;

    div.text +=`<h2><a href=${idlink} target="_blank" style='text-decoration:none'><center>${title}&nbsp<i class='fa fa-external-link' style='font-size:15px'></i></center></a></h2>`;

    div.text += `<p><strong><center>Answer: <a style='color:blue; font-style:italic'>${detail_data.data['answer'][selection_index]}</a></center></strong></p>`;

    div.text += `<br>`;

    div.text += `<p><strong><center>Abstract</center></strong></p>`;

    div.text += `<p>${abstract}</p>`;



""")

#    border:1px solid black; border-top-left-radius: 10px; border-top-right-radius: 10px;



source.selected.js_on_change('indices', select_callback)

select.js_on_change('value', task_select_callback)



scoretype_select.js_on_change('value', scoretype_callback)

output_file('res.html')



title = Div(text = "<h1><center>Covid-19 Tasks' Candidate Answers<center></h1>")

show(column(title, row(column(row(select,scoretype_select), data_table),paper_detail)))
### Noun phrase generation function

from wordcloud import WordCloud

##########################

def noun_phrase_generation(boosted_all):

    '''

    

    '''

    res = {i:[] for i in task_list}

    for task_index in task_list:

        task_answers = boosted_all[boosted_all[task_index+'_can_handle_flag']==1][task_index+'_answer']

    ##

        grammar = "NP: {<DT|VB>?<JJ>*<NN|NNS|NNP.*>}"

        cp = nltk.RegexpParser(grammar)

        for answer in task_answers:

            answer_token = nltk.word_tokenize(answer)

            answer_tag = nltk.pos_tag(answer_token)

            answer_chunked = cp.parse(answer_tag)

            for chunk in answer_chunked:

                if str(chunk)[1:-1].startswith('NP'):

                    try:

                        str_chunk_list = str(chunk).split(' ')[1:]

                        remove_sep = []

                        for i in str_chunk_list:

                            if '/' in i:

                                remove_sep.append(i[0:i.index('/')])

                            else:

                                continue

                        res[task_index].append(' '.join(remove_sep))

                    except:

                        return chunk

    return res



### noun_phrase_generation function usage example and wordcloud generation

res_noun = noun_phrase_generation(boosted_all)

wordcloud_noun_phrase_all = {i:None for i in task_list.keys()}

for key in wordcloud_noun_phrase_all.keys():

    wordcloud_noun_phrase_all[key] = WordCloud(background_color="white",width=1000, height=800, margin=2).generate(" ".join(res_noun[key]))

    wordcloud_noun_phrase_all[key].to_file(f'./{key}_all_wordcloud.png')
from bokeh.layouts import gridplot

output_notebook()

wordcloud_vis = {key: None for key in task_list.keys()}

for task_index in task_list.keys():

    worldcloud_source = ColumnDataSource({

        'url': [f'./{task_index}_all_wordcloud.png']

    })

    p = figure(x_range=(0,1), y_range=(0,1), width=400, height=400)

    p.image_url(url='url', x=0, y=1, w=1, h=0.8, source=worldcloud_source)

    p.xaxis.visible = False

    p.yaxis.visible = False

    wordcloud_vis[task_index] = p



grid = gridplot(list(wordcloud_vis.values()), ncols=2, plot_width=300, plot_height=300)

show(grid)
### Generate Lexical feature

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics import pairwise

from scipy.sparse import hstack

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from random import shuffle

from gensim import models

import seaborn as sns

### title vectorizer

df_covid = biorxiv_medrxiv

df_covid.fillna('UK', inplace=True)

titles_vectorizer = CountVectorizer(max_df = 0.5, min_df = 2, stop_words = 'english', ngram_range=(1,1)) ### Unigram used here, could be fine-tunning

titles_vec = titles_vectorizer.fit_transform(df_covid['abstract_all'])

titles_simi = pairwise.cosine_similarity(titles_vec)

print(f'title_vec generation finished!')

### abstract vectorizer

abstracts_vectorizer = CountVectorizer(max_df = 0.5, min_df = 2, stop_words='english', ngram_range=(1, 1)) ### Unigram used here, could be fine-tunning

abstracts_vec = abstracts_vectorizer.fit_transform(df_covid['abstract_all'])

abstracts_simi = pairwise.cosine_similarity(abstracts_vec)

print(f'abstract_vec generation finished!')

### author vectorizer

authors_vectorizer = CountVectorizer(max_df = 0.9, min_df = 3, stop_words='english') ### Unigram used here, could be fine-tunning

authors_vec = authors_vectorizer.fit_transform(df_covid['authors'])

authors_simi = pairwise.cosine_similarity(authors_vec)

print(f'authors_vec generation finished!')

### body_text feature

body_text_vectorizer = CountVectorizer(max_df = 0.5, min_df = 2, stop_words='english', ngram_range=(1,3)) ### Uni to tri-grams used here

body_text_vec = body_text_vectorizer.fit_transform(df_covid['body_text'])

body_text_simi = pairwise.cosine_similarity(body_text_vec)

print(f'body_text_vec generation finished!')



### journal feature

# journal_vectorizer = CountVectorizer(max_df = 0.9, min_df = 1, stop_words='english')

# journal_vec = journal_vectorizer.fit_transform(df_covid['journal'])

# journal_simi = pairwise.cosine_similarity(journal_vec)

# print(f'journal_vec generation finished!')



### Combine tile, absract and author feature together

combine_vec = hstack((titles_vec, abstracts_vec, authors_vec, body_text_vec))

combine_simi = titles_simi*0.6 + abstracts_simi + authors_simi*0.5 + body_text_simi # the weights are adjustable

print(f'combine_vec and combine_simi generation finished!') 
# Doc2Vec for title

titles_sentences = []

label_index = 0

for title in df_covid['title_all']:

    sentence = models.doc2vec.TaggedDocument(words = title.replace('.', '').split(), tags = ['Title_%s' % label_index])

    titles_sentences.append(sentence)

    label_index += 1

model_doc2vec = models.Doc2Vec(alpha=0.025, min_alpha=0.025)  # use fixed learning rate

model_doc2vec.build_vocab(titles_sentences)

### Start to train own Doc2Vec

for epoch in range(20): 

    model_doc2vec.train(titles_sentences, total_examples=len(titles_sentences), epochs=1)

    model_doc2vec.alpha -= 0.001

    model_doc2vec.min_alpha = model_doc2vec.alpha

    shuffle(titles_sentences)

    print(f'-----------------------epoch {epoch} finish: title_doc2vec----------------------')



titles_doc2vec = [model_doc2vec.docvecs[i[1][0]] for i in titles_sentences]



### Dimensionality reduction for titles feature with T-SNE

tsne = TSNE(n_components = 2, init='pca', perplexity=100, random_state = 0)

np.set_printoptions(suppress = True)

title_2d = tsne.fit_transform(titles_doc2vec)



# Doc2Vec for abstract

abstracts_sentences = []

label_index = 0

for abstract in df_covid['abstract_all']:

    sentence = models.doc2vec.TaggedDocument(words = abstract.replace('.', '').split(), tags = ['Abstract_%s' % label_index])

    abstracts_sentences.append(sentence)

    label_index += 1

model_doc2vec = models.Doc2Vec(alpha=0.025, min_alpha=0.025)

model_doc2vec.build_vocab(abstracts_sentences)



for epoch in range(20): # run for 20 passes for better performance

    model_doc2vec.train(abstracts_sentences, total_examples=len(abstracts_sentences), epochs=1)

    model_doc2vec.alpha -= 0.001

    model_doc2vec.min_alpha = model_doc2vec.alpha

    shuffle(abstracts_sentences)

    print(f'-----------------------epoch {epoch} finish: abstract_doc2vec----------------------')

    

abstracts_doc2vec = [model_doc2vec.docvecs[i[1][0]] for i in abstracts_sentences]



### Dimensionality reduction for abstracts feature with T-SNE

abstract_2d = tsne.fit_transform(abstracts_doc2vec)



# Doc2Vec for combination of Title and Abstract

texts_sentences = []

label_index = 0

for (title, abstract) in zip(df_covid['title_all'], df_covid['abstract_all']):

    sentence = models.doc2vec.TaggedDocument(words = title.replace('.', '').split() + abstract.replace('.', '').split(), tags = ['Text_%s' % label_index])

    texts_sentences.append(sentence)

    label_index += 1

model_doc2vec = models.Doc2Vec(alpha=0.025, min_alpha=0.025) 

model_doc2vec.build_vocab(texts_sentences)

for epoch in range(10): # run for 20 passes for better performance

    model_doc2vec.train(texts_sentences, total_examples=len(texts_sentences), epochs=1)

    model_doc2vec.alpha -= 0.002

    model_doc2vec.min_alpha = model_doc2vec.alpha

    shuffle(texts_sentences)

    print(f'-----------------------epoch {epoch} finish: texts_doc2vec----------------------')



texts_doc2vec = [model_doc2vec.docvecs[i[1][0]] for i in texts_sentences]

text_2d = tsne.fit_transform(texts_doc2vec)
from igraph import Graph, plot

import networkx as nx



g = Graph()

g_nx = nx.Graph()

g.add_vertices(label_index)

combine_simi_edge = np.zeros((label_index, label_index))

for i in range(0, label_index):

    g_nx.add_node(i)

combine_simi_copy = combine_simi

for index in range(0, label_index):

    combine_simi_copy[index][index] = 0

flag_10percent_value = np.zeros(label_index)



for index1 in range(0, label_index):

        flag_10percent_index = int(0.1 * (label_index-1))

        flag_10percent_value[index1] = np.partition(combine_simi_copy[index1], int(-flag_10percent_index))[int(-flag_10percent_index)]

        for index2 in range(0, label_index):

            if combine_simi_copy[index1][index2] < flag_10percent_value[index1]:

                combine_simi_copy[index1][index2] = 0

                

degree = np.zeros(label_index)

for index in range(0, label_index):

    degree[index] = np.count_nonzero(combine_simi_copy[index])

count_edge = np.zeros(label_index)

combine_simi_final = np.zeros((label_index, label_index))



for index1 in range(0, label_index):

    count_edge[index1] = int(pow(degree[index1], 0.5)) + 1

    #if count_edge[index1] == 0:

    #	count_edge[index1] = 1

    threshold_value = np.partition(combine_simi_copy[index1], int(-count_edge[index1]))[int(-count_edge[index1])]

    for index2 in range(0, label_index):

        #if combine_simi_copy[index1][index2] >= threshold_value and index2 > index1:

        if combine_simi_copy[index1][index2] >= threshold_value:

            g.add_edge(index1, index2, weight = combine_simi_copy[index1][index2])

            g_nx.add_edge(index1, index2, weight = combine_simi_copy[index1][index2])

            combine_simi_final[index1][index2] = combine_simi_copy[index1][index2]

            

for index1 in range(0, label_index):

    for index2 in range(0, label_index):

        if index1 < index2 and combine_simi_final[index1][index2] > 0:

            combine_simi_edge[index1][index2] = combine_simi_final[index1][index2]

        if index1 > index2 and combine_simi_final[index1][index2] > 0 and combine_simi_final[index2][index1] == 0:

            #print(str(index1) + "-" + str(index2))

            combine_simi_edge[index2][index1] = combine_simi_final[index1][index2]
###

! pip install fa2

from fa2 import ForceAtlas2

import gensim

################################



weight = g.es['weight']

layout_fr = g.layout_fruchterman_reingold(weights=weight)

layout_kk = g.layout("kk")

layout_lgl = g.layout("lgl")

forceatlas2 = ForceAtlas2(

    # Behavior alternatives

    outboundAttractionDistribution=False,  # Dissuade hubs

    linLogMode=False,  # NOT IMPLEMENTED

    adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)

    edgeWeightInfluence=1.0,



    # Performance

    jitterTolerance=1.0,  # Tolerance

    barnesHutOptimize=True,

    barnesHutTheta=1.2,

    multiThreaded=False,  # NOT IMPLEMENTED



    # Tuning

    scalingRatio=2.0,

    strongGravityMode=False,

    gravity=1.0,



    # Log

    verbose=False

)

positions = forceatlas2.forceatlas2_networkx_layout(g_nx, pos=None, iterations=2000)

layout_fa = np.zeros((label_index, 2))

for i in range(label_index):

    layout_fa[i][0] = positions[i][0]

    layout_fa[i][1] = positions[i][1]
memberships = []

number_level = 0

weight = g.es['weight']

level_size = np.zeros(number_level)



community = g.community_multilevel(weights=weight) ### Louvain method (Blondel 2008)

modularity = community.modularity

membership = community.membership



#Multilevel Community Detection with Louvain method (network clustering) for sparsified article networks

communities_raw = g.community_multilevel(weights=weight, return_levels=True) # Louvain method (Blondel 2008)

communities = []



if(len(communities_raw) == 3):

    communities = communities_raw

elif(len(communities_raw) > 3):

    for i in range(len(communities_raw) - 2, len(communities_raw)):

        communities.append(communities[i])

elif(len(communities_raw) == 2):

    communities.append(communities_raw[0])

    communities.append(communities_raw[0])

    communities.append(communities_raw[1])

elif(len(communities_raw) == 1):

    communities.append(communities_raw[0])

    communities.append(communities_raw[0])

    communities.append(communities_raw[0])

else:

    print("error in community detection")



modularities = []



optimal_level_index = 0

optimal_modularity = 0

for level in communities: # iterate through different levels of clustering

    number_level += 1

    if level.modularity > optimal_modularity:

        optimal_modularity = level.modularity

        optimal_level_index = number_level - 1

    modularities.append(level.modularity)

    memberships.append(level.membership)

level_size = np.zeros(number_level)

for i in range(0, number_level):

    level_size[i] = len(set(memberships[i]))



community_map= np.zeros((int(max(level_size)), number_level)) # global map: level0, level1, level2

community_map = community_map - 1

for index in range(0, label_index):

    community = int(memberships[0][index])

    if community not in community_map[:,0]:

        community_map[community][0] = community

        for level in range(1, number_level):

            community_map[community][level] = memberships[level][index]

community_maps = [] # the map between every level to the top level, the format is: current_level_cluster_id, top_level_cluster_id, sub_id_within_the_top_cluster

for level in range(0, number_level):

    community_map_current = np.zeros((int(level_size[level]), 3))

    community_map_current = community_map_current - 1

    cluster_count = np.zeros(int(level_size[number_level - 1]))

    for index in range(0, label_index):

        community = int(memberships[level][index])

        if community not in community_map_current[:,0]:

            community_map_current[community][0] = community

            top_community = memberships[number_level-1][index]

            community_map_current[community][1] = top_community

            community_map_current[community][2] = cluster_count[top_community]

            cluster_count[top_community] += 1

    community_maps.append(community_map_current)

    

    

# Useful results: memberships/level_size
# for memberships3, community_map3 and community_maps3

memberships3 = []

for level in range(0, number_level):

    clustering = AgglomerativeClustering(linkage = 'ward', n_clusters = int(level_size[level]))

    #clustering.fit(text_2d)

    clustering.fit(texts_doc2vec)

    #plot_clustering(combine_2d, combine_vec.toarray(), clustering.labels_, "%s linkage" % 'ward')

    memberships3.append(clustering.labels_)

community_map3 = np.zeros((int(max(level_size)), number_level)) # global map: level0, level1, level2

community_map3 = community_map3 - 1

for index in range(0, label_index):

    community = int(memberships3[0][index])

    if community not in community_map3[:,0]:

        #print ("index-" + str(index) + ", community0-" + str(community) + ", community1-" + str(memberships2[1][index]) + ", community2-" + str(memberships2[2][index]));

        community_map3[community][0] = community

        for level in range(1, number_level):

            community_map3[community][level] = memberships3[level][index]

community_maps3 = [] # the map between every level to the top level, the format is: current_level_cluster_id, top_level_cluster_id, sub_id_within_the_top_cluster

for level in range(0, number_level):

    community_map_current = np.zeros((int(level_size[level]), 3))

    community_map_current = community_map_current - 1

    cluster_count = np.zeros(int(level_size[number_level - 1]))

    for index in range(0, label_index):

        community = int(memberships3[level][index])

        if community not in community_map_current[:,0]:

            community_map_current[community][0] = community

            top_community = memberships3[number_level-1][index]

            community_map_current[community][1] = top_community

            community_map_current[community][2] = cluster_count[top_community]

            cluster_count[top_community] += 1

    community_maps3.append(community_map_current)
current_index = label_index

layout_tsne_text = text_2d

layout_tsne_text_adjusts = []

layout_tsne_text_adjusts2 = []

 # get the graph center

graph_center = [0,0]

for index in range(0, current_index):

    graph_center += layout_tsne_text[index]

graph_center = graph_center/current_index

# get the cluster centers (consider different clustering levels)

for level in range(0, number_level):

    cluster_number = int(level_size[level])

    cluster_center = np.zeros((cluster_number, 2))

    cluster_diff = np.zeros((cluster_number, 2))

    cluster_member_count = np.zeros(cluster_number)

    for index in range(0, current_index):

        cluster_id = memberships3[level][index]

        cluster_center[cluster_id] += layout_tsne_text[index]

        cluster_member_count[cluster_id] += 1

    for index in range(0, cluster_number):

        cluster_center[index] = cluster_center[index]/cluster_member_count[index]

        cluster_diff[index] = cluster_center[index] - graph_center



    layout_adjust = np.zeros((current_index, 2))

    for index in range(0, current_index):

        cluster_id = memberships3[level][index]

        layout_adjust[index] = layout_tsne_text[index] + 0.5*cluster_diff[cluster_id]

    layout_tsne_text_adjusts.append(layout_adjust)



    layout_adjust2 = np.zeros((current_index, 2))

    for index in range(0, current_index):

        cluster_id = memberships3[level][index]

        layout_adjust2[index] = layout_tsne_text[index] + 1*cluster_diff[cluster_id]

    layout_tsne_text_adjusts2.append(layout_adjust2)
### Helper function and Parameters for rake-based topic modeling

import operator

my_stopwords = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "about", "above", "addition", "after", "again", "against", "ain", "all",

"also", "although", "am", "among", "an", "and", "any", "approach", "approached", "approaches", "approaching", "are", "aren", "as", "at", "b", "based", "be",

"because", "been", "before", "being", "below", "between", "both", "but", "by", "c", "called", "can",

"consider", "considers", "consideres", "considering", "corresponding", "could", "couldn", "d",

"develop", "developed", "developing", "develops", "did", "didn", "do", "does", "doesn", "doing", "don", "down", "during", "e",

"each", "f", "few", "first", "for", "from", "further", "g", "go", "goes", "h", "had", "hadn", "has", "hasn", "have", "haven",

"having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "however", "i", "if", "in", "include", "included",

"includes", "including", "into", "is", "isn", "it", "its", "itself", "j", "just", "k", "l", "ll", "m", "m", "ma", "many", "may", "me",

"mg", "might", "mightn", "more", "most", "much", "must", "mustn", "my", "myself", "n", "needn", "never", "new", "no", "none", "nor",

"not", "now", "o", "of", "off", "on", "once", "one", "ones", "only", "or", "other", "others", "otherwise", "our", "ours", "ourselves",

"out", "over", "over", "own", "p", "particular", "present", "presented", "presenting", "presents", "propose", "proposed", "proposes",

"proposing", "provide", "provided", "provides", "providing", "q", "r", "re", "result", "resulted", "resulting", "results", "s", "same",

"shall", "shalln", "shan", "she", "should", "shouldn", "show", "showed", "showing", "shows", "since", "so", "some", "studied", "studies",

"study", "studying", "sub", "such", "sup", "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",

"they", "this", "those", "though", "through", "throughout", "to", "too", "two", "u", "under", "until", "up", "use", "used", "uses", "using",

"v", "ve", "very", "via", "w", "was", "wasn", "we", "well", "were", "weren", "what", "when", "where", "whether", "which", "while", "who",

"whom", "why", "will", "with", "without", "won", "would", "wouldn", "x", "y", "you", "your", "yours", "yourself", "yourselves", "z"]

#########################################



def removeNonAscii(s):

    return "".join(i for i in s if ord(i) < 128)





# define the function to extract Noun Phrases (NP) from text, e.g., title and abstract

# use more grammars, no stemming is applied, noun phrases are in the format of adj + noun

def NPextractor2(text):

    text = removeNonAscii(text)

    if len(text) == 0:

        return text



    tok = nltk.word_tokenize(text)

    pos = nltk.pos_tag(tok)



    # the original grammar, to get shorter NPs

    grammar1 = r"""

      NP: {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun

          {<NNP>+}                # chunk sequences of proper nouns

    """





    # the self defined grammar based on the previous version above, to get longer NPs (as supplements)

    grammar2 = r"""

      NP: {<DT|PP\$>?<JJ>*<NN|NNS|NNP|NNPS>+}   # chunk determiner/possessive, adjectives and noun(s)

          {<NNP>+}                # chunk sequences of proper nouns

    """



    chunker1 = nltk.RegexpParser(grammar1)

    tree1 = chunker1.parse(pos)

    chunker2 = nltk.RegexpParser(grammar2)

    tree2 = chunker2.parse(pos)



    nps = [] # word and pos_tag

    nps_words = [] # only word



    for subtree in tree1.subtrees(filter=lambda t: t.label() == 'NP'):

        nps.append(subtree.leaves())

        current_np = []

        for item in subtree.leaves():

            current_np.append(item[0])

        nps_words.append(current_np)



    for subtree in tree2.subtrees(filter=lambda t: t.label() == 'NP'):

        if subtree.leaves() in nps:

            continue

        nps.append(subtree.leaves())

        current_np = []

        for item in subtree.leaves():

            current_np.append(item[0])

        nps_words.append(current_np)



    refined_words = []

    #stopwords = nltk.corpus.stopwords.words('english')

    stopwords = my_stopwords

    for np in nps_words:

        if len(np) < 1:

            continue

        current_np = []

        for word in np:

            if (2 <= len(word) <= 40) and (word.lower() not in stopwords):

                current_np.append(word.lower())

        if len(current_np) >= 1:

            refined_words.append(current_np)

    return refined_words



#############################################

# Tell whether a word is punctuation or not.

def isPunct(word):

    return len(word) == 1 and word in string.punctuation

#############################################

# Tell whether a word is numeric or not.

def isNumeric(word):

    try:

        float(word) if '.' in word else int(word)

        return True

    except ValueError:

        return False

############################################



# Tell whether the a contains numerical part.

def containNumeric(word):

    return any(char.isdigit() for char in word)



# Define the RAKE method for keyword extraction

# Reference: Reference: Automatic keyword extraction from individual documents

class RakeKeywordExtractor:

    def __init__(self):

        #self.stopwords = set(nltk.corpus.stopwords.words())

        self.stopwords = set(my_stopwords)

        self.top_fraction = 1 # consider top third candidate keywords by score

    #########################################

    '''

    Chunk each sentence into phrases using punctuations and stopwords.

    Upper_length restricts phrase length. If model == all, we also include additional noun phrases based on NLP grammars;

    if mode == np, we only consider noun phrases, and RAKE-chunked phrases will be ignored.

    '''

    def _generate_candidate_keywords(self, sentences, upper_length, mode):

        phrase_list = []

        for sentence in sentences:

            # Additional Noun phrases if they won't be detected by the Rake method below

            nps = NPextractor2(sentence)

            if mode == "np":

                if len(nps) > 0:

                    for item in nps:

                        if len(item) > 0 and len(item) <= upper_length:  # restrict the length of phrase to be 1~5

                            phrase_list.append(item)

                continue



            words = map(lambda x: "|" if x in my_stopwords else x, nltk.word_tokenize(sentence.lower()))

            phrase = []

            for word in words:

                #if word == "|" or isPunct(word):

                if word == "|" or isPunct(word) or isNumeric(word) or containNumeric(word):

                    #if len(phrase) > 0:

                    if len(phrase) > 0 and len(phrase) <= upper_length: # restrict the length of phrase to be 1~5

                        if phrase not in nps:

                            phrase_list.append(phrase)

                        phrase = []

                else:

                    phrase.append(word)

            if len(nps) > 0:

                #phrase_list += nps

                for item in nps:

                    if len(item) > 0 and len(item) <= upper_length:  # restrict the length of phrase to be 1~5

                        phrase_list.append(item)

        return phrase_list

    ###########################################

    '''

    For each phrase consisting of multiple words, calculate the score of each word, reflecting the word’s content meaningfulness.

    '''

    def _calculate_word_scores(self, phrase_list):

        word_freq = nltk.FreqDist()

        word_degree = nltk.FreqDist()

        for phrase in phrase_list:

            filterlist = list(filter(lambda x: not isNumeric(x) and not containNumeric(x), phrase))

            degree = len(filterlist) - 1

            for word in phrase:

                #word_freq.inc(word)

                word_freq[word] += 1

                #word_degree.inc(word, degree) # other words

                word_degree[word] += degree

        for word in word_freq.keys():

            word_degree[word] = word_degree[word] + word_freq[word] # itself

    # word score = deg(w) / freq(w) (favor long phrases), or word score = deg(w) (not that favor long phrases)

        word_scores = {}

        for word in word_freq.keys():

            #word_scores[word] = word_degree[word] / word_freq[word] # (favor long phrases)

            word_scores[word] = word_degree[word]

        return word_scores

    #################################################################

    '''

    For each phrase consisting of multiple words, combine word scores into the phrase score, which represents the phrase’s content meaningfulness.

    '''

    def _calculate_phrase_scores(self, phrase_list, word_scores):

        phrase_scores = {}

        for phrase in phrase_list:

            phrase_score = 0

            for word in phrase:

                phrase_score += word_scores[word]

            phrase_scores[" ".join(phrase)] = phrase_score

            #phrase_scores[" ".join(phrase)] = phrase_score/len(phrase)

        return phrase_scores

    #################################################################

    '''

    Extract keywords (key phrases) from the input text, which can consist of multiple sentences.

    '''

    def extract(self, text, incl_scores=False, number=30, upper_length=5, mode="all"):

        sentences = nltk.sent_tokenize(text) # break a text (paragraph) into an array of single sentences ending with a period

        phrase_list = self._generate_candidate_keywords(sentences, upper_length, mode)

        word_scores = self._calculate_word_scores(phrase_list)

        phrase_scores = self._calculate_phrase_scores(phrase_list, word_scores)

        sorted_phrase_scores = sorted(phrase_scores.items(), key=operator.itemgetter(1), reverse=True)

        n_phrases = len(sorted_phrase_scores)

        if incl_scores:

            #return sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]

            return sorted_phrase_scores[0:number]

        else:

            #return map(lambda x: x[0], sorted_phrase_scores[0:int(n_phrases/self.top_fraction)])

            return map(lambda x: x[0], sorted_phrase_scores[0:number])
cluster_title_keywords3 = []

cluster_abstract_keywords3 = []

cluster_titles3 = [["" for j in range(len(community_maps3[i]))] for i in range(number_level)]

cluster_abstracts3 = [["" for j in range(len(community_maps3[i]))] for i in range(number_level)]

cluster_title_keywords3 = [[[] for j in range(len(community_maps3[i]))] for i in range(number_level)]

cluster_abstract_keywords3 = [[[] for j in range(len(community_maps3[i]))] for i in range(number_level)]

for level in range(0, number_level):

    for index in range(0, current_index):

        cluster_id = int(memberships3[level][index])

        cluster_titles3[level][cluster_id] += biorxiv_medrxiv['title_raw'][index]

        cluster_titles3[level][cluster_id] += " "

        cluster_abstracts3[level][cluster_id] += biorxiv_medrxiv['abstract_raw'][index]

        cluster_abstracts3[level][cluster_id] += " "

rake = RakeKeywordExtractor()



for level in range(0, number_level):

    for cluster in range(len(community_maps3[level])):

        cluster_title_keywords3[level][cluster] = rake.extract(cluster_titles3[level][cluster], incl_scores=False)

        cluster_abstract_keywords3[level][cluster] = rake.extract(cluster_abstracts3[level][cluster], incl_scores=False)
cluster_title_keywords = []

cluster_abstract_keywords = []

for level in range(number_level):

    tmp_title = []

    tmp_abstract = []

    for cluster_ in memberships3[level]:

        tmp_title.append(', '.join(list(cluster_title_keywords3[level][cluster_])[0:5]))

        tmp_abstract.append(', '.join(list(cluster_abstract_keywords3[level][cluster_])[0:5]))

    cluster_title_keywords.append(tmp_title)

    cluster_abstract_keywords.append(tmp_abstract)
output_notebook()

selected_num = 2

y_labels = memberships3[selected_num]

### Data source

source = ColumnDataSource(data = {

    'x' : layout_tsne_text_adjusts[selected_num][:,0],

    'y' : layout_tsne_text_adjusts[selected_num][:,1],

    'title' : df_covid['title_raw'],

    'desc' : y_labels,

    'author' : df_covid['authors'],

    'journal' : df_covid['journal'],

    'labels' : ['Cluster '+str(x) for x in y_labels],

#     'title_keywords': cluster_title_keywords[selected_num],

#     'abstract_keywords': cluster_abstract_keywords[selected_num]

})



### Hover information

hover = HoverTool(tooltips=[

    ("Title", "@title"),

    ("Author(s)", "@author"),

    ("Journal", "@journal"),

#     ("Keywords of Cluster: title", '@title_keywords'),

#     ("Keywords of Cluster: abstract", '@abstract_keywords')

],point_policy="follow_mouse")



### Map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))





p = figure(plot_width=800, plot_height=800, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'lasso_select'], 

           title="COVID-19 Semantic Cluster - Fine", 

           toolbar_location="right")



# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black",

          legend = 'labels')





# source.selected.js_on_change('indices', CustomJS(args={'x':layout_tsne_text_adjusts[selected_num][:,0], 'y': layout_tsne_text_adjusts[selected_num][:,1]}, code="""

#         var inds = cb_obj.indices;

        

#     """)

# )



callback = CustomJS(args={'source':source, 'db':layout_tsne_text_adjusts, 'memberships3':memberships3, 'cluster_title_keywords':cluster_title_keywords, 'cluster_abstract_keywords':cluster_abstract_keywords}, code='''

    var data = source.data;

    var x = data['x']

    var y = data['y']

    var f = cb_obj.value;

    var desc = data['desc']

    var labels = data['labels']

    var selected_num = 0;

    switch(f){

        case 0:

            selected_num=0

            break;

        

        case 1:

            selected_num=1

            break;

            

        case 2:

            selected_num=2

            break;

        

    };

    var db = db;

    for (var i = 0; i < x.length; i++) {

        x[i] = db[selected_num][i][0];

        y[i] = db[selected_num][i][1];

        labels[i] = 'Cluster '+memberships3[selected_num][i];

        desc[i] = memberships3[selected_num][i];

        

    }

    //source.data['title_keywords'] = cluster_title_keywords[selected_num];

    //source.data['abstract_keywords'] = cluster_abstract_keywords[selected_num];

    source.change.emit();

''')



#header

header = Div(text="""<h1>COVID-19 Semantic Cluster</h1>""")



slider = Slider(start=0, end=2, value=1, step=1, title="Num of clusters: large to small")

slider.js_on_change('value', callback)



layout = column(header, row(column(slider)), p)

#show

show(layout)
# def get_desc_col_name(cluster_level):

#     return 'all_memberships_size'+str(cluster_level)



# def get_x_col_name(cluster_level, sample=''):

#     return f'all_layout_tsne_text_adjusts_{str(cluster_level)}_x'



# def get_y_col_name(cluster_level, sample=''):

#     return f'all_layout_tsne_text_adjusts_{str(cluster_level)}_y'



# def get_keyword_col_name(cluster_level):

#     return f'all_cluster_keywords_for_mem{str(cluster_level)}'



# wordcloud_library = {key: [ f'./{key}_all_wordcloud.png']* boosted_all[boosted_all[key+'_can_handle_flag']==1].shape[0] for key in task_list}



# from bokeh.palettes import Spectral6

# %config InlineBackend.figure_format = 'retina'

# output_notebook()



# cluster_level = 10

# ### Data source

# source_tsne = ColumnDataSource(data = {

#     'x' : boosted_all[get_x_col_name(cluster_level)],

#     'y' : boosted_all[get_y_col_name(cluster_level)],

#     'title' : boosted_all['title_raw'],

#     'desc' : boosted_all[get_desc_col_name(cluster_level)],

#     'author' : boosted_all['authors'],

#     'journal' : boosted_all['journal'],

#     'keyword': boosted_all[get_keyword_col_name(cluster_level)],

#     'labels' : boosted_all[get_desc_col_name(cluster_level)]

# })



# ### Hover information

# hover = HoverTool(tooltips=[

#     ("Title", "@title"),

#     ("Author(s)", "@author"),

#     ("Journal", "@journal"),

#     ("Keyword of cluster", "@keyword")

# ],point_policy="follow_mouse")



# ### Map colors

# mapper = linear_cmap(field_name='desc', 

#                      palette=Category20[20],

#                      low=min(boosted_all[get_desc_col_name(cluster_level)]) ,high=max(boosted_all[get_desc_col_name(cluster_level)]))





# p = figure(plot_width=600, plot_height=600, 

#            tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'lasso_select'], 

#            title="COVID-19 Semantic Cluster - Fine", 

#            toolbar_location="right")



# # plot

# render = p.circle('x', 'y', size=5,

#           source=source_tsne,

#           fill_color=mapper,

#           line_alpha=0.3,

#           line_color="black",

#           legend = 'labels',color=Spectral6)





# subclass_select = Select(title="Subclass:", value="All", options=['All'] + [str(i) for i in range(cluster_level)], width=100)



# callback = CustomJS(args={'source':source_tsne, 'db':boosted_all_dict, 'subclass_select': subclass_select}, code='''

#     var sliderIndex = cb_obj.value;

#     var x_col_name = "all_layout_tsne_text_adjusts_" + sliderIndex.toString() + '_x';

#     var y_col_name = "all_layout_tsne_text_adjusts_" + sliderIndex.toString() + '_y';

#     var desc_col_name = "all_memberships_size" + sliderIndex.toString();

#     source.data['x'] = Object.values(db[x_col_name]);

#     source.data['y'] = Object.values(db[y_col_name]);

#     source.data['desc'] = Object.values(db[desc_col_name]);

#     source.data['labels'] = Object.values(db[desc_col_name]);

#     var tmp = ['All'];

#     let i = 0;

#     while(i < sliderIndex){

#         tmp.push(i.toString());

#         i = i + 1;

#     }

#     subclass_select.options = tmp; 

#     source.change.emit();

# ''')







# subclass_callback = CustomJS(args = dict(source=source_tsne, boosted_all_dict=boosted_all_dict), code="""

#     Object.filter = function( obj, predicate) {

#         var result = {};

#         for (let key in obj) {

#             if (obj.hasOwnProperty(key) && predicate(obj[key])) {

#                 result[key] = obj[key];

#             }

#         }

#         return result;

#     };

    

#     Object.batch_select = function(obj, list){

#         var res = [];

#         for(let key of list){

#             res.push(obj[key]);

#         }

#         return res;

#     };

    

#     var selected_index = cb_obj.value;

#     var cluster_level = parseInt(cb_obj.options[cb_obj.options.length-1]) + 1;

#     var target = [];

#     if (selected_index === "All"){

#         source.selected.indices = [];

#     } else {    

#         selected_index = parseInt(selected_index);

#         target = Object.keys(Object.filter(boosted_all_dict['all_memberships_size'+ cluster_level.toString()], item=>item == selected_index));

#         source.selected.indices = target;

#     }

    

# """)





# #header

# header = Div(text=f"<h3>COVID-19 Candidate Corpus Semantic Cluster - {boosted_all.shape[0]} total points</h3>")



# slider = Slider(start=5, end=20, value=10, step=1, title="Num of clusters",width=500 )

# slider.js_on_change('value', callback)

# subclass_select.js_on_change('value', subclass_callback)

# layout = column(header, row(slider, subclass_select), p)

# show(layout)

# # layout = column(header, row(subclass_select), p)

# # ######

# # wordcloud_source = ColumnDataSource(data={

# #     'url': wordcloud_library['task_1']

# # })



# # data = {

# #     'answer': list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['task_1_answer']),

# #     'score':list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['task_1_start_score_prob']),

# #     'index_from_original': list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1].index)

# # }



# # detail_data = ColumnDataSource({

# #     'title':list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['title_raw']),

# #     'answer': list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['task_1_answer']),

# #     'abstract':list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['abstract_raw']),

# #     'id': list(boosted_all[boosted_all['task_1_can_handle_flag'] == 1]['paper_id'])

# # })



# # # worldcloud_source = ColumnDataSource({

# # #     'url': res['task_1'][-1]

# # # })





# # source = ColumnDataSource(data)



# # columns = [

# #     TableColumn(field='answer', title='Possible Answer', formatter=StringFormatter(font_style="bold")),

# #     TableColumn(field='score', title='Score', width=5)

# # ]

# # data_table =  DataTable(source=source, columns=columns, selectable=True, index_header="", width=500,height=600, fit_columns=True, scroll_to_selection=True, height_policy='auto', editable=True)



# # select = Select(title='', value=list(task_list.values())[0], options=list(task_list.values()), height=50, width=420)



# # scoretype_select = Select(title='Score type', value='standard', options=['standard', 'normalized', 'norm+adjust'], width=80)





# # scoretype_callback = CustomJS(args = dict(source=source, select = select, task_to_index=task_to_index, boosted_all_dict=boosted_all_dict), code="""

    

# #     Object.filter = function( obj, predicate) {

# #         var result = {};

# #         for (let key in obj) {

# #             if (obj.hasOwnProperty(key) && predicate(obj[key])) {

# #                 result[key] = obj[key];

# #             }

# #         }

# #         return result;

# #     };

    

# #     Object.batch_select = function(obj, list){

# #         var res = [];

# #         for(let key of list){

# #             res.push(obj[key]);

# #         }

# #         return res;

# #     };

    

# #     var selected_type = cb_obj.value;

# #     var selected_task = select.value;

# #     var selected_index = task_to_index[selected_task];

# #     var target_indexs = Object.keys(Object.filter(boosted_all_dict[selected_index+'_can_handle_flag'], item => item == 1));

# #     var answer = Object.batch_select(boosted_all_dict[selected_index+'_answer'], target_indexs);

# #     var score = [];

# #     if (selected_type == 'standard'){

# #         score = Object.batch_select(boosted_all_dict[selected_index+'_start_score_prob'], target_indexs);

# #     }else if(selected_type == 'normalized'){

# #         score = Object.batch_select(boosted_all_dict[selected_index+'_score_normalized'], target_indexs);

# #     }else if(selected_type == 'norm+adjust'){

# #         score = Object.batch_select(boosted_all_dict[selected_index+'_score_normalized_adjusted'], target_indexs);

# #     }

# #     var title = Object.batch_select(boosted_all_dict['title_raw'], target_indexs);

# #     var abstract = Object.batch_select(boosted_all_dict['abstract_raw'], target_indexs);

# #     var id = Object.batch_select(boosted_all_dict['paper_id'], target_indexs);

# #     source.data = {

# #         'answer': answer,

# #         'score': score,

# #         'index_from_original': target_indexs

# #     };

# #     detail_data.data = {

# #        'title':title,

# #        'answer':answer,

# #        'abstract':abstract,

# #        'id':id

# #     };

# #     source.change.emit();

# #     detail_data.change.emit();

# # """)





# # subclass_callback = CustomJS(args = dict(source=source_tsne, boosted_all_dict=boosted_all_dict), code="""

# #     Object.filter = function( obj, predicate) {

# #         var result = {};

# #         for (let key in obj) {

# #             if (obj.hasOwnProperty(key) && predicate(obj[key])) {

# #                 result[key] = obj[key];

# #             }

# #         }

# #         return result;

# #     };

    

# #     Object.batch_select = function(obj, list){

# #         var res = [];

# #         for(let key of list){

# #             res.push(obj[key]);

# #         }

# #         return res;

# #     };

    

# #     var selected_index = cb_obj.value;

# #     var cluster_level = parseInt(cb_obj.options[cb_obj.options.length-1]) + 1;

# #     var target = [];

# #     if (selected_index === "All"){

# #         source.selected.indices = [];

# #     } else {    

# #         selected_index = parseInt(selected_index);

# #         target = Object.keys(Object.filter(boosted_all_dict['all_memberships_size'+ cluster_level.toString()], item=>item == selected_index));

# #         source.selected.indices = target;

# #     }

    

# # """)









# # task_select_callback = CustomJS(args=dict(source=source,detail_data=detail_data, source_tsne=source_tsne, task_to_index=task_to_index, scoretype_select=scoretype_select, boosted_all_dict=boosted_all_dict, wordcloud_source=wordcloud_source, wordcloud_library=wordcloud_library), code="""

# #     Object.filter = function( obj, predicate) {

# #         var result = {};

# #         for (let key in obj) {

# #             if (obj.hasOwnProperty(key) && predicate(obj[key])) {

# #                 result[key] = obj[key];

# #             }

# #         }

# #         return result;

# #     };

    

# #     Object.batch_select = function(obj, list){

# #         var res = [];

# #         for(let key of list){

# #             res.push(obj[key]);

# #         }

# #         return res;

# #     };

# #     var selected_type = scoretype_select.value;

# #     var selected_task = cb_obj.value;

# #     var selected_index = task_to_index[selected_task];

# #     var target_indexs = Object.keys(Object.filter(boosted_all_dict[selected_index+'_can_handle_flag'], item => item == 1));

# #     var answer = Object.batch_select(boosted_all_dict[selected_index+'_answer'], target_indexs);

# #     var score = [];

# #     if (selected_type == 'standard'){

# #         score = Object.batch_select(boosted_all_dict[selected_index+'_start_score_prob'], target_indexs);

# #     }else if(selected_type == 'normalized'){

# #         score = Object.batch_select(boosted_all_dict[selected_index+'_score_normalized'], target_indexs);

# #     }else if(selected_type == 'norm+adjust'){

# #         score = Object.batch_select(boosted_all_dict[selected_index+'_score_normalized_adjusted'], target_indexs);

# #     }

# #     var title = Object.batch_select(boosted_all_dict['title_raw'], target_indexs);

# #     var abstract = Object.batch_select(boosted_all_dict['abstract_raw'], target_indexs);

# #     var id = Object.batch_select(boosted_all_dict['paper_id'], target_indexs);

# #     source.data = {

# #         'answer': answer,

# #         'score': score,

# #         'index_from_original': target_indexs

# #     };

# #     detail_data.data = {

# #        'title':title,

# #        'answer':answer,

# #        'abstract':abstract,

# #        'id':id

# #     };

# #     wordcloud_source.data['url'] = wordcloud_library[selected_index];

# #     source_tsne.selected.indices = target_indexs;

# #     wordcloud_source.change.emit();

# #     """) 

    

# # paper_detail = Div(text="Paper info shows here", margin=(50,10,10,30), style={'border':'1px solid black', 'width':'400px', 'height':'600px', 'padding':'20px', 'border-top-left-radius':'5px','border-top-right-radius':'5px','text-align':'center','overflow-y':'auto'})

# # select_callback=CustomJS(args=dict(source=source, div=paper_detail, source_tsne=source_tsne, detail_data=detail_data), code="""

# #     var selection_index=source.selected.indices[0];

# #     var answer = detail_data.data['answer'][selection_index];

# #     var title = detail_data.data['title'][selection_index];

# #     var abstract = detail_data.data['abstract'][selection_index] || "";

# #     var idlink = "";

# #     var id = detail_data.data['id'][selection_index];

# #     if(id.startsWith("PMC")){

# #         idlink = "https://www.ncbi.nlm.nih.gov/pmc/articles/" + id;

# #     }else{

# #         idlink = 'https://www.semanticscholar.org/paper/' + id;

# #     }

# #     var index = 0;

# #     if(title.toLowerCase().indexOf(answer.toLowerCase()) !== -1){

# #         index = title.toLowerCase().indexOf(answer.toLowerCase())

# #         title = title.slice(0,index) + `<a href=${idlink} target="_blank" style='text-decoration:none; background: linear-gradient(90deg, rgb(147, 222, 241), rgb(147, 222, 23)); background-image: linear-gradient(90deg, rgb(147, 222, 241), rgb(147, 222, 23)); background-position-x: initial; background-position-y: initial;background-size: initial; background-repeat-x: initial; background-repeat-y: initial; background-attachment: initial;background-origin: initial; background-clip: initial; background-color: initial; margin: 0 0.25em; line-height: 1.5; padding: 0px 3px !important; border-radius: 5rem !important;'><strong>${title.slice(index, index + answer.length)}</strong></a>` + title.slice(index + answer.length);

# #     }

# #     if(abstract.toLowerCase().indexOf(answer.toLowerCase()) !== -1){

# #         index = abstract.toLowerCase().indexOf(answer.toLowerCase());

# #         abstract = abstract.slice(0, index) + `<a style='text-decoration:none; background: linear-gradient(90deg, rgb(147, 222, 241), rgb(147, 222, 23)); background-image: linear-gradient(90deg, rgb(147, 222, 241), rgb(147, 222, 23)); background-position-x: initial; background-position-y: initial;background-size: initial; background-repeat-x: initial; background-repeat-y: initial; background-attachment: initial;background-origin: initial; background-clip: initial; background-color: initial; margin: 0 0.25em; line-height: 1.5; padding: 0px 3px !important; border-radius: 5rem !important;'><strong>${abstract.slice(index, index + answer.length)}</strong></a>` + abstract.slice(index + answer.length);

# #     }

# #     div.text = `<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">`;

# #     div.text +=`<h2><a href=${idlink} target="_blank" style='text-decoration:none'><center>${title}&nbsp<i class='fa fa-external-link' style='font-size:15px'></i></center></a></h2>`;

# #     div.text += `<p><strong><center>Answer: <a style='color:blue; font-style:italic'>${detail_data.data['answer'][selection_index]}</a></center></strong></p>`;

# #     div.text += `<br>`;

# #     div.text += `<p><strong><center>Abstract</center></strong></p>`;

# #     div.text += `<p>${abstract}</p>`;

# #     source_tsne.selected.indices = [source.data['index_from_original'][selection_index]];

# # """)





# # header_wc = Div(text=f"<h3>Noun/Subject phrase wordcloud from candidate answers</h3>")    

# # wordcloudpng = figure(x_range=(0,1), y_range=(0,1), width=300, height=300)

# # wordcloudpng.image_url(url='url', x=0, y=1, w=1, h=0.8, source=wordcloud_source)

# # wc_layout = column(header_wc, Div(text=""), wordcloudpng)

# # source.selected.js_on_change('indices', select_callback)

# # select.js_on_change('value', task_select_callback)

# # subclass_select.js_on_change('value', subclass_callback)

# # scoretype_select.js_on_change('value', scoretype_callback)

# # output_file('all_valid.html')



# # # show(column(row(layout, wc_layout),row(column(row(select,scoretype_select), data_table),column(paper_detail))))

# # # grid = gridplot([layout, wc_layout,column(row(select,scoretype_select), data_table),paper_detail], ncols=2, plot_width=250, plot_height=250)

# # # show(row(column(layout), column(row(select,scoretype_select), data_table, row(paper_detail), wc_layout)))

# # title = Div(text = "<h1><center>Covid-19 Tasks' Candidate Answers<center></h1>")

# # show(column(title, row(layout, wc_layout),row(column(row(select,scoretype_select), data_table),paper_detail)))

# # # show(row(layout, column(select, data_table, paper_detail)))

# # # show(column(row(column(select, data_table), paper_detail)))