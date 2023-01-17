!pip uninstall tensorflow -y
# !pip uninstall tensorflow-gpu -y
# !conda uninstall tensorflow
# !conda uninstall tensorflow-gpu
!nvcc -V
!nvidia-smi
!python -V
!gcc --version
!pip install tensorflow-gpu==1.15.2
!pip install caireCovid==0.1.7
import tensorflow as tf
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
import requests
import math
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, pos_tag
import pandas as pd
from IPython.core.display import display, HTML

import caireCovid

from caireCovid import QaModule, get_rank_score
from caireCovid.qa_utils import stop_words
# define our function

def retrieve_paragraph(query):
    url = "http://hlt027.ece.ust.hk:5000/query_paragraph"

    payload = "{\n\t\"text\": \""+query+"\"\n}"
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache",
        'Postman-Token': "696fa512-5fed-45ca-bbe7-b7a1b4d19fe4"
    }
    response = requests.request("POST", url, data=payload, headers=headers)

    response = response.json()
    return response

def information_retrieval(file_name):

    with open(file_name) as f:
        json_file = json.load(f)
    subtasks = json_file["sub_task"]
    
    all_results = []
    data_for_qa = []
    for item in subtasks:
        questions = item["questions"]
        for query in questions:
            result_item = {"question" : query}
            retri_result = retrieve_paragraph(query)
            result_item["data"] = retri_result

            qa_item = {"question": query}
            context = []
            titles = []
            doi = []
            count = 1
            for item in retri_result:
                #context.append(item["paragraph"] if "paragraph" in item and len(item["paragraph"]) > 0 else item["abstract"])
                if count>10:
                    break
                if 'abstract' in item and len(item['abstract']) > 0:
                    context.append(item['abstract'])
                    doi.append(item["doi"])
                    titles.append(item["title"])
                    count+=1
                if 'paragraphs' in item:
                    # for para in item['paragraphs']:
                    #     context.append(para['text'])
                    #     count+=1
                    #     if count>20:
                    #         break
                    context.append(item['paragraphs'][0]['text'])   
                    doi.append(item["doi"])
                    titles.append(item["title"])
                    count+=1

            qa_item["data"] = {"answer": "", "context": context, "doi": doi, "titles": titles}

            all_results.append(result_item)
            data_for_qa.append(qa_item)

    return data_for_qa

def parse_ir_results(query, retri_result, topk = 10):
    qa_item = {'question': query}
    temp_doi = {}
    contexts = []
    titles = []
    doi = []
    count = 1
    for item in retri_result:
        if count>10:
                break
        if 'abstract' in item and len(item['abstract']) > 0:
                contexts.append(item['abstract'])
                doi.append(item["doi"])
                titles.append(item["title"])
                count+=1
        if 'paragraphs' in item:
                if item["doi"] in temp_doi:
                        if temp_doi[item['doi']] > 1:
                                continue
                        else:
                                temp_doi[item['doi']]+=1
                else:
                        temp_doi[item['doi']] = 1
                contexts.append(item['paragraphs'][0]['text'])
                doi.append(item["doi"])
                titles.append(item["title"])
                count+=1
    #print(len(doi), len(titles))
    qa_item['data'] = {'answer': '', 'context':contexts, 'doi': doi, 'titles': titles}
    data_for_qa = [qa_item]
    return data_for_qa


    
def information_retrieval_query(query):

    retri_result = retrieve_paragraph(query)
    data_for_qa = parse_ir_results(query, retri_result ,topk = 20)
    
    return data_for_qa

# QA System
class QA_System():
    def __init__(self):
        # Load the QA models. Please refer to [Github](https://github.com/yana-xuyan/caireCovid) for details.
        self.model = QaModule(['mrqa', 'biobert'], ["/kaggle/input/cairecovidqa/caire-MRQA-tpu/caire-MRQA-tpu", "/kaggle/input/cairecovidqa/BioBERT/BioBERT/exported-tf-model"], \
                              "/kaggle/input/xlnetlargecased/xlnet_cased_L-24_H-1024_A-16/spiece.model", "/kaggle/input/cairecovidqa/BioBERT/BioBERT/bert_config.json", \
                              "/kaggle/input/cairecovidqa/BioBERT/BioBERT/vocab.txt")
    def getAnswer(self, query):
        data_for_qa = information_retrieval_query(query)
        print(data_for_qa)
        answers =  self.model.getAnswers(data_for_qa)
        return answers
    def getAnswers(self, filename):
        data_for_qa = information_retrieval(query)
        answers = self.model.getAnswers(data_for_qa)
        return answers
    
    def makeFormatAnswers(self, answers):
        format_answers = []
        for i in range(len(answers[0]['data']['answer'])):
                format_answer = {}
                format_answer['question'] = answers[0]['question']
                format_answer['answer'] = answers[0]['data']['answer'][i]
                format_answer['context'] = answers[0]['data']['context'][i]
                format_answer['doi'] = answers[0]['data']['doi'][i]
                format_answer['title'] = answers[0]['data']['title'][i]
                format_answer["confidence"] = answers[0]['data']['confidence'][i]
                format_answer["raw"] = answers[0]['data']['raw'][i]
                format_answer["matching_score"] = answers[0]['data']['matching_score'][i]
                format_answer["rerank_score"] = answers[0]['data']['rerank_score'][i]
                format_answers.append(format_answer)
        return format_answers
    
    def rankAnswers(self, answers):
        return get_rank_score(answers)

def get_QA_answer_api(query):
    url = "http://eez114.ece.ust.hk:5000/query_qa"
    payload = "{\n\t\"text\": \""+query+"\"\n}"
    headers = {
        'Content-Type': "application/json",
        'cache-control': "no-cache",
        'Postman-Token': "696fa512-5fed-45ca-bbe7-b7a1b4d19fe4"
    }
    response = requests.request("POST", url, data=payload, headers=headers)
    response = response.json()
    return response

def display_QA(result):
    pdata = []
    for i in range(len(result)):
        line = []
        context_1 = "<div> "
#         best_paragraph_idx = result[i]['all_score'].index(max(result[i]['all_score']))
        
#         if best_paragraph_idx == 0:
        golden = result[i]['answer']
        context = result[i]['context']
        context_sents = sent_tokenize(context)
        golden_sents = sent_tokenize(golden)
        for sent in context_sents:

            if sent not in golden:
                context_1 += sent
            else:
                context_1 += "<font color='red'> "
                context_1 += sent
                context_1 += " </font>"
        context_1 += " </div>"

        line.append(context_1)
        # <a href="https://doi.org/10.1186/1471-2458-7-208" target="_blank">Non-pharmaceutical public health interventions for pandemic influenza: an evaluation of the evidence base</a>
        context_2 = '<a href= "https://doi.org/'
        context_2 += result[i]['doi']
        context_2 += ' target="_blank">'
        context_2 += result[i]['title']
        context_2 += '</a>'
        line.append(context_2)
        pdata.append(line)
    df = pd.DataFrame(pdata, columns = ['QA results', 'title'])
    HTML(df.to_html(render_links=True, escape=False))
#     display(HTML(df.to_html(render_links=True, escape=False)))
    df = df.style.set_properties(**{'text-align': 'left'})
    display(df)

QA_model = QA_System()
query = "How long is the incubation period of covid-19?"
answers1 = QA_model.getAnswer(query)
scored_answers1 = QA_model.rankAnswers(answers1)
format_answers1 = QA_model.makeFormatAnswers(scored_answers1)
display_QA(format_answers1)
# Call QA API
query = "How movement control strategies help to prevent COVID-19 secondary transmission in health care and community settings?"
answers2 = get_QA_answer_api(query)
def rankAnswers(answers):
    for item in answers:
        query = item["question"]
        context = item['context']
        # make new query with only n. and adj.
        tokens = word_tokenize(query.lower())
        tokens = [word for word in tokens if word not in stop_words]
        tagged = pos_tag(tokens)
        query_token = [tag[0] for tag in tagged if 'NN' in tag[1] or 'JJ' in tag[1] or 'VB' in tag[1]]

        text = context.lower()
        count = 0
        text_words = word_tokenize(text)
        for word in text_words:
            if word in query_token:
                count += 1
            
        matching_score = count / (1 + math.exp(-len(text_words)+50))/ 5 
        item['matching_score'] = matching_score
        item['rerank_score'] = matching_score + 0.4 * item['confidence']
        
    # sort QA results
    answers.sort(key=lambda k: k["rerank_score"], reverse=True)
    return answers
scored_answers2 = rankAnswers(answers2)
# format_answers2 = makeFormatAnswers(scored_answers2)
display_QA(scored_answers2)
