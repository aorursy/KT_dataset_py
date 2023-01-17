import json,csv,re,os
from zipfile import ZipFile
import re
a=[]
for k in os.walk("/kaggle/input/CORD-19-research-challenge/"):
    
    if 'pdf_json' in k[0] and len(k[2])>0:
        print(len(k[2]),k[0])
        for z in k[2]:
            if z.endswith(".json") :
                
                loaded_json=json.load((open(k[0]+'/'+z,"r")))
                body_string=[]
                
                title_string=re.sub('[^a-zA-Z0-9 \n\.]','',loaded_json['metadata']['title'])
                for x in loaded_json["body_text"]:
                    body_string.append(re.sub('[^a-zA-Z0-9 \n\.]','',x['text'])) 
                #print()
                if 'abstract' in loaded_json and len(loaded_json['abstract'])>0:
                   abstract_string=re.sub('[^a-zA-Z0-9 \n\.]','',loaded_json['abstract'][0]['text'] )
                else:
                   abstract_string=''
                a.append((title_string,abstract_string,body_string))
csv_read=open("/kaggle/working/read_csv_final1.csv","w")
for (i,j,k) in a:
    if i==a[0][0]:
      csv_read.write('"{}","{}","{}"'.format("title","abstract","paragraphs"))
      csv_read.write("\n")
    csv_read.write('"{}","{}","{}"'.format(i,j,k))
    csv_read.write("\n")
csv_read.close()
!pip install cdqa
import os
import pandas as pd
from ast import literal_eval

from cdqa.utils.filters import filter_paragraphs
from cdqa.pipeline import QAPipeline
from cdqa.utils.download import download_model, download_bnpp_data

download_bnpp_data(dir='./data/bnpp_newsroom_v1.1/')
download_model(model='bert-squad_1.1', dir='./models')
df = pd.read_csv('/kaggle/working/read_csv_final1.csv',converters={'paragraphs': literal_eval})
print(df.head())
df2 = filter_paragraphs(df)
print(df2.head())

# print(df.head())
cdqa_pipeline = QAPipeline(reader='./models/bert_qa.joblib')
cdqa_pipeline.fit_retriever(df=df2)
from IPython.display import display, Markdown, Latex, HTML

def show(y,x):
  # w=
  # dh(w)
  z="""<div><div class="question_title">{}</div><div class="single_answer">{}</div></div>""".format(y ,"<span class='answer'>" +x + "</span>")
  dh(z)
def layout_style():
    style = """
        div {
            color: black;
        }
        .single_answer {
            border-left: 3px solid #dc7b15;
            padding-left: 10px;
            font-family: Arial;
            font-size: 16px;
            color: #777777;
            margin-left: 5px;

        }
        .answer{
            color: #dc7b15;
        }
        .question_title {
            color: grey;
            display: block;
            text-transform: none;
        }      
        div.output_scroll { 
            height: auto; 
        }
    """
    return "<style>" + style + "</style>"

def dh(z): display(HTML(layout_style() + z))


queries = [
    'What is known about transmission, incubation, and environmental stability?',
    'What do we know about COVID-19 risk factors?',
    'What do we know about virus genetics, origin, and evolution?',
    'What do we know about vaccines and therapeutics?',
    'What do we know about non-pharmaceutical interventions?',
    'What has been published about medical care?',
    'What do we know about diagnostics and surveillance?'
    'What has been published about information sharing and inter-sectoral collaboration?',
    'What has been published about ethical and social science considerations?'
]
for query in queries:
  prediction= cdqa_pipeline.predict(query, n_predictions=20,retriever_score_weight=0.6)
  print('Query: {}'.format(query))
  # for x,y,z in zip(prediction[0][:-1],prediction[1][:-1],prediction[2][:-1]):
  show('Answer',str(prediction[0][-2]))
  show('Title',str(prediction[1][-2]))
  show('Paragraph',str(prediction[2][-2]))
    # if x!=prediction[0][-2]:
      # print('-------------Next Prediction-------------')
  if query!=queries[-1]:
    print('---------------Next Query---------------------')

  