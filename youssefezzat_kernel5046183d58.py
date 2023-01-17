import numpy as np 
import pandas as pd 
import sys
import os
import re
from nltk.tokenize import word_tokenize
import fitz
sys.path.append(os.path.abspath(os.path.dirname('/kaggle/input/pdf')))

folderpath='/kaggle/input/pdf'

            
def pdf2text(path):
    
    text=list()
    doc = fitz.open(path)
    for page in doc:                            
        text.append(page.getText())
        
    return text
def get_pdf(folderpath):
    
    final=list()
    result=None
    #invalids = ['1124.pdf','1648.pdf']
    nFiles = len(os.listdir(folderpath))
 
    for f in os.listdir(folderpath):
        filepath = os.path.join(folderpath, f)
        #print(f)
        #if(not str(f)in invalids):
        try:
            final.append(pdf2text(filepath))
        except:
            continue
    #print("No. of files = ", nFiles) 
    #print("No. of parsed files = ", len(final)) 
    return final
def get_email(final):
    email = None
    match = re.search(r'[\w\.-]+@[\w\.-]+',final)
    if match is not None:
        email = match.group(0)
    return email
def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    return tokens
    
result=list()  
emails=list()
tokenized_Cvs=list()
result=get_pdf(folderpath)
for cv in result:
    #text=None
    #text=str(cv)
    #emails.append(get_email(text))
    tokenized_Cvs.append(tokenize(str(cv)))
                         
print(len(tokenized_Cvs))
                         
for t in tokenized_Cvs :
    print(t)
    break
import docx2txt
from os import listdir

folder_path = '/kaggle/input/word'
file_path = '/kaggle/input/word/1.docx'

my_text = ''
doc_data = list()
for file in listdir(folder_path):
    my_text = docx2txt.process(file_path)
    doc_data.append(my_text)

print(len(doc_data))