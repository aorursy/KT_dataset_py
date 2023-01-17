!pip install wxconv
!pip install googletrans
import os
import pandas as pd
from wxconv import WXC
import re
import http.client
from googletrans import Translator
from tqdm import tqdm
translator = Translator(service_urls=[
      'translate.google.com','translate.google.co.kr', ])

prog = re.compile(r'^[a-zA-Z]+$')
listOfWX = []
itc = 'te-t-i0-und'

def request(word):
    conn = http.client.HTTPSConnection('inputtools.google.com')
    conn.request('GET', '/request?text=' + word + '&itc=' + itc + '&num=1&cp=0&cs=1&ie=utf-8&oe=utf-8&app=test')
    res = conn.getresponse()
    return res.read()
path = "../input/code-mixed-telugu/"

chats = []
for i in os.listdir(path):
    chats.append(i)
print(chats)

df = pd.DataFrame(columns=['text','target'])

eng_words = pd.read_csv('../input/corncob-english-lowercase/corncob_lowercase.txt')
eng_words=eng_words.values.tolist()
con = WXC(order='utf2wx', lang='tel')
chats.sort()
chats
for p in chats:
    print(p)
    
    fp = open(path+p,'r',encoding='utf8')
    
    ls = fp.read()
    ls = ls.split("\n")
    fp.close()
    
    ls_ = []
    
    
    for x in ls:
        y = x.split(":")
        ls_.append(y[-1].lower())
        
        
    tel_sen = []      
    telugu = []
    for input_string in tqdm(ls_):
        telugu = []
        eng_trans = []
        input_string = list(input_string.split(" "))
        translator = Translator()
        for i in input_string:
            i = i.replace("'s","")
            i = i.replace("?"," ")
            i = i.replace("'ll"," will")
            i = i.replace("."," ")
            i = i.replace(","," ")
            i = i.replace("!"," ")
            i = i.replace("<"," ")
            i = i.replace(">"," ")
            i = i.replace("*"," ")
            i = i.replace("i'm","i am")
            if [i] in eng_words or i.isnumeric():
                eng_trans.append(i)
            else:
                if len(eng_trans)>0:
                    try:
                        telugu.append(translator.translate(" ".join(eng_trans), src='en', dest='te').text)
                    except:
                        print(" ".join(eng_trans))
                    eng_trans = []
                if prog.match(i):
                    i = str(request(i), encoding = 'utf-8')[14+4+len(i):-31] 
                telugu.append(i)
        if len(eng_trans)>0:
                try:
                    telugu.append(translator.translate(" ".join(eng_trans), src='en', dest='te').text)
                except:
                    print(2," ".join(eng_trans))
                eng_trans = []
        tel_sen.append(" ".join(telugu))   

    data = pd.DataFrame()
    data['text'] = ls_
    data['target'] = tel_sen
    data.to_csv(p[:-4]+"-out-"+str(fc)+".csv", index=False, encoding = 'utf-8')
    fc+=1
files=[]
path = "../input/code-mixed-telugu/preprocessed data/"
for i in os.listdir(path):
    files.append(i)
files.sort()
files
df=pd.DataFrame()
for i in files:
    data = pd.read_csv(path+i).fillna("")
    df=pd.concat([df,data],ignore_index=True)
df.drop(index=df[df.text==""].index,inplace=True)
df.drop(index=df[df.text==" <media omitted>"].index,inplace=True)
df.reset_index(drop=True,inplace=True)
c_text=[]
for k in range(df.shape[0]):
    i=df.loc[k,'text']
    i = i.replace("?"," ")
    i = i.replace("'ll"," will")
    i = i.replace("."," ")
    i = i.replace(","," ")
    i = i.replace("!"," ")
    i = i.replace("<"," ")
    i = i.replace(">"," ")
    i = i.replace("*"," ")
    i = i.replace("i'm","i am")
    i = " ".join(i.split())
    c_text.append(i)
df["text"] = c_text
df.to_csv("cleaned data.csv", index=False)