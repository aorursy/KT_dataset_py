import pandas as pd

import numpy as np

import json

import os.path

import re



metadata=pd.read_csv("C:/temp/CORD-19-research-challenge/metadata.csv")

metadata=metadata[metadata["sha"]==metadata["sha"]] #filters out entries having sha=NaN

metadata=metadata[metadata["has_full_text"]] #filters out entries for which the full text is not available

metadata.shape

metadata=metadata[20000:28462]

#Verified upto 0 to 20K

def path(shaid): #returns path of .json file

    for dirname, _, files in os.walk('C:/temp/CORD-19-research-challenge/'):

        if shaid+'.json' in files:

            return os.path.join(dirname,shaid+".json")

metadata["Path"]=metadata.apply(lambda x: path(x["sha"]),axis=1) #this takes a while unfotunately

metadata=metadata[metadata["Path"]==metadata["Path"]]







Texts={} #dictionary: {id: "Text"}; adapted from cristian's notebook (https://www.kaggle.com/crprpr/vaccine-data-filter)

valid_id=[]

for shaid,file in zip(metadata["sha"],metadata["Path"]):

    with open(file, 'r') as f:

        doc=json.load(f)

    MainText=''

    for item in doc["body_text"]:

        MainText=MainText+(re.sub('[^a-zA-Z0-9]', ' ', item["text"].lower()))

    match1=re.search(r'therapeutic|therapy', MainText)

    if match1 is not None:

        match2=re.search(r'COVID-19 |SARS |SARS-CoV2 |AS-SCoV2 |SARS-CoV-2 ', MainText, re.IGNORECASE)

        if match2 is not None:

            Texts[shaid]=MainText

            valid_id.append(shaid)



print (Texts)





data1=open("C:/temp/CORD-19-research-challenge/level1.txt", 'r')

f = open("C:/temp/CORD-19-research-challenge/level4.txt","w",encoding="utf8")



data2=[1]

p=1

while len(data2)>0:

    data2=data1.read(20000000)

    if data2 is not None:

        data3 =  data2.split('waseem')

    for x1 in data3:

        x2=x1.split('\n')

        if x2 is not None:

            for x in x2:

                match1=re.search(r'therapeutic|therapy', x)

                if match1 is not None:

                    y=x2.index(x)

                    if len(x2)>2:

                        try:

                            z=' '.join([x2[y-2],x2[y-1],x2[y],x2[y+1],x2[y+2]])

                            match2=re.search(r'COVID|CoV2 |SCoV2 |CoV-2 ', z, re.IGNORECASE)

                            if match2 is not None:

                                f.write(z)

                                f.write('\n')

                        except IndexError:

                            pass

    quit

            

f.close()

 


