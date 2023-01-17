import matplotlib.pyplot as plt

%matplotlib inline

import pandas

import numpy

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
debate = pandas.read_csv('../input/debate.csv',encoding = 'iso-8859-1')

debate.head(10)
Debate=debate[debate['Date']=='2016-09-26']
print(Debate['Speaker'].unique())

print(Debate[Debate['Speaker']=='Audience']['Text'].unique())

print(len(Debate[Debate['Speaker']=='Clinton']))

print(len(Debate[Debate['Speaker']=='Trump']))

print(len(Debate[Debate['Speaker']=='Holt']))

print(len(Debate[Debate['Speaker']=='Audience']))
Debate=Debate.iloc[7:350,:].reset_index(drop=True)
Interrupt_clinton=[]

Interrupt_trump=[]

for i in range(len(Debate)-1):

    if Debate['Speaker'][i]=='Clinton' and Debate['Speaker'][i+1]=='Holt':

        Interrupt_clinton.append(i)

    elif Debate['Speaker'][i]=='Trump' and Debate['Speaker'][i+1]=='Holt':

        Interrupt_trump.append(i)
Laugh_clinton=[]

Applaud_clinton=[]

Laugh_Trump=[]

Applaud_Trump=[]

for i in range(len(Debate)-1):

    if Debate['Speaker'][i]=='Clinton' and Debate['Text'][i+1]=='(APPLAUSE)':

        Applaud_clinton.append(i)

    elif Debate['Speaker'][i]=='Trump' and Debate['Text'][i+1]=='(APPLAUSE)':

        Applaud_Trump.append(i)

    elif Debate['Speaker'][i]=='Clinton' and Debate['Text'][i+1]=='(LAUGHTER)':

        Laugh_clinton.append(i)

    elif Debate['Speaker'][i]=='Trump' and Debate['Text'][i+1]=='(LAUGHTER)':

        Laugh_Trump.append(i)
Laugh=[]

Interupted=[]

Applause=[]

Interuptted_text=[]

for i in range(len(Debate)):

    if i in Laugh_clinton or i in Laugh_Trump:

        Laugh.append(1)

    else:

        Laugh.append(0)

    if i in Applaud_clinton or i in Applaud_Trump:

        Applause.append(1)

    else:

        Applause.append(0)

    if i in Interrupt_clinton or i in Interrupt_trump:

        Interupted.append(1)

        Interuptted_text.append(Debate['Text'][i+1])

    else:

        Interupted.append(0)

        Interuptted_text.append('No Interruption')

    
Debate.insert(4,'Laugh',Laugh)

Debate.insert(5,'Interupted',Interupted)

Debate.insert(6,'Interupted Text',Interuptted_text)

Debate.insert(7,'Applause',Applause)

del Debate['Line']

del Debate['Date']
Debate=Debate[Debate['Speaker']!='Holt']

Debate=Debate[Debate['Speaker']!='Audience']

Debate=Debate[Debate['Speaker']!='CANDIDATES']
Debate.head()
from wordcloud import WordCloud

import re

import nltk

from nltk.corpus import stopwords
def to_words(content):

    letters_only = re.sub("[^a-zA-Z]", " ", content) 

    words = letters_only.lower().split()                             

    stops = set(stopwords.words("english"))                  

    meaningful_words = [w for w in words if not w in stops] 

    return( " ".join( meaningful_words )) 
def wordcloud(candidate):

    df=Debate[Debate['Speaker']==candidate]

    clean_text=[]

    for each in df['Text']:

        clean_text.append(to_words(each))

    if candidate=='Trump':

        color='black'

    else:

        color='white'

    wordcloud = WordCloud(background_color=color,

                      width=3000,

                      height=2500

                     ).generate(clean_text[0])

    print('==='*30)

    print('word cloud of '+candidate+' is plotted below')

    plt.figure(1,figsize=(8,8))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()
wordcloud('Trump')

wordcloud('Clinton')
ind = numpy.arange(3)

trump=(len(Laugh_Trump),len(Applaud_Trump),len(Interrupt_trump))

clinton=(len(Laugh_clinton),len(Applaud_clinton),len(Interrupt_clinton))

fig, ax = plt.subplots()

width=0.35

rects1 = ax.bar(ind, trump,width, color='r')

rects2 = ax.bar(ind+width , clinton, width,color='y')

ax.set_ylabel('Counts')

ax.set_title('Counts of behavior of mediator and audience')

ax.set_xticks(ind)

ax.set_xticklabels(('Making laugh','Making applaud','Be interrupted'),rotation=45)

ax.legend((rects1[0], rects2[0]), ('Trump', 'Clinton'))

plt.show()
def interruption_analytic(candidate):

    if candidate=='Trump':

        color1='black'

        color2='r'

    else:

        color1='white'

        color2='y'

    df=Debate[Debate['Speaker']==candidate]

    df=df[df['Interupted']==1]

    length=[]

    text=[]

    for each in df['Text']:

        text.append(to_words(each))

        length.append(len(to_words(each).split()))

    print("="*40+'Analytic of '+candidate+'='*40)

    plt.hist(length,facecolor=color2)

    plt.title("Histogram of the count of words when being interrupted/questioned.")

    plt.xlabel("Value")

    plt.ylabel("Frequency")

    plt.figure(1,figsize=(8,8))

    wordcloud = WordCloud(background_color=color1,

                      width=3000,

                      height=2500

                     ).generate(text[0])

    plt.figure(2,figsize=(8,8))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    
interruption_analytic('Trump')
interruption_analytic('Clinton')
trump_interupt=Debate[Debate['Speaker']=='Trump']

trump_interupt=trump_interupt[trump_interupt['Interupted']==1].reset_index(drop=True)

clinton_interupt=Debate[Debate['Speaker']=='Clinton']

clinton_interupt=clinton_interupt[clinton_interupt['Interupted']==1].reset_index(drop=True)
print('='*30+'Trump part'+'='*30)

print('Trump '+'\n'+trump_interupt['Text'][11])

print('Holt'+'\n'+trump_interupt['Interupted Text'][11])

print('Trump '+'\n'+trump_interupt['Text'][23])

print('Holt'+'\n'+trump_interupt['Interupted Text'][23])

print('Trump '+'\n'+trump_interupt['Text'][38])

print('Holt'+'\n'+trump_interupt['Interupted Text'][38])

print('Trump '+'\n'+trump_interupt['Text'][44])

print('Holt'+'\n'+trump_interupt['Interupted Text'][44])

print('='*30+'Clinton part'+'='*30)

print('Clinton '+'\n'+clinton_interupt['Text'][12])

print('Holt'+'\n'+clinton_interupt['Interupted Text'][12])

print('Clinton '+'\n'+clinton_interupt['Text'][17])

print('Holt'+'\n'+clinton_interupt['Interupted Text'][17])