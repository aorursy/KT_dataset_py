import copy

import operator

import os,glob,re

import numpy as np

from sklearn import metrics

from string import punctuation

from collections import defaultdict

from collections import Counter

from collections import OrderedDict

from textblob.classifiers import NaiveBayesClassifier

import pandas as pd
dataset_train = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
dataset_test = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
train22neg = open('train_neg.txt')
def ajuste (x):

    return 'ID-' + x

    
dataset_test['id']=dataset_test['id'].astype('str').apply(ajuste)
dataset_train['id']=dataset_train['id'].astype('str').apply(ajuste)

dataset_train['label_target'] = pd.cut(dataset_train['target'], [-1,0.4999,1], labels=['no_toxic','toxic'])

df_train_pos = dataset_train.loc[(dataset_train['label_target'])=='toxic']

train_pos = df_train_pos[['id','comment_text']]

df_train_neg = dataset_train.loc[(dataset_train['label_target'])=='no_toxic']

train_neg = df_train_neg[['id','comment_text']]

# Listas

list_negative_word = []

list_positive_word = []

Unique_list_positive_word = []

Unique_list_negative_word = []

cwd = os.getcwd() 

count_positive_ID = 0

count_negative_ID = 0

Negative_Word_Dict = defaultdict()

Positive_Word_Dict = defaultdict()

Sentence_Sentiment = {}

Evaluation_Dict = OrderedDict()



# Lista de palavras frequentes

frequent_word_list = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']

ID_Val = []



# Limpeza de dados usando expressões regulares. 

# Removemos as palavras frequentemente repetidas como descrito na lista acima. 

# Eliminamos todos os tipos de pontuação - como.,!, -, etc.

for line in open("../input/bases_v1 /train_pos.txt", encoding="utf8").readlines():

    if line.strip():

        word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()

        for each_word in word_split:

                        if re.match(r'[IDidIdiD].*-[0-9].*',each_word):

                            count_negative_ID+=1

                            ID_Val.append(each_word)

                            continue

                        elif re.match(r'(\d+)\.(\d+)+',each_word):

                            continue

                        elif re.match(r'(\d+)+',each_word):

                            continue 

                        elif re.match(r'$(\d+)+',each_word):

                            continue  

                        elif re.match(r'[A-Za-z]*\.+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'[A-Za-z]*\?+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'[A-Za-z]*\!+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'-+',each_word):

                            continue

                        elif re.match(r'[A-Za-z]*\)+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'\(+[A-Za-z]*',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'[A-Za-z]*\,+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                            

                        if each_word in frequent_word_list:

                            continue

                        else:

                            list_negative_word.append(each_word)



# Alimenta as listas

List_Size_Negative = len(list_negative_word)

Negative_Word_Dict =  Counter(list_negative_word)

Denominator_Sum_Neg = sum(Negative_Word_Dict.values())

Unique_list_negative_word = set(list_negative_word)

Unique_List_Size_Negative = len(Unique_list_negative_word)

list_positive_word = []



# Limpeza de dados usando expressões regulares. 

for line in open("../input/bases_v1 /train_neg.txt", encoding="utf8").readlines():

    if line.strip():

        word_split = line.split()

        word_split = line.replace(',',' ').replace('.',' ').replace('!',' ').replace('--',' ').split()

        for each_word in word_split:

                        if re.match(r'ID-[0-9].*',each_word):

                            count_positive_ID+=1

                            ID_Val = each_word

                            continue

                        elif re.match(r'(\d+)\.(\d+)+',each_word):

                            continue

                        elif re.match(r'(\d+)+',each_word):

                            continue 

                        elif re.match(r'$(\d+)+',each_word):

                            continue  

                        elif re.match(r'[A-Za-z]*\.+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'[A-Za-z]*\?+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'[A-Za-z]*\!+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'-+',each_word):

                            continue

                        elif re.match(r'[A-Za-z]*\)+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'\(+[A-Za-z]*',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'[A-Za-z]*\,+',each_word):

                            each_word = re.sub(r'[^\w\s]','',each_word)

                        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',each_word):

                            list_word = each_word.split('/')

                            if list_word[0] not in frequent_word_list:

                                list_negative_word.append(list_word [0])

                            else:

                                continue

                            

                            if list_word[1] not in frequent_word_list:

                                list_negative_word.append(list_word [1])

                            else:

                                continue

                        if each_word in frequent_word_list:

                            continue

                        else:

                            list_positive_word.append(each_word)
# Alimenta as listas

List_Size_Positive = len(list_positive_word)

Positive_Word_Dict = Counter(list_positive_word)

Denominator_Sum_Pos = sum(Positive_Word_Dict.values())

Unique_list_positive_word = set(list_positive_word)

Unique_List_Size_Positive = len(Unique_list_positive_word)

Total_No_of_Docs = count_positive_ID + count_negative_ID

# Imprime um resumo até aqui

print("\n************ Executando Exemplo1 ************")

print("\nResumo do Dataset: ")

print ("count_positive_ID - ",count_positive_ID)

print ("count_negative_ID - ",count_negative_ID)

print ("Número Total de docs - ",Total_No_of_Docs)



Log_Prior_Positive = np.log10(float(count_positive_ID)/float(Total_No_of_Docs))

Log_Prior_Negative = np.log10(float(count_negative_ID)/float(Total_No_of_Docs))



Negative_Word_Dict_final = defaultdict()

Positive_Word_Dict_final = defaultdict()

# Criando valores de estimativa de máxima verossimilhança para cada palavra no conjunto de treinamento positivo

for key in Positive_Word_Dict:    

    val = np.log10 (float(Positive_Word_Dict[key] + 1)/float(Denominator_Sum_Pos + Unique_List_Size_Positive))

    Positive_Word_Dict_final.update({key:val})



# Criando valores de estimativa de máxima verossimilhança para cada palavra no conjunto de treinamento negativo

for key in Negative_Word_Dict:    

    val =  np.log10(float(Negative_Word_Dict[key] + 1)/float(Denominator_Sum_Neg + Unique_List_Size_Negative))

    Negative_Word_Dict_final.update({key:val})



Negative_Probability_Sentence = 0.00

Positive_Probability_Sentence = 0.00

Sentence_Token_List = []

Sentence_List = []



# Abre o dataset de teste

with open("teste/teste.txt",'r',encoding="utf8") as f:

   token =  [line.split() for line in f]

   for each_word in token:

        Sentence_Token_List.append(each_word)



Test_Review_Dict = OrderedDict()

Test_Review_Class = OrderedDict()

temp_dict = {}

List_Tokens_ID = []

temp = []



# Processo de limpeza nos dados de teste

for i in range (0,len(Sentence_Token_List)):

    for j in range (0,len(Sentence_Token_List[i])):

        word = Sentence_Token_List[i][j]

        if re.match(r'ID-[0-9].*',word):

            Id_Value = word

            continue

        elif re.match(r'(\d+)\.(\d+)+',word):

            continue

        elif re.match(r'(\d+)+',word):

            continue 

        elif re.match(r'$(\d+)+',word):

            continue  

        elif re.match(r'[A-Za-z]*\.+',word):

            word = re.sub(r'[^\w\s]','',word)

        elif re.match(r'[A-Za-z]*\?+',word):

            word = re.sub(r'[^\w\s]','',word)

        elif re.match(r'[A-Za-z]*\!+',word):

            word = re.sub(r'[^\w\s]','',word)

        elif re.match(r'-+',word):

            continue

        elif re.match(r'[A-Za-z]*\)+',word):

            word = re.sub(r'[^\w\s]','',word)

        elif re.match(r'\(+[A-Za-z]*',word):

            word = re.sub(r'[^\w\s]','',word)

        elif re.match(r'[A-Za-z]*\,+',word):

            word = re.sub(r'[^\w\s]','',word)

        elif re.match(r'[A-Za-z]*\/+[A-Za-z]*',word):

            list_word = word.split('/')

            if list_word[0] in frequent_word_list:

                continue

            else:

                List_Tokens_ID.append(list_word[0])

            if list_word[1] in frequent_word_list:

                continue

            else:

                List_Tokens_ID.append(list_word[0])

                

        if (word not in frequent_word_list):

                List_Tokens_ID.append(word)

    temp = copy.copy(List_Tokens_ID)

    Test_Review_Dict.update ({Id_Value:temp})

    List_Tokens_ID[:] = []



for each_key,each_value in Test_Review_Dict.items():

    ID_Value = each_key

    for val in each_value:

        if ((val in Negative_Word_Dict_final) and (val in Positive_Word_Dict_final)):

            Negative_Probability_Sentence += Negative_Word_Dict_final[val]

            Positive_Probability_Sentence += Positive_Word_Dict_final[val]



    Positive_Probability_Sentence+=Log_Prior_Positive

    Negative_Probability_Sentence+=Log_Prior_Negative

    temp_dict.update({'0': Positive_Probability_Sentence})

    temp_dict.update({'1': Negative_Probability_Sentence})

    #print ("ID- Value : {}, Negative : {} , Positive : {}".format(ID_Value, Negative_Probability_Sentence,Positive_Probability_Sentence))

    key = [k for k,v in temp_dict.items() if v==max(temp_dict.values())][0] 

    Test_Review_Class.update ({ID_Value:key})

    temp_dict.clear()

    Positive_Probability_Sentence = 0.0

    Negative_Probability_Sentence = 0.0

    

with open("output/classificacoes_math.txt",'w') as ofile:

        for keys,values in Test_Review_Class.items():

            ofile.write((str(keys) + '\t' + values + '\n'))
