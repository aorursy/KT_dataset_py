import os
import numpy as np
import re
import random
import pandas as pd
from csv import reader
def SegmentLineToWordsList(string):
    return list([x.lower() for x in re.split(r'[\s]\s*',string.strip()) if x])
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            if row[3] == '':
                continue
            if row[2] == 'rating':
                continue
            sentense = re.sub("[%s]+"%('"|#|$|%|&|\|(|)|\[|\]|*|+|\-|/|<|=|>|@|^|`|{|}|~|_|,|.|?|!|:|;'), ' ', row[3])
            sentense = re.sub("[%s]+"%('\''),'',sentense)
            pattern = r'\w+'
            ascii_pattern = re.compile(pattern, re.ASCII)
            if len(ascii_pattern.findall(sentense)) == len(SegmentLineToWordsList(sentense)):
                index = round(float(row[2]) / 2)
                index = int(index)
                if index == 5:
                    index = 4
                dataset.append([sentense, index])
    return dataset

dataset_org = load_csv('../input/boardgamegeek-reviews/bgg-13m-reviews.csv')
print(len(dataset_org))
def splitDataset(dataset, ratio_train):
    random.shuffle(dataset)
    cnt_train = round(len(dataset) * ratio_train ,0)
    train = []
    test = []
    for i in range(int(cnt_train)):
        train.append(dataset[i])
    for i in range(int(cnt_train) ,len(dataset)):
        test.append(dataset[i])
    return train, test

train = []
test = []
train, test = splitDataset(dataset_org, 0.75)
print(len(train))
print(len(test))
stopSet = set({'i', 'im', 'me', 'my', 'myself', 'we', 'our', 'ours', 'us', 'ourselves', 'you', 'your', 'yours', 
               'yourself', "youve", 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 
               'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 
               'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 
               'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
               'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 
               'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 
               'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 
               'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
               'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 
               'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
               'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 
               'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 
               'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn', 
               "havent", "wont", 'mustnt', "neednt", 'couldnt', 'doesnt', "shouldnt", "wasnt", 'wouldnt', "shes",
               "shouldve", "werent", "isnt", "dont", "arent", "thatll", "hasnt", "didnt", "mightnt", "hadnt", 'youre', 'theyre', })
def SegmentLineToWordsSet(sentense):
    sentense = re.sub("[%s]+"%('"|#|$|%|&|\|(|)|\[|\]|*|+|\-|/|<|=|>|@|^|`|{|}|~|,|.|?|!|:|;'), ' ', sentense)
    sentense = re.sub("[%s]+"%('\''),'',sentense)
    #return set([x.lower() for x in re.split(r'[\s|,|;|.|/|\[|\]|;|\!|?|\'|\\|\)|\(|\"|@|&|#|-|=|*|%|>|<|^|-]\s*',sentense.strip()) if x and x not in stopSet and len(x) > 1])
    return set([x.lower() for x in re.split(r'[\s]\s*',sentense.strip()) if x])

def buildVocabularyList(dataset):
    dict_list = {}
    pattern = re.compile('[0-9]+')
    for row in dataset:
        words = list(SegmentLineToWordsList(str(row[0]))) #Words that appear multiple times in the same comment are counted only once
        #words = set()
        #words = words.union(SegmentLineToWordsSet(str(row[0])))
        for word in words:
            if word in stopSet or len(word) == 1:
            #if len(word) == 1:
                continue
            if pattern.findall(word):
                continue
            if word not in dict_list:
                dict_list[word] = [0,0,0,0,0,0] #0-10 is rating,11 is sum
            dict_list[word][row[1]] += 1
            dict_list[word][len(dict_list[word])-1] += 1
    for word in list(dict_list.keys()):
        if dict_list[word][len(dict_list[word])-1] < len(train) * 0.00002:
            del dict_list[word]
    return dict_list
train_dict = buildVocabularyList(train)
train_dict
def getRatingProbability(dataset):
    rating_num = [0,0,0,0,0]
    for row in dataset:
        rating_num[row[1]] += 1
    return rating_num
rating_num = getRatingProbability(dataset_org)
print(rating_num)
def getClassWordNum(dataset):
    word_num = [0,0,0,0,0]
    for word in list(train_dict.keys()):
        for i in range(0,len(word_num)):
            word_num[i] += train_dict[word][i]
    return word_num
word_num = getClassWordNum(dataset_org)
print(word_num)
lambda_value = 0.0005
lambda_cag = len(rating_num) * lambda_value
def getConditionalProbabilityUsingSmoothing(word):
    conditional_probability = list()
    for i in range(0,len(rating_num)):
        if word not in train_dict:
            pro = lambda_value/(len(train_dict)*lambda_value+word_num[i])
        else:
            pro = (lambda_value + train_dict[word][i])/(len(train_dict)*lambda_value+word_num[i])
        conditional_probability.append(pro)
    return conditional_probability
def predict(review):
    words = set()
    words = words.union(SegmentLineToWordsSet(review))
    probability = np.array(rating_num) / len(train)
    pattern = re.compile('[0-9]+')
    for word in words:
        if pattern.findall(word):
                continue
        if word not in stopSet and len(word) > 1:
        #if len(word) > 1:
            probability *= getConditionalProbabilityUsingSmoothing(word)
    probability = list(probability)
    return probability.index(max(probability))
def accuracy_metric(test_dataset):
    correct = 0
    for row in test_dataset:
        if row[1] == predict(str(row[0])):
            correct += 1
    return correct / float(len(test_dataset)) * 100.0
train_part = list()
for i in range(1,1000):
    train_part.append(train[i])
print('Accuracy: %.3f%%' % accuracy_metric(train_part))
test_part = list()
for i in range(1,10000):
    test_part.append(test[i])
print('Accuracy: %.3f%%' % accuracy_metric(test_part))
print('Accuracy: %.3f%%' % accuracy_metric(test))
f = open('dict_file.txt','w')
f.write(str(train_dict))
f.close()
f = open('dict_file.txt','r')
a = f.read()
read_dictionary = eval(a)
f.close()
print(read_dictionary['greatest'])
print(len(read_dictionary))
print(len(train_dict))