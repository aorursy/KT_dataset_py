import json

import os

import heapq



import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
path_train = '/kaggle/input/tensorflow2-question-answering/simplified-nq-train.jsonl'

train = []

with open(path_train, 'r') as file:

    for i in range(1000):

        train.append(json.loads(file.readline()))
train[0].keys()
train[0]['long_answer_candidates'][:5]
def extract_corpus(doc):

    document_text = doc['document_text']

    long_answer_candidates = doc['long_answer_candidates']

    tokens = document_text.split(' ')

    corpus = []

    for candidate in long_answer_candidates:

        start_token = candidate['start_token']

        end_token = candidate['end_token']

        corpus.append(" ".join(tokens[start_token:end_token]))

    return corpus



def get_long_answer(doc):

    document_text = doc['document_text']

    tokens = document_text.split(' ')

    # even though annotatations is an array, it seems to be all length of 1

    long_answer_anno = doc['annotations'][0]['long_answer']

    start_token = long_answer_anno['start_token']

    end_token = long_answer_anno['end_token']

    long_answer = " ".join(tokens[start_token:end_token])

    return long_answer
# testing out extract_corpus

corpus = extract_corpus(train[0])

len(corpus)
corpus[0]
corpus[1]
corpus[2]
def get_top_n_candidates(corpus, question_text, n):

    tfidf = TfidfVectorizer(stop_words='english')

    X_corpus = tfidf.fit_transform(corpus)

    X_question_text = tfidf.transform([question_text])

    similarity = cosine_similarity(X_corpus, X_question_text)

    top_n_idx = heapq.nlargest(n, range(len(similarity)), similarity.take)

    top_n_candidates = [corpus[i] for i in top_n_idx]

    return top_n_candidates
def print_ranking(doc, n):

    question_text = doc['question_text']

    print('Question:')

    print(question_text)

    print()



    long_answer = get_long_answer(doc)

    print('Expected long answer:')

    print(long_answer)

    print()



    corpus = extract_corpus(doc)

    top_candidates = get_top_n_candidates(corpus, question_text, n)

    print('Ranked long answers:')

    found = False

    for idx, candidate in enumerate(top_candidates):

        if long_answer == candidate:

            print("CORRECT LONG ANSWER FOUND :)")

            found = True

        print(f"#{idx + 1}:")

        print(candidate)

        print()

    if not found:

        print("correct long answer not found :(")
print_ranking(train[0], 3)
print_ranking(train[1], 3)
print_ranking(train[2], 3)
print_ranking(train[3], 3)
def find_long_answer(doc, n):

    question_text = doc['question_text']

    long_answer = get_long_answer(doc)

    corpus = extract_corpus(doc)

    top_candidates = get_top_n_candidates(corpus, question_text, n)

    candidate_match = [candidate for candidate in top_candidates if long_answer == candidate]

    found = True if len(candidate_match) > 0 else False

    return found

    

def calc_find_score(docs, n):

    num_found = 0

    for doc in docs:

        if find_long_answer(doc, n):

            num_found += 1

    return num_found / len(docs)
calc_find_score(train, 3)
calc_find_score(train, 5)
calc_find_score(train, 10)
calc_find_score(train, 20)