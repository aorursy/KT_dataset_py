import numpy as np

from gensim.models.word2vec import Word2Vec

from gensim.models.callbacks import CallbackAny2Vec

from os import path

from pprint import pprint

from scipy.linalg import svd, norm, orthogonal_procrustes

from scipy.spatial.distance import cosine, euclidean

from sklearn.decomposition import PCA

import seaborn as sns

import matplotlib.pyplot as plt

import spacy



class EpochLogger(CallbackAny2Vec):

    def __init__(self):

        self.epoch = 0

    def on_epoch_begin(self, model):

        print("Epoch #{} start".format(self.epoch))

    def on_epoch_end(self, model):

        print("Epoch #{} end".format(self.epoch))

        self.epoch += 1



MODEL_PATH = "../input/domain-word2vec"



models = dict()



models['cs'] = Word2Vec.load(path.join(MODEL_PATH, "cs.bin"))

models['med'] = Word2Vec.load(path.join(MODEL_PATH, "medicine.bin"))

models['sport'] = Word2Vec.load(path.join(MODEL_PATH, "sports.bin"))

models['ele'] = Word2Vec.load(path.join(MODEL_PATH, "ee.bin"))

models['mec'] = Word2Vec.load(path.join(MODEL_PATH, "me.bin"))
for model in models:

    print(model, len(models[model].wv.vocab))
for model in models:

    

    # Mean centering

    words = []

    matrix = []

    for word in models[model].wv.vocab:

        words.append(word)

        matrix.append(models[model].wv[word])

    mean = np.mean(matrix, axis=0)

    models[model].wv.add(words, matrix - mean, replace=True)

    

    # Length normalization

    models[model].init_sims(replace=True)
nlp = spacy.load('en')

tags = dict()



def get_tag(word):

    if word in tags:

        return tags[word]

    tag = list(nlp(word))[0].pos_

    tags[word] = tag

    return tag
def get_sorted_vocab(domain):

    with_freq = [(word, vocab_entry.count) for word, vocab_entry in models[domain].wv.vocab.items()]

    return sorted(with_freq, key=lambda x: -x[1])
def get_frequent_shared_words_spacy(domains, size=100):

    sorted_vocabs = list()

    for domain in domains:

        sorted_vocabs.append(get_sorted_vocab(domain))

    min_vocab_size = min([len(sorted_vocab) for sorted_vocab in sorted_vocabs])

    head = size

    while True:

        common = set([word for word, freq in sorted_vocabs[0][:head]])

        for sorted_vocab in sorted_vocabs[1:]:

            common = common.intersection(set([word for word, freq in sorted_vocab[:head]]))

        if len(common) < size:

            head += 1

            continue

        to_remove = set()

        for word in common:

            if len(word) == 1:

                to_remove.add(word)

            elif word.isnumeric():

                to_remove.add(word)

            elif get_tag(word) not in ['NOUN', 'PROPN', 'VERB' 'ADJ']:

                to_remove.add(word)

        common.difference_update(to_remove)

        if len(common) >= size or head >= min_vocab_size:

            return common

        head += 1
# def get_ambiguity_scores(domains, min_freq=1000, min_ratio=0.3):



#     vocab = set(models[domains[0]].wv.vocab.keys())

#     vectors = dict()

#     for word in models[domains[0]].wv.vocab:

#         vectors[word + '_' + domains[0]] = models[domains[0]].wv[word] 

#     matrix = dict()

#     domains_left = domains[1:]

#     domains_added = [domains[0]]

#     for domain in domains_left:

#         X = []

#         y = []

#         for word in models[domain].wv.vocab:

#             avg = np.zeros((50,))

#             n = 0

#             for added_domain in domains_added:

#                 if word + '_' + added_domain in vectors:

#                     avg += vectors[word + '_' + added_domain]

#                     n += 1

#             if n:

#                 avg /= n

#                 X.append(models[domain].wv[word])

#                 y.append(avg)

#         X, y = np.array(X), np.array(y)

#         domains_added.append(domain)

#         vocab = vocab.union(models[domain].wv.vocab.keys())



#         U, s, VT = svd(np.matmul(y.T, X))

#         matrix[domain] = np.matmul(VT.T, U.T)

#         for word in models[domain].wv.vocab:

#             vectors[word + '_' + domain] = np.matmul(models[domain].wv[word], matrix[domain])



#     matrix[domains[0]] = np.eye(50)

    

#     word_scores = []    

#     for word in vocab:

#         if len(word) == 1 or word.isnumeric() or get_tag(word) not in ['NOUN', 'VERB', 'ADJ']:

#             continue

#         v = []

#         d = []

# #         max_freq = get_max_freq(word, domains)

# #         if max_freq < min_freq:

# #             continue

#         counts = []

#         for domain in domains:

#             try:

#                 counts.append(models[domain].wv.vocab[word].count)

#             except KeyError:

#                 counts.append(0)

#         counts.sort()

#         if counts[-1] < min_freq or counts[-2] < min_ratio * counts[-1]:

#             continue

#         for domain in domains:

#             if word + '_' + domain in vectors:# and models[domain].wv.vocab[word].count >= min_ratio * max_freq:

#                 d.append(domain)

#                 v.append(vectors[word + '_' + domain])

#         l = len(d)

#         if l > 1:

#             cos = 0

#             for i in range(l - 1):

#                 for j in range(i + 1, l):

#                     cos += cosine(v[i], v[j])

#             word_scores.append([word, cos / ((l * (l - 1)) / 2), d])

    

#     return sorted(word_scores, key=lambda x : x[1], reverse=True), matrix
def get_ambiguity_scores(domains, min_freq=800, min_ratio=0.3, th=0.001):



    vocab = set()

    for domain in domains:

        vocab = vocab.union(models[domain].wv.vocab.keys())

    Y = dict()

    c = dict()

    for word in vocab:

        Y[word] = np.zeros((50,))

        c[word] = 0

    for domain in domains:

        for word in models[domain].wv.vocab:

            Y[word] += models[domain].wv[word]

            c[word] += 1

    for word in vocab:

        Y[word] /= c[word]

    vectors = dict()

    for word in models[domains[0]].wv.vocab:

        vectors[word + '_' + domains[0]] = models[domains[0]].wv[word] 

    matrix = dict()

    iteration = 1

    errors = []

    while True:

        for domain in domains:

            X = []

            y = []

            for word in models[domain].wv.vocab:

                X.append(models[domain].wv[word])

                y.append(Y[word])

            X, y = np.array(X), np.array(y)

            

            matrix[domain], _ = orthogonal_procrustes(X, y, check_finite=False)

        

        for word in vocab:

            Y[word] = np.zeros((50,))

        for domain in domains:

            for word in models[domain].wv.vocab:

                Y[word] += np.matmul(models[domain].wv[word], matrix[domain])

        for word in vocab:

            Y[word] /= c[word]

        

        e = 0

        for domain in domains:

            X = []

            y = []

            for word in models[domain].wv.vocab:

                X.append(models[domain].wv[word])

                y.append(Y[word])

            e += norm(y - np.matmul(X, matrix[domain])) / ((len(y) * 50) ** 0.5)

        e /= len(domains)

        errors.append(e)

        if iteration > 1:

            if e_prev - e < th:

                break

        e_prev = e

        iteration += 1

    

    for domain in domains:

        for word in models[domain].wv.vocab:

            vectors[word + '_' + domain] = np.matmul(models[domain].wv[word], matrix[domain])



    word_scores = []    

    for word in vocab:

        if len(word) == 1 or word.isnumeric() or get_tag(word) not in ['NOUN', 'PROPN', 'VERB', 'ADJ']:

            continue

        v = []

        d = []

        counts = []

        for domain in domains:

            try:

                counts.append(models[domain].wv.vocab[word].count)

            except KeyError:

                counts.append(0)

        counts.sort()

        if counts[-1] < min_freq or counts[-2] < min_ratio * counts[-1]:

            continue

        for domain in domains:

            if word + '_' + domain in vectors:# and models[domain].wv.vocab[word].count >= min_ratio * max_freq:

                d.append(domain)

                v.append((domain, word, vectors[word + '_' + domain]))

        l = len(d)

        if l > 1:

            cos = 0

            weight_sum = 0

            for i in range(l - 1):

                for j in range(i + 1, l):

                    weight = models[v[i][0]].wv.vocab[v[i][1]].count + models[v[j][0]].wv.vocab[v[j][1]].count

                    cos += cosine(v[i][2], v[j][2]) * weight

                    weight_sum += weight

            word_scores.append([word, cos / weight_sum, d])

    

    return sorted(word_scores, key=lambda x : x[1], reverse=True), matrix, errors
def get_ambiguity_scores_ferrari(domains, min_freq=800, min_ratio=0.3, w2v_topn=100):

    output = list()

    vocab = set()

    for domain in domains:

        vocab = vocab.union(models[domain].wv.vocab.keys())

        

    for word in vocab:

        if len(word) == 1 or word.isnumeric() or get_tag(word) not in ['NOUN', 'PROPN', 'VERB', 'ADJ']:

            continue

        counts = []

        for domain in domains:

            try:

                counts.append(models[domain].wv.vocab[word].count)

            except KeyError:

                counts.append(0)

        counts.sort()

        if counts[-1] < min_freq or counts[-2] < min_ratio * counts[-1]:

            continue

        sorted_tops = list()

        sorted_words = list()

        tops = list()

        for domain in domains:

            try:

                sorted_tops.append(models[domain].wv.most_similar(word, topn=w2v_topn))

            except:

                continue

            sorted_words.append([word for word, score in sorted_tops[-1]])

            tops.append(dict(sorted_tops[-1]))

        

        shared = set()

        for top_word in tops:

            shared.update(top_word.keys())



        mse = 0

        for shared_word in shared:

            min_rank = w2v_topn + 1



            for sorted_word in sorted_words:

                try:

                    min_rank = min(min_rank, sorted_word.index(shared_word) + 1)

                except:

                    pass

            scores = list()

            for top in tops:

                scores.append(top.get(shared_word, 0))



            mse += np.var(scores) / min_rank

        counts = list()

        for domain in domains:

            try:

                counts.append(models[domain].wv.vocab[word].count)

            except KeyError:

                counts.append(0)

        output.append((word, len(shared), mse, counts))

    return sorted(output, key=lambda x: -x[2])
matrices = []

errors = []

scenarios = {'CS_EEN': ['cs', 'ele'],

             'CS_MEN': ['cs', 'mec'],

             'CS_MED': ['cs', 'med'],

             'CS_SPO': ['cs', 'sport'],

             'medical_device': ['cs', 'ele', 'med'],

             'medical_robot': ['cs', 'ele', 'mec', 'med'],

             'sport_rehab_machine': ['cs', 'ele', 'mec', 'med', 'sport']}

ambiguous = []

for scenario_name in scenarios:   

    ambiguous, domain_matrix, e = get_ambiguity_scores(scenarios[scenario_name], 1000, 0.5, 0.001)

    ambiguous_ferrari = get_ambiguity_scores_ferrari(scenarios[scenario_name], 1000, 0.5)

    matrices.append(domain_matrix)

    errors.append(e)

    print(scenario_name, len(ambiguous), '\n')

    pprint([(term, score, domains) for (term, score, domains) in ambiguous[:10]], width=200)

    pprint([(term, score, domains) for (term, score, domains) in ambiguous[len(ambiguous)-10:]], width=200)

    

#     l = []

#     for i in range(len(ambiguous)):

#         for j in range(len(ambiguous_ferrari)):

#             if ambiguous_ferrari[j][0]==ambiguous[i][0]:

#                 l.append([ambiguous[i][0], i, j, abs(i-j)])

#     l.sort(key=lambda x:x[3])

#     print()

#     pprint([(w,i,j,d) for (w,i,j,d) in l[len(l)-10:]], width=200)
sns.set(style='dark')

plt.figure(figsize=(10, 10))

ax = sns.lineplot(x=range(1,len(errors[0]) + 1), y=errors[0], label=list(scenarios.keys())[0])

for i in range(1, 7):

    sns.lineplot(x=range(1,len(errors[i]) + 1), y=errors[i], label=list(scenarios.keys())[i], ax=ax)
vec = dict()

domains = ['cs', 'ele', 'mec', 'med', 'sport']

for domain in domains:

    for w in models[domain].wv.vocab.keys():

        vec[w + '_' + domain] = models[domain].wv[w]



pca = PCA(n_components=2)

pcavec = pca.fit(list(vec.values()))
sns.set(style='dark')

fig, axs = plt.subplots(ncols=5, figsize=(75, 15))

c = ['Reds', 'Oranges', 'Blues', 'Purples', 'Greens']

for i in range(5):

    a = []

    for w in models[domains[i]].wv.vocab.keys():

        a.append(vec[w + '_' + domains[i]])

    sns.kdeplot(pca.transform(a)[:,0], pca.transform(a)[:,1], cmap=c[i], shade=True, ax=axs[i])
vec = dict()

domains = ['cs', 'ele', 'mec', 'med', 'sport']

for domain in domains:

    for w in models[domain].wv.vocab.keys():

        vec[w + '_' + domain] = np.matmul(matrices[6][domain], models[domain].wv[w])
fig, axs = plt.subplots(ncols=2, figsize=(30, 15))

c = ['Reds', 'Oranges', 'Blues', 'Purples', 'Greens']

for i in range(2):

    a = []

    for w in models[domains[i]].wv.vocab.keys():

        a.append(vec[w + '_' + domains[i]])

    sns.kdeplot(pca.transform(a)[:,0], pca.transform(a)[:,1], cmap=c[i], shade=True, ax=axs[i])
# dist = []

# for v in vec.keys():

#     dist.append([v, cosine(vec['kingdom_ele'], vec[v])])

# dist = sorted(dist, key=lambda x:x[1])

# dist[:100]

# # for i in range(1000):

# #     if dist[i][0][-1] != 's':

# #         print(i, dist[i])
# models['cs'].wv.most_similar('kingdom', topn=500)