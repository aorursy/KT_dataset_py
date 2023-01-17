import nltk
import numpy as np

import os

from nltk.corpus import machado

import unicodedata



os.environ['NLTK_DATA'] = '../input/machado/'



# Remove acentos e coloca palavras em minúsculas

def strip_accents_and_lower(s):

    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()



machado_sents = map(lambda sent: list(map(strip_accents_and_lower, sent)), machado.sents())



# 'Executa' o mapeamento da lista

%time machado_sents = list(machado_sents)
import gensim



# Tamanho do 'embedding'

N = 200



# Número de palavras anteriores a serem consideradas

C = 7



%time model = gensim.models.Word2Vec(machado_sents, sg=0, size=N, window=C, min_count=5, hs=0, negative=14)
# Funções auxiliares



# Embedding de uma palavra

def word_embedding(word):

    return model[word]



# Pega apenas as palavras a partir do resultado da função 'most_similar'

def strip_score(result):

    return [w for w, s in result]



# Lista as palavras mais próximas

def closest_words(word, num=5):

    word_score_pair = model.most_similar(word, topn=num)

    return strip_score(word_score_pair)
# Exemplo de um embedding

emb = word_embedding('homem')



print(emb.shape)

print(emb)
# Exibe algumas palavras próximas daquelas contidas nesta lista

test_words = ['seja', 'foi', 'amou', 'aquele', 'foram', 'homem', 'rua', 'marcela']



for w in test_words:

    print(w, closest_words(w))
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.random_projection import GaussianRandomProjection



import matplotlib.pyplot as plt



%matplotlib inline



seed_words = set([

    # Personagens:

    'quincas', 'cubas',

        

    # Verbos:

    'estar', 'encontrar',

    

    # Objetos

    'bolsa', 'relogio'

])



# Vamos montar uma lista com as palavras raiz + 3 palavras próximas

all_words = []

for w in seed_words:

    all_words.append(w)

    all_words.extend(closest_words(w, 3))

    

all_words = set(all_words)



# Converte cada palavra para sua representação (embedding)

high_dim_embs = np.array(list(map(word_embedding, all_words)))



# Cria os gráficos (3 linas e 1 coluna)

fig, axes = plt.subplots(3, 1, figsize=(16, 16))



# Função que irá desenhar as palavras

def plot_labels(ax, high_dim_embs, words, dim_reduction):

    # Faz a redução para 2 dimensões (espera-se que o parâmetro `dim_reduction`

    # esteja corretamente configurado)

    low_dim_embs = dim_reduction.fit_transform(high_dim_embs)

    

    ax.set_title(dim_reduction.__class__.__name__)

    

    # Agora vamos "desenhar" cada palavra em sua posição x, y

    for (x, y), w in zip(low_dim_embs, words):

        ax.scatter(x, y, c='red' if w in seed_words else 'blue')

        ax.annotate(w,

                     xy=(x, y),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

        

# Chama a rotina de desenho, modificando o algoritmo de redução de dimensão



## PCA:

plot_labels(axes[0], high_dim_embs, all_words, PCA(2))



## TSNE:

plot_labels(axes[1], high_dim_embs, all_words, TSNE(2, perplexity=10, method='exact'))



## GaussianRandomProjection:

plot_labels(axes[2], high_dim_embs, all_words, GaussianRandomProjection(2))
# Faz a conjungação do verbo no passado, a partir do exemplo do verbo 'amar' -> 'amou'



def past(verb):

    result = model.most_similar(positive=['amou', verb], negative=['amar'], topn=5)

    return strip_score(result)



verbs = ['amar', 'explicar', 'contar', 'falar']



for verb in verbs:

    print(verb, past(verb))
# Agora vamos calcular o passado de um verbo, só que utilizando vários exemplos

# A idéia é calcular um vetor médio que leve da região do 'infinitivo' para o 'passado'

past_table = [

    ('andar', 'andou'),

    ('chegar', 'chegou'),

    ('sair', 'saiu'),

    ('sentir', 'sentiu'),

    ('perder', 'perdeu'),

    ('lembrar', 'lembrou')

]



positives = list(map(lambda v: v[1], past_table))

negatives = list(map(lambda v: v[0], past_table))



def past2(verb):

    # Aqui iremos incluir todos os exemplos positivos (relacionados ao 'passado'), os negativos

    # (relacionados ao 'infinitivo'), e o verbo que se quer 'calcular o passado'.

    # Note que temos que multiplicar o verbo pelo número de exemplos que temos a fim de se 

    # manter a proporção

    result = model.most_similar(positive=positives + [verb] * len(past_table), negative=negatives, topn=5)

    return strip_score(result)



for verb in verbs:

    print(verb, past2(verb))
past2('levantar')