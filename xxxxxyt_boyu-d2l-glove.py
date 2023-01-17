import torch

import torchtext.vocab as vocab



print([key for key in vocab.pretrained_aliases.keys() if "glove" in key])

glove = vocab.GloVe(name='6B', dim=50)

print("一共包含%d个词。" % len(glove.stoi))

print(glove.stoi['beautiful'], glove.itos[3366])
def knn(W, x, k):

    '''

    @params:

        W: 所有向量的集合

        x: 给定向量

        k: 查询的数量

    @outputs:

        topk: 余弦相似性最大k个的下标

        [...]: 余弦相似度

    '''

    cos = torch.matmul(W, x.view((-1,))) / (

        (torch.sum(W * W, dim=1) + 1e-9).sqrt() * torch.sum(x * x).sqrt())

    _, topk = torch.topk(cos, k=k)

    topk = topk.cpu().numpy()

    return topk, [cos[i].item() for i in topk]



def get_similar_tokens(query_token, k, embed):

    '''

    @params:

        query_token: 给定的单词

        k: 所需近义词的个数

        embed: 预训练词向量

    '''

    topk, cos = knn(embed.vectors,

                    embed.vectors[embed.stoi[query_token]], k+1)

    for i, c in zip(topk[1:], cos[1:]):  # 除去输入词

        print('cosine sim=%.3f: %s' % (c, (embed.itos[i])))



get_similar_tokens('chip', 3, glove)
get_similar_tokens('baby', 3, glove)
get_similar_tokens('beautiful', 3, glove)
def get_analogy(token_a, token_b, token_c, embed):

    '''

    @params:

        token_a: 词a

        token_b: 词b

        token_c: 词c

        embed: 预训练词向量

    @outputs:

        res: 类比词d

    '''

    vecs = [embed.vectors[embed.stoi[t]] 

                for t in [token_a, token_b, token_c]]

    x = vecs[1] - vecs[0] + vecs[2]

    topk, cos = knn(embed.vectors, x, 1)

    res = embed.itos[topk[0]]

    return res



get_analogy('man', 'woman', 'son', glove)
get_analogy('beijing', 'china', 'tokyo', glove)
get_analogy('bad', 'worst', 'big', glove)
get_analogy('do', 'did', 'go', glove)