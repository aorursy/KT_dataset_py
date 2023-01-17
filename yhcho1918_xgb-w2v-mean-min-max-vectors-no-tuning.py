import pandas as pd
import numpy as np
df_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
df_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
y_train = pd.read_csv('../input/y_train.csv').gender
IDtest = df_test.custid.unique()

df_train.head()
p_level = 'corner_nm'  # 상품 분류 수준

# W2V 학습을 하기에는 데이터(즉 corpus)가 부족하여 
# 고객별로 구매한 상품 목록으로부터 n배 oversampling을 수행
def oversample(x, n):
    if n == 0:
        return list(x)
    uw = np.unique(x)
    bs = np.array([])
    for j in range(n):
        bs = np.append(bs, np.random.choice(uw, len(uw), replace=False))
    return list(bs)

X_train = list(df_train.groupby('custid')[p_level].agg(oversample, 20))
X_test = list(df_test.groupby('custid')[p_level].agg(oversample, 20))
num_features = 100 # 단어 벡터 차원 수
min_word_count = 1 # 최소 단어 수
context = 5 # 학습 윈도우(인접한 단어 리스트) 크기

# 초기화 및 모델 학습
from gensim.models import word2vec

# 모델 학습
model = word2vec.Word2Vec(X_train, 
                          size=num_features, 
                          min_count=min_word_count,
                          window=context,)
# 필요없는 메모리 unload
model.init_sims(replace=True)
#model.wv.most_similar('여자',topn=20)
# 구매상품에 해당하는 벡터의 평균/최소/최대 벡터를 feature로 만드는 전처리기(pipeline에서 사용 가능)
class EmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = num_features
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.hstack([
                np.max([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.min([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)                
            ]) 
            for words in X
        ])  
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

model = Pipeline([
    ("W2V vectorizer", EmbeddingVectorizer(model.wv)),
    ("xgboost", XGBClassifier())])

model.fit(X_train, y_train)
pred = model.predict_proba(X_test)[:,1]
fname = 'submissions.csv'
submissions = pd.concat([pd.Series(IDtest, name="custid"), pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))