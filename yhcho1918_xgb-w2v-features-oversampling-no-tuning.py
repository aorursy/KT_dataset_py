import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
df_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
df_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
y_train = pd.read_csv('../input/y_train.csv').gender
IDtest = df_test.custid.unique()

# 상품코드 타입을 문자열로 변환
#df_train.goodcd = df_train.goodcd.apply(lambda x: str(x)) 
#df_test.goodcd = df_test.goodcd.apply(lambda x: str(x)) 

df_train.head()
level = 'corner_nm' # 상품 분류 수준

# W2V 학습을 하기에는 데이터(즉 corpus)가 부족하여 
# 고객별로 구매한 상품 목록으로부터 20배 oversampling을 수행
sentences = []
df_all = pd.concat([df_train, df_test])
for id in df_all.custid.unique():
#    uw = df_all.query('custid == @id')[level].unique()
#    bs = np.array([])
#    for j in range(20):
#        bs = np.append(bs, np.random.choice(uw, len(uw), replace=False))
#    sentences.append(list(bs))
    sentences.append(list(df_all.query('custid == @id')[level].values))
max_features = 300 # 문자 벡터 차원 수
min_word_count = 1 # 최소 문자 수
num_workers = 4 # 병렬 처리 스레드 수
context = 3 # 문자열 창 크기
downsampling = 1e-3 # 문자 빈도수 Downsample

from gensim.models import word2vec

# 모델 학습
model = word2vec.Word2Vec(sentences, 
                          workers=num_workers, 
                          size=max_features, 
                          min_count=min_word_count,
                          window=context,
                          sample=downsampling)
# 필요없는 메모리 unload
model.init_sims(replace=True)
def gender_vec():
    truth = pd.read_csv('../input/y_train.csv')

    sentences = []
    df_all = df_train
    for id in df_all.custid.unique():
        x = df_all.query('custid == @id')[level].unique()
        y = np.where(truth.query('custid == @id').gender == 0, "여자", "남자")
        for j in range(20):
            y = np.append(y, np.random.choice(x, len(x), replace=False))
        sentences.append(list(y))
#model.wv.most_similar('여자',topn=20)
# 고객별로 구매한 상품의 평균벡터를 feature로 사용한다.
features = []
for id in df_train.custid.unique():
    features.append(df_all.query('custid == @id')[level] \
                              .apply(lambda x: model.wv[x]).mean())
X_train = np.array(features)

features = []
for id in df_test.custid.unique():
    features.append(df_all.query('custid == @id')[level] \
                              .apply(lambda x: model.wv[x]).mean())
X_test = np.array(features)
from xgboost import XGBClassifier

model = XGBClassifier().fit(X_train, y_train)
pred = model.predict_proba(X_test)[:,1]
fname = 'submissions.csv'
submissions = pd.concat([pd.Series(IDtest, name="custid"), pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))