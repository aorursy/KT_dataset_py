from IPython.core.display import display, HTML

display(HTML("<style>.container { width:90% !important; }</style>"))
%matplotlib inline



import pandas as pd

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns



sns.set() # seaborn 속성을 기본값으로 설정
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



#from enum import Enum

class Columns:

    # 원래 존재하는 항목

    PassengerId = "PassengerId"

    Survived = "Survived"

    Pclass = "Pclass"

    Name = "Name"

    Sex = "Sex"

    Age = "Age"

    SibSp = "SibSp"

    Parch = "Parch"

    Ticket = "Ticket"

    Fare = "Fare"

    Cabin = "Cabin"

    Embarked = "Embarked"

    

    # 새로 생성하는 항목

    Title = "Title"

    FareBand = "FareBand"

    Family = "Family"

    Deck = "Deck" # Cabin의 알파벳을 떼서 Deck을 지정한다.

    CabinExists = "CabinExists"
train.head()
test.head()
print(train[[Columns.Pclass, Columns.Survived]].head())

train[[Columns.Pclass, Columns.Survived]].groupby([Columns.Pclass]).mean().plot.bar()
train[[Columns.Sex, Columns.Survived]].groupby([Columns.Sex]).mean().plot.bar()
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))

sns.countplot(x=Columns.Sex, hue=Columns.Survived, data=train, ax=ax[0])

sns.countplot(x=Columns.Sex, hue=Columns.Pclass, data=train, ax=ax[1])

sns.countplot(x=Columns.Pclass, hue=Columns.Survived, data=train, ax=ax[2])
train[Columns.Age].plot.kde() # 가우시안 커널 함수를 사용해서 KDE plot을 그림
df = train[train[Columns.Age].isnull() == False] # Age가 없는 데이터는 뺀다.

#df.describe()



bincount = 12

# 나이대로 나누어서 출력해 본다.

age_min = df[Columns.Age].min().astype('int')

age_max = df[Columns.Age].max().astype('int')

print("Age :", age_min, " ~ ", age_max)

gap = ((age_max - age_min) / bincount).astype(int)

print('gap:', gap)



bins = [-1]

for i in range(bincount):

    bins.append(i * gap)

bins.append(np.inf)

print(bins)



_df = df

_df['AgeGroup'] = pd.cut(_df[Columns.Age], bins) #bins로 구분된 'AgeGroup' category column을 생성한다.

fig, ax = plt.subplots(figsize=(20, 10))

sns.countplot(x='AgeGroup', hue=Columns.Survived, data=_df, ax=ax) #'AgeGroup'을 기준으로 생존/사망 수를 표시한다.
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(16, 20))

sns.violinplot(x=Columns.Pclass, y=Columns.Age, hue=Columns.Survived, data=train, scale='count', split=True, ax=ax[0])

sns.violinplot(x=Columns.Sex, y=Columns.Age, hue=Columns.Survived, data=train, scale='count', split=True, ax=ax[1])

sns.violinplot(x=Columns.Pclass, y=Columns.Sex, hue=Columns.Survived, data=train, scale='count', split=True, ax=ax[2])

_train = train

_train['Family'] = _train[Columns.SibSp] + _train[Columns.Parch] + 1 #형제 + 직계 + 자기자신(1)

_train[['Family', Columns.Survived]].groupby('Family').mean().plot.bar()
train[Columns.Age].plot.hist() # 승객들의 나이 분포
sns.countplot(x='Family', data=_train)
sns.countplot(x='Family', hue=Columns.Survived, data=_train)
train_len = train.shape[0]

'''

ignore_index=True로 하지 않으면 test의 인덱스가 0부터 시작해서 iterrow()등의 반복자를 

사용할 때 오동작한다.(0이 두개가 되는 등)

'''



merged = train.append(test, ignore_index=True) 

print("train len : ", train.shape[0])

print("test len : ", test.shape[0])

print("merged len : ", merged.shape[0])
merged[Columns.Family] = merged[Columns.Parch] + merged[Columns.SibSp] + 1

if Columns.Parch in merged:    

    merged = merged.drop([Columns.Parch], axis=1)

if Columns.SibSp in merged:

    merged = merged.drop([Columns.SibSp], axis=1)

    

merged.head()
most_embarked_label = merged[Columns.Embarked].value_counts().index[0]



merged = merged.fillna({Columns.Embarked : most_embarked_label})

merged.describe(include="all")
# Name에서 Title 추출(그냥 알파벳 끝에 .이 붙어 있는걸 추출한다.)

merged[Columns.Title] = merged.Name.str.extract('([A-Za-z]+)\. ', expand=False) # expand:True면 DataFrame을, False면 Series를 리턴한다.



print("initial titles : ", merged[Columns.Title].value_counts().index)

#initial titles :  Index(['Mr', 'Miss', 'Mrs', 'Master', 'Dr', 'Rev', 'Col', 'Ms', 'Mlle', 'Major',

#                         'Sir', 'Jonkheer', 'Don', 'Mme', 'Countess', 'Lady', 'Dona', 'Capt'],



# 정리(희귀한 title을 모아서 정리한다.)

merged[Columns.Title] = merged[Columns.Title].replace(['Lady', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

merged[Columns.Title] = merged[Columns.Title].replace(['Countess', 'Lady', 'Sir'], 'Royal')

merged[Columns.Title] = merged[Columns.Title].replace(['Miss', 'Mlle', 'Ms', 'Mme'], 'Mrs')



print("Survival rate by title:")

print("========================")

print(merged[[Columns.Title, Columns.Survived]].groupby(Columns.Title).mean())



idxs = merged[Columns.Title].value_counts().index # 많은 순서대로 정렬해서 오름차순으로 값을 매김

print(idxs)



# 숫자값으로 변경한다.

mapping = {}

for i in range(len(idxs)):

    mapping[idxs[i]] = i + 1

print("Title mapping : ", mapping)

merged[Columns.Title] = merged[Columns.Title].map(mapping)



if Columns.Name in merged:

    merged = merged.drop([Columns.Name], axis=1)

    

merged.head()
sns.countplot(x=Columns.Title, hue=Columns.Survived, data=merged)

print(merged[Columns.Title].value_counts())
mapping = {'male':0, 'female':1}

merged[Columns.Sex] = merged[Columns.Sex].map(mapping)
merged.head(n=10)
# {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5, 'Royal': 6}



mapping = {1:21, 2:28, 3:28, 4:40, 5:50, 6:60}

def guess_age(row):

    return mapping[row[Columns.Title]]



def fixup_age(df):

    for idx, row in df[df[Columns.Age].isnull() == True].iterrows():

        df.loc[idx, Columns.Age] = guess_age(row)

    return df

    

merged = fixup_age(merged)

merged.describe(include='all')
def make_deck(df):

    '''

    Cabin에서 알파벳을 떼서 Deck 알파벳을 생성한다.

    '''

    df[Columns.Deck] = df[Columns.Cabin].str.extract('([A-Za-z]+)', expand=True)

    return df



merged = make_deck(merged)

merged.describe(include='all')
merged[[Columns.Deck, Columns.Fare]].groupby(Columns.Deck).mean().sort_values(by=Columns.Fare)
sns.countplot(x=Columns.Deck, hue=Columns.Survived, data=merged)
print("total survived rate: ", merged[Columns.Survived].mean())

print("deck survived rate: ", merged[merged[Columns.Deck].isnull() == False][Columns.Survived].mean())

print("no deck survived rate: ", merged[merged[Columns.Deck].isnull()][Columns.Survived].mean())



fig, ax = plt.subplots(2, 1, figsize=(16, 16))

merged[[Columns.Deck, Columns.Survived]].groupby(Columns.Deck).mean().plot.bar(ax=ax[0])



def generate_fare_group(df, slicenum):

    if "FareGroup" in df:

        df.drop("FareGroup", axis=1)

    # 나이대로 나누어서 출력해 본다.

    _min = df[Columns.Fare].min().astype('int')

    _max = df[Columns.Fare].max().astype('int')

    print("Fare :", _min, " ~ ", _max)

    gap = ((_max - _min) / slicenum).astype(int)

    print('gap:', gap)



    bins = [-1]

    for i in range(slicenum):

        bins.append(i * gap)

    bins.append(np.inf)

    print(bins)

    df['FareGroup'] = pd.cut(df[Columns.Fare], bins)    

    return df



df = generate_fare_group(merged.copy(), 16)



# deck 정보가 없는 사람들의 요금에 따른 생존자/사망자 수

sns.countplot(x="FareGroup", hue=Columns.Survived, data=df[df[Columns.Deck].isnull()], ax=ax[1])

merged[Columns.CabinExists] = (merged[Columns.Cabin].isnull() == False)

merged[Columns.CabinExists] = merged[Columns.CabinExists].map({True:1, False:0})
merged.head()
merged[merged[Columns.Fare].isnull()]
merged.loc[merged[Columns.Fare].isnull(), [Columns.Fare]] = merged[Columns.Fare].mean()
merged.head()
sns.distplot(merged[Columns.Fare])
'''

log를 취하는 방법

'''



#merged[Columns.Fare] = merged[Columns.Fare].map(lambda i : np.log(i) if i > 0 else 0)



'''

등급을 4단계로 나누는 방법

'''

merged[Columns.FareBand] = pd.qcut(merged[Columns.Fare], 4, labels=[1,2,3,4]).astype('float')

#merged[Columns.Fare] = merged[Columns.FareBand]



merged.head(n=20)

merged[Columns.Fare] = merged[Columns.FareBand]

merged = merged.drop([Columns.FareBand], axis=1)

merged.head()
merged.head()
sns.distplot(merged[Columns.Fare])
merged.head()
if Columns.Ticket in merged:

    merged = merged.drop(labels=[Columns.Ticket], axis=1)

if Columns.Cabin in merged:

    merged = merged.drop(labels=[Columns.Cabin], axis=1)

if Columns.Deck in merged:

    merged = merged.drop(labels=[Columns.Deck], axis=1)



# passengerId는 나중에 삭제한다.

# if Columns.PassengerId in merged:

#     merged = merged.drop(labels=[Columns.PassengerId], axis=1)
merged.describe(include='all')
merged.head()
merged = pd.get_dummies(merged, columns=[Columns.Pclass], prefix='Pclass')

merged = pd.get_dummies(merged, columns=[Columns.Title], prefix='Title')

merged = pd.get_dummies(merged, columns=[Columns.Embarked], prefix='Embarked')

merged = pd.get_dummies(merged, columns=[Columns.Sex], prefix='Sex')

merged.head()
from sklearn.preprocessing import MinMaxScaler



class NoColumnError(Exception):

    """Raised when no column in dataframe"""

    def __init__(self, value):

        self.value = value

    # __str__ is to print() the value

    def __str__(self):

        return(repr(self.value))



# normalize AgeGroup

def normalize_column(data, columnName):

    scaler = MinMaxScaler(feature_range=(0, 1))    

    if columnName in data:

        aaa = scaler.fit_transform(data[columnName].values.reshape(-1, 1)) # 입력을 2D 데이터로 넣어야 하므로 reshape해 준다.

        aaa = aaa.reshape(-1,) # 다시 원복해서 넣어주지만, 그냥 넣어도 알아서 제대로 들어간다...

        #print(aaa.shape)

        data[columnName] = aaa

        return data

    else:

        raise NoColumnError(str(columnName) + " is not exists!")



def normalize(dataset, columns):

    for col in columns:

        dataset = normalize_column(dataset, col)

    return dataset
merged.head()
merged = normalize(merged, [Columns.Age, Columns.Fare, Columns.Family])
merged.head(n=10)
train = merged[:train_len]

test = merged[train_len:]

test = test.drop([Columns.Survived], axis=1)



train = train.drop([Columns.PassengerId], axis=1)



test_passenger_id = test[Columns.PassengerId]

test = test.drop([Columns.PassengerId], axis=1)



print(train.shape)

print(test.shape)
train_X = train.drop([Columns.Survived], axis=1).values #Series.values는 numpy array 타입의 데이터임

train_Y = train[Columns.Survived].values.reshape(-1, 1)

print(train_X.shape)

print(train_Y.shape)
test.shape
test.describe(include='all')
train.head()
test.head()
import keras

from keras import layers, models



from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold



train_X = train_X.astype(np.float32)

train_Y = train_Y.astype(np.float32)



class SimpleNN(models.Sequential):

    def __init__(self, input_shape, dropout):

        super().__init__()

        

        self.add(layers.Dense(units=20, activation='relu', input_shape=input_shape))

        self.add(layers.Dropout(dropout))

        

        self.add(layers.Dense(units=8, activation='relu'))

        self.add(layers.Dropout(dropout))

        

        self.add(layers.Dense(units=1, activation='sigmoid'))

        self.add(layers.Dropout(dropout))

        

        self.compile(loss=keras.losses.binary_crossentropy, optimizer='adam', metrics=['accuracy'])



batch_size = 32

epochs = 30

dropout = 0.0

n_splits = 10



kfold = KFold(n_splits=n_splits, shuffle=True, random_state=7)



cvscores = []

models = []

acc_hist = []

for _train, _test in kfold.split(train_X, train_Y):

    model = SimpleNN(input_shape=(train_X.shape[1],), dropout=dropout)

    history = model.fit(train_X[_train], train_Y[_train], epochs=epochs, batch_size=batch_size, verbose=0)

    #print("history=", history.history['acc'])

    acc_hist.append(history.history['acc'])

    #evaluate model:

    scores = model.evaluate(train_X[_test], train_Y[_test], verbose=0)    

    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

    cvscores.append(scores[1] * 100)

    models.append(model)



    

fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(n_splits, 8 * n_splits))

for i in range(len(acc_hist)):

    title = "model " + str(i)    

    ax[i].plot(acc_hist[i])

    ax[i].set_title(title)

plt.show()



print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

maxidx = np.argmax(cvscores)

print("Max : [", maxidx, "] : ", "{0:.2f}".format(cvscores[maxidx]))



# 가장 점수가 높은 모델로 test 데이터를 돌린다.

pred = models[maxidx].predict(test, batch_size=test.shape[0], verbose=0)



from sklearn.preprocessing import Binarizer

binarizer=Binarizer(0.5)



test_predict_result=binarizer.fit_transform(pred)

test_predict_result=test_predict_result.astype(np.int32)

#print(test_predict_result[:10])

submission = pd.DataFrame({"PassengerId" : test_passenger_id, "Survived":test_predict_result.reshape(-1)})

submission.to_csv('submission.csv', index=False)
