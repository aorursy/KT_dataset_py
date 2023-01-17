# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# read in the training and testing data into Pandas.DataFrame objects

input_df = pd.read_csv('../input/train.csv', header=0)

submit_df  = pd.read_csv('../input/test.csv',  header=0)
# merge the two DataFrames into one

df = pd.concat([input_df, submit_df])
# re-number the combined data set so there aren't duplicate indexes

df.reset_index(inplace=True)
# reset_index() generates a new column that we don't want, so let's get rid of it

df.drop('index', axis=1, inplace=True)
# the remaining columns need to be reindexed so we can access the first column at '0' instead of '1'

# 就是原本 Columns 的順序在做 reset_index() 的時候被打亂掉了，

# 所以再用原本 DataFrame 的 Columns 的順序，重新排列目前的 DataFrame 的 Columns 順序

## 其實是不太需要，不過作者好像很喜歡用 index 的數字來拿資料，所以才會這樣做吧

## 有空可以測試看看，做跟不做之間的執行時間差距

df = df.reindex_axis(input_df.columns, axis=1)
print(df.shape[1], "columns:", df.columns.values)
print("Row count:", df.shape[0])
# 1.

# Replace missing values with "U0"

# 跟原本教學的程式碼不一樣了，原本的可能是舊版的寫法

df['Cabin'] = df['Cabin'].fillna('U0')
# 3.

# Take the median of all non-null Fares and use that for all missing values

df['Fare'] = df['Fare'].fillna(df['Fare'].median())



# or



# Replace missing values with most common port

# df.Embarked.fillna(df.Embarked.dropna().mode().values)
# 4.

from sklearn.ensemble import RandomForestRegressor

 

### Populate missing ages using RandomForestClassifier

def setMissingAges(df):

    

    # Grab all the features that can be included in a Random Forest Regressor

    ## TODO: Sex 跟 Embarked 的資料如果轉換成數字，可能可以做迴歸？

    age_df = df[['Age','SibSp','Parch','Fare']]

    

    # Split into sets with known and unknown Age values

    ## NOTE: 以下很多程式碼被改過

    ## 基本上我很討厭原作者喜歡用 Index Number 來擷取資料的方法

    ## 程式碼很難懂，而且可能因為 Columns 的順序跑掉而出錯

    knownAge = age_df.loc[ (df.Age.notnull()) ]

    unknownAge = age_df.loc[ (df.Age.isnull()) ]

    unknownAge.drop('Age', axis=1, inplace=True)

    

    # All age values are stored in a target array

    y = knownAge['Age'].values

    

    # All the other values are stored in the feature array

    X = knownAge[['SibSp','Parch','Fare']].values

    

    # Create and fit a model

    rtr = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    rtr.fit(X, y)

    

    # Use the fitted model to predict the missing values

    predictedAges = rtr.predict(unknownAge[['SibSp','Parch','Fare']].values)

    

    # Assign those predictions to the full data set

    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 

    

    return df



df = setMissingAges(df)
# 1.

import pandas as pd



# Create a dataframe of dummy variables for each distinct value of 'Embarked'

dummies_df = pd.get_dummies(df['Embarked'])



# Rename the columns from 'S', 'C', 'Q' to 'Embarked_S', 'Embarked_C', 'Embarked_Q'

dummies_df = dummies_df.rename(columns=lambda x: 'Embarked_' + str(x))



# Add the new variables back to the original data set

df = pd.concat([df, dummies_df], axis=1)



# (or written as a one-liner):

# df = pd.concat([df, pd.get_dummies(df['Embarked']).rename(columns=lambda x: 'Embarked_' + str(x))], axis=1)
# 2.

import re



# Replace missing values with "U0"

# NOTE: 我之前做了

# df['Cabin'] = df['Cabin'].fillna('U0')



# create feature for the alphabetical part of the cabin number

## NOTE: 其實我覺得這做得太複雜了，擷取第一個字就好了

## string = 'HELLO', string[0] = 'H'

df['CabinLetter'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())



# convert the distinct cabin letters with incremental integer values

## NOTE: 這會把 Category 的名字依照索引到的順序，轉換成數字

## 像是這個 Catrgories 的 List: ['A', 'B', 'D', 'C', 'D']

## 就會變成: [0, 1, 2, 3, 2]

df['CabinLetter'] = pd.factorize(df['CabinLetter'])[0]
# 3.

# StandardScaler will subtract the mean from each value then scale to the unit variance

from sklearn import preprocessing



scaler = preprocessing.StandardScaler()

df['Age_scaled'] = scaler.fit_transform(df['Age'].reshape(-1,1))
# 4.

# Divide all fares into quartiles

df['Fare_bin'] = pd.qcut(df['Fare'], 4)



# qcut() creates a new variable that identifies the quartile range, but we can't use the string so either

# factorize or create dummies from the result

## NOTE: 網站上這邊好像有寫錯，忘了在 factorized() 處理完的物件後面加上 "[0]"

df['Fare_bin_id'] = pd.factorize(df['Fare_bin'])[0]



df = pd.concat([df, pd.get_dummies(df['Fare_bin']).rename(columns=lambda x: 'Fare_' + str(x))], axis=1)
# how many different names do they have? 

df['Names'] = df['Name'].map(lambda x: len(re.split(' ', x)))
# What is each person's title? 

df['Title'] = df['Name'].map(lambda x: re.compile(", (.*?)\.").findall(x)[0])



# Group low-occuring, related titles together

## TODO:

## 這裡會出現警告，盡量改用 .loc() 來寫入值（不過我其實不是很會......再查一下 .loc() 的基本用法吧）

## Ref: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy

df['Title'][df.Title == 'Jonkheer'] = 'Master'

df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'

df['Title'][df.Title == 'Mme'] = 'Mrs'

df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'

df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'



# Build binary features

df = pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)
# Replace missing values with "U0"

## NOTE: 我之前做了

# df['Cabin'][df.Cabin.isnull()] = 'U0'



# Create a feature for the deck

df['Deck'] = df['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())

df['Deck'] = pd.factorize(df['Deck'])[0]



# Create binary features for each deck

decks = pd.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_' + str(x))

df = pd.concat([df, decks], axis=1)



# Create feature for the room number

## NOTE: 有 6 筆資料的 Cabin 欄位在正規表示式分析出來以後，因為沒有帶數字（本身資料就只有一個英文字沒數字），

## 在做 .group() 的時候會出錯（值為 NoneType)

## Test Code 1: df['Cabin'].map( lambda x : re.compile("([0-9]+)").search(x)).isnull().sum()

## Test Code 2: df[df['Cabin'].map( lambda x : re.compile("([0-9]+)").search(x)).isnull() == True]['Cabin']

# df['Room'] = df['Cabin'].map( lambda x : re.compile("([0-9]+)").search(x).group()).astype(int) + 1
def processTicket(df):    

    # extract and massage the ticket prefix

    ## NOTE: 稍微修改了一下，主要把 global 換成 return 的寫法

    df['TicketPrefix'] = df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))

    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('[\.?\/?]', '', x) )

    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )

        

    # create binary features for each prefix

    prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))

    df = pd.concat([df, prefixes], axis=1)

    

    # factorize the prefix to create a numerical categorical variable

    df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]

    

    # extract the ticket number

    df['TicketNumber'] = df['Ticket'].map( lambda x: getTicketNumber(x) )

    

    # create a feature for the number of digits in the ticket number

    df['TicketNumberDigits'] = df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)

    

    # create a feature for the starting number of the ticket number

    df['TicketNumberStart'] = df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)

    

    # The prefix and (probably) number themselves aren't useful

    df.drop(['TicketPrefix', 'TicketNumber'], axis=1, inplace=True)

    

    return df



def getTicketPrefix(ticket):

    match = re.compile("([a-zA-Z\.\/]+)").search(ticket)

    if match:

        return match.group()

    else:

        return 'U'



def getTicketNumber(ticket):

    match = re.compile("([\d]+$)").search(ticket)

    if match:

        return match.group()

    else:

        return '0'



df = processTicket(df)
## NOTE: 這應該是想要去看兩兩資料在經過加減乘除以後，有沒有相關性，

## 不過實在是沒什麼必要。一來是這應該透過回歸分析去做，

## 再來就是，很多在這邊列舉的資料，之前的步驟如果不是沒有處理，

## 就是其實本身為類別資料，只是被轉換成數值了，

## 所以整個跳過，用接下來的方法就好。



# numerics = df.loc[:, ['Age_scaled', 'Fare_scaled', 'Pclass_scaled', 'Parch_scaled', 'SibSp_scaled', 

#                       'Names_scaled', 'CabinNumber_scaled', 'Age_bin_id_scaled', 'Fare_bin_id_scaled']]



# # for each pair of variables, determine which mathmatical operators to use based on redundancy

# for i in range(0, numerics.columns.size-1):

#     for j in range(0, numerics.columns.size-1):

#         col1 = str(numerics.columns.values[i])

#         col2 = str(numerics.columns.values[j])

#         # multiply fields together (we allow values to be squared)

#         if i <= j:

#             name = col1 + "*" + col2

#             df = pd.concat([df, pd.Series(numerics.iloc[:,i] * numerics.iloc[:,j], name=name)], axis=1)

#         # add fields together

#         if i < j:

#             name = col1 + "+" + col2

#             df = pd.concat([df, pd.Series(numerics.iloc[:,i] + numerics.iloc[:,j], name=name)], axis=1)

#         # divide and subtract fields from each other

#         if not i == j:

#             name = col1 + "/" + col2

#             df = pd.concat([df, pd.Series(numerics.iloc[:,i] / numerics.iloc[:,j], name=name)], axis=1)

#             name = col1 + "-" + col2

#             df = pd.concat([df, pd.Series(numerics.iloc[:,i] - numerics.iloc[:,j], name=name)], axis=1)
# calculate the correlation matrix (ignore survived and passenger id fields)

## NOTE: 計算每個數值資料之間的迴歸性質

df_corr = df.drop(['Survived', 'PassengerId'],axis=1).corr(method='spearman')



# create a mask to ignore self-

## NOTE: 因為同樣的數值之間的迴歸值必定為 1.0（自己跟自己的值是一樣的 R，當然是絕對相關）

## 所以用這兩行程式碼，來把這些值的回歸數值設定為 0，

## 才不會在下一步驟誤判為高度迴歸而把資料捨去

mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)

df_corr = mask * df_corr



drops = []

# loop through each variable

for col in df_corr.columns.values:

    # if we've already determined to drop the current variable, continue

    if np.in1d([col],drops):

        continue

    

    # find all the variables that are highly correlated with the current variable 

    # and add them to the drop list 

    ## NOTE: 看看兩兩欄位之間，有沒有高度相關的（corr > 0.98）

    ## 有的話就捨棄掉

    corr = df_corr[abs(df_corr[col]) > 0.98].index

    drops = np.union1d(drops, corr)



print("\nDropping", drops.shape[0], "highly correlated features...\n", drops)

## NOTE: 最後把剛剛產生出來的資料中，有高度相關的資料丟掉

## WHY? 應該是因為如果有高度相關，就不用再分析一次資料，還放到原本的 DataFrame 裡了

df.drop(drops, axis=1, inplace=True)