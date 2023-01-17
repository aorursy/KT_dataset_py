import numpy as np 

import pandas as pd 

import seaborn as sns 

import matplotlib.pyplot as plt

from subprocess import check_output

from PIL import Image

from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfTransformer

import nltk

import collections

from nltk.corpus import stopwords

import threading

from scipy.cluster import hierarchy

from scipy import ndimage

from subprocess import check_output

#print(check_output(["ls",'../input']).decode("utf8"))

training = pd.DataFrame.from_csv('../input/mercari-price-suggestion-challenge/train.tsv', sep='\t')

test = pd.DataFrame.from_csv('../input/mercari-price-suggestion-challenge/test.tsv', sep='\t')

training.head(5)
print("Shape : "+str(training.shape))
training.info()
training.describe()
# Lets check if shipping has any impact on prices 

fig, ax = plt.subplots(figsize=(11, 7), sharex=True, sharey=True)

sns.distplot(np.log(training.loc[training['shipping']==1]['price'].values+1),  color='blue', label='shipping')

sns.distplot(np.log(training.loc[training['shipping']==0]['price'].values+1),  color='green', label='No shipping')

plt.ylabel("price")

plt.title("price variation with shipping mode")

ax.legend(loc=0)

plt.show()

#Engineered feature

training['has_description'] = [1]*training.shape[0]

training.loc[((training['item_description']=='No description yet') | training['item_description'].isnull()), 'has_description'] = 0
plt.figure(figsize=(20, 15))

bins=50

plt.hist(training[training['has_description']==1]['price'], bins, normed=True,range=[0,250],

         alpha=0.6, label='price when has_description==1')

plt.hist(training[training['has_description']==0]['price'], bins, normed=True,range=[0,250],

         alpha=0.6, label='price when has_description==0')

plt.title('Train price X has_description type distribution', fontsize=15)

plt.xlabel('Price', fontsize=15)

plt.ylabel('Samples', fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.legend(fontsize=15)

plt.show()
fig, ax = plt.subplots(figsize= (20,10))

plt.hist(training['price'], bins=50, range=[0,250], label='price')

plt.title('Train "price" distribution', fontsize=15)

plt.xlabel('Price', fontsize=15)

plt.ylabel('Samples', fontsize=15)

ax.legend(loc='best')

plt.show()
fig, ax = plt.subplots(figsize= (10,10))

plt.title('Relation between declared condition and shipping')

plt.xlabel("item's condition")

plt.ylabel('ln(price)')

ax = sns.violinplot(x="item_condition_id", y=np.log(training["price"].values+1), hue='shipping', data=training , scale="count", palette="muted", split=True, x_jitter = True, y_jitter = True)

plt.show()


mask = np.array(Image.open('../input/machinejpg/machine-learning.jpg'))

stopwords = stopwords.words('english')

cloud = WordCloud(width=1440, height=1080,mask=mask, stopwords=stopwords).generate(" ".join(training['item_description'].astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')

bins=100

plt.figure(figsize=(20, 15))

plt.hist(training["item_description"].apply(lambda x: len(str(x).split())), bins, range=[0,100], label='train')

plt.hist(test["item_description"].apply(lambda x: len(str(x).split())), bins, alpha=0.6,range=[0,100], label='test')

plt.title("Histogram of description's word count", fontsize=15)

plt.ylabel('Number of words', fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.legend(fontsize=15)

plt.show()

plt.figure(figsize=(20, 15))

trainingplt = training.copy(deep=True)

trainingplt["has_Brand"] = ([True]*(training.shape)[0])

trainingplt.loc[trainingplt['brand_name'].isnull() == True, 'has_Brand'] = False

sns.set(style="darkgrid")

sns.countplot(x="has_Brand", data=trainingplt)

plt.title("Brand Repartition ", fontsize=15)

plt.ylabel('Number of records', fontsize=15)

plt.xlabel('is Brand feature filled', fontsize=15)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.legend(fontsize=15)

plt.show()

#Reducing Diff NIKE == nike == Nike 

def treat(elt):

    if(isinstance(elt,str)):

        return elt.lower()

    return str(elt).lower()



training["brand_name"] = training["brand_name"].apply(lambda x: np.NaN if(pd.isnull(x)) else treat(x))

training["category_name"] = training["category_name"].apply(lambda x: np.NaN if(pd.isnull(x)) else treat(x))

training["name"] = training["name"].apply(lambda x: np.NaN if(pd.isnull(x)) else treat(x))

training["item_description"] = training["item_description"].apply(lambda x: np.NaN if(pd.isnull(x)) else treat(x))

training.head(5)

#Defining a function which will split into three features out categories

def transform_category_name(category_name):

    try:

        mainCategory, subCategory, subsubCategory = category_name.split('/')

        return mainCategory, subCategory, subsubCategory

    except:

        return np.nan, np.nan, np.nan
#Creating Three Columns with the corresponding values

training['category_main'], training['category_sub1'], training['category_sub2'] = zip(*training['category_name'].apply(transform_category_name))
training.head()
def plotBarHDisctionnary(liste11,category):

    plt.figure(figsize=(20, 20))

    cpt=0

    liste1=liste11[:30]

    size = len(liste1[:30])

    x = [elt[0] for elt in liste1]

    for [xx,yy] in liste1:

        plt.barh(cpt, yy, align='center', alpha=0.5)

        cpt+=1    

    plt.yticks(range(0,size),x, fontsize=15)

    plt.xticks(fontsize=15)

    plt.title("Top 30 "+category+" categories sorted by its Article's Number ", fontsize=15)

    plt.xlabel('Count of articles', fontsize=15)

    plt.ylabel(category+' categorie', fontsize=15)

    plt.legend(fontsize=15)

    plt.show()

grouped_df = training.astype(str).groupby(["category_main"])

grouped_df = pd.DataFrame(grouped_df.size().reset_index(name = "Group_Count"))

result = grouped_df.sort_values(by='Group_Count', ascending=False,na_position='first')

resultMain = result[["category_main","Group_Count"]].values.tolist()
plotBarHDisctionnary(resultMain,'Main')
grouped_df = training.astype(str).groupby(["category_sub1"])

grouped_df = pd.DataFrame(grouped_df.size().reset_index(name = "Group_Count"))

result = grouped_df.sort_values(by='Group_Count', ascending=False,na_position='first')

resultSub = result[["category_sub1","Group_Count"]].values.tolist()
plotBarHDisctionnary(resultSub,'Sub')
grouped_df = training.astype(str).groupby(["category_sub2"])

grouped_df = pd.DataFrame(grouped_df.size().reset_index(name = "Group_Count"))

result = grouped_df.sort_values(by='Group_Count', ascending=False,na_position='first')

resultSubSub = result[["category_sub2","Group_Count"]].values.tolist()

plotBarHDisctionnary(resultSubSub,'SubSub')
grouped_df = training.astype(str).groupby(["brand_name"])

grouped_df = pd.DataFrame(grouped_df.size().reset_index(name = "Group_Count"))

result = grouped_df.sort_values(by='Group_Count', ascending=False,na_position='first')

result = result[["brand_name","Group_Count"]].values.tolist()
def brandPlot(liste11):

    plt.figure(figsize=(20, 20))

    cpt=0

    liste1=liste11[:30]

    size = len(liste1[:30])

    x = [elt[0] for elt in liste1]

    for [xx,yy] in liste1:

        plt.barh(cpt, yy, align='center', alpha=0.5)

        cpt+=1    

    plt.yticks(range(0,size),x, fontsize=15)

    plt.xticks(fontsize=15)

    plt.title("Top 30 Brands sorted by its Article's Number ", fontsize=15)

    plt.xlabel('Count of articles', fontsize=15)

    plt.ylabel('Brand', fontsize=15)

    plt.legend(fontsize=15)

    plt.show()
brandPlot(result)
#Creating another Field before Drowping Brands Name and NaN Brands

#We suppose that every Record has a Brand's Name

training["has_Brand"] = ([1]*(training.shape)[0]) #Conceptually True

training.loc[training['brand_name'].isnull(), 'has_Brand'] = 0

training["has_Category"] = ([1]*(training.shape)[0]) #Conceptually True

training.loc[training['category_name'].isnull(), 'has_Category'] = 0





for toTreat in ["brand_name","category_main","category_sub1","category_sub2"]:  

    dic = {}

    cpt = 0

    for elt in (training[toTreat].unique()):

        dic[elt] = cpt

        cpt+=1

    training[toTreat] = training[toTreat].apply(lambda x: -1 if(pd.isnull(x)) else dic[x])





training.head()

training = training.drop(['name', 'category_name',"item_description"], axis=1)

from sklearn.model_selection import KFold

import lightgbm as lgbm

def runLGBM(train_X, train_y, test_X, seed_val=42):

    params = {

        'boosting_type': 'gbdt', 'objective': 'regression', 'nthread': -1, 'verbose': 0,

        'num_leaves': 31, 'learning_rate': 0.05, 'max_depth': -1,

        'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.6, 

        'reg_alpha': 1, 'reg_lambda': 0.001, 'metric': 'rmse',

        'min_split_gain': 0.5, 'min_child_weight': 1, 'min_child_samples': 10, 'scale_pos_weight': 1}

    

    pred_test_y = np.zeros(test_X.shape[0])

    

    train_set = lgbm.Dataset(train_X, label = np.asarray(train_y))

        

    model = lgbm.train(params, train_set=train_set, num_boost_round=300)

    pred_test_y = model.predict(test_X, num_iteration = model.best_iteration)

        

    return pred_test_y , model
#Engineered feature

test['has_description'] = [1]*test.shape[0]

test.loc[((test['item_description']=='No description yet') | test['item_description'].isnull()), 'has_description'] = 0

test["brand_name"] = test["brand_name"].apply(lambda x: np.NaN if(pd.isnull(x)) else treat(x))

test["category_name"] = test["category_name"].apply(lambda x: np.NaN if(pd.isnull(x)) else treat(x))

test["name"] = test["name"].apply(lambda x: np.NaN if(pd.isnull(x)) else treat(x))

test["item_description"] = test["item_description"].apply(lambda x: np.NaN if(pd.isnull(x)) else treat(x))

test['category_main'], test['category_sub1'], test['category_sub2'] = zip(*test['category_name'].apply(transform_category_name))

test["has_Brand"] = ([1]*(test.shape)[0]) #Conceptually True

test.loc[test['brand_name'].isnull(), 'has_Brand'] = 0

test["has_Category"] = ([1]*(test.shape)[0]) #Conceptually True

test.loc[test['category_name'].isnull(), 'has_Category'] = 0

for toTreat in ["brand_name","category_main","category_sub1","category_sub2"]:  

    dic = {}

    cpt = 0

    for elt in (test[toTreat].unique()):

        dic[elt] = cpt

        cpt+=1

    test[toTreat] = test[toTreat].apply(lambda x: -1 if(pd.isnull(x)) else dic[x])

test1 = test.drop(['name', 'category_name',"item_description"], axis=1)

id_test = test1.index

usedColumn = test1.columns.values.tolist()

explicativeValues = training[usedColumn]

predictedValue = training[["price"]]

xtrainMatrix = np.asarray(explicativeValues.as_matrix())

ytrainMatrix = np.asarray(predictedValue.as_matrix()).flatten()

xtestMatrix = np.asarray(test1.as_matrix())

predictions, model = runLGBM(xtrainMatrix, ytrainMatrix,xtestMatrix , seed_val=42)

submission = pd.DataFrame(np.column_stack([id_test,predictions]),columns=["test_id","price"])

submission["test_id"] = submission["test_id"].astype(int)

submission["price"] = submission["price"].astype(float).round(3)
submission.to_csv("LGBM.csv",index=False)
print("First Submission, TO IMPROVE")