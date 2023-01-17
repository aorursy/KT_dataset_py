# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import warnings

warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score,train_test_split

import nltk

#nltk.download('stopwords')

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer,SnowballStemmer

from bs4 import BeautifulSoup

from tqdm import tqdm

import re

import os

import sqlite3

import time

from sklearn.manifold import TSNE

import matplotlib.patches as mpatches

from sklearn.model_selection import cross_val_score,train_test_split

from sklearn.neighbors import KNeighborsClassifier

import datetime

from sklearn.metrics import confusion_matrix,f1_score,precision_recall_curve,precision_score,recall_score,roc_auc_score,roc_curve,accuracy_score,classification_report



import scikitplot.metrics as skplt

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def LoadData(filename,FullDataLoad=False):

    path=os.path.join('/kaggle/input','amazon-fine-food-reviews',filename)

    Connection=sqlite3.connect(path)

    if not FullDataLoad:

        return pd.read_sql_query("""SELECT * FROM Reviews WHERE Score !=3 LIMIT 5000""",Connection),Connection

    return pd.read_sql_query("""SELECT * FROM Reviews WHERE Score !=3""",Connection),Connection    
Dataset_Name='database.sqlite'

Amazon_Food_Reviews,Table_Connection = LoadData(Dataset_Name,FullDataLoad=False)

Amazon_Food_Reviews['Score']= Amazon_Food_Reviews['Score'].apply(lambda x: 1 if x>3 else 0)
Amazon_Food_Reviews.head()
Amazon_Food_Reviews_Sorted_Data= Amazon_Food_Reviews.sort_values('ProductId',axis=0, inplace=False, kind='quicksort', na_position='last')

Amazon_Food_Reviews_Duplicates_Dropped= Amazon_Food_Reviews_Sorted_Data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep='first',inplace=False)

print("{:07.3f} % of data removed during duplicates removal operation".format((1-(Amazon_Food_Reviews_Duplicates_Dropped.shape[0])/(Amazon_Food_Reviews_Sorted_Data.shape[0]))*100))
Amazon_Food_Reviews_Preprocessed=Amazon_Food_Reviews_Duplicates_Dropped[Amazon_Food_Reviews_Duplicates_Dropped.HelpfulnessNumerator<=Amazon_Food_Reviews_Duplicates_Dropped.HelpfulnessDenominator]

print("{:07.3f} % of data removed".format(np.round((1-(Amazon_Food_Reviews_Preprocessed.shape[0])/(Amazon_Food_Reviews_Duplicates_Dropped.shape[0]))*100)))
def DeContraction(document):

    document= re.sub(r"won\'t","will not",document)

    document= re.sub(r"can\'t","can not",document)

    document= re.sub(r"\'t","not",document)

    document= re.sub(r"\'m","am",document)

    document= re.sub(r"\'re","are",document)

    document= re.sub(r"\'ve","have",document)

    document= re.sub(r"\'d","would",document)

    document= re.sub(r"\'s","is",document)

    document= re.sub(r"\'ll","will",document)

    return document
Custom_StopWords = [word for word in stopwords.words('english') if word not in ['no','nor','not']]
from tqdm import tqdm

def Preprocessing(Text_Data):

    Preprocessed_Text =[]

    for document in tqdm(Text_Data):

        document= re.sub(r"http\S+","",document)

        document = BeautifulSoup(document,'lxml').get_text()

        document=DeContraction(document)

        document = re.sub(r"\S*\d\S*","",document).strip()

        document = re.sub("[^a-zA-Z]+",' ',document)

        document = ' '.join(word.lower() for word in document.split() if word.lower() not in Custom_StopWords)

        Preprocessed_Text.append(document)

    return Preprocessed_Text
Preprocessed_Amazon_Reviews = Preprocessing(Amazon_Food_Reviews_Preprocessed['Text'].values)

np.random.seed(42)

print("{} :{}".format(np.random.randint(Amazon_Food_Reviews_Preprocessed.shape[0]),Preprocessed_Amazon_Reviews[np.random.randint(Amazon_Food_Reviews_Preprocessed.shape[0])]))
from sklearn.feature_extraction.text import CountVectorizer

cnt_vectorizer= CountVectorizer()

Count_Vectorized_Reviews=cnt_vectorizer.fit_transform(Preprocessed_Amazon_Reviews)

print("Type of Output:{},Size:{}".format(type(Count_Vectorized_Reviews),Count_Vectorized_Reviews.get_shape()))
#Count Vector With Bi-Grams,Words considering whose frequency greater than 10

Cnt_Vectorizer_Bigrams=CountVectorizer(ngram_range=(1,2),min_df=10)

Count_Vectorized_Reviews_Bigram = Cnt_Vectorizer_Bigrams.fit_transform(Preprocessed_Amazon_Reviews)

print("Type of Output:{},Size:{}".format(type(Count_Vectorized_Reviews_Bigram),Count_Vectorized_Reviews_Bigram.get_shape()))
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer_bigram = TfidfVectorizer(ngram_range=(1,2),min_df=10)

TFIDF_Vectorized_Reviews_Bigram=tfidf_vectorizer_bigram.fit_transform(Preprocessed_Amazon_Reviews)

print("Type of Output:{},Size:{}".format(type(TFIDF_Vectorized_Reviews_Bigram),TFIDF_Vectorized_Reviews_Bigram.get_shape()))
List_Of_document=[]

for document in Preprocessed_Amazon_Reviews:

    List_Of_document.append(document.split())
from gensim.models import Word2Vec

def Train_Word2Vec(List_Of_document,want_to_train_own_word2vec_model=True):

    if want_to_train_own_word2vec_model:

        return Word2Vec(List_Of_document,min_count=5,size=128,workers=os.cpu_count()-2)
Word2Vec_Model = Train_Word2Vec(List_Of_document)

w2v_words=list(Word2Vec_Model.wv.vocab)
def Weighted_Word2Vec(Word2Vec_Model,List_Of_document):

    Documents_Vector=[]

    for document in tqdm(List_Of_document):

        Document_Vector=np.zeros(128)

        count_words=0

        for word in document:

            if word in w2v_words:

                word_vector=Word2Vec_Model.wv[word]

                Document_Vector+=word_vector

                count_words+=1

        if count_words!=0:

            Document_Vector/=count_words

        Documents_Vector.append(Document_Vector)

    return Documents_Vector
Weighted_Word2Vec_Vectorization=Weighted_Word2Vec(Word2Vec_Model,List_Of_document)
tfidf_model =TfidfVectorizer()

tfidf_model.fit(Preprocessed_Amazon_Reviews)

IDF_Dictionary= dict(zip(tfidf_model.get_feature_names(),tfidf_model.idf_))
def TFIDF_WORD2VEC(tfidf_model,IDF_Dictionary,List_Of_document):

    TFIDF_FEATURES = tfidf_model.get_feature_names()

    TFIDF_Documents_Vector=[]

    row=0

    for Document in tqdm(List_Of_document):

        Document_Vector=np.zeros(128)

        weighted_sum=0

        for word in Document:

            if word in w2v_words and word in TFIDF_FEATURES:

                word_vector=Word2Vec_Model.wv[word]

                word_tfidf= IDF_Dictionary[word]*(Document.count(word)/len(Document))

                Document_Vector+=word_vector*word_tfidf

                weighted_sum+=word_tfidf

        if weighted_sum!=0:

            Document_Vector/=weighted_sum

        TFIDF_Documents_Vector.append(Document_Vector)

        row+=1

    return TFIDF_Documents_Vector
TFIDF_Word2Vec_Vectorization=TFIDF_WORD2VEC(tfidf_model,IDF_Dictionary,List_Of_document)
import time

os.mkdir('/kaggle/working/Visualization/')

def Visualization(Dataset,Score,Title,no_of_dims=2,perplexity=50,learning_rate=200):

    plt.figure(figsize=(12,10))

    TSNE_Model = TSNE(n_components=no_of_dims,perplexity=perplexity,learning_rate=learning_rate,n_jobs=-1,n_iter=5000)

    start_time=time.time()

    X_TSNE=TSNE_Model.fit_transform(Dataset)

    stop_time=time.time()

    Data = np.hstack((X_TSNE,Score.reshape(-1,1)))

    TSNE_DF=pd.DataFrame(Data,columns=['Dimension_x','Dimension_y','Score'])

    colors = {0:'red', 1:'blue'}

    plt.scatter(TSNE_DF['Dimension_x'], TSNE_DF['Dimension_y'], c=TSNE_DF['Score'].apply(lambda x: colors[x]))

    plt.title(Title)

    green_patch = mpatches.Patch(color='green', label='perplexity :'+str(perplexity))

    black_patch = mpatches.Patch(color='black', label='learning rate :'+str(learning_rate))

    plt.xlabel('Dimension_x')

    plt.ylabel('Dimension_y')

    plt.legend(handles=[green_patch, black_patch])

    Image = os.path.join('/kaggle/working/Visualization/',Title+'n_components'+'_'+str(no_of_dims)+'_'+'perplexity'+'_'+str(perplexity)+'_'+'learning_rate'+'_'+str(learning_rate)+'.png')

    plt.savefig(fname =Image,dpi=500,format='png')

    Total_Time=(stop_time-start_time)/60

    print('Execution Time:{}'.format(Total_Time))
Visualization(Dataset=Count_Vectorized_Reviews.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='Count Vectoization TSNE',no_of_dims=2,perplexity=10,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='Count Vectoization TSNE',no_of_dims=2,perplexity=30,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='Count Vectoization TSNE',no_of_dims=2,perplexity=20,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='Count Vectoization TSNE',no_of_dims=2,perplexity=40,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='Count Vectoization TSNE')
Visualization(Dataset=Count_Vectorized_Reviews.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='Count Vectoization TSNE',no_of_dims=2,perplexity=100,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM Count Vectoization TSNE',no_of_dims=2,perplexity=10,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM Count Vectoization TSNE',no_of_dims=2,perplexity=20,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM Count Vectoization TSNE',no_of_dims=2,perplexity=30,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM Count Vectoization TSNE',no_of_dims=2,perplexity=40,learning_rate=200)
Visualization(Dataset=Count_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM Count Vectoization TSNE')
Visualization(Dataset=Count_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM Count Vectoization TSNE',no_of_dims=2,perplexity=100,learning_rate=200)
Visualization(Dataset=TFIDF_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM TFIDF Vectoization TSNE',no_of_dims=2,perplexity=10,learning_rate=200)
Visualization(Dataset=TFIDF_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM TFIDF Vectoization TSNE',no_of_dims=2,perplexity=20,learning_rate=200)
Visualization(Dataset=TFIDF_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM TFIDF Vectoization TSNE',no_of_dims=2,perplexity=30,learning_rate=200)
Visualization(Dataset=TFIDF_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM TFIDF Vectoization TSNE',no_of_dims=2,perplexity=40,learning_rate=200)
Visualization(Dataset=TFIDF_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM TFIDF Vectoization TSNE')
Visualization(Dataset=TFIDF_Vectorized_Reviews_Bigram.toarray(),Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='BiGRAM TFIDF Vectoization TSNE',no_of_dims=2,perplexity=100,learning_rate=200)
Visualization(Dataset=Weighted_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='AVG WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=10,learning_rate=200)
Visualization(Dataset=Weighted_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='AVG WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=20,learning_rate=200)
Visualization(Dataset=Weighted_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='AVG WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=30,learning_rate=200)
Visualization(Dataset=Weighted_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='AVG WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=40,learning_rate=200)
Visualization(Dataset=Weighted_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='AVG WORD2VEC Vectoization TSNE')
Visualization(Dataset=Weighted_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='AVG WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=100,learning_rate=200)
Visualization(Dataset=TFIDF_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='TFIDF WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=10,learning_rate=200)
Visualization(Dataset=TFIDF_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='TFIDF WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=20,learning_rate=200)
Visualization(Dataset=TFIDF_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='TFIDF WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=30,learning_rate=200)
Visualization(Dataset=TFIDF_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='TFIDF WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=40,learning_rate=200)
Visualization(Dataset=TFIDF_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='TFIDF WORD2VEC Vectoization TSNE')
Visualization(Dataset=TFIDF_Word2Vec_Vectorization,Score=Amazon_Food_Reviews_Preprocessed['Score'].values,Title='TFIDF WORD2VEC Vectoization TSNE',no_of_dims=2,perplexity=100,learning_rate=200)
def Cleaning_Pipeline(filename,FullDataLoad=False):

    Amazon_Food_Reviews,Table_Connection = LoadData(Dataset_Name,FullDataLoad=FullDataLoad)

    Amazon_Food_Reviews['Score']= Amazon_Food_Reviews['Score'].apply(lambda x: 1 if x>3 else 0)

    print(Amazon_Food_Reviews.head())

    

    Amazon_Food_Reviews_Sorted_Data= Amazon_Food_Reviews.sort_values('ProductId',axis=0, inplace=False, kind='quicksort', na_position='last')

    Amazon_Food_Reviews_Duplicates_Dropped= Amazon_Food_Reviews_Sorted_Data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"},keep='first',inplace=False)

    print("{:07.3f} % of data removed during duplicates removal operation".format((1-(Amazon_Food_Reviews_Duplicates_Dropped.shape[0])/(Amazon_Food_Reviews_Sorted_Data.shape[0]))*100))

    Amazon_Food_Reviews_Preprocessed=Amazon_Food_Reviews_Duplicates_Dropped[Amazon_Food_Reviews_Duplicates_Dropped.HelpfulnessNumerator<=Amazon_Food_Reviews_Duplicates_Dropped.HelpfulnessDenominator]

    print("{:07.3f} % of data removed".format(np.round((1-(Amazon_Food_Reviews_Preprocessed.shape[0])/(Amazon_Food_Reviews_Duplicates_Dropped.shape[0]))*100)))

    return Amazon_Food_Reviews_Preprocessed
Dataset_Name='database.sqlite'

#cleaning 5000 records

Amazon_Food_Reviews_Preprocessed= Cleaning_Pipeline(Dataset_Name,FullDataLoad=True)
def Balanced_Data(Total_Dataset):

    return pd.concat([Total_Dataset[Total_Dataset['Score']==1].sample(frac=0.035,random_state=1),Total_Dataset[Total_Dataset['Score']==0].sample(frac=0.15,random_state=1)],axis=0)
Balanced_Amazon_Data_Preprocessed=Balanced_Data(Total_Dataset=Amazon_Food_Reviews_Preprocessed)
Balanced_Data_Time_Sorted=Balanced_Amazon_Data_Preprocessed.sort_values('Time',axis=0,kind='quicksort',ascending=True).reset_index(drop=True)
Balanced_Data_Time_Sorted['Score'].value_counts().plot(kind='bar')
def Find_Optimal_K_In_KNN(X_Train,Y_Train,K_Values_List):

    #creating odd list of K values

    neighbors= list(filter(lambda k: k%2!=0,K_Values_List))

    CV_Scores=[]

    for k in tqdm(neighbors):

        knn = KNeighborsClassifier(n_neighbors=k,n_jobs=-1)

        scores = cross_val_score(knn,X_Train,Y_Train,cv=10,scoring='accuracy',n_jobs=-1)

        CV_Scores.append(scores.mean())

    MSE = [1-x for x in CV_Scores]

    Optimal_K = neighbors[MSE.index(min(MSE))]

    print('\nThe optimal number of neighbors is {}'.format(Optimal_K))

    plt.figure(figsize=(12,10))

    plt.plot(neighbors,MSE,color='blue', linestyle='dashed',marker='o',markerfacecolor='red', markersize=10)

    plt.title('Error Rate vs. K Value')

    plt.xlabel('K')

    plt.ylabel('Error Rate')

    print("the misclassification error for each k value is : ", np.round(MSE,3))

    return Optimal_K
X_Train,X_Test,Y_Train,Y_Test = train_test_split(Balanced_Data_Time_Sorted['Text'],Balanced_Data_Time_Sorted['Score'],test_size=0.25)
from sklearn.preprocessing import StandardScaler



cnt_vect = CountVectorizer(ngram_range=(1,2),min_df=10,max_features=5000)

X_Train_Features=cnt_vect.fit_transform(X_Train)

X_Test_Features = cnt_vect.transform(X_Test)
K_Neighbours_List = list(range(0,20))

Optimal_K = Find_Optimal_K_In_KNN(X_Train_Features.toarray() ,Y_Train,K_Neighbours_List)
knn = KNeighborsClassifier(n_neighbors=Optimal_K)

knn.fit(X_Train_Features, Y_Train)

kn_pred = knn.predict(X_Test_Features)


knn_confusion_matrix=confusion_matrix(Y_Test,kn_pred)

skplt.plot_confusion_matrix(Y_Test,kn_pred)
from sklearn.metrics import accuracy_score,classification_report

print(accuracy_score(Y_Test,kn_pred))
print(classification_report(Y_Test,kn_pred))
from sklearn.feature_extraction.text import TfidfVectorizer

#X_Train,X_Test,Y_Train,Y_Test = train_test_split(Balanced_Data_Time_Sorted['Text'],Balanced_Data_Time_Sorted['Score'],test_size=0.25)

TFIDF_vect = TfidfVectorizer(ngram_range=(1,2),min_df=10,max_features=5000)

X_Train_Features=TFIDF_vect.fit_transform(X_Train)

X_Test_Features = TFIDF_vect.transform(X_Test)
K_Neighbours_List = list(range(0,20))

Optimal_K = Find_Optimal_K_In_KNN(X_Train_Features.toarray() ,Y_Train,K_Neighbours_List)

knn = KNeighborsClassifier(n_neighbors=Optimal_K)

knn.fit(X_Train_Features, Y_Train)

knn_TFIDF_pred = knn.predict(X_Test_Features)


knn_confusion_matrix=confusion_matrix(Y_Test,knn_TFIDF_pred)

skplt.plot_confusion_matrix(Y_Test,knn_TFIDF_pred)

print(accuracy_score(Y_Test,knn_TFIDF_pred))
print(classification_report(Y_Test,knn_TFIDF_pred))