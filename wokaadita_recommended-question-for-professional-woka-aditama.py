import pandas as pd

import operator

import re

from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.externals import joblib

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt







#Load files and datasets

answer=pd.read_csv('answers.csv')

professional=pd.read_csv('professionals.csv')

tags=pd.read_csv('tags.csv')

email=pd.read_csv('emails.csv')

match=pd.read_csv('matches.csv')

question_score=pd.read_csv('question_scores.csv')

answer_score=pd.read_csv('answer_scores.csv')

questi=pd.read_csv('questions.csv')

tag_=pd.read_csv('tag_questions.csv')

tag_user=pd.read_csv('tag_users.csv')
questi[questi['questions_body']=='<p>I am a sophomore in Boston and I am not sure what I want to do yet. I really think I want to be a pediatric nurse but I am not sure if I will want to go further into the health and medicine field if I really like it. </p>']
questi[questi['questions_body'].str.contains('http://www.typeoflawyer.com/different-types-of-law-careers/')]
questi.isnull().sum() 
#Rename the name of column in 'tag_users.csv' table to be merged with 'tags.csv'

tag_user=tag_user.rename(index=str, columns={'tag_users_tag_id':'tags_tag_id'})

tag_professional=tag_user.merge(tags, how='left', on='tags_tag_id')



#Merging the tag_professional for question with 'professionals.csv'

tag_professional=tag_professional.rename(index=str, columns={'tag_users_user_id':'professionals_id'})

final_professional=professional.merge(tag_professional, how='left', on='professionals_id')



#Rename the name of column in 'Professionals.csv' table to be merged with 'answers.csv'

final_professional=final_professional.rename(index=str, columns={'professionals_id':'answers_author_id'})

answer_prof_merged=answer.merge(final_professional, how='left', on='answers_author_id')



#Rename the name of column in 'answer_score.csv' table to be merged with 'answer_prof_merged'

answer_score1=answer_score.rename(index=str, columns={'id':'answers_id'})

answer_prof_merged=answer_prof_merged.merge(answer_score1, how='left', on='answers_id')



#Rename the name of column in 'question_scores.csv' table to be merged with 'questions.csv'

question_score1=question_score.rename(index=str, columns={'id':'questions_id'})

questi=questi.merge(question_score1, how='left', on='questions_id')



#Rename the name of column in 'tag_questions.csv' table to be merged with 'tags.csv'

tag_=tag_.rename(index=str, columns={'tag_questions_question_id':'questions_id'})

tag1=tags.rename(index=str, columns={'tags_tag_id':'tag_questions_tag_id'})



final_tag=tag_.merge(tag1, how='left', on='tag_questions_tag_id')





#Merging the tags for question with 'questions.csv'

questi1=questi.merge(final_tag, how='left', on='questions_id')
import nltk



temporary1=[]

temporary2=[]

end=[]

question_new_list=[]



#Cleaned Texts from HTML tags, URL, Hashtag, and then translating all of them to english as the default 

def cleaned_list(file,name):

    text_=list(str(x) for x in file[name])

    question_new_list=[]

    for i in text_:

        cleaned_str=BeautifulSoup(i)

        cleaned_text=cleaned_str.get_text()

        result= re.sub(r"http\S+", "", cleaned_text)

        result0=re.sub("(\\d|\\W)+"," ",result)

        result1=result0.replace('#','')

        question_new_list.append(result1)

    return question_new_list

                   

#Some of questions are posted as title rather than body, so we merged it both of them as 'merged text'

def merged_body_and_title(file):

    bodies= cleaned_list(file,'questions_body')

    titles= cleaned_list(file,'questions_title')

    for i in range(0, len(titles)):

        nltk_tokens_body = nltk.word_tokenize(bodies[i])

        nltk_tokens_title = nltk.word_tokenize(titles[i])

        for f in nltk_tokens_title :

            temporary1.append(f)

            t=' '.join(temporary1)

        for v in nltk_tokens_body :

            temporary2.append(v)

            t1=' '.join(temporary2)

            t3=t+'.'+t1

        end.append(t3)

        temporary1.clear()

        temporary2.clear()

    return end



questi1['merged_question']=merged_body_and_title(questi1)

answer_prof_merged['merged_question']=cleaned_list(answer_prof_merged,'answers_body')
nan_df=questi1[questi1['tags_tag_name'].isnull()]

non_nan_df=questi1[questi1['tags_tag_name'].notnull()]
tag_name=list(str(x) for x in questi1['tags_tag_name'].drop_duplicates())

tag_name.sort()

dict_tag={}



for i in tag_name:

    dict_tag[i]=[u for u in questi1[questi1['tags_tag_name']==i].merged_question]
non_nan_list=list(str(x) for x in questi1['merged_question'].drop_duplicates())

answer_list=list(str(x) for x in answer_prof_merged['answers_body'])



tfidf=TfidfVectorizer(stop_words='english')

X_idf = tfidf.fit_transform(non_nan_list)
# Sum_of_squared_distances = []

# K = range(1,200)

# for k in K:

#     print(k)

#     model = KMeans(n_clusters=k)

#     km=model.fit(X_idf)

#     Sum_of_squared_distances.append(km.inertia_)
# plt.figure(figsize=(80,80))

# plt.plot(K, Sum_of_squared_distances, 'bx-')

# plt.xlabel('k')

# plt.ylabel('Sum_of_squared_distances')

# plt.title('Elbow Method For Optimal k')

# plt.show()


true_k =138

model = KMeans(n_clusters=true_k, init='k-means++',max_iter=300, n_init=10)

model.fit(X_idf)

    

print("Top terms per cluster:")

order_centroids = model.cluster_centers_.argsort()[:, ::-1]

terms = tfidf.get_feature_names()



# print(order_centroids)

for i in range(true_k):

    print("Cluster %d:" % i),

    for t in order_centroids[i, :10]:

        print(' %s' % terms[t]),

print
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

 

def similarity(X_list, y_list):

    X = tfidf.transform(X_list)

    y=tfidf.transform(y_list)

    

    d=cosine_similarity(X, y)

    return d

 
# true_k =125

# model = KMeans(n_clusters=true_k, init='k-means++',max_iter=300, n_init=10)

# model.fit(X_idf)



full_list_question=list(str(x) for x in questi1['merged_question'])



def labeling(name_list):

    clustering=[]

    for i in name_list:

        Y = tfidf.transform([i])

        prediction = model.predict(Y)

        clustering.append(prediction)

    return clustering







questi1['label']=labeling(full_list_question)

answer2=answer_prof_merged.rename(index=str, columns={'answers_question_id':'questions_id'})



answer_and_question=answer2.merge(questi1,how='left', on='questions_id')



answer_and_question
answer_and_question.isnull().sum()
from datetime import datetime

interval=[]

for i in range (0, len(answer_and_question)):

    answer_date=datetime.strptime(str(answer_and_question['answers_date_added'][i]),'%Y-%m-%d %H:%M:%S UTC+0000')

    sent_date=datetime.strptime(str(answer_and_question['questions_date_added'][i]),'%Y-%m-%d %H:%M:%S UTC+0000')

    diff= (answer_date-sent_date).days

    interval.append(diff)

answer_and_question['interval']=interval
answer_recommend=answer_and_question.groupby('answers_author_id').agg({'interval':'mean'})

answer_final_end=answer_and_question.merge(answer_recommend, how='left', on='answers_author_id')
answer_recommend.head()
def not_answerred_prof(label):

    label=answer_and_question[answer_and_question['label'].isin(label)]

    list_tags=list(str(x) for x in label['tags_tag_name_x'].drop_duplicates())

    for i in list_tags:

        if i is not np.nan:

            final=final_professional[final_professional['tags_tag_name'].isin(list_tags)]

            final1=final[['answers_author_id','tags_tag_name']].drop_duplicates(subset='answers_author_id')

    return final1
def find_the_professional(text_list):

    dictionary={}

    list_similarity=[]

    label_list=labeling(text_list)

    for i in range(0,len(text_list)):

        dictionary[text_list[i]]=label_list[i]

        df=answer_final_end[answer_final_end['label'].isin(label_list)]

        sim_score=similarity(text_list,list(str(x) for x in df['merged_question_y']))

        for i in sim_score:

            for c in i:

                list_similarity.append(c)

        df['similarity_score']=list_similarity

        df1=df.sort_values(by=['similarity_score'], ascending=False)

        professional_result=df1[['answers_author_id','interval_y', 'similarity_score', 'score_x']].drop_duplicates(subset='answers_author_id')

        professional_result=professional_result.rename(columns={'interval_y':'interval mean', 'score_x':'heart'})

        additional_prof=not_answerred_prof(label_list)

    print('\033[1m','List of Professional For This Question:','\033[1m')

    return professional_result, additional_prof



a,b=find_the_professional(['Do you know how to be a good data scientist expert?'])  

display(a)

print('\033[1m','--------------------------------------------------------------------------------------------------------------','\033[1m')

print('\033[1m','Additional Professional:','\033[1m')

display(b)