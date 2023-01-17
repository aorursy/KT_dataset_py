# 载入库

import numpy as np 

import pandas as pd

import matplotlib.mlab as mlab

import matplotlib.pyplot as plt

import os



# 读入文件

file_dir='../input/2020mcm/microwave.tsv'

data=pd.read_csv(file_dir,engine='python',sep='\t')

data.info()
data['verified_purchase'].value_counts()
labels=['verfied','unverified']

X=[data[data['verified_purchase']=='Y'].count()[0],data[data['verified_purchase']=='N'].count()[0]]

fig = plt.figure()

plt.pie(X,labels=labels,autopct='%1.2f%%')

plt.title("PieChart.jpg")

plt.show()

verified_data=data[data['verified_purchase'].isin(['Y'])]

verified_data['verified_purchase'].value_counts()
slim_data = verified_data.loc[:,['review_id','product_id','vine','review_body','star_rating','helpful_votes','total_votes','review_date','star_ratings']]

slim_data.info()
sales_number=slim_data['product_id'].value_counts()

len(sales_number)
sales_number=slim_data.groupby(['product_id'],as_index=False)['product_id'].agg({'cnt':'count'})

sales_number
small_sales=sales_number[(sales_number['cnt']>=1)&(sales_number['cnt']<=5)]

small_sales.info()
sales_one=sales_number[sales_number['cnt']==1]

sales_one.info()

sales_one.shape
labels=sales_number['product_id']

X=sales_number['cnt']

fig = plt.figure(figsize=(15,15))

plt.pie(X,labels=labels,autopct='%1.2f%%')

plt.title("the sales number pie chart"+" microwave")

plt.savefig("all microwave share.jpg")

plt.show()
X.sum() # 这个X的求取是下面的X=top_XX['cnt']
top_XX=sales_number.nlargest(20,columns='cnt').tail(20)
X=top_XX['cnt']

labels=top_XX['product_id']

fig=plt.figure(figsize=(15,15))

plt.pie(X,labels=labels,autopct='%1.2f%%')

plt.title("slim hair microwave share")

plt.savefig("slim hair microwave share.jpg")

plt.show()
top_XX_data=slim_data[slim_data['product_id'].isin(labels)]

top_XX_data.info()
labels
top_XX_data.to_csv("./top_50_microwave.csv",index=False)
tmp=pd.read_csv('./top_50_microwave.csv',sep=',')

tmp.info()
from textblob import TextBlob

from wordcloud import WordCloud

import pandas as pd

import numpy as np

import csv

from os import listdir



def getComments(filename): # 获取评论列表、评论中所有的单词，以空格分隔

    comments = np.zeros(0)

    words = ''

    com_file = pd.read_csv(filename,sep=',')

    comments = np.append(comments, com_file['review_body'])

    for each in comments:

        words += each

    replace_list = [',', '.', '\'', '\"']

    for each in replace_list:

        words = words.replace(each, ' ')

    return comments, words



def getWordCloud(text_str, picture_name): # 生成词云

    wordcloud = WordCloud(background_color="white",width=1980, height=1080, margin=1, random_state=0).generate(text_str)

    wordcloud.to_file(picture_name)



def get_p_or_n(comments): # 获取情绪极化评分，并划定阈值确定是积极、消极或中立

    with open('result.csv', 'w', encoding='utf-8') as csvfile:

        id = 0

        writer = csv.writer(csvfile)

        writer.writerow(['id', 'result', 'score', 'comment'])

        with open('samples.csv', 'w', encoding='utf-8') as samples_file:

            writer_samples = csv.writer(samples_file)

            writer_samples.writerow(['id', 'result', 'score', 'OurJudge', 'comment'])

            for each in comments:

                judge = TextBlob(each)

                # print(each)

                result = ''

                score = judge.sentiment.polarity

                if score > 0.05:

                    result = '积极'

                elif score < -0.03:

                    result = '消极'

                else:

                    result = '中立'

                id += 1

                writer.writerow([id, result, score, each])

                if id%5 == 0:

                    writer_samples.writerow([id, result, score, '', each])

    result_df=pd.read_csv('result.csv',sep=',')

    init_df=pd.read_csv('./top_50_microwave.csv',sep=',')

    init_df['score']=result_df['score']

    init_df.to_csv('./microwave_score.csv',index=False)

                    

                    

def main():

    filename = "./top_50_microwave.csv"

    comments, words = getComments(filename)

    print(len(comments))

    getWordCloud(words, "WordCloud_microwave.png")

    get_p_or_n(comments)



if __name__ == "__main__":

    main()

data=pd.read_csv('./microwave_score.csv',sep=',')

data['score']=(data['score']+1)/2
plt.hist(data['score'])
def transfer_vine(data):

    if data['vine']=='Y':

        return 1.2

    else:

        return 1.0

data['vine']=data.apply(transfer_vine,axis=1)    
data['vine'].value_counts()
data_test=data.loc[:,['helpful_votes','total_votes']]
def calculate_helpful(df):

    if df['total_votes']==0:

        return 1

    else:

        return 1+df['helpful_votes']/df['total_votes']*0.5

    

data_test['helpful rating']=data.apply(calculate_helpful,axis=1)  

data_test['helpful rating']
data['helpfulness rating']=data_test['helpful rating']
data_test=data.loc[:,'star_rating']

def transfer(data):

    return data['star_rating']/5

data_test['star_rating']=data.apply(transfer,axis=1)

data['star_rating']=data_test['star_rating']
data['star_rating'].value_counts()
labels=['5 stars','4 stars','3 stars','2 stars','1 stars']

X=[data[data['star_rating']==1.0].count()[0],

   data[data['star_rating']==0.8].count()[0],

  data[data['star_rating']==0.6].count()[0],

  data[data['star_rating']==0.4].count()[0],

  data[data['star_rating']==0.2].count()[0]]

fig=plt.figure()

plt.pie(X,labels=labels,autopct='%1.2f%%')

plt.title("stars share")

# plt.savefig("file_dir")

plt.show()
data['score']=data['score']*100
data.to_csv("ini_review_microwave.csv",index=False)
data['final_review_score']=data['vine']*data['score']*data['helpfulness rating']
plt.hist(data['final_review_score'])
data.to_csv('./every_review_info.csv',index=False)

data_2b=data.loc[:,['product_id','review_id','review_date','final_review_score','review_body','review_date','star_rating']]

data_2b.to_csv('./2b_data.csv',index=False)
data_tmp=data.loc[:,['product_id','final_review_score']]
data_tmp.info()
# 只留下需要处理的列

cols = [col for col in data_tmp.columns if col in['final_review_score']]

# 分组的列

gp_col = 'product_id'

# 根据分组计算平均值

df_mean = data_tmp.groupby(gp_col)[cols].mean()



df_mean
df_final_score=df_mean.sort_values(by='final_review_score',ascending=False)
df_mean.sort_values(by='final_review_score',ascending=False)
df_mean.sort_values(by='final_review_score',ascending=False).head(10)
df_final_score.to_csv("final_microwave_score.csv")
1+1