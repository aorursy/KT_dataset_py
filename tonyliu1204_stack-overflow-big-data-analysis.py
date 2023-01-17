import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
stackOverflow = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="stackoverflow")
def del_order_mark(df):
    new_Reputation = []
    for i in range(len(df)): # delete the order mark in "repulation"
        new_Reputation.append(df['Reputation'][i][1:])
    df.Reputation = new_Reputation
    return df
bq_assistant = BigQueryHelper("bigquery-public-data", "stackoverflow")
bq_assistant.list_tables()
# How many users are questioner ?
query1 = '''
select count(distinct q.owner_user_id)
from `bigquery-public-data.stackoverflow.posts_questions` q
left join `bigquery-public-data.stackoverflow.posts_answers` a
on q.owner_user_id = a.owner_user_id
where a.owner_user_id is null
'''

# How many users are answerer ?
query2 = '''
select count(distinct a.owner_user_id)
from `bigquery-public-data.stackoverflow.posts_answers` a
left join `bigquery-public-data.stackoverflow.posts_questions` q
on a.owner_user_id = q.owner_user_id
where q.owner_user_id is null
'''

# How many users are question-and-answerer ?
query3='''
select count( distinct q.owner_user_id)
from `bigquery-public-data.stackoverflow.posts_questions` q
inner join `bigquery-public-data.stackoverflow.posts_answers` a 
on q.owner_user_id = a.owner_user_id
'''

# How many users are do-nothinger ?
query4='''
select count(id)
from `bigquery-public-data.stackoverflow.users` u
left join (
    select distinct owner_user_id
    from `bigquery-public-data.stackoverflow.posts_answers`
    union all
    select distinct owner_user_id
    from `bigquery-public-data.stackoverflow.posts_questions`) b
on u.id = b.owner_user_id
where b.owner_user_id is null
'''

# Execute the queries
questioner = stackOverflow.query_to_pandas_safe(query1).iat[0,0]
answerer = stackOverflow.query_to_pandas_safe(query2).iat[0,0]
question_and_answerer = stackOverflow.query_to_pandas_safe(query3).iat[0,0]
do_nothinger = stackOverflow.query_to_pandas_safe(query4).iat[0,0]
num_user = stackOverflow.query_to_pandas_safe("select count(*) from `bigquery-public-data.stackoverflow.users` ").iat[0,0]

# Show result
user_type_df = pd.DataFrame({"Number of Users": [questioner, answerer, question_and_answerer, do_nothinger, num_user]})
user_type_df["Percentage(%)"] = round(user_type_df["Number of Users"] / num_user * 100,2)
user_type_df.index = ["Questioner", "Answerer", "Question-and-answerer", "Do-nothinger", "Total"]
display(user_type_df)
query1 = '''
SELECT
    rep_range AS Reputation,
    COUNT(*) AS Users,
    SUM(asked) AS Asked_question,
    SUM(unanswered) AS Unanswered_question,
    SUM(answered) AS Contributed_answer
FROM(
    SELECT 
        CASE
            WHEN reputation BETWEEN 1 AND 100 THEN '11- 100'
            WHEN reputation BETWEEN 101 AND 1000 THEN '2101- 1000'
            WHEN reputation BETWEEN 1001 AND 10000 THEN '31001- 10000'
            WHEN reputation BETWEEN 10001 AND 100000 THEN '410001- 100000'
            WHEN reputation > 100000 THEN '5> 100000'
        END AS rep_range,
        asked,
        answered,
        unanswered
    FROM(    
        SELECT id AS user_id, reputation, asked, answered, unanswered
        FROM `bigquery-public-data.stackoverflow.users` u
        LEFT JOIN(
            SELECT owner_user_id AS user_id, COUNT(*) AS asked
            FROM `bigquery-public-data.stackoverflow.posts_questions`
            GROUP BY user_id
        ) q ON u.id = q.user_id
        LEFT JOIN(
            SELECT owner_user_id AS user_id, COUNT(*) AS answered
            FROM `bigquery-public-data.stackoverflow.posts_answers`
            GROUP BY user_id
        ) a ON u.id = a.user_id
        LEFT JOIN(
            SELECT owner_user_id AS user_id, COUNT(*) AS unanswered 
            FROM (
                SELECT owner_user_id
                FROM `bigquery-public-data.stackoverflow.posts_questions`
                WHERE answer_count=0
            )
            GROUP BY user_id
        ) ua ON u.id = ua.user_id
    )
)
GROUP BY rep_range
ORDER BY rep_range
'''

profile = del_order_mark(stackOverflow.query_to_pandas_safe(query1))
profile.index = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
display(profile)

# normalize the profile
profile_per = profile.copy()
profile_per.Users = round(100 * profile_per.Users / profile_per.Users.sum(),5)
profile_per.Asked_question = round(100 * profile_per.Asked_question / profile_per.Asked_question.sum(),2)
profile_per.Unanswered_question = round(100 * profile_per.Unanswered_question / profile_per.Unanswered_question.sum(),2)
profile_per.Contributed_answer = round(100 * profile_per.Contributed_answer / profile_per.Contributed_answer.sum(),)
profile_per.rename(columns = {'Users':'Users(%)','Asked_question':'Asked_question(%)','Unanswered_question':'Unanswered_question(%)','Contributed_answer':'Contributed_answer(%)'},inplace=True)
display(profile_per)
query1 = '''
select a.rep_range as Reputation, (num_ans + num_que) as num_post
from(
    select rep_range, sum(num_ans) as num_ans
    from(
        select 
            case
                when reputation between 1 and 100 then '11- 100'
                when reputation between 101 and 1000 then '2101- 1000'
                when reputation between 1001 and 10000 then '31001- 10000'
                when reputation between 10001 and 100000 then '410001- 100000'
                when reputation > 100000 THEN '5> 100000'
            end as rep_range,
            num_ans
        from(
        select reputation, num_ans
        from `bigquery-public-data.stackoverflow.users`
        inner join(
            select owner_user_id, count(*) as num_ans
            from `bigquery-public-data.stackoverflow.posts_answers`
            group by owner_user_id)
        on id = owner_user_id)
        )
    group by rep_range) a
inner join(
    select rep_range, sum(num_que) as num_que
    from(
        select 
            case
                when reputation between 1 and 100 then '11- 100'
                when reputation between 101 and 1000 then '2101- 1000'
                when reputation between 1001 and 10000 then '31001- 10000'
                when reputation between 10001 and 100000 then '410001- 100000'
                when reputation > 100000 THEN '5> 100000'
            end as rep_range,
            num_que
        from(
        select reputation, num_que
        from `bigquery-public-data.stackoverflow.users`
        inner join(
            select owner_user_id, count(*) as num_que
            from `bigquery-public-data.stackoverflow.posts_questions`
            group by owner_user_id)
        on id = owner_user_id)
        )
    group by rep_range) b
on a.rep_range = b.rep_range
order by Reputation
'''
num_post = del_order_mark(stackOverflow.query_to_pandas_safe(query1))
num_post.index = ["Level 1", "Level 2", "Level 3", "Level 4", "Level 5"]
num_post["per_user"] = num_post.num_post / profile.Users 
num_post["per_user_per_month"] = num_post["per_user"] / (11 * 12+5) # the oldest post is 2008.7.31, the latest is 2019.12.1
display(num_post)
query1 = """SELECT
  EXTRACT(YEAR FROM creation_date) AS Year,
  COUNT(*) AS Number_of_Questions,
  ROUND(100 * SUM(IF(answer_count > 0, 1, 0)) / COUNT(*), 1) AS Percent_Questions_with_Answers
FROM
  `bigquery-public-data.stackoverflow.posts_questions`
GROUP BY
  Year
ORDER BY
  Year;
        """
answer_rate = stackOverflow.query_to_pandas_safe(query1)
display(answer_rate)
fig = plt.figure()
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.grid()
plt.rc('grid', linestyle="dotted", color='gray')
plt.xlabel('Year', fontsize=15)
plt.ylabel('Answer rate', fontsize=15)
plt.plot(answer_rate['Year'], answer_rate['Percent_Questions_with_Answers'],'ro-')
fig.set_size_inches(16, 8)
profile_per["Answer_rate(%)"] = round((profile["Asked_question"] - profile["Unanswered_question"]) / profile["Asked_question"] * 100, 2)
display(profile_per)
# the distribution of comment number

query1 = '''
select comment_count, count(*) as num
from `bigquery-public-data.stackoverflow.posts_questions`
where answer_count = 0
group by comment_count
order by comment_count asc
'''

query2 = '''
select comment_count, count(*) as num
from `bigquery-public-data.stackoverflow.posts_questions`
where answer_count > 0
group by comment_count
order by comment_count asc
'''


comment_count_unanswered = stackOverflow.query_to_pandas_safe(query1)
comment_count_answered = stackOverflow.query_to_pandas_safe(query2)


labelsize = 15
plt.figure(figsize=(16,8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

x1 = comment_count_answered.comment_count
y1 = comment_count_answered.num

x2 = comment_count_unanswered.comment_count
y2 = comment_count_unanswered.num

plt.plot(x1,y1, label="answered question",linestyle='--')
plt.plot(x2,y2, label="unanswered question")
# plt.xscale("log")
plt.yscale("log")
plt.xlabel('Comment count',fontsize = labelsize)
plt.ylabel('Number of question',fontsize = labelsize)
plt.grid(True)
plt.legend(fontsize=15)
plt.title("Frequency distribution of comment count", fontsize=labelsize)

labelsize = 15
plt.figure(figsize=(16,8))
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

x1 = comment_count_answered.comment_count
y1 = np.cumsum(comment_count_answered.num)
y1 = 100 * y1 / y1.max()

x2 = comment_count_unanswered.comment_count
y2 = np.cumsum(comment_count_unanswered.num)
y2 = 100 * y2 / y2.max()

plt.plot(x1,y1, label="answered question",linestyle='--')
plt.plot(x2,y2, label="unanswered question")
# plt.xscale("log")
# plt.yscale("log")
plt.xlabel('Comment count',fontsize = labelsize)
plt.ylabel('CDF(%)',fontsize = labelsize)
plt.grid(True)
plt.legend(fontsize=15)
plt.title("Cumulative distribution function of comment count", fontsize=labelsize)
# Reputation of the user making comments to answered questions
query1 = '''
select 
    case
        when uc.reputation between 1 and 100 then '11- 100'
        when uc.reputation between 101 and 1000 then '2101- 1000'
        when uc.reputation between 1001 and 10000 then '31001- 10000'
        when uc.reputation between 10001 and 100000 then '410001- 100000'
        when uc.reputation > 100000 THEN '5> 100000'
    end as Reputation,
    sum(uc.num) as num
from(    
select u.reputation, count(*) as num
from `bigquery-public-data.stackoverflow.users` u
inner join(
    select c.user_id
    from `bigquery-public-data.stackoverflow.comments` c
    inner join (
        select id from `bigquery-public-data.stackoverflow.posts_questions`
        where answer_count > 0) q
    on post_id = q.id)
on id = user_id
group by reputation
order by reputation asc) uc
group by Reputation
order by Reputation
'''

temp = stackOverflow.query_to_pandas(query1)
temp = del_order_mark(temp)

labels = temp.Reputation
sizes = round(100 * temp.num / temp.num.sum(),2)
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff6666']
explode = (0.05,0.05,0.05,0.05,0.05)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode)
ax1.axis('equal')  
plt.tight_layout()
plt.title("Reputation of the user making comments to answered questions", fontsize = 14)

# Reputation of the user making comments to unanswered questions
query2 = '''
select 
    case
        when uc.reputation between 1 and 100 then '11- 100'
        when uc.reputation between 101 and 1000 then '2101- 1000'
        when uc.reputation between 1001 and 10000 then '31001- 10000'
        when uc.reputation between 10001 and 100000 then '410001- 100000'
        when uc.reputation > 100000 THEN '5> 100000'
    end as Reputation,
    sum(uc.num) as num
from(    
select u.reputation, count(*) as num
from `bigquery-public-data.stackoverflow.users` u
inner join(
    select c.user_id
    from `bigquery-public-data.stackoverflow.comments` c
    inner join (
        select id from `bigquery-public-data.stackoverflow.posts_questions`
        where answer_count = 0) q
    on post_id = q.id)
on id = user_id
group by reputation
order by reputation asc) uc
group by Reputation
order by Reputation
'''

temp2 = stackOverflow.query_to_pandas(query2)
temp2 = del_order_mark(temp2)

labels = temp2.Reputation
sizes = round(100 * temp2.num / temp2.num.sum(),2)
colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99','#ff6666']
explode = (0.05,0.05,0.05,0.05,0.05)

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90, explode=explode)
ax1.axis('equal')  
plt.tight_layout()
plt.title("Reputation of the user making comments to unanswered questions", fontsize = 14)
# pick out top 25 tags
query1 = '''
select tag_name, count
from `bigquery-public-data.stackoverflow.tags`
order by count desc
limit 25
'''

# pick out top 50 tags
query2 = '''
select tag_name, count
from `bigquery-public-data.stackoverflow.tags`
order by count desc
limit 50
'''

# pick out top 100 tags
query3 = '''
select tag_name, count
from `bigquery-public-data.stackoverflow.tags`
order by count desc
limit 100
'''

top_25_tag = stackOverflow.query_to_pandas(query1)
top_50_tag = stackOverflow.query_to_pandas(query2)
top_100_tag = stackOverflow.query_to_pandas(query3)

top_25_list = top_25_tag.tag_name.tolist()
top_50_list = top_50_tag.tag_name.tolist()
top_100_list = top_100_tag.tag_name.tolist()


# The following command are used for SQL query (I know it is kinda brute force......)
top_25_query_command = ''
for s in top_25_list:
    top_25_query_command += 'tags like \'%%|%s|%%\' or tags like \'%%|%s\' or tags like \'%s|%%\' or ' % (s,s,s)

top_50_query_command = ''
for s in top_50_list:
    top_50_query_command += 'tags like \'%%|%s|%%\' or tags like \'%%|%s\' or tags like \'%s|%%\' or ' % (s,s,s)

top_100_query_command = ''
for s in top_100_list:
    top_100_query_command += 'tags like \'%%|%s|%%\' or tags like \'%%|%s\' or tags like \'%s|%%\' or ' % (s,s,s)

    
# Delete the ' or ' at the end of the command
top_25_query_command = top_25_query_command[:-4]
top_50_query_command = top_50_query_command[:-4]
top_100_query_command = top_100_query_command[:-4]
  
    
# Feel free to have a look at the query statement
# print(top_25_query_command,end='\n\n')
# print(top_50_query_command,end='\n\n')
# print(top_100_query_command,end='\n\n')


query_top_25 = '''
select count(*) as num
from `bigquery-public-data.stackoverflow.posts_questions`
where  %s''' % top_25_query_command

query_top_50 = '''
select count(*) as num
from `bigquery-public-data.stackoverflow.posts_questions`
where  %s''' % top_50_query_command

query_top_100 = '''
select count(*) as num
from `bigquery-public-data.stackoverflow.posts_questions`
where  %s''' % top_100_query_command

query_num_question='''
select count(*) as num
from `bigquery-public-data.stackoverflow.posts_questions`
'''
top_25_num = stackOverflow.query_to_pandas(query_top_25).iat[0,0]
top_50_num = stackOverflow.query_to_pandas(query_top_50).iat[0,0]
top_100_num = stackOverflow.query_to_pandas(query_top_100).iat[0,0]
num_question = stackOverflow.query_to_pandas(query_num_question).iat[0,0]

temp = pd.DataFrame({"Top_N_tag":[25,50,100], "Number of Relevant Questions":[top_25_num, top_50_num, top_100_num]})
temp["Percentage(%)"] = round(temp["Number of Relevant Questions"] / num_question * 100, 2)
display(temp)
print('Top 25 tags: \n',top_25_list)

# Feel free to check out top 50 tags and top 100 tags
# print('Top 50 tags: \n',top_50_list)
# print('Top 100 tags: \n',top_100_list)
# ================= !!! NOTE !!! ==============
# The code in this part is not efficient at all. 
# Running the code for the top 6 tags('javascript', 'java', 'c#', 'python', 'php','android') is okay 
# but doing it for all the top 25 tags is very time-consuming

import datetime
from dateutil.relativedelta import *

# tag_list = ['javascript', 'java', 'c#', 'python', 'php', 'android', 'jquery', 'html', 'c++', 'css', 'ios', 'mysql', 'sql', 'asp.net', 'r', 'c', 'arrays', 'ruby-on-rails', 'node.js', '.net', 'objective-c', 'json', 'sql-server', 'angularjs', 'swift']
tag_list = ['javascript', 'java', 'c#', 'python', 'php','android']

result = pd.DataFrame()
date_string_list = []

for tag in tag_list:
    start_date = datetime.datetime(2008,8,1)
    end_date = datetime.datetime(2008,11,1)
    final_date = datetime.datetime(2019,12,1)
    print('Now working on %s ......'%tag)
    
    temp = []
    while end_date < final_date:
        start_date_string = start_date.strftime('%Y-%m-%d')
        end_date_string = end_date.strftime('%Y-%m-%d')
        
        date_string_list.append(start_date_string)
        
        query = '''
        select count(*) as num
        from `bigquery-public-data.stackoverflow.posts_questions`
        where (tags like '%%|%s|%%' or tags like '%%|%s' or tags like '%s|%%')
        and date(creation_date) >= '%s' and date(creation_date) < '%s'
        ''' % (tag, tag, tag,start_date_string ,end_date_string)
        temp.append(stackOverflow.query_to_pandas(query).iat[0,0])
        
        start_date += relativedelta(months=+3)
        end_date += relativedelta(months=+3)
        
    result[tag] = temp
result.index = date_string_list[:len(date_string_list) // len(tag_list)]
print('Complete')

# Plot the topcial trend

matplotlib.style.use('default')
labelsize = 15
tag_list = result.columns.tolist()
x = result.index.tolist()

NUM_COLORS = len(tag_list)
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)

cm = plt.get_cmap('tab10')
fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(18.5, 10.5)
for i in range(NUM_COLORS):
    lines = ax.plot(x,result[tag_list[i]])
    lines[0].set_color(cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])

plt.setp(ax.xaxis.get_majorticklabels(), rotation=70 )
plt.legend(fontsize=11.38)
plt.grid(True)
plt.title("Number of question of Top 25 tags", fontsize=labelsize)
plt.xlabel('time',fontsize = labelsize)
plt.ylabel('Number of question',fontsize = labelsize)
plt.savefig('Number of question of Top 25 tags c')
# plot percentage of each tag in every three months
labelsize = 15

tag_list = result.columns.tolist()
x = result.index.tolist()
result_new = result.div(result.sum(axis=1), axis=0) * 100 # normalize 

NUM_COLORS = len(tag_list)
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)

cm = plt.get_cmap('tab10')
fig = plt.figure()
ax = fig.add_subplot(111)
fig.set_size_inches(18.5, 10.5)
for i in range(NUM_COLORS):

    lines = ax.plot(x,result_new[tag_list[i]])
    lines[0].set_color(cm(i//NUM_STYLES*float(NUM_STYLES)/NUM_COLORS))
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])

plt.setp(ax.xaxis.get_majorticklabels(), rotation=70 )
plt.legend(fontsize=11.38, bbox_to_anchor=(1, 1))
plt.grid(True)
plt.title("Percentage of question of Top 25 tags", fontsize=labelsize)
plt.xlabel('time',fontsize = labelsize)
plt.ylabel('Percentage of question(%)',fontsize = labelsize)
plt.savefig('Percentage of question of Top 25 tags')
# ================= !!! NOTE !!! ==============
# The code in this part is even less efficient than the previous part. 
# Running the code may run out of your quota of query in the current month
# You may need more than one kaggle account to work out the complete result yourself
# Or you can try impoving the code yourself. I am not really good at sql :p

import datetime
from dateutil.relativedelta import *

# tag_list = ['javascript', 'java', 'c#', 'python', 'php', 'android', 'jquery', 'html', 'c++', 'css', 'ios', 'mysql', 'sql', 'asp.net', 'r', 'c', 'arrays', 'ruby-on-rails', 'node.js', '.net', 'objective-c', 'json', 'sql-server', 'angularjs', 'swift']
tag_list = ['javascript', 'java', 'c#', 'python', 'php','android']

result_var = pd.DataFrame()
date_string_list = []

for tag in tag_list:
    start_date = datetime.datetime(2008,8,1)
    end_date = datetime.datetime(2008,9,1)
    final_date = datetime.datetime(2019,12,1)
    print('Now working on %s ......'%tag)
    
    temp = []
    while end_date < final_date:
        start_date_string = start_date.strftime('%Y-%m-%d')
        end_date_string = end_date.strftime('%Y-%m-%d')
        
        date_string_list.append(start_date_string)
        
        query = '''
        select variance(num) as var
        from(
            select creation_date, count(*) as num
            from(
                select date(creation_date) as creation_date
                from `bigquery-public-data.stackoverflow.posts_questions`
                where (tags like '%%|%s|%%' or tags like '%%|%s' or tags like '%s|%%')
                and date(creation_date) >= '%s' and date(creation_date) < '%s')
            group by creation_date
            )
        ''' % (tag, tag, tag,start_date_string ,end_date_string)
        temp.append(stackOverflow.query_to_pandas(query).iat[0,0])
        
        start_date += relativedelta(days=+7)
        end_date += relativedelta(days=+7)
        
    result_var[tag] = temp
result_var.index = date_string_list[:len(date_string_list) // len(tag_list)]
print('Complete')
result_var.to_csv('result_var.csv')
print('Save Complete')