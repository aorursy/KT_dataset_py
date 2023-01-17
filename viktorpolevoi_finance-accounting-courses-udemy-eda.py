import pandas as pd
import pandas_profiling as pp
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
import plotly
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
pyo.init_notebook_mode()
import os
data = pd.read_csv('../input/finance-accounting-courses-udemy-13k-course/udemy_output_All_Finance__Accounting_p1_p626.csv')
pp.ProfileReport(data)
data[['title','num_subscribers', 'num_reviews', 'rating', 'is_paid']] \
.sort_values(by = 'num_subscribers',ascending = False)[:10] \
.set_index('title').style \
    .format("{:.2f}", subset = ['rating']) \
    .background_gradient(cmap='Blues', subset = ['rating']) \
    .set_caption('Most subscribed courses') \
    .set_properties(padding="15px", border='2px solid white', width='150px')
data[data.num_subscribers > 10000][['title','num_subscribers', 'num_reviews', 'rating', 'is_paid']] \
.sort_values(by = 'rating',ascending = False)[:10] \
.set_index('title').style \
    .format("{:.2f}", subset = ['rating']) \
    .background_gradient(cmap='Blues', subset = ['num_subscribers']) \
    .set_caption('Top rated courses') \
    .set_properties(padding="15px", border='2px solid white', width='150px')
data['rating_diff'] = data.avg_rating_recent - data.avg_rating
data[data.num_subscribers > 10000][['title','num_subscribers', 'avg_rating', 'avg_rating_recent','rating_diff']] \
.sort_values(by = 'rating_diff',ascending = False)[:10] \
.set_index('title').style \
    .format("{:.4f}", subset = ['avg_rating', 'avg_rating_recent','rating_diff']) \
    .background_gradient(cmap='Blues', subset = ['num_subscribers']) \
    .bar(align='mid', color=['#FCC0CB', '#90EE90'], subset = ['rating_diff']) \
    .set_caption('Positive rating change') \
    .set_properties(padding="15px", border='2px solid white', width='150px')
data[data.num_subscribers > 10000][['title','num_subscribers', 'avg_rating', 'avg_rating_recent','rating_diff']] \
.sort_values(by = 'rating_diff')[:10] \
.set_index('title').style \
    .format("{:.4f}", subset = ['avg_rating', 'avg_rating_recent','rating_diff']) \
    .background_gradient(cmap='Blues', subset = ['num_subscribers']) \
    .bar(align='mid', color=['#FCC0CB', '#90EE90'], subset = ['rating_diff']) \
    .set_caption('Nagative rating change') \
    .set_properties(padding="15px", border='2px solid white', width='150px')
pie_cnt = data.is_paid.value_counts()
pie_cnt.rename({True: 'Paid', False: 'Free'}, inplace=True)

fig = go.Figure(data=[go.Pie(labels=pie_cnt.index, values=pie_cnt, hole=.4, textinfo='label+percent')])
fig.update_layout(title_text="Free/paid courses")
fig.show()
data_free = data[data.is_paid == False].sort_values(by = 'num_subscribers',ascending = False)[:10]
fig = go.Figure(go.Bar(
            x=data_free.num_subscribers,
            y=data_free.title,
            orientation='h'))
fig.update_layout(yaxis=dict(autorange="reversed"), title='Top 10 most subscribed free courses')

fig.show()
data['date'] = [x.split('T')[0] for x in data.published_time]
data['date'] = pd.to_datetime(data['date'])
data['date'] = data['date'].apply(lambda x: x.strftime('%Y-%m'))
data_date = data.groupby(['date']).size()
fig = px.line(data_date, 
              x=data_date.index, y=data_date, line_shape = 'linear', title='Created courses', labels={'y': 'Courses'})
fig.update_layout(hovermode='x')
fig.update_xaxes(
    rangeslider_visible=True
)

fig.show()
data_2020 = data[data.date > '2019-12'].sort_values(by = 'num_subscribers',ascending = False)[:10]

fig = go.Figure(go.Bar(
            x=data_2020.num_subscribers,
            y=data_2020.title,
            orientation='h'))
fig.update_layout(yaxis=dict(autorange="reversed"), title='Most subscribed courses that was created in 2020')

fig.show()
data_python = data[data.title.str.contains('Python', regex=False)] \
                .sort_values(by = 'num_subscribers',ascending = False)[:10]

data_python[['title', 'num_subscribers','num_reviews', 'rating', 'is_paid']] \
.set_index('title').style \
    .format("{:.2f}", subset = ['rating']) \
    .background_gradient(cmap='Blues', subset = ['rating']) \
    .set_caption('Most subscribed Python courses') \
    .set_properties(padding="15px", border='2px solid silver', width='150px')
data_python_free = data[data.title.str.contains('Python', regex=False) & (data.is_paid == False)] \
                .sort_values(by = 'num_subscribers',ascending = False)

data_python_free[['title', 'num_subscribers','num_reviews', 'rating','published_time']] \
.set_index('title').style \
    .format("{:.2f}", subset = ['rating']) \
    .set_caption('Free Python courses') \
    .set_properties(padding="15px", border='2px solid silver', width='150px')
comment_words = ''
stopwords = set(STOPWORDS)
for val in data['title']:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 700, height = 700,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

plt.figure(figsize = (7, 7), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
comment_words = ''
stopwords = set(STOPWORDS)
for val in data[data.date > '2019-12']['title']:
    val = str(val)
    tokens = val.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
    comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 700, height = 700,
                background_color ='white',
                stopwords = stopwords,
                min_font_size = 10).generate(comment_words)

plt.figure(figsize = (7, 7), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.title('Couses that was created in 2020', fontsize = 15)
plt.show()