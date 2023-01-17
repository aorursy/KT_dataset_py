import pandas as pd
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
from wordcloud import WordCloud, STOPWORDS 
import matplotlib.pyplot as plt
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('../input/windows-store/msft.csv')
data
data.isnull().sum()
data.dropna(inplace=True)
data.Price[data["Price"] == "Free"] = 0
data["Price"] = data["Price"].str.replace("₹ ", "")
data["Price"] = data["Price"].str.replace(",","")
data["Price"].fillna(0, inplace=True)
data['Price'] = pd.to_numeric(data['Price'])
data['Date'] = pd.to_datetime(data['Date'])
data['year'] = pd.DatetimeIndex(data['Date']).year
data.info()
data_category_cnt = data.Category.value_counts()
fig = px.pie(data_category_cnt, values=data_category_cnt, names=data_category_cnt.index,
            title="Distribution of Categories")
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
data_free_cnt = data.Category[data.Price == 0].value_counts()
data_pay_cnt = data.Category[data.Price != 0].value_counts()
fig = go.Figure(data=[
    go.Bar(name='Free', x=data_free_cnt.index, y=data_free_cnt),
    go.Bar(name='Paid', x=data_pay_cnt.index, y=data_pay_cnt)
])
fig.update_traces(texttemplate='%{value}', textposition='outside')
fig.update_layout(barmode='group', hovermode='x', title_text='Free/Paid Apps')
fig.show()
cm = sns.light_palette("green", as_cmap=True)
table = data.groupby('Category').agg({'Rating': 'mean', 'No of people Rated': 'sum'})
table.columns = ['Rating (mean)', 'No of people Rated (sum)']
table.style.background_gradient(cmap=cm).format({"Rating (mean)": lambda x: '{:,.2f}'.format(x)})
table = pd.pivot_table(data, values='Name', index='Category',
                    columns=['year'], aggfunc='count', fill_value=0)
table.style.background_gradient(cmap=cm)
table_2 = data.groupby('year').agg({'Rating': 'mean', 'No of people Rated': 'sum'})
fig = make_subplots(specs=[[{"secondary_y": True}]])
trace1 = go.Bar(x=table_2.index, y=table_2['No of people Rated'],
                name='No of people Rated (sum)',
                marker=dict(color= '#3AA03A',
                            line= dict(width= 1)))
trace2 = go.Scatter(x=table_2.index, y=table_2['Rating'].apply(lambda x: '{:,.2f}'.format(x)),
                    marker= dict(line= dict(width= 1), size= 8),
                    line=dict(color= '#636EFA', width= 1.5),
                    name= 'Rating (mean)')
fig.add_trace(trace1, secondary_y=False)
fig.add_trace(trace2, secondary_y=True)
fig.update_layout(hovermode='x', xaxis = dict(tickmode = 'linear'))
fig.update_yaxes(title= 'No of people Rated (sum)', secondary_y=False);
fig.update_yaxes(showgrid= False, title= 'Rating (mean)', secondary_y=True)
comment_words = ''
stopwords = set(STOPWORDS)
for val in data['Name']:
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