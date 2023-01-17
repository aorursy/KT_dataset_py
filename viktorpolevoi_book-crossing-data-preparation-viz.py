import pandas as pd
!pip install --quiet pycountry_convert
from pycountry_convert import country_alpha2_to_country_name, country_name_to_country_alpha3, country_alpha3_to_country_alpha2
import warnings
warnings.filterwarnings('ignore')
import plotly
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import plotly.offline as pyo
pyo.init_notebook_mode()
users = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Users.csv', sep=';', encoding='latin-1', low_memory=False)
books_r = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Book-Ratings.csv', sep=';', encoding='latin-1', low_memory=False)
b_cols = ['ISBN', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']
books = pd.read_csv('../input/bookcrossing-dataset/Book reviews/BX-Books.csv', sep=';', names=b_cols, encoding='latin-1',low_memory=False)
users[:5]
books_r[:5]
books[:5]
def get_alpha3(col):
    try:
        iso_3 =  country_name_to_country_alpha3(col, cn_name_format="lower")
    except:
        iso_3 = 'Unknown'
    return iso_3

def get_name(col):
    try:
        name =  country_alpha2_to_country_name(country_alpha3_to_country_alpha2(col))
    except:
        name = 'Unknown'
    return name
users['country'] = [x.split(', ')[-1] for x in users.Location]
users['alpha_3'] = users['country'].apply(lambda x: get_alpha3(x))
users['alpha_3'].loc[users.country == 'usa'] = 'USA'
users['country'] = users['alpha_3'].apply(lambda x: get_name(x))
users[:5]
books.drop(['img_s', 'img_m', 'img_l'], axis=1, inplace=True)
books.drop(books.index[0], inplace=True)
data = pd.merge(users, books_r, on='User-ID')
data = pd.merge(data, books, on='ISBN')
data[:5]
users_map = users.groupby(['country', 'alpha_3'])['alpha_3'].agg(Users='count').reset_index()
users_map['Users(log)'] = np.log10(users_map["Users"])

fig = px.choropleth(users_map, locations="alpha_3",
                    color='Users(log)',
                    hover_name="country",
                    hover_data=["Users"],
                    color_continuous_scale='Cividis')
fig.update_layout(title_text="Unique Users")
fig.update_layout(coloraxis_colorbar=dict(title='Users', tickprefix='1.e'))
fig.show()
data_map = data.groupby(['country', 'alpha_3'])['alpha_3'].agg(Users_activity='count').reset_index()
data_map['Users_activity(log)'] = np.log10(data_map["Users_activity"])

fig = px.choropleth(data_map, locations="alpha_3",
                    color='Users_activity(log)',
                    hover_name="country",
                    hover_data=["Users_activity"],
                    color_continuous_scale='Cividis')
fig.update_layout(title_text="Users activity")
fig.update_layout(coloraxis_colorbar=dict(title='Users activity', tickprefix='1.e'))
fig.show()
data_bar = pd.merge(users_map, data_map, on='country')

x = ['Unique Users', 'Users_activity']
y = [data_bar.Users.sum(), data_bar.Users_activity.sum()]

fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            text=y,
            textposition='auto',
        )])
fig.update_layout(title_text='Users/Users activity')
fig.show()
data_bar = data_bar[data_bar.country != 'Unknown']
data_bar = data_bar.sort_values(by = 'Users',ascending = False)[:15]

fig = go.Figure(data=[
    go.Bar(name='Users', x=data_bar.country, y=data_bar.Users),
    go.Bar(name='Users activity', x=data_bar.country, y=data_bar.Users_activity)
])

fig.update_layout(barmode='group', title_text='Users/Users activity per country (top 15)')
fig.show()
data['Age'].loc[data.Age >= 100] = np.NaN

fig = px.histogram(data, x="Age", nbins=10, title='Histogram of Age')
fig.show()
fig = px.histogram(data, x="Book-Rating", nbins=10, title='Histogram of Rating')
fig.show()
data_barh = data.book_title.value_counts().reset_index()[:10]

fig = go.Figure(go.Bar(
            x=data_barh.book_title,
            y=data_barh['index'],
            orientation='h'))
fig.update_layout(yaxis=dict(autorange="reversed"), title='Most voted books (top 10)')

fig.show()
print('Most voted books per country (top 15)')
data[data.country.isin(data_bar.country)].groupby(['country'])['book_title'].agg(lambda x:x.value_counts().index[0]).reset_index()
data_barh = data.book_author.value_counts().reset_index()[:10]

fig = go.Figure(go.Bar(
            x=data_barh.book_author,
            y=data_barh['index'],
            orientation='h'))
fig.update_layout(yaxis=dict(autorange="reversed"), title='Most voted authors (top 10)')

fig.show()
print('Most voted authors per country (top 15)')
data[data.country.isin(data_bar.country)].groupby(['country'])['book_author'].agg(lambda x:x.value_counts().index[0]).reset_index()