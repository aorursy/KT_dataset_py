import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

from datetime import datetime


%matplotlib inline 
data = pd.read_csv('../input/youtube-new/INvideos.csv')
# Peek at the data 
data.head(5)

# Check the dimension of our data , dtypes etc...

print('Number of observations in hand : {}'.format( data.shape[0] ))
print('Number of columns : {}'.format( data.shape[1] )) 

print('Dtypes \n' , data.dtypes)
# statistical summary 

data.describe( include = 'all' ).T
# missing value check

print(data.isnull().sum() )
print('Percentage share of Nullity in description  : ' , (561 / data.shape[0])*100 ,'%'  )


sns.heatmap( data.isnull() , cmap = 'viridis' , yticklabels= False , cbar = True )

data['category_id'].value_counts()
#since description and title almost matches in context , i would rather remove the description column .
#It already has great deal of null values , its better we drop it . No harm done :-)

Columns2Delete = [ 'thumbnail_link' , 'description' , 'tags'   ]
data4EDA = data.drop( labels = Columns2Delete , axis = 1  )
data4EDA.head()
data4EDA['publish_time'] = pd.to_datetime( data4EDA.publish_time , format= '%Y-%m-%dT%H:%M:%S' )
print( data4EDA.publish_time.dtypes )
data4EDA.head()
like_bins = [-1, 100000, 500000, 1000000, 5000000]
view_bins = [-1, 300000, 5000000, 10000000, 500000000]
dislike_bins = [-1, 100000, 500000, 1000000, 5000000]
comment_bins = [-1, 10000, 50000, 500000 , 1000000]


data4EDA['like_BandWidth'] = pd.cut( data4EDA.likes  , labels= ['Poor','Improving', 'Good', 'Very Good'] , bins= like_bins )
data4EDA['view_BandWidth'] = pd.cut( data4EDA.views  , labels= ['Poor','Improving', 'Good', 'Very Good'] , bins= view_bins )
data4EDA['dislikes_BandWidth'] = pd.cut( data4EDA.dislikes  , labels= ['Normal','Critical', 'Bad', 'Worse'] , bins= dislike_bins )
data4EDA['comment_BandWidth'] = pd.cut( data4EDA.comment_count  , labels= ['Poor','Improving', 'Good', 'Very Good'] , bins= comment_bins ) 


data4EDA['comment_BandWidth'] = data4EDA['comment_BandWidth'].astype('object') 
data4EDA.loc[ data4EDA['comment_BandWidth'].isna() , 'comment_BandWidth' ] = data4EDA.loc[ data4EDA['comment_BandWidth'].isna() , 'comment_BandWidth' ].astype('object').replace( np.NaN , 'Disabled' )
data4EDA.head()
# To check out what content is most avail in our dataset .
data4EDA['title'].value_counts()
data4EDA['video_id'].value_counts()
data4EDA['like_BandWidth'] = data4EDA['like_BandWidth'].astype('object') 
data4EDA['view_BandWidth'] = data4EDA['view_BandWidth'].astype('object') 
data4EDA['dislikes_BandWidth'] = data4EDA['dislikes_BandWidth'].astype('object') 

data4EDA.dtypes
data4EDA.isnull().sum()
sns.heatmap( data4EDA.corr() , annot = True )
# distribution of views 
plt.figure( figsize= (30,10) )
sns.kdeplot( data = data4EDA.views , label = 'views' , shade = True ) 

# sclale = 10 crore 
plt.figure( figsize= (30,10) )
sns.kdeplot( data = data4EDA.likes , label = 'likes in million' , shade = True  ) 
sns.kdeplot( data = data4EDA.dislikes , label = 'dislikes in million' , shade = True  ) 

#scale : 1 million
plt.figure( figsize= (30,10) )
sns.kdeplot( data = data4EDA.views , label = 'views' , shade = True ,  ) 
sns.kdeplot( data = data4EDA.likes , label = 'likes in million' , shade = True  ) 
sns.kdeplot( data = data4EDA.dislikes , label = 'dislikes in million' , shade = True  ) 

plt.xlim( ( 0 , 3e6) )
plt.ylim( ( 0 , 20e-8 ) )
# like_bins = [0, 100000, 500000, 1000000, 5000000] : { 'for reference' }
# labels= ['Poor','Improving', 'Good', 'Very Good']

px.box( data_frame= data4EDA , x = 'like_BandWidth' , y = 'views'  , color = 'video_error_or_removed' )
#dislike_bins = [0, 100000, 500000, 1000000, 5000000]
#labels= ['Normal','Critical', 'Bad', 'Worse']

px.box( data_frame= data4EDA , x = 'dislikes_BandWidth' , y = 'views' , color = 'video_error_or_removed' )
# Double click of the legends to isolate the plots you wish 

px.line( data_frame= data4EDA , x = 'trending_date' , y = 'views' , color= 'category_id' )
px.line( data_frame= data4EDA , x = 'publish_time' , y = 'views' , color= 'category_id' )
from wordcloud import WordCloud, STOPWORDS
text = data4EDA.title.values
wordcloud = WordCloud(
    width = 3000,
    height = 2000,
    background_color = 'black',
    stopwords = STOPWORDS).generate(str(text))
fig = plt.figure( figsize = (30, 20),facecolor = 'k', edgecolor = 'k')
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()
px.violin( data_frame = data4EDA , y = 'views' ) 
ColumnsWeCareAbout = ['title','view_BandWidth','like_BandWidth','dislikes_BandWidth','comment_BandWidth'] 
data4EDA[ColumnsWeCareAbout]
pd.crosstab( index = [ data4EDA.like_BandWidth , data4EDA.dislikes_BandWidth , data4EDA.comment_BandWidth ] , 
           columns = data4EDA.view_BandWidth ).style.background_gradient(cmap='summer_r')
pd.crosstab( index = [ data4EDA.like_BandWidth , data4EDA.comments_disabled , data4EDA.ratings_disabled ] , 
           columns = data4EDA.view_BandWidth ).style.background_gradient(cmap='summer_r')