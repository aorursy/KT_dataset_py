import numpy as np
import pandas as pd
import seaborn as sns
app = pd.read_csv("../input/AppleStore.csv")
app.head()
app = app.drop("Unnamed: 0" ,axis = 1)
app.columns.values

### Classification of genres.
a = app['prime_genre'].value_counts()
print(a)
import matplotlib.pyplot as plt

##Since there are a lot of categories we consider only the top 5 and club the remaining into others.
### Clubbing the categories in the variable prime genre.
b = app.prime_genre.value_counts().index[:5]
def genre(x):
    if x in b:
        return x
    else:
        return "others"
app["genre_new"]=app.prime_genre.apply(lambda x: genre(x))
app.head()

### New variable paid or unpaid apps
def payment(x):
    if x == 0:
        return "free"
    else:
        return "paid"
app["payment"]=app.price.apply(lambda x: payment(x))
# app.head()

##distribution of free and paid apps
paid_free = app['payment'].value_counts()
paid_free.plot.bar()
##What is the average price of each genre? / Which genres have high prices
###genre of the app with the corresponding price range
gen_price = app.loc[:,"price":"rating_count_tot"]
gen_price["genre_new"] = app["genre_new"]
gen_price = gen_price.drop(columns = "rating_count_tot")
gen_price.head()
q = gen_price.groupby(by = gen_price['genre_new']).mean()
q
genre = q.index
q["genre"] = genre
## bar chart of genre_new and price
sns.barplot(x = "genre", y = "price", data = q)

#The average price of education apps is more than all other genres
###rating of the app and size of the app
##Does the rating of the app depend on the size of the app, with respect to genres?

#Converting size_bytes from bytes to MB
app['MB']= app.size_bytes.apply(lambda x : x/1048576)

sns.lmplot(x = "user_rating", y = "MB",col='genre_new', data = app,col_wrap=2,hue = "genre_new")
plt.show()


##Removing the outlier for price
#Since there were only 7 apps who had prices greater than 50, the extreme values affected the overall data. Thus these outliers had to be removed

### price and size of the app 
appdata = app[((app.price<50) & (app.price>0))]
sns.lmplot(x = "price", y = "MB",col='genre_new', data = appdata,col_wrap=2,hue = "genre_new", fit_reg=False)
plt.show()

###Do costly apps have higher rating?

sns.lmplot(x = "price", y = "user_rating",col='genre_new', data = appdata,col_wrap=2,hue = "genre_new", fit_reg=False)

###Paid and unpaid apps rating
# rating_paid = app.loc[:,"user_rating":"user_rating_ver"]
# rating_paid["payment"] = app["payment"]
# rating_paid = rating_paid.drop(columns = "user_rating_ver")
# rating_paid.head()
#report stating number of free apps and paid apps in each genre
sub_data=app.genre_new.value_counts()
sub_data=sub_data.to_frame()
type(sub_data)
p=pd.crosstab(index=app['genre_new'],columns=app['payment'])
sub_data2=pd.concat([sub_data,p],axis=1,sort=False)
sub_data2.columns=['Total','Free','Paid']
sub_data2

a=[]
for i in range(0,6):
    a.append(sub_data2['Free'][i])
    a.append(sub_data2['Paid'][i])
    
# sub_data2=app.payment.value_counts()
# sub_data2=sub_data2.to_frame()
# sub_data2
# Libraries
import matplotlib.pyplot as plt

# First Ring (outside)
fig, ax = plt.subplots()
ax.axis('equal')
mypie, _ = ax.pie(sub_data.values, radius=2.5, labels=sub_data.index )
plt.setp( mypie, width=0.8, edgecolor='white')
sub= ['#45cea2','#fdd470']
# mypie2, _ = ax.pie(sub_data2.values, labels=sub_data2.index, labeldistance=0.7)
# plt.margins(0,0)
mypie2, _ = ax.pie(a, radius=2.5-0.8, labels=6*['Free','Paid'], labeldistance=0.7,colors=6*sub)
plt.setp( mypie2, width=0.4, edgecolor='white')
plt.margins(0,0)
plt.show()
#The following graph represents the distribution of free and paid apps in each genre
#grouping the apps according to the genre
d=app.prime_genre.value_counts()
df1 = pd.DataFrame(data=d.index, columns=['genre'])
df2 = pd.DataFrame(data=d.values, columns=['number_of_apps'])
df = pd.merge(df1, df2, left_index=True, right_index=True)
df
plot=sns.barplot(x=df.genre,y=df.number_of_apps)
plot.set_xticklabels(df['genre'], rotation=90, ha="center")
plot

#Among all the genres, the game genre has the large number of apps
#heatmap 
import seaborn as sns
appdata=appdata[((app.price<50) & (app.price>0))]
appdata.head()
df = appdata.iloc[:,[4,5,7,12,13,14,18]]
#Correlation Matrix
corr = df.corr()
corr = (corr)
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            cmap="Spectral", center=0)
plt.show()

#from the heatmap, it can be seen that the price,size and rating of the apps are not related to each other 
# Getting the mean user ratings for the different App genre categories
mean_user_rating=appdata.groupby('genre_new')['user_rating'].mean().reset_index().sort_values(by=['user_rating'])
mean_user_rating
#The mean user rating of all the genre are approximately same
appdata['cont_rating']=appdata['cont_rating'].str.replace("+","")
appdata['cont_rating']=appdata['cont_rating'].astype(float)

# Getting the mean user ratings for the different App genre categories

mean_content_rating=appdata.groupby('genre_new')['cont_rating'].mean().reset_index().sort_values(by=['cont_rating'])
mean_content_rating
sns.barplot(x=mean_content_rating['genre_new'],y=mean_content_rating['cont_rating'])
#Games genre has high content rating than any other genre
