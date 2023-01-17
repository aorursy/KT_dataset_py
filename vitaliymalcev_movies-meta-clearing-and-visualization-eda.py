

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm, skew

import matplotlib.pyplot as plt

pd.options.display.max_columns = None

from wordcloud import WordCloud



df = pd.read_csv("/kaggle/input/movies-meta-data/movie_metadata.csv")
df.info()
df['color']= df['color'].fillna("Color")

df['director_name']= df['director_name'].fillna("None")

df['duration']= df['duration'].fillna(df['duration'].median())

# df['num_critic_for_reviews'] keep Nan

df['director_facebook_likes']= df['director_facebook_likes'].fillna(0)

df['actor_3_facebook_likes'] = df['actor_3_facebook_likes'].fillna(0)

df['actor_2_name']= df['actor_2_name'].fillna("None")

df['actor_1_facebook_likes'] = df['actor_1_facebook_likes'].fillna(0)

#df['gross'] keep Nan

df['actor_1_name']= df['actor_1_name'].fillna("None")

df['actor_3_name']= df['actor_3_name'].fillna("None")

df['facenumber_in_poster']= df['facenumber_in_poster'].fillna(df['facenumber_in_poster'].median())

df['plot_keywords']= df['plot_keywords'].fillna("None")

df['num_user_for_reviews'] = df['num_user_for_reviews'].fillna(0)

df['language']= df['language'].fillna("English")

df['country']= df['country'].fillna("USA")

df['content_rating']= df['content_rating'].fillna("Not Rated")

#df['budget'] keep Nan

df['title_year']= df['title_year'].fillna(df['title_year'].median()) # no idia

df['aspect_ratio'] = df['aspect_ratio'].fillna(df['aspect_ratio'].median())

df['title_year']= df['title_year'].fillna(df['title_year'].median())

df['budget']= df['budget'].fillna(df['budget'].median())

df['gross']= df['gross'].fillna(df['gross'].median())

df["title_year"] = df["title_year"].astype(int)

corrmat = df.corr()

plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=0.9, square=True)

# interesting that budget doesnt correl woth anything 
sns.distplot(df['gross'] , fit=norm);



# Get the fitted parameters used by the function

(mu, sigma) = norm.fit(df['gross'])

print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))



#Now plot the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],

            loc='best')

#QQ-plot

fig = plt.figure()

res = stats.probplot(df['gross'], plot=plt)

plt.show()
ax = sns.lineplot(x="title_year", y="budget", data=df)

ax = sns.lineplot(x="title_year", y="gross", data=df)

ax.set(xlabel='year', ylabel='money')

ax.legend(["budget","gross"])
df2 = pd.pivot_table(df, values='budget', index="country",aggfunc="sum")

df2 =  df2.sort_values("budget").tail(10)

plot = df2.plot.pie(y='budget', figsize=(5, 5), legend = False,sharey = True, label = "top10 countries by Total budget")
df2 = pd.pivot_table(df, values=['gross','budget'], index="country",aggfunc="sum")



df2["proficit"] = df2['gross']/df2['budget']

df2 =  df2.sort_values("proficit").tail(15)

sns.barplot(data = df2, x=df2.index, y="proficit")

plt.xticks(rotation=75)
def wordcoluder(col):

    words = []

    new = col.split("|")

    for i in range(len(new)):

        words.append(new[i])

    #for i in new:

        #words.append[i]

        #print[i]

    return words



df['words'] = df['plot_keywords'].apply(wordcoluder)



text = " ".join(review for review in df.plot_keywords)

print ("There are {} words in the combination of all review.".format(len(text)))



wordcloud = WordCloud( background_color="white").generate(text)



plt.figure(figsize=[15,20])

plt.title("Keywords for all movies",size= 30)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()

text = " ".join(review for review in df[(df['title_year']>=1920)&(df['title_year']<1950) ]['plot_keywords'])

wordcloud = WordCloud( background_color="white").generate(text)

plt.figure(figsize=[15,20])

plt.title("Cinema keywords in 1920-1950th",size= 30)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
text = " ".join(review for review in df[(df['title_year']>=1990)&(df['title_year']<2020) ]['plot_keywords'])



wordcloud = WordCloud( background_color="black").generate(text)



plt.figure(figsize=[15,20])

plt.title("Cinema keywords in 1990-2020",size= 30)

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()