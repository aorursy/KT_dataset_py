import pandas as pd
import numpy as np
rat=pd.read_csv('../input/movieratings/movrat.csv')
rat.head()
mov=pd.read_csv('../input/movies/movies.csv')
mov.head()
rat.isnull().sum()
def euclid(x,y):
    l=len(x)
    s=0
    for i in range(0,l):
        s=s+(x[i]-y[i])**2
    return np.sqrt(s)
def similiarity_users(o,t):
    id1=rat[rat.userId==o].movieId
    id2=rat[rat.userId==t].movieId
    idf=set(id1).intersection(set(id2))
    np.seterr(divide='ignore', invalid='ignore')
    if len(idf)!=0:
        r1=rat[(rat.userId==o) & (rat.movieId.isin(idf))].rating
        r2=rat[(rat.userId==t) & (rat.movieId.isin(idf))].rating
        return np.corrcoef(r1,r2)[0,1]
        #return euclid(r1,r2)
    else:
        return -2
def sim_keys(u):
    scores={}
    cor=0
    for oth in rat.userId.unique():
        if oth!=u:
            cor=similiarity_users(u,oth)
            if cor>0.80:
                scores[oth]=cor
    return scores.keys()
def recommend_user(u):
    ixs=sim_keys(u)
    ids={}
    uid=rat[rat.userId==u].movieId
    for i in ixs:
        idp=rat[rat.userId==i].movieId
        ids[i]=list(set(idp)-set(uid))
    inter={}
    for k,v in ids.items():
        for idk in v:
            if idk not in inter:
                inter[idk]=1
            else:
                inter[idk]+=1
    ints=inter.copy()
    ints=[(v,k) for k,v in ints.items()]
    ints=sorted(ints)
    ts=[kl[1] for kl in ints]
    ts.reverse()
    print("USERS LIKE YOU ALSO WATCHED:")
    print(' ')
   # print(set(ts))
    tit=list(mov[mov.movieId.isin(ts[0:50])]['title'])
    for name in tit:
        print(name)

def topMatches(person,n=5):
    scores={}
    for other in rat.userId.unique():
        #print(other)
        if other!=person:
            scores[other]=similiarity_users(person,other)
    return scores
#recommendations for user 2 by user-based collaborative filtering
recommend_user(2)
#recommendations for user 777 by user-based collaborative filtering
recommend_user(777)
def similiarity_movies(m1,m2):
    #r1,r2=[],[]
    df1=rat[(rat.movieId==m1)]
    df2=rat[(rat.movieId==m2)]
    ids=set(df1.userId).intersection(df2.userId)
    if len(ids)!=0:
        r1=df1[df1.userId.isin(ids)].rating
        r2=df2[df2.userId.isin(ids)].rating
        return np.corrcoef(r1,r2)[0,1]
    else:
        return -2
def recommend_movies(m):
    rats={}
    for mo in rat.movieId.unique():
        if mo!=m:
            cor=similiarity_movies(m,mo)
            if cor>0.80:
                rats[mo]=cor
    ints=rats.copy()
    ints=[(v,k) for k,v in ints.items()]
    ints=sorted(ints)
    ts=[kl[1] for kl in ints]
    ts.reverse()
    print("YOU MAY ALSO LIKE THESE MOVIES:")
    print(' ')
    tit=list(mov[mov.movieId.isin(ts[0:50])]['title'])
    for name in tit:
        print(name)
#when user downloads the movie it reccomends him the movies that are similiar to that movie
#this is item based collaborative filtering
recommend_movies(1320)
#based on the previous movies rated by the users
#it can predict the rating the person gives for any movie that person has not rated
def predict_rating(person,movie):
    ids=rat[rat.userId==person].movieId
    dic={}
    di=[]
    for i in sorted(ids):
        dic[i]=(similiarity_movies(movie,i))
        di.append(dic[i])
    r=rat[rat.userId==person].sort_values('movieId').rating
    r=np.array(r)
    di=np.array(di)
    r=di*r
    return sum(r)/sum(di),dic
predict_rating(100,100)