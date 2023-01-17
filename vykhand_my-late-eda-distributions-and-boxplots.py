import pandas as pd

import numpy as np

from scipy.stats import norm

from scipy.stats import beta

from scipy.stats import chi2

from scipy.stats import gamma

from scipy.stats import triang



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#weight functions https://www.kaggle.com/zhehaoliu/santas-uncertain-bags/over-weight-probability-function

#and https://www.kaggle.com/cpmpml/santas-uncertain-bags/optimal-expected-submission-value

#weight functions

def get_weight(t):

    if t == 'horse':

        return max(0, np.random.normal(5,2,1)[0])

    elif t == 'ball':

        return max(0, 1 + np.random.normal(1,0.3,1)[0])

    elif t == 'bike':

        return max(0, np.random.normal(20,10,1)[0])

    elif t == 'train':

        return max(0, np.random.normal(10,5,1)[0])

    elif t == 'coal':

        return 47 * np.random.beta(0.5,0.5,1)[0]

    elif t == 'book':

        return np.random.chisquare(2,1)[0]

    elif t == 'doll':

        return np.random.gamma(5,1,1)[0]

    elif t == 'blocks':

        return np.random.triangular(5,10,20,1)[0]

    elif t == 'gloves':

        return 3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]

    else:

        raise ValueError('Not a valid gift type!')

        

#calculating the probability of a single gift over certain weight

def over_weight_prob(t, weight):

    if t == 'horse':

        return norm.cdf(weight, 5, 2)

    elif t == 'ball':

        return norm.cdf(weight, 1, 0.3)

    elif t == 'bike':

        return norm.cdf(weight, 20,10)

    elif t == 'train':

        return norm.cdf(weight, 10, 5)

    elif t == 'coal':

        return beta.cdf(weight/47, 0.5, 0.5)

    elif t == 'book':

        return chi2.cdf(weight, 2)

    elif t == 'doll':

        return gamma.cdf(weight,5,1)

    elif t == 'blocks':

        return triang.cdf(weight,c = 1.0/3, loc = 5, scale = 15)

    elif t == 'gloves':

        random_series = pd.Series(np.random.rand(5000)).apply(lambda x: x+3 if x < 0.3 else x)

        return 1.0 * sum(random_series > weight ) / len(random_series)

    else:

        raise ValueError('Not a valid gift type!')



def gift_distributions(gift, ngift, n=1):

    if ngift == 0:

        return np.array([0.0])

    

    #np.random.seed(100)

    

    if gift == "horse":

        dist = np.maximum(0, np.random.normal(5,2,(n, ngift)))

    if gift == "ball":

        dist = np.maximum(0, 1 + np.random.normal(1,0.3,(n, ngift)))

    if gift == "bike":

        dist = np.maximum(0, np.random.normal(20,10,(n, ngift)))

    if gift == "train":

        dist = np.maximum(0, np.random.normal(10,5,(n, ngift)))

    if gift == "coal":

        dist = 47 * np.random.beta(0.5,0.5,(n, ngift))

    if gift == "book":

        dist = np.random.chisquare(2,(n, ngift))

    if gift == "doll":

        dist = np.random.gamma(5,1,(n, ngift))

    if gift == "blocks":

        dist = np.random.triangular(5,10,20,(n, ngift))

    if gift == "gloves":

        gloves1 = 3.0 + np.random.rand(n, ngift)

        gloves2 = np.random.rand(n, ngift)

        gloves3 = np.random.rand(n, ngift)

        dist = np.where(gloves2 < 0.3, gloves1, gloves3)

    for j in range(1, ngift):

        dist[:,j] += dist[:,j-1]

    return dist

        
get_weight("coal")
gift_distributions("coal",1,10)
df = pd.read_csv('../input/gifts.csv')

df.head()
df['typ'] = df.GiftId.str.split("_").map(lambda x: x[0]).astype("category")

df['n'] = df.GiftId.str.split("_").map(lambda x: x[1]).astype('int')

#df.drop("GiftId", axis = 1, inplace=True)
df.head()
df.groupby('typ')['n'].agg(["min", "max", "count"])
df.shape
def plot_pdf(t, ax,  n = 100000):

    dt = [get_weight(t) for i in range(n)]

    ax.set_xlim(0,50)

    #ax.set_ylim(0, 1.4)

    return sns.distplot(dt,bins=100, ax = ax)
nrows = df.typ.nunique() # 9

fig, axs = plt.subplots(nrows = nrows, ncols = 1, figsize =(8,30))

#fig.set_figheight(30)

#fig.set_figwidth(15)



for i,t in enumerate(df.typ.unique()):

    ax  = plot_pdf(t, ax = axs[i])

    ax.set_title(t + " distribution")
typs = df.typ.unique()

fig = plt.figure(figsize=(8,10))

n=100000

dt = pd.DataFrame()

for t in typs:

    dt = pd.concat((dt, pd.Series([get_weight(t) for i in range(10000)], name=t)), axis=1)

sns.boxplot(data = dt)
nsim = 10000

weights = []

for i in range(nsim):

    weight = (gift_distributions("ball", 1, 1100).sum() +

         gift_distributions("bike",1,500).sum() +

         gift_distributions("blocks",1,1000).sum()  +

         gift_distributions("book", 1, 1200).sum() +

         gift_distributions("coal",1,166).sum() +

         gift_distributions("doll", 1,1000).sum() +

         gift_distributions("gloves",1,200).sum() +

         gift_distributions("horse",1,1000).sum() +

         gift_distributions("train", 1, 1000).sum()

         )

    weights.append(weight)

sns.distplot(weights)