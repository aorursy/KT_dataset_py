import pandas as pd

import matplotlib.pyplot as plt

from scipy import stats

cereal = pd.read_csv('../input/80-cereals/cereal.csv')
cereal.describe(include='all')
def createhist (df,row,col,feature):

    fig,ax = plt.subplots(figsize = (10,10))

    temp = row*col

    idx = list(range(1,temp+1))

    features = feature

    for feature,idx in zip (feature,idx):

        ax = plt.subplot (row,col,idx)

        ax.hist(cereal[feature])

        ax.set_title(feature)
feature = ['calories','sodium','fiber','carbo','sugars','potass']

createhist (cereal,3,2,feature)
ax,figsize = plt.subplots(figsize=(10,5))

ax1 = plt.subplot(121)

ax2 = plt.subplot(122)

ax1.hist(cereal.sugars)

ax1.set_title('Sugars')

stats.probplot(cereal.sugars, dist='norm',fit=True, plot=plt)

len(cereal.sugars)
stats.normaltest(cereal.sugars)
from scipy.stats import ttest_ind

ttest_ind(cereal.calories, cereal.protein)
import seaborn as sns

ax=sns.countplot(cereal.type)

ax.set_title('Cereal Type')
ax=sns.countplot(cereal.mfr)
import scipy.stats

chisquare(cereal.type.value_counts())
chisquare(cereal.mfr.value_counts())
contingencyTable = pd.crosstab(cereal.type, cereal.mfr)
scipy.stats.chi2_contingency(contingencyTable)