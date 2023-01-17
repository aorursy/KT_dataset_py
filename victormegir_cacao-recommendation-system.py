import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import pearsonr

from scipy.stats import norm



plt.rcParams['figure.figsize'] = [10, 10]

plt.rcParams.update({'font.size': 16})
cacao = pd.read_csv('../input/chocolate-bar-ratings/flavors_of_cacao.csv')
cacao.columns = ['Company', 'Name', 'Ref', 'Review Date', 'Cocoa percent', 'Country', 'Rating', 'Bean Type', 'Bean Origin']
sns.scatterplot(cacao['Cocoa percent'].apply(lambda x: float(x.split('%')[0])), cacao['Rating']);
pearsonr(cacao['Cocoa percent'].apply(lambda x: float(x.split('%')[0])), cacao['Rating'])
countries = cacao.groupby('Country').agg({

    'Rating': 'mean',

    'Name': 'count'

})
countries.sort_values(by='Rating', ascending=False).head(10)
countries.sort_values(by='Name', ascending=False).head(10)
features = cacao.copy(deep=True)
features['Cocoa percent'] = features['Cocoa percent'].apply(lambda x: float(x.split('%')[0]))



all_companies = list(features['Company'].unique())

features['Company'] = features['Company'].apply(lambda x: all_companies.index(x))



all_countries = list(features['Country'].unique())

features['Country'] = features['Country'].apply(lambda x: all_countries.index(x))



all_beans = list(features['Bean Origin'].unique())

features['Bean Origin'] = features['Bean Origin'].apply(lambda x: all_beans.index(x))
features.head()
features_as_array = features.drop(columns=['Name', 'Bean Type']).to_numpy()

Similarities = cosine_similarity(features_as_array, features_as_array)
Similarities.shape
def get_index(cacao_name):

    return features.loc[features['Name'] == cacao_name].index[0]



print('Index of cacao flavor: {}'.format(get_index('Atsane')))
def recommend(cacao_name, N=10):

    # Get the index of the chocolate

    index = get_index(cacao_name)

    

    # Put all the similarity scores in a list with their index

    sim_scores = list(enumerate(Similarities[index]))

    

    # Sort the list by the similarity score

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    

    # Get the first N items

    sim_scores = sim_scores[1:N+1]

    

    # Put their indices in a list

    cacao_indeces = [x[0] for x in sim_scores]

    

    # Return the chocolate names

    return list(features['Name'].iloc[cacao_indeces])
recommend('Atsane')