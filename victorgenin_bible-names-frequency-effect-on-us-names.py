import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import RegexpTokenizer

HIGH_CORR_THRESHOLD = 0.85
TOP_BIBLE_NAMES = 20
prefix = ''
bibleNamesURL = prefix + 'People.csv'
newTestamentURL = prefix + 'newTestament.txt'
oldTestamentURL = prefix + 'oldTestament.txt'

prefix = '../input/'
stateNamesURL = prefix + 'StateNames.csv'
nationalNamesURL = prefix + 'NationalNames.csv'

nationalNamesDS = pd.read_csv(nationalNamesURL)
stateNamesDS = pd.read_csv(stateNamesURL)

try:
    bibleNamesDS = pd.read_csv(bibleNamesURL)
    # retrieve all words starting with capital letter and having atleast length of 3
    tokenizer = RegexpTokenizer("[A-Z][a-z]{2,}")
    # load new testament
    file = open(newTestamentURL)
    bibleData = file.read()
    file.close()
    newTestamentWordsCount = pd.DataFrame(tokenizer.tokenize(bibleData)).apply(pd.value_counts)

    # load old testament
    file = open(oldTestamentURL)
    bibleData = file.read()
    file.close()
    oldTestamentWordsCount = pd.DataFrame(tokenizer.tokenize(bibleData)).apply(pd.value_counts)
    bibleData = None
except:
    pass
try:
    # remove irrelevant columns
    stateNamesDS.drop(['Id', 'Gender'], axis=1, inplace=True)
    nationalNamesDS.drop(['Id', 'Gender'], axis=1, inplace=True)

    # retrieve unique names count of each testament
    bibleNames = pd.Series(bibleNamesDS['Name'].unique())
    # filtering out Bible names
    newTestamentNamesCount = pd.merge(newTestamentWordsCount, pd.DataFrame(bibleNames), right_on=0, left_index=True)
    newTestamentNamesCount = newTestamentNamesCount.ix[:, 0:2]
    newTestamentNamesCount.columns = ['Name', 'BibleCount']

    oldTestamentNamesCount = pd.merge(oldTestamentWordsCount, pd.DataFrame(bibleNames), right_on=0, left_index=True)
    oldTestamentNamesCount = oldTestamentNamesCount.ix[:, 0:2]
    oldTestamentNamesCount.columns = ['Name', 'BibleCount']
except:
    pass
try:
    # plot top TOP_BIBLE_NAMES old testament names
    topOldTestamentNamesCount = oldTestamentNamesCount.sort_values('BibleCount', ascending=False).head(TOP_BIBLE_NAMES)
    topOldTestamentNamesCount.plot(kind='bar', x='Name', legend=False, title='Old Testament names count')

    # remove God/Israel
    oldTestamentNamesCount = oldTestamentNamesCount.drop(oldTestamentNamesCount[(oldTestamentNamesCount.Name == 'God') | (oldTestamentNamesCount.Name == 'Israel')].index)
except:
    pass
try:
    # plot top TOP_BIBLE_NAMES new testament names
    topNewTestamentNamesCount = newTestamentNamesCount.sort_values('BibleCount', ascending=False).head(TOP_BIBLE_NAMES)
    topNewTestamentNamesCount.plot(kind='bar', x='Name', legend=False, title='New Testament names count')

    # remove God/Jesus
    newTestamentNamesCount = newTestamentNamesCount.drop(newTestamentNamesCount[(newTestamentNamesCount.Name == 'God') | (newTestamentNamesCount.Name == 'Jesus')].index)
except:
    pass
try:
    # get state data of new testament names
    newTestamentStateNamesCount = pd.merge(newTestamentNamesCount, stateNamesDS, right_on='Name', left_on='Name')

    # get state data of old testament names
    oldTestamentStateNamesCount = pd.merge(oldTestamentNamesCount, stateNamesDS, right_on='Name', left_on='Name')

    # remove name column
    newTestamentStateNamesCount = newTestamentStateNamesCount.ix[:, 1:5]
    oldTestamentStateNamesCount = oldTestamentStateNamesCount.ix[:, 1:5]
except:
    pass
# scale and calculate plot states with high corr
def plotStateCorr(stateNamesCount, title):
    stateNamesCount[['Count','BibleCount']] = stateNamesCount[['Count','BibleCount']].apply(lambda x: MinMaxScaler().fit_transform(x))
    stateNamesCount = stateNamesCount.groupby(['Year', 'State']).corr()
    stateNamesCount = stateNamesCount[::2]
    highCorrStateNamesCount = stateNamesCount[stateNamesCount.Count > HIGH_CORR_THRESHOLD]
    highCorrStateNamesCount.drop(['BibleCount'], axis=1, inplace=True)
    highCorrStateNamesCount = highCorrStateNamesCount.unstack()
    highCorrStateNamesCount = highCorrStateNamesCount.reset_index()
    fg = sns.FacetGrid(data=highCorrStateNamesCount, hue='State', size=5)
    fg.map(pyplot.scatter, 'Year', 'Count').add_legend().set_axis_labels('Year', 'Correlation coefficient')
    sns.plt.title(title)

try:
    plotStateCorr(newTestamentStateNamesCount, 'Correlation of New Testament and US state names')
    plotStateCorr(oldTestamentStateNamesCount, 'Correlation of Old Testament and US state names')
    oldTestamentStateNamesCount = None
    newTestamentStateNamesCount = None
    stateNamesDS = None
except:
    pass
try:
    # get national data of new testament names
    newTestamentNationalNamesCount = pd.merge(newTestamentNamesCount, nationalNamesDS, right_on='Name', left_on='Name')

    # get national data of old testament names
    oldTestamentNationalNamesCount = pd.merge(oldTestamentNamesCount, nationalNamesDS, right_on='Name', left_on='Name')

    # remove name column
    newTestamentNationalNamesCount = newTestamentNationalNamesCount.ix[:, 1:4]
    oldTestamentNationalNamesCount = oldTestamentNationalNamesCount.ix[:, 1:4]
except:
    pass
# scale and calculate plot states with high corr
def plotNationalCorr(nationalNamesCount, title):
    nationalNamesCount[['Count','BibleCount']] = nationalNamesCount[['Count','BibleCount']].apply(lambda x: MinMaxScaler().fit_transform(x))
    nationalNamesCount = nationalNamesCount.groupby('Year').corr()
    nationalNamesCount = nationalNamesCount[::2]
    nationalNamesCount.unstack().plot(kind='line', y='Count', legend=False, title=title)

try:    
    plotNationalCorr(newTestamentNationalNamesCount, 'New Testament national correlation')
    plotNationalCorr(oldTestamentNationalNamesCount, 'Old Testament national correlation')
except:
    pass