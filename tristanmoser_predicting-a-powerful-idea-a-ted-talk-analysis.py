import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
import ast
from gensim.models import KeyedVectors
pd.options.mode.chained_assignment = None

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.linear_model import SGDRegressor, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split, GridSearchCV
data = pd.read_csv('../input/ted-talks/ted_main.csv')
data.head(1)
data['ratings'][0]
#list of all ratings
ratings = ['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Longwinded', 'Confusing',
           'Informative', 'Fascinating', 'Unconvincing', 'Persuasive', 'Jaw-dropping', 'OK',
           'Obnoxious', 'Inspiring']
rate = []
#For this function (x) denotes the rating of interest
def parse(x):
    #Loop over all talks in dataframe
    for ll in range(len(data)):
        #First we split on the rating of interest and subsequent splits work to isolate the rating counts
        splitting = data['ratings'][ll].split(x)
        splitting2 = splitting[1].split(':')
        splitting3 = splitting2[1].split(" ")
        splitting4 = splitting3[1].split("}")
        #Isolate number of ratings
        rate.append(splitting4[0])
        series = pd.Series(rate)
        #Create column in dataframe for rating
        data[x] = series
for rating in ratings:
    parse(rating)
    rate.clear()
data[ratings].head()
#Convert ratings from string to integers so we can use mathematical operations
data[ratings] = data[ratings].astype(int)
#Categorize the ratings into the three broad categories
positive = ['Funny','Beautiful','Ingenious','Courageous','Inspiring','Jaw-dropping','Fascinating']
negative = ['Longwinded','Unconvincing','Obnoxious','Confusing']
moderate = ['Informative','OK','Persuasive']
#Create new columns that sum the ratings appropriately
data['Positive'] = data['Informative'] + data['Persuasive'] + data['Funny'] + data['Beautiful'] + data['Ingenious'] + data['Courageous'] + data['Inspiring'] + data['Jaw-dropping'] + data['Fascinating']
data['Moderate'] = data['OK']
data['Negative'] = data['Longwinded'] + data['Unconvincing'] + data['Obnoxious'] + data['Confusing']
#lets look at the first 5 talks.
data[['Positive','Moderate','Negative']].head()
data[['Positive', 'Moderate', 'Negative']].describe()
data['related_talks'][0]
data['related_views'] = 0
for ii in range(len(data)):
    #Remove string
    less = ast.literal_eval(data['related_talks'][ii])
    related_views = []
    for ll in range(len(less)):
        #Add view counts for each related talk into list
        related_views.append(less[ll]['viewed_count'])
        data[['related_views']][ii] = np.mean(related_views)    
data['event'].unique()
#Create column with default value of "Other"
data['event_class'] = 'Other'
#Loop over every talk and assign event category based on name of event
for ii in range(len(data)):
    if data['event'][ii].count('TED20') >0:
        data['event_class'][ii] = 'Yearly TED Conference'
    elif data['event'][ii].count('TED19') >0:
        data['event_class'][ii] = 'Yearly TED Conference'
    elif data['event'][ii].count('TEDx') >0:
        data['event_class'][ii] = 'TEDx'
    elif data['event'][ii].count('TEDGlobal') >0:
        data['event_class'][ii] = 'TEDGlobal'
    elif data['event'][ii].count('TEDWomen') >0:
        data['event_class'][ii] = 'TEDWomen'
    if data['event'][ii].count('TEDSalon') >0:
        data['event_class'][ii] = 'TEDSalon'
    if data['event'][ii].count('TEDNYC') >0:
        data['event_class'][ii] = 'TEDNYC'
    else:
        pass
data['event_class'].unique()
data['tags'].head()
data['tags'][9]
destring = []
for number in range(len(data)):
    #Remove string
    destring.append(ast.literal_eval(data['tags'][number]))
data['Tags'] = pd.Series(destring)
from gensim.models import KeyedVectors
#Load Google's vectors
model = KeyedVectors.load_word2vec_format("../input/trained-embeddings/word-embeddings/GoogleNews-vectors-negative300.bin.gz", binary=True)
#Extract all tags from list of lists
listed = [item for sublist in destring for item in sublist]
listed = pd.Series(listed)

#Only take unique tags
lists = list(listed.unique())

#Remove phrases and hyphenated words
lists2 = [ x for x in lists if " " not in x ]
lists2 = [ x for x in lists2 if "-" not in x ]
#Remove anomaly words
lists2.remove('archaeology')
lists2.remove('TEDYouth')
lists2.remove('deextinction')
lists2.remove('blockchain')
lists2.remove('TEDNYC')
#List containing each word
labels = []
#List containing the vector representation of each word
tokens = []

#Populate lists 
for word in lists2:
    tokens.append(model[word])
    labels.append(word)

#T-SNE model for 2D representation
tsne_model = TSNE(perplexity=50, n_components=2, init='pca', n_iter=105000, random_state=17,learning_rate=5500)
new_values = tsne_model.fit_transform(tokens)

#K-Means model to assign similar clusters
kmeans = KMeans(n_clusters=15,n_init=200)
kmeans.fit(tokens)
clusters = kmeans.predict(tokens)

#DataFrame we will use to plot
df_tsne = pd.DataFrame(new_values, columns=['1st_Comp', '2nd_Comp'])
df_tsne['Cluster'] = clusters

sns.lmplot(x='1st_Comp', y='2nd_Comp', data=df_tsne, hue='Cluster', fit_reg=False)
plt.title("Tag Clusters")
#Word to Cluster
convert = {labels[word]: clusters[word] for word in range(len(labels))}
#Comparison DataFrame
comp = pd.DataFrame(labels)
comp['cluster'] = clusters
#Cluster to Group Title
comp_conver = {0:'Organizing/Perceiving Information',1:'animals/organisms',2:'exploration',3:'Scientific Fields',
              4:'media/entertainment',5:'arts/creativity',6:'Epidemics',7:'Humanity/Progress',8:'Vices/Prejudices',
              9:'robots/prosthetics',10:'music',11:'philanthropy/religion',12:'Middle East',13:'Global issues',
              14:'Outer-Space',15:'NA'}
#Add group titles to DataFrame
comp['group'] = 'None'
for ii in range(len(comp)):
    comp['group'][ii] = comp_conver[comp['cluster'][ii]]
    
#Only take unique group titles for dummy variables
unique = comp['group'].unique()
for group in unique:
    #Create dummy variable for tag group
    data[group+'_tag'] = 0
    for item in range(len(data['Tags'])):
        #Loop through list of tags for each talk
        for ii in data['Tags'][item]:
            #Convert word to cluster
            try:
                clust = convert[ii]
            #If tag is not in vocabulary then we assign it 15 which voids the conversion
            except KeyError:
                clust = 15
            #Convert cluster to group title
            grouping = comp_conver[clust]
            if grouping == group:
                data[group+'_tag'][item] = 1
#look at the tags
data.filter(like='_tag', axis=1).head()
#Current columns
data.columns
#Drop unnecessary columns
data_final = data.drop(['description','event','film_date','num_speaker','ratings','related_talks','tags','url',
                        'Tags','main_speaker','name','Funny','Beautiful','Ingenious','Courageous','Longwinded',
                       'Confusing','Informative','Fascinating','Unconvincing','Persuasive','Jaw-dropping',
                       'OK','Obnoxious','Inspiring'],axis=1)
data_final.columns
#Save data
#data_final.to_csv("..../input/ted-condensed-data/cleaned_data_.csv")
data_final = pd.read_csv("../input/ted-condensed-data/cleaned_data_.csv")
#Positive Ratings vs. View Count
f, ax = plt.subplots(figsize = (8,6))
sns.regplot(data=data_final,x='Positive',y='views')
plt.title("Relationship of Positive Ratings and View Count", fontsize = 16)
plt.ylabel("Number of Views", fontsize = 14)
plt.xlabel("Number of Positive Ratings", fontsize = 14)
data_final['views_per_positive_rating'] = data_final['views']/data_final['Positive']
neg_views = len(data[data['Negative']>data['Positive']])
pos_views = len(data[data['Negative']<data['Positive']])
mod_views = len(data[(data['Moderate']> (data['Positive'] + data['Negative']))])
print()
print('Number of Positively Rated Talks :{}' .format(pos_views))
print()
print('Number of Moderately Rated Talks:{}' .format(mod_views))
print()
print('Number of Negatively Rated Talks:{}' .format(neg_views))
print()
#Languages vs. View Count
f, ax = plt.subplots(figsize = (8,6))
sns.regplot(data=data_final,x='languages',y='views')
plt.title("Relationship Between Languages and View Count", fontsize = 16)
plt.ylabel("Number of Views: Ten Millions ", fontsize = 14)
plt.xlabel("Number of Languages", fontsize = 14)
data_final['per_language_views'] = data_final['views']/data_final['languages']
#comments vs. View Counts
f, ax = plt.subplots(figsize = (8,6))
sns.regplot(data=data_final,x='comments',y='views')
plt.title("Relationship Between Comments and View Count", fontsize = 16)
plt.ylabel("Number of Views: Ten Millions ", fontsize = 14)
plt.xlabel("Number of Comments", fontsize = 14)
data_final['per_comment_views'] = data_final['views']/data_final['comments']
#Related Views vs. View Counts
f, ax = plt.subplots(figsize = (8,6))
sns.regplot(data=data_final,x='related_views',y='views')
plt.title("Relationship Between Related Views and Views", fontsize = 16)
plt.ylabel("Number of Views: Ten Millions ", fontsize = 14)
plt.xlabel("Number of Related Views", fontsize = 14)
#View Count by Event Class
f, ax = plt.subplots(figsize = (14,8))
sns.barplot(data=data_final,x='event_class',y='views')
plt.title("Views by Event Class", fontsize = 18)
plt.ylabel("Number of Views", fontsize = 14)
plt.xlabel("Event Class", fontsize = 14)
#Isloate Tag Variables
vis = data_final[['views','Humanity/Progress_tag',
       'arts/creativity_tag', 'philanthropy/religion_tag', 'music_tag',
       'Global issues_tag', 'Scientific Fields_tag', 'media/entertainment_tag',
       'Organizing/Perceiving Information_tag', 'Middle East_tag',
       'Epidemics_tag', 'Outer-Space_tag', 'Vices/Prejudices_tag',
       'exploration_tag', 'robots/prosthetics_tag', 'animals/organisms_tag']]
#View Count by Content Tag
fig = plt.figure(figsize=(21,14))
#fig, ax = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(6, 6))
#Create a barplot for each tag showing the mean number of views
for i in np.arange(15):
    ax = fig.add_subplot(3,5,i+1)
    sns.barplot(x=vis.iloc[:,i+1], y=vis['views'])
    plt.ylabel('',)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.suptitle("Tag Evaluation", fontsize = 18)
plt.show()
#Define target variables
target_views = np.log(data_final.views)
target_positive = np.log(data_final.Positive)
target_comments = np.log(data_final.comments)
#Drop target variables and variables with no real significance
data_final = data_final.drop(['Unnamed: 0', 'speaker_occupation', 'title', 'views','Positive','Negative','Moderate','comments','views_per_positive_rating','per_language_views','per_comment_views'], axis=1)
#Create dummy variables for our categorical variables
data_final = pd.get_dummies(data_final)

data_final.head(1)
def cross_val(df,target_param):
    #Cross-Validation set based on target variable of interest
    X_train, X_test, y_train, y_test = train_test_split(df, target_param, test_size=0.2, random_state=42)
    #Scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    scaled_train = scaler.transform(X_train)
    scaled_cv = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaled_train, scaled_cv
    
def Model_Prediction(model):
    model.fit(scaled_train, y_train)
    predictions = model.predict(scaled_train)
    mse = mean_squared_error(y_train, predictions)
    cv_predictions = model.predict(scaled_cv)
    cv_mse = mean_squared_error(y_test,cv_predictions)
    print("Model Results")
    print("RMSE: {}".format(np.sqrt(mse)))
    print("CV_RMSE: {}".format(np.sqrt(cv_mse)))
    return model
    
def plot_coef(model):
    #Plot coefficients if model uses Gradient Descent
    try:
        series_coef = pd.Series(model.coef_,index = X_train.columns)
        series_coef = series_coef.sort_values()
        series_coef.plot(kind='barh',figsize=(12,7),fontsize=12)
        #plt.title('Impact of Isolated Features',fontsize=22)
        plt.ylabel('Feature',fontsize=18)
        plt.xlabel('Size of Coefficient',fontsize=18)
    except AttributeError:
        pass
    #Plot feature importance if model uses Random Forest
    try:
        series_coefs = pd.Series(model.feature_importances_,index = X_train.columns)
        series_coefs = series_coefs.sort_values()
        series_coefs.plot(kind='barh',figsize=(14,9),fontsize=12)
        #plt.title('Impact of Isolated Features',fontsize=22)
        plt.ylabel('Feature',fontsize=18)
        plt.xlabel('Level of Importance',fontsize=18)
    except AttributeError:
        pass
#Standard Gradient Descent
gradient_descent = SGDRegressor(tol=.0001, eta0=.01)

X_train, X_test, y_train, y_test, scaled_train, scaled_cv = cross_val(data_final,target_views)
Model_Prediction(gradient_descent)
plot_coef(gradient_descent)
plt.title('Impact of Features on Views: Gradient Descent',fontsize=22)
#Elastic Net
ENSTest = ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .3, .5, .7, .9, .99], max_iter=50000)

Model_Prediction(ENSTest)
plot_coef(ENSTest)
plt.title('Impact of Features on Views: Elastic Net',fontsize=22)
data_final = data_final.drop(['languages','published_date','duration','related_views'],axis=1)
#Elastic Net
ENSTest = ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .3, .5, .7, .9, .99], max_iter=50000)

X_train, X_test, y_train, y_test, scaled_train, scaled_cv = cross_val(data_final,target_views)
Model_Prediction(ENSTest)
plot_coef(ENSTest)
plt.title('Impact of Features on Views: Elastic Net',fontsize=22)
forest = RandomForestRegressor(max_depth = 16,max_features = 14, n_estimators=2000)

X_train, X_test, y_train, y_test, scaled_train, scaled_cv = cross_val(data_final,target_views)
Model_Prediction(forest)
plot_coef(forest)
plt.title("Importance of Features on Views: Random Forest",fontsize=22)
#Standard Gradient Descent
gradient_descentP = SGDRegressor(tol=.0001, eta0=.01)

X_train, X_test, y_train, y_test, scaled_train, scaled_cv = cross_val(data_final,target_positive)
Model_Prediction(gradient_descentP)
plot_coef(gradient_descentP)
plt.title('Impact of Features on Positive Ratings: Gradient Descent',fontsize=22)
#Elastic Net
ENSTestP = ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .3, .5, .7, .9, .99], max_iter=50000)

Model_Prediction(ENSTestP)
plot_coef(ENSTestP)
plt.title('Impact of Features on Positive Ratings: Elastic Net',fontsize=22)
forestP = RandomForestRegressor(max_depth=16,max_features=12,n_estimators=2500)

Model_Prediction(forestP)
plot_coef(forestP)
plt.title("Importance of Features on Positive Ratings: Random Forest",fontsize=22)
#Standard Gradient Descent
gradient_descentc = SGDRegressor(tol=.0001, eta0=.01)

X_train, X_test, y_train, y_test, scaled_train, scaled_cv = cross_val(data_final,target_comments)
Model_Prediction(gradient_descentc)
plot_coef(gradient_descentc)
plt.title('Impact of Features on Comments: Gradient Descent',fontsize=22)
#Elastic Net
ENSTestC = ElasticNetCV(alphas=[0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .3, .5, .7, .9, .99], max_iter=50000)

X_train, X_test, y_train, y_test, scaled_train, scaled_cv = cross_val(data_final,target_comments)
Model_Prediction(ENSTestC)
plot_coef(ENSTestC)
plt.title('Impact of Features on Comments: Elastic Net',fontsize=22)
forestc = RandomForestRegressor(max_depth = 16,max_features = 14, n_estimators=2000)

X_train, X_test, y_train, y_test, scaled_train, scaled_cv = cross_val(data_final,target_comments)
Model_Prediction(forestc)
plot_coef(forestc)
plt.title("Importance of Features on Comments: Random Forest",fontsize=22)
#Get all variables again
data_final = pd.read_csv("../input/ted-condensed-data/cleaned_data_.csv")

#describe the data to contextualize the RMSE 
data_final[['views','Positive','comments']].describe()
series_coef = pd.DataFrame(ENSTestP.coef_,index = X_train.columns)
series_coef.columns=['Views']
series_coef['Positive'] = ENSTest.coef_
series_coef['Comments'] = ENSTestC.coef_
series_coef = series_coef.sort_values(by='Views')
series_coef = series_coef.reset_index()
events_coef = series_coef[series_coef['index'].str.contains('event')]
tags_coef = series_coef[series_coef['index'].str.contains('tag')]
events_coef = events_coef.set_index('index')
tags_coef = tags_coef.set_index('index')
events_coef.plot(kind='barh',figsize=(18,8),fontsize=14)
plt.title("Comparative Effects of Location on Targets",fontsize=18)
plt.xlabel("Size of Coefficient",fontsize=14)
plt.ylabel(' ')
tags_coef.plot(kind='barh',figsize=(18,8),fontsize=14)
plt.title("Comparative Effects of Content on Targets",fontsize=18)
plt.xlabel("Size of Coefficient",fontsize=14)
plt.ylabel(' ')
