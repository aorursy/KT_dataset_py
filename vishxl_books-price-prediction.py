import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import gc
import warnings
warnings.filterwarnings('ignore')

train = pd.read_excel('../input/book-price-machinehack/Data_Train.xlsx')
test = pd.read_excel('../input/book-price-machinehack/Data_Test.xlsx')
import pandas as pd
import numpy as np
#combining test and train for cleaning
combined = pd.concat([train, test], sort=False)
combined.reset_index(drop=True, inplace=True)
combined['Reviews'] = combined['Reviews'].str.split(' ').str.get(0).astype(float)
combined['Ratings'] = combined['Ratings'].str.split(' ').str.get(0).str.replace(',','').astype(float)
combined[['Issue_type','Issue_date']] = combined['Edition'].str.split(',– ',expand=True)
combined['Issue_type'] = combined.Issue_type.str.extract(r'(^[a-zA-Z|\s]*)')[0]
combined['Issue_type'].unique()
#Month & Year
#combined['Issue_year'] = combined.Issue_date.str[-4:].astype(float) ##Alternative Method
combined['Issue_year'] = combined.Issue_date.str.extract('.*(\d{4})', expand = False)
combined['Issue_year'] = combined['Issue_year'].fillna(2004).astype(int) #2004 being the mean


combined['Book_age'] = 2019 - combined['Issue_year']

combined['Issue_month'] = pd.to_datetime(combined['Issue_date'], errors='coerce').dt.month
#for printing the entire table
"""
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(combined.Issue_year) """
#Genre
combined['Genre'] = combined['Genre'].str.replace(r"\(.*\)","")  #removing '(Books)' from Genre 
combined['Genre'] = combined['Genre'].str.replace(' &',',') #replace '&' with ,
combined['Genre_count'] = combined['Genre'].apply(lambda x: len(x.split(',')))  #Counting Genre

#BookCategory 
combined['BookCategory'] = combined['BookCategory'].str.replace(r"\(.*\)","")  #removing '(Books)' from Genre 
combined['BookCategory'] = combined['BookCategory'].str.replace(' &',',') #replace '&' with ,
combined['BookCategory_count'] = combined['BookCategory'].apply(lambda x: len(x.split(',')))
combined.describe(include = 'all').head(2)
combined['Net_Rating'] = round(combined['Ratings']*combined['Reviews'], 2)
#Authors
author_replacements = {' & ':', ',"0":"other","2":"other",'A. P. J. Abdul Kalam':'A.P.J. Abdul Kalam','APJ Abdul Kalam':'A.P.J. Abdul Kalam','Agrawal P. K.': 'Agrawal P.K','Ajay K Pandey': 'Ajay K. Pandey','Aravinda Anantharaman': 'Aravinda Anatharaman','Arthur Conan Doyle': 'Sir Arthur Conan Doyle','B A Paris': 'B. A. Paris','E L James': 'E. L. James','E.L. James':'E. L. James','Eliyahu M Goldratt': 'Eliyahu M. Goldratt','Ernest Hemingway': 'Ernest Hemmingway','Frank Miler': 'Frank Miller','Fyodor Dostoevsky': 'Fyodor Dostoyevsky','George R R Martin': 'George R. R. Martin','George R.R. Martin':'George R. R. Martin','H. G. Wells': 'H.G. Wells','Johann Wolfgang Von Goethe': 'Johann Wolfgang von Goethe','John Le Carré': 'John le Carré','Judith McNaught': 'Judith Mcnaught','Keith Giffen': 'Kieth Giffen','Ken Hultgen': 'Ken Hultgren','Kentaro Miura': 'Kenturo Miura','Kohei Horikoshi': 'Kouhei Horikoshi','M.K Gandhi': 'M.K. Gandhi','Matthew K Manning': 'Matthew Manning','Michael Crichton': 'Micheal Crichton','N.K Aggarwala': 'N.K. Aggarwala','Oxford University Press (India)': 'Oxford University Press India','P D James': 'P. D. James','Paramahansa Yogananda': 'Paramhansa Yogananda','R K Laxman': 'R. K. Laxman','R.K. Laxman': 'R. K. Laxman','R. M. Lala': 'R.M. Lala','Raina Telgemaeier': 'Raina Telgemeier','Rajaraman': 'Rajaraman V','Rajiv M. Vijayakar': 'Rajiv Vijayakar','Ramachandra Guha': 'Ramchandra Guha','Rene Goscinny': 'René Goscinny','Richard P Feynman': 'Richard P. Feynman','S Giridhar': 'S. Giridhar','S Hussain Zaidi': 'S. Hussain Zaidi','S. A. Chakraborty': 'S. Chakraborty','Santosh Kumar K': 'Santosh Kumar K.',"S.C. Gupta" : "S. C. Gupta",'Shiv Prasad Koirala': 'Shivprasad Koirala','Shivaprasad Koirala': 'Shivprasad Koirala','Simone De Beauvoir': 'Simone de Beauvoir','Sir Arthur Conan Doyle': 'Arthur Conan Doyle',"Terry O' Brien": "Terry O'Brien",'Thich Nhat Hahn': 'Thich Nhat Hanh','Trinity College Lond': 'Trinity College London',"Trinity College London Press" : "Trinity College London",'Ursula K. Le Guin': 'Ursula Le Guin','Willard A Palmer': 'Willard A. Palmer','Willard Palmer': 'Willard A. Palmer','William Strunk Jr': 'William Strunk Jr.','Yashavant Kanetakr': 'Yashavant Kanetkar','Yashavant P. Kanetkar': 'Yashavant Kanetkar','Yashwant Kanetkar': 'Yashavant Kanetkar','et al': 'et al.',' et al': 'et al.','Peter Clutterbuck': ' Peter Clutterbuck','Scholastic': 'Scholastic ','Ullekh N. P.': 'Ullekh N.P.','Shalini Jain': 'Dr. Shalini Jain','Kevin Mitnick': 'Kevin D. Mitnick'}
combined['Author'] = combined['Author'].replace(author_replacements,regex=True)
combined['Author_count'] = combined['Author'].apply(lambda x: len(x.split(',')))
combined['Review_Impact']= combined['Reviews']*combined['Book_age']
#combined = pd.get_dummies(combined, drop_first=True)
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
combined[['Issue_type','Issue_month']] = combined[['Issue_type','Issue_month']].apply(enc.fit_transform)
from sklearn.feature_extraction.text import CountVectorizer
t_vectorizer = CountVectorizer(max_features=10, lowercase=True)
t_vector = t_vectorizer.fit_transform(combined['Title']).toarray()
df_title = pd.DataFrame(data=t_vector,
                        columns=t_vectorizer.get_feature_names())
bc_vectorizer = CountVectorizer(lowercase=True, tokenizer=lambda x: x.split(', '))
book_cat_vector = bc_vectorizer.fit_transform(combined['BookCategory']).toarray()
df_book_cat = pd.DataFrame(data=book_cat_vector,
                      columns=bc_vectorizer.get_feature_names())
genre_vectorizer = CountVectorizer(max_features=10,
                                   lowercase=True, tokenizer=lambda x: x.split(', '))
genre_vector = genre_vectorizer.fit_transform(combined['Genre']).toarray()
df_genre = pd.DataFrame(data=genre_vector,
                        columns=genre_vectorizer.get_feature_names())
author_vectorizer = CountVectorizer(max_features=10, lowercase=True,
                                    tokenizer=lambda x: x.split(', '))
author_vector = author_vectorizer.fit_transform(combined['Author']).toarray()
df_author = pd.DataFrame(data=author_vector,
                         columns=author_vectorizer.get_feature_names())
synopsis_vectorizer = CountVectorizer(max_features=10,
                                      stop_words='english', 
                                      strip_accents='ascii', 
                                      lowercase=True)
synopsis_vector = synopsis_vectorizer.fit_transform(combined['Synopsis']).toarray()
df_syn = pd.DataFrame(data=synopsis_vector,
                           columns=synopsis_vectorizer.get_feature_names())
final = combined[['Price','Reviews', 'Ratings', 'Issue_type','Issue_month', 'Genre_count',
       'BookCategory_count', 'Net_Rating','Review_Impact', 'Author_count', 
       'Book_age']]
final = pd.concat([
    final,# dummy encoded features
    df_title,
    df_book_cat,
    df_genre,
    df_author,
    df_syn,
    ], axis=1)
final.reset_index(drop=True, inplace=True)
final.head()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
correlation_mat = final.corr()
top_corr_features = correlation_mat.index
plt.figure(figsize=(20,20))
g= sns.heatmap(final[top_corr_features].corr(),annot=True,cmap="coolwarm")
#Splitting Train, Test

X_full = final.iloc[:,1:]
X = X_full.iloc[:6237,1:]
y = final.iloc[:6237,0]
X_eval = X_full.iloc[6237:,1:]
print(X_full.shape, X.shape, y.shape, X_eval.shape)
###Feature Importance
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_) 
#plot graph of feature importance for better visualization
feat_importances = pd.Series(model.feature_importances_,index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()
#train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.15)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
n_estimators= [int(x) for x in np.linspace(start = 100,stop = 1200, num=12)] 
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num =6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]
bootstrap = [True, False]
from sklearn.model_selection import RandomizedSearchCV
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
rf = RandomForestRegressor()
model_tuned = RandomizedSearchCV(estimator = rf, 
                                 param_distributions = random_grid, 
                                 scoring='neg_mean_squared_error', 
                                 n_iter = 10, 
                                 cv = 5, 
                                 verbose=2, 
                                 random_state=42, 
                                 n_jobs = 2)
model_tuned.fit(X_train, y_train)
#prediction
predictions = model_tuned.predict(X_test)
X_eval.shape
eval = model_tuned.predict(X_eval)
eval
sns.distplot(y_test)
sns.distplot(eval)
from sklearn.metrics import r2_score
r2_score(y_test, predictions)
from sklearn.metrics import mean_squared_log_error
np.sqrt(mean_squared_log_error( y_test, predictions ))
submission = pd.DataFrame(columns=['Price'])
submission['Price'] = eval
submission.to_excel('to_submit.xlsx',index=False)