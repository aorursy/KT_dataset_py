# This Python 3 environment comes with many helpful analytics libraries installed



import pandas as pd

import matplotlib.pyplot as plt

import re

import time

import warnings

import numpy as np

from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.manifold import TSNE

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from imblearn.over_sampling import SMOTE

from collections import Counter

from scipy.sparse import hstack

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold 

from collections import Counter, defaultdict

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import math

from sklearn.metrics import normalized_mutual_info_score

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")



from mlxtend.classifier import StackingClassifier



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression
# exploring the data files

from subprocess import check_output

# checking for subfolder name

print(check_output(["ls", "../input"]).decode("utf8"))
# exploring available data files

print(check_output(["ls", "../input/msk-redefining-cancer-treatment"]).decode("utf8"))
# unzipping training data

import zipfile



# declaring datasets to unzip

Datasets = [

    "../input/msk-redefining-cancer-treatment/" + "training_text.zip",

    "../input/msk-redefining-cancer-treatment/" + "training_variants.zip"

]



# unzipping declared datasets

for dataset in Datasets:

    with zipfile.ZipFile(dataset,"r") as z:

        z.extractall(".")
# Checking that unziped file is created

for dataset in Datasets:

    # dataset name is in both cases the 3 index after split "/" in this particular example

    print(check_output(["ls",dataset.split("/")[3][:-4]]).decode("utf8"))
# Loading training data

data_variants = pd.read_csv('training_variants')

# training_text dataset uses "||" as a seperator and has the headers seperated by "," so we skip that row and declare headers

data_text =pd.read_csv("training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
data_variants.head(3)
data_variants.info()
data_variants.describe()
# Checking dimention of data

# even though we already know from count and columns exploration (data_variants.head(3))

data_variants.shape
# Clecking column in above data set

# even though we already know from data_variants.head(3)

data_variants.columns
data_text.head(3)
data_text.info()
# data_text.describe() is not useful for text data values

data_text.columns
# checking the dimentions (which we already know)

data_text.shape
# Confirmation of aviable results

data_variants.Class.unique()
# We would like to remove all stop words like a, is, an, the, ... 

# so we collecting all of them from nltk library

stop_words = set(stopwords.words('english'))
# defining function to remove all stop words from data

def data_text_preprocess(total_text, ind, col):

    # Remove int values from text data as that might not be imp

    if type(total_text) is not int:

        string = ""

        # replacing all special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))

        # replacing multiple spaces with single space

        total_text = re.sub('\s+',' ', str(total_text))

        # bring whole text to same lower-case scale.

        total_text = total_text.lower()

        

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from text

            if not word in stop_words:

                string += word + " "

        

        data_text[col][ind] = string
# applying data_text_preprocess to data_text

for index, row in data_text.iterrows():

    if type(row['TEXT']) is str:

        data_text_preprocess(row['TEXT'], index, 'TEXT')
# merging both gene_variations and text data based on ID

result = pd.merge(data_variants, data_text,on='ID', how='left')

result.head()
# checking for missing values

# missing values may create qualitive problematics in our final analysis

result[result.isnull().any(axis=1)]
# Imputing the missing values as "Gene" + "Variation" text

# This is the example imputation used in the course (check note in the beggining)

result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']
# Confirming that tere are no missing values

result[result.isnull().any(axis=1)]
y_true = result['Class'].values

# replacing spaces with "_"

result.Gene      = result.Gene.str.replace('\s+', '_')

result.Variation = result.Variation.str.replace('\s+', '_')
# Splitting the data into train and test set 

X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2)

# split the train data now into train validation and cross validation

train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
print('Number of data points in train data:', train_df.shape[0])

print('Number of data points in test data:', test_df.shape[0])

print('Number of data points in cross validation data:', cv_df.shape[0])
# counting class values for each set and sorting for better comparison

train_class_distribution = train_df['Class'].value_counts().sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True)

test_class_distribution = test_df['Class'].value_counts().sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True)

cv_class_distribution = cv_df['Class'].value_counts().sort_index(axis=0, level=None, ascending=True, inplace=False, kind='quicksort', na_position='last', sort_remaining=True)
train_class_distribution
# Visualizing train class distrubution

my_colors = 'rgbkymc'

train_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel(' Number of Data points per Class')

plt.title('Distribution of yi in train data')

plt.grid()

plt.show()
# Printing distribution in percentage form

sorted_yi = np.argsort(-train_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')
# test set diribution visualization

my_colors = 'rgbkymc'

test_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Number of Data points per Class')

plt.title('Distribution of yi in test data')

plt.grid()

plt.show()
# test set distribution in percentage form

sorted_yi = np.argsort(-test_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3), '%)')
# cross validation set diribution visualization

my_colors = 'rgbkymc'

cv_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Number of Data points per Class')

plt.title('Distribution of yi in test data')

plt.grid()

plt.show()
# cross validation set distribution in percentage form

sorted_yi = np.argsort(-cv_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3), '%)')
# getting the length of our datasets

test_data_len = test_df.shape[0]

cv_data_len = cv_df.shape[0]



# creating an output array that has exactly same size as the CV data

cv_predicted_y = np.zeros((cv_data_len,9))

for i in range(cv_data_len):

    rand_probs = np.random.rand(1,9)

    # setting random values so each row sum is equal to 1

    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

# Evaluating and printing the Log Loss for worst model

# All models that will be generated should not be worse than worst model

print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))



# Test-Set error (worst model).

# creating output array that has exactly same as the test data

test_predicted_y = np.zeros((test_data_len,9))

for i in range(test_data_len):

    rand_probs = np.random.rand(1,9)

    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))
# Lets get the index of max probablity

predicted_y =np.argmax(test_predicted_y, axis=1)

predicted_y
predicted_y = predicted_y + 1
C = confusion_matrix(y_test, predicted_y)

C
# Displaying predictions

labels = [1,2,3,4,5,6,7,8,9]

plt.figure(figsize=(20,7))

sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()
# generating precision matrix

B =(C/C.sum(axis=0))

# diplaying precision matrix

plt.figure(figsize=(20,7))

sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()
# generating recall matrix

A =(((C.T)/(C.sum(axis=1))).T)

# diplaying recall matrix

plt.figure(figsize=(20,7))

sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()
# Creating unique genes series

unique_genes = train_df['Gene'].value_counts()

# measuring the length of unique genes matrix

print('Number of Unique Genes :', unique_genes.shape[0])

# the top 10 genes that occured most

print(unique_genes.head(10))
# Cumulative distribution of unique genes

s = sum(unique_genes.values);

h = unique_genes.values/s;

c = np.cumsum(h)

plt.plot(c,label='Cumulative distribution of Genes')

plt.grid()

plt.legend()

plt.show()
# one-hot encoding of Gene feature.

gene_vectorizer = CountVectorizer()

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])

test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])

cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])
# checking classified train set shape

train_gene_feature_onehotCoding.shape
# plotting added column names

gene_vectorizer.get_feature_names()
# ----Notes----

# code for response coding with Laplace smoothing.

# alpha : used for laplace smoothing

# feature: ['gene', 'variation']

# df: ['train_df', 'test_df', 'cv_df']

# algorithm

# ----------

# Consider all unique values and the number of occurances of given feature in train data dataframe

# build a vector (1*9) , the first element = (number of times it occured in class1 + 10*alpha / number of time it occurred in total data+90*alpha)

# gv_dict is like a look up table, for every gene it store a (1*9) representation of it

# for a value of feature in df:

# if it is in train data:

# we add the vector that was stored in 'gv_dict' look up table to 'gv_fea'

# if it is not there is train:

# we add [1/9, 1/9, 1/9, 1/9,1/9, 1/9, 1/9, 1/9, 1/9] to 'gv_fea'

# return 'gv_fea'

# ----------------------





# get_gv_fea_dict: Get Gene varaition Feature Dict

def get_gv_fea_dict(alpha, feature, df):

    # value_count: it contains a dict like

    # print(train_df['Gene'].value_counts())

    # output:

    #        {BRCA1      174

    #         TP53       106

    #         EGFR        86

    #         BRCA2       75

    #         PTEN        69

    #         ...}

    # print(train_df['Variation'].value_counts())

    # output:

    # {

    # Truncating_Mutations                     63

    # Deletion                                 43

    # Amplification                            43

    # Fusions                                  22

    # Overexpression                            3

    # E17K                                      3

    # Q61L                                      3

    # S222D                                     2

    # P130S                                     2

    # ...

    # }

    value_count = train_df[feature].value_counts()

    

    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation

    gv_dict = dict()

    

    # denominator will contain the number of time that particular feature occured in whole data

    for i, denominator in value_count.items():

        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to perticular class

        # vec is 9 diamensional vector

        vec = []

        for k in range(1,10):

            # print(train_df.loc[(train_df['Class']==1) & (train_df['Gene']=='BRCA1')])

            #         ID   Gene             Variation  Class  

            # 2470  2470  BRCA1                S1715C      1   

            # 2486  2486  BRCA1                S1841R      1   

            # 2614  2614  BRCA1                   M1R      1   

            # 2432  2432  BRCA1                L1657P      1   

            # 2567  2567  BRCA1                T1685A      1   

            # 2583  2583  BRCA1                E1660G      1   

            # 2634  2634  BRCA1                W1718L      1   

            # cls_cnt.shape[0] will return the number of rows



            cls_cnt = train_df.loc[(train_df['Class']==k) & (train_df[feature]==i)]

            

            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data

            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))



        # we are adding the gene/variation to the dict as key and vec as value

        gv_dict[i]=vec

    return gv_dict



# Get Gene variation feature

def get_gv_feature(alpha, feature, df):

    # print(gv_dict)

    #     {'BRCA1': [0.20075757575757575, 0.03787878787878788, 0.068181818181818177, 0.13636363636363635, 0.25, 0.19318181818181818, 0.03787878787878788, 0.03787878787878788, 0.03787878787878788], 

    #      'TP53': [0.32142857142857145, 0.061224489795918366, 0.061224489795918366, 0.27040816326530615, 0.061224489795918366, 0.066326530612244902, 0.051020408163265307, 0.051020408163265307, 0.056122448979591837], 

    #      'EGFR': [0.056818181818181816, 0.21590909090909091, 0.0625, 0.068181818181818177, 0.068181818181818177, 0.0625, 0.34659090909090912, 0.0625, 0.056818181818181816], 

    #      'BRCA2': [0.13333333333333333, 0.060606060606060608, 0.060606060606060608, 0.078787878787878782, 0.1393939393939394, 0.34545454545454546, 0.060606060606060608, 0.060606060606060608, 0.060606060606060608], 

    #      'PTEN': [0.069182389937106917, 0.062893081761006289, 0.069182389937106917, 0.46540880503144655, 0.075471698113207544, 0.062893081761006289, 0.069182389937106917, 0.062893081761006289, 0.062893081761006289], 

    #      'KIT': [0.066225165562913912, 0.25165562913907286, 0.072847682119205295, 0.072847682119205295, 0.066225165562913912, 0.066225165562913912, 0.27152317880794702, 0.066225165562913912, 0.066225165562913912], 

    #      'BRAF': [0.066666666666666666, 0.17999999999999999, 0.073333333333333334, 0.073333333333333334, 0.093333333333333338, 0.080000000000000002, 0.29999999999999999, 0.066666666666666666, 0.066666666666666666],

    #      ...

    #     }

    gv_dict = get_gv_fea_dict(alpha, feature, df)

    # value_count is similar in get_gv_fea_dict

    value_count = train_df[feature].value_counts()

    

    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data

    gv_fea = []

    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea

    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea

    for index, row in df.iterrows():

        if row[feature] in dict(value_count).keys():

            gv_fea.append(gv_dict[row[feature]])

        else:

            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])

#             gv_fea.append([-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return gv_fea
# response-coding of the Gene feature

# alpha is used for laplace smoothing

alpha = 1

# train gene feature

train_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", train_df))

# test gene feature

test_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", test_df))

# cross validation gene feature

cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", cv_df))
# exploring training gene shape

train_gene_feature_responseCoding.shape
# We need a hyperparemeter for SGD classifier.

# giving alpha a set of ranges to compare

alpha = [10 ** x for x in range(-5, 1)]
# We will be using SGD classifier

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# We will also be using Calibrated Classifier to get the result into probablity format t be used for log loss

cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_gene_feature_onehotCoding, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_gene_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
# Lets plot the same to check the best Alpha value

#fig, ax = plt.subplots()

#ax.plot(alpha, cv_log_error_array,c='g')

#for i, txt in enumerate(np.round(cv_log_error_array,3)):

#    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

#plt.grid()

#plt.title("Cross Validation Error for each alpha")

#plt.xlabel("Alpha i's")

#plt.ylabel("Error measure")

#plt.show()





# because the graph is not clear this section is commented and written to just give the best alpha

def print_best_alpha(alpha_arr, loss_arr):

    print("The best alpha is: " + str(alpha_arr[loss_arr.index(min(loss_arr))]) + "\nThe best alpha index is: " + str(loss_arr.index(min(loss_arr))))

print_best_alpha(alpha, cv_log_error_array)
# Lets use best alpha value as we can see from above graph and compute log loss

# Building a very simple model using just gene column to check error decrease from worst model

# building a model with just one feature gives information on how much that feature is meaningful for the final result

best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_gene_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_gene_feature_onehotCoding, y_train)



predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

# checking for overlaping between train set and [cross_validation, test] set

test_coverage=test_df[test_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]

cv_coverage=cv_df[cv_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]
print('1. In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)

print('2. In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)
unique_variations = train_df['Variation'].value_counts()

print('Number of Unique Variations :', unique_variations.shape[0])

# the top 10 variations that occured most

print(unique_variations.head(10))
# looking at the comulative distribution of unique variation values

s = sum(unique_variations.values);

h = unique_variations.values/s;

c = np.cumsum(h)

print(c)

plt.plot(c,label='Cumulative distribution of Variations')

plt.grid()

plt.legend()

plt.show()
unique_variations = train_df['Variation'].value_counts()

print('Number of Unique Variations :', unique_variations.shape[0])

# the top 10 variations that occured most

print(unique_variations.head(10))
# ploting the distribution of variation values

s = sum(unique_variations.values);

h = unique_variations.values/s;

c = np.cumsum(h)

print(c)

plt.plot(c,label='Cumulative distribution of Variations')

plt.grid()

plt.legend()

plt.show()
# one-hot encoding of variation values.

variation_vectorizer = CountVectorizer()

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])

test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])

cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
# The shape of one hot encoder column for variation

train_variation_feature_onehotCoding.shape
# Response encoding of variation values.

# alpha is used for laplace smoothing

alpha = 1

# train gene feature

train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", train_df))

# test gene feature

test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", test_df))

# cross validation gene feature

cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", cv_df))
# the shape of this response encoding result

train_variation_feature_responseCoding.shape
# Lets again build the model with only column name of variation column

# We need a hyperparemeter for SGD classifier.

alpha = [10 ** x for x in range(-5, 1)]

# We will be using SGD classifier

# http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# We will also be using Calibrated Classifier to get the result into probablity format t be used for log loss

cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_variation_feature_onehotCoding, y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_variation_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

    

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
print_best_alpha(alpha, cv_log_error_array)
# checking for error on a simple model buided using just variation column

best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_variation_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_variation_feature_onehotCoding, y_train)



predict_y = sig_clf.predict_proba(train_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
# checking overlaping of training set with other sets

test_coverage=test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

cv_coverage=cv_df[cv_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

print('1. In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)

print('2. In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)
# cls_text is a data frame

# for every row in data fram consider the 'TEXT'

# split the words by space

# make a dict with those words

# increment its count whenever we see that word



def extract_dictionary_paddle(cls_text):

    dictionary = defaultdict(int)

    for index, row in cls_text.iterrows():

        for word in row['TEXT'].split():

            dictionary[word] +=1

    return dictionary



#https://stackoverflow.com/a/1602964

def get_text_responsecoding(df):

    text_feature_responseCoding = np.zeros((df.shape[0],9))

    for i in range(0,9):

        row_index = 0

        for index, row in df.iterrows():

            sum_prob = 0

            for word in row['TEXT'].split():

                sum_prob += math.log(((dict_list[i].get(word,0)+10 )/(total_dict.get(word,0)+90)))

            text_feature_responseCoding[row_index][i] = math.exp(sum_prob/len(row['TEXT'].split()))

            row_index += 1

    return text_feature_responseCoding
# building a CountVectorizer with all the words that occured minimum 3 times in train data

text_vectorizer = CountVectorizer(min_df=3)

train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])

# getting all the feature names (words)

train_text_features= text_vectorizer.get_feature_names()



# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1



# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured

text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))



print("Total number of unique words in train data :", len(train_text_features))
dict_list = []

# dict_list =[] contains 9 dictoinaries each corresponds to a class

for i in range(1,10):

    cls_text = train_df[train_df['Class']==i]

    # build a word dict based on the words in that class

    dict_list.append(extract_dictionary_paddle(cls_text))

    # append it to dict_list



# dict_list[i] is build on i'th  class text data

# total_dict is buid on whole training text data

total_dict = extract_dictionary_paddle(train_df)





confuse_array = []

for i in train_text_features:

    ratios = []

    max_val = -1

    for j in range(0,9):

        ratios.append((dict_list[j][i]+10 )/(total_dict[i]+90))

    confuse_array.append(ratios)

confuse_array = np.array(confuse_array)
# response coding of text features

# text column calculations teke a long time

train_text_feature_responseCoding  = get_text_responsecoding(train_df)

test_text_feature_responseCoding  = get_text_responsecoding(test_df)

cv_text_feature_responseCoding  = get_text_responsecoding(cv_df)
# https://stackoverflow.com/a/16202486

# we convert each row values such that they sum to 1  

train_text_feature_responseCoding = (train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T

test_text_feature_responseCoding = (test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T

cv_text_feature_responseCoding = (cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T
# don't forget to normalize every feature

train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)



# we use the same vectorizer that was trained on train data

test_text_feature_onehotCoding = text_vectorizer.transform(test_df['TEXT'])

# don't forget to normalize every feature

test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)



# we use the same vectorizer that was trained on train data

cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])

# don't forget to normalize every feature

cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)
#https://stackoverflow.com/a/2258273/4084039

sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))

sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))
# Number of words for a given frequency.

print(Counter(sorted_text_occur))
cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_text_feature_onehotCoding, y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_text_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))
print_best_alpha(alpha, cv_log_error_array)
# Simple model with only text data to evaluate its importance and check error

best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_text_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_text_feature_onehotCoding, y_train)



predict_y = sig_clf.predict_proba(train_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
# Checking text overlap

def get_intersec_text(df):

    df_text_vec = CountVectorizer(min_df=3)

    df_text_fea = df_text_vec.fit_transform(df['TEXT'])

    df_text_features = df_text_vec.get_feature_names()



    df_text_fea_counts = df_text_fea.sum(axis=0).A1

    df_text_fea_dict = dict(zip(list(df_text_features),df_text_fea_counts))

    len1 = len(set(df_text_features))

    len2 = len(set(train_text_features) & set(df_text_features))

    return len1,len2



len1,len2 = get_intersec_text(test_df)

print(np.round((len2/len1)*100, 3), "% of word of test data appeared in train data")

len1,len2 = get_intersec_text(cv_df)

print(np.round((len2/len1)*100, 3), "% of word of Cross Validation appeared in train data")
# Functions delcaration



def report_log_loss(train_x, train_y, test_x, test_y,  clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    sig_clf_probs = sig_clf.predict_proba(test_x)

    return log_loss(test_y, sig_clf_probs, eps=1e-15)



# This function plots the confusion matrices given y_i, y_i_hat.

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    

    A =(((C.T)/(C.sum(axis=1))).T)

    

    B =(C/C.sum(axis=0)) 

    labels = [1,2,3,4,5,6,7,8,9]

    # representing A in heatmap format

    print("-"*20, "Confusion matrix", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    # representing B in heatmap format

    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()





def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    pred_y = sig_clf.predict(test_x)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

    print("Log loss :",log_loss(test_y, sig_clf.predict_proba(test_x)))

    # calculating the number of data points that are misclassified

    print("Number of mis-classified points :", np.count_nonzero((pred_y- test_y))/test_y.shape[0])

    plot_confusion_matrix(test_y, pred_y)



# this function will be used just for naive bayes

# for the given indices, we will print the name of the features

# and we will check whether the feature present in the test point text or not

def get_impfeature_names(indices, text, gene, var, no_features):

    gene_count_vec = CountVectorizer()

    var_count_vec = CountVectorizer()

    text_count_vec = CountVectorizer(min_df=3)

    

    gene_vec = gene_count_vec.fit(train_df['Gene'])

    var_vec  = var_count_vec.fit(train_df['Variation'])

    text_vec = text_count_vec.fit(train_df['TEXT'])

    

    fea1_len = len(gene_vec.get_feature_names())

    fea2_len = len(var_count_vec.get_feature_names())

    

    word_present = 0

    for i,v in enumerate(indices):

        if (v < fea1_len):

            word = gene_vec.get_feature_names()[v]

            yes_no = True if word == gene else False

            if yes_no:

                word_present += 1

                print(i, "Gene feature [{}] present in test data point [{}]".format(word,yes_no))

        elif (v < fea1_len+fea2_len):

            word = var_vec.get_feature_names()[v-(fea1_len)]

            yes_no = True if word == var else False

            if yes_no:

                word_present += 1

                print(i, "variation feature [{}] present in test data point [{}]".format(word,yes_no))

        else:

            word = text_vec.get_feature_names()[v-(fea1_len+fea2_len)]

            yes_no = True if word in text.split() else False

            if yes_no:

                word_present += 1

                print(i, "Text feature [{}] present in test data point [{}]".format(word,yes_no))



    print("Out of the top ",no_features," features ", word_present, "are present in query point")
# merging gene, variance and text features



# building train, test and cross validation data sets

# a = [[1, 2], 

#      [3, 4]]

# b = [[4, 5], 

#      [6, 7]]

# hstack(a, b) = [[1, 2, 4, 5],

#                [ 3, 4, 6, 7]]



train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))

test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))

cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))



train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()

train_y = np.array(list(train_df['Class']))



test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()

test_y = np.array(list(test_df['Class']))



cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()

cv_y = np.array(list(cv_df['Class']))





train_gene_var_responseCoding = np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))

test_gene_var_responseCoding = np.hstack((test_gene_feature_responseCoding,test_variation_feature_responseCoding))

cv_gene_var_responseCoding = np.hstack((cv_gene_feature_responseCoding,cv_variation_feature_responseCoding))



train_x_responseCoding = np.hstack((train_gene_var_responseCoding, train_text_feature_responseCoding))

test_x_responseCoding = np.hstack((test_gene_var_responseCoding, test_text_feature_responseCoding))

cv_x_responseCoding = np.hstack((cv_gene_var_responseCoding, cv_text_feature_responseCoding))





print("One hot encoding features :")

print("(number of data points * number of features) in train data = ", train_x_onehotCoding.shape)

print("(number of data points * number of features) in test data = ", test_x_onehotCoding.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_onehotCoding.shape)

print(" Response encoding features :")

print("(number of data points * number of features) in train data = ", train_x_responseCoding.shape)

print("(number of data points * number of features) in test data = ", test_x_responseCoding.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_responseCoding.shape)
# http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    # MultinomialNB is used for multi class classification

    clf = MultinomialNB(alpha=i)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 



print_best_alpha(alpha, cv_log_error_array)
best_alpha = np.argmin(cv_log_error_array)

clf = MultinomialNB(alpha=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)





predict_y = sig_clf.predict_proba(train_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = MultinomialNB(alpha=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)

sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

# to avoid rounding error while multiplying probabilites we use log-probability estimates

print("Log Loss :",log_loss(cv_y, sig_clf_probs))

print("Number of missclassified point :", np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0])

plot_confusion_matrix(cv_y, sig_clf.predict(cv_x_onehotCoding.toarray()))
# checked item is defined apriory, without any meaning

# we can check whichever item we want

# defining item to be checked

test_point_index = 2

# important features to be printed

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
alpha = [5, 11, 15, 21, 31, 41, 51, 99]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = KNeighborsClassifier(n_neighbors=i)

    clf.fit(train_x_responseCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_responseCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(cv_y, sig_clf_probs))

print("")

print_best_alpha(alpha, cv_log_error_array)
best_alpha = np.argmin(cv_log_error_array)

clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_responseCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_responseCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_responseCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

predict_and_plot_confusion_matrix(train_x_responseCoding, train_y, cv_x_responseCoding, cv_y, clf)
# Lets look at few test points

clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



test_point_index = 1

predicted_cls = sig_clf.predict(test_x_responseCoding[0].reshape(1,-1))

print("Predicted Class :", predicted_cls[0])

print("Actual Class :", test_y[test_point_index])

neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), alpha[best_alpha])

print("The ",alpha[best_alpha]," nearest neighbours of the test points belongs to classes",train_y[neighbors[1][0]])

print("Fequency of nearest points :",Counter(train_y[neighbors[1][0]]))
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



test_point_index = 100



predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))

print("Predicted Class :", predicted_cls[0])

print("Actual Class :", test_y[test_point_index])

neighbors = clf.kneighbors(test_x_responseCoding[test_point_index].reshape(1, -1), alpha[best_alpha])

print("the k value for knn is",alpha[best_alpha],"and the nearest neighbours of the test points belongs to classes",train_y[neighbors[1][0]])

print("Fequency of nearest points :",Counter(train_y[neighbors[1][0]]))
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 

print("")

print_best_alpha(alpha, cv_log_error_array)
best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
def get_imp_feature_names(text, indices, removed_ind = []):

    word_present = 0

    tabulte_list = []

    incresingorder_ind = 0

    for i in indices:

        if i < train_gene_feature_onehotCoding.shape[1]:

            tabulte_list.append([incresingorder_ind, "Gene", "Yes"])

        elif i< 18:

            tabulte_list.append([incresingorder_ind,"Variation", "Yes"])

        if ((i > 17) & (i not in removed_ind)) :

            word = train_text_features[i]

            yes_no = True if word in text.split() else False

            if yes_no:

                word_present += 1

            tabulte_list.append([incresingorder_ind,train_text_features[i], yes_no])

        incresingorder_ind += 1

    print(word_present, "most importent features are present in our query point")

    print("-"*50)

    print("The features that are most importent of the ",predicted_cls[0]," class:")

    print (tabulate(tabulte_list, headers=["Index",'Feature name', 'Present or Not']))
# from tabulate import tabulate

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
alpha = [10 ** x for x in range(-6, 1)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 

print("")

print_best_alpha(alpha, cv_log_error_array)
best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
alpha = [10 ** x for x in range(-5, 3)]

cv_log_error_array = []

for i in alpha:

    print("for C =", i)

#     clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')

    clf = SGDClassifier( class_weight='balanced', alpha=i, penalty='l2', loss='hinge', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(cv_y, sig_clf_probs))

print("")

print_best_alpha(alpha, cv_log_error_array)
best_alpha = np.argmin(cv_log_error_array)

# clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42,class_weight='balanced')

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,cv_y, clf)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

# test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.coef_)[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
alpha = [100,200,500,1000,2000]

max_depth = [5, 10]

cv_log_error_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_onehotCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_onehotCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 
best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,cv_y, clf)
# test_point_index = 10

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



test_point_index = 1

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.feature_importances_)

print("-"*50)

get_impfeature_names(indices[:no_feature], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
alpha = [10,50,100,200,500,1000]

max_depth = [2,3,5,10]

cv_log_error_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_responseCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_responseCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)

        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 





best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42, n_jobs=-1)

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
clf = RandomForestClassifier(max_depth=max_depth[int(best_alpha%4)], n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_features='auto',random_state=42)

predict_and_plot_confusion_matrix(train_x_responseCoding, train_y,cv_x_responseCoding,cv_y, clf)
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42, n_jobs=-1)

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)





test_point_index = 1

no_feature = 27

predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_responseCoding[test_point_index].reshape(1,-1)),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.feature_importances_)

print("-"*50)

for i in indices:

    if i<9:

        print("Gene is important feature")

    elif i<18:

        print("Variation is important feature")

    else:

        print("Text is important feature")
clf1 = SGDClassifier(alpha=0.001, penalty='l2', loss='log', class_weight='balanced', random_state=0)

clf1.fit(train_x_onehotCoding, train_y)

sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")



clf2 = SGDClassifier(alpha=1, penalty='l2', loss='hinge', class_weight='balanced', random_state=0)

clf2.fit(train_x_onehotCoding, train_y)

sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")





clf3 = MultinomialNB(alpha=0.001)

clf3.fit(train_x_onehotCoding, train_y)

sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")



sig_clf1.fit(train_x_onehotCoding, train_y)

print("Logistic Regression :  Log Loss: %0.2f" % (log_loss(cv_y, sig_clf1.predict_proba(cv_x_onehotCoding))))

sig_clf2.fit(train_x_onehotCoding, train_y)

print("Support vector machines : Log Loss: %0.2f" % (log_loss(cv_y, sig_clf2.predict_proba(cv_x_onehotCoding))))

sig_clf3.fit(train_x_onehotCoding, train_y)

print("Naive Bayes : Log Loss: %0.2f" % (log_loss(cv_y, sig_clf3.predict_proba(cv_x_onehotCoding))))

print("-"*50)

alpha = [0.0001,0.001,0.01,0.1,1,10] 

best_alpha = 999

for i in alpha:

    lr = LogisticRegression(C=i)

    sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

    sclf.fit(train_x_onehotCoding, train_y)

    print("Stacking Classifer : for the value of alpha: %f Log Loss: %0.3f" % (i, log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))))

    log_error =log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))

    if best_alpha > log_error:

        best_alpha = log_error
lr = LogisticRegression(C=0.1)

sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

sclf.fit(train_x_onehotCoding, train_y)



log_error = log_loss(train_y, sclf.predict_proba(train_x_onehotCoding))

print("Log loss (train) on the stacking classifier :",log_error)



log_error = log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))

print("Log loss (CV) on the stacking classifier :",log_error)



log_error = log_loss(test_y, sclf.predict_proba(test_x_onehotCoding))

print("Log loss (test) on the stacking classifier :",log_error)



print("Number of missclassified point :", np.count_nonzero((sclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0])

plot_confusion_matrix(test_y=test_y, predict_y=sclf.predict(test_x_onehotCoding))
#Refer:http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

from sklearn.ensemble import VotingClassifier

vclf = VotingClassifier(estimators=[('lr', sig_clf1), ('svc', sig_clf2), ('rf', sig_clf3)], voting='soft')

vclf.fit(train_x_onehotCoding, train_y)

print("Log loss (train) on the VotingClassifier :", log_loss(train_y, vclf.predict_proba(train_x_onehotCoding)))

print("Log loss (CV) on the VotingClassifier :", log_loss(cv_y, vclf.predict_proba(cv_x_onehotCoding)))

print("Log loss (test) on the VotingClassifier :", log_loss(test_y, vclf.predict_proba(test_x_onehotCoding)))

print("Number of missclassified point :", np.count_nonzero((vclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0])

plot_confusion_matrix(test_y=test_y, predict_y=vclf.predict(test_x_onehotCoding))