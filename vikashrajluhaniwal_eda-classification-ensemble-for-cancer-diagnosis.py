import warnings

warnings.filterwarnings("ignore")

from nltk.corpus import stopwords

import re

import math

from collections import Counter, defaultdict

from scipy.sparse import hstack

import numpy as np

import pandas as pd

from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score, log_loss

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import StackingClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
df = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip')

print('Dataframe shape: ', df.shape)

print('Features names: ', df.columns.values)

df.head()
df_text =pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_text.zip",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)

print('Text data shape: ', df_text.shape)

print('Features names : ', df_text.columns.values)

df_text.head()
stop_words = set(stopwords.words('english'))

for i,text in enumerate(df_text["TEXT"]):

    if type(df_text["TEXT"][i]) is not str:

        print("no text description available at index : ", i)

    else:

        string = ""

        df_text["TEXT"][i] = str(df_text["TEXT"][i]).lower()

        df_text["TEXT"][i] = re.sub("\W"," ",df_text["TEXT"][i])

        df_text["TEXT"][i] = re.sub('\s+',' ', df_text["TEXT"][i])

        for word in df_text["TEXT"][i].split():

            if not word in stop_words:

                string += word + " "

        df_text["TEXT"][i] = string
df["Gene"] = df["Gene"].str.replace('\s+', '_')

df["Variation"] = df["Variation"].str.replace('\s+', '_')
final_df = pd.merge(df, df_text,on='ID', how='left')

final_df.head()
final_df[final_df.isna().any(axis=1)]
#imputing the missing values

final_df.loc[final_df['TEXT'].isna(),'TEXT'] = final_df['Gene'] +' '+final_df['Variation']
y_label = final_df["Class"].values

X_train, X_test, y_train, y_test = train_test_split(final_df, y_label, stratify=y_label, test_size=0.2)

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
print('Training data size :', X_train.shape)

print('test data size :', X_test.shape)

print('Validation data size :', X_cv.shape)
def dist_class(df, name):

    sns.countplot(df["Class"])

    plt.title("Bar plot of Class using {} data".format(name))

    print("Frequency of each class in {} data".format(name))

    for i in df["Class"].value_counts().index:

        print("Number of observations in class ", i," is : ",df["Class"].value_counts()[i], "(", np.round((df["Class"].value_counts()[i] / len(df["Class"]))*100,2), "%)")
dist_class(X_train,"training")
dist_class(X_cv,"validation")
dist_class(X_test,"test")
#user defined function to plot confusion matrix, precision and recall for a ML model

def plot_confusion_recall_precision(cm):

    labels = [1,2,3,4,5,6,7,8,9]

    print("="*30, "Confusion matrix", "="*30)

    plt.figure(figsize=(16,8))

    sns.heatmap(cm, annot=True, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    precision_matrix =(cm/cm.sum(axis=0))

    print("="*30, "Precision matrix (columm sum=1)", "="*30)

    plt.figure(figsize=(16,8))

    sns.heatmap(precision_matrix, annot=True, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    recall_matrix =(((cm.T)/(cm.sum(axis=1))).T)

    print("="*30, "Recall matrix (row sum=1)", "="*30)

    plt.figure(figsize=(16,8))

    sns.heatmap(recall_matrix, annot=True, cmap=plt.cm.Blues, xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
test_len = X_test.shape[0]

cv_len = X_cv.shape[0]

cv_y_pred = np.zeros((cv_len,9))

for i in range(cv_len):

    rand_probs = np.random.rand(1,9)

    cv_y_pred[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on validation data using Random Model",log_loss(y_cv,cv_y_pred, eps=1e-15))

test_y_pred = np.zeros((test_len,9))

for i in range(test_len):

    rand_probs = np.random.rand(1,9)

    test_y_pred[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Test Data using Random Model",log_loss(y_test,test_y_pred, eps=1e-15))

y_pred = np.argmax(test_y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred+1)

plot_confusion_recall_precision(cm)
#user defined functions to fet feature representations

def get_column_fea_dict(alpha, column, df):

    freq = X_train[column].value_counts()

    column_dict = dict()

    for i, denominator in freq.items():

        vec = []

        for k in range(1,10):

            subset = X_train.loc[(X_train['Class']==k) & (X_train[column]==i)]

            vec.append((subset.shape[0] + alpha*10)/ (denominator + 90*alpha))

        column_dict[i]=vec

    return column_dict

def get_column_feature(alpha, column, df):

    column_dict = get_column_fea_dict(alpha, column, df)

    freq = X_train[column].value_counts()

    column_fea = []

    for index, row in df.iterrows():

        if row[column] in dict(freq).keys():

            column_fea.append(column_dict[row[column]])

        else:

            column_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])

    return column_fea
print('Number of Unique Genes :', X_train["Gene"].nunique())

freq_genes = X_train['Gene'].value_counts()

print("Top 10 genes with highest frequency")

freq_genes.head(10)
plt.figure(figsize = (20,8))

sns.countplot(freq_genes)

plt.xticks(rotation = 90)

plt.xlabel('Index of a Gene based on their decreasing order of frequency')

plt.title('Bar plot of most oftenly occuring Genes')
alpha = 1

train_gene_feat_resp_coding = np.array(get_column_feature(alpha, "Gene", X_train))

val_gene_feat_resp_coding = np.array(get_column_feature(alpha, "Gene", X_cv))

test_gene_feat_resp_coding = np.array(get_column_feature(alpha, "Gene", X_test))
print("shape of training gene feature after response coding :", train_gene_feat_resp_coding.shape)
gene_vectorizer = CountVectorizer()

train_gene_feat_onehot_en = gene_vectorizer.fit_transform(X_train['Gene'])

val_gene_feat_onehot_en = gene_vectorizer.transform(X_cv['Gene'])

test_gene_feat_onehot_en = gene_vectorizer.transform(X_test['Gene'])
gene_vectorizer.get_feature_names()
print("shape of training gene feature after one hot encoding :", train_gene_feat_onehot_en.shape)
alpha = [10 ** x for x in range(-5, 1)]

val_log_loss_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_gene_feat_onehot_en, y_train)

    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

    calib_clf.fit(train_gene_feat_onehot_en, y_train)

    y_pred = calib_clf.predict_proba(val_gene_feat_onehot_en)

    val_log_loss_array.append(log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

    print('For alpha = ', i, "The log loss is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

plt.plot(alpha, val_log_loss_array)

for i, logloss in enumerate(np.round(val_log_loss_array,3)):

    plt.annotate((alpha[i],np.round(logloss,3)), (alpha[i],logloss))

plt.grid()

plt.title("Validation log loss for different values of alpha")

plt.xlabel("Alpha")

plt.ylabel("Log loss")
best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_gene_feat_onehot_en, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(train_gene_feat_onehot_en, y_train)



y_pred = calib_clf.predict_proba(train_gene_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is : ",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(val_gene_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is: ",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(test_gene_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is : ",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
print("Number of observations in test and validation datasets covered by the unique ", X_train["Gene"].nunique(), " genes in train dataset")



test_cover=X_test[X_test['Gene'].isin(list(X_train['Gene'].unique()))].shape[0]

validation_cover=X_cv[X_cv['Gene'].isin(list(X_train['Gene'].unique()))].shape[0]



print('In test data',test_cover, 'out of',X_test.shape[0], ":",(test_cover/X_test.shape[0])*100)

print('2. In cross validation data',validation_cover, 'out of ',X_cv.shape[0],":" ,(validation_cover/X_cv.shape[0])*100)
print('Number of Unique Variations :', X_train["Variation"].nunique())

freq_variations = X_train['Variation'].value_counts()

print("Top 10 variations with highest frequency")

freq_variations.head(10)
total_variations = sum(freq_variations.values)

fraction = freq_variations.values/total_variations

plt.plot(fraction, label="Histrogram of Variations")

plt.xlabel('Index of a variations based on their decreasing order of frequency')

plt.ylabel('Frequency')

plt.legend()

plt.grid()
alpha = 1

train_variation_feat_resp_coding = np.array(get_column_feature(alpha, "Variation", X_train))

val_variation_feat_resp_coding = np.array(get_column_feature(alpha, "Variation", X_cv))

test_variation_feat_resp_coding = np.array(get_column_feature(alpha, "Variation", X_test))
print("shape of training variation feature after response coding :", train_variation_feat_resp_coding.shape)
variation_vectorizer = CountVectorizer()

train_variation_feat_onehot_en = variation_vectorizer.fit_transform(X_train['Variation'])

val_variation_feat_onehot_en = variation_vectorizer.transform(X_cv['Variation'])

test_variation_feat_onehot_en = variation_vectorizer.transform(X_test['Variation'])
variation_vectorizer.get_feature_names()
print("shape of training varaition feature after one hot encoding :", train_variation_feat_onehot_en.shape)
alpha = [10 ** x for x in range(-5, 1)]

val_log_loss_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_variation_feat_onehot_en, y_train)

    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

    calib_clf.fit(train_variation_feat_onehot_en, y_train)

    y_pred = calib_clf.predict_proba(val_variation_feat_onehot_en)

    val_log_loss_array.append(log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

    print('For alpha = ', i, "The log loss is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

plt.plot(alpha, val_log_loss_array)

for i, logloss in enumerate(np.round(val_log_loss_array,3)):

    plt.annotate((alpha[i],np.round(logloss,3)), (alpha[i],logloss))

plt.grid()

plt.title("Validation log loss for different values of alpha")

plt.xlabel("Alpha")

plt.ylabel("Log loss")
best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_variation_feat_onehot_en, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(train_variation_feat_onehot_en, y_train)



y_pred = calib_clf.predict_proba(train_variation_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(val_variation_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(test_variation_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
print("Number of observations in test and validation datasets covered by the unique ", X_train["Variation"].nunique(), " variations in train dataset")



test_cover=X_test[X_test['Variation'].isin(list(X_train['Variation'].unique()))].shape[0]

validation_cover=X_cv[X_cv['Variation'].isin(list(X_train['Variation'].unique()))].shape[0]



print('In test data',test_cover, 'out of',X_test.shape[0], ":",(test_cover/X_test.shape[0])*100)

print('2. In cross validation data',validation_cover, 'out of ',X_cv.shape[0],":" ,(validation_cover/X_cv.shape[0])*100)
#using Bag of words approach

text_vectorizer = CountVectorizer(min_df=3)

train_text_feat_onehot_en = text_vectorizer.fit_transform(X_train['TEXT'])

train_text_features= text_vectorizer.get_feature_names()

train_text_feat_counts = train_text_feat_onehot_en.sum(axis=0).A1

text_feat_dict = dict(zip(list(train_text_features),train_text_feat_counts))

print("Total number of unique words in TEXT feature of training data :", len(train_text_features))
def get_word_count_dictionary(df_cls):

    dic = defaultdict(int)

    for index, row in df_cls.iterrows():

        for word in row['TEXT'].split():

            dic[word] +=1

    return dic
def get_text_resp_coding(df):

    text_feat_resp_coding = np.zeros((df.shape[0],9))

    for i in range(0,9):

        row_index = 0

        for index, row in df.iterrows():

            total_prob = 0

            for word in row['TEXT'].split():

                total_prob += math.log(((dic_list[i].get(word,0)+10 )/(total_dic.get(word,0)+90)))

            text_feat_resp_coding[row_index][i] = math.exp(total_prob/len(row['TEXT'].split()))

            row_index += 1

    return text_feat_resp_coding
dic_list = []

for i in range(1,10):

    subset_cls = X_train[X_train['Class']==i]

    dic_list.append(get_word_count_dictionary(subset_cls))

total_dic = get_word_count_dictionary(X_train)
train_text_feat_resp_coding  = get_text_resp_coding(X_train)

val_text_feat_resp_coding  = get_text_resp_coding(X_cv)

test_text_feat_resp_coding  = get_text_resp_coding(X_test)
train_text_feat_resp_coding = (train_text_feat_resp_coding.T/train_text_feat_resp_coding.sum(axis=1)).T

val_text_feat_resp_coding = (val_text_feat_resp_coding.T/val_text_feat_resp_coding.sum(axis=1)).T

test_text_feat_resp_coding = (test_text_feat_resp_coding.T/test_text_feat_resp_coding.sum(axis=1)).T
train_text_feat_onehot_en = normalize(train_text_feat_onehot_en, axis=0)

test_text_feat_onehot_en = text_vectorizer.transform(X_test['TEXT'])

test_text_feat_onehot_en = normalize(test_text_feat_onehot_en, axis=0)

val_text_feat_onehot_en = text_vectorizer.transform(X_cv['TEXT'])

val_text_feat_onehot_en = normalize(val_text_feat_onehot_en, axis=0)
sorted_text_feat_dict = dict(sorted(text_feat_dict.items(), key=lambda x: x[1] , reverse=True))

sorted_text_occur = np.array(list(sorted_text_feat_dict.values()))
# Number of words to a given frequency

print(Counter(sorted_text_occur))
alpha = [10 ** x for x in range(-5, 1)]

val_log_loss_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_text_feat_onehot_en, y_train)

    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

    calib_clf.fit(train_text_feat_onehot_en, y_train)

    y_pred = calib_clf.predict_proba(val_text_feat_onehot_en)

    val_log_loss_array.append(log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

    print('For alpha = ', i, "The log loss is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

plt.plot(alpha, val_log_loss_array)

for i, logloss in enumerate(np.round(val_log_loss_array,3)):

    plt.annotate((alpha[i],np.round(logloss,3)), (alpha[i],logloss))

plt.grid()

plt.title("Validation log loss for different values of alpha")

plt.xlabel("Alpha")

plt.ylabel("Log loss")

best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_text_feat_onehot_en, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(train_text_feat_onehot_en, y_train)



y_pred = calib_clf.predict_proba(train_text_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(val_text_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(test_text_feat_onehot_en)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
def get_common_word(df):

    text_vectorizer = CountVectorizer(min_df=3)

    df_text_feat_onehot_en = text_vectorizer.fit_transform(df['TEXT'])

    df_text_features = text_vectorizer.get_feature_names()



    df_text_feat_counts = df_text_feat_onehot_en.sum(axis=0).A1

    df_text_fea_dict = dict(zip(list(df_text_features),df_text_feat_counts))

    df_len = len(set(df_text_features))

    common_words_len = len(set(train_text_features) & set(df_text_features))

    return df_len,common_words_len
cv_len,common_words_len = get_common_word(X_cv)

print(np.round((common_words_len/cv_len)*100, 3), "% of word of validation appeared in train data")

test_len,common_words_len = get_common_word(X_test)

print(np.round((common_words_len/test_len)*100, 3), "% of word of test data appeared in train data")
#user defined function to calculate confusion matrix, precision and recall and also to plot

def predict_and_plot_confusion_recall_precision(X_train, y_train,X_test, y_test, classifier):

    classifier.fit(X_train, y_train)

    calib_clf = CalibratedClassifierCV(classifier, method="sigmoid")

    calib_clf.fit(X_train, y_train)

    y_pred = calib_clf.predict(X_test)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

    print("Log loss :",log_loss(y_test, calib_clf.predict_proba(X_test)))

    # calculating the number of data points that are misclassified

    print("Number of mis-classified points :", np.count_nonzero((y_pred- y_test))/y_test.shape[0])

    cm = confusion_matrix(y_test, y_pred)

    plot_confusion_recall_precision(cm)
#user defined function to calculate log loss

def calculate_log_loss(X_train, y_train,X_test, y_test, classifier):

    classifier.fit(X_train, y_train)

    calib_clf = CalibratedClassifierCV(classifier, method="sigmoid")

    calib_clf.fit(X_train, y_train)

    calib_clf_probs = calib_clf.predict_proba(X_test)

    return log_loss(y_test, calib_clf_probs, eps=1e-15)
# user defined function to get important feature 

def get_impfeature_names(indices, text, gene, var, no_features):

    gene_count_vectorizer = CountVectorizer()

    var_count_vectorizer = CountVectorizer()

    text_count_vectorizer = CountVectorizer(min_df=3)

    

    gene_vec_onehot = gene_count_vectorizer.fit(X_train['Gene'])

    var_vec_onehot  = var_count_vectorizer.fit(X_train['Variation'])

    text_vec_onehot = text_count_vectorizer.fit(X_train['TEXT'])

    

    feat1_len = len(gene_count_vectorizer.get_feature_names())

    feat2_len = len(var_count_vectorizer.get_feature_names())

    

    word_present = 0

    for i,v in enumerate(indices):

        if (v < feat1_len):

            word = gene_count_vectorizer.get_feature_names()[v]

            flag = True if word == gene else False

            if flag:

                word_present += 1

                print(i, "Gene feature [{}] present in test data point [{}]".format(word,flag))

        elif (v < feat1_len+feat2_len):

            word = var_count_vectorizer.get_feature_names()[v-(feat1_len)]

            flag = True if word == var else False

            if flag:

                word_present += 1

                print(i, "variation feature [{}] present in test data point [{}]".format(word,flag))

        else:

            word = text_count_vectorizer.get_feature_names()[v-(feat1_len+feat2_len)]

            flag = True if word in text.split() else False

            if flag:

                word_present += 1

                print(i, "Text feature [{}] present in test data point [{}]".format(word,flag))



    print("Out of the top ",no_features," features ", word_present, "are present in query point")
train_gene_and_var_onehot_en = hstack((train_gene_feat_onehot_en,train_variation_feat_onehot_en))

val_gene_and_var_onehot_en = hstack((val_gene_feat_onehot_en,val_variation_feat_onehot_en))

test_gene_and_var_onehot_en = hstack((test_gene_feat_onehot_en,test_variation_feat_onehot_en))
X_train_onehotCoding = hstack((train_gene_and_var_onehot_en, train_text_feat_onehot_en)).tocsr()

y_train = np.array(X_train['Class'])

X_test_onehotCoding = hstack((test_gene_and_var_onehot_en, test_text_feat_onehot_en)).tocsr()

y_test = np.array(X_test['Class'])

X_cv_onehotCoding = hstack((val_gene_and_var_onehot_en, val_text_feat_onehot_en)).tocsr()

y_cv = np.array(X_cv['Class'])
train_gene_and_var_responseCoding = np.hstack((train_gene_feat_resp_coding,train_variation_feat_resp_coding))

test_gene_and_var_responseCoding = np.hstack((test_gene_feat_resp_coding,test_variation_feat_resp_coding))

val_gene_and_var_responseCoding = np.hstack((val_gene_feat_resp_coding,val_variation_feat_resp_coding))



X_train_responseCoding = np.hstack((train_gene_and_var_responseCoding, train_text_feat_resp_coding))

X_test_responseCoding = np.hstack((test_gene_and_var_responseCoding, test_text_feat_resp_coding))

X_cv_responseCoding = np.hstack((val_gene_and_var_responseCoding, val_text_feat_resp_coding))
print("Overview about one hot encoding features : ")

print("Size of one hot encoded train data : ", X_train_onehotCoding.shape)

print("Size of one hot encoded test data : ", X_test_onehotCoding.shape)

print("Size of one hot encoded validation data : ", X_cv_onehotCoding.shape)
print(" Overview about response coded features :")

print("Size of response coded train data : ", X_train_responseCoding.shape)

print("Size of response coded test data : ", X_test_responseCoding.shape)

print("Size of response coded validation data ", X_cv_responseCoding.shape)
alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]

val_log_loss_array = []

for i in alpha:

    print("for alpha =", i)

    clf = MultinomialNB(alpha=i)

    clf.fit(X_train_onehotCoding, y_train)

    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

    calib_clf.fit(X_train_onehotCoding, y_train)

    calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)

    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 

plt.plot(np.log10(alpha), val_log_loss_array)

for i, logloss in enumerate(np.round(val_log_loss_array,3)):

    plt.annotate((alpha[i],str(logloss)), (np.log10(alpha[i]),logloss))

plt.grid()

plt.xticks(np.log10(alpha))

plt.title("Validation log loss for different values of alpha")

plt.xlabel("Alpha")

plt.ylabel("Log loss")



best_alpha = np.argmin(val_log_loss_array)

clf = MultinomialNB(alpha=i)

clf.fit(X_train_onehotCoding, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(X_train_onehotCoding, y_train)



y_pred = calib_clf.predict_proba(X_train_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_cv_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_test_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
best_alpha = np.argmin(val_log_loss_array)

clf = MultinomialNB(alpha=alpha[best_alpha])

predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)
random_index = 1

no_feature = 100

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]

print("="*50)

get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
random_index = 100

no_feature = 100

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]

print("="*50)

get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
alpha = [5, 11, 15, 21, 31, 41, 51, 99]

val_log_loss_array = []

for i in alpha:

    print("for alpha =", i)

    clf = KNeighborsClassifier(n_neighbors=i)

    clf.fit(X_train_responseCoding, y_train)

    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

    calib_clf.fit(X_train_responseCoding, y_train)

    calib_clf_probs = calib_clf.predict_proba(X_cv_responseCoding)

    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 

plt.plot(alpha, val_log_loss_array)

for i, logloss in enumerate(np.round(val_log_loss_array,3)):

    plt.annotate((alpha[i],str(logloss)), (alpha[i],logloss))

plt.grid()

plt.xticks(alpha)

plt.title("Validation log loss for different values of alpha")

plt.xlabel("Alpha")

plt.ylabel("Log loss")



best_alpha = np.argmin(val_log_loss_array)

clf = KNeighborsClassifier(n_neighbors=i)

clf.fit(X_train_responseCoding, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(X_train_responseCoding, y_train)



y_pred = calib_clf.predict_proba(X_train_responseCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_cv_responseCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_test_responseCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
best_alpha = np.argmin(val_log_loss_array)

clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

predict_and_plot_confusion_recall_precision(X_train_responseCoding, y_train,X_cv_responseCoding, y_cv, clf)
random_index = 1

pred_cls = calib_clf.predict(X_test_responseCoding[random_index].reshape(1,-1))

print("Predicted Class :", pred_cls[0])

print("Actual Class :", y_test[random_index])

nearest_neighbors = clf.kneighbors(X_test_responseCoding[random_index].reshape(1, -1), alpha[best_alpha])

print("The ",alpha[best_alpha]," nearest neighbours of the random test point belongs to classes",y_train[nearest_neighbors[1][0]])

print("Fequency of nearest points :",Counter(y_train[nearest_neighbors[1][0]]))
random_index = 100

pred_cls = calib_clf.predict(X_test_responseCoding[random_index].reshape(1,-1))

print("Predicted Class :", pred_cls[0])

print("Actual Class :", y_test[random_index])

nearest_neighbors = clf.kneighbors(X_test_responseCoding[random_index].reshape(1, -1), alpha[best_alpha])

print("The ",alpha[best_alpha]," nearest neighbours of the random test point belongs to classes",y_train[nearest_neighbors[1][0]])

print("Fequency of nearest points :",Counter(y_train[nearest_neighbors[1][0]]))
alpha = [10 ** x for x in range(-6, 3)]

val_log_loss_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(X_train_onehotCoding, y_train)

    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

    calib_clf.fit(X_train_onehotCoding, y_train)

    calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)

    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 

plt.plot(alpha, val_log_loss_array)

for i, logloss in enumerate(np.round(val_log_loss_array,3)):

    plt.annotate((alpha[i],str(logloss)), (alpha[i],logloss))

plt.grid()

plt.xticks(alpha)

plt.title("Validation log loss for different values of alpha")

plt.xlabel("Alpha")

plt.ylabel("Log loss")



best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(X_train_onehotCoding, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(X_train_onehotCoding, y_train)



y_pred = calib_clf.predict_proba(X_train_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_cv_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_test_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)
random_index = 1

no_feature = 500

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]

print("="*50)

get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
random_index = 100

no_feature = 500

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]

print("="*50)

get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
alpha = [10 ** x for x in range(-6, 1)]

val_log_loss_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(X_train_onehotCoding, y_train)

    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

    calib_clf.fit(X_train_onehotCoding, y_train)

    calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)

    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 

plt.plot(alpha, val_log_loss_array)

for i, logloss in enumerate(np.round(val_log_loss_array,3)):

    plt.annotate((alpha[i],str(logloss)), (alpha[i],logloss))

plt.grid()

plt.xticks(alpha)

plt.title("Validation log loss for different values of alpha")

plt.xlabel("Alpha")

plt.ylabel("Log loss")



best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(X_train_onehotCoding, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(X_train_onehotCoding, y_train)



y_pred = calib_clf.predict_proba(X_train_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_cv_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_test_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)
random_index = 1

no_feature = 500

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]

print("="*50)

get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
random_index = 100

no_feature = 500

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]

print("="*50)

get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
alpha = [10 ** x for x in range(-5, 3)]

val_log_loss_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(class_weight='balanced',alpha=i, penalty='l2', loss='hinge', random_state=42)

    clf.fit(X_train_onehotCoding, y_train)

    calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

    calib_clf.fit(X_train_onehotCoding, y_train)

    calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)

    val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))

    print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 

plt.plot(alpha, val_log_loss_array)

for i, logloss in enumerate(np.round(val_log_loss_array,3)):

    plt.annotate((alpha[i],str(logloss)), (alpha[i],logloss))

plt.grid()

plt.xticks(alpha)

plt.title("Validation log loss for different values of alpha")

plt.xlabel("Alpha")

plt.ylabel("Log loss")



best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)

clf.fit(X_train_onehotCoding, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(X_train_onehotCoding, y_train)



y_pred = calib_clf.predict_proba(X_train_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_cv_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_test_onehotCoding)

print('For the best alpha of alpha = ', alpha[best_alpha], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
best_alpha = np.argmin(val_log_loss_array)

clf = SGDClassifier(class_weight='balanced',alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)

predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)
random_index = 1

no_feature = 500

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]

print("="*50)

get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
random_index = 100

no_feature = 500

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(abs(-clf.coef_))[pred_cls-1][:,:no_feature]

print("="*50)

get_impfeature_names(indices[0], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
alpha = [100,200,500,1000,2000]

max_depth = [5, 10]

val_log_loss_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42)

        clf.fit(X_train_onehotCoding, y_train)

        calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

        calib_clf.fit(X_train_onehotCoding, y_train)

        calib_clf_probs = calib_clf.predict_proba(X_cv_onehotCoding)

        val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 



best_alpha = np.argmin(val_log_loss_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42)

clf.fit(X_train_onehotCoding, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(X_train_onehotCoding, y_train)
y_pred = calib_clf.predict_proba(X_train_onehotCoding)

print('For the best alpha of alpha = ', alpha[int(best_alpha/2)], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_cv_onehotCoding)

print('For the best alpha of alpha = ', alpha[int(best_alpha/2)], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_test_onehotCoding)

print('For the best alpha of alpha = ', alpha[int(best_alpha/2)], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42)

predict_and_plot_confusion_recall_precision(X_train_onehotCoding, y_train,X_cv_onehotCoding, y_cv, clf)
random_index = 1

no_feature = 100

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(-clf.feature_importances_)

print("="*50)

get_impfeature_names(indices[:no_feature], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
random_index = 100

no_feature = 100

pred_cls = calib_clf.predict(X_test_onehotCoding[random_index])

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_onehotCoding[random_index]),4))

print("Actual Class :", y_test[random_index])

indices=np.argsort(-clf.feature_importances_)

print("="*50)

get_impfeature_names(indices[:no_feature], X_test['TEXT'].iloc[random_index],X_test['Gene'].iloc[random_index],X_test['Variation'].iloc[random_index], no_feature)
alpha = [10,50,100,200,500,1000]

max_depth = [2,3,5,10]

val_log_loss_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42)

        clf.fit(X_train_responseCoding, y_train)

        calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

        calib_clf.fit(X_train_responseCoding, y_train)

        calib_clf_probs = calib_clf.predict_proba(X_cv_responseCoding)

        val_log_loss_array.append(log_loss(y_cv, calib_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(y_cv, calib_clf_probs)) 



best_alpha = np.argmin(val_log_loss_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42)

clf.fit(X_train_responseCoding, y_train)

calib_clf = CalibratedClassifierCV(clf, method="sigmoid")

calib_clf.fit(X_train_responseCoding, y_train)



y_pred = calib_clf.predict_proba(X_train_responseCoding)

print('For the best alpha of alpha = ', alpha[int(best_alpha/4)], "The log loss on training data is:",log_loss(y_train, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_cv_responseCoding)

print('For the best alpha of alpha = ', alpha[int(best_alpha/4)], "The log loss on validation data is:",log_loss(y_cv, y_pred, labels=clf.classes_, eps=1e-15))

y_pred = calib_clf.predict_proba(X_test_responseCoding)

print('For the best alpha of alpha = ', alpha[int(best_alpha/4)], "The log loss on test data is:",log_loss(y_test, y_pred, labels=clf.classes_, eps=1e-15))
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42)

predict_and_plot_confusion_recall_precision(X_train_responseCoding, y_train,X_cv_responseCoding, y_cv, clf)
random_index = 1

pred_cls = calib_clf.predict(X_test_responseCoding[random_index].reshape(1,-1))

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_responseCoding[random_index].reshape(1,-1)),4))

print("Actual Class :", y_test[random_index])

indices = np.argsort(-clf.feature_importances_)

print("="*50)

for i in indices:

    if i<9:

        print("Gene is important feature")

    elif i<18:

        print("Variation is important feature")

    else:

        print("Text is important feature")
random_index = 100

pred_cls = calib_clf.predict(X_test_responseCoding[random_index].reshape(1,-1))

print("Predicted Class :", pred_cls[0])

print("Predicted Class Probabilities:", np.round(calib_clf.predict_proba(X_test_responseCoding[random_index].reshape(1,-1)),4))

print("Actual Class :", y_test[random_index])

indices = np.argsort(-clf.feature_importances_)

print("="*50)

for i in indices:

    if i<9:

        print("Gene is important feature")

    elif i<18:

        print("Variation is important feature")

    else:

        print("Text is important feature")
clf1 = SGDClassifier(alpha=0.001, penalty='l2', loss='log', class_weight='balanced', random_state=42)

clf1.fit(X_train_onehotCoding, y_train)

calib_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")



clf2 = SGDClassifier(alpha=1, penalty='l2', loss='hinge', class_weight='balanced', random_state=42)

clf2.fit(X_train_onehotCoding, y_train)

calib_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")





clf3 = MultinomialNB(alpha=0.001)

clf3.fit(X_train_onehotCoding, y_train)

calib_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")



calib_clf1.fit(X_train_onehotCoding, y_train)

print("Logistic Regression :  Log Loss: %0.2f" % (log_loss(y_cv, calib_clf1.predict_proba(X_cv_onehotCoding))))

calib_clf2.fit(X_train_onehotCoding, y_train)

print("SVM : Log Loss: %0.2f" % (log_loss(y_cv, calib_clf2.predict_proba(X_cv_onehotCoding))))

calib_clf3.fit(X_train_onehotCoding, y_train)

print("Naive Bayes : Log Loss: %0.2f" % (log_loss(y_cv, calib_clf3.predict_proba(X_cv_onehotCoding))))

print("="*50)

alpha = [0.0001,0.001,0.01,0.1,1,10] 

best_alpha = 999

for i in alpha:

    lr = LogisticRegression(C=i)

    stack_clf = StackingClassifier(classifiers=[calib_clf1, calib_clf2, calib_clf3], meta_classifier=lr, use_probas=True)

    stack_clf.fit(X_train_onehotCoding, y_train)

    print("Stacking Classifer : for the value of alpha: %f Log Loss: %0.3f" % (i, log_loss(y_cv, stack_clf.predict_proba(X_cv_onehotCoding))))

    logloss =log_loss(y_cv, stack_clf.predict_proba(X_cv_onehotCoding))

    if best_alpha > logloss:

        best_alpha = logloss
lr = LogisticRegression(C=best_alpha)

stack_clf = StackingClassifier(classifiers=[calib_clf1, calib_clf2, calib_clf3], meta_classifier=lr, use_probas=True)

stack_clf.fit(X_train_onehotCoding, y_train)



logloss = log_loss(y_train, stack_clf.predict_proba(X_train_onehotCoding))

print("Log loss of training data using the stacking classifier :",logloss)



logloss = log_loss(y_cv, stack_clf.predict_proba(X_cv_onehotCoding))

print("Log loss of validation data using the stacking classifier :",logloss)



logloss = log_loss(y_test, stack_clf.predict_proba(X_test_onehotCoding))

print("Log loss of test data using the stacking classifier :",logloss)



print("Number of missclassified point :", np.count_nonzero((stack_clf.predict(X_test_onehotCoding)- y_test))/y_test.shape[0])

cm = confusion_matrix(y_test, stack_clf.predict(X_test_onehotCoding))

plot_confusion_recall_precision(cm)
voting_clf = VotingClassifier(estimators=[('lr', calib_clf1), ('svc', calib_clf2), ('rf', calib_clf3)], voting='soft')

voting_clf.fit(X_train_onehotCoding, y_train)

print("Log loss (train) on the VotingClassifier :", log_loss(y_train, voting_clf.predict_proba(X_train_onehotCoding)))

print("Log loss (CV) on the VotingClassifier :", log_loss(y_cv, voting_clf.predict_proba(X_cv_onehotCoding)))

print("Log loss (test) on the VotingClassifier :", log_loss(y_test, voting_clf.predict_proba(X_test_onehotCoding)))

print("Number of missclassified point :", np.count_nonzero((voting_clf.predict(X_test_onehotCoding)- y_test))/y_test.shape[0])

cm = confusion_matrix(y_test, voting_clf.predict(X_test_onehotCoding))

plot_confusion_recall_precision(cm)