import numpy as np

import pandas as pd

import json

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

#from keras.models import Sequential

#from keras import layers

from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

from IPython import display



%matplotlib inline

import os

print(os.listdir("../input/mobilefull/mobile_image_resized/"))

with open('../input/ndsc-beginner/categories.json', 'r') as f:

    categories_dict = json.load(f)

    

category_list = []

for i in range(0,58):

    for cat_name, upper_dict in categories_dict.items():

        for subcat_name, value in upper_dict.items():

            if value == i:

                category_list.append((cat_name,subcat_name))
train_df=pd.read_csv('../input/ndsc-beginner/train.csv')

train_df['image_path'] = train_df['image_path'].str.replace('.jpg', '') + '.jpg'

train_df['image_path'] = train_df['image_path'].str.replace('mobile_image','../input/mobilefull/mobile_image_resized/mobile_image_resized/train') 

train_df.head()
def category_lookup(row):

    return category_list[row['Category']][1]



def main_category_lookup(row):

    return category_list[row['Category']][0]

    

train_df['Category_Name'] = train_df.apply(lambda row: category_lookup(row), axis=1)

train_df['Main_Category_Name'] = train_df.apply(lambda row: main_category_lookup(row), axis=1)

train_df.head()
def train_maincat(df,main_category):

    return train_df[train_df['Main_Category_Name']==main_category]



beauty_df = train_maincat(train_df,'Beauty')

fashion_df = train_maincat(train_df,'Fashion')

mobile_df = train_maincat(train_df,'Mobile')

mobile_df = mobile_df.sample(10000)

print(len(beauty_df.index),len(fashion_df.index),len(mobile_df.index))
mobile_df_train, mobile_df_valid = train_test_split(mobile_df, test_size=0.1)

fashion_df_train, fashion_df_valid = train_test_split(fashion_df, test_size=0.1)

beauty_df_train, beauty_df_valid = train_test_split(beauty_df, test_size=0.1)

overall_df_train, overall_df_valid = train_test_split(train_df, test_size=0.1)
def return_xy(df,output_cat):

    return df['title'].values, df[output_cat].values



mobile_sentences_train, mobile_y_train = return_xy(mobile_df_train,'Category_Name')

mobile_sentences_test, mobile_y_test = return_xy(mobile_df_valid,'Category_Name')

beauty_sentences_train, beauty_y_train = return_xy(beauty_df_train,'Category_Name')

beauty_sentences_test, beauty_y_test = return_xy(beauty_df_valid,'Category_Name')

fashion_sentences_train, fashion_y_train = return_xy(fashion_df_train,'Category_Name')

fashion_sentences_test, fashion_y_test = return_xy(fashion_df_valid,'Category_Name')

overall_sentences_train, overall_y_train = return_xy(overall_df_train,'Category_Name')

overall_sentences_test, overall_y_test = return_xy(overall_df_valid,'Category_Name')
mobile_le = LabelEncoder()

fashion_le = LabelEncoder()

beauty_le = LabelEncoder()

overall_le = LabelEncoder()

mobile_le.fit(mobile_y_train)

fashion_le.fit(fashion_y_train)

beauty_le.fit(beauty_y_train)

overall_le.fit(overall_y_train)
def generate_fit_classifier(sentences_train, sentences_test, y_train, y_test, min_df = 1):

    vectorizer = CountVectorizer(min_df=min_df)

    vectorizer.fit(sentences_train)

    X_train = vectorizer.transform(sentences_train)

    X_test  = vectorizer.transform(sentences_test)

    classifier = LogisticRegression(solver = 'lbfgs',multi_class='multinomial',max_iter = 3000)

    classifier.fit(X_train, y_train)

    probs = classifier.predict_proba(X_test)

    print(classifier.score(X_test, y_test))

    return classifier, vectorizer, probs
mobile_classifier, mobile_vectorizer, mobile_predict_probs = generate_fit_classifier(mobile_sentences_train, mobile_sentences_test, mobile_y_train, mobile_y_test,min_df = 3)
#fashion_classifier, fashion_vectorizer, fashion_predict_probs = generate_fit_classifier(fashion_sentences_train, fashion_sentences_test, fashion_y_train, fashion_y_test,min_df = 3)
#beauty_classifer, beauty_vectorizer, beauty_predict_probs = generate_fit_classifier(beauty_sentences_train, beauty_sentences_test, beauty_y_train, beauty_y_test,min_df = 3)
mobile_predict_prob = np.max(mobile_predict_probs, axis=1)

mobile_predict_class = mobile_le.inverse_transform(np.argmax(mobile_predict_probs, axis=1))

np.sum(mobile_predict_class==mobile_y_test)/len(mobile_y_test)

mobile_df_valid['Log_Reg_Prediction'] = mobile_predict_class

mobile_df_valid['Prediction_Probability'] = mobile_predict_prob

mobile_df_valid['Correct'] = mobile_predict_class == mobile_y_test



#fashion_predict_prob = np.max(fashion_predict_probs, axis=1)

#fashion_predict_class = fashion_le.inverse_transform(np.argmax(fashion_predict_probs, axis=1))

#np.sum(fashion_predict_class==fashion_y_test)/len(fashion_y_test)

#fashion_df_valid['Log_Reg_Prediction'] = fashion_predict_class

#fashion_df_valid['Prediction_Probability'] = fashion_predict_prob

#fashion_df_valid['Correct'] = fashion_predict_class == fashion_y_test



#beauty_predict_prob = np.max(beauty_predict_probs, axis=1)

#beauty_predict_class = beauty_le.inverse_transform(np.argmax(beauty_predict_probs, axis=1))

#np.sum(beauty_predict_class==beauty_y_test)/len(beauty_y_test)

#beauty_df_valid['Log_Reg_Prediction'] = beauty_predict_class

#beauty_df_valid['Prediction_Probability'] = beauty_predict_prob

#beauty_df_valid['Correct'] = beauty_predict_class == beauty_y_test
mobile_df_wrong = mobile_df_valid[mobile_df_valid['Correct']==False]

#beauty_df_wrong = beauty_df_valid[beauty_df_valid['Correct']==False]

#fashion_df_wrong = fashion_df_valid[fashion_df_valid['Correct']==False]
# def plot_gallery(df, h = 12, w = 12, n_row=3, n_col=4):

#     """Helper function to plot a gallery of portraits"""

#     subset_df = df.sample(n_row * n_col)

#     plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))

#     plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)

#     for i in range(n_row * n_col):

#         datarow = mobile_df_wrong.iloc[i]

#         image = mpimg.imread(datarow['image_path'])

#         plt.subplot(n_row, n_col, i + 1)

#         plt.imshow(image)

#         title_string = '\n'+datarow['Category_Name']+'\n'+datarow['Log_Reg_Prediction']+'\n'+'{0:.2f}'.format(datarow['Prediction_Probability'])

#         plt.title(title_string, size=12)

#         plt.xticks(())

#         plt.yticks(())



def plot_sample(df):

    subset_df = df.sample(1)

    datarow = subset_df.iloc[0]

    image = mpimg.imread(datarow['image_path'])

    plt.figure(figsize=(20,10))

    plt.imshow(image)

    title_string = datarow['title']+'\nActual: '+datarow['Category_Name']+'\nPredicted: '+datarow['Log_Reg_Prediction']+'\nProb: '+'{0:.2f}'.format(datarow['Prediction_Probability'])

    plt.title(title_string)

    plt.axis('off')
plot_sample(mobile_df_wrong)
import seaborn as sn

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report
def conf_matrix(y_test,predict_class):



    class_names = list(set(y_test))

    cm = confusion_matrix(y_test, predict_class, labels=class_names)

    df_cm = pd.DataFrame(cm, index = [i for i in class_names],

                  columns = [i for i in class_names])

    plt.figure(figsize = (25,25))

    

    return print(classification_report(y_test, predict_class, target_names=class_names))

    return sn.heatmap(df_cm,annot=True,cmap='Blues', fmt='g',annot_kws={"size": 16})
conf_matrix(mobile_y_test, mobile_predict_class)

#conf_matrix(fashion_y_test, fashion_predict_class)

#conf_matrix(beauty_y_test, beauty_predict_class)
mobile_class_names = list(set(mobile_y_test))
mobile_class_names = list(set(mobile_y_test))



mobile_cm = confusion_matrix(mobile_y_test, mobile_predict_class, labels=mobile_class_names)

df_cm = pd.DataFrame(mobile_cm, index = [i for i in mobile_class_names],

                  columns = [i for i in mobile_class_names])

plt.figure(figsize = (25,25))

sn.heatmap(df_cm,annot=True,cmap='Blues', fmt='g')