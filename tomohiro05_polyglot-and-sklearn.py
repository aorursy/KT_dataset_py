!pip install --upgrade pip

!apt-get install -y libicu-dev

!pip install polyglot

!pip install pyicu

!pip install pycld2

!pip install morfessor
!polyglot download embeddings2.en pos2.en

!polyglot download ner2.en

!polyglot download sentiment2.en
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib as plt

from polyglot.text import Text, Word

from polyglot.detect import Detector

from polyglot.downloader import downloader



from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.decomposition import PCA



# Any results you write to the current directory are saved as out
word_class = ['VERB', 'DET', 'NOUN', 'PRON', 'PUNCT', 'ADP', 'ADV', 'ADJ']

word_class
class_index_col = []

ordinal_num_max = 3



for class_name in word_class:

    i = 0

    for i in range(ordinal_num_max):

        class_index_col.append(class_name + "_" + "INX" + str(i))



class_index_col
class_count_col = []



for class_name in word_class:

    class_count_col.append(class_name + "_" + "CNT")



class_count_col
class_percent_col = []



for class_name in word_class:

    class_percent_col.append(class_name + "_" + "PER")



class_percent_col
posi_index_col = []



i = 0

for i in range(ordinal_num_max):

    posi_index_col.append("POSI_INX" +  str(i))



posi_index_col
nega_index_col = []



i = 0

for i in range(ordinal_num_max):

    nega_index_col.append("NEGA_INX" +  str(i))



nega_index_col
def train_data_read(columns_data, skiprows, nrows):

    data = pd.read_csv("../input/train.csv",

                   names = columns_data,

                   skiprows = skiprows,

                   nrows = nrows

                  )

    return data



def index_cal(index_list, th_number):

    if ((th_number) < (len(index_list))):

        return index_list[th_number]

    else:

        return 101



def flat_reshape(input_list):

    i = 0

    work_area = input_list

    output_list = []

    for i in range(len(input_list)):

        output_list.append(work_area[i].flatten())

    

    return output_list
data = pd.read_csv("../input/train.csv",

               nrows = 10000

              )    



y_target = np.array(data["target"],dtype = "float16")



text = data.comment_text

text = text.values



i = 0

j = 0

words = 0

tags = 0



class_index_all = []

posi_index_all = []

nega_index_all = []



class_count_all = []

posi_count_all = []

nega_count_all = []



text_length = []



for i in range(len(text)):

    try:

        twords    = Text(text[i])  #comment_data

        words     = twords.words   #comment divided into words

        tag_words =  (np.array(twords.pos_tags).T[0])  #tags of tagged words

        tags      =  (np.array(twords.pos_tags).T[1])  #words of tagged words



        posi_index_0 = []

        nega_index_0 = []

        posi_index_1 = []

        nega_index_1 = []

        class_order_0 = []

        class_count_0 = []

        posi_count_0 = 0

        nega_count_0 = 0

        check_count_0 = 0





        for i,w in enumerate(words):

            if w.polarity == 1:

                posi_count_0 += 1

                posi_index_0.append(i)

            if w.polarity == -1:

                nega_count_0 += 1

                nega_index_0.append(i)



        j = 0

        for j in range(ordinal_num_max):

            posi_work = index_cal(posi_index_0,j)

            nega_work = index_cal(nega_index_0,j)



            posi_index_1.append(posi_work)

            nega_index_1.append(nega_work)



        k = 0

        for w in word_class:

            word_work = np.where(tags == w)[0]

            class_order_1 = []

            class_count_0.append(len(word_work))



            for k in range(ordinal_num_max):

                word_index = index_cal(word_work,k)

                class_order_1.append(word_index)

            class_order_0.append(class_order_1)



        posi_count_all.append(posi_count_0)

        nega_count_all.append(nega_count_0)

        posi_index_all.append(posi_index_1)

        nega_index_all.append(nega_index_1)



        class_count_all.append(np.array(class_count_0))

        class_index_all.append(np.array(class_order_0))

        text_length.append(len(words))



    except: 

        posi_index_0 = []

        nega_index_0 = []

        posi_index_1 = []

        nega_index_1 = []

        class_order_0 = []

        class_count_0 = []

        posi_count_0 = 0

        nega_count_0 = 0

        check_count_0 = 0





        for j in range(ordinal_num_max):

            posi_work = index_cal(posi_index_0,j)

            nega_work = index_cal(nega_index_0,j)



            posi_index_1.append(posi_work)

            nega_index_1.append(nega_work)



        for w in word_class:

            class_order_1 = []

            class_count_0.append(0)

            for k in range(ordinal_num_max):

                word_index = 100

                class_order_1.append(word_index)

            class_order_0.append(class_order_1)



        posi_count_all.append(0)

        nega_count_all.append(0)

        posi_index_all.append(posi_index_1)

        nega_index_all.append(nega_index_1)



        class_count_all.append(np.array(class_count_0))

        class_index_all.append(np.array(class_order_0))

        text_length.append(0) 







l = 0

class_index_all_work = class_index_all

class_index_all = []

for i in range(len(class_index_all_work)):

    class_index_all.append(class_index_all_work[i].flatten())



flat_reshape(class_index_all)



text_length_wide = []

l = 0

for i in range(len(np.array(class_count_all).T)):

    text_length_wide.append (text_length)

text_length_wide



class_per_all = []

l = 0

for i in range(len(class_count_all)):

    class_per_all.append(np.array(class_count_all[i], dtype = "int16") * 1000 / (np.array(text_length_wide,  dtype = "int16").T))



class_per_all = class_per_all[0]





posi_per_all = np.array(posi_count_all, dtype = "int16") * 1000 / (np.array(text_length,  dtype = "int16"))

nega_per_all = np.array(nega_count_all, dtype = "int16") * 1000 / (np.array(text_length,  dtype = "int16"))





target_data = pd.DataFrame(y_target,columns = ["target"])

analyze = pd.DataFrame(class_index_all,columns = class_index_col)



for i,w in enumerate(posi_index_col):

    analyze[w] = np.array(posi_index_all).T[i]



for i,w in enumerate(nega_index_col):

    analyze[w] = np.array(nega_index_all).T[i]



for i,w in enumerate(class_count_col):

    analyze[w] = np.array(class_count_all).T[i]



analyze["posi_count"] = posi_count_all

analyze["nega_count"] = nega_count_all  



#for i,w in enumerate(class_percent_col):

#    analyze[w] = np.array(class_per_all).T[i]



#analyze["posi_percent"] = posi_per_all

#analyze["nega_percent"] = nega_per_all

        

    
analyze
target_data
np_analyze = analyze.values

np_analyze = np_analyze.astype(np.int16)

np_analyze
np_target = target_data.values

np_target
poly_features = PolynomialFeatures(degree=2, include_bias = False)

X_poly = poly_features.fit_transform(np_analyze)

X_poly
lin_reg = LinearRegression()

lin_reg.fit(X_poly,y_target)

lin_reg.intercept_, lin_reg.coef_
train_predict = lin_reg.predict(X_poly)

train_predict
test = pd.read_csv("../input/test.csv",

               nrows = 100

              )    



text = test.comment_text

text = text.values



i = 0

j = 0

words = 0

tags = 0



class_index_all = []

posi_index_all = []

nega_index_all = []



class_count_all = []

posi_count_all = []

nega_count_all = []



text_length = []



for i in range(len(text)):

    try:

        twords    = Text(text[i])  #comment_data

        words     = twords.words   #comment divided into words

        tag_words =  (np.array(twords.pos_tags).T[0])  #tags of tagged words

        tags      =  (np.array(twords.pos_tags).T[1])  #words of tagged words



        posi_index_0 = []

        nega_index_0 = []

        posi_index_1 = []

        nega_index_1 = []

        class_order_0 = []

        class_count_0 = []

        posi_count_0 = 0

        nega_count_0 = 0

        check_count_0 = 0





        for i,w in enumerate(words):

            if w.polarity == 1:

                posi_count_0 += 1

                posi_index_0.append(i)

            if w.polarity == -1:

                nega_count_0 += 1

                nega_index_0.append(i)



        j = 0

        for j in range(ordinal_num_max):

            posi_work = index_cal(posi_index_0,j)

            nega_work = index_cal(nega_index_0,j)



            posi_index_1.append(posi_work)

            nega_index_1.append(nega_work)



        k = 0

        for w in word_class:

            word_work = np.where(tags == w)[0]

            class_order_1 = []

            class_count_0.append(len(word_work))



            for k in range(ordinal_num_max):

                word_index = index_cal(word_work,k)

                class_order_1.append(word_index)

            class_order_0.append(class_order_1)



        posi_count_all.append(posi_count_0)

        nega_count_all.append(nega_count_0)

        posi_index_all.append(posi_index_1)

        nega_index_all.append(nega_index_1)



        class_count_all.append(np.array(class_count_0))

        class_index_all.append(np.array(class_order_0))

        text_length.append(len(words))



    except: 

        posi_index_0 = []

        nega_index_0 = []

        posi_index_1 = []

        nega_index_1 = []

        class_order_0 = []

        class_count_0 = []

        posi_count_0 = 0

        nega_count_0 = 0

        check_count_0 = 0





        for j in range(ordinal_num_max):

            posi_work = index_cal(posi_index_0,j)

            nega_work = index_cal(nega_index_0,j)



            posi_index_1.append(posi_work)

            nega_index_1.append(nega_work)



        for w in word_class:

            class_order_1 = []

            class_count_0.append(0)

            for k in range(ordinal_num_max):

                word_index = 100

                class_order_1.append(word_index)

            class_order_0.append(class_order_1)



        posi_count_all.append(0)

        nega_count_all.append(0)

        posi_index_all.append(posi_index_1)

        nega_index_all.append(nega_index_1)



        class_count_all.append(np.array(class_count_0))

        class_index_all.append(np.array(class_order_0))

        text_length.append(0) 







l = 0

class_index_all_work = class_index_all

class_index_all = []

for i in range(len(class_index_all_work)):

    class_index_all.append(class_index_all_work[i].flatten())



flat_reshape(class_index_all)



text_length_wide = []

l = 0

for i in range(len(np.array(class_count_all).T)):

    text_length_wide.append (text_length)

text_length_wide



class_per_all = []

l = 0

for i in range(len(class_count_all)):

    class_per_all.append(np.array(class_count_all[i], dtype = "int16") * 1000 / (np.array(text_length_wide,  dtype = "int16").T))



class_per_all = class_per_all[0]





posi_per_all = np.array(posi_count_all, dtype = "int16") * 1000 / (np.array(text_length,  dtype = "int16"))

nega_per_all = np.array(nega_count_all, dtype = "int16") * 1000 / (np.array(text_length,  dtype = "int16"))





target_data = pd.DataFrame(y_target,columns = ["target"])

test_analyze = pd.DataFrame(class_index_all,columns = class_index_col)



for i,w in enumerate(posi_index_col):

    test_analyze[w] = np.array(posi_index_all).T[i]



for i,w in enumerate(nega_index_col):

    test_analyze[w] = np.array(nega_index_all).T[i]



for i,w in enumerate(class_count_col):

    test_analyze[w] = np.array(class_count_all).T[i]



test_analyze["posi_count"] = posi_count_all

test_analyze["nega_count"] = nega_count_all  



#for i,w in enumerate(class_percent_col):

#    test_analyze[w] = np.array(class_per_all).T[i]



#test_analyze["posi_percent"] = posi_per_all

#test_analyze["nega_percent"] = nega_per_all
test_analyze
np_test_analyze = test_analyze.values

np_test_analyze
poly_features = PolynomialFeatures(degree=2, include_bias = False)

X_poly = poly_features.fit_transform(np_test_analyze)

X_poly
test_predict = lin_reg.predict(X_poly)

test_predict