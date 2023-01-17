import os

from time import time



def load_data(rel_data_path, project_data_location = '/kaggle/input/data/'):

    sorted_filenames = sorted(os.listdir(project_data_location+rel_data_path))

    

    data = []

    for filename in sorted_filenames:

        with open(project_data_location+rel_data_path+filename,'r',encoding = "latin-1") as review:

            data.append(review.read())

    

    sorted_review_ids = [fn[:fn.index('.')] for fn in sorted_filenames]

    

    return data,sorted_review_ids



def codify(label):

    return {'bad'    :0,

            'average':1,

            'good'   :2}[label]



def load_labels(filename,ids,project_data_location = '/kaggle/input/data/'):

    labels_dict = dict()

    with open(project_data_location+filename,'r') as labels_file:

         labels_file.readline() # skip header  

         for line in labels_file:

             review_id,rating = line.split(',')

             labels_dict[review_id.strip()] = codify(rating.strip())

    

    labels = [labels_dict[rid] for rid in ids]

    

    return labels



t0 = time()

data_train, ids_train = load_data('data/train/')

labels_train = load_labels('data/labels_train.csv', ids_train)

data_test, ids_test = load_data('data/test/')

print('Data loaded successfully in',round(time()-t0, 3), 's')
from sklearn.model_selection import train_test_split



data_train_split,data_validation,labels_train_split,labels_validation = train_test_split(data_train,labels_train, test_size = 0.1, random_state = 3)

print('Data splitted successfully into train and validation sets with sizes:',len(data_train_split),'&',len(data_validation))
from nltk.tokenize import RegexpTokenizer

from sklearn.feature_extraction.text import CountVectorizer



def preprocess_text(text):

    text_prep = ''



    #TODO: Use text processing techniques to improve the quality of the data for our problem.

    #      One example below.



    # compute stop words list:

    sw = {'to', 'the', 'a'}

    

    # tokenize text, removing punctuation but keeping dash (-) separated words and abbreviations:

    tokenizer  = RegexpTokenizer("(?:[A-Z]\.)+|\w+(?:[-]\w+)*")

    words_list = tokenizer.tokenize(text)



    # remove stop words:

    for word in words_list:

        if word.lower() not in sw:

            text_prep += word + ' '



    return text_prep



def preprocess(train_data,validation_data,test_data):

    #preprocess content of reviews

    train_data_prep      = [preprocess_text(text) for text in train_data     ]

    validation_data_prep = [preprocess_text(text) for text in validation_data]

    test_data_prep       = [preprocess_text(text) for text in test_data      ]

        

    # vectorize data

    vectorizer = CountVectorizer(max_df = 0.95)

    vectorizer.fit(train_data_prep)

    

    # transform data according to the vectorization

    train_data_vect      = vectorizer.transform(train_data_prep     )

    validation_data_vect = vectorizer.transform(validation_data_prep)

    test_data_vect       = vectorizer.transform(test_data_prep      )



    # returning the preprocessed data

    return train_data_vect, validation_data_vect, test_data_vect



t0 = time()

data_train_split_prep, data_validation_prep, data_test_prep = preprocess(data_train_split, data_validation, data_test)

print('Data preprocessed successfully in',round(time()-t0, 3), 's')
from sklearn.linear_model import LogisticRegression



t0 = time()

clf = LogisticRegression()

clf.fit(data_train_split_prep,labels_train_split)

print('Model fitted successfully in',round(time()-t0, 3), 's')
from sklearn.metrics import accuracy_score



t0 = time()

predictions_validation = clf.predict(data_validation_prep)

print('Predictions on the validation set done successfully in',round(time()-t0, 3), 's')



accuracy = accuracy_score(labels_validation, predictions_validation)         

print('Classifier\'s accuracy score on validation set:',accuracy*100,'%')
def decode(code):

    return {0:'bad'    ,

            1:'average',

            2:'good'   }[code]



t0 = time()

predictions_test = clf.predict(data_test_prep)

print('Predictions on the test set done successfully in',round(time()-t0, 3), 's')



submission_file_name = 'mtg_submission.csv'

with open(submission_file_name,'w') as submission_file:

    submission_file.write('ReviewID,Rating\n')

    for pred, review_id in zip(predictions_test,ids_test):

        submission_file.write(review_id+','+decode(pred)+'\n')

print('Submission file generated successfully')