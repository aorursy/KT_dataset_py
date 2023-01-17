import numpy as np 

import pandas as pd

import spacy

import en_core_web_lg



data= pd.read_csv('../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv', delimiter= ',')

N,d= np.shape(data)

print(N,d)



nlp= en_core_web_lg.load() #large starter model trained on web text
#create new dataframe, same as original but every Messages entry will have only lemmatized words and stop words will be removed

data_clean= pd.DataFrame(columns=['Category', 'Message'])



for i in range(N):

    doc= nlp(data.iloc[i,1]) #col 1 has the text to be preprocessed and classified

    

    clean_text= []

    for tok in doc:

        lex= nlp.vocab[tok.lemma] #consider lemmatized words only

        if lex.is_stop == False: #don't include stop words

            clean_text.append(tok.lemma_)

    

    add_row= pd.Series({'Category': data.iloc[i,0], 'Message': clean_text}) #for each entry in the new dataframe, keep the label the same

    data_clean= data_clean.append(add_row, ignore_index= True) 
#right now each message in data_clean is a list of strings. convert each message to be one string each

for i in range(N):

    data_clean.iloc[i, 1]= " ".join(data_clean.iloc[i, 1])



#convert each message to a word vector

#for each document, first get the word-level embeddings and then use the average of each word vector in the document as the document-level embedding

#(even though there are multiple words in each message, we want a single vector representing all of the words in each message. taking the average of the vectors representing each of the words allows us to do this)

message_vectors= np.array([nlp(j).vector for j in data_clean.iloc[:,1]])

N,d= np.shape(message_vectors)

print(N,d)
# for each iteration of the 10 folds, first standardize all feature values to have a mean of 0 and unit variance

# then train and evaluate the model's performance



from sklearn.model_selection import StratifiedKFold #for cross validation train/test splits

from sklearn.preprocessing import StandardScaler #for feature normalization

from sklearn.neural_network import MLPClassifier 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.svm import LinearSVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score, recall_score #model performance metrics





def eval_model(features, labels, model):

    get_folds= StratifiedKFold(n_splits= 10, random_state= 2, shuffle= True)

    scaler= StandardScaler()

    

    precisions= np.array([])

    recalls= np.array([])



    for train_idx, test_idx in get_folds.split(features, labels):

        # get training and test data

        x_train, x_test= features[train_idx], features[test_idx]

        y_train, y_test= labels[train_idx], labels[test_idx]

    

        # scale training and test features

        x_train_scaled= scaler.fit_transform(x_train)

        x_test_scaled= scaler.transform(x_test) # no data leakage here! using the training data to transform the test data

    

        #fit model using training data

        model.fit(x_train_scaled, y_train)

    

        #get predictions on the test set and return precision and recall for each of the 10 folds

        predicts= model.predict(x_test_scaled)

        precisions= np.append(precisions, precision_score(y_test, predicts, average= 'weighted'))

        recalls= np.append(recalls, recall_score(y_test, predicts, average= 'weighted'))

    

    # return precision and recall for each of the 10 folds in the cross-validation

    return precisions, recalls

    

labels= data_clean.iloc[:,0] # message_vectors is the matrix of corresponding features





#create 2-layer neural network. hyperparameter settings are arbitrary

neural_net= MLPClassifier(hidden_layer_sizes= (50,), activation= 'relu', solver= 'adam', alpha= 0.001, max_iter= 250, shuffle= True, random_state= 2)



#create decision tree. hyperparameter settings are arbitrary

d_tree= DecisionTreeClassifier(random_state= 2)



#create support vector machine. hyperparamter settings are arbitrary

svm= LinearSVC(dual= False, fit_intercept= False, random_state= 2, max_iter= 250)



#create kNN model. hyperparameter settings are arbitrary

knn= KNeighborsClassifier()





#for each of the models, get the precision and recall for each of the 10 folds

nn_precisions, nn_recalls= eval_model(message_vectors, labels, neural_net)

print('Neural net precisions and recalls:', '\n', nn_precisions, '\n', nn_recalls)



tree_precisions, tree_recalls= eval_model(message_vectors, labels, d_tree)

print('Decision tree precisions and recalls:', '\n', tree_precisions, '\n', tree_recalls)



svm_precisions, svm_recalls= eval_model(message_vectors, labels, svm)

print('SVM precisions and recalls:', '\n', svm_precisions, '\n', svm_recalls)



knn_precisions, knn_recalls= eval_model(message_vectors, labels, knn)

print('kNN precisions and recalls:', '\n', knn_precisions, '\n', knn_recalls)
import matplotlib.pyplot as plt



#visualize precisions

plt.figure()

plt.boxplot([nn_precisions, tree_precisions, svm_precisions, knn_precisions], notch= True, labels= ['Neural net', 'Decision tree', 'SVM', 'k-Nearest neighbors'])

plt.title('PRECISIONS for the 10 folds for each of the models')

plt.ylim([0.9,1.0])

plt.show()



#visualize recalls

plt.figure()

plt.boxplot([nn_recalls, tree_recalls, svm_recalls, knn_recalls], notch= True, labels= ['Neural net', 'Decision tree', 'SVM', 'k-Nearest neighbors'])

plt.title('RECALLS for the 10 folds for each of the models')

plt.ylim([0.75, 1.0])

plt.show()



#get average precision and recall for all models

print('Average precisions: ')

for i in [nn_precisions, tree_precisions, svm_precisions, knn_precisions]:

    print(np.mean(i))



print('\n', 'Average recalls: ')

for i in [nn_recalls, tree_recalls, svm_recalls, knn_recalls]:

    print(np.mean(i))