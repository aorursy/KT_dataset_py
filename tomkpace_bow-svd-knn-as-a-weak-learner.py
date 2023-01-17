import os 



import numpy as np



import pandas as pd



from sklearn.feature_extraction.text import CountVectorizer



from sklearn.decomposition import TruncatedSVD



from sklearn.neighbors import KNeighborsClassifier



from sklearn import metrics
raw_train = pd.read_csv('../input/train.csv')



raw_test = pd.read_csv('../input/test.csv')
vocab_size = 100000



max_length = 220



text_column = 'comment_text'



target_column = 'target'



dimensionality = 300



"""Validation"""



val_fraction = 0.2



n_components_list = [1, 2, 5]



k_list = [1, 2, 5]
def create_val_data(x_train, y_train, val_fraction):

    

    x_train, y_train = joint_shuffle(x_train, y_train)

    

    val_size = int(val_fraction*x_train.shape[0])

    

    partial_x_train = x_train[val_size:]

    

    partial_y_train = y_train[val_size:]

    

    val_x_train = x_train[:val_size]

    

    val_y_train = y_train[:val_size]

    

    return partial_x_train, partial_y_train, val_x_train, val_y_train





def bow_prepare_data(train_data, test_data, text_column, target_column, vocab_size):

    

    """Prepares train_data and test_data DataFrames into bag-of-words

    SciPy CSR sparse matrices."""

    

    raw_x_test = test_data[text_column].astype(str)

    

    raw_x_train = train_data[text_column].astype(str)

    

    y_train = train_data[target_column].values

    

    word_corpus = list(raw_x_test) + list(raw_x_train)

    

    vectorizer = CountVectorizer(strip_accents='ascii', max_features=vocab_size)



    vectorizer.fit_transform(word_corpus)

    

    bow_x_test = vectorizer.transform(raw_x_test)

    

    bow_x_train = vectorizer.transform(raw_x_train)

    

    return bow_x_train, y_train, bow_x_test, vectorizer





def SVD(bow_x_train, bow_x_test, n_components):

    

    svd = TruncatedSVD(n_components=n_components).fit(bow_x_train)

    

    svd_train_data = svd.transform(bow_x_train)

    

    svd_test_data = svd.transform(bow_x_test)

    

    return svd_train_data, svd_test_data





def sgn(y):

    

    new_y = np.zeros(shape=y.shape)

    

    for i in range(len(y)):

        

        if y[i] >= 0.5:

            

            new_y[i] = 1

            

    return new_y





def joint_shuffle(x_data, y_data):

    

    if x_data.shape[0] == y_data.shape[0]:

    

        p = np.random.permutation(x_data.shape[0])

    

    return x_data[p], y_data[p]





def svd_knn_val_test(x_train, y_train, x_test, y_test, n_components, k_list):

    

    class_y_train = sgn(y_train)

    

    class_y_test = sgn(y_test)

    

    svd_x_train, svd_x_test = SVD(x_train, x_test, n_components)

    

    training_auc = []

    

    val_auc = []

    

    for k in k_list:

        

        nbh = KNeighborsClassifier(n_neighbors=k)

    

        nbh.fit(svd_x_train, class_y_train)

    

        training_auc.append(metrics.roc_auc_score(class_y_train, nbh.predict(svd_x_train)))

    

        val_auc.append(metrics.roc_auc_score(class_y_test, nbh.predict(svd_x_test)))

    

    return training_auc, val_auc





def svd_knn_val(x_train, y_train, x_test, y_test, n_components_list, k_list):

    

    training_auc = np.zeros(shape=(len(n_components_list), len(k_list)))

    

    val_auc = np.zeros(shape=(len(n_components_list), len(k_list)))

    

    for i in range(len(n_components_list)):

        

        print("Performing validation on n_component =", n_components_list[i])

            

        training_auc[i], val_auc[i] = svd_knn_val_test(x_train, 

                                                       y_train,

                                                       x_test, 

                                                       y_test, 

                                                       n_components_list[i], 

                                                       k_list)

    

    return training_auc, val_auc





def val_assess(val_auc, hyper_p_list_1, hyper_p_list_2):

    

    val_index = np.unravel_index(np.argmax(val_auc, axis=None), val_auc.shape)

    

    best_hyper_p_1 = hyper_p_list_1[val_index[0]]

    

    best_hyper_p_2 = hyper_p_list_2[val_index[1]]

        

    return best_hyper_p_1, best_hyper_p_2





def predict_svd_knn(x_train, y_train, x_test, raw_test, best_n_component, best_k):

    

    class_y_train = sgn(y_train)

    

    svd_x_train, svd_x_test = SVD(x_train, x_test, best_n_component)

    

    nbh = KNeighborsClassifier(n_neighbors=best_k)

    

    nbh.fit(svd_x_train, class_y_train)

    

    predicted_y = nbh.predict(svd_x_test)

    

    submission = pd.DataFrame.from_dict({'id': raw_test.id, 'prediction': predicted_y})

    

    submission.to_csv('submission.csv', index=False)

    

    return submission
bow_x_train, y_train, bow_x_test, vectorizer = bow_prepare_data(raw_train, 

                                                                raw_test, 

                                                                text_column, 

                                                                target_column, 

                                                                vocab_size)
partial_x_train, partial_y_train, val_x_train, val_y_train = create_val_data(bow_x_train, 

                                                                             y_train, 

                                                                             val_fraction)
training_auc, val_auc = svd_knn_val(partial_x_train, 

                                    partial_y_train, 

                                    val_x_train, 

                                    val_y_train, 

                                    n_components_list, 

                                    k_list)



best_n_component, best_k = val_assess(val_auc, n_components_list, k_list)



submission = predict_svd_knn(bow_x_train, y_train, bow_x_test, raw_test, best_n_component, best_k)
