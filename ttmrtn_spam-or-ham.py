import pandas as pd
train = pd.read_csv("../input/train_data.csv")

test = pd.read_csv("../input/test_features.csv")
train
train.describe()
train.info()
train.drop('Id', axis = 1, inplace = True)

#test.drop('Id', axis = 1, inplace = True)
train.corr()['ham'].sort_values(ascending=True).drop('ham')
(((train.corr()['ham']).sort_values(ascending=True)).drop('ham')).plot(kind = 'bar')
train['ham'].describe()
spam = train.copy()[train['ham'] == False]  # Avoids annoying warning
spam
spam.describe()
ham = train.copy()[train['ham'] == True]  # Avoids annoying warning
ham
ham.describe()
#(spam.drop(labels=['ham'], axis=1)).mean().nlargest(58)
#(ham.drop(labels=['ham'], axis=1).mean()).nlargest(58)
spam_stats = pd.DataFrame((spam.drop(labels=['ham'], axis=1)).mean(), columns = ['mean'])

spam_stats['std'] = (spam.drop(labels=['ham'], axis=1)).std()

spam_stats['max'] = (spam.drop(labels=['ham'], axis=1)).max()

spam_stats.sort_values(by = 'mean', ascending = False)
ham_stats = pd.DataFrame((ham.drop(labels=['ham'], axis=1)).mean(), columns = ['mean'])

ham_stats['std'] = (ham.drop(labels=['ham'], axis=1)).std()

ham_stats['max'] = (ham.drop(labels=['ham'], axis=1)).max()

ham_stats.sort_values(by = 'mean', ascending = False)
#[['word_freq_your', 'word_freq_000', 'char_freq_$', 'word_freq_remove', 'word_freq_you', 'word_freq_free', 'word_freq_business', 'capital_run_length_total', 'word_freq_order', 'word_freq_receive', 'word_freq_our', 'char_freq_!', 'word_freq_over', 'word_freq_credit', 'word_freq_money', 'capital_run_length_longest', 'word_freq_internet', 'word_freq_hpl', 'word_freq_hp']]



#[['word_freq_all', 'word_freq_addresses']]



# Selected features (abs(corr) > 0.19)
Xtreino1 = (train.drop('ham', axis = 1))[['word_freq_your', 'word_freq_000', 'char_freq_$', 'word_freq_remove', 'word_freq_you', 'word_freq_free', 'word_freq_business', 'capital_run_length_total', 'word_freq_order', 'word_freq_receive', 'word_freq_our', 'char_freq_!', 'word_freq_over', 'word_freq_credit', 'word_freq_money', 'capital_run_length_longest', 'word_freq_internet', 'word_freq_hpl', 'word_freq_hp', 'word_freq_all', 'word_freq_addresses']]
Ytreino1 = train['ham']
Xtest1 = test.drop('Id', axis = 1)[['word_freq_your', 'word_freq_000', 'char_freq_$', 'word_freq_remove', 'word_freq_you', 'word_freq_free', 'word_freq_business', 'capital_run_length_total', 'word_freq_order', 'word_freq_receive', 'word_freq_our', 'char_freq_!', 'word_freq_over', 'word_freq_credit', 'word_freq_money', 'capital_run_length_longest', 'word_freq_internet', 'word_freq_hpl', 'word_freq_hp', 'word_freq_all', 'word_freq_addresses']]
%%time 

import sklearn

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score



gnb = GaussianNB()

gnb_scores = cross_val_score(gnb, Xtreino1, Ytreino1, cv = 10, n_jobs=-1)

print('Gaussian Naive Bayes CV accuracy: {0:1.4f} +-{1:2.5f}\n'.format(gnb_scores.mean(), gnb_scores.std()))
import scikitplot as skplt

skplt.metrics.plot_confusion_matrix(rtr, y_pred_gnb,cmap="cool",figsize=(6,6))
gnb.fit(Xtreino1, Ytreino1)

y_predict_gnb = gnb.predict(Xtest1)
df_pred_gnb = pd.DataFrame({'Id': test['Id'], 'ham':y_predict_gnb})

df_pred_gnb.to_csv("gnb_prediction.csv", index = False)
df_pred_gnb.head()
#high_corr = train.corr()['ham'].sort_values(ascending=True).drop('ham').apply(lambda x: abs(x)>0.19)

#high_corr