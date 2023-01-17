import pandas as pd

import string

import numpy as np
dataset = pd.read_csv('../input/spamraw.csv')
def text_preprocess(text):

    for punctuation in string.punctuation:

        text = text.replace(punctuation, '')

        text = str.lower(text)

    return text
dataset.text = dataset.text.apply(text_preprocess)
words = ['win','prize','award','free']
dataset_count = dataset.text.count()

train_set = dataset.head(int(dataset_count * 0.8))

test_set = dataset.tail(dataset_count - int(dataset_count * 0.8))

spam_set = train_set[train_set['type'] == 'spam']

nonspam_set = train_set[train_set['type'] == 'ham']
spam_count = spam_set.text.count()

nonspam_count = nonspam_set.text.count()

print('Spam count: ' + str(spam_count))

print('Nonspam count: ' + str(nonspam_count))
def check_word(word,text):

    if word in text:

        return True

    return False
for i in words:

    spam_set['has' + i] = np.vectorize(check_word)(i, spam_set.text)    

    nonspam_set['has' + i] = np.vectorize(check_word)(i, nonspam_set.text)    
spam_set.head()
nonspam_set.head()
p_win_spam = spam_set[spam_set['haswin'] == True].haswin.count() / float(spam_count)

p_win_nonspam = nonspam_set[nonspam_set['haswin'] == True].haswin.count() / float(nonspam_count)
p_prize_spam = spam_set[spam_set['hasprize'] == True].hasprize.count() / float(spam_count)

p_prize_nonspam = nonspam_set[nonspam_set['hasprize'] == True].hasprize.count() / float(nonspam_count)
p_award_spam = spam_set[spam_set['hasaward'] == True].hasaward.count() / float(spam_count)

p_award_nonspam = nonspam_set[nonspam_set['hasaward'] == True].hasaward.count() / float(nonspam_count)
p_free_spam = spam_set[spam_set['hasfree'] == True].hasfree.count() / float(spam_count)

p_free_nonspam = nonspam_set[nonspam_set['hasfree'] == True].hasfree.count() / float(nonspam_count)
p_spam = spam_count / float(spam_count + nonspam_count)

p_nonspam = nonspam_count / float(spam_count + nonspam_count)
p_win = spam_set[spam_set['haswin'] == True].haswin.count() / float(spam_count + nonspam_count)

p_prize = spam_set[spam_set['hasprize'] == True].hasprize.count() / float(spam_count + nonspam_count)

p_award = spam_set[spam_set['hasaward'] == True].hasaward.count() / float(spam_count + nonspam_count)

p_free = spam_set[spam_set['hasfree'] == True].hasfree.count() / float(spam_count + nonspam_count)
p_spam_win = (p_win_spam * p_spam)/float(p_win)

p_nonspam_win = (p_win_nonspam * p_nonspam)/float(p_win)
p_spam_prize = (p_prize_spam * p_spam)/float(p_win)

p_nonspam_prize = (p_prize_nonspam * p_nonspam)/float(p_win)
p_spam_award = (p_award_spam * p_spam)/float(p_win)

p_nonspam_award = (p_award_nonspam * p_nonspam)/float(p_win)
p_spam_free = (p_free_spam * p_spam)/float(p_win)

p_nonspam_free = (p_free_nonspam * p_nonspam)/float(p_win)
test_set.head()
def predict(text):

    if 'win' in text:

        if p_spam_win > p_nonspam_win:

            return 'spam'

        else:

            return 'ham'

    elif 'prize' in text:

        if p_spam_prize > p_nonspam_prize:

            return 'spam'

        else:

            return 'ham'

    elif 'award' in text:

        if p_spam_award > p_nonspam_award:

            return 'spam'

        else:

            return 'ham'

    elif 'free' in text:

        if p_spam_free > p_nonspam_free:

            return 'spam'

        else:

            return 'ham'

    else:

        return 'ham'
test_set['predict'] = test_set.text.apply(predict)
def result_prediction(ideal,predict):

    if ideal == predict:

        return True

    else:

        return False
test_set.head(15)
test_set['result'] = np.vectorize(result_prediction)(test_set.type, test_set.predict)    
test_set.head()
true_pre = test_set[test_set['result'] == True].result.count()

false_pre = test_set[test_set['result'] == False].result.count()
accuracy_score = float(true_pre) / (true_pre + false_pre)

print('Accuracy Score: ' + str(accuracy_score))