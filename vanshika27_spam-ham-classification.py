import re

import string

import numpy as np

import pandas as pd

from nltk import pos_tag

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import accuracy_score, confusion_matrix
email_data = pd.read_csv('../input/spam.csv', encoding = "ISO-8859-1")
email_data = email_data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis = 1)
email_data.head()
# Apostrophe Dictionary

apostrophe_dict = {

"ain't": "am not / are not",

"aren't": "are not / am not",

"can't": "cannot",

"can't've": "cannot have",

"'cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had / he would",

"he'd've": "he would have",

"he'll": "he shall / he will",

"he'll've": "he shall have / he will have",

"he's": "he has / he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has / how is",

"i'd": "I had / I would",

"i'd've": "I would have",

"i'll": "I shall / I will",

"i'll've": "I shall have / I will have",

"i'm": "I am",

"i've": "I have",

"isn't": "is not",

"it'd": "it had / it would",

"it'd've": "it would have",

"it'll": "it shall / it will",

"it'll've": "it shall have / it will have",

"it's": "it has / it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't": "might not",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had / she would",

"she'd've": "she would have",

"she'll": "she shall / she will",

"she'll've": "she shall have / she will have",

"she's": "she has / she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"so's": "so as / so is",

"that'd": "that would / that had",

"that'd've": "that would have",

"that's": "that has / that is",

"there'd": "there had / there would",

"there'd've": "there would have",

"there's": "there has / there is",

"they'd": "they had / they would",

"they'd've": "they would have",

"they'll": "they shall / they will",

"they'll've": "they shall have / they will have",

"they're": "they are",

"they've": "they have",

"to've": "to have",

"wasn't": "was not",

"we'd": "we had / we would",

"we'd've": "we would have",

"we'll": "we will",

"we'll've": "we will have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what shall / what will",

"what'll've": "what shall have / what will have",

"what're": "what are",

"what's": "what has / what is",

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where has / where is",

"where've": "where have",

"who'll": "who shall / who will",

"who'll've": "who shall have / who will have",

"who's": "who has / who is",

"who've": "who have",

"why's": "why has / why is",

"why've": "why have",

"will've": "will have",

"won't": "will not",

"won't've": "will not have",

"would've": "would have",

"wouldn't": "would not",

"wouldn't've": "would not have",

"y'all": "you all",

"y'all'd": "you all would",

"y'all'd've": "you all would have",

"y'all're": "you all are",

"y'all've": "you all have",

"you'd": "you had / you would",

"you'd've": "you would have",

"you'll": "you shall / you will",

"you'll've": "you shall have / you will have",

"you're": "you are",

"you've": "you have"

}
short_word_dict = {

"121": "one to one",

"a/s/l": "age, sex, location",

"adn": "any day now",

"afaik": "as far as I know",

"afk": "away from keyboard",

"aight": "alright",

"alol": "actually laughing out loud",

"b4": "before",

"b4n": "bye for now",

"bak": "back at the keyboard",

"bf": "boyfriend",

"bff": "best friends forever",

"bfn": "bye for now",

"bg": "big grin",

"bta": "but then again",

"btw": "by the way",

"cid": "crying in disgrace",

"cnp": "continued in my next post",

"cp": "chat post",

"cu": "see you",

"cul": "see you later",

"cul8r": "see you later",

"cya": "bye",

"cyo": "see you online",

"dbau": "doing business as usual",

"fud": "fear, uncertainty, and doubt",

"fwiw": "for what it's worth",

"fyi": "for your information",

"g": "grin",

"g2g": "got to go",

"ga": "go ahead",

"gal": "get a life",

"gf": "girlfriend",

"gfn": "gone for now",

"gmbo": "giggling my butt off",

"gmta": "great minds think alike",

"h8": "hate",

"hagn": "have a good night",

"hdop": "help delete online predators",

"hhis": "hanging head in shame",

"iac": "in any case",

"ianal": "I am not a lawyer",

"ic": "I see",

"idk": "I don't know",

"imao": "in my arrogant opinion",

"imnsho": "in my not so humble opinion",

"imo": "in my opinion",

"iow": "in other words",

"ipn": "I’m posting naked",

"irl": "in real life",

"jk": "just kidding",

"l8r": "later",

"ld": "later, dude",

"ldr": "long distance relationship",

"llta": "lots and lots of thunderous applause",

"lmao": "laugh my ass off",

"lmirl": "let's meet in real life",

"lol": "laugh out loud",

"ltr": "longterm relationship",

"lulab": "love you like a brother",

"lulas": "love you like a sister",

"luv": "love",

"m/f": "male or female",

"m8": "mate",

"milf": "mother I would like to fuck",

"oll": "online love",

"omg": "oh my god",

"otoh": "on the other hand",

"pir": "parent in room",

"ppl": "people",

"r": "are",

"rofl": "roll on the floor laughing",

"rpg": "role playing games",

"ru": "are you",

"shid": "slaps head in disgust",

"somy": "sick of me yet",

"sot": "short of time",

"thanx": "thanks",

"thx": "thanks",

"ttyl": "talk to you later",

"u": "you",

"ur": "you are",

"uw": "you’re welcome",

"wb": "welcome back",

"wfm": "works for me",

"wibni": "wouldn't it be nice if",

"wtf": "what the fuck",

"wtg": "way to go",

"wtgp": "want to go private",

"ym": "young man",

"gr8": "great"

}
remove_words = []

remove_words += stopwords.words('english')

remove_words += list(string.punctuation)
# Utlity to simplify the lemmatization

def get_simple_pos_tag(tag):

    if tag.startswith('J'):

        return wordnet.ADJ

    elif tag.startswith('V'):

        return wordnet.VERB

    elif tag.startswith('N'):

        return wordnet.NOUN

    elif tag.startswith('R'):

        return wordnet.ADJ

    else:

        return wordnet.NOUN



# Utility to remove stopwords

def remove_patterns(pattern, data):

    pattern_words = re.findall(pattern, data) 

    for word in pattern_words:

            data = re.sub(word, '', data)

    return data



# Utlity to perform lemmatization

def word_lemma_(data):

    lemmatizer = WordNetLemmatizer()

    text = ''

    for word in data:

        pos = pos_tag([word])

        clean_word = lemmatizer.lemmatize(word, pos = get_simple_pos_tag(pos[0][1]))

        text += clean_word+' '

    return text
# Remove all email ID's from data

email_data['v2'] = np.vectorize(remove_patterns)('\w+[@]\w+[.]\w+', email_data['v2'])
# Manipulate Data

def manipulate_text_(dictionary, text):

    for word in text.split(' '):

        if word.lower() in dictionary:

            text = text.replace(word, dictionary[word.lower()])

    return text
# Remove all apostrophes from data

email_data['clean_apostrophe'] = email_data['v2'].apply(lambda x: manipulate_text_(apostrophe_dict, x))
# Update all short words in data

email_data['clean_short_words'] = email_data['clean_apostrophe'].apply(lambda x: manipulate_text_(short_word_dict, x))
# Update all emoticons in data

#email_data['clean_emoticons'] = email_data['clean_short_words'].apply(lambda x: manipulate_text_(emoticon_dict, x))
# Remove all extra characters in data

email_data['clean_emails'] = email_data['clean_short_words'].apply(lambda x: re.sub(r'[^a-zA-Z0-9]',' ',x))
# Remove all stopwords and punctuations from data

email_data['clean_emails'] = email_data['clean_emails'].apply(lambda x: [word for word in x.split(' ') if len(word) > 1 if word not in remove_words])
email_data.head()
# Lemmatize the Data

email_data['lemmatized_emails'] = email_data['clean_emails'].apply(lambda x: word_lemma_(x))
tfidf_vectorizer = TfidfVectorizer(max_df=0.9, min_df=0.1, max_features=1500)
tfidf_vectorizer
apostrphe_transformed = tfidf_vectorizer.fit_transform(email_data['clean_apostrophe'])

a_x_train, a_x_test, a_y_train, a_y_test = train_test_split(apostrphe_transformed, email_data['v1'])



apostrphe_clf = LogisticRegression()

apostrphe_clf.fit(a_x_train, a_y_train)

a_y_pred = apostrphe_clf.predict(a_x_test)



a_accuracy = accuracy_score(a_y_test, a_y_pred)

print('Accuracy is: ',a_accuracy)



a_cm_ = confusion_matrix(a_y_test, a_y_pred)

print('Confusion Matrix: ',a_cm_)
email_text_transformed = tfidf_vectorizer.fit_transform(email_data['lemmatized_emails'])

x_train, x_test, y_train, y_test = train_test_split(email_text_transformed, email_data['v1'])



clf = LogisticRegression()

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)



accuracy = accuracy_score(y_test, y_pred)

print('Accuracy is: ',accuracy)



cm_ = confusion_matrix(y_test, y_pred)

print('Confusion Matrix: ',cm_)