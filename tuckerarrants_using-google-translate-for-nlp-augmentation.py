GEN_BACK_TR = True



GEN_UPSAMPLE = False



GEN_EN_ONLY = False
#python basics

from matplotlib import pyplot as plt

import math, os, re, time

import numpy as np, pandas as pd, seaborn as sns



#nlp augmentation

!pip install --quiet googletrans

from googletrans import Translator



#model evaluation

from sklearn.model_selection import train_test_split, StratifiedKFold



#for fast parallel processing

from dask import bag, diagnostics
def back_translate(sequence, PROB = 1):

    languages = ['en', 'fr', 'th', 'tr', 'ur', 'ru', 'bg', 'de', 'ar', 'zh-cn', 'hi',

                 'sw', 'vi', 'es', 'el']

    

    #instantiate translator

    translator = Translator()

    

    #store original language so we can convert back

    org_lang = translator.detect(sequence).lang

    

    #randomly choose language to translate sequence to  

    random_lang = np.random.choice([lang for lang in languages if lang is not org_lang])

    

    if org_lang in languages:

        #translate to new language and back to original

        translated = translator.translate(sequence, dest = random_lang).text

        #translate back to original language

        translated_back = translator.translate(translated, dest = org_lang).text

    

        #apply with certain probability

        if np.random.uniform(0, 1) <= PROB:

            output_sequence = translated_back

        else:

            output_sequence = sequence

            

    #if detected language not in our list of languages, do nothing

    else:

        output_sequence = sequence

    

    return output_sequence



#check performance

for i in range(5):

    output = back_translate('I genuinely have no idea what the output of this sequence of words will be')

    print(output)
#applies above define function with Dask

def back_translate_parallel(dataset):

    prem_bag = bag.from_sequence(dataset['premise'].tolist()).map(back_translate)

    hyp_bag =  bag.from_sequence(dataset['hypothesis'].tolist()).map(back_translate)

    

    with diagnostics.ProgressBar():

        prems = prem_bag.compute()

        hyps = hyp_bag.compute()



    #pair premises and hypothesis

    dataset[['premise', 'hypothesis']] = list(zip(prems, hyps))

    

    return dataset
twice_train_aug = pd.read_csv('../input/contradictorywatsontwicetranslatedaug/twice_translated_aug_train.csv')

twice_test_aug = pd.read_csv('../input/contradictorywatsontwicetranslatedaug/twice_translated_aug_test.csv')
if GEN_BACK_TR:

#now we apply translation augmentation

    train_thrice_aug = twice_train_aug.pipe(back_translate_parallel)

    test_thrice_aug = twice_test_aug.pipe(back_translate_parallel)

    

    train_thrice_aug.to_csv('thrice_translation_aug_train.csv')

    test_thrice_aug.to_csv('thrice_translation_aug_test.csv')
#offline loading

train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv")

test = pd.read_csv("../input/contradictory-my-dear-watson/test.csv")



train_aug = pd.read_csv("../input/contradictorywatsontwicetranslatedaug/translation_aug_train.csv")

test_aug = pd.read_csv("../input/contradictorywatsontwicetranslatedaug/translation_aug_test.csv")



train_twice_aug = pd.read_csv("../input/contradictorywatsontwicetranslatedaug/twice_translated_aug_train.csv")

test_twice_aug = pd.read_csv("../input/contradictorywatsontwicetranslatedaug/twice_translated_aug_test.csv")



#view original

print(train.shape)

train.head()
#view first aug

print(train_aug.shape)

train_aug.head()
#view second aug

print(train_twice_aug.shape)

train_twice_aug.head()
#view third aug

print(train_thrice_aug.shape)

train_thrice_aug.head()
#check most undersampled languages in training dataset

train['language'].value_counts()
#check most undersampled languages in test dataset

test['language'].value_counts()
def translation(sequence, lang):

    

    #instantiate translator

    translator = Translator()

    

    org_lang = translator.detect(sequence).lang

    

    if lang is not org_lang:

        #translate to new language and back to original

        translated = translator.translate(sequence, dest = lang).text

        

    else:

        translated = sequence

    

    return translated



def translation_parallel(dataset, lang):

    prem_bag = bag.from_sequence(dataset['premise'].tolist()).map(lambda x: translation(x, lang = lang))

    hyp_bag =  bag.from_sequence(dataset['hypothesis'].tolist()).map(lambda x: translation(x, lang = lang))

    

    with diagnostics.ProgressBar():

        prems = prem_bag.compute()

        hyps = hyp_bag.compute()



    #pair premises and hypothesis

    dataset[['premise', 'hypothesis']] = list(zip(prems, hyps))

    

    return dataset
#translate to Vietnamese

prem_bag_vi = bag.from_sequence(train['premise'].tolist()).map(lambda x: translation(x, lang = 'vi'))

hyp_bag_vi =  bag.from_sequence(train['hypothesis'].tolist()).map(lambda x: translation(x, lang = 'vi'))



#translate to Hindi

prem_bag_hi = bag.from_sequence(train['premise'].tolist()).map(lambda x: translation(x, lang = 'hi'))

hyp_bag_hi =  bag.from_sequence(train['hypothesis'].tolist()).map(lambda x: translation(x, lang = 'hi'))



#translate to Bulgarian

prem_bag_bg = bag.from_sequence(train['premise'].tolist()).map(lambda x: translation(x, lang = 'bg'))

hyp_bag_bg =  bag.from_sequence(train['hypothesis'].tolist()).map(lambda x: translation(x, lang = 'bg'))



#and compute

if GEN_UPSAMPLE:

    with diagnostics.ProgressBar():

        print('Translating train to Vietnamese...')

        prems_vi = prem_bag_vi.compute()

        hyps_vi = hyp_bag_vi.compute()

        print('Done'); print('')

    

        print('Translating train to Hindi...')

        prems_hi = prem_bag_hi.compute()

        hyps_hi = hyp_bag_hi.compute()

        print('Done'); print('')

    

        print('Translating train to Bulgarian...')

        prems_bg = prem_bag_bg.compute()

        hyps_bg = hyp_bag_bg.compute()

        print('Done')

        

else:

    train_vi = pd.read_csv("../input/contradictorytranslatedtrain/train_vi.csv")

    train_hi = pd.read_csv("../input/contradictorytranslatedtrain/train_hi.csv")

    train_bg = pd.read_csv("../input/contradictorytranslatedtrain/train_bg.csv")
if GEN_UPSAMPLE:

    #sanity check

    train_vi = train

    train_vi[['premise', 'hypothesis']] = list(zip(prems_vi, hyps_vi))

    train_vi[['lang_abv', 'language']] = ['vi', 'Vietnamese']

    train_vi.to_csv('train_vi.csv', index = False)

train_vi.head()
if GEN_UPSAMPLE:

    #sanity check

    train_hi = train

    train_hi[['premise', 'hypothesis']] = list(zip(prems_hi, hyps_hi))

    train_hi[['lang_abv', 'language']] = ['hi', 'Hindi']

    train_hi.to_csv('train_hi.csv', index = False)

train_hi.head()
if GEN_UPSAMPLE:

    #sanity check

    train_bg = train

    train_bg[['premise', 'hypothesis']] = list(zip(prems_bg, hyps_bg))

    train_bg[['lang_abv', 'language']] = ['bg', 'Bulgarian']

    train_bg.to_csv('train_bg.csv', index = False)

train_bg.head()
#translate to English

prem_bag_en = bag.from_sequence(train['premise'].tolist()).map(lambda x: translation(x, lang = 'en'))

hyp_bag_en =  bag.from_sequence(train['hypothesis'].tolist()).map(lambda x: translation(x, lang = 'en'))



if GEN_EN_ONLY:

    #sanity check

    train_en = train

    train_en[['premise', 'hypothesis']] = list(zip(prems_en, hyps_en))

    train_en[['lang_abv', 'language']] = ['en', 'English']

    train_en.to_csv('train_en.csv', index = False)



else:

    train_en = pd.read_csv("../input/contradictorytranslatedtrain/train_en.csv")

    

#sanity check

train_en.head()