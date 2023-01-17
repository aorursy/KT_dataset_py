import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# !python -m spacy download en_core_web_sm

!python -m spacy download en_core_web_lg
import spacy

import numpy as np

import pandas as pd
# Creating a spaCy object

nlp = spacy.load('en_core_web_lg')
movie_description = "Mrs. Warren becomes concerned for Helen's safety when a rash of murders involving 'women with afflictions' hits the neighborhood. All three lives change when Vicky joins Karan and Sneha in college, and Sneha begins feeling drawn towards Vicky. Karan feels he has lost the only woman he has ever loved but knows he cannot do anything about it. The former tenant claims she's been murdered, but there's no record of a murder or even her death.. A singing notebook tells 3 puppets to be creative."
doc = nlp(movie_description)

spacy.displacy.render(doc, style='ent',jupyter=True)
def unique(list1): 

    x = np.array(list1) 

    return np.unique(x)

    

def get_people(doc):

  people = []

  for ent in doc.ents:

    if ent.label_ == 'PERSON':

      people.append(ent.text)

  return unique(people)



people_from_story = get_people(doc)

print(people_from_story)
def get_new_names(name_list):

  total_name_count = len(people_from_story)

  name_datasets = pd.read_csv(r'/kaggle/input/nationalnames/NationalNames.csv')

  names_to_use = name_datasets.sample(total_name_count)

  return list(names_to_use['Name'])



names = get_new_names(people_from_story)

print(names)
def replace_names(new, old, text_to_replace_in):

  new_text = text_to_replace_in

  for i in range(len(old)):

    new_text = new_text.replace(old[i], new[i])

  return new_text

    

new_movie_description = replace_names(names, people_from_story, movie_description)
doc = nlp(new_movie_description)

spacy.displacy.render(doc, style='ent',jupyter=True)
movieset_df = pd.DataFrame(columns=['Character Names', 'Roles'])

movieset_df['Character Names'] = names

print(movieset_df)