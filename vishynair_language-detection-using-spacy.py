import pandas as pd
#Reading the Dataset
df = pd.read_csv("/kaggle/input/contradictory-my-dear-watson/train.csv")
#Head of the dataset
df.head()
#Just pulling in first few records
df = df[:50]
#PIP installing the Spacy-Landetect for spaCy
!pip install spacy-langdetect
#importing spacy and LanguageDetector from spacy_langdetect
import spacy
from spacy_langdetect import LanguageDetector
def is_english(text: str) -> bool:
    """Simple Function to detect whether the language of the text is english or not.
    Returns a Boolean output for the same.Input the text column"""
    nlp = spacy.load('en')
    nlp.add_pipe(LanguageDetector(),name='language_detector',last=True)
    return nlp(text)._.language['language'] == 'en'
#Applying the function
df['hypothesis'].apply(is_english)