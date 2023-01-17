import numpy as np 
import pandas as pd
import os
import string
from unidecode import unidecode
import dill
from IPython.display import Image as IPythonImage, display
from PIL import Image
import pytesseract
from Levenshtein import jaro_winkler
import matplotlib.pyplot as plt
# Look at the labels
validation_df = pd.read_csv("../input/handwriting-recognition/written_name_validation_v2.csv")
validation_df.head()
# Display a few randomly chosen images along with their labels
PATH = '../input/handwriting-recognition/validation_v2/validation'

# Print randomly selected images and their labels
NUMBER_OF_IMAGES = 4
random_index = np.random.choice(validation_df.shape[0], NUMBER_OF_IMAGES, False)
for row in validation_df.loc[random_index, :].itertuples(index=False):
    filename = os.path.join(PATH, row.FILENAME)
    print(row.IDENTITY)
    display(IPythonImage(filename=filename))
# Iterate over the data-frame, running Tesseract OCR and recording the output of Tesseract
# Takes a fair bit of time (2 hours)
output = {}
for row in validation_df.itertuples():
    file_path = os.path.join(PATH, row.FILENAME)
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)
    output[row.FILENAME] = text
    if ((row.Index + 1) % 5000) == 0:
        print(f"Processed {row.Index + 1} rows")
# Convert into df
result_df = pd.DataFrame()
for i, (k, v) in enumerate(output.items()):
    if (i + 1) % 5000 == 0:
        print(f"Converted output from {i + 1} images")
    
    text = [t for t in v.split('\n') if t not in ['', ' ', '\n', '\x0c']]
    
    temp_df = pd.DataFrame({
        'FILENAME': [k] * len(text), 
        'TEXT': text
    })
    result_df = pd.concat([result_df, temp_df], ignore_index=True)

# Look at the top 10 text tokens (these will probably be generic terms like "nom")
result_df['TEXT'].value_counts().nlargest(10)
# Filter and get possible variations of "Nom", "Prenom" etc
result_df\
    .loc[result_df['TEXT'].str.upper().str.contains('NOM|PRENOM|DATE DE NAISSANCE CLASSE') , "TEXT"]\
    .head(30)
# Helper function to remove tokens that are common words and stand-alone punctuation
def filter_tokens(token_list, tokens_to_filter, remove_punctuation = True):
    # Punctuation and empty text
    punctuation_list = list(string.punctuation) + ['']
    result = []
    for t in token_list:
        # Remove stand-alone punctuation/empty string
        # Unicode has some fancy punctuation marks (slanting quotes, for example)
        # remove them using unidecode
        if remove_punctuation and unidecode(t) in punctuation_list:
            continue
        
        # Remove if any token in tokens_to_filter list is a substring of current token
        found = 0
        for t_filter in tokens_to_filter:
            if t_filter.upper() in t.upper():
                # token is matched in tokens_to_filter list
                found = 1
                break
        
        if found == 1:
            continue
        else:
            result.append(t)
    
    return result


# Define text to remove
SINGLE_TOKENS_REMOVE = ['NOM', 'PRENOM']
MULTI_TOKENS_REMOVE = ['DATE DE NAISSANCE CLASSE']

# Remove tokens
result_df['CLEAN_TEXT'] = result_df['TEXT']\
    .str\
    .split(' ')\
    .apply(lambda c: ' '.join(filter_tokens(c, SINGLE_TOKENS_REMOVE)))

for multi_token in MULTI_TOKENS_REMOVE:
    result_df['CLEAN_TEXT'] = np.where(result_df['CLEAN_TEXT'].str.contains(multi_token), 
                                       '', 
                                       result_df['CLEAN_TEXT'])

result_df.head(40)
# Combine remaining multiple words into 1 (per filename)
clean_result = result_df\
    .groupby('FILENAME')['CLEAN_TEXT']\
    .apply(''.join)\
    .reset_index()

clean_result.head(20)
# Create 1 dataframe with both actual and OCR labels
ocr_vs_actual = validation_df.merge(clean_result, how='left', on='FILENAME')

# Remove labels which do not exist
ocr_vs_actual = ocr_vs_actual.loc[ocr_vs_actual['IDENTITY'].notnull(), :]

# Remove spaces in OCR output
ocr_vs_actual['CLEAN_TEXT'] = ocr_vs_actual['CLEAN_TEXT'].str.replace('\\s', '', regex=True)
ocr_vs_actual.head(10)
# Create jaro-winkler similarity score
vectorized_jaro_winkler = np.vectorize(jaro_winkler)

ocr_vs_actual['SIMILARITY_SCORE'] = vectorized_jaro_winkler(ocr_vs_actual['IDENTITY'].str.upper(), 
                                                            np.where(ocr_vs_actual['CLEAN_TEXT'].isnull(), 
                                                                     '', 
                                                                     ocr_vs_actual['CLEAN_TEXT'].str.upper()))
ocr_vs_actual.head(10)
# Plot histogram of similarity scores to see how well we did
plt.style.use('seaborn-white')
plt.figure(figsize=(8,3), dpi=120)
plt.hist(ocr_vs_actual['SIMILARITY_SCORE'], bins=50, alpha=0.5, color='steelblue', edgecolor='none')
plt.title('Histogram of Jaro-Winkler similarity score between label and OCR-results')
plt.show()
# Create bins. Non-uniform width.
second_lowest_score = ocr_vs_actual.loc[(ocr_vs_actual['SIMILARITY_SCORE'] != 0), 'SIMILARITY_SCORE'].min()

ocr_vs_actual['BINS'] = pd.cut(ocr_vs_actual['SIMILARITY_SCORE'], 
                               bins=[0] + np.linspace(second_lowest_score, 0.95, 9).tolist() + [0.96, 0.97, 0.98, 0.99, 1.01], 
                               labels = ['no-match', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', 'best-match'],
                               right=False)
ocr_vs_actual['BINS'].value_counts().sort_index()
# Highest similarity score images
NUMBER_OF_IMAGES = 5
random_filename = np.random.choice(ocr_vs_actual.loc[ocr_vs_actual['BINS'] == 'best-match', 'FILENAME'].tolist(), NUMBER_OF_IMAGES, False)
for row in ocr_vs_actual.loc[ocr_vs_actual['FILENAME'].isin(random_filename), :].itertuples(index=False):
    filename = os.path.join(PATH, row.FILENAME)
    print(f"""Filename: {row.FILENAME}\nActual: {row.IDENTITY}\nOCR: {row.CLEAN_TEXT}""")
    display(IPythonImage(filename=filename))
# Mid-range similarity score images
NUMBER_OF_IMAGES = 5
random_filename = np.random.choice(ocr_vs_actual.loc[ocr_vs_actual['BINS'] == '8', 'FILENAME'].tolist(), NUMBER_OF_IMAGES, False)
for row in ocr_vs_actual.loc[ocr_vs_actual['FILENAME'].isin(random_filename), :].itertuples(index=False):
    filename = os.path.join(PATH, row.FILENAME)
    print(f"""Filename: {row.FILENAME}\nActual: {row.IDENTITY}\nOCR: {row.CLEAN_TEXT}""")
    display(IPythonImage(filename=filename))
# No matches
NUMBER_OF_IMAGES = 5
random_filename = np.random.choice(ocr_vs_actual.loc[ocr_vs_actual['BINS'] == 'no-match', 'FILENAME'].tolist(), NUMBER_OF_IMAGES, False)
for row in ocr_vs_actual.loc[ocr_vs_actual['FILENAME'].isin(random_filename), :].itertuples(index=False):
    filename = os.path.join(PATH, row.FILENAME)
    print(f"""Filename: {row.FILENAME}\nActual: {row.IDENTITY}\nOCR: {row.CLEAN_TEXT}""")
    display(IPythonImage(filename=filename))