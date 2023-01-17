import json

import pickle



import numpy as np

import pandas as pd

from tqdm import tqdm_notebook as tqdm

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import text, sequence
def build_test(test_path):

    with open(test_path) as f:

        processed_rows = []



        for line in tqdm(f):

            line = json.loads(line)



            text = line['document_text'].split(' ')

            question = line['question_text']

            example_id = line['example_id']



            for candidate in line['long_answer_candidates']:

                start = candidate['start_token']

                end = candidate['end_token']



                processed_rows.append({

                    'text': " ".join(text[start:end]),

                    'question': question,

                    'example_id': example_id,

                    'PredictionString': f'{start}:{end}'



                })



        test = pd.DataFrame(processed_rows)

    

    return test
directory = '/kaggle/input/tensorflow2-question-answering/'

test_path = directory + 'simplified-nq-test.jsonl'

test = build_test(test_path)

submission = pd.read_csv("../input/tensorflow2-question-answering/sample_submission.csv")



test.head()
def compute_text_and_questions(test, tokenizer):

    test_text = tokenizer.texts_to_sequences(test.text.values)

    test_questions = tokenizer.texts_to_sequences(test.question.values)

    

    test_text = sequence.pad_sequences(test_text, maxlen=300)

    test_questions = sequence.pad_sequences(test_questions)

    

    return test_text, test_questions
model = load_model('/kaggle/input/tf-qa-new-start/model.h5')

model.summary()
with open('/kaggle/input/tf-qa-new-start/tokenizer.pickle', 'rb') as f:

    tokenizer = pickle.load(f)
test_text, test_questions = compute_text_and_questions(test, tokenizer)
del test['text']

del test['question']
%%time

test_target = model.predict([test_text, test_questions], batch_size=512)
test['target'] = test_target



result = (

    test.query('target > 0.3')

    .groupby('example_id')

    .max()

    .reset_index()

    .loc[:, ['example_id', 'PredictionString']]

)



result.head()
result = pd.concat([

    result.assign(example_id=lambda example_id: example_id + '_long'),

    result.assign(example_id=lambda example_id: example_id + '_short')

])
final_submission = (

    submission.drop(columns='PredictionString')

    .merge(result, on=['example_id'], how='left')

)



final_submission.head()
final_submission.to_csv("submission.csv", index=False)