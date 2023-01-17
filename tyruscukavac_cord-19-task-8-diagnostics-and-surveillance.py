from pathlib import Path

input_data_dirpath = Path("..","input")

cord_19_literature_data_dirpath = input_data_dirpath / "CORD-19-research-challenge"

files = [file for file in cord_19_literature_data_dirpath.glob("*/**")]
import pandas as pd

def get_metadata(cord_19_literature_data_dirpath):

    return pd.read_csv(

            cord_19_literature_data_dirpath / "metadata.csv",

            low_memory=False

            )
metadata = get_metadata(cord_19_literature_data_dirpath)

metadata.columns

metadata.head()
comm_use_subset_filepaths = list((

        cord_19_literature_data_dirpath

        / "comm_use_subset"

        / "comm_use_subset"

        ).glob("pdf_json/*.json")

        )

data_filepaths = comm_use_subset_filepaths
import json
from tqdm import tqdm

def get_data(data_filepaths):

    data = []

    for data_filepath in tqdm(data_filepaths):

        with open(data_filepath) as data_file:

            data_object = json.load(data_file)

            abstract = data_object.get("abstract")

            if abstract:

                for abstract_item in abstract:

                    data.append({  

                    "paper_id": data_object["paper_id"],

                    "text": abstract_item["text"],

                    "text_type": "abstract"

                    })

            body_text = data_object.get("body_text")

            if body_text:

                for body_text_item in body_text:

                    data.append({

                        "paper_id":data_object["paper_id"],

                        "text":body_text_item["text"],

                        "text_type": "body"

                    })

    data = pd.DataFrame.from_dict(data=data)

    return data
data = get_data(data_filepaths)
def sample_data(data, frac, random_state=1):

    return data.sample(frac=frac, random_state=random_state)
development_data = sample_data(data,frac=.001)
!pip install scispacy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import keras.preprocessing.sequence

import numpy as np

import scispacy

import spacy
nlp = spacy.load("en_core_sci_lg", disable=['parser', 'tagger', 'ner'])

sentencizer = nlp.create_pipe("sentencizer")

nlp.add_pipe(sentencizer)
def get_sentence_vectors_from_data(data, maxlen=None):

    """

    Apply a language model to the data and vectorize by sentence.

    Returns a numpy array of sentence vectors, padded by 0.0

    Keyword arguments:

    data -- a pandas DataFrame with a "text" column 

    """

    texts = data["text"]

    vectors = np.array([

        [sent.vector for sent in nlp(text).sents]

        for text in tqdm(texts)

        ])

    vectors = keras.preprocessing.sequence.pad_sequences(

        vectors,

        dtype='float32',

        value=0.0,

        maxlen=maxlen

        )

    return vectors
development_data_sentence_vectors = get_sentence_vectors_from_data(development_data)
from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(development_data_sentence_vectors, random_state=1)
import keras.losses 

import keras.metrics

import keras.optimizers

from keras.layers import Input, LSTM, RepeatVector

from keras.models import Model
def get_autoencoder(

        timesteps,

        input_dim,

        encoding_dim

        ):

    '''

    Factory function for the default auto encoder model 

    (Heavily inspired by the Keras AutoEncoder tutorial by Francois Chollet)

    '''

    encoder_inputs = Input((timesteps, input_dim))

    encoded_1 = LSTM(input_dim, return_sequences=True)(encoder_inputs)

    encoded_2 = LSTM(encoding_dim, return_sequences=True)(encoded_1)

    encoded_3 = LSTM(encoding_dim)(encoded_2)

    decoder_inputs = RepeatVector(timesteps)(encoded_3)

    decoded_1 = LSTM(encoding_dim,

                     return_sequences=True)(decoder_inputs)

    decoded_2 = LSTM(input_dim,

                     return_sequences=True)(decoded_1)

    decoded_3 = LSTM(input_dim, return_sequences=True)(decoded_2)

    autoencoder_model = Model(encoder_inputs, decoded_3)

    encoder_model = Model(encoder_inputs, encoded_3)

    for model in autoencoder_model, encoder_model:

        model.compile(

            optimizer=keras.optimizers.rmsprop(),

            loss=keras.losses.mean_squared_error,

            # loss=keras.losses.cosine_proximity,

            #loss=keras.losses.logcosh,

            metrics=[

                keras.metrics.cosine_proximity,

                keras.metrics.mean_squared_error,

                keras.metrics.logcosh

                ]

            )

    return autoencoder_model, encoder_model
autoencoder_model, encoder_model = get_autoencoder(

        X_train.shape[1], # The number of time steps

        X_train.shape[2], # The dimension of the vectors

        X_train.shape[2] // 2 , # The dimension to encode to (half of a word vector)

        )
import keras.callbacks
history = autoencoder_model.fit(

        X_train, 

        X_train, 

        epochs=5, # Reduced for sake of demo

        batch_size=32,

        shuffle=True,

        validation_split=.1,

        callbacks = [

            keras.callbacks.callbacks.EarlyStopping(patience=10),

            ]

        )

maxlen = encoder_model.get_layer(index=0).input_shape[1] 
autoencoder_model.evaluate(X_test, X_test)
from IPython.display import SVG

from keras.utils import model_to_dot

SVG(model_to_dot(autoencoder_model, dpi=72).create(prog='dot', format='svg'))

SVG(model_to_dot(encoder_model, dpi=72).create(prog='dot', format='svg'))
querying_data = sample_data(data, frac=.001) 

    # For simplicity of the demo, we'll sample .1% of the total data

    # The full query system uses the entire data set in comm_use_subset

    # However, vectorization takes a bit of time.
from scipy import spatial

class QueryEngine:

    def __init__(self, raw_data, metadata, encode):

        # Let's fix an inconsistency with the metadata 

        metadata = metadata.rename(columns={"sha":"paper_id"})

        self.data = raw_data.merge(metadata, how="left", on="paper_id")

        self.encode = encode



        # Encoding the querying data.

        data_encoded = self.encode(

                self.data

                )

        # Creating a spatial tree to store the data for querying.

        self.data_space = spatial.KDTree(

            data_encoded,

            leafsize=500

            )

    def query(self, query_string, max_results=20):

        # Turn the query string into the appropriate type:

        query_object = pd.DataFrame([{"text": query_string}])

        # Encode the query

        query_encoded = self.encode(query_object)

        query_results = self.data_space.query(query_encoded, k=max_results)

        query_results_data = pd.DataFrame([

            self.data.loc[query_result_indices][["text","text_type","title","abstract","paper_id"]]

            for query_result_indices in query_results[1][0] # The second value in the tuple

                                                           # are the indices

            ])

        return query_results_data
def make_encode(model):

    def encode(texts):

        maxlen = model.get_layer(index=0).input_shape[1]

        return model.predict(

                get_sentence_vectors_from_data(

                    texts,

                    maxlen=maxlen

                    )

                )

    return encode
tiny_query_engine = QueryEngine(

        querying_data, 

        metadata,

        encode=make_encode(encoder_model)

        )
query = ("""

What do we know about diagnostics and surveillance? What has been<br>

published concerning systematic, holistic approach to diagnostics (from the<br>

public health surveillance perspective to being able to predict clinical outcomes)?  <br>

""")
query_results = tiny_query_engine.query(query)
query_results[["abstract","text"]].head()
# from keras.models import load_model

#query_engine = QueryEngine(

#    data,

#    metadata,

#    make_encode(

#        load_model(

#            (input_data_dirpath / "models" / "encoder").with_suffix(".h5")

#            )

#        ),

#    )
task_queries = [

   """

   What do we know about diagnostics and surveillance? What has been published

   concerning systematic, holistic approach to diagnostics (from the public

   health surveillance perspective to being able to predict clinical

   outcomes)?

   """,

   """

   How widespread current exposure is to be able to make immediate policy

   recommendations on mitigation measures. Denominators for testing and a

   mechanism for rapidly sharing that information, including demographics, to

   the extent possible. Sampling methods to determine asymptomatic disease

   (e.g., use of serosurveys (such as convalescent samples) and early

       detection of disease (e.g., use of screening of neutralizing antibodies

           such as ELISAs).

   """,

   """

   Efforts to increase capacity on existing diagnostic platforms and tap into

   existing surveillance platforms.

   """,

   """

   Recruitment, support, and coordination of local expertise and capacity

   (public, private—commercial, and non-profit, including academic), including

   legal, ethical, communications, and operational issues.

   """,

   """

   National guidance and guidelines about best practices to states (e.g., how

   states might leverage universities and private laboratories for testing

   purposes, communications to public health officials and the public).

   """,

   """

   Development of a point-of-care test (like a rapid influenza test) and rapid

   bed-side tests, recognizing the tradeoffs between speed, accessibility, and

   accuracy.

   """,

   """

   Rapid design and execution of targeted surveillance experiments calling for

   all potential testers using PCR in a defined area to start testing and

   report to a specific entity. These experiments could aid in collecting

   longitudinal samples, which are critical to understanding the impact of ad

   hoc local interventions (which also need to be recorded).

   """,

   """

   Separation of assay development issues from instruments, and the role of

   the private sector to help quickly migrate assays onto those devices.

   """,

   """

   Efforts to track the evolution of the virus (i.e., genetic drift or

   mutations) and avoid locking into specific reagents and

   surveillance/detection schemes.

   """,

   """

   Latency issues and when there is sufficient viral load to detect the

   pathogen, and understanding of what is needed in terms of biological and

   environmental sampling.

   """,

   """

   Use of diagnostics such as host response markers (e.g., cytokines) to

   detect early disease or predict severe disease progression, which would be

   important to understanding best clinical practice and efficacy of

   therapeutic interventions.

   """,

   """

   Policies and protocols for screening and testing.

   """,

   """

   Policies to mitigate the effects on supplies associated with mass testing,

   including swabs and reagents.

   """,

   """

   Technology roadmap for diagnostics.

   """,

   """

   Barriers to developing and scaling up new diagnostic tests (e.g., market

   forces), how future coalition and accelerator models (e.g.,

   Coalition for Epidemic Preparedness Innovations) could

   provide critical funding for diagnostics, and opportunities for a

   streamlined regulatory environment.

   """,

   """

   New platforms and technology (e.g., CRISPR) to improve response times and

   employ more holistic approaches to COVID-19 and future diseases.

   """,

   """

   Coupling genomics and diagnostic testing on a large scale.

   """,

   """

   Enhance capabilities for rapid sequencing and bioinformatics to target

   regions of the genome that will allow specificity for a particular variant.

   """,

   """

   Enhance capacity (people, technology, data) for sequencing with advanced

   analytics for unknown pathogens, and explore capabilities for

   distinguishing naturally-occurring pathogens from intentional.

   """,

   """

   One Health surveillance of humans and potential sources of future spillover

   or ongoing exposure for this organism and future pathogens, including both

   evolutionary hosts (e.g., bats) and transmission hosts (e.g., heavily

   trafficked and farmed wildlife and domestic food and companion

   species), inclusive of environmental, demographic, and occupational

   risk factors.

   """

   ]
#task_queries_results = [query_engine.query(query) for task_query in

#        task_queries]

#task_queries_results_data_frame = pd.concat([

#        query_results.assign(query=query)

#        for query, query_results in zip(task_queries, task_queries_results)

#    ])
loaded_task_queries_results_data_frame = pd.read_csv(

        (input_data_dirpath

         / "results"

         / "task_queries_results_data_frame").with_suffix(".csv")

        )



loaded_task_queries_results_data_frame.head()
def print_result_text(query, results):

    results = results.loc[results["query"] == query]

    for index, result in enumerate(results.iterrows()):

        title = result[1]["title"]

        text = result[1]["text"]

        text_type = result[1]["text_type"]

        print(f"Result {index}: ")

        print(f"\tText: {text}")

        print(f"\tText Type: {text_type}")

        print(f"\tArticle Title: {title}")

    return results
print_result_text(task_queries[0], loaded_task_queries_results_data_frame)
print_result_text(task_queries[1], loaded_task_queries_results_data_frame)
print_result_text(task_queries[2], loaded_task_queries_results_data_frame)
print_result_text(task_queries[3], loaded_task_queries_results_data_frame)
print_result_text(task_queries[4], loaded_task_queries_results_data_frame)
print_result_text(task_queries[5], loaded_task_queries_results_data_frame)
print_result_text(task_queries[6], loaded_task_queries_results_data_frame)
print_result_text(task_queries[7], loaded_task_queries_results_data_frame)
print_result_text(task_queries[8], loaded_task_queries_results_data_frame)
print_result_text(task_queries[9], loaded_task_queries_results_data_frame)
print_result_text(task_queries[10], loaded_task_queries_results_data_frame)
print_result_text(task_queries[11], loaded_task_queries_results_data_frame)
print_result_text(task_queries[12], loaded_task_queries_results_data_frame)
print_result_text(task_queries[13], loaded_task_queries_results_data_frame)
print_result_text(task_queries[14], loaded_task_queries_results_data_frame)
print_result_text(task_queries[15], loaded_task_queries_results_data_frame)
print_result_text(task_queries[16], loaded_task_queries_results_data_frame)
print_result_text(task_queries[17], loaded_task_queries_results_data_frame)
print_result_text(task_queries[18], loaded_task_queries_results_data_frame)
print_result_text(task_queries[19], loaded_task_queries_results_data_frame)