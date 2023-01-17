# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import torch

from torch import nn

import time

import io

!pip install pyhash

import pyhash
PROJECT_ID = "jefferson-1790"

OUTPUT_BUCKET = "jefferson-1790-examples"

OUTPUT_PATH = "abstract-to-cpc.csv"

FORMAT = "CSV" # "NEWLINE_DELIMITED_JSON" is also supported for results with arrays

COMPRESSION = "GZIP"



from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)
if False:

    # Only re-run if needed.

    query = """

      SELECT publication_number, abstract, ARRAY_TO_STRING(ARRAY(SELECT cpcs.code FROM UNNEST(cpc) AS cpcs), "|") AS cpc_list

      FROM `patents-public-data.google_patents_research.publications`

      WHERE abstract != "" AND ARRAY_LENGTH(cpc) > 0

    """



    bigquery_client.create_dataset("%s.exports" % PROJECT_ID, exists_ok=True)

    table_id = "%s.exports.data_%d" % (PROJECT_ID, int(time.time()))

    job_config = bigquery.QueryJobConfig(allow_large_results=True, destination=table_id)

    job = bigquery_client.query(query, job_config)

    query_result = job.result()

    query_pages = query_result.pages

    print("Result rows: %d" % query_result.total_rows)

    print("Result schema: %s" % query_result.schema)

    first_page = next(query_pages)

    for i, item in enumerate(first_page):

        print(item)

        if i > 5:

            break

    extract_config = bigquery.job.ExtractJobConfig(

        destination_format=FORMAT, compression=COMPRESSION)

    extract_result = bigquery_client.extract_table(

        bigquery.table.TableReference.from_string(table_id),

        destination_uris=["gs://%s/%s-*" % (OUTPUT_BUCKET, OUTPUT_PATH)],

        job_config=extract_config)

    extract_result.result()

    blobs = storage_client.list_blobs(OUTPUT_BUCKET, prefix=OUTPUT_PATH)

    for blob in blobs:

        print(blob.name)

    bigquery_client.delete_table(table_id)
from torchtext.data.utils import ngrams_iterator

from torchtext.data.utils import get_tokenizer



tokenizer = get_tokenizer("basic_english")

NGRAMS = 1



file_cache = {}

def get_training_shards():

    for blob in storage_client.list_blobs(OUTPUT_BUCKET, prefix=OUTPUT_PATH):

        print("Loading %s" % blob.name)

        if blob.name in file_cache:

            yield file_cache[blob.name]

        file_obj = io.BytesIO()

        storage_client.download_blob_to_file(

            "gs://%s/%s" % (OUTPUT_BUCKET, blob.name), file_obj)

        file_obj.seek(0)

        df_compression = None

        if COMPRESSION == "GZIP":

            df_compression = "gzip"

        df = pd.read_csv(file_obj, compression=df_compression)

        print(df.head())

        file_cache[blob.name] = df

        yield df



def get_batch():

    for df in get_training_shards():

        for row in df.itertuples(index=False):

            text = [x for x in ngrams_iterator(tokenizer(row.abstract), NGRAMS)]

            pos_classes = row.cpc_list.split("|")

            yield (text, pos_classes)
for (text, pos_classes) in get_batch():

    print(text)

    print(pos_classes)

    break
from collections import namedtuple

import random



class TextEmbed(nn.Module):

    def __init__(self, vocab_size, embed_dim):

        super().__init__()

        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, max_norm=1.0, norm_type=2.0, sparse=True, mode="sum")

        self.embedding.weight.data.uniform_(-0.05, 0.05)



    def forward(self, text):

        # sum + l2 norm

        emb = self.embedding(text, torch.tensor([0]))

        return nn.functional.normalize(emb, p=2)



class WARPLoss(nn.Module):

    def __init__(self, embedding, num_classes, margin, max_tries):

        super().__init__()

        self.num_classes = num_classes

        self.margin = margin

        self.max_tries = max_tries

        self.embedding = embedding



    def similarity(self, doc_emb, cls_emb):

        return (doc_emb * cls_emb).sum()



    def forward(self, doc_emb, pos_classes):

        num_samples = 0

        # Calculate similarity between doc and a random positive class.

        pos_class = pos_classes[torch.tensor(random.randint(0, pos_classes.size()[0]-1))]

        pos_emb = self.embedding(pos_class)

        doc_pos_sim = self.similarity(doc_emb, pos_emb)

        while num_samples < self.max_tries:

            # Sample one id in range 0, NUM_CLASS.

            neg_id = torch.tensor([random.randint(0, self.num_classes-1)])

            if neg_id in pos_classes:

                continue

            # Reject if id is a positive.

            num_samples += 1

            # Calculate similarity between doc and negative sample.

            neg_emb = self.embedding(neg_id)

            doc_neg_sim = self.similarity(doc_emb, neg_emb)

            # If < margin away from positive score, calculate loss.

            loss = doc_neg_sim - (doc_pos_sim - self.margin)

            if loss > 0:

                loss *= 1.0 / float(num_samples)

                break

            else:

                loss = None

        return loss, num_samples, pos_class, doc_pos_sim, neg_id, doc_neg_sim





class VocabCount:

    __slots__ = ["vocab", "count"]

    def __init__(self, vocab=None, count=None):

        self.vocab = vocab

        self.count = count



class Vocab:

    """On-the-fly vocab bucket assignment when your dataset is too large."""

    def __init__(self, max_vocab, oov_buckets):

        self.hasher = pyhash.fnv1_32() 

        self.max_vocab = max_vocab

        self.vocab_to_id_count = {}

        self.id_to_vocab = {}

        self.oov_buckets = oov_buckets

    

    def to_id(self, vocab, count=True):

        # Hash to a bucket.

        h = self.hasher(vocab)

        bucket = h % self.max_vocab

        # Increment our count in that bucket.

        matching_vocabs = self.vocab_to_id_count.get(bucket, None)

        if not matching_vocabs:

            vc = VocabCount(vocab=vocab, count=1)

            self.vocab_to_id_count[bucket] = [vc]

            matching_vocabs = [vc]

        for vc in matching_vocabs:

            if vc.vocab == vocab and count:

                vc.count += 1

        highest_vocab = max(matching_vocabs, key=lambda x: x.count)

        # If we are the highest count in our bucket, return the hash id.

        if highest_vocab.vocab == vocab:

            self.id_to_vocab[bucket] = highest_vocab

            return bucket

        # If we aren't the highest count, return one of the oov buckets.

        return self.max_vocab + (h % self.oov_buckets)



    def to_vocab(self, bucket):

        # Return the highest vocab for the bucket. This doesn't take into

        # account collisions.

        if bucket < self.max_vocab:

            if bucket not in self.id_to_vocab:

                return "UNK"

            return self.id_to_vocab[bucket].vocab

        # If id is an oov bucket, return "OOV"

        return "OOV%d" % (bucket - self.max_vocab)





device = "cpu"

VOCAB_SIZE = 100000

VOCAB_SIZE_OOV = 4000

EMBED_DIM = 32

NUM_CLASS = 1000000

text_vocab = Vocab(VOCAB_SIZE, VOCAB_SIZE_OOV)

class_vocab = Vocab(NUM_CLASS, 1)

doc_model = TextEmbed(VOCAB_SIZE+VOCAB_SIZE_OOV, EMBED_DIM).to(device)

class_emb = nn.Embedding(NUM_CLASS+1, EMBED_DIM, max_norm=1.0, norm_type=2.0, sparse=True).to(device)

class_emb.weight.data.uniform_(-0.05, 0.05)

loss_model = WARPLoss(class_emb, NUM_CLASS+1, 0.1, 4000).to(device)



optimizer = torch.optim.SGD(list(doc_model.parameters()) + list(class_emb.parameters()) + list(loss_model.parameters()), lr=1.0)



MAX_STEPS = 100000

PRINT_STEPS = 1000

step = 0

num_samples_sum = 0.0

loss_sum = 0.0

doc_pos_sim_sum = 0.0

doc_neg_sim_sum = 0.0



while step < MAX_STEPS:

    for (text, pos_classes) in get_batch():

        if step >= MAX_STEPS:

            break

        if step % 1000 == 0:

            print(f"Step:\t{step}\tLoss:\t{loss_sum/PRINT_STEPS}\tSamples:\t{num_samples_sum/PRINT_STEPS}\tPos:\t{doc_pos_sim_sum/PRINT_STEPS}\tNeg:\t{doc_neg_sim_sum/PRINT_STEPS}")

            num_samples_sum = 0.0

            loss_sum = 0.0

            doc_pos_sim_sum = 0.0

            doc_neg_sim_sum = 0.0

        # Map to vocab.

        text_ids = torch.tensor([text_vocab.to_id(x) for x in text]).to(device)

        class_ids = torch.tensor([class_vocab.to_id(x) for x in pos_classes]).to(device)

        optimizer.zero_grad()

        doc_emb = doc_model(text_ids)

        loss, num_samples, pos_id, doc_pos_sim, neg_id, doc_neg_sim = loss_model(doc_emb, class_ids)

        num_samples_sum += num_samples

        doc_pos_sim_sum += doc_pos_sim.item()

        doc_neg_sim_sum += doc_neg_sim.item()

        if loss:

            loss_sum += loss.item()

            loss.backward()

            optimizer.step()

        if step % 10000 == 0:

            text = [x for x in ngrams_iterator(tokenizer("A visual prosthesis apparatus and a method for limiting power consumption in a visual prosthesis apparatus. The visual prosthesis apparatus comprises a camera for capturing a video image, a video processing unit associated with the camera, the video processing unit configured to convert the video image to stimulation patterns, and a retinal stimulation system configured to stop stimulating neural tissue in a subject's eye based on the stimulation patterns when an error is detected in a forward telemetry received from the video processing unit."), NGRAMS)]

            clss = ["A61F9/08", "A61N1/3787", "H04N5/2254", "F24H7/00", "C12N15/8509", "C07K14/53"]

            with torch.no_grad():

                test_text_ids = torch.tensor([text_vocab.to_id(x, count=False) for x in text]).to(device)

                test_class_ids = torch.tensor([class_vocab.to_id(x, count=False) for x in pos_classes]).to(device)

                test_emb = doc_model(test_text_ids)

                print("Sample US-8000000-B2")

                for cls in clss:

                    cls_id = torch.tensor([class_vocab.to_id(cls)]).to(device)

                    cls_emb = class_emb(cls_id)

                    sim = loss_model.similarity(test_emb, cls_emb)

                    print(cls, sim)

        step += 1
