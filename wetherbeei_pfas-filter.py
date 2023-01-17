# Output exported as CSV to "gs://jefferson-1790-public/pfas/3flourines.csv.*"

OUTPUT_BUCKET = "jefferson-1790-public"

OUTPUT_PATH = "pfas/3flourines_count.csv"

SMARTS = ["C(F)(F).C(F)(F)", "C(F)(F)(F)C(F)C(F)(F)(F)", "C(F)(F)(F)C(F)", "C(F)(F)(F)CC(F)(F)(F)", "C(F)(F)C(F)C(F)(F)", "C(F)(F)OC(F)(F)"]

#SMARTS = ["[!H]C(F)(F)C(F)-[!H]-[!H]"]
!wget -c https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh

!chmod +x Miniconda3-py37_4.8.3-Linux-x86_64.sh

!bash ./Miniconda3-py37_4.8.3-Linux-x86_64.sh -b -f -p /usr/local

!conda install -q -y -c conda-forge rdkit



import sys

sys.path.append('/usr/local/lib/python3.7/site-packages/')
from rdkit import Chem

from rdkit.Chem.Draw import IPythonConsole

from rdkit import RDLogger



RDLogger.DisableLog('rdApp.*')

Chem.MolFromSmiles('c1ccccc1')
import time

import io

import pandas as pd

import numpy as np



from google.cloud import storage

storage_client = storage.Client(project="jefferson-1790")
blobs = storage_client.list_blobs(OUTPUT_BUCKET, prefix=OUTPUT_PATH)

for blob in blobs:

    print(blob.name)
df_shards = []

for blob in storage_client.list_blobs(OUTPUT_BUCKET, prefix=OUTPUT_PATH):

    print("Loading %s" % blob.name)

    file_obj = io.BytesIO()

    storage_client.download_blob_to_file(

        "gs://%s/%s" % (OUTPUT_BUCKET, blob.name), file_obj)

    file_obj.seek(0)

    df_compression = None # can also set "gzip"

    df = pd.read_csv(file_obj, compression=df_compression)

    print(df.head())

    df_shards.append(df)

df = pd.concat(df_shards)

df_shards = None

df
matches = []

queries = [(s, Chem.MolFromSmarts(s)) for s in SMARTS]



errors = 0

frags = 0

for i, row in df.iterrows():

    if i % 1000 == 0:

        print(f"#{i}, frags: {frags}, matches: {len(matches)}, errors: {errors}")

    smiles = row["smiles"]

    # Each fragment.

    for frag_smiles in smiles.split("."):

        frags += 1

        if "N=[N+]=[N-]" in frag_smiles:

            errors += 1

            continue

        candidate = None

        try:

            candidate = Chem.MolFromSmiles(frag_smiles)

        except:

            pass

        if not candidate:

            errors += 1

            continue

        smarts_match = []

        for name, query in queries:

            if candidate.HasSubstructMatch(query):

                smarts_match.append(name)

        if len(smarts_match):

            try:

                inchi_key = Chem.inchi.MolToInchiKey(candidate)

                matches.append({

                    "smiles_frag": frag_smiles,

                    "inchi_key": inchi_key,

                    "smiles_orig": smiles,

                    "sample_pubs": row["sample_pubs"],

                    "pub_count": row["pub_count"],

                    "smarts_match": '|'.join(smarts_match),

                })

            except:

                errors += 1

                continue
out_df = pd.DataFrame(matches)

out_df
df_non_zero = out_df[out_df['inchi_key'] != '']

def process_group(dg):

    return pd.DataFrame([[

                        dg['smiles_frag'].str.cat(sep='|'),

                        dg['smiles_orig'].str.cat(sep='|'),

                        dg['sample_pubs'].str.cat(sep='|'),

                        dg['pub_count'].sum(),

                        '|'.join(set([y for x in dg['smarts_match'].to_list() for y in x.split('|')])),

                        ]], columns=['smiles_frag', 'smiles_orig', 'sample_pubs', 'pub_count', 'smarts_match'])



out_df_reduced = df_non_zero.groupby(['inchi_key']).apply(process_group)
out_df_reduced
out_df_reduced['log_pub_count'] = np.log2(out_df_reduced['pub_count'])

out_df_reduced.hist(column='log_pub_count', bins=100)
# Get rows with more than 5 publications referencing the same InChI.

out_df_over_5 = out_df_reduced[out_df_reduced['pub_count'] > 5]

out_df_over_5.hist(column='log_pub_count', bins=100)
# How many rows are there?

out_df_over_5.shape[0]
out_df_reduced[out_df_reduced['pub_count'] > 20]
out_df_reduced[out_df_reduced['pub_count'] > 100]
out_df_reduced.to_csv('matches.csv')

out_df_over_5.to_csv('matches_over_5.csv')