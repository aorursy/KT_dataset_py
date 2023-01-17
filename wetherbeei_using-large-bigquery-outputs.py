PROJECT_ID = "jefferson-1790"

OUTPUT_BUCKET = "jefferson-1790-examples"

OUTPUT_PATH = "data-export-test.csv"

FORMAT = "CSV" # "NEWLINE_DELIMITED_JSON" is also supported for results with arrays

COMPRESSION = "NONE" # "GZIP" is also supported for compressed results



query = """

  SELECT

    ARRAY_TO_STRING(cd.parents, "|"),

    CAST(FLOOR(filing_date / 10000) AS INT64) AS filing_year

  FROM (

    SELECT ANY_VALUE(cpc) AS cpc, ANY_VALUE(filing_date) AS filing_date

    FROM `patents-public-data.patents.publications`

    WHERE application_number != ""

    GROUP BY application_number

  ), UNNEST(cpc) AS cpcs

  JOIN `patents-public-data.cpc.definition` cd ON cd.symbol = cpcs.code

  WHERE cpcs.first = TRUE AND filing_date > 0

"""
import time

import io

import pandas as pd



from google.cloud import bigquery

bigquery_client = bigquery.Client(project=PROJECT_ID)

from google.cloud import storage

storage_client = storage.Client(project=PROJECT_ID)
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
df_shards = []

for blob in storage_client.list_blobs(OUTPUT_BUCKET, prefix=OUTPUT_PATH):

    print("Loading %s" % blob.name)

    file_obj = io.BytesIO()

    storage_client.download_blob_to_file(

        "gs://%s/%s" % (OUTPUT_BUCKET, blob.name), file_obj)

    file_obj.seek(0)

    df_compression = None

    if COMPRESSION == "GZIP":

        df_compression = "gzip"

    df = pd.read_csv(file_obj, compression=df_compression)

    print(df.head())

    df_shards.append(df)

df = pd.concat(df_shards)

df_shards = None

df
for blob in storage_client.list_blobs(OUTPUT_BUCKET, prefix=OUTPUT_PATH):

    print("Deleting %s from %s" % (blob.name, OUTPUT_BUCKET))

    storage_client.bucket(OUTPUT_BUCKET).delete_blob(blob.name)