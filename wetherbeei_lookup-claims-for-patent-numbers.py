from google.cloud import bigquery

# client = bigquery.Client(project="my-project-name")
client = bigquery.Client()
query = """
SELECT patent_number FROM `patents-public-data.examples.google_opn_pledged`
"""

df = client.query(query).to_dataframe()
df
# Match a list of messy numbers to their DOCDB format publication number to join with BigQuery.
import requests

docdb_numbers = []
for row, messy_num in df.itertuples():
    docdb_numbers.append(requests.get(f"https://patents.google.com/api/match?pubnum={messy_num}").text)
print(docdb_numbers)
number_string = ', '.join([f'"{num}"' for num in docdb_numbers])

query = f"SELECT title_localized, abstract_localized, claims_localized FROM `patents-public-data.patents.publications` WHERE publication_number IN ({number_string})"
print(query)
df_text = client.query(query).to_dataframe()
df_text
# Analyze full text here!
df_text.to_csv("data.csv")