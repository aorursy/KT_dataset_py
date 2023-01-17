from google.cloud import bigquery



client = bigquery.Client()
query = """

SELECT country_code, COUNT(*) AS cnt

FROM `patents-public-data.patents.publications`

GROUP BY country_code;

"""



df = client.query(query).to_dataframe()

df.head(20)

df.sort_values('cnt', ascending=False).head(20).plot.bar(x='country_code', y='cnt')
df.to_csv("data.csv")