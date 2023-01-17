# all we need is bigquery

from google.cloud import bigquery



# there's no need for passing any credentials since access is managed by kaggle

bq = bigquery.Client()
# just write your sql query

sql_query = """

SELECT

  DISTINCT(word)

FROM

  `bigquery-public-data.samples.shakespeare`

WHERE

  word LIKE '%ing' OR word LIKE '%ed';

"""

# and run the query to get the result in a dataframe

results = bq.query(sql_query).to_dataframe()
# now this result dataframe can be used as you want

results.head(10)
# let's try another sample query

all_python_repos_activity_query = """

SELECT

  *

FROM

  `bigquery-public-data.samples.github_timeline`

WHERE

  repository_url LIKE '%python%'

"""

all_python_repos_activity_df = bq.query(all_python_repos_activity_query).to_dataframe()
# once you get the response dataframe, you can keep utilizing it as you would normally use pandas dataframe

all_python_repos_activity_df.head(10)
# let's do a normal describe

all_python_repos_activity_df.describe()
# let's get unique values in a column

all_python_repos_activity_df['repository_url'].unique()
# get smaller dataframe which contains one row per repository URL

unique_repositories_df = all_python_repos_activity_df.drop_duplicates(subset='repository_url', keep='last')
unique_repositories_df['repository_watchers'].plot()
unique_repositories_df.plot.scatter('repository_forks', 'repository_open_issues')
repo_languages_df = unique_repositories_df.groupby(['repository_language']).mean()

repo_languages_df.head(100)
repo_languages_df['repository_forks'].plot.bar()
repo_languages_df['repository_size'].plot.density()