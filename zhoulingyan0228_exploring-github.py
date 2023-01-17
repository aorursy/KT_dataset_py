import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import bq_helper
github = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="github_repos")
query = ("""
        SELECT L.license, COUNT(sf.path) AS number_of_files
        FROM `bigquery-public-data.github_repos.files` as sf
        INNER JOIN `bigquery-public-data.github_repos.licenses` as L 
            ON sf.repo_name = L.repo_name
        GROUP BY L.license
        ORDER BY number_of_files DESC
        """)

file_count_by_license = github.query_to_pandas_safe(query, max_gb_scanned=200)
sns.barplot(x='license', y='number_of_files', data=file_count_by_license);
plt.xticks(rotation=45);
query = ("""
        WITH tmp1 AS (
        SELECT repo_name, SUM(size) AS total_size
        FROM `bigquery-public-data.github_repos.files` f
        INNER JOIN `bigquery-public-data.github_repos.contents` c
          ON f.id = c.id
        GROUP BY repo_name
        ), tmp2 AS (
        SELECT tmp1.repo_name, SUM(size) AS binary_size
        FROM `bigquery-public-data.github_repos.files` f
        INNER JOIN `bigquery-public-data.github_repos.contents` c
          ON f.id = c.id
          AND c.binary=True
        INNER JOIN tmp1
          ON tmp1.repo_name = f.repo_name
        GROUP BY repo_name
        ), tmp3 AS(
        SELECT tmp1.repo_name, SUM(size) AS text_size
        FROM `bigquery-public-data.github_repos.files` f
        INNER JOIN `bigquery-public-data.github_repos.contents` c
          ON f.id = c.id
          AND c.binary=False
        INNER JOIN tmp1
          ON tmp1.repo_name = f.repo_name
        GROUP BY repo_name
        )
        SELECT tmp1.repo_name, total_size, binary_size, text_size
        FROM tmp1
        INNER JOIN tmp2
          ON tmp1.repo_name = tmp2.repo_name
        INNER JOIN tmp3
          ON tmp2.repo_name = tmp3.repo_name
        ORDER BY total_size
        LIMIT 20
        """)

repo_sizes = github.query_to_pandas_safe(query, max_gb_scanned=200)

g = sns.factorplot(x='repo_name', y='value', hue='variable', 
               data=repo_sizes.melt(id_vars=['repo_name'], value_vars=['total_size', 'binary_size', 'text_size']),
               size=10,
               kind='bar'
              )
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=90)
query = ("""
        WITH tmp as (
        SELECT language.name AS name, language.bytes as bytes
        FROM `bigquery-public-data.github_repos.languages` l
        CROSS JOIN UNNEST(l.language) as language
        )
        SELECT name, SUM(bytes) as total_bytes
        FROM tmp
        GROUP BY name
        ORDER BY total_bytes DESC
        """)

language_bytes = github.query_to_pandas_safe(query, max_gb_scanned=200)
plt.pie(x=language_bytes['total_bytes'][:10], labels=language_bytes['name'][:10], autopct='%.0f%%');
language_bytes
query = ("""
        SELECT repo_name, COUNT(1) as commit_count
        FROM `bigquery-public-data.github_repos.commits` c
        CROSS JOIN UNNEST(c.repo_name) as repo_name
        GROUP BY repo_name
        ORDER BY commit_count DESC
        LIMIT 10
        """)

repo_commits = github.query_to_pandas_safe(query, max_gb_scanned=200)
g = sns.factorplot(x='repo_name', y='commit_count', 
               data=repo_commits,
               size=10,
               kind='bar'
              )
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=90)
query = ("""
        SELECT repo_name, COUNT(DISTINCT author.name) as author_count
        FROM `bigquery-public-data.github_repos.commits` c
        CROSS JOIN UNNEST(c.repo_name) as repo_name
        GROUP BY repo_name
        ORDER BY author_count DESC
        LIMIT 10
        """)

repo_commits = github.query_to_pandas_safe(query, max_gb_scanned=200)
g = sns.factorplot(x='repo_name', y='author_count', 
               data=repo_commits,
               size=10,
               kind='bar'
              )
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=90)
query = ("""
        SELECT author.name as author, COUNT(1) as commit_count
        FROM `bigquery-public-data.github_repos.commits` c
        GROUP BY author
        ORDER BY commit_count DESC
        LIMIT 10
        """)

repo_commits = github.query_to_pandas_safe(query, max_gb_scanned=200)
g = sns.factorplot(x='author', y='commit_count', 
               data=repo_commits,
               size=10,
               kind='bar'
              )
for ax in g.axes.flat:
    labels = ax.get_xticklabels()
    ax.set_xticklabels(labels, rotation=90)
query = ("""
        WITH tmp as (
        SELECT CAST(SUM(language.bytes)/1024 AS INT64) as kilos
        FROM `bigquery-public-data.github_repos.languages` l
        CROSS JOIN UNNEST(l.language) as language
        GROUP BY repo_name
        HAVING kilos < 1024
        )
        SELECT kilos
          , COUNT(1) AS count
        FROM tmp
        GROUP BY kilos
        ORDER BY kilos ASC
        """)

repo_commits = github.query_to_pandas_safe(query, max_gb_scanned=200)
g = sns.factorplot(x='kilos', y='count', 
               data=repo_commits,
               size=10,
               kind='bar'
              )
for ax in g.axes.flat:
    ax.set_xticklabels([], rotation=90)