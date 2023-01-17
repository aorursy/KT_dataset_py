import pandas as pd # data processing

import collections

import itertools



import bq_helper # accessing bigQuery database



stackoverflow = bq_helper.BigQueryHelper("bigquery-public-data","stackoverflow")
query4 = """SELECT tags, extract(dayofweek from creation_date) = 1 or extract(dayofweek from creation_date) = 7 as is_weekend

         FROM `bigquery-public-data.stackoverflow.posts_questions`

         LIMIT 1000000;

         """

alltags = stackoverflow.query_to_pandas_safe(query4)

tags = ((tag, row['is_weekend']) for _, row in alltags.iterrows() for tag in row['tags'].split('|'))

counter=collections.Counter(tags)
most_common = counter.most_common(10000)



weekend_tags = {}

weekday_tags = {}

for (tag, is_weekend), count in most_common:

    if is_weekend:

        weekend_tags[tag] = count

    else:

        weekday_tags[tag] = count



relative_tags = {}

for tag in iter(weekend_tags):

    if tag in weekday_tags:

        weekend_count = weekend_tags[tag]

        weekday_count = weekday_tags[tag]

        relative_tags[tag] = weekday_count + weekend_count



relative_tags_list = list(relative_tags.items())

relative_tags_list.sort(key = lambda entry: entry[1], reverse = True)



for row in relative_tags_list:

    print("{},{}".format(row[0], row[1]))
prog_langs = ["coldfusion", "vb6", "sed", "typescript", "vba", "powershell", "kotlin", "excel", "bash", "awk", "sql", "r", "rust", "c#", "vb.net", "processing", "ruby", "python", "c++", "haskell", "vbscript", "java", "oracle", "javascript", "matlab", "lua", "php", "scala", "c", "glsl", "swift", "erlang", "tcl", "elixir", "go", "common-lisp", "fortran", "f#", "prolog", "scheme", "racket"]



prog_langs_table = []

for tag in prog_langs:

    weekend_count = weekend_tags[tag]

    weekday_count = weekday_tags[tag]

    total = weekend_count + weekday_count

    prog_langs_table.append(

        [tag, total, 100 * weekend_count / total]

    )

prog_langs_table.sort(key = lambda row: row[2], reverse = True)

pd.DataFrame(prog_langs_table, columns=["Tag", "Count", "% of Qs on weekends"])