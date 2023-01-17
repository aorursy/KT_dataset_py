# import functions
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql.functions import udf, explode, flatten, explode_outer
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructType, StructField, DoubleType, LongType, MapType, BooleanType
from pyspark.sql import functions as F
from pyspark.sql import Row
from pyspark.sql.functions import col, concat
import json

# create local session
spark = SparkSession.builder.appName('lol').\
        master("local").\
        getOrCreate()
sqlContext = SQLContext(spark)
# partial shcmea for match data
match_schema = StructType(
    [
        StructField('_c0', IntegerType(), True),
        StructField('gameCreation', DoubleType(), True),
        StructField('gameDuration', DoubleType(), True),
        StructField('gameId', DoubleType(), True),
        StructField('gameMode', StringType(), True),
        StructField('gameType', StringType(), True),
        StructField('gameVersion', StringType(), True),
        StructField('mapId', DoubleType(), True),
        StructField('participantIdentities', StringType(), True),
        StructField('participants',  StringType(), True),
        StructField('platformId', StringType(), True),
        StructField('queueId', DoubleType(), True),
        StructField('seasonId', DoubleType(), True),
        StructField('status.message', StringType(), True),
        StructField('status.status_code', StringType(), True)
    ]
)

# shcmea for itens data
itens_schema = StructType(
    [
        StructField('_c0', IntegerType(), True),
        StructField('item_id', IntegerType(), True),
        StructField('name', StringType(), True),
        StructField('upper_item', StringType(), True),
        StructField('explain', StringType(), True),
        StructField('buy_price', IntegerType(), True),
        StructField('sell_price', IntegerType(), True),
        StructField('tag', StringType(), True)
    ]
)

# shcmea for champions data

champions_schema = StructType(
    [
        StructField('_c0', IntegerType(), True),
        StructField('version', StringType(), True),
        StructField('id', StringType(), True),
        StructField('key', IntegerType(), True),
        StructField('name', StringType(), True),
        StructField('title', StringType(), True),
        StructField('blurb', StringType(), True),
        StructField('tags', StringType(), True),
        StructField('partype', StringType(), True),
        StructField('info.attack', IntegerType(), True),
        StructField('info.defense', IntegerType(), True),
        StructField('info.magic', IntegerType(), True),
        StructField('info.difficulty', IntegerType(), True),
        StructField('image.full', StringType(), True),
        StructField('image.sprite', StringType(), True),
        StructField('image.group', StringType(), True),
        StructField('image.x', IntegerType(), True),
        StructField('image.y', IntegerType(), True),
        StructField('image.w', IntegerType(), True),
        StructField('image.h', IntegerType(), True),
        StructField('stats.hp', DoubleType(), True),
        StructField('stats.hpperlevel', IntegerType(), True),
        StructField('stats.mp', DoubleType(), True),
        StructField('stats.mpperlevel', DoubleType(), True),
        StructField('stats.movespeed', IntegerType(), True),
        StructField('stats.armor', DoubleType(), True),
        StructField('stats.armorperlevel', DoubleType(), True),
        StructField('stats.spellblock', DoubleType(), True),
        StructField('stats.spellblockperlevel', DoubleType(), True),
        StructField('stats.attackrange', IntegerType(), True),
        StructField('stats.hpregen', DoubleType(), True),
        StructField('stats.hpregenperlevel', DoubleType(), True),
        StructField('stats.mpregen', DoubleType(), True),
        StructField('stats.mpregenperlevel', DoubleType(), True),
        StructField('stats.crit', IntegerType(), True),
        StructField('stats.critperlevel', IntegerType(), True),
        StructField('stats.attackdamage', DoubleType(), True),
        StructField('stats.attackdamageperlevel', DoubleType(), True),
        StructField('stats.attackspeedperlevel', DoubleType(), True),
        StructField('stats.attackspeed', DoubleType(), True),
    ]
)
# read match data
match_data = spark.read.csv("../data/match_data_version1.csv",
                    header='true',
                    schema=match_schema)

# https://www.kaggle.com/tk0802kim/kerneld01a1ec7ad
itens = spark.read.csv("../data/riot_item.csv",
                    header='true',
                    schema=itens_schema)

champions = spark.read.csv("../data/riot_champion.csv",
                    header='true',
                    schema=champions_schema)

from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F

# Convenience function for turning JSON strings into DataFrames.
# https://docs.databricks.com/_static/notebooks/transform-complex-data-types-scala.html
def jsonToDataFrame(json_input, schema=None):
    # SparkSessions are available with Spark 2.0+
    reader = spark.read
    if schema:
        reader.schema(schema)
    return reader.json(sc.parallelize([json_input]))


# Convenience function flatten dataframes with structs.
#https://stackoverflow.com/questions/38753898/how-to-flatten-a-struct-in-a-spark-dataframe
def flatten_df(nested_df):
    stack = [((), nested_df)]
    columns = []

    while len(stack) > 0:
        parents, df = stack.pop()

        flat_cols = [
            col(".".join(parents + (c[0],))).alias("_".join(parents + (c[0],)))
            for c in df.dtypes
            if c[1][:6] != "struct"
        ]

        nested_cols = [
            c[0]
            for c in df.dtypes
            if c[1][:6] == "struct"
        ]

        columns.extend(flat_cols)

        for nested_col in nested_cols:
            projected_df = df.select(nested_col + ".*")
            stack.append((parents + (nested_col,), projected_df))

    return nested_df.select(columns)

# Convenience function transform dict array string into rows
def transform_colum(df, column):
    df_select = df.select(col(column))
    str_ = df_select.take(1)[0].asDict()[column]
    df_select = jsonToDataFrame(json.dumps(eval(str_)))
    schema = df_select.schema
    
    eval_column = udf(lambda x : eval(x), ArrayType(schema))

    df = df.withColumn(column, eval_column(col(column)))
    
    return df, schema
# Rename columns with '.'
match_data = match_data.withColumnRenamed("status.message", "status_message")
match_data = match_data.withColumnRenamed("status.status_code", "status_status_code")
# transform string rows into psypsark rows
match_data, schema_partifipants = transform_colum(match_data, "participants")
match_data, schema_identities = transform_colum(match_data, "participantIdentities")
# new schema
match_data.printSchema()
# here we have to array columns. Before flatten our dataset we need first concatanate this arrays into a single array, and them explode their rows. 
combine = udf(lambda x, y: list(zip(x, y)),ArrayType(StructType([StructField("ids", schema_partifipants),
                                    StructField("info", schema_identities)]))
             )
match_data = match_data.withColumn("participants_info", combine("participants", "participantIdentities"))

# remove the old columns
columns_to_drop = ['participants', 'participantIdentities']
match_data = match_data.drop(*columns_to_drop)
match_data = match_data.withColumn("participants_info", explode("participants_info"))

match_data.printSchema()
# flatten structs
match_data=flatten_df(match_data)
match_data.printSchema()
# get the dictionary with itens names and keys
itens_dict = itens.select("item_id", "name").distinct().collect()
itens_dict = {v["item_id"]:v["name"] for v in itens_dict}

# help function to translate a item key into a item name
def transform_itens(x):
    try:
        value = itens_dict[int(x)] 
    except:
        value = "Name Not Found"
    return value


new_cols_itens = udf(lambda x : transform_itens(x), StringType())

# apply the translate function for each item column
match_data = match_data.withColumn("name_item0", new_cols_itens(col("participants_info_ids_stats_item0")))
match_data = match_data.withColumn("name_item1", new_cols_itens(col("participants_info_ids_stats_item1")))
match_data = match_data.withColumn("name_item2", new_cols_itens(col("participants_info_ids_stats_item2")))
match_data = match_data.withColumn("name_item3", new_cols_itens(col("participants_info_ids_stats_item3")))
match_data = match_data.withColumn("name_item4", new_cols_itens(col("participants_info_ids_stats_item4")))
match_data = match_data.withColumn("name_item5", new_cols_itens(col("participants_info_ids_stats_item5")))
match_data = match_data.withColumn("name_item6", new_cols_itens(col("participants_info_ids_stats_item6")))
# get the dictionary with champions names and keys

champions_dict = champions.select("key", "name").distinct().collect()
champions_dict = {v["key"]:v["name"] for v in champions_dict}

# help function to translate a champion key into a champion name
def transform_champions(x):
    try:
        value = champions_dict[int(x)] 
    except:
        value = "Name Not Found"
    return value


new_cols_champions = udf(lambda x : transform_champions(x), StringType())

# apply the translate function for champion column
match_data = match_data.withColumn("name_champion", new_cols_champions(col("participants_info_ids_championId")))
# Register the DataFrame as a SQL temporary view
match_data.createOrReplaceTempView("match_data")

# SQL querrie to extract the victory stats for each  champion
champions = sqlContext.sql("""
                              SELECT victorys.name_champion as name_champion, victorys.won_matches, matches.total_matches, victorys.won_matches/matches.total_matches as win_rate \
                              FROM \
                                  (SELECT match_data.name_champion as name_champion, COUNT(DISTINCT(match_data.gameId)) as won_matches \
                                  FROM match_data \
                                  WHERE match_data.participants_info_ids_stats_win == true \
                                  GROUP BY match_data.name_champion) as victorys \
                              LEFT JOIN (SELECT match_data.name_champion as name_champion, COUNT(DISTINCT(match_data.gameId)) as total_matches \
                                         FROM match_data \
                                         GROUP BY match_data.name_champion) as matches \
                              ON victorys.name_champion = matches.name_champion
                              ORDER BY matches.total_matches DESC
                          """) 
champions.createOrReplaceTempView("champions")
champions.show()
# SQL querrie to extract the victory stats for each  player with an especific champion
players = sqlContext.sql("""
                            SELECT victorys.id as user, victorys.name_champion as name_champion, victorys.won_matches, matches.total_matches, victorys.won_matches/matches.total_matches as win_rate \
                            FROM \
                                (SELECT match_data.participants_info_info_player_accountId as id, match_data.name_champion as name_champion, COUNT(DISTINCT(match_data.gameId)) as won_matches \
                                FROM match_data \
                                WHERE match_data.participants_info_ids_stats_win == true \
                                GROUP BY match_data.participants_info_info_player_accountId, match_data.name_champion) as victorys \
                            LEFT JOIN (SELECT match_data.participants_info_info_player_accountId as id, match_data.name_champion as name_champion, COUNT(DISTINCT(match_data.gameId)) as total_matches \
                                       FROM match_data \
                                       GROUP BY match_data.participants_info_info_player_accountId, match_data.name_champion) as matches \
                            ON victorys.id=matches.id AND victorys.name_champion = matches.name_champion
                            ORDER BY matches.total_matches DESC
                        """) 
players.createOrReplaceTempView("players")
players.show()
# SQL querrie to extract the most common first iten for each champion

connector = "-"
build_first_item = sqlContext.sql("""
                                     SELECT build.championName, build.build_name as first_item, COUNT(build.build_name) as total_matches \
                                     FROM \
                                         (SELECT match_data.name_champion as championName, match_data.name_item0  as build_name \
                                         FROM match_data \
                                         WHERE match_data.participants_info_info_player_accountId \
                                         IN ( \
                                              SELECT players.user
                                              FROM players \
                                              WHERE players.win_rate > 0.5 AND players.total_matches > 2)) as build \
                                     GROUP BY build.championName, build.build_name \
                                     ORDER BY total_matches DESC
                                 """)
build_first_item = build_first_item.dropDuplicates((['championName'])).sort((['championName']))
build_first_item.show()
# SQL querrie to extract the most common full build for each champion

connector = "-"
build = sqlContext.sql("""
                          SELECT build.championName, build.build_name, COUNT(build.build_name) as total_matches \
                          FROM \
                              (SELECT match_data.name_champion as championName, CONCAT(match_data.name_item0, "%s",
                                             match_data.name_item1, "%s",
                                             match_data.name_item2, "%s",
                                             match_data.name_item3, "%s",
                                             match_data.name_item4, "%s",
                                             match_data.name_item5, "%s",
                                             match_data.name_item6) as build_name \
                              FROM match_data \
                              WHERE match_data.participants_info_info_player_accountId \
                              IN ( \
                                   SELECT players.user
                                   FROM players \
                                   WHERE players.win_rate > 0.5 AND players.total_matches > 2)) as build \
                         GROUP BY build.championName, build.build_name \
                         ORDER BY total_matches DESC
                    """ % tuple([connector]*6))
build = build.dropDuplicates((['championName'])).sort((['championName']))
build.show()
players.write.parquet("../players.parquet")
champions.write.parquet("../champions.parquet")
build_first_item.write.parquet("../build_first_item.parquet")
build.write.parquet("../build.parquet")