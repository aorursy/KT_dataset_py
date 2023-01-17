import numpy as np

import pandas as pd

import matplotlib.pyplot as plot

import seaborn as sns





from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
# Can't seem to load the events it exceeds max kernel memory :(

#events = pd.read_csv('../input/events.csv', dtype={'uuid': np.str, 'display_id': np.str, 'document_id': np.str, 'geo_location': np.str})
documents_cat = pd.read_csv('../input/documents_categories.csv').rename(columns={'confidence_level': 'category_confidence_level'})

documents_cat
documents_ent = pd.read_csv('../input/documents_entities.csv').rename(columns={'confidence_level': 'entity_confidence_level'})

documents_ent
documents_meta = pd.read_csv('../input/documents_meta.csv', parse_dates=['publish_time'], dtype={'source_id': np.str, 'publisher_id': np.str, 'document_id': np.str})

documents_meta
document_topics = pd.read_csv('../input/documents_topics.csv', dtype={'document_id': np.str, 'topic_id': np.str}).rename(columns={'confidence_level': 'topic_confidence_level'})

document_topics
promoted_content = pd.read_csv('../input/promoted_content.csv', dtype={'ad_id': np.str, 'document_id': np.str, 'campaign_id': np.str, 'advertiser_id': np.str})

promoted_content
promoted_joined = pd.merge(promoted_content, document_topics, how='outer', on=['document_id'])
promoted_joined
promoted_joined = pd.merge(promoted_joined, documents_meta, how='outer', on=['document_id'])
promoted_joined