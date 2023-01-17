from google.cloud import bigquery

from IPython import display

import jinja2



PROJECT_ID = "jefferson-1790"

client = bigquery.Client(PROJECT_ID)
# List datasets

projects = ["patents-public-data", "ncbi-research-pubchem", "erudite-marker-539", "innography-174118", "nlp-parsing", "open-targets-genetics", "sciwalker-open-data"]
# TODO: move this into a kaggle script so multiple documentation kernels can use it

for project in projects:

    for dataset in client.list_datasets(project):

        dataset_id = "%s.%s" % (project, dataset.dataset_id)

        dd = client.get_dataset(dataset_id)

        print("-"*80)

        print(dataset_id)

        print(dd.description)

        # List tables

        print("\nTables:")

        for table in client.list_tables(dataset_id):

            table_id = "%s.%s" % (dataset_id, table.table_id)

            td = client.get_table(table_id)

            print(table_id)

            def print_field(field, indent):

                print(("\t" * indent) + "%s: %s" % (field.name, field.field_type))

                for sf in field.fields:

                    print_field(sf, indent+1)

            for field in td.schema:

                print_field(field, 2)



# <dataset, table, description>



# <dataset, table, field, tags>



# Highlight join columns