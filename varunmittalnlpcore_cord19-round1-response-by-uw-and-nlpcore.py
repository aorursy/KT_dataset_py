import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests # process http requests

from time import sleep # timer functions

import json # process json objects

from tqdm import tqdm # progress bar

from hashlib import md5 # md5 hash for caching

import networkx as nx # generate d3 compatible graph

import IPython.display

from IPython.core.display import display, HTML, Javascript

from string import Template

import networkx as nx

from difflib import SequenceMatcher

from os import path
# ************************************** FUNCTION DEFINITIONS ***********************************************

"""

User defined Topics that forces search engine to look in their neighborhood also

"""

select_topics = set(['ACTIVITY', 'ADE', 'AGENT', 'ANIMAL', 'ANIMALS', 'ANTAGONIST', 'ANTIVIRAL', 'ASYMPTOMATIC',

                     'BAT', 'BINDING', 'BUFFER', 'CELL', 'CELLS', 'CIRCULATION', 'CLARITHROMYCIN', 'CO-INFECTIONS',

                     'CO-MORBIDITIES', 'DISEASE', 'DRUG', 'DRUGS', 'ENVIRONMENT', 'ENZYME', 'ENZYMES', 'EXPERIMENTAL',

                     'FARMERS', 'GENOME', 'HIGH-RISK', 'HISTONE', 'HOST', 'HYDROPHILIC', 'HYDROPHOBIC', 'INFECTION',

                     'INTERACTIONS','IMMUNE', 'LIGAND', 'LIVESTOCK', 'MINOCYCLINE', 'MODEL', 'NAGOYA', 'NAPROXEN',

                     'NEONATES', 'NUCLEOTIDE', 'PATIENT', 'PATHOGENESIS', 'PEPTIDE', 'PEPTIDES', 'PHENOTYPE', 'PLATES',

                     'POLYPROTEIN', 'PPE', 'PRE-EXISTING', 'PREGNANCY', 'PROTEIN', 'PROTOCOL', 'PROPHYLAXIS',

                     'PULMONARY', 'RBD', 'RANGE', 'REAGENT', 'REAGENTS', 'RECEPTER', 'REPLICATION', 'RESIDUES', 'RESPONSE'

                     'SEQUENCING', 'SHEDDING', 'SMOKING', 'STRAIN', 'STRUCTURES', 'THERAPEUTIC', 'TRACKING',

                     'TRANSCRIBE', 'TRANSCRIPTASE', 'TRANSMISSION', 'TREATMENT', 'VACCINE', 'VIRAL', 'VIRUS',

                     'WILDLIFE', 'UNIVERSAL'])



"""

Extract concepts and topics and their relationship from the search engine including user-defined topics

"""

def get_graph(project_name="cord19-dataset", source="coronavirus", target="transmission", auth="test-key"):



    params = {

        'auth': auth,

        'u_name': source,

        "v_name": target,

        "return_dataframe": True,

        "return_type": "dataframe",        

        "additional_topics": ",".join(map(lambda word: word.lower(), select_topics)),

        "project_name": project_name

    }

    r = requests.get("https://apis.nlpcore.com/apis/get_graph/", params=params)

    if r.status_code != 200:

        raise RuntimeError("Failed to get_graph, please try again., %s" % r.content)

    dataframe = pd.DataFrame(json.loads(r.content))

    return dataframe



"""

Filter rows in a dataframe to specific topics

"""

def subset_dataframe(dataframe, given_topics):



    select_dataframe_rows = []

    for _,row in dataframe.iterrows():

        source_topics = set(row['source_topics'])

        target_topics = set(row['target_topics'])

        if (given_topics & source_topics or given_topics & target_topics):

            select_dataframe_rows.append(row)

    return pd.DataFrame(select_dataframe_rows)



"""

Get Article metadata/attributes for a given document id, store results in caches for repeated calls

"""

def document_metadata(project_name, document_id, auth="test-key"):

    

    cache_key_str = "%s-%s" % (project_name, document_id)

    cache_key = md5(cache_key_str.encode()).hexdigest()

    cache_path = "/tmp/metadata2-%s.json" % cache_key



    try:

        return json.load(open(cache_path))

    except FileNotFoundError:

        pass



    r = requests.get("https://apis.nlpcore.com/apis/get_document_metadata/", params={'project_name': project_name,

                                                                            'auth': auth,

                                                                            'd': document_id})

    if r.status_code == 200:

        reference_data = r.json()

        json.dump(reference_data, open(cache_path, "w"))



    return r.json()



"""

Dataframe returned from the above calls has a list of concepts and their references. For each reference we can request 

text segments. The parameter "r" is a comma seperated list of integers which are senetence numbers.

Cache references for repeated calls.

"""

def get_references(project_name, document_id, r, auth="test-key"):

    

    cache_key_str = "%s-%s-%s" % (project_name, r, document_id)

    cache_key = md5(cache_key_str.encode()).hexdigest()

    cache_path = "/tmp/%s.json" % cache_key

    

    try:

        return json.load(open(cache_path))

    except FileNotFoundError:

        pass

    

    r = requests.get("https://apis.nlpcore.com/apis/get_references/", params={'project_name': project_name,

                                                                             'auth': auth, 'r': r,

                                                                             'd': document_id})

    if r.status_code == 200:

        reference_data = r.json()

        json.dump(reference_data, open(cache_path, "w"))

    

    return r.json()



"""

Augment the dataframe with article and sentence references for each of the co-occuring concepts in each row

"""

def refine_dataframe(project_name, dataframe, auth="test-key"):

    select_dataframe_rows = []

    for _,row in tqdm(list(dataframe.iterrows())):

        source_topics = set(row['source_topics'])

        target_topics = set(row['target_topics'])

        if (select_topics & source_topics and select_topics & target_topics) and row['source_idf'] < 3 and row['target_idf'] < 3:            

            reference_texts = []

            for document_id,references in row['references'].items():

                title = document_metadata(project_name=project_name, document_id=document_id)['title'] or "<No Title>"

                sections = {}

                for reference in references[:]:

                    r = "%d,%d" % (reference['u_curr_ref'], reference['v_curr_ref'])

                    text = get_references(project_name=project_name, document_id=document_id, r=r)

                    for section in text.values():

                        try:

                            section_title = section['section_title']

                        except Exception as e:

                            raise e

                        try:

                            bucket = sections[section_title]

                        except KeyError:

                            bucket = []

                            sections[section_title] = bucket                        

                        bucket.append(section['sentence'])

                reference_texts.append({'title': title, 'sections': sections})                    

            select_dataframe_rows.append({'source': row['u_name'], 'target': row['v_name'], 'source_types': ", ".join(source_topics),

                                         'target_types': ", ".join(target_topics), 'count': row['count'],

                                         'references': reference_texts})

    return pd.DataFrame(select_dataframe_rows)



"""

Augment the dataframe with select sentences that match keywords from task

"""

def search_task_words(dataframe, given_topics):

    

    select_dataframe_rows = []

    given_topics = [word.lower() for word in given_topics] 

    for _,row in dataframe.iterrows():

        matched_sentences = []

        matched_words = []

        for reference_obj in row['references']:

            for section_title, sentences in reference_obj['sections'].items():

                for sentence in sentences:

                    matched = [word for word in given_topics if word in sentence.lower()]

                    if matched:

                        matched_sentences.append(sentence)

        row['sentences'] = matched_sentences

        select_dataframe_rows.append(row.to_dict())

    return pd.DataFrame(select_dataframe_rows)        



def convert_df(dataframe):

    def similar(a, b):

        return SequenceMatcher(None, a, b).ratio()



    g = nx.DiGraph()

    groups = {}



    def get_group(group_name):

        try:

            group_id = groups[group_name]

        except KeyError:

            group_id = len(groups) + 1

            groups[group_name] = group_id

        return group_id



    def add_node(concept_name, group_name):

        concept_name = concept_name.lower()

        

        if concept_name in g.nodes:

            g.nodes[concept_name]['size'] += 1

        else:

            sim_scores = [(_node, similar(_node, concept_name)) for _node in g.nodes]

            if len(sim_scores) > 0:

                _node, score = max(sim_scores, key=lambda item: item[1])

                if score > 0.7:

                    return add_node(_node, g.nodes[_node]['group'])

            g.add_node(concept_name, size=1, group=get_group(group_name))

        return concept_name



    for _, row in dataframe.iterrows():

        source_id = add_node(row['source'], row['source_types'])

        target_id = add_node(row['target'], row['target_types'])

        edge = g.get_edge_data(source_id, target_id)

        if edge:

            edge['value'] += 1

        else:

            g.add_edge(source_id, target_id, value=1)



    dataframe_rows = []

    for node_id in g.nodes:

        node = g.nodes[node_id]

        name = "project.%d.%s" % (node['group'], node_id)

        dataframe_rows.append({'id': name, 'value': node['size'], 'value1': node['size']})

    dataframe_rows = sorted(dataframe_rows, key=lambda item: item['value'], reverse=True)[:100]

    return pd.DataFrame(dataframe_rows)



def return_bubble_data(csv_file_path, html_element_id):

    html = """<!DOCTYPE html><svg id="%s" width="760" height="760" font-family="sans-serif" font-size="10" text-anchor="middle"></svg>""" % html_element_id

    js = """require.config({paths: {d3: "https://d3js.org/d3.v4.min"}});require(["d3"], function(d3) {var svg=d3.select("#%s"),width=+svg.attr("width"),height=+svg.attr("height"),format=d3.format(",d"),color=d3.scaleOrdinal(d3.schemeCategory20c);console.log(color);var pack=d3.pack().size([width,height]).padding(1.5);d3.csv("%s",function(t){if(t.value=+t.value,t.value)return t},function(t,e){if(t)throw t;var n=d3.hierarchy({children:e}).sum(function(t){return t.value}).each(function(t){if(e=t.data.id){var e,n=e.lastIndexOf(".");t.id=e,t.package=e.slice(0,n),t.class=e.slice(n+1)}}),a=(d3.select("body").append("div").style("position","absolute").style("z-index","10").style("visibility","hidden").text("a"),svg.selectAll(".node").data(pack(n).leaves()).enter().append("g").attr("class","node").attr("transform",function(t){return"translate("+t.x+","+t.y+")"}));a.append("circle").attr("id",function(t){return t.id}).attr("r",function(t){return t.r}).style("fill",function(t){return color(t.package)}),a.append("clipPath").attr("id",function(t){return"clip-"+t.id}).append("use").attr("xlink:href",function(t){return"#"+t.id}),a.append("svg:title").text(function(t){return t.value}),a.append("text").attr("clip-path",function(t){return"url(#clip-"+t.id+")"}).selectAll("tspan").data(function(t){return t.class.split(/(?=[A-Z][^A-Z])/g)}).enter().append("tspan").attr("x",0).attr("y",function(t,e,n){return 13+10*(e-n.length/2-.5)}).text(function(t){return t})});});""" % (html_element_id, csv_file_path)

    return html, js



# ************************************** END OF FUNCTION DEFINITIONS ***********************************************
"""

Filter the results to only focus on most relvant topics for this challenge

"""

task_topics = set(['ANIMAL', 'ANIMALS', 'ASYMPTOMATIC', 'MODEL', 'TRANSMISSION', 'INCUBATION', 

                   'SHEDDING' 'HYDROPHILIC', 'HYDROPHOBIC', 'VIRUS', 'DISEASE', 'PHENOTYPE',

                   'PPE'])

project_name="cord19-dataset"

source="coronavirus"

target="transmission"

auth="test-key"
"""

Get the initial dataframe and filter it down to topics of interest and add article references

"""

if path.isfile("/kaggle/input/nlpcore-cord19-output/task_1.csv"):

    task_df = pd.read_csv("/kaggle/input/nlpcore-cord19-output/task_1.csv", index_col=0)

else:

    df = get_graph(project_name, source, target, auth)

    task_df = refine_dataframe(project_name, subset_dataframe(df, task_topics), auth)

    task_df = search_task_words(task_df, task_topics)

    task_df.to_csv("/kaggle/working/task_1.csv")



# Print results



graph_data = convert_df(task_df)
task_df
"""

Filter the results to only focus on most relvant topics for this challenge

"""

task_topics = set(['SMOKING', 'PRE-EXISTING', 'PULMONARY', 'DISEASE', 'CO-INFECTIONS', 'CO-MORBIDITIES', 'NEONATES', 'PREGNANCY', 

                   'HIGH-RISK', 'PATIENT'])

project_name="cord19-dataset"

source="coronavirus"

target="disease"

auth="test-key"
"""

Get the initial dataframe and filter it down to topics of interest and add article references

"""

if path.isfile("/kaggle/input/nlpcore-cord19-output/task_2.csv"):

    task_df = pd.read_csv("/kaggle/input/nlpcore-cord19-output/task_2.csv", index_col=0)

else:

    df = get_graph(project_name, source, target, auth)

    task_df = refine_dataframe(project_name, subset_dataframe(df, task_topics), auth)

    task_df = search_task_words(task_df, task_topics)

    task_df.to_csv("/kaggle/working/task_2.csv")



task_df.to_csv("/kaggle/working/task_2.csv")

graph_data = convert_df(task_df)
graph_data.to_csv("task_2_graph.csv")

html, js = return_bubble_data("task_2_graph.csv", "graph_2_csv")



h = display(HTML(html))

j = IPython.display.Javascript(js)

IPython.display.display_javascript(j)
task_df
"""

Filter the results to only focus on most relvant topics for this challenge

"""

task_topics = set(['GENOME', 'TRACKING', 'STRAIN', 'CIRCULATION', 'NAGOYA', 'LIVESTOCK', 'RECEPTER', 'BINDING', 

                   'FARMERS' 'WILDLIFE', 'HOST', 'RANGE', 'EXPERIMENTAL', 'INFECTION', 'ANIMAL', 'PROTOCOL'

                   'HOST'])

project_name="cord19-dataset"

source="coronavirus"

target="strain"

auth="test-key"
"""

Get the initial dataframe and filter it down to topics of interest and add article references

"""

if path.isfile("/kaggle/input/nlpcore-cord19-output/task_3.csv"):

    task_df = pd.read_csv("/kaggle/input/nlpcore-cord19-output/task_1.csv", index_col=0)

else:

    df = get_graph(project_name, source, target, auth)

    task_df = refine_dataframe(project_name, subset_dataframe(df, task_topics), auth)

    task_df = search_task_words(task_df, task_topics)

    task_df.to_csv("/kaggle/working/task_3.csv")



task_df.to_csv("/kaggle/working/task_3.csv")
task_df
"""

Filter the results to only focus on most relvant topics for this challenge

"""

task_topics = set(['NAPROXEN', 'CLARITHROMYCIN', 'MINOCYCLINE', 'ADE', 'THERAPEUTIC', 'ANTIVIRAL', 'AGENT', 'UNIVERSAL'

                  'VACCINE', 'PROPHYLAXIS', 'IMMUNE', 'RESPONSE'])

project_name="cord19-dataset"

source="coronavirus"

target="vaccine"

auth="test-key"
"""

Get the initial dataframe and filter it down to topics of interest and add article references

"""

if path.isfile("/kaggle/input/nlpcore-cord19-output/task_4.csv"):

    task_df = pd.read_csv("/kaggle/input/nlpcore-cord19-output/task_4.csv", index_col=0)

else:

    df = get_graph(project_name, source, target, auth)

    task_df = refine_dataframe(project_name, subset_dataframe(df, task_topics), auth)

    task_df = search_task_words(task_df, task_topics)

    task_df.to_csv("/kaggle/working/task_4.csv")



task_df.to_csv("/kaggle/working/task_4.csv")
task_df