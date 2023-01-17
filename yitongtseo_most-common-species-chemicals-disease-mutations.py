#! /bin/env python3

import json
import os
import pprint
from dataclasses import dataclass
from typing import List
from enum import Enum


class EntityType(Enum):
    CELL_LINE = 'CellLine'
    MUTATION = 'Mutation'
    SPECIES = 'Species'
    GENUS = 'Genus'
    STRAIN = 'Strain'
    GENE = 'Gene'
    DOMAIN_MOTIF = 'DomainMotif'
    CHEMICAL = 'Chemical'
    DISEASE = 'Disease'

class SectionType(Enum):
    TITLE = 'title'
    ABSTRACT = 'abstract'
    INTRODUCTION = 'introduction'
    METHODS = 'methods'
    CONCLUSION = 'conclusion'
    RESULTS = 'results'
    DISCUSSION = 'discussion'
    REFERENCES = 'references'
    OTHER = 'other'

@dataclass
class Location:
    offset: int
    length: int

@dataclass
class Entity:
    type: EntityType
    text: str
    location: List[Location]
    source: SectionType

@dataclass
class Section:
    text: str
    section_type: SectionType

@dataclass
class Paper:
    id: str
    _id: str
    entities: List[Entity]
    sections: List[Section]
    

def resolveSectionType(section_str: str) -> SectionType:
    for st in SectionType:
        if section_str.lower().find(st.value):
            return st
    return SectionType.OTHER

        
paper_limit = 3000#float("inf")
current_index = 0

processed_papers:List[Paper] = []
for dirname, _, filenames in os.walk('/kaggle/input/cord19-named-entities/entities/pmcid'):
    for filename in filenames:
        current_index += 1
        if current_index > paper_limit:
            break
        with open(os.path.join(dirname, filename), 'r') as f:
            data = json.load(f)
            title = abstract = None
            entities = []
            sections = []
            for passages in data['passages']:
                section_type = resolveSectionType(passages['infons']['section'])
                sections.append(
                    Section(
                        section_type,
                        passages['text']
                    )
                )
                for extracted_entity in passages['annotations']:
                    entities.append(
                        Entity(
                            EntityType(extracted_entity['infons']['type']),
                            extracted_entity['text'],
                            [Location(l['offset'], l['length']) for l in extracted_entity['locations']],
                            section_type,
                        )
                    )
            processed_papers.append(
                Paper(
                    data['id'],
                    data['_id'],
                    entities,
                    sections,
                )
            )

print('--data loaded--')
print('# of papers: ', len(processed_papers))
entity_number = sum([
    len(paper.entities)
    for paper in processed_papers
])
print('# of entities: ', entity_number)
import csv

interested_entities = [
    ('species', [EntityType.SPECIES]),
    ('chemicals', [EntityType.CHEMICAL]),
    ('diseases_strains', [EntityType.DISEASE, EntityType.STRAIN]),
    (
        'mutations_genes_cellLines',
        [EntityType.MUTATION, EntityType.GENE, EntityType.CELL_LINE]
    ),
]

for entity_names, entity_types in interested_entities:
    entities_by_paper = [
        set([(e.text, e.type) for e in entities])
        for entities in [
            paper.entities
            for paper in processed_papers
        ]
    ]
    entities = [
        entity[0]
        for entity_set in entities_by_paper 
        for entity in entity_set
        if entity[1] in entity_types
    ]
    counts = [
        (entity, entities.count(entity))
        for entity in set(entities)
    ]
    # Take the top 1K
    most_common = sorted(
        counts, 
        key=lambda s: s[1], 
        reverse=True
    )[:1000]
    
    with open(entity_names + '.tsv', 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(["Type", "Count"])
        for row in most_common:
            writer.writerow(row)

    print('Total of', len(entities), 'tallied  for', entity_names)