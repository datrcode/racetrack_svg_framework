import pandas as pd
import polars as pl
import numpy as np
import networkx as nx
import json
import time
import os
import sys
sys.path.insert(1, '../../rtsvg')
from rtsvg import *
from rt_ontologies_mixin import jsonAbsolutePath, fillJSONPathElements
rt = RACETrack()

# Read the original json file and make a graph representation of the json structure
print('Reading JSON File... ')
_all_ = open('../../../data/kaggle_arXiv/arxiv-metadata-oai-snapshot.json', encoding='utf-8').read()
print('Separating into lines...')
_jsons_ = []
for _line_ in _all_.split('\n'):
    if len(_line_) == 0: continue
    _jsons_.append(json.loads(_line_))
    # if len(_jsons_) > 1000: break # short circuit while developing...
print(f'{len(_jsons_)=}')
print('Analyzing json structure...')
_json_repr_ = rt.jsonRepr(_jsons_)
print('Making graph representation...')
df, relates, labels = _json_repr_.starPathGraphDataFrame()

xform_spec = """
__id__          = '$.id'         | PaperID   | uniq
__journalref__  = '$.journal-ref'| JournalID | uniq
__fullname__    = fullName('$.authors_parsed[*][0]', '$.authors_parsed[*][1]', '$.authors_parsed[*][2]')
__versionNode__ = versionNode('$.id', '$.versions[*].version') | xsd:string | uniq
__id__          --- hasTitle          --- stripString('$.title')    | xsd:string   | uniq @@@ __journalref__ ^^^ 'kaggle_arXiv'
__id__          --- hasAbstract       --- stripString('$.abstract') | xsd:string   | cont @@@ __journalref__ ^^^ 'kaggle_arXiv'
__id__          --- hasSubmitter      --- '$.submitter'             | xsd:string   | ambi @@@ __journalref__ ^^^ 'kaggle_arXiv'
__id__          --- hasDOI            --- '$.doi'                   | xsd:string   | uniq @@@ __journalref__ ^^^ 'kaggle_arXiv'
__id__          --- hasLicense        --- '$.license'               | xsd:string   | uniq @@@ __journalref__ ^^^ 'kaggle_arXiv'
__id__          --- updateDate        --- '$.update_date'           | xsd:date     | date @@@ __journalref__ ^^^ 'kaggle_arXiv'
__id__          --- hasAuthors        --- '$.authors'               | xsd:string   | ambi @@@ __journalref__ ^^^ 'kaggle_arXiv'
__id__          --- hasAuthor         --- __fullname__              | xsd:string   | ambi @@@ __journalref__ ^^^ 'kaggle_arXiv'
__id__          --- hasVersion        --- __versionNode__                                 @@@ __journalref__ ^^^ 'kaggle_arXiv'
__versionNode__ --- hasTimestamp      --- '$.versions[*].created'   | xsd:dateTime | dttm @@@ __journalref__ ^^^ 'kaggle_arXiv'
"""
def fullName(last, first, middle) -> str: return f'{last} {first} {middle}'
def versionNode(id, version) -> str: return f'{id}|{version}'
def stripString(s) -> str: 
    s = s.strip()
    s = s.replace('\\n', ' ')
    return ' '.join(s.split())
ofv = rt.ontologyFrameworkInstance(xform_spec=xform_spec, funcs={'fullName': fullName, 'stripString': stripString, 'versionNode': versionNode})
ofv.parse(_jsons_)
print(len(ofv.df_triples))

print('Writing to disk...')
ofv.to_files('../../../data/kaggle_arXiv/20240609_ontology')
