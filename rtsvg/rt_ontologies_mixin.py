# Copyright 2024 David Trimm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pandas as pd
import polars as pl
import numpy as np
import json
import time
import os

__name__ = 'rt_ontologies_mixin'

#
# Ontologies Mixin
#
class RTOntologiesMixin(object):
    #
    # Constructor
    #
    def __ontologies_mixin_init__(self):
        pass

    #
    # Ontology Framework
    # - "base_filename" -- forces a load of the specified base path
    #
    def ontologyFrameworkInstance(self, **kwargs):
        if    'xform_spec'    in kwargs: return RTOntology(self, **kwargs)
        elif  'base_filename' in kwargs: 
            rt_ontology = RTOntology(self)
            rt_ontology.fm_files(kwargs['base_filename'])
            return rt_ontology
        else: return RTOntology(self)
    
# scanForward() - finds the next unescaped version of character c in x starting at i
def scanForward(x, i, c):
    in_escape = False
    while i < len(x):
        if   x[i] == '\\' and in_escape == False: in_escape = True
        else:
            if x[i] == c and in_escape == False: return i
            in_escape = False
        i += 1
    return None

# literalize() - converts any single or double quoted strings into unique literal names
# ... fails if inputs contain four underscore names that overlap with the format "____lit{num}____"
def literalize(x):
    l, lu = [], {}
    i = 0
    while i < len(x):
        c = x[i]
        if   c == "'":
            j = scanForward(x, i+1, "'")
            if j is None: raise Exception(f'RTOntology.literalize() - unterminated string literal "{x}"')
            _literal_name_ = f'____lit{len(lu.keys())}____' # Surely, no one would ever use four underscores in a literal... and don't call me Surely
            lu[_literal_name_] = x[i+1:j]
            l.append(_literal_name_)
            i = j + 1
        elif c == '"':
            j = scanForward(x, i+1, '"')
            if j is None: raise Exception(f'RTOntology.literalize() - unterminated string literal "{x}"')
            _literal_name_ = f'____lit{len(lu.keys())}____' # Surely, no one would ever use four underscores in a literal... and don't call me Surely
            lu[_literal_name_] = x[i+1:j]
            l.append(_literal_name_)
            i = j + 1
        else:
            l.append(c)
            i += 1
    return ''.join(l), lu

# fillLiteratals() - fill in the literal values (opposite of literalize but not guaranteed to keep spaces)
def fillLiterals(x, lu):
    for k, v in lu.items():
        x = x.replace(k, v)
    return x

# findClosingParen() - find the next closing paren taking other open/closes into consideration
# ... requires that literals were taken care of... [see literalize function]
def findClosingParen(s, i):
    stack = 0
    while i < len(s):
        if   s[i] == '(':               stack += 1
        elif s[i] == ')' and stack > 0: stack -= 1
        elif s[i] == ')':               return i
        i += 1
    raise Exception(f'RTOntology.findClosingParen() - no closing paren found for "{s}"')

# tokenizeParameters() - create a token list of function parameters
# ... requires that literals were taken care of... [see literalize function]
def tokenizeParameters(x):
    r = []
    while ',' in x or '(' in x:
        if   ',' in x and '(' in x: # both... process the one that occurs first
            i = x.index(',')
            j = x.index('(')
            if i < j:
                r.append(x[:i])
                x = x[i+1:]
            else:
                k = findClosingParen(x,j+1)
                r.append(x[:k+1].strip())
                x = x[k+1:]
                if ',' in x: x = x[x.index(',')+1:] # if there's another comma, consume it
        elif ',' in x: # just literals from here on out...
            r.append(x[:x.index(',')].strip())
            x = x[x.index(',')+1:]
        elif '(' in x: # just one function call from here on out...
            i = x.index('(')
            j = findClosingParen(x,i+1)
            r.append(x[:j+1].strip())
            x = x[j+1:]
            if ',' in x: x = x[x.index(',')+1:] # if there's another comma, consume it
    x = x.strip()
    if len(x) > 0:
        r.append(x)
    return r

# parseTree() - create a parse tree representation of a ontology node description
def parseTree(x, node_value=None, node_children=None, node_name=None, lit_lu=None):
    if node_value is None:
        node_value = {}
        node_children = {}
        node_name = 'root'

    if lit_lu is None:
        x, lit_lu = literalize(x)
    if '(' in x:
        i          = x.index('(')
        j          = findClosingParen(x, i+1)
        fname      = x[0:i].strip()
        parms      = tokenizeParameters(x[i+1:j])
        node_value   [node_name] = lit_lu[fname] if fname in lit_lu else fname
        node_children[node_name] = []    # functions have children... even if it's an empty list of children
        for child_i in range(len(parms)):
            child_name = f'{node_name}.{child_i}'
            node_children[node_name].append(child_name)
            parseTree(parms[child_i], node_value, node_children, child_name, lit_lu)
    else:
        x                        = x.strip()
        node_value   [node_name] = lit_lu[x] if x in lit_lu else x
        node_children[node_name] = None # literals have no children
    return node_value, node_children

# upToStar() - upto the cth '[*]'
def upToStar(x, c):
    i = 0
    while c > 0:
        j = x.index('[*]', i)
        i = j + 3
        c -= 1
    return x[:i]

# fillStars() - fill the the stars in the specified order
def fillStars(x, i, j=None, k=None):
    if '[*]' not in x: return x # for example ... "$.id"
    _index_ = x.index('[*]')
    x = x[:_index_] + f'[{i}]' + x[_index_+3:]
    if j is not None and '[*]' in x:
        _index_ = x.index('[*]')
        x = x[:_index_] + f'[{j}]' + x[_index_+3:]
    if k is not None and '[*]' in x:
        _index_ = x.index('[*]')
        x = x[:_index_] + f'[{k}]' + x[_index_+3:]
    return x

# isJsonPath() - check if the string is a jsonpath
def isJsonPath(_str_): 
    return _str_.startswith('$.') or _str_.startswith('$[')

#
# fillJSONPathElementsByJSONPath() - unoptimized version using jsonpath-ng
# - to_fill should only have values with a [*] in them
#
def fillJSONPathElementsByJSONPath(to_fill, myjson):
    longest_by_star_path, min_stars, max_stars = None, None, None
    filled = {}
    for x in to_fill:
        star_count = x.count('[*]')
        if min_stars is None or star_count < min_stars: min_stars = star_count
        if max_stars is None or star_count > max_stars: max_stars = star_count
        if longest_by_star_path is None or len(x) > len(longest_by_star_path): longest_by_star_path = x
        filled[x] = []

    # fill in the json values into the filled dict
    if min_stars == max_stars: # shortcut if we only do the same number of stars
        _length_to_fill_ = None
        for v in filled.keys():
            if isJsonPath(v) and '[*]' in v:
                filled[v] = [match.value if match.value != {} else None for match in parse(v).find_or_create(myjson)]
                _this_length_ = len(filled[v])
                if   _length_to_fill_ is None:          _length_to_fill_ = _this_length_
                elif _length_to_fill_ != _this_length_: raise Exception(f'RTOntology.__applyTemplate__() - unequal number of values for {v} ({_length_to_fill_=} vs {_this_length_=})')
        for v in filled.keys():
            if   isJsonPath(v) and '[*]' not in v:
                _match_ = parse(v).find(myjson)[0].value
                filled[v] = [_match_] * _length_to_fill_
            elif isJsonPath(v) == False:
                filled[v] = [v] * _length_to_fill_
    else:
        star_count = longest_by_star_path.count('[*]')
        if star_count   == 1:
            for i in range(len(parse(upToStar(longest_by_star_path, 1)).find(myjson))):
                for v in filled.keys():
                    if isJsonPath(v):
                        _matches_ = parse(fillStars(v, i)).find(myjson)
                        if len(_matches_) == 1: filled[v].append(_matches_[0].value)
                        else:                   filled[v].append(None)
                    else:
                        filled[v].append(v)
        elif star_count == 2:
            for i in range(len(parse(upToStar(longest_by_star_path, 1)).find(myjson))):
                for j in range(len(parse(upToStar(fillStars(longest_by_star_path,i), 1)).find(myjson))):
                    for v in filled.keys():
                        if isJsonPath(v):
                            _matches_ = parse(fillStars(v, i, j)).find(myjson)
                            if len(_matches_) == 1: filled[v].append(_matches_[0].value)
                            else:                   filled[v].append(None)
                        else:
                            filled[v].append(v)
        elif star_count == 3:
            for i in range(len(parse(upToStar(longest_by_star_path, 1)).find(myjson))):
                for j in range(len(parse(upToStar(fillStars(longest_by_star_path,i), 1)).find(myjson))):
                    for k in range(len(parse(upToStar(fillStars(longest_by_star_path,i,j), 1)).find(myjson))):
                        for v in filled.keys():
                            if isJsonPath(v):
                                _matches_ = parse(fillStars(v, i, j, k)).find(myjson)
                                if len(_matches_) == 1: filled[v].append(_matches_[0].value)
                                else:                   filled[v].append(None)
                            else:
                                filled[v].append(v)
        else:
            raise Exception(f'RTOntology.__applyTemplate__() - max of three stars supported -- {star_count} found')
    return filled

#
# isLiteral() - true if it's a proper literal for json key string
#
def isLiteral(v):
    if v == '': return False
    for i in range(len(v)):
        if i == 0 and v[i] >= '0' and v[i] <= '9': return False # can't start with a number
        if (v[i] >= 'a' and v[i] <= 'z') or (v[i] >= 'A' and v[i] <= 'Z') or (v[i] >= '0' and v[i] <= '9') or (v[i] == '_') or (v[i] == '-'): pass
        else: return False
    return True

#
# endsWithAny() - does a string end with any of these?
#
def endsWithAny(_str_, _set_):
    return any(_str_.endswith(x) for x in _set_)

#
# jsonAbsolutePath() - return the json value at the given path
# ... has to be an absolute path
#
def jsonAbsolutePath(_path_, _json_, _so_far_=None):
    if len(_path_) == 0:
        _to_eval_ = '_json_' + ''.join(_so_far_)
        try:    return eval(_to_eval_)
        except: return None
    if _so_far_ is None: _so_far_ = []
    if _path_.startswith('$'): _path_ = _path_[1:]
    if _path_.startswith('.'): _path_ = _path_[1:]
    if _path_.startswith('['):
        _index_ = _path_[1:_path_.index(']')]
        _so_far_.append(f'[{_index_}]')
        return jsonAbsolutePath(_path_[_path_.index(']')+1:], _json_, _so_far_)
    else:
        if   '.' in _path_ and '[' in _path_:
            dot_pos = _path_.index('.')
            brk_pos = _path_.index('[')
            if dot_pos < brk_pos: _var_ = _path_[:dot_pos]
            else:                 _var_ = _path_[:brk_pos]
        elif '.' in _path_: _var_ = _path_[:_path_.index('.')]
        elif '[' in _path_: _var_ = _path_[:_path_.index('[')]
        else:               _var_ = _path_
        _so_far_.append(f'["{_var_}"]')
        return jsonAbsolutePath(_path_[len(_var_):], _json_, _so_far_)

#
# fillJSONPathElements() - uses self modifying code to optimize the filling of the structures based on jsonpath specifications.
#
def fillJSONPathElements(to_fill, myjson):
    filled = {}
    for x in to_fill: filled[x] = [] 
    filled_list = list(filled.keys())
    longest_by_star_path  = filled_list[0]
    for i in range(1, len(filled_list)): 
        if longest_by_star_path.count('[*]') <  filled_list[i].count('[*]'): longest_by_star_path = filled_list[i]
        if longest_by_star_path.count('[*]') == filled_list[i].count('[*]') and \
           longest_by_star_path.count('.')   <  filled_list[i].count('.'):   longest_by_star_path = filled_list[i]

    to_eval, indent, _index_, _loop_vars_, _loop_i_, _path_, _star_path_, vars_set = [], 0, 1, ['i','j','k','l'], 0, '', '$', 0
    while _index_ < len(longest_by_star_path):
        _rest_ = longest_by_star_path[_index_:]
        if   _rest_.startswith('[*]'):
            to_eval.append(' '*indent+'for '+_loop_vars_[_loop_i_]+f' in range(len(myjson{_path_})):')
            _path_      += f'[{_loop_vars_[_loop_i_]}]'
            _star_path_ += f'[*]'
            _index_, _loop_i_, indent = _index_+3, _loop_i_+1, indent+4
            if _rest_.endswith('[*]'):
                for i in range(len(filled_list)):
                    if filled_list[i] == _star_path_:
                        to_eval.append(' '*indent+f'_var{i}_ = myjson{_path_}')
                        vars_set += 1
            for i in range(len(filled_list)):
                _filled_rest_ = filled_list[i][_index_:]
                if _filled_rest_.count('[*]') == 0 and '.' not in _filled_rest_ and len(_filled_rest_) > 0:
                    to_eval.append(' '*indent+f'_var{i}_ = myjson{_path_}' + _filled_rest_)
                    vars_set += 1
        elif _rest_[0] == '.':
            _star_path_ += '.'
            for i in range(len(filled_list)):
                lit_maybe = filled_list[i][len(_star_path_):]
                if isLiteral(lit_maybe):
                    to_eval.append(' '*indent+f'if "{lit_maybe}" in myjson{_path_}:')
                    to_eval.append(' '*(indent+4)+f'_var{i}_ = myjson{_path_}["{lit_maybe}"]')
                    to_eval.append(' '*indent+f'else: _var{i}_ = None')
                    vars_set += 1
            _index_ += 1
        elif _rest_[0].isalpha() or _rest_[0] == '_':
            l = len(_rest_)
            if '.' in _rest_:                           l = _rest_.index('.')
            if '[' in _rest_ and _rest_.index('[') < l: l = _rest_.index('[')
            lit = _rest_[:l]
            to_eval.append(' '*indent+f'if "{lit}" in myjson{_path_}:')
            _path_      += f'["{lit}"]'
            _star_path_ += f'{lit}'
            _index_, indent = _index_+l, indent+4
        elif _rest_[0] == '[':
            _json_index_ = _rest_[1:_rest_.index(']')]
            to_eval.append(' '*indent+f'if len(myjson{_path_}) > {_json_index_}:')
            _path_      += f'[{_json_index_}]'
            _star_path_ += f'[{_json_index_}]'
            _index_, indent = _index_ + _rest_.index(']')+1, indent + 4
            if _index_ == len(longest_by_star_path):
                for i in range(len(filled_list)):
                    if filled_list[i] == _star_path_:
                        to_eval.append(' '*indent+f'_var{i}_ = myjson{_path_}')
                        vars_set += 1
        else:
            print('Exception for the following script:\n')
            print(f'_path_      = "{_path_}"')
            print(f'_star_path_ = "{_star_path_}"')
            print(f'_rest_      = "{_rest_}"')
            print('\n'.join(to_eval)) 
            raise Exception(f'RTOntology.fillJSONPathElements() - parse error at {i}')

        if vars_set >= len(filled_list):
            for i in range(len(filled_list)):
                to_eval.append(' '*indent+f'filled["{filled_list[i]}"].append(_var{i}_)')
            break

    if to_eval[-1].endswith(':'): to_eval.append(' '*indent+'pass')
    #print('\n'.join(to_eval)) # debug
    exec('\n'.join(to_eval))
    #print(f'{filled.keys()=}') # debug
    #for x in filled: print(f'{x=}: {len(filled[x])=}') # debug
    return filled

#
# RTOntology() - ontology framework class
#
class RTOntology(object):
    # __init__() - prepare transform spec for use and initial instance variables
    def __init__(self, rt_self, xform_spec=None, labeling_verbs=None, funcs=None):
        self.rt_self        = rt_self
        self.labeling_verbs = set() if labeling_verbs is None else labeling_verbs
        if xform_spec is not None: self.xform_spec_lines = self.__substituteDefines__(xform_spec)
        else:                      self.xform_spec_lines = []
        if funcs is not None:      self.funcs = funcs
        else:                      self.funcs = {}
        self.df_triples_schema = {'uid':   pl.Int64,
                                  'sbj':   pl.Int64,
                                  'stype': pl.String,
                                  'sdisp': pl.String,
                                  'vrb':   pl.String,
                                  'obj':   pl.Int64,
                                  'otype': pl.String,
                                  'odisp': pl.String,
                                  'grp':   pl.Int64,
                                  'gdisp': pl.String,
                                  'src':   pl.String}

        # RDF triples
        self.df_triples = pl.DataFrame(schema=self.df_triples_schema)

        # Needs to match the df_triples schema
        # Needs to be the same as what is cleared later in the appendBufferedTriplesAndClearBuffer()
        self.buffered_triples = {'uid':  [],
                                 'sbj':  [],
                                 'stype': [],
                                 'sdisp': [],
                                 'vrb':   [],
                                 'obj':   [],
                                 'otype': [],
                                 'odisp': [],
                                 'grp':   [],
                                 'gdisp': [],
                                 'src':   []}
        # Unique identifiers lookups and reverse
        self.uid_lu     = {}
        self.rev_uid_lu = {}
        # Labeling information
        self.labeling_uids = {}
        self.labeling_sbjs = {}
        # Dispositions
        self._dispositions_ = {'uniq', # unique entity
                               'ambi', # ambiguous entity
                               'anon', # anonymous entity
                               'yyyy', # year                  - xsd:date
                               'dura', # duration              - xsd:duration
                               'cata', # categorical
                               'valu', # value (ints, floats)  - xsd:integer, xsd:float
                               'cont', # content (e.g., text)  - xsd:string
                               'date', # yyyy-mm-dd            - xsd:date
                               'dttm'} # timestamp             - xsd:dateTime

        # Validation errors
        self.validation_errors = set()

        # Tabular Data Supplied By User
        self.tables         = {}
        self.table_mappings = {}
        self.id_to_uid_lu   = {} # for any user-provided identifier's labeled as "uniq"

        # Performance measurements
        self.time_lu    = {}
        for x in ['fill.trace_json_paths', 'fill.collapse', 'fill.parse']: self.time_lu[x] = 0

    #
    # __repr__()
    #
    def __repr__(self):
        _strs_ = [f'RTOntology(triple_count={len(self.df_triples)}, buffered_triple_count={len(self.buffered_triples["uid"])}, uids={len(self.uid_lu)}, rev_uids={len(self.rev_uid_lu)})']
        for x in self.validation_errors: _strs_.append(f'  Validation Error: "{x}"')
        return '\n'.join(_strs_)

    # to_files() - write state to several files
    def to_files(self, _base_name_):
        # RDF triples
        self.df_triples.write_parquet(f'{_base_name_}.triples.parquet')

        # uids information
        _lu_ = {'uid':[], 't0':[], 't1':[], 't2':[], 't0_type':[]}
        for _uid_ in self.uid_lu:
            _lu_['uid'].append(_uid_)
            _lu_['t0'].append(str(self.uid_lu[_uid_][0]))
            if        self.uid_lu[_uid_][0] is None: _lu_['t0_type'].append('none')
            elif type(self.uid_lu[_uid_][0]) == str: _lu_['t0_type'].append('str')
            elif type(self.uid_lu[_uid_][0]) == int: _lu_['t0_type'].append('int')
            else: raise Exception(f'Unexpected type for "{self.uid_lu[_uid_][0]}" -- type is {type(self.uid_lu[_uid_][0])}')

            _lu_['t1'].append(self.uid_lu[_uid_][1])
            _lu_['t2'].append(self.uid_lu[_uid_][2])
        pd.DataFrame(_lu_).to_parquet(f'{_base_name_}.uids.parquet')

        # labeling information - two versions of this ... the internal UID version and then the original subject version
        _lu_ = {'uid':[], 'label':[]}
        for _uid_ in self.labeling_uids:
            _lu_['uid'].append(_uid_)
            _lu_['label'].append(self.labeling_uids[_uid_])
        pd.DataFrame(_lu_).to_parquet(f'{_base_name_}.labels.parquet')

        _lu_ = {'sbj':[], 'label':[]}
        for _sbj_ in self.labeling_sbjs:
            _lu_['sbj'].append(_sbj_)
            _lu_['label'].append(self.labeling_sbjs[_sbj_])
        pd.DataFrame(_lu_).to_parquet(f'{_base_name_}.sbjs.parquet')

        # xform spec
        if len(self.xform_spec_lines) > 0:
            with open(f'{_base_name_}.xform_spec', 'wt') as f: f.write('\n'.join(self.xform_spec_lines))

        # tabular data
        if len(self.tables) > 0:
            _lu_ = {'table_id':[], 'sbj':[], 'vrb':[], 'obj':[], 'grp':[], 'src':[]}
            for _table_id_ in self.table_mappings:
                for _map_ in self.table_mappings[_table_id_]:
                    _lu_['table_id'].append(_table_id_)
                    _lu_['sbj'].append(_map_[0])
                    _lu_['vrb'].append(_map_[1])
                    _lu_['obj'].append(_map_[2])
                    if len(_map_) > 3: _lu_['grp'].append(_map_[3])
                    else:              _lu_['grp'].append(None)
                    if len(_map_) > 4: _lu_['src'].append(_map_[5])
                    else:              _lu_['src'].append(None)
            pl.DataFrame(_lu_).write_parquet(f'{_base_name_}.table_mappings.parquet')
            for _table_id_ in self.tables:
                self.tables[_table_id_].write_parquet(f'{_base_name_}.{_table_id_}.parquet')
        
        return self

    # fm_files() - read state from several files
    def fm_files(self, _base_name_):
        # RDF triples
        self.df_triples = pl.read_parquet(f'{_base_name_}.triples.parquet')

        # uids information
        _lu_ = pd.read_parquet(f'{_base_name_}.uids.parquet')
        uid_v, t0_v, t1_v, t2_v, t0_types = _lu_['uid'].values, _lu_['t0'].values, _lu_['t1'].values, _lu_['t2'].values, _lu_['t0_type'].values
        for i in range(len(uid_v)):
            if    t0_types[i] == 'none': self.uid_lu[uid_v[i]] = (None,         t1_v[i], t2_v[i])
            elif  t0_types[i] == 'str':  self.uid_lu[uid_v[i]] = (    t0_v[i],  t1_v[i], t2_v[i])
            elif  t0_types[i] == 'int':  self.uid_lu[uid_v[i]] = (int(t0_v[i]), t1_v[i], t2_v[i])
            else: raise Exception(f'Unexpected type for "{t0_v[i]}" -- type is {t0_types[i]}')
            if t2_v[i] == 'uniq':
                _key_ = str(t0_v[i]) + '|' + str(t1_v[i])
                self.rev_uid_lu[_key_] = uid_v[i]

        # labeling information
        _lu_ = pd.read_parquet(f'{_base_name_}.labels.parquet')
        uid_v, label_v = _lu_['uid'].values, _lu_['label'].values
        for i in range(len(uid_v)):
            self.labeling_uids[uid_v[i]] = label_v[i]
            
        _lu_ = pd.read_parquet(f'{_base_name_}.sbjs.parquet')
        uid_v, label_v = _lu_['sbj'].values, _lu_['label'].values
        for i in range(len(uid_v)):
            self.labeling_sbjs[uid_v[i]] = label_v[i]
        
        # tabular data
        if os.path.exists(f'{_base_name_}.table_mappings.parquet'):
            _df_ = pl.read_parquet(f'{_base_name_}.table_mappings.parquet')
            for i in range(len(_df_)):
                _table_id_, _sbj_, _vrb_, _obj_, _grp_, _src_ = _df_['table_id'][i], _df_['sbj'][i], _df_['vrb'][i], _df_['obj'][i], _df_['grp'][i], _df_['src'][i]
                if _table_id_ not in self.table_mappings: self.table_mappings[_table_id_] = []
                if   _src_ is None and _grp_ is None:  self.table_mappings[_table_id_].append((_sbj_, _vrb_, _obj_))
                elif _src_ is None:                    self.table_mappings[_table_id_].append((_sbj_, _vrb_, _obj_, _grp_))
                else:                                  self.table_mappings[_table_id_].append((_sbj_, _vrb_, _obj_, _grp_, _src_))

            for _table_id_ in self.tables:
                self.tables[_table_id_] = pl.read_parquet(f'{_base_name_}.{_table_id_}.parquet')
        
        return self

    # __substituteDefines__() - subsitute defines
    def __substituteDefines__(self, _txt_):
        lines     = _txt_.split('\n')
        subs      = {}
        completes = []
        for _line_ in lines:
            tokens = _line_.split()
            if len(tokens) >= 3 and tokens[1] == '=':
                subs[tokens[0]] = ' '.join(tokens[2:])
            else:
                for r in subs:
                    if r in _line_:
                        _line_ = _line_.replace(r, subs[r])
                if len(_line_) > 0:
                    completes.append(_line_)
        return completes


    # solveParseTree() - evaluate a parse tree
    def solveParseTree(self, values, children, filled, i, node=None):
        if node is None: node = 'root'
        if   children[node] is None and isJsonPath(values[node]):
            return filled[values[node]][i]  # jsonpath filled in value from the json
        elif children[node] is None:
            return values[node]             # constant / literal
        else:
            parms = [self.solveParseTree(values, children, filled, i, x) for x in children[node]]
            return self.funcs[values[node]](*parms)


    # __applyTemplate__() - apply templated line in the transform to the json representation
    def __applyTemplate__(self, 
                          myjson,        # json representation
                          s_values,      s_children,    s_type,     s_disp, # subject params
                          v_values,      v_children,                        # verb params   (it's only a string, no typing, unique to the schema)
                          o_values,      o_children,    o_type,     o_disp, # object params
                          g_values,      g_children,    g_type,     g_disp, # group params
                          src_values,    src_children,                      # source params (it's only a string, no typing, unique to this ontological instance)
                          ):
        # resolve the jsonpath values        
        all_values  = set(s_values.values()) | set(v_values.values()) | set(o_values.values())
        if g_values   is not None: all_values |= set(g_values.values())
        if src_values is not None: all_values |= set(src_values.values())

        t0 = time.time()
        path_values, starred_path_values, longest_by_star_path = [], [], None
        for x in all_values:
            if isJsonPath(x):
                if '[*]' in x:
                    starred_path_values.append(x)
                    if   longest_by_star_path is None:                          longest_by_star_path = x
                    elif longest_by_star_path.rindex('[*]') < x.rindex('[*]'):  longest_by_star_path = x
                else:
                    path_values.append(x)

        # ensure that all jsonpath values are substrings of the longest star path
        for x in starred_path_values:
            x_until_last_star = x[:x.rindex('[*]')+3] # get the close bracket too
            if longest_by_star_path[:len(x_until_last_star)] != x_until_last_star:
                raise Exception(f'RTOntology.__applyTemplate__() - jsonpath are not subsets "{x}" vs "{longest_by_star_path}"')

        # fill in the json values into the filled dict
        if len(starred_path_values) > 0: filled = fillJSONPathElements(starred_path_values, myjson)
        else:                            filled = {}

        # ... double check that they are the same length
        fill_len = None
        for x in filled.keys():
            if fill_len is None: fill_len = len(filled[x])
            if len(filled[x]) != fill_len: raise Exception(f'OntologyForViz.__applyTemplate__() - unequal number of values for {x}')
        if fill_len is None: fill_len = 1 # if there are no starred paths, then we need at least one filler (it's a constant path)

        # Fix up the filled with either constants or with static json paths
        for v in all_values:
            if isJsonPath(v) and '[*]' in v: continue
            if    isJsonPath(v): to_fill = [jsonAbsolutePath(v, myjson)]
            else:                to_fill = [v]
            filled[v] = to_fill * fill_len
        t1 = time.time()
        self.time_lu['fill.trace_json_paths'] += (t1-t0)

        # collapse the parse trees based on the filled values
        # ... double check that they are the same length
        t0 = time.time()
        l = None
        for v in filled.keys():
            if l is None: l = len(filled[v])
            if len(filled[v]) != l: raise Exception(f'RTOntology.__applyTemplate__() - unequal number of values for {v}')
        pre_df = {}
        pre_df['sbj']    = [self.solveParseTree(s_values,   s_children,   filled, i) for i in range(l)]
        pre_df['vrb']    = [self.solveParseTree(v_values,   v_children,   filled, i) for i in range(l)]
        pre_df['obj']    = [self.solveParseTree(o_values,   o_children,   filled, i) for i in range(l)]
        if g_values   is not None: pre_df['grp'] = [self.solveParseTree(g_values,   g_children,   filled, i) for i in range(l)]
        if src_values is not None: pre_df['src'] = [self.solveParseTree(src_values, src_children, filled, i) for i in range(l)]
        t1 = time.time()
        self.time_lu['fill.collapse'] += (t1-t0)

        t0 = time.time()
        for_df = {'uid': [], 'sbj': [], 'stype': [], 'sdisp': [], 'vrb': [], 'obj': [], 'otype': [], 'odisp': [], 'grp':[], 'gdisp':[], 'src':[]}
        for i in range(l):
            #
            # UID (for this table row)
            # ... needs to be refactored for consistency across uses...
            #
            my_uid = 100_000 + len(self.uid_lu.keys())
            for_df['uid'].append(my_uid)
            _tuple_ = (my_uid, '__triple__', 'uniq')
            self.uid_lu[my_uid]      = _tuple_
            self.rev_uid_lu[_tuple_] = my_uid

            #
            # Subject (Required)
            #
            _sbj_, _sbj_type_, _sbj_disp_ = pre_df['sbj'][i], s_type, s_disp
            if type(_sbj_) == tuple:
                _sbj_type_ = _sbj_[1] if len(_sbj_) > 1 else s_type
                _sbj_disp_ = _sbj_[2] if len(_sbj_) > 2 else s_disp
                _sbj_      = _sbj_[0]
            _sbj_uid_ = self.resolveUniqIdAndUpdateLookups(_sbj_, _sbj_type_, _sbj_disp_, 'sbj')
            for_df['sbj'].append(_sbj_uid_), for_df['stype'].append(_sbj_type_), for_df['sdisp'].append(_sbj_disp_)

            #
            # Verb (Required)
            #
            _vrb_ = pre_df['vrb'][i]
            for_df['vrb'].append(_vrb_)

            #
            # Object (Required)
            #
            _obj_, _obj_type_, _obj_disp_ = pre_df['obj'][i], o_type, o_disp
            if type(_obj_) == tuple:
                _obj_type_ = _obj_[1] if len(_obj_) > 1 else o_type
                _obj_disp_ = _obj_[2] if len(_obj_) > 2 else o_disp
                _obj_      = _obj_[0]
            _obj_uid_ = self.resolveUniqIdAndUpdateLookups(_obj_, _obj_type_, _obj_disp_, 'obj')            
            for_df['obj'].append(_obj_uid_), for_df['otype'].append(_obj_type_), for_df['odisp'].append(_obj_disp_)

            #
            # Labeling bookkeeping
            #
            if self.labeling_verbs is not None and _vrb_ in self.labeling_verbs:
                _label_ = str(_obj_)
                if _obj_disp_ == 'anon': # anonymous
                    _label_ += f' [anon{_obj_uid_}]'
                if _obj_disp_ == 'ambi': # ambiguous
                    _label_ += f' [ambi{_obj_uid_}]'
                if _sbj_uid_ in self.labeling_uids and self.labeling_uids[_sbj_uid_] != _label_:
                    self.validation_errors.add(f'RTOntologies.__applyTemplate__() - label conflict for {_sbj_uid_} - {_label_} vs {self.labeling_uids[_sbj_uid_]}')
                self.labeling_uids[_sbj_uid_] = _label_
                if _obj_uid_ in self.labeling_sbjs and self.labeling_sbjs[_obj_uid_] != _label_:
                    self.validation_errors.add(f'RTOntologies.__applyTemplate__() - label conflict for {_obj_uid_} - {_label_} vs {self.labeling_sbjs[_obj_uid_]}')
                self.labeling_uids[_obj_uid_] = _label_
                self.labeling_sbjs[_sbj_]     = _label_

            #
            # Grouping (Optional)
            #
            if g_values is not None:
                _grp_, _grp_type_, _grp_disp_ = pre_df['grp'][i], g_type, g_disp
                if type(_grp_) == tuple:
                    _grp_type_ = _grp_[1] if len(_grp_) > 1 else g_type
                    _grp_disp_ = _grp_[2] if len(_grp_) > 2 else g_disp
                    _grp_      = _grp_[0]
                _grp_uid_ = self.resolveUniqIdAndUpdateLookups(_grp_, _grp_type_, _grp_disp_, 'grp')
                for_df['grp'].append(_grp_uid_)
                for_df['gdisp'].append(_grp_disp_)
            else:
                for_df['grp'].append(None)
                for_df['gdisp'].append(None)

            #
            # Sourcing (Optional)
            #
            if src_values is not None:
                _src_ = pre_df['src'][i]
                for_df['src'].append(str(_src_))
            else:
                for_df['src'].append(None)

        t1 = time.time()
        self.time_lu['fill.parse'] += (t1-t0)

        _df_ = pl.DataFrame(for_df, schema=self.df_triples_schema)
        return _df_

    #
    # resolveIdAndUpdateLookups() - resolve id and update lookups
    # self.uid_lu[<interger>] = (id-from-input, type-from-input, disposition-from-input)
    # _occurs_in_ == 'sbj' or 'obj' or 'sbj,obj'
    #
    def resolveUniqIdAndUpdateLookups(self, _id_, _type_, _disp_, _occurs_in_):
        _uniq_key_ = str(_id_)+'|'+str(_type_)+'|'+str(_disp_)
        if _uniq_key_ in self.rev_uid_lu: return self.rev_uid_lu[_uniq_key_]
        my_uid = 100_000 + len(self.uid_lu.keys())
        self.uid_lu[my_uid] = (_id_, _type_, _disp_)
        self.rev_uid_lu[_uniq_key_] = my_uid
        if _disp_ == 'uniq': self.id_to_uid_lu[_id_] = my_uid
        return my_uid

    #
    # resolveUniqId() - resolve id -- should have been a unique
    # ... currently not saved w/ state... so only good for initialization
    #
    def resolveUniqId(self, _id_):
        return self.id_to_uid_lu[_id_]

    #
    # createId() - create an id
    # - used to create reference non-named unique nodes
    #
    def createId(self, _type_):
        my_uid = 100_000 + len(self.uid_lu.keys())
        _uniq_key_ = str(my_uid)+'|'+str(_type_)+'|'+str('uniq')        
        self.uid_lu[my_uid] = (my_uid, _type_, 'uniq')
        self.rev_uid_lu[_uniq_key_] = my_uid
        return my_uid

    #
    # bufferTripleToAddLater() - add triple to an intermediate buffer...
    # - all values should have been created by the resolveUniqIdAndUpdateLookups()
    # - only three are required -- sbj, vrb, obj
    #
    def bufferTripleToAddLater(self, sbj, vrb, obj, grp=None, src=None):
        if type(sbj) is not int: raise Exception(f'bufferTripleToAddLater() - sbj is {type(sbj)}')
        if type(obj) is not int: raise Exception(f'bufferTripleToAddLater() - obj is {type(obj)}')

        # Create a unique id for the triple
        my_uid = 100_000 + len(self.uid_lu.keys())

        # Resolve the ids
        sbj_tuple = self.uid_lu[sbj]
        obj_tuple = self.uid_lu[obj]
        self.buffered_triples['uid'].   append(my_uid)
        self.buffered_triples['sbj'].   append(sbj)
        self.buffered_triples['stype']. append(sbj_tuple[1])
        self.buffered_triples['sdisp']. append(sbj_tuple[2])
        self.buffered_triples['vrb'].   append(vrb)
        self.buffered_triples['obj'].   append(obj)
        self.buffered_triples['otype']. append(obj_tuple[1])
        self.buffered_triples['odisp']. append(obj_tuple[2])
        self.buffered_triples['grp'].   append(grp)
        if grp is not None:
            grp_tuple = self.uid_lu[grp]
            grp_disp  = grp_tuple[2]
        else: grp_disp = None
        self.buffered_triples['gdisp']. append(grp_disp)
        self.buffered_triples['src'].   append(src)

        # As a tuple (for this event row) -- return the id for reference
        _tuple_ = (my_uid, '__triple__', 'uniq')
        self.uid_lu[my_uid]      = _tuple_
        self.rev_uid_lu[_tuple_] = my_uid

        return my_uid

    #
    # appendBufferedTriplesAndClearBuffer() - append buffered triples and clear buffers
    #
    def appendBufferedTriplesAndClearBuffer(self):
        # Add the triples
        df_buffered     = pl.DataFrame(self.buffered_triples, schema=self.df_triples.schema)
        self.df_triples = pl.concat([self.df_triples, df_buffered])

        # Clear the buffer / needs to be the same as the initialization
        self.buffered_triples = {'uid':  [],
                                 'sbj':  [],
                                 'stype': [],
                                 'sdisp': [],
                                 'vrb':   [],
                                 'obj':   [],
                                 'otype': [],
                                 'odisp': [],
                                 'grp':   [],
                                 'gdisp': [],
                                 'src':   []}

    #
    # addTabularDataFrame() - add a tabular dataframe
    #
    def addTabularDataFrame(self, 
                            df, 
                            mapping, 
                            tabular_id=None): # user supplied string
        if tabular_id is None: tabular_id_internal = self.createId('tabular')
        else:                  tabular_id_internal = self.resolveUniqIdAndUpdateLookups(tabular_id, 'tabular', 'uniq')

        # Validate the mapping -- each map should be either three, four, or five elements
        # -- [<sbj>, <vrb>, <obj>] .... should be the same as [<sbj>, <vrb>, <obj>, None, None]
        # -- [<sbj>, <vrb>, <obj>, <grp>]
        # -- [<sbj>, <vrb>, <obj>, <grp>, <src>]
        # -- elements that start with a "@" represent columns in the dataframe
        # -- elements that start with a "@__id..." represent columns in the dataframe that should have ID's previously created
        # -- elements that are 'None' are ignored and won't be represented by the ontology
        for _map_ in mapping:
            if len(_map_) != 3 and len(_map_) != 4 and len(_map_) != 5: raise ValueError('mapping must be a list of 3, 4, or 5 elements')
            for _elem_ in _map_:
                if    _elem_ is None: continue
                elif  _elem_.startswith('@__id'):
                    if _elem_[1:] not in df.columns: raise ValueError(f'mapping column "{_elem_}" not found in dataframe [1]')
                    _set_ = set(df[_elem_[1:]])
                    for _id_ in _set_: 
                        if _id_ not in self.uid_lu: raise ValueError(f'mapping id "{_id_}" from column "{_elem_}" not found in dataframe [1]')
                elif  _elem_.startswith('@'):
                    if _elem_[1:] not in df.columns: raise ValueError(f'mapping column "{_elem_}" not found in dataframe [2]')

        # Add the tabular dataframe & the related mappings
        # -- mappings should enable the dataframe to be completely represented within the triples if necessary
        self.tables[tabular_id_internal]         = df
        self.table_mappings[tabular_id_internal] = mapping

    #
    # nodeLabels() - return node labels for the sbj, obj in the df_triples structure
    #
    def nodeLabels(self, subset=None):
        if subset is None: subset = set(self.df_triples['sbj']) | set(self.df_triples['obj'])
        _node_labels_ = {}
        for _node_ in subset: 
            if self.uid_lu[_node_][0] is not None:
                _tuple_ = self.uid_lu[_node_]
                if type(_tuple_[0]) == int and _tuple_[2] == 'uniq': _node_labels_[_node_] = str(_tuple_[1])
                else:                                                _node_labels_[_node_] = str(_tuple_[0])
        return _node_labels_

    # parse() - parse json into ontology via specification
    def parse(self, jlist):
        spec_to_parse_count = {}
        if type(jlist) != list: jlist = [jlist]
        _dfs_ = []
        for j in jlist:
            for l in self.xform_spec_lines:
                spec_orig = l
                l, lu = literalize(l) # get rid of any literal values so it doesn't mess up the delimiters
                if '#' in l: l = l[:l.index('#')].strip() # comments... hope the hash symbol doesn't occur anywhere in the template that isn't a comment
                if len(l) == 0: continue

                # Sourcing Information
                src_values = src_children = None
                if '^^^' in l:
                    src = l[l.index('^^^')+3:]
                    l   = l[:l.index('^^^')].strip()
                    src_values, src_children = parseTree(fillLiterals(src, lu))

                # Grouping Information
                g_values = g_children = g_type = g_disp = None
                if '@@@' in l:
                    grp = l[l.index('@@@')+3:]
                    l   = l[:l.index('@@@')].strip()
                    g_uniq = None
                    if endsWithAny(grp, self._dispositions_) and '|' in grp:
                        g_disp = grp[grp.rindex('|')+1:].strip()
                        grp    = grp[:grp.rindex('|')]
                    else: g_disp = 'ambi'
                    g_type = None
                    if '|' in grp:
                        g_type = grp[grp.rindex('|')+1:].strip()
                        grp   = grp[:grp.rindex('|')]
                    g_node = grp
                    g_values, g_children = parseTree(fillLiterals(g_node, lu))
                    
                svo = [x.strip() for x in l.split('---')]
                if len(svo) == 3:
                    s, v, o = svo[0], svo[1], svo[2]

                    # Subject
                    s_uniq = None
                    if endsWithAny(s, self._dispositions_) and '|' in s:
                        s_disp = s[s.rindex('|')+1:].strip()
                        s      = s[:s.rindex('|')]
                    else: s_disp = 'ambi'
                    s_type = None
                    if '|' in s:
                        s_type = s[s.rindex('|')+1:].strip()
                        s      = s[:s.rindex('|')]
                    s_node = s
                    s_values, s_children = parseTree(fillLiterals(s_node, lu))

                    # Verb
                    v_values, v_children = parseTree(fillLiterals(v, lu))

                    # Object
                    o_uniq = None
                    if endsWithAny(o, self._dispositions_) and '|' in o:
                        o_disp = o[o.rindex('|')+1:].strip()
                        o      = o[:o.rindex('|')]
                    else: o_disp = 'ambi'
                    if '|' in o:
                        o_type = o[o.rindex('|')+1:].strip()
                        o      = o[:o.rindex('|')]
                    o_node = o
                    o_values, o_children = parseTree(fillLiterals(o_node, lu))
                    _df_ = self.__applyTemplate__(j, s_values, s_children, s_type, s_disp, 
                                                     v_values, v_children, 
                                                     o_values, o_children, o_type, o_disp,
                                                     g_values, g_children, g_type, g_disp,
                                                     src_values, src_children)
                    if _df_ is not None:
                        if len(_df_) > 0: 
                            _dfs_.append(_df_)
                        if spec_orig not in spec_to_parse_count: spec_to_parse_count[spec_orig] = []
                        spec_to_parse_count[spec_orig].append(len(_df_))
                else:
                    raise Exception(f'RTOntology.parse() - line "{l}" does not have three parts')

        # print out the counts (more for debugging)
        #for spec in spec_to_parse_count:
            #counts = spec_to_parse_count[spec]
            # print(f'{" ".join(spec.split())} - sum:{sum(counts)} | min:{min(counts)} | max:{max(counts)} | avg:{sum(counts) / len(counts):0.2f} | files:{len(counts)})')

        # put it all together at once
        parsed_len = 0
        if len(_dfs_) > 0:
            concatted_dfs = pl.concat(_dfs_)
            parse_len     = concatted_dfs.shape[0]
            if self.df_triples is None or self.df_triples.shape[0] == 0: self.df_triples = concatted_dfs
            else:                                                        self.df_triples = pl.concat([self.df_triples, concatted_dfs])
        return parsed_len
