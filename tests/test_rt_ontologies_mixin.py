import pandas as pd
import polars as pl
import numpy as np
import json
from rtsvg import *
rt = RACETrack()
def lEquals(l1, l2):
    if len(l1) != len(l2): return False
    for i in range(len(l1)):
        if type(l1[i]) != type(l2[i]): return False
        if type(l1[i]) == list:
            l_equals = lEquals(l1[i],l2[i])
            if l_equals == False: return False
        if type(l1[i]) == dict:
            d_equals = dEquals(l1[i],l2[i])
            if d_equals == False: return False
        if l1[i] != l2[i]: return False
    return True
def dEquals(d1, d2):
    # Checks keys first
    d1set, d2set = set(d1.keys()), set(d2.keys())
    if len(d1set) != len(d2set): return False
    for k in d1set: 
        if k not in d2set: return False

    # Checks values
    for k in d1set:
        v1, v2 = d1[k], d2[k]
        if type(v1) != type(v2): return False
        if type(v1) == list:
            l_equals = lEquals(v1,v2)
            if l_equals == False: return False
        if type(v1) == dict:
            d_equals = dEquals(v1,v2)
            if d_equals == False: return False
        if v1 != v2: return False
    
    return True

my_json = json.loads('''
{
  "id":      1,
  "id_str": "1",
  "array":  [1, 2, 3],
  "dict":   {"a": 1, "b": 2},
  "empty_stuff":[],
  "empty_dict":{},
  "more-stuff":[ {"id":100, "name":"mary"},
                 {"id":101, "name":"joe"},
                 {"id":102, "name":"fred",  "jobs":["scientist"]},
                 {"id":103},
                 {"id":104, "name":"sally", "jobs":["developer", "manager", "accountant"]} ],
  "arr_win_arr": [[1, 2, 3], [4, 5, 6]],
  "arr_deeper":  [ {"value": 2.3, "stuff": [1, 2, 3]},
                   {"value": 4.5, "stuff": [4, 5, 6]}                       
  ]
}
''')

assert jsonAbsolutePath("$.id",                     my_json) == 1
assert jsonAbsolutePath("$.more-stuff[1].id",       my_json) == 101
assert jsonAbsolutePath("$.more-stuff[3].name",     my_json) is None
assert jsonAbsolutePath("$.more-stuff[4].jobs[1]",  my_json) == 'manager'
assert jsonAbsolutePath("$.more-stuff[4].jobs[3]",  my_json) is None
assert jsonAbsolutePath("$.arr_win_arr[1]",         my_json) == [4, 5, 6]
assert jsonAbsolutePath("$.arr_deeper[0].value",    my_json) == 2.3
_results_ = fillJSONPathElements(["$.more-stuff[*].name"], my_json) 
assert dEquals(_results_, {'$.more-stuff[*].name': ['mary', 'joe', 'fred', None, 'sally']})
_results_ = fillJSONPathElements(["$.more-stuff[*].name", "$.more-stuff[*].id"], my_json)
assert dEquals(_results_, {'$.more-stuff[*].name': ['mary', 'joe', 'fred', None, 'sally'],'$.more-stuff[*].id': [100, 101, 102, 103, 104]})
_results_ = fillJSONPathElements(["$.more-stuff[*].jobs[*]", "$.more-stuff[*].id"], my_json) 
assert dEquals(_results_, {'$.more-stuff[*].jobs[*]': ['scientist', 'developer', 'manager', 'accountant'], '$.more-stuff[*].id': [102, 104, 104, 104]})
_results_ = fillJSONPathElements(["$.arr_deeper[0].stuff[*]", "$.arr_deeper[0].value"], my_json)
assert dEquals(_results_, {'$.arr_deeper[0].stuff[*]': [1, 2, 3], '$.arr_deeper[0].value': [2.3, 2.3, 2.3]})
_results_ = fillJSONPathElements(["$.more-stuff[*].jobs[0]", "$.more-stuff[*].id"], my_json)
assert dEquals(_results_, {'$.more-stuff[*].jobs[0]': ['scientist', 'developer'], '$.more-stuff[*].id': [102, 104]})
_results_ = fillJSONPathElements(["$.more-stuff[*].jobs[1]", "$.more-stuff[*].id"], my_json)
assert dEquals(_results_, {'$.more-stuff[*].jobs[1]': ['manager'], '$.more-stuff[*].id': [104]})
_results_ = fillJSONPathElements(["$.more-stuff[*].jobs[5]", "$.more-stuff[*].id"], my_json)
assert dEquals(_results_, {'$.more-stuff[*].jobs[5]': [], '$.more-stuff[*].id': []})
