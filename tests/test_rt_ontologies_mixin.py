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
import unittest
import pandas as pd
import polars as pl
import numpy as np
import json
from rtsvg import *

class Testrt_ontologies_mixin(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.rt_self = RACETrack()
    _my_json_str_ = '''
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
      '''
    self.my_json = json.loads(_my_json_str_)

  def test_ontologyFrameworkInstance(self):
    self.assertTrue(jsonAbsolutePath("$.id", self.my_json) == 1)
    self.assertTrue(jsonAbsolutePath("$.more-stuff[1].id", self.my_json) == 101)
    self.assertTrue(jsonAbsolutePath("$.more-stuff[3].name",     self.my_json) is None)
    self.assertTrue(jsonAbsolutePath("$.more-stuff[4].jobs[1]",  self.my_json) == 'manager')
    self.assertTrue(jsonAbsolutePath("$.more-stuff[4].jobs[3]",  self.my_json) is None)
    self.assertTrue(jsonAbsolutePath("$.arr_win_arr[1]",         self.my_json) == [4, 5, 6])
    self.assertTrue(jsonAbsolutePath("$.arr_deeper[0].value",    self.my_json) == 2.3)
    _results_ = fillJSONPathElements(["$.more-stuff[*].name"], self.my_json) 
    self.assertDictEqual(_results_, {'$.more-stuff[*].name': ['mary', 'joe', 'fred', None, 'sally']})
    _results_ = fillJSONPathElements(["$.more-stuff[*].name", "$.more-stuff[*].id"], self.my_json)
    self.assertDictEqual(_results_, {'$.more-stuff[*].name': ['mary', 'joe', 'fred', None, 'sally'],'$.more-stuff[*].id': [100, 101, 102, 103, 104]})
    _results_ = fillJSONPathElements(["$.more-stuff[*].jobs[*]", "$.more-stuff[*].id"], self.my_json) 
    self.assertDictEqual(_results_, {'$.more-stuff[*].jobs[*]': ['scientist', 'developer', 'manager', 'accountant'], '$.more-stuff[*].id': [102, 104, 104, 104]})
    _results_ = fillJSONPathElements(["$.arr_deeper[0].stuff[*]", "$.arr_deeper[0].value"], self.my_json)
    self.assertDictEqual(_results_, {'$.arr_deeper[0].stuff[*]': [1, 2, 3], '$.arr_deeper[0].value': [2.3, 2.3, 2.3]})
    _results_ = fillJSONPathElements(["$.more-stuff[*].jobs[0]", "$.more-stuff[*].id"], self.my_json)
    self.assertDictEqual(_results_, {'$.more-stuff[*].jobs[0]': ['scientist', 'developer'], '$.more-stuff[*].id': [102, 104]})
    _results_ = fillJSONPathElements(["$.more-stuff[*].jobs[1]", "$.more-stuff[*].id"], self.my_json)
    self.assertDictEqual(_results_, {'$.more-stuff[*].jobs[1]': ['manager'], '$.more-stuff[*].id': [104]})
    _results_ = fillJSONPathElements(["$.more-stuff[*].jobs[5]", "$.more-stuff[*].id"], self.my_json)
    self.assertDictEqual(_results_, {'$.more-stuff[*].jobs[5]': [], '$.more-stuff[*].id': []})

  def test_mappingAlgorithm(self):
    _json_txt_ = '''
    {"id":1,
    "people":[{"first":"John", "last":"Smith", "id":10, "citescore":2.3, "age":30, "city":"nyc",          "state":"ny", "country":"us"},
              {"first":"Joe",  "last":"Smith", "id":20, "citescore":1.8, "age":35,                        "state":"ny", "country":"us"},
              {"first":"Mary", "last":"Jones", "id":30, "age":32, "city":"philadelphia", "state":"pa", "country":"us"}],
    "knowsFrom":[[10, 20, "Conference A"], 
                  [20, 30, "Conference B"]],
    "education":[{"id":10, "degreeReceived":"Ph.D. in Computer Science",   "university":"Stanford University"},
                  {"id":10, "degreeReceived":"Masters in Computer Science", "university":"University of Pennsylvania"}],
    "total_people":3
    }'''
    _json_simple_  = json.loads(_json_txt_)

    def concatNames(_last_,_first_):
      return _last_ + ' ' + _first_
    def combineAddress(_city_,_state_,_country_):
      s = ''
      if _city_    is not None: s += _city_
      if _state_   is not None: s += ', ' + _state_    if (len(s) > 0) else _state_
      if _country_ is not None: s += ', ' + _country_  if (len(s) > 0) else _country_
      return s if (len(s) > 0) else 'Not Supplied'
    _xform_simple_ = '''
_id_ = '$.people[*].id' | PersonID | uniq
'$.id'                                --- "hasEntryCount"    --- '$.total_people' | xsd:integer                                                                           ^^^ "IN_TEMPLATE"
_id_                                  --- "hasName"          --- concatNames('$.people[*].last', '$.people[*].first') | xsd:string                                        ^^^ "IN_TEMPLATE"
_id_                                  --- "hasCitationScore" --- '$.people[*].citescore' | xsd:float   | valu                                                             ^^^ '$.id'    
_id_                                  --- "hasAge"           --- '$.people[*].age'       | xsd:integer | valu                                                             ^^^ '$.id'
_id_                                  --- "isFrom"           --- combineAddress('$.people[*].city', '$.people[*].state', '$.people[*].country') | CityStateCountry | uniq ^^^ '$.id'
_id_                                  --- "isFromCity"       --- '$.people[*].city'      | City                                                                           ^^^ '$.id'
'$.knowsFrom[*][0]' | PersonID | uniq --- "knows"            --- '$.knowsFrom[*][1]'     | PersonID    | uniq                 @@@ '$.knowsFrom[*][2]' | xsd:string | uniq ^^^ '$.id'
'''

    ofv_simple = self.rt_self.ontologyFrameworkInstance(xform_spec=_xform_simple_, funcs={'concatNames': concatNames, 'combineAddress': combineAddress})
    ofv_simple.parse(_json_simple_)

if __name__ == '__main__':
    unittest.main()
