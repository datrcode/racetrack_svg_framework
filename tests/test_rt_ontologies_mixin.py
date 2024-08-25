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
      self.my_json = json.loads('''
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

if __name__ == '__main__':
    unittest.main()
