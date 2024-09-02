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
import numpy  as np
import polars as pl
import unittest

from rtsvg import *

class TestRTSVG(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()
        lu = {'a':[10,   20,   12,   15,   18,   100,  101],
              'b':['a',  'b',  'c',  'a',  'b',  'c',  'a'],
              'c':[1,    2,    3,    1,    2,    3,    1]}
        self.df_pd   = pd.DataFrame(lu)
        self.df_pl   = pl.DataFrame(lu)

    def test_isPandas(self):
        self.assertTrue (self.rt_self.isPandas(self.df_pd))
        self.assertFalse(self.rt_self.isPandas(self.df_pl))
    
    def test_isPolars(self):
        self.assertFalse(self.rt_self.isPolars(self.df_pd))
        self.assertTrue (self.rt_self.isPolars(self.df_pl))

    def test_polarsCounter(self):
        df = pl.DataFrame({'a':[ 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5],
                           'b': 'a  b  c  d  e  f  g  h  i  j  k  l  m  m'.split(),
                           'c': 'a  a  a  a  a  a  b  b  b  b  b  c  c  c'.split(),
                           'd':[ 10,11,12,13,14,15,16,17,18,19,20,21,22,23]})
        # Count By Rows (into column 'a')
        _df_ = self.rt_self.polarsCounter(df, 'a')
        self.assertEqual(_df_.shape, (5, 2))
        _lu_, _luc_ = {}, {}        
        for i in range(len(_df_)): _lu_[_df_['a'][i]] = _df_['__count__'][i]
        for _tuple_ in [(5,1), (4,4), (3,3), (2,3), (1,3)]: _luc_[_tuple_[0]] = _tuple_[1]
        self.assertEqual(_lu_, _luc_)

        # Count by Numbers (into column 'a' ... from column 'a')
        _df_ = self.rt_self.polarsCounter(df,'a', count_by='a')
        _lu_, _luc_ = {}, {}        
        for i in range(len(_df_)): _lu_[_df_['a'][i]] = _df_['__count__'][i]
        for _tuple_ in [(5,5), (4,16), (3,9), (2,6), (1,3)]: _luc_[_tuple_[0]] = _tuple_[1]
        self.assertEqual(_lu_, _luc_)

        # Count by Numbers (into column 'a' ... from column 'a')
        _df_ = self.rt_self.polarsCounter(df,'a', count_by='a', count_by_set=True)
        _lu_, _luc_ = {}, {}
        for i in range(len(_df_)): _lu_[_df_['a'][i]] = _df_['__count__'][i]
        for _tuple_ in [(5,1), (4,1), (3,1), (2,1), (1,1)]: _luc_[_tuple_[0]] = _tuple_[1]
        self.assertEqual(_lu_, _luc_)

        # MANY MORE NEEDED ...


if __name__ == '__main__':
    unittest.main()
