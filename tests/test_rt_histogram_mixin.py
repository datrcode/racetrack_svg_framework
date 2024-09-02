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
import datetime

from rtsvg import *

class Testrt_histogram_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

    def test_batch_1(self):
        df = pd.DataFrame({'bin':   ['a',   'a',     'b',   'b',     'c',   'c',    'c',      'd',   'd',     'd',    'd'],
                           'bin2':  ['x',   'x',     'x',   'x',     'x',   'y',    'z',      'x',   'y',     'z',    'w'],
                           'amt':   [10,    5,       10,    30,      1,     2,      1,        1,      2,      3,      4],
                           'amt2':  [1,     2,       1,     2,       1,     2,      3,        1,      2,      3,      4],
                           'color': ['red', 'blue',  'red', 'red',   'red', 'blue', 'yellow', 'blue', 'blue', 'blue', 'red']})
        df_pl = pl.DataFrame(df)
        _parms_ = {'w':160,'h':96}


        self.rt_self.histogram(df, bin_by='bin',                   **_parms_).renderSVG()
        self.rt_self.histogram(df, bin_by='bin', count_by='bin2',  **_parms_).renderSVG()
        self.rt_self.histogram(df, bin_by='bin', count_by='amt',   **_parms_).renderSVG()
        self.rt_self.histogram(df, bin_by='bin', count_by='amt2',  **_parms_).renderSVG()
        self.rt_self.histogram(df, bin_by='bin', count_by='color', **_parms_).renderSVG()

        self.rt_self.histogram(df_pl, bin_by='bin',                   **_parms_).renderSVG()
        self.rt_self.histogram(df_pl, bin_by='bin', count_by='bin2',  **_parms_).renderSVG()
        self.rt_self.histogram(df_pl, bin_by='bin', count_by='amt',   **_parms_).renderSVG()
        self.rt_self.histogram(df_pl, bin_by='bin', count_by='amt2',  **_parms_).renderSVG()
        self.rt_self.histogram(df_pl, bin_by='bin', count_by='color', **_parms_).renderSVG()

        _parms_['color_by'] = 'color'

        self.rt_self.histogram(df, bin_by='bin',                   **_parms_).renderSVG()
        self.rt_self.histogram(df, bin_by='bin', count_by='bin2',  **_parms_).renderSVG()
        self.rt_self.histogram(df, bin_by='bin', count_by='amt',   **_parms_).renderSVG()
        self.rt_self.histogram(df, bin_by='bin', count_by='amt2',  **_parms_).renderSVG() # second row shows a little bit more than the row on top...  both should be 6
        self.rt_self.histogram(df, bin_by='bin', count_by='color', **_parms_).renderSVG()

        self.rt_self.histogram(df_pl, bin_by='bin',                   **_parms_).renderSVG()
        self.rt_self.histogram(df_pl, bin_by='bin', count_by='bin2',  **_parms_).renderSVG()
        self.rt_self.histogram(df_pl, bin_by='bin', count_by='amt',   **_parms_).renderSVG()
        self.rt_self.histogram(df_pl, bin_by='bin', count_by='amt2',  **_parms_).renderSVG() # second row shows a little bit more than the row on top...  both should be 6
        self.rt_self.histogram(df_pl, bin_by='bin', count_by='color', **_parms_).renderSVG()


    def test_batch_2(self):
        df = pd.DataFrame({'bin':   ['a',   'a',     'b',   'b',     'c',   'c',    'c',      'd',   'd',     'd',    'd'],
                           'bin2':  ['x',   'x',     'x',   'x',     'x',   'y',    'z',      'x',   'y',     'z',    'w'],
                           'amt':   [10,    5,       10,    30,      1,     2,      1,        1,      2,      3,      4],
                           'amt2':  [1,     2,       1,     2,       1,     2,      3,        1,      2,      3,      4],
                           'color': ['red', 'blue',  'red', 'red',   'red', 'blue', 'yellow', 'blue', 'blue', 'blue', 'red']})
        df_pl = pl.DataFrame(df)

        _parms_ = {'w':160,'h':128}
        
        _str_ = 'bin'
        self.rt_self.histogram(df, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()
        _str_ = 'bin2'
        self.rt_self.histogram(df, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()
        _str_ = 'amt'
        self.rt_self.histogram(df, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()
        _str_ = 'amt2'
        self.rt_self.histogram(df, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()
        _str_ = 'color'
        self.rt_self.histogram(df, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()

        _str_ = 'bin'
        self.rt_self.histogram(df_pl, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()
        _str_ = 'bin2'
        self.rt_self.histogram(df_pl, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()
        _str_ = 'amt'
        self.rt_self.histogram(df_pl, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()
        _str_ = 'amt2'
        self.rt_self.histogram(df_pl, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()
        _str_ = 'color'
        self.rt_self.histogram(df_pl, bin_by=_str_, count_by=_str_, color_by=_str_, **_parms_).renderSVG()


    def test_batch_3(self):
        df = pd.DataFrame({'bin':  ['a','a','b','b','b','c'], 
                          'bin2': [ 1,  2,  1,  2,  3,  4],
                          'count':[10, 5,   8,  3,  1,  4]})

        params = {'df':df, 'bin_by':'bin', 'count_by':'count', 'w':128, 'h':108}
        self.rt_self.histogram(**params).renderSVG()
        labels = {'a':'expansion', 'b':'way too many words', 'c':'even more more more more more words go here'}
        self.rt_self.histogram(labels=labels, **params).renderSVG()
        params['bin_by'] = ['bin','bin2']
        self.rt_self.histogram(**params).renderSVG()
        labels = {'a | 1':'first a', 'c | 4':'fourth "c"'}
        self.rt_self.histogram(labels=labels, **params).renderSVG()
        hst = self.rt_self.histogram(labels=labels, **params)
        self.rt_self.tile([self.rt_self.annotateEntities(hst, ['first a', 'b | 3'])])._repr_svg_()

    def test_batch_4(self):
        pass # this is that self referential section that has an exception in it (2024-09-02)

if __name__ == '__main__':
    unittest.main()
