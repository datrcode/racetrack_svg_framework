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
import random
from math import sin, cos, pi

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

    def test_flattenTuple(self):
        self.assertEqual(self.rt_self.flattenTuple(('fm','to'))                , ('fm', 'to'))
        self.assertEqual(self.rt_self.flattenTuple(('fm','to','other'))        , ('fm', 'to', 'other'))
        self.assertEqual(self.rt_self.flattenTuple(('a', ('b','c',('d','e')))) , ('a', 'b', 'c', 'd', 'e'))

    def test_concatDataFrames(self):
        _lu_a_ = {'a':[1,2,3], 'x':['a','b','c']}
        _lu_b_ = {'b':[4,5,6], 'y':['e','f','g']}
        _lu_c_ = {'c':[7,8,9], 'x':['x','y','z'], 'a':[11,12,13]}

        _lu_merged_ = {'a':[1,    2,    3,    None, None, None, 11,   12,   13], 
                    'x':['a',  'b',  'c',  None, None, None, 'x',  'y',  'z'],
                    'b':[None, None, None, 4,    5,    6,    None, None, None], 
                    'y':[None, None, None, 'e',  'f',  'g',  None, None, None],
                    'c':[None, None, None, None, None, None, 7,    8,    9]}

        df = self.rt_self.concatDataFrames([pd.DataFrame(_lu_a_), pd.DataFrame(_lu_b_), pd.DataFrame(_lu_c_)]).reset_index(drop=True)
        self.assertTrue(df.equals(pd.DataFrame(_lu_merged_)))

        df = self.rt_self.concatDataFrames([pl.DataFrame(_lu_a_), pl.DataFrame(_lu_b_), pl.DataFrame(_lu_c_)])
        self.assertTrue(df.equals(pl.DataFrame(_lu_merged_)))

    def test_columnsAreTimestamps(self):
        _examples_ = ['2001', 
                      '2003-02', 
                      '2004-05-01', 
                      '1998-12-01T00',            '1998-12-02 00', 
                      '1998-12-05T23:12',         '1998-12-06 02:59', 
                      '2024-12-10T13:59:50',      '2024-12-11 13:59:50',
                      '2024-12-10T13:59:50Z',     '2024-12-11 13:59:50Z',
                      '2024-12-20T00:00:00.000',  '2024-12-20 00:00:00.000',
                      '2024-12-20T00:00:00.000Z', '2024-12-20 00:00:00.000Z']
        # Single types
        for _ts_ in _examples_:
            self.rt_self.columnsAreTimestamps(pd.DataFrame({'timestamp':[_ts_]}), 'timestamp')
            self.rt_self.columnsAreTimestamps(pl.DataFrame({'timestamp':[_ts_]}), 'timestamp')
        # Mixed types
        self.rt_self.columnsAreTimestamps(pd.DataFrame({'timestamp':_examples_}), 'timestamp')
        self.rt_self.columnsAreTimestamps(pl.DataFrame({'timestamp':_examples_}), 'timestamp')

        # This comes from the tests in the temporal_barchart_mixin but fails for certain configurations on macos (homebrew python maybe?)
        _ts_, _td_, d = pd.to_datetime('2023-01-01'), pd.Timedelta(days=1), 0.0
        timestamps, colors, counts = [], [], []
        for i in range(360):
            timestamps.append(_ts_), counts.append(2.4 + sin(d)),            colors.append('red')
            timestamps.append(_ts_), counts.append(2.8 + cos(d)),            colors.append('green')
            timestamps.append(_ts_), counts.append(3   + cos(d) + 2*sin(d)), colors.append('blue')
            d    += pi/16
            _ts_ += _td_
        self.df    = pd.DataFrame({'timestamp':timestamps, 'count':counts, 'color':colors})
        self.df    = self.rt_self.columnsAreTimestamps(self.df, 'timestamp')
        self.df_pl = pl.DataFrame(self.df)
        self.df_pl = self.rt_self.columnsAreTimestamps(self.df_pl, 'timestamp')

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

    def test_hashcode(self):
        self.rt_self.hashcode('')
        self.rt_self.hashcode('abc')
        self.rt_self.hashcode('abcdef')
        self.rt_self.hashcode('abc'*5000)

    def test_stringEncodeDecode(self):
        _txt_ = '<abc!@#$%^&*()_+-={}[]|\\def:";\'<>?,./xyz0123456789'
        _enc_ = self.rt_self.stringEncode(_txt_)
        _dec_ = self.rt_self.stringDecode(_enc_)
        self.assertEqual(_txt_, _dec_)

    def test_encSVGID(self):
        _txt_ = '<abc!@#$%^&*()_+-={}[]|\\def:";\'<>?,./xyz0123456789'
        _enc_ = self.rt_self.encSVGID(_txt_)
        _dec_ = self.rt_self.decSVGID(_enc_)
        self.assertEqual(_txt_, _dec_)

    def test_transforms(self):
        _lu_ = {'timestamp':['2022-03-01 12:32:08','2020-12-30 02:08:59']}
        _answers_ = {"day_of_week":['Tue', 'Wed'],
                    "day_of_week_hour":['Tue-12', 'Wed-02'],
                    "year":['2022', '2020'],
                    "quarter":['Q1', 'Q4'],
                    "year_quarter":['2022Q1', '2020Q4'],
                    "month":['Mar', 'Dec'],
                    "year_month":['2022-03', '2020-12'],
                    "year_month_day":['2022-03-01', '2020-12-30'],
                    "year_month_day_hour":['2022-03-01 12', '2020-12-30 02'],
                    "day":['01', '30'],
                    "day_of_year":['060', '365'],
                    "day_of_year_hour":['060_12', '365_02'],
                    "hour":['12', '02'],
                    "minute":['32', '08'],
                    "second":['08', '59'],}
        df_pd = self.rt_self.columnsAreTimestamps(pd.DataFrame(_lu_), 'timestamp')
        df_pl = self.rt_self.columnsAreTimestamps(pl.DataFrame(_lu_), 'timestamp')
        for _transform_ in self.rt_self.transforms:
            if _transform_.startswith('ipv4') or _transform_ == 'log_bins': continue
            tfield           = self.rt_self.createTField('timestamp', _transform_)
            _df_ , new_field = self.rt_self.applyTransform(df_pd, tfield)
            self.assertEqual(list(_df_[new_field]), _answers_[_transform_])
            _df_, new_field = self.rt_self.applyTransform(df_pl, tfield)
            self.assertEqual(list(_df_[new_field]), _answers_[_transform_])

if __name__ == '__main__':
    unittest.main()
