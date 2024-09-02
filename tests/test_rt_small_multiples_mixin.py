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
import random

from rtsvg import *

class Testrt_small_multiples_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

    def test_batch_1(self):
        df = pd.DataFrame({'cat':['a',  'a',     'b',   'b',   'c',   'c',   'd',   'd',   'd'],
                           'bin':['x',  'y',     'x',   'y',   'x',   'y',   'x',   'y',   'z'],
                           'num':[80,   20,      70,    30,    200,   200,   100,   800,   100]})
        df_pl = pl.DataFrame(df)
        params = {'category_by':'cat', 'sm_type':'pieChart', 'color_by':'bin', 'count_by':'num', 'w_sm_override':64, 'h_sm_override':64, 'w':512}
        self.rt_self.smallMultiples(df,    **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, **params).renderSVG()
        self.rt_self.smallMultiples(df,    sort_by='records', **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, sort_by='records', **params).renderSVG()
        self.rt_self.smallMultiples(df,    sort_by='field', sort_by_field='num', **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, sort_by='field', sort_by_field='num', **params).renderSVG()
        self.rt_self.smallMultiples(df,    show_df_multiple=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, show_df_multiple=False, **params).renderSVG()
        self.rt_self.smallMultiples(df,    max_categories=2,       **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, max_categories=2,       **params).renderSVG()
        params = {'category_by':'cat', 'sm_type':'histogram', 'sm_params':{'bin_by':'bin'}, 'color_by':'bin', 'count_by':'num', 'w_sm_override':64, 'h_sm_override':64, 'w':512}
        self.rt_self.smallMultiples(df,    **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, **params).renderSVG()
        self.rt_self.smallMultiples(df,    y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df,    y_axis_independent=False, x_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, y_axis_independent=False, x_axis_independent=False, **params).renderSVG()

    def test_batch_2(self):
        df = pd.DataFrame({'cat':[ 1,   1,   1,     2,   2,   2,   2,    2,    2,    2,    2,    2],
                        'fm': ['a', 'b', 'c',   'w', 'x', 'y', 'z',  'w',  'w',  'w',  'z',  'z'],
                        'to': ['b', 'c', 'a',   'x', 'y', 'z', 'z0', 'w0', 'w1', 'w2', 'z1', 'z2']})
        df_pl = pl.DataFrame(df)
        relates = [('fm','to')]
        pos = {'a':[0,0], 'b':[2,0], 'c':[2,2], 'w':[5,5], 'x':[6,6], 'y':[7,7], 'z':[8,8], 'w0':[4,4], 'w1':[4,5], 'w2':[4,6], 'z0':[9,9], 'z1':[8,9], 'z2':[9,8]}
        params = {'category_by':'cat', 'sm_type':'linkNode', 'sm_params':{'relationships':relates, 'pos':pos}, 'w_sm_override':128, 'h_sm_override':128, 'w':512}
        self.rt_self.smallMultiples(df,    **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, **params).renderSVG()
        self.rt_self.smallMultiples(df,    x_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, x_axis_independent=False, **params).renderSVG()

    def test_batch_3(self):
        df = pd.DataFrame({'bin':['a',          'a',          'a',          'a',          'a',          'a',          'a',
                                  'b',          'b',          'b',          'b',          'b',          'b',          'b',
                                  'c',          'c',          'c',          'c',          'c',          'c',          'c'],
                           'ts': ['2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07',
                                  '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07',
                                  '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-04', '2021-01-05', '2021-01-06', '2021-01-07'],
                           'no': [5,            6,            6.3,          6,            5,            4,            4,
                                  1,            1.2,          1.3,          1.35,         1.3,          1.2,          1,
                                  9,            8,            7,            7,            7,            8,            9]})
        df    = self.rt_self.columnsAreTimestamps(df, 'ts')
        df_pl = pl.DataFrame(df)
        params = {'category_by':'bin', 'sm_type':'periodicBarChart', 'sm_params':{'time_period':'day_of_week'}, 
                'count_by':'no', 'color_by':'bin', 'h_sm_override':96, 'w_sm_override':128, 'w':640}
        self.rt_self.smallMultiples(df,    **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, **params).renderSVG()
        self.rt_self.smallMultiples(df,    x_axis_independent=True,  **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, x_axis_independent=True,  **params).renderSVG()
        self.rt_self.smallMultiples(df,    y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, y_axis_independent=False, **params).renderSVG()
        params = {'category_by':'bin', 'sm_type':'temporalBarChart', 'count_by':'no', 'color_by':'bin', 'h_sm_override':96, 'w_sm_override':128, 'w':640}
        self.rt_self.smallMultiples(df,    **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, **params).renderSVG()
        self.rt_self.smallMultiples(df,    x_axis_independent=True,  **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, x_axis_independent=True,  **params).renderSVG()
        self.rt_self.smallMultiples(df,    y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, y_axis_independent=False, **params).renderSVG()

    def test_batch_4(self):
        df = pd.DataFrame({'ts': ['2021-03-01','2021-03-02','2021-03-03','2021-03-04','2021-03-20','2021-03-21','2021-03-21','2021-03-22','2021-04-10','2021-04-10','2021-04-11','2021-04-11'],
                          'cat': [1,           1,            1,           1,          2,           2,            2,           2,          3,           3,            3,           3],
                          'num': [10,          12,           13,          11,         5,           5,            4,           3,          3,           2,            3,           5],
                          'col': ['red',       'red',        'red',       'red',      'blue',      'blue',       'yellow',    'blue',     'black',     'yellow',     'black',     'yellow']})
        df = self.rt_self.columnsAreTimestamps(df, 'ts')
        df_pl = pl.DataFrame(df)
        params = {'category_by':'cat', 'sm_type':'temporalBarChart', 'sm_params':{'ts_field':'ts'}, 
                'w_sm_override':384, 'h_sm_override':96, 'count_by':'num', 'color_by':'col', 'w':1024}
        spacer = '<svg x="0" y="0" width="800" height="32"><rect x="0" y="0" width="800" height="32" fill="#000000" /></svg>'
        self.rt_self.smallMultiples(df,    **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, **params).renderSVG()
        self.rt_self.smallMultiples(df,    x_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, x_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df,    y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df,    x_axis_independent=False, y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, x_axis_independent=False, y_axis_independent=False, **params).renderSVG()
        params = {'category_by':'cat', 'sm_type':'xy', 
                'sm_params':{'x_field':'ts', 'y_field':'num', 'line_groupby_field':'col', 'line_groupby_w':3}, 
                'w_sm_override':256, 'h_sm_override':96, 'count_by':'num', 'color_by':'col', 'w':768}
        spacer = '<svg x="0" y="0" width="518" height="32"><rect x="0" y="0" width="800" height="32" fill="#000000" /></svg>'
        self.rt_self.smallMultiples(df,    **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, **params).renderSVG()
        self.rt_self.smallMultiples(df,    x_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, x_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df,    y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df,    x_axis_independent=False, y_axis_independent=False, **params).renderSVG()
        self.rt_self.smallMultiples(df_pl, x_axis_independent=False, y_axis_independent=False, **params).renderSVG()

    def test_batch_5(self):
        df = pd.DataFrame({'fm':['a', 'b', 'c', 'a',  'a',  'a',  'd',  'd',  'd'],
                           'to':['b', 'c', 'd', 'a0', 'a1', 'a1', 'd0', 'd0', 'd2'],
                           'no':[ 10,  5,   5,   20,   6,    3,    4,    5,    8],
                           'co':['r', 'r', 'r', 'b',  'g',  'y',  'g',  'b',  'y']})
        relates = [('fm','to')]
        pos     = {'a':(0,0), 'b':(1,1), 'c':(2,2), 'd':(3,3), 'a0':(0,1), 'a1':(1,0), 'd0':(3,2), 'd2':(2,3)}
        self.rt_self.linkNode(df, relates, pos, 
                              w=512, h=512, x_ins=64, y_ins=64, node_size=24,
                              sm_type='pieChart', link_shape='curve',
                              color_by='co', count_by='no').renderSVG()
        self.rt_self.linkNode(pl.DataFrame(df), relates, pos, 
                              w=512, h=512, x_ins=64, y_ins=64, node_size=24,
                              sm_type='pieChart', link_shape='curve',
                              color_by='co', count_by='no').renderSVG()
        self.rt_self.linkNode(df, relates, pos, 
                              w=512, h=512, x_ins=64, y_ins=64, node_size=24,
                              sm_type='histogram', sm_params={'bin_by':'co'}, link_shape='curve',
                              color_by='co', count_by='no').renderSVG()
        self.rt_self.linkNode(pl.DataFrame(df), relates, pos, 
                              w=512, h=512, x_ins=64, y_ins=64, node_size=24,
                              sm_type='histogram', sm_params={'bin_by':'co'}, link_shape='curve',
                              color_by='co', count_by='no').renderSVG()


    def test_batch_6(self):
        df = pd.DataFrame({'dt':['2021-01-01', '2021-01-01', '2021-01-02', '2021-01-03', '2021-01-03'],
                           'no':[150,          50,           175,          75,           75],
                           'x' :[1,            2,            2,            3,            5],
                           'y' :[1,            2,            2,            5,            3],
                           'co':['r',          'b',          'g',          'r',          'b']})
        df = self.rt_self.columnsAreTimestamps(df, 'dt')
        self.rt_self.temporalBarChart(df, color_by='co', count_by='no', sm_type='xy', 
                                        sm_params={'x_field':'x', 'y_field':'y', 'dot_size':'large', 'draw_border':False}).renderSVG()
        self.rt_self.temporalBarChart(pl.DataFrame(df), color_by='co', count_by='no', sm_type='xy', 
                                        sm_params={'x_field':'x', 'y_field':'y', 'dot_size':'large', 'draw_border':False}).renderSVG()
        self.rt_self.xy(df, x_field='x', y_field='y', count_by='no', color_by='co', 
                            sm_type='temporalBarChart', sm_w=96, sm_h=80, w=512, h=512, x_ins=64, y_ins=64).renderSVG()
        self.rt_self.xy(pl.DataFrame(df), x_field='x', y_field='y', count_by='no', color_by='co', 
                            sm_type='temporalBarChart', sm_w=96, sm_h=80, w=512, h=512, x_ins=64, y_ins=64).renderSVG()

if __name__ == '__main__':
    unittest.main()
