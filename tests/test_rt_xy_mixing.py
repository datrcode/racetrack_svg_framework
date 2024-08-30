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
import random

from rtsvg import *

class Testrt_xy_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

        # Batch 1
        self._bg_lu  = {'square':[(25,25),(25,35),(35,35),(35,25)],'triangle':[(15,10),(30,10),(22.5,20)]}
        self._bg2_lu = {'pill':'M 20 20 L 40 20 C 50 20 50 30 40 30 L 20 30 C 10 30 10 20 20 20 Z'}
        self.df_b1   = pd.DataFrame({'x':[10, 20, 30, 40, 50, 25], 
                                     'y':[ 5, 35, 40, 10, 20, 25], 
                                     's':[ 0,  1,  2,  3,  4,  5]})
        self.df_pl_b1   = pl.DataFrame(self.df_b1)
        self.my_vars_b1 = {'w':128,'h':128}

    def test_bg_shapes(self):
        self.rt_self.xy(self.df_b1,    x_field='x', y_field='y', dot_size='large', bg_shape_lu=self._bg_lu,  bg_shape_opacity=0.1, bg_shape_fill='#ff0000', **self.my_vars_b1)
        self.rt_self.xy(self.df_pl_b1, x_field='x', y_field='y', dot_size='large', bg_shape_lu=self._bg_lu,  bg_shape_opacity=0.1, bg_shape_fill='#ff0000', **self.my_vars_b1)

    def test_bg_shapes_2(self):
        self.rt_self.xy(self.df_b1,    x_field='x', y_field='y', dot_size='large', bg_shape_lu=self._bg_lu,  bg_shape_opacity=0.5, bg_shape_fill='vary', bg_shape_stroke='#ff0000', bg_shape_stroke_w=2.0, **self.my_vars_b1)
        self.rt_self.xy(self.df_pl_b1, x_field='x', y_field='y', dot_size='large', bg_shape_lu=self._bg_lu,  bg_shape_opacity=0.5, bg_shape_fill='vary', bg_shape_stroke='#ff0000', bg_shape_stroke_w=2.0, **self.my_vars_b1)

    def test_bg_shapes_3(self):
        self.rt_self.xy(self.df_b1,    x_field='x', y_field='y', dot_size='large', bg_shape_lu=self._bg_lu,  bg_shape_label_color='#101010', bg_shape_opacity=0.5, bg_shape_fill=None, bg_shape_stroke='#a0a0a0', bg_shape_stroke_w=2.0, **self.my_vars_b1)
        self.rt_self.xy(self.df_pl_b1, x_field='x', y_field='y', dot_size='large', bg_shape_lu=self._bg_lu,  bg_shape_label_color='#101010', bg_shape_opacity=0.5, bg_shape_fill=None, bg_shape_stroke='#a0a0a0', bg_shape_stroke_w=2.0, **self.my_vars_b1)

    def test_bg_shapes_4(self):
        self.rt_self.xy(self.df_b1,    x_field='x', y_field='y', dot_size='large', bg_shape_lu=self._bg2_lu, bg_shape_label_color='vary', bg_shape_opacity=0.5, bg_shape_fill=None, bg_shape_stroke='vary',bg_shape_stroke_w=2.0, **self.my_vars_b1)
        self.rt_self.xy(self.df_pl_b1, x_field='x', y_field='y', dot_size='large', bg_shape_lu=self._bg2_lu, bg_shape_label_color='vary', bg_shape_opacity=0.5, bg_shape_fill=None, bg_shape_stroke='vary',bg_shape_stroke_w=2.0, **self.my_vars_b1)

    def test_dot_shapes(self):
        def _my_shape_func(_df, _tuple, _x, _y, _w, _color, _opacity):
            if   rt.isPandas(_df): v = _df.iloc[0]['s']
            elif rt.isPolars(_df): v = _df['s'][0]
            if   v == 0: return 'plus'
            elif v == 1: return 'x'
            elif v == 2: return 'ellipse'
            elif v == 3: return 'square'
            elif v == 4: return 'diamond'
            elif v == 5: return 'triangle'
            else:        return 'triangle'

        self.rt_self.xy(self.df_b1,    x_field='x', y_field='y', dot_size='large', dot_shape=_my_shape_func, **self.my_vars_b1)
        self.rt_self.xy(self.df_pl_b1, x_field='x', y_field='y', dot_size='large', dot_shape=_my_shape_func, **self.my_vars_b1)

    def test_timestamps(self):
        df  = pd.DataFrame({'timestamp':['2023-01-01', '2023-01-02', '2023-01-03'], 'value':    [10,           20,           15],   'gb':['a','a','a']})
        df  = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)
        df2 = pd.DataFrame({'timestamp':['2023-01-01', '2023-01-02', '2023-01-03'], 'value':    [1020,         900,          1030], 'gb':['b','b','b']})
        df2 = self.rt_self.columnsAreTimestamps(df2, 'timestamp')
        df2_pl = pl.DataFrame(df2)

        self.rt_self.xy(df,    x_field='timestamp', y_field='value', line_groupby_field='gb', df2=df2,    x2_field='timestamp',y2_field='value',line2_groupby_field='gb', **self.my_vars_b1)
        self.rt_self.xy(df_pl, x_field='timestamp', y_field='value', line_groupby_field='gb', df2=df2_pl, x2_field='timestamp',y2_field='value',line2_groupby_field='gb', **self.my_vars_b1)

    def test_timestamps_2(self):
        x_values   = [1,      2,       3,     1,      2,      3,      1,      2,      3]
        y_values   = [10,     20,     25,     15,     22,     15,     11,     10,     12]
        groups     = ['a',    'a',    'a',    'b',    'b',    'b',    'c',    'c',    'c']
        timestamps = ['1980', '1985', '1990', '1980', '1985', '1990', '1980', '1985', '1990']
        df = pd.DataFrame({'x':x_values,'y':y_values,'group':groups,'timestamp':timestamps})
        df = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)

        self.rt_self.xy(df,    x_field='x', y_field='y', line_groupby_w=4, line_groupby_field='group', color_by='group', dot_size='large', **self.my_vars_b1)
        self.rt_self.xy(df_pl, x_field='x', y_field='y', line_groupby_w=4, line_groupby_field='group', color_by='group', dot_size='large', **self.my_vars_b1)

        self.rt_self.xy(df,    x_field='timestamp', y_field='y', line_groupby_w=4, line_groupby_field='group', color_by='group', dot_size='large', **self.my_vars_b1)
        self.rt_self.xy(df_pl, x_field='timestamp', y_field='y', line_groupby_w=4, line_groupby_field='group', color_by='group', dot_size='large', **self.my_vars_b1)

        df2 = pd.DataFrame({'x':        [1,     1.4,   1.8,   2.2,   2.6,   3],
                            'y':        [500,   510,   505,   600,   640,   610],
                            'other':    ['z',   'z',   'z',   'z',   'z',   'z'],
                            'timestamp':['1980','1982','1984','1986','1988','1990']})
        df2 = self.rt_self.columnsAreTimestamps(df2, 'timestamp')
        df2_pl = pl.DataFrame(df2)

        self.rt_self.xy(df, df2=df2, x_field='timestamp', y_field='y', y2_field='y', 
                        line_groupby_w=4, line_groupby_field='group', 
                        line2_groupby_field='other', line2_groupby_color='#000000',
                        color_by='group', dot_size='large', **self.my_vars_b1)
        self.rt_self.xy(df_pl, df2=df2_pl, x_field='timestamp', y_field='y', y2_field='y', 
                        line_groupby_w=4, line_groupby_field='group', 
                        line2_groupby_field='other', line2_groupby_color='#000000',
                        color_by='group', dot_size='large', **self.my_vars_b1)

        self.rt_self.xy(df, df2=df2, x_field='timestamp', y_field='y', y2_field='y', 
                        line_groupby_w=4, line_groupby_field='group', 
                        line2_groupby_field='other', line2_groupby_color='other', line2_groupby_w=4,
                        color_by='group', dot_size='large', **self.my_vars_b1)
        self.rt_self.xy(df_pl, df2=df2_pl, x_field='timestamp', y_field='y', y2_field='y', 
                        line_groupby_w=4, line_groupby_field='group', 
                        line2_groupby_field='other', line2_groupby_color='other', line2_groupby_w=4,
                        color_by='group', dot_size='large', **self.my_vars_b1)

    def test_timestamps_3(self):
        x_values   = [1,      2,       3,     3]
        y_values   = [10,     20,     25,     15]
        groups     = ['a',    'a',    'a',    'a']
        timestamps = ['1980', '1985', '1990', '1990']
        df = pd.DataFrame({'x':x_values,'y':y_values,'group':groups,'timestamp':timestamps})
        df = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)

        self.rt_self.xy(df,    x_field='x', y_field='y', line_groupby_w=4, line_groupby_field='group', color_by='group', dot_size='large', **self.my_vars_b1)
        self.rt_self.xy(df_pl, x_field='x', y_field='y', line_groupby_w=4, line_groupby_field='group', color_by='group', dot_size='large', **self.my_vars_b1)

    def test_timestamp_4(self):
        x_values   = [10,     10,     20,     20,     15]
        y_values   = [10,     20,     20,     10,     10]
        groups     = ['a',    'a',    'a',    'a',    'a']
        timestamps = ['1980', '1981', '1982', '1983', '1984']
        df = pd.DataFrame({'x':x_values,'y':y_values,'group':groups,'timestamp':timestamps})
        df = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)

        self.rt_self.xy(df,    x_field='x', y_field='y', line_groupby_w=4, line_groupby_field=['group','timestamp'], color_by='group', dot_size='large', **self.my_vars_b1)
        self.rt_self.xy(df_pl, x_field='x', y_field='y', line_groupby_w=4, line_groupby_field=['group','timestamp'], color_by='group', dot_size='large', **self.my_vars_b1)

    def test_timestamp_5(self):
        x_values   = [1,      2,       3,     2,      1,      1,      2,      3,      1,      2,      3]
        y_values   = [10,     20,     25,     30,     12,     15,     22,     15,     11,     10,     12]
        groups     = ['a',    'a',    'a',    'a',    'a',    'b',    'b',    'b',    'c',    'c',    'c']
        timestamps = ['1980', '1985', '1990', '1995', '2000', '1980', '1985', '1990', '1980', '1985', '1990']
        df = pd.DataFrame({'x':x_values,'y':y_values,'group':groups,'timestamp':timestamps})
        df = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)

        self.rt_self.xy(df,    x_field='x', y_field='y', line_groupby_w=4, line_groupby_field=['group','timestamp'], color_by='group', dot_size='large', **self.my_vars_b1)
        self.rt_self.xy(df_pl, x_field='x', y_field='y', line_groupby_w=4, line_groupby_field=['group','timestamp'], color_by='group', dot_size='large', **self.my_vars_b1)

        self.rt_self.xy(df,    x_field='timestamp', y_field='y', line_groupby_w=4, line_groupby_field=['group','timestamp'], color_by='group', dot_size='large', **self.my_vars_b1)
        self.rt_self.xy(df_pl, x_field='timestamp', y_field='y', line_groupby_w=4, line_groupby_field=['group','timestamp'], color_by='group', dot_size='large', **self.my_vars_b1)

    def test_timestamp_6(self):
        x_values   = [1,      2,       3,   ]
        y_values   = [10,     20,     25,   ]
        groups     = ['a',    'a',    'a',  ]
        timestamps = ['1980', '1990', '2000']
        df = pd.DataFrame({'x':x_values,'y':y_values,'group':groups,'timestamp':timestamps})
        df = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)
        x_values   = [3,      4,       5,   ]
        y_values   = [4,      2,       3,   ]
        groups     = ['a',    'a',    'a',  ]
        timestamps = ['1970', '1990', '2010']
        df2 = pd.DataFrame({'x2':x_values,'y2':y_values,'group2':groups,'timestamp2':timestamps})
        df2 = self.rt_self.columnsAreTimestamps(df2, 'timestamp2')
        df2_pl = pl.DataFrame(df2)

        self.rt_self.xy(df,         x_field ='timestamp',  y_field ='y',  line_groupby_field ='group', 
                        df2=df2,    x2_field='timestamp2', y2_field='y2', line2_groupby_field='group2', **self.my_vars_b1)
        self.rt_self.xy(df_pl,      x_field ='timestamp',  y_field ='y',  line_groupby_field ='group', 
                        df2=df2_pl, x2_field='timestamp2', y2_field='y2', line2_groupby_field='group2', **self.my_vars_b1)

    def test_order(self):
        df  = pd.DataFrame({'cat':   ['abc', 'def', 'ghi', 'jkl'], 'value': [10,    11,    12,    13], 'color':['red',  'red',  'red',  'red']})
        df_pl = pl.DataFrame(df)
        df2 = pd.DataFrame({'cat2':  ['ghi', 'jkl', 'mno', 'pqr'], 'value2':[100,   98,    96,    94], 'color':['blue', 'blue', 'blue', 'blue']})
        df2_pl = pl.DataFrame(df2)
        x_order = sorted(list(set(df['cat']) | set(df2['cat2'])))

        self.rt_self.xy(df,    x_field='cat', y_field='value', x_order=x_order, df2=df2,    x2_field='cat2', y2_field='value2', dot_size='large', color_by='color', **self.my_vars_b1)
        self.rt_self.xy(df_pl, x_field='cat', y_field='value', x_order=x_order, df2=df2_pl, x2_field='cat2', y2_field='value2', dot_size='large', color_by='color', **self.my_vars_b1)

    def test_poly_fit(self):
        _x0,_x1,_xinc = 10.0, 20.0, 0.1
        _xs,_ys = [],[]
        _x = _x0
        _samples = 20
        while _x <= _x1:
            for i in range(0,_samples):
                _y = 1.2 * _x * _x - 2.3 * _x + 4.5 + random.random()*100.0
                _xs.append(_x)
                _ys.append(_y)
            _x += _xinc
        df = pd.DataFrame({'x':_xs,'y':_ys})
        df_pl = pl.DataFrame(df)

        self.rt_self.xy(df,   x_field='x',y_field='y',dot_size='small', opacity=0.1, poly_fit_degree=1, **self.my_vars_b1)
        self.rt_self.xy(df_pl,x_field='x',y_field='y',dot_size='small', opacity=0.1, poly_fit_degree=1, **self.my_vars_b1)

        self.rt_self.xy(df,   x_field='x',y_field='y',dot_size='small', opacity=0.1, poly_fit_degree=2, **self.my_vars_b1)
        self.rt_self.xy(df_pl,x_field='x',y_field='y',dot_size='small', opacity=0.1, poly_fit_degree=2, **self.my_vars_b1)

        self.rt_self.xy(df,   x_field='x',y_field='y',dot_size='small', opacity=0.1, poly_fit_degree=3, **self.my_vars_b1)
        self.rt_self.xy(df_pl,x_field='x',y_field='y',dot_size='small', opacity=0.1, poly_fit_degree=3, **self.my_vars_b1)

        self.rt_self.xy(df,   x_field='x',y_field='y',dot_size='small', opacity=0.1, poly_fit_degree=4, **self.my_vars_b1)
        self.rt_self.xy(df_pl,x_field='x',y_field='y',dot_size='small', opacity=0.1, poly_fit_degree=4, **self.my_vars_b1)
