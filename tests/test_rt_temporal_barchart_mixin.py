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

from math import cos, sin, pi

from rtsvg import *

class Testrt_temporal_barchart_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

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

    def test_batch_1(self):
        self.rt_self.temporalBarChart(self.df,                                      count_by='count',     color_by='color', w=1280).renderSVG()
        self.rt_self.temporalBarChart(self.df_pl,                                   count_by='count',     color_by='color', w=1280).renderSVG()
        self.rt_self.temporalBarChart(self.df.query('color=="red"'),                count_by='count',     color_by='color', w=1280, h=64).renderSVG()
        self.rt_self.temporalBarChart(self.df_pl.filter(pl.col('color') =="red"),   count_by='count',     color_by='color', w=1280, h=64).renderSVG()
        self.rt_self.temporalBarChart(self.df.query('color=="green"'),              count_by='count',     color_by='color', w=1280, h=64).renderSVG()
        self.rt_self.temporalBarChart(self.df_pl.filter(pl.col('color') =="green"), count_by='count',     color_by='color', w=1280, h=64).renderSVG()
        self.rt_self.temporalBarChart(self.df.query('color=="blue"'),               count_by='count',     color_by='color', w=1280, h=64).renderSVG()
        self.rt_self.temporalBarChart(self.df_pl.filter(pl.col('color') =="blue"),  count_by='count',     color_by='color', w=1280, h=64).renderSVG()
        self.rt_self.xy              (self.df, x_field ='timestamp', y_field='count',  color_by='color', dot_size=None, line_groupby_field='color', w=1280,h=128).renderSVG()

    def test_batch_2(self):
        df_agg = self.rt_self.temporalStatsAggregationWithGBFields(self.df, ts_field='timestamp', fields='count', freq='W', gb_fields='color').reset_index()
        self.rt_self.temporalBarChart(self.df.query('color == "red"'),             count_by='count', color_by='color', w=768, ignore_unintuitive=False).renderSVG()
        self.rt_self.temporalBarChart(self.df_pl.filter(pl.col('color') == "red"), count_by='count', color_by='color', w=768, ignore_unintuitive=False).renderSVG()
        self.rt_self.xy              (df_agg.query('color == "red"'), x_field ='timestamp', y_field='count_sum',  color_by='color', dot_size='small', line_groupby_field='color', w=768,h=128).renderSVG()

    def test_batch_3(self):
        self.df['day_of_month'] = self.df['timestamp'].apply(lambda x: x.day)
        self.df['month']        = self.df['timestamp'].apply(lambda x: x.month)

        df_pl_mod, df_pl_mod_dom = self.rt_self.applyTransform(self.df_pl,  self.rt_self.createTField('timestamp', 'day'))
        df_pl_mod, df_pl_mod_mon = self.rt_self.applyTransform(df_pl_mod,   self.rt_self.createTField('timestamp', 'month'))

        self.rt_self.temporalBarChart(self.df.query('day_of_month < 10 or day_of_month > 20'),         count_by='count', color_by='color', w=1280).renderSVG()
        self.rt_self.temporalBarChart(df_pl_mod.filter((pl.col(df_pl_mod_dom) < '10') | (pl.col(df_pl_mod_dom) > '20')), count_by='count', color_by='color', w=1280).renderSVG()

        self.rt_self.temporalBarChart(self.df.query('month == 1 or day_of_month > 5'),                 count_by='count', color_by='color', w=1280).renderSVG()
        self.rt_self.temporalBarChart(df_pl_mod.filter((pl.col(df_pl_mod_mon) == 'Jan') | (pl.col(df_pl_mod_dom) > '05')), count_by='count', color_by='color', w=1280).renderSVG()

        self.rt_self.temporalBarChart(self.df.query('month == 1 or month == 12 or day_of_month < 20'), count_by='count', color_by='color', w=1280).renderSVG()
        self.rt_self.temporalBarChart(df_pl_mod.filter((pl.col(df_pl_mod_mon) == 'Jan') | (pl.col(df_pl_mod_mon) == 'Dec') | (pl.col(df_pl_mod_dom) < '20')), count_by='count', color_by='color', w=1280).renderSVG()


    def test_batch_4(self):
        timestamps = ['2022-01-01', '2022-01-01', '2022-01-01',   '2022-01-02', '2022-01-02',   '2022-01-03', '2022-01-03', '2022-01-03', '2022-01-03', '2022-01-03',   '2022-01-04', '2022-01-04', '2022-01-04']
        colors     = ['red',        'red',        'red',          'red',        'red',          'red',        'red',        'red',        'blue',       'blue',         'green',      'red',        'blue']
        setops     = ['a',          'a',          'b',            'a',          'a',            'a',          'b',          'c',          'a',          'a',            'a',          'a',          'a']
        counts     = [1,            2,            3,              5,            2,              2,            1,            1,            4,            1,              10,           1,            3]
        df = pd.DataFrame({'timestamp':timestamps, 'color':colors, 'setop':setops, 'count':counts})
        df['timestamp'] = df['timestamp'].astype('datetime64[ms]')
        df_pl = pl.DataFrame(df)

        self.rt_self.temporalBarChart(df, color_by='color', count_by='setop',                      w=256).renderSVG()
        self.rt_self.temporalBarChart(df, color_by='color',                                        w=256).renderSVG()
        self.rt_self.temporalBarChart(df, color_by='color', count_by='count',                      w=256).renderSVG()
        self.rt_self.temporalBarChart(df, color_by='color', count_by='count',   count_by_set=True, w=256).renderSVG()

        self.rt_self.temporalBarChart(df_pl, color_by='color', count_by='setop',                      w=256).renderSVG()
        self.rt_self.temporalBarChart(df_pl, color_by='color',                                        w=256).renderSVG()
        self.rt_self.temporalBarChart(df_pl, color_by='color', count_by='count',                      w=256).renderSVG()
        self.rt_self.temporalBarChart(df_pl, color_by='color', count_by='count',   count_by_set=True, w=256).renderSVG()

    def test_batch_5(self):
        timestamps = ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', '2022-01-06', '2022-01-07', '2022-01-08', '2022-01-09', '2022-01-10', '2022-01-11', '2022-01-12', '2022-01-13']
        counts     = [1,            2,            3,            5,            4,            2,            1,            1.5,          4,            5,            6,            3.5,          3]
        groups     = ['a',          'a',          'a',          'a',          'a',          'a',          'a',          'a',         'a',           'a',          'a',          'a',          'a']
        df    = pd.DataFrame({'timestamp':timestamps, 'count':counts, 'group':groups})
        df    = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)

        timestamps, counts, groups = [], [], []
        a, d, t = 0.0, pd.Timedelta(hours=4), pd.to_datetime('2021-12-29')
        for i in range(128):
            timestamps.append(t)
            counts.append(10.0 + 5.0 * cos(a))
            groups.append('b')
            counts.append(8.0 + 2.0 * sin(2*a))
            timestamps.append(t)
            groups.append('c')
            a += pi / 16
            t += d

        df2 = pd.DataFrame({'ts':timestamps, 'ct':counts, 'group':groups})
        df2 = self.rt_self.columnsAreTimestamps(df2, 'ts')
        df2_pl = pl.DataFrame(df2)

        _svgs_ = []
        self.rt_self.temporalBarChart(df,    count_by='count').renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='count').renderSVG()
        self.rt_self.temporalBarChart(df,    count_by='count',             y2_field='count', line2_groupby_field='group', line2_groupby_color='#000000').renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='count',             y2_field='count', line2_groupby_field='group', line2_groupby_color='#000000').renderSVG()
        self.rt_self.temporalBarChart(df,    count_by='count', df2=df2,    y2_field='ct',    line2_groupby_field='group', line2_groupby_color='#000000').renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='count', df2=df2_pl, y2_field='ct',    line2_groupby_field='group', line2_groupby_color='#000000').renderSVG()
        self.rt_self.co_mgr.str_to_color_lu['b'] = '#ff0000'
        self.rt_self.co_mgr.str_to_color_lu['c'] = '#00ff00'
        self.rt_self.temporalBarChart(df,    count_by='count', df2=df2,    y2_field='ct',    line2_groupby_field='group', line2_groupby_color='group', line2_groupby_dasharray=None, dot2_size=None).renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='count', df2=df2_pl, y2_field='ct',    line2_groupby_field='group', line2_groupby_color='group', line2_groupby_dasharray=None, dot2_size=None).renderSVG()
        self.rt_self.table(_svgs_, per_row=2)

        _ts_min_, _ts_max_ = '2022-01-03 00:00:00', '2022-01-05 23:59:59'
        self.rt_self.temporalBarChart(df,    count_by='count', ts_min=_ts_min_, ts_max=_ts_max_).renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='count', ts_min=_ts_min_, ts_max=_ts_max_).renderSVG()

        _ts_min_, _ts_max_ = '2020-01-03 00:00:00', '2022-01-05 23:59:59'
        self.rt_self.temporalBarChart(df,    count_by='count', ts_min=_ts_min_, ts_max=_ts_max_).renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='count', ts_min=_ts_min_, ts_max=_ts_max_).renderSVG()

        _ts_min_, _ts_max_ = '2022-01-03 00:00:00', '2025-01-05 23:59:59'
        self.rt_self.temporalBarChart(df,    count_by='count', ts_min=_ts_min_, ts_max=_ts_max_).renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='count', ts_min=_ts_min_, ts_max=_ts_max_).renderSVG()

        _ts_min_, _ts_max_ = '2021-12-20', '2022-01-15'
        self.rt_self.temporalBarChart(df,    count_by='count', ts_min=_ts_min_, ts_max=_ts_max_).renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='count', ts_min=_ts_min_, ts_max=_ts_max_).renderSVG()

    def test_batch_6(self):
        from datetime import timedelta, date
        import random
        import pandas as pd
        my_date  = date(2020, 1, 1)
        one_day = timedelta(days=1)
        base    = 10.0
        inc     = 0.5
        ts, val = [], []
        for i in range(0,60):
            my_date =  my_date + one_day
            if (i%2) == 0:
                sd = 1.0
            else:
                sd = 6.0
            for x in np.random.normal(base, sd, random.randint(100,1000)):
                val.append(x)
                ts.append(str(my_date) + f' {random.randint(0,23):02}:{random.randint(0,59):02}')
            base += inc
        df    = pd.DataFrame({'ts': ts, 'val': val})
        df    = self.rt_self.columnsAreTimestamps(df, 'ts')
        df_pl = pl.DataFrame(df)
        self.rt_self.temporalBarChart(df,    count_by='val', style='boxplot').renderSVG()
        self.rt_self.temporalBarChart(df_pl, count_by='val', style='boxplot').renderSVG()

if __name__ == '__main__':
    unittest.main()
