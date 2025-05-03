# Copyright 2025 David Trimm
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

class Testrt_piechart_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()
        _lu_ = {'bin':[], 'count':[], 'color':[]}
        for i in range(1000):
            _lu_['bin']  .append(chr(ord('A') + random.randint(0,25)))
            _lu_['count'].append(1.0 + 100.0 * random.random())
            _lu_['color'].append(random.choice(['red', 'green', 'blue']))
        self.df    = pd.DataFrame(_lu_)
        self.df_pl = pl.DataFrame(self.df)

    def test_basicPieChartPandas(self):
        self.rt_self.pieChart(self.df, color_by='bin',   count_by=None)   .renderSVG()
        self.rt_self.pieChart(self.df, color_by='bin',   count_by='count').renderSVG()
        self.rt_self.pieChart(self.df, color_by='color', count_by=None)   .renderSVG()
        self.rt_self.pieChart(self.df, color_by='color', count_by='count').renderSVG()

    def test_basicPieChartPolars(self):
        self.rt_self.pieChart(self.df_pl, color_by='bin',   count_by=None)   .renderSVG()
        self.rt_self.pieChart(self.df_pl, color_by='bin',   count_by='count').renderSVG()
        self.rt_self.pieChart(self.df_pl, color_by='color', count_by=None)   .renderSVG()
        self.rt_self.pieChart(self.df_pl, color_by='color', count_by='count').renderSVG()

    def test_basicDonutChartPandas(self):
        self.rt_self.pieChart(self.df, color_by='bin',   count_by=None,    style="donut").renderSVG()
        self.rt_self.pieChart(self.df, color_by='bin',   count_by='count', style="donut").renderSVG()
        self.rt_self.pieChart(self.df, color_by='color', count_by=None,    style="donut").renderSVG()
        self.rt_self.pieChart(self.df, color_by='color', count_by='count', style="donut").renderSVG()

    def test_basicDonutChartPolars(self):
        self.rt_self.pieChart(self.df_pl, color_by='bin',   count_by=None,    style="donut").renderSVG()
        self.rt_self.pieChart(self.df_pl, color_by='bin',   count_by='count', style="donut").renderSVG()
        self.rt_self.pieChart(self.df_pl, color_by='color', count_by=None,    style="donut").renderSVG()
        self.rt_self.pieChart(self.df_pl, color_by='color', count_by='count', style="donut").renderSVG()

    def test_basicWaffleChartPandas(self):
        self.rt_self.pieChart(self.df, color_by='bin',                     style='waffle').renderSVG()
        self.rt_self.pieChart(self.df, color_by='bin',   count_by='count', style='waffle').renderSVG()
        self.rt_self.pieChart(self.df, color_by='color',                   style='waffle').renderSVG()
        self.rt_self.pieChart(self.df, color_by='color', count_by='count', style='waffle').renderSVG()

    def test_basicWaffleChartPolars(self):
        ... # never implemented

if __name__ == '__main__':
    unittest.main()
