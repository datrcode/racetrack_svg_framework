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

class Testrt_periodic_barchart_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

    #
    # test_batch_1() - from vunit_test
    #
    def test_batch_1(self):
        timestamps, counts, colors = [], [], []
        for x in range(10):
            for y in range(24):
                if   y <  4:
                    to_add = 1.5
                elif y < 12:
                    to_add = 4.5
                elif y < 18:
                    to_add = 8
                else:
                    to_add = 3
                counts.append(to_add)
                if   (x%3) == 0:
                    color = 'red'
                elif (x%3) == 1:
                    color = 'green'
                else:
                    color = 'blue'
                if x == 9 and y < 12:
                    color = 'black'
                colors.append(color)
                year, month, day, minute, second = random.randint(2000,2020), random.randint(1,12), random.randint(1,27), random.randint(0,59), random.randint(0,59)
                timestamps.append(f'{year:04}-{month:02}-{day:02} {y:02}:{minute:02}:{second:02}')
        df    = pd.DataFrame({'timestamp':timestamps, 'count':counts, 'color':colors})
        df    = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)
        self.rt_self.periodicBarChart(df,                      time_period='hour').renderSVG()
        self.rt_self.periodicBarChart(df_pl,                   time_period='hour').renderSVG()
        self.rt_self.periodicBarChart(df,                      time_period='hour', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df_pl,                   time_period='hour', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df,    count_by='count', time_period='hour').renderSVG()
        self.rt_self.periodicBarChart(df_pl, count_by='count', time_period='hour').renderSVG()
        self.rt_self.periodicBarChart(df,    count_by='count', time_period='hour', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df_pl, count_by='count', time_period='hour', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df,    count_by='color', time_period='hour', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df_pl, count_by='color', time_period='hour', color_by='color').renderSVG()

    #
    # test_batch_2() - from vunit_test
    #
    def test_batch_2(self):
        timestamps, counts, colors = [], [], []
        for x in range(10):
            for y in range(1,13):
                if   y in set([1, 4, 7]):
                    to_add = 1.5
                elif y in set([2, 5, 8]):
                    to_add = 4.5
                elif y in set([3, 6, 9]):
                    to_add = 8
                else:
                    to_add = 3
                counts.append(to_add)
                if   (x%3) == 0:
                    color = 'red'
                elif (x%3) == 1:
                    color = 'green'
                else:
                    color = 'blue'
                if x == 9 and y <= 6:
                    color = 'black'
                colors.append(color)
                year, hour, day, minute, second = random.randint(2000,2020), random.randint(0,23), random.randint(1,27), random.randint(0,59), random.randint(0,59)
                timestamps.append(f'{year:04}-{y:02}-{day:02} {hour:02}:{minute:02}:{second:02}')
        df    = pd.DataFrame({'timestamp':timestamps, 'count':counts, 'color':colors})
        df    = self.rt_self.columnsAreTimestamps(df, 'timestamp')
        df_pl = pl.DataFrame(df)
        self.rt_self.periodicBarChart(df,                      time_period='month').renderSVG()
        self.rt_self.periodicBarChart(df_pl,                   time_period='month').renderSVG()
        self.rt_self.periodicBarChart(df,                      time_period='month', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df_pl,                   time_period='month', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df,    count_by='count', time_period='month').renderSVG()
        self.rt_self.periodicBarChart(df_pl, count_by='count', time_period='month').renderSVG()
        self.rt_self.periodicBarChart(df,    count_by='count', time_period='month', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df_pl, count_by='count', time_period='month', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df,    count_by='color', time_period='month', color_by='color').renderSVG()
        self.rt_self.periodicBarChart(df_pl, count_by='color', time_period='month', color_by='color').renderSVG()

if __name__ == '__main__':
    unittest.main()
