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
from shapely.geometry import Polygon

from rtsvg import *

class Testrt_choroplethmap_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()
    def test_rt_choroplethmap_mixin_pandas(self):
        df = pd.DataFrame({'count':[1, 2, 3, 4, 1, 2, 3, 4], 
                           'shape':[0, 1, 2, 3, 0, 1, 2, 3]})
        self.basics(df)

    def test_rt_choroplethmap_mixin_polars(self):
        df = pl.DataFrame({'count':[1, 2, 3, 4, 1, 2, 3, 4], 
                           'shape':[0, 1, 2, 3, 0, 1, 2, 3]})
        self.basics(df)

    def basics(self, df):
        shape_lu = {}
        x_orig, y_orig = 0.0, 0.0
        shape_lu[0] = Polygon([(x_orig,       y_orig),       (x_orig + 0.5, y_orig + 0.0), 
                            (x_orig + 0.5, y_orig + 0.5), (x_orig + 0.0, y_orig + 0.5)])
        x_orig, y_orig = 0.5, 0.0
        shape_lu[1] = Polygon([(x_orig,       y_orig),       (x_orig + 0.5, y_orig + 0.0), 
                            (x_orig + 0.5, y_orig + 0.5), (x_orig + 0.0, y_orig + 0.5)])
        x_orig, y_orig = 0.0, 0.5
        shape_lu[2] = Polygon([(x_orig,       y_orig),       (x_orig + 0.5, y_orig + 0.0), 
                            (x_orig + 0.5, y_orig + 0.5), (x_orig + 0.0, y_orig + 0.5)])
        x_orig, y_orig = 0.5, 0.5
        shape_lu[3] = Polygon([(x_orig,       y_orig),       (x_orig + 0.5, y_orig + 0.0), 
                            (x_orig + 0.5, y_orig + 0.5), (x_orig + 0.0, y_orig + 0.5)])
        x_orig, y_orig = -1.0, -1.0
        shape_lu[4] = Polygon([(x_orig,       y_orig),       (x_orig + 0.5, y_orig + 0.0), 
                            (x_orig + 0.5, y_orig + 0.5), (x_orig + 0.0, y_orig + 0.5)])

        tiles = []

        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu))
        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, count_by='count'))
        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, count_by='count', count_by_set=True))

        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, bounds_from_all_shapes=False))
        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, bounds_from_all_shapes=False, count_by='count'))
        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, bounds_from_all_shapes=False, count_by='count', count_by_set=True))

        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, draw_outlines=False))
        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, draw_outlines=False, count_by='count'))
        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, draw_outlines=False, count_by='count', count_by_set=True))

        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, outline_all_shapes=False))
        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, outline_all_shapes=False, count_by='count'))
        tiles.append(self.rt_self.choroplethMap(df, 'shape', shape_lu, outline_all_shapes=False, count_by='count', count_by_set=True))

        self.rt_self.table(tiles, per_row=3)._repr_svg_() # Force the render


if __name__ == '__main__':
    unittest.main()
