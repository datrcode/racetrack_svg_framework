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

class Testrt_shapes_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

    def test_renderShapePolars(self):
        _shapes_ = ['square', 'triangle',  'utriangle', 'diamond', 'plus',  'x',      'circle', 'whatsit']
        _colors_ = ['red',    '#00ff00', 'blue',      'white',   'black', 'yellow', 'orange', 'gray']
        _coords_ = []
        for x in range(len(_shapes_)): _coords_.append(3.0*float(x))
        dfs = []
        for y in range(len(_shapes_)):
            _ys_        = []
            _szs_       = []
            _fills_     = []
            _strokes_   = []
            _stroke_ws_ = []
            for i in range(len(_shapes_)): 
                _ys_.append(3.0*float(y))
                _szs_.append(0.5 + 0.4*float(y)/len(_shapes_))
                _fills_.append(_colors_[i])
                _strokes_.append('#ff0000')
                _stroke_ws_.append(0.01)
            _df_ = pl.DataFrame({'shape':_shapes_, 'x':_coords_, 'y':_ys_, 'sz':_szs_, 'fill':_fills_, 'stroke':_strokes_, 'stroke_width':_stroke_ws_})
            dfs.append(_df_)
        df  = pl.concat(dfs)
        _tiles_ = []

        svg = ['<svg x="0" y="0" width="256" height="256" viewBox="-1.0 -1.0 25.0 25.0">']
        svg.append('<rect x="-1.0" y="-1.0" width="27.0" height="27.0" fill="white" stroke="none" />')
        for _shape_ in set(df['shape']):
            _df_     = df.filter(pl.col('shape') == _shape_)
            _str_op_ = self.rt_self.renderShapePolars(_shape_, pl.col('x'), pl.col('y'), 0.8, fill='red', stroke_width=0.1, stroke='black')
            _df_     = _df_.with_columns(pl.concat_str(_str_op_, separator='').alias('svg'))
            svg.extend(_df_['svg'])
        svg.append("</svg>")
        _tiles_.append(''.join(svg))

        svg = ['<svg x="0" y="0" width="256" height="256" viewBox="-1.0 -1.0 25.0 25.0">']
        svg.append('<rect x="-1.0" y="-1.0" width="27.0" height="27.0" fill="white" stroke="none" />')
        for _shape_ in set(df['shape']):
            _df_     = df.filter(pl.col('shape') == _shape_)
            _str_op_ = self.rt_self.renderShapePolars(_shape_, pl.col('x'), pl.col('y'), pl.col('sz'), fill='red', stroke_width=0.1, stroke='black')
            _df_     = _df_.with_columns(pl.concat_str(_str_op_, separator='').alias('svg'))
            svg.extend(_df_['svg'])
        svg.append("</svg>")
        _tiles_.append(''.join(svg))

        svg = ['<svg x="0" y="0" width="256" height="256" viewBox="-1.0 -1.0 25.0 25.0">']
        svg.append('<rect x="-1.0" y="-1.0" width="27.0" height="27.0" fill="white" stroke="none" />')
        for _shape_ in set(df['shape']):
            _df_     = df.filter(pl.col('shape') == _shape_)
            _str_op_ = self.rt_self.renderShapePolars(_shape_, pl.col('x'), pl.col('y'), pl.col('sz'), fill=pl.col('fill'), stroke_width=0.1, stroke='black')
            _df_     = _df_.with_columns(pl.concat_str(_str_op_, separator='').alias('svg'))
            svg.extend(_df_['svg'])
        svg.append("</svg>")
        _tiles_.append(''.join(svg))

        svg = ['<svg x="0" y="0" width="256" height="256" viewBox="-1.0 -1.0 25.0 25.0">']
        svg.append('<rect x="-1.0" y="-1.0" width="27.0" height="27.0" fill="white" stroke="none" />')
        for _shape_ in set(df['shape']):
            _df_     = df.filter(pl.col('shape') == _shape_)
            _str_op_ = self.rt_self.renderShapePolars(_shape_, pl.col('x'), pl.col('y'), pl.col('sz'), fill=pl.col('fill'), stroke_width=pl.col('stroke_width'), stroke='black')
            _df_     = _df_.with_columns(pl.concat_str(_str_op_, separator='').alias('svg'))
            svg.extend(_df_['svg'])
        svg.append("</svg>")
        _tiles_.append(''.join(svg))

        svg = ['<svg x="0" y="0" width="256" height="256" viewBox="-1.0 -1.0 25.0 25.0">']
        svg.append('<rect x="-1.0" y="-1.0" width="27.0" height="27.0" fill="white" stroke="none" />')
        for _shape_ in set(df['shape']):
            _df_     = df.filter(pl.col('shape') == _shape_)
            _str_op_ = self.rt_self.renderShapePolars(_shape_, pl.col('x'), pl.col('y'), pl.col('sz'), fill=pl.col('fill'), stroke_width=0.1, stroke=pl.col('stroke'))
            _df_     = _df_.with_columns(pl.concat_str(_str_op_, separator='').alias('svg'))
            svg.extend(_df_['svg'])
        svg.append("</svg>")
        _tiles_.append(''.join(svg))

        self.rt_self.tile(_tiles_, spacer=10)._repr_svg_()
