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
import networkx as nx
import datetime

from rtsvg import *

class Testrt_liknode_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

        # Batch 1
        self.df_b1 = pd.DataFrame({'fm': ['a',    'b',    'c',    'a',    'a',    'a',    'b',    'c',    'd',    'd',    'd',    'd',    'd',    'd'],
                                   'fmi':[0,      1,      2,      0,      0,      0,      1,      3,      3,      3,      3,      3,      3,      3],
                                    'to': ['b',    'c',    'd',    'a0',   'a1',   'a2',   'b1',   'c0',   'd0',   'd1',   'd2',   'd3',   'd3',   'd3'],
                                    'toi':[1,      2,      3,      10,     11,     12,     21,     30,     40,     41,     42,     43,     43,     43],
                                    'ct': [5,      5,      5,      1,      1,      1,      3,      2,      2,      2,      2,      2,      2,      2],
                                    'co': ['core', 'core', 'core', 'sub1', 'sub1', 'sub1', 'spur', 'spur', 'sub2', 'sub2', 'sub2', 'sub2', 'sub2', 'sub2']})
        self.df_pl_b1 = pl.DataFrame(self.df_b1)
        self.relates_b1 = [('fm','to')]
        self.pos_b1     = {"a":  [-0.13,-0.72],           "a0": [-0.13,-1.00],           "a1": [ 0.09,-0.87],           "a2": [-0.35,-0.88],
                           "b":  [-0.12,-0.30],           "b1": [-0.37,-0.30],           "c":  [-0.00, 0.12],           "d":  [ 0.13, 0.55],
                           "d0": [ 0.31, 0.82],           "d1": [ 0.41, 0.48],           "d2": [ 0.04, 0.84],           "d3": [-0.13, 0.64],
                           "c0": [ 0.24, 0.10]}

        # Batch 2
        self.relates_b2 = [('fmi','toi')]
        self.pos_b2     = {"0":  [-0.13,-0.72],           "10": [-0.13,-1.00],           "11": [ 0.09,-0.87],           "12": [-0.35,-0.88],
                        "1":  [-0.12,-0.30],           "21": [-0.37,-0.30],           "2":  [-0.00, 0.12],           "3":  [ 0.13, 0.55],
                        "40": [ 0.31, 0.82],           "41": [ 0.41, 0.48],           "42": [ 0.04, 0.84],           "43": [-0.13, 0.64],
                        "30": [ 0.24, 0.10]}
        self.pos_i_b2   = {0:  [-0.13,-0.72],              10: [-0.13,-1.00],           11: [ 0.09,-0.87],           12: [-0.35,-0.88],
                        1:  [-0.12,-0.30],              21: [-0.37,-0.30],           2:  [-0.00, 0.12],           3:  [ 0.13, 0.55],
                        40: [ 0.31, 0.82],              41: [ 0.41, 0.48],           42: [ 0.04, 0.84],           43: [-0.13, 0.64],
                        30: [ 0.24, 0.10]}
        
        # Batch 3
        df_a_b3    = pd.DataFrame({'fm':['a','b','c'],
                                   'to':['b','c','d']})
        df_b_b3    = pd.DataFrame({'src':['a','a','a','d','d','d'],
                                   'dst':[1,  2,  3,  4,  5,  6]})
        self.df_b3      = self.rt_self.concatDisparateDataFrames([df_a_b3, df_b_b3])
        df_a_pl_b3 = pl.DataFrame(df_a_b3)
        df_b_pl_b3 = pl.DataFrame(df_b_b3)
        self.df_pl_b3   = self.rt_self.concatDisparateDataFrames([df_a_pl_b3, df_b_pl_b3])
        self.pos_b3 = {'a': [0.0,0.0], 1:[-0.5,-0.5], 2:[-0.5,0.0], 3:[-0.5,0.5],
                       'b': [1.0,1.0],
                       'c': [0.0,2.0],
                       'd': [1.0,3.0], 4:[1.5, 2.5],  5:[1.5, 3.0],  6:[1.5, 3.5]}
        self.relates_b3 = [('fm','to'),('src','dst')]

        # Batch 4
        _df1_    = pd.DataFrame({'fm':['a','b','c'],      'to':['b','c','d'],      'count':[10, 20, 1]})
        _df2_    = pd.DataFrame({'src':['x','y','z','a'], 'dst':['1','2','3','4'], 'count':[8,   2, 4, 12]})
        self.df1_b4      = self.rt_self.concatDisparateDataFrames([_df1_, _df2_])
        _df1_pl_ = pl.DataFrame(_df1_)
        _df2_pl_ = pl.DataFrame(_df2_)
        self.df1_pl_b4   = self.rt_self.concatDisparateDataFrames([_df1_pl_, _df2_pl_])
        self.multi_relates_b4 = [('fm','to'),('src','dst')]
        self.node_shapes_b4   = {'fm':'ellipse','to':'square','src':'triangle','dst':'plus'}
        self.pos_b4 = {'a':[0,0],'b':[1,0],'c':[1,1],'d':[0,1],'x':[0,0.4],'1':[0.2,0.4],'y':[0,0.2],'2':[0.2,0.2],'z':[0,0.6],'3':[0.2,0.6],'4':[0.6,0.2]}

        # Batch 5
        self.df_b5 = pd.DataFrame({'fm':  ['a','b','c','d','c', 'a', 'a', 'a', 'd', 'd', 'd', 'd', 'd'],
                                   'to':  ['b','c','d','e','c0','a0','a1','a2','d0','d1','d2','d3','d4'],
                                   'fm_i':[1,  1,  1,  1,  1,   1,   1,   2,   1,   2,   3,   4,   5]})
        self.df_pl_b5 = pl.DataFrame(self.df_b5)
        self.relates_b5 = [('fm','to')]
        self.pos_b5  = {'a':  [ 0.74, -0.38], 'a0': [ 0.98, -0.54], 'a1': [ 0.75, -0.65], 'a2': [ 1.  , -0.28],
                        'b':  [ 0.38, -0.16], 'c':  [-0.00,  0.06], 'c0': [ 0.18,  0.24], 'd':  [-0.47,  0.21],
                        'd0': [-0.73,  0.39], 'd1': [-0.69,  0.00], 'd2': [-0.78 , 0.19], 'd3': [-0.33,  0.45],
                        'd4': [-0.47, -0.05], 'e':  [-0.55,  0.51]}
        self._labels_b5_ = {'a':['More','Even More','Most'],
                            'c':'Additional'}

        self.params_b5 = {'relationships':self.relates_b5, 'pos':self.pos_b5, 'x_ins':20, 'y_ins':20, 'w':384, 'h':384, 'draw_labels':True}

        # Batch 6 (shares a little with batch 5)
        self.df_b6 = pd.DataFrame({'fm':['a','b'],
                                'to':['b','a'],
                                'txt':['This is a longer text string describing node "a".  It\'s meant to be much longer and therefore to be split by the labeler.',
                                        'This is another long string.\nThis time there\'s carriage returns.\nTo see if the lines gets separated properly.\nHope it works.\n'],
                                'txt2':['ABC','DEF']})
        self.df_pl_b6 = pl.DataFrame(self.df_b6)
        self.params_b6 = {'relationships':self.relates_b5, 'pos':{'a':[0,1], 'b':[1,0]}, 'bounds_percent':0.3, 'w':384, 'h':384, 'draw_labels':True}

        # Batch 7
        fms, tos, timestamps = [], [], []
        for x in [datetime.datetime(2010,1,  1) + datetime.timedelta(days=x) for x in range(4)]:
            fms.append('a'), tos.append('b'), timestamps.append(x)
        for x in [datetime.datetime(2010,1,  5) + datetime.timedelta(days=x) for x in range(8)]:
            fms.append('b'), tos.append('a'), timestamps.append(x)
        for x in [datetime.datetime(2010,1, 13) + datetime.timedelta(days=x) for x in range(3)]:
            fms.append('a'), tos.append('b'), timestamps.append(x)
        for x in [datetime.datetime(2010,1, 1) + datetime.timedelta(days=3*x) for x in range(5)]:
            fms.append('a'), tos.append('c'), timestamps.append(x)
        for x in [datetime.datetime(2010,1, 8, 0, 0)  + datetime.timedelta(hours=x) for x in range(64)]:
            fms.append('c'), tos.append('d'), timestamps.append(x)
        for x in [datetime.datetime(2010,1, 6, 0, 0)  + datetime.timedelta(hours=x) for x in range(64)]:
            fms.append('d'), tos.append('c'), timestamps.append(x)

        self.df_b7    = pd.DataFrame({'fm':fms, 'to':tos, 'ts':timestamps})
        self.df_pl_b7 = pl.DataFrame(self.df_b7)

        self.relates_b7 = [('fm','to')]
        self.pos_b7     = {'a': (0,0), 'b':(1,0), 'c':(1,1), 'd':(0,1)}
        self.params_b7  = {'relationships':self.relates_b7, 'pos':self.pos_b7, 'ts_field':'ts', 'timing_marks':True, 'x_ins':32, 'y_ins':32}

        # Batch 8
        self.df_b8 = pd.DataFrame({'fm':  ['a','a','a',  'b','b','b',  'c','c','c'],
                                   'fm2': [ 1,  1,  2,    1,  2,  3,    1,  1,  2],
                                   'to':  ['a','b','c',  'a','b','c',  'a','b','c'],
                                   'to2': [ 2,  1,  2,    1,  1,  1,    2,  3,  1]})
        self.my_pos_b8 = {'b|2':(10,0),          'b|1':(10,5),          'a|1':(8, 5),
                          'a|2':(5, 5),          'c|2':(3, 3),          'c|1':(0, 5),
                          'b|3':(0, 0)}

        # Batch 9
        self.df_b9 = pd.DataFrame({'fm':['a','a','a', 'b','b','b', 'c','c','c'],
                                    'to':['b','b','c', 'a','a','d', 'b','b','a'],
                                    'x': [ 1,  2,  1,   1,  2,  3,   1,  1,  1],
                                    'y': [ 1,  2,  2,   1,  2,  3,   3,  3,  3]})
        self.df_pl_b9  = pl.DataFrame(self.df_b9)
        self.pos_b9    = {'a':(0,0), 'b':(1,0), 'c':(1,1), 'd':(0.75, 0.25)}
        self.params_b9 = {'relationships':[('fm','to')], 'pos':self.pos_b9, 'bounds_percent':0.2, 'link_shape':'curve', 'sm_type':'xy',
                          'sm_params':{'x_field':'x', 'y_field':'y', 'draw_border':False, 'dot_size':'large'}, 'w':384, 'h':384}




    def test_simple(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b1, self.pos_b1).renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b1, self.pos_b1).renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b1, self.pos_b1).renderSVG()

    def test_simpleSizes(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b1, self.pos_b1, count_by='ct', link_size='vary',  node_size='vary').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b1, self.pos_b1, count_by='ct', link_size='vary',  node_size='vary').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b1, self.pos_b1, count_by='ct', link_size='vary',  node_size='vary').renderSVG()

    def test_simpleSizesAndColors(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  link_color='vary', link_size_min=3, link_size_max=5, node_size='vary').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  link_color='vary', link_size_min=3, link_size_max=5, node_size='vary').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  link_color='vary', link_size_min=3, link_size_max=5, node_size='vary').renderSVG()

    def test_simpleColorsAndFixedSizes(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b1, self.pos_b1, color_by='co', link_size='large', link_color='vary', node_size=None).renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size='large', link_color='vary', node_size=None).renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size='large', link_color='vary', node_size=None).renderSVG()

    def test_simpleColorsAndNoLinks(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b1, self.pos_b1, color_by='co', link_size=None,    node_size='medium').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size=None,    node_size='medium').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size=None,    node_size='medium').renderSVG()

    def test_simpleColorsAndNoLinksAndSquareNodes(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b1, self.pos_b1, color_by='co', link_size=None,    node_size='small',  node_shape='square').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size=None,    node_size='small',  node_shape='square').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size=None,    node_size='small',  node_shape='square').renderSVG()

    def test_simpleColorsAndCurvedLinks(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  node_size='medium', link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  node_size='medium', link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  node_size='medium', link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary').renderSVG()

    def test_simpleColorsAndCurvedLinksAndArrow(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  node_size='small',  link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary', link_arrow=True).renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  node_size='small',  link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary', link_arrow=True).renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b1, self.pos_b1, color_by='co', link_size='vary',  node_size='small',  link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary', link_arrow=True).renderSVG()

    def test_simpleIntegers(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b2, self.pos_b2).renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b2, self.pos_b2).renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b2, self.pos_i_b2).renderSVG()

    def test_simpleIntegersSizes(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b2, self.pos_b2,   count_by='ct', link_size='vary',  node_size='vary').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b2, self.pos_b2,   count_by='ct', link_size='vary',  node_size='vary').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b2, self.pos_i_b2, count_by='ct', link_size='vary',  node_size='vary').renderSVG()

    def test_simpleIntegersColors(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b2, self.pos_b2,   color_by='co', link_size='vary',  link_color='vary', link_size_min=3, link_size_max=5, node_size='vary').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b2, self.pos_b2,   color_by='co', link_size='vary',  link_color='vary', link_size_min=3, link_size_max=5, node_size='vary').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b2, self.pos_i_b2, color_by='co', link_size='vary',  link_color='vary', link_size_min=3, link_size_max=5, node_size='vary').renderSVG()

    def test_simpleIntegersColorsAndSizes(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b2, self.pos_b2,   color_by='co', link_size='large', link_color='vary', node_size=None).renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b2, self.pos_b2,   color_by='co', link_size='large', link_color='vary', node_size=None).renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b2, self.pos_i_b2, color_by='co', link_size='large', link_color='vary', node_size=None).renderSVG()

    def test_simpleIntegersColorsAndHiddenLinks(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b2, self.pos_b2,   color_by='co', link_size=None,    node_size='medium').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b2, self.pos_b2,   color_by='co', link_size=None,    node_size='medium').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b2, self.pos_i_b2, color_by='co', link_size=None,    node_size='medium').renderSVG()

    def test_simpleIntegersHiddenLinksAndNodeShapes(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b2, self.pos_b2,   color_by='co', link_size=None,    node_size='small',  node_shape='square').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b2, self.pos_b2,   color_by='co', link_size=None,    node_size='small',  node_shape='square').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b2, self.pos_i_b2, color_by='co', link_size=None,    node_size='small',  node_shape='square').renderSVG()


    def test_simpleIntegersCurvedLinks(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b2, self.pos_b2,   color_by='co', link_size='vary',  node_size='medium', link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary').renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b2, self.pos_b2,   color_by='co', link_size='vary',  node_size='medium', link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary').renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b2, self.pos_i_b2, color_by='co', link_size='vary',  node_size='medium', link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary').renderSVG()

    def test_simpleIntegersCurvedLinksAndArrow(self):
        self.rt_self.linkNode(self.df_b1,    self.relates_b2, self.pos_b2,   color_by='co', link_size='vary',  node_size='small',  link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary', link_arrow=True).renderSVG()
        self.rt_self.linkNode(self.df_pl_b1, self.relates_b2, self.pos_b2,   color_by='co', link_size='vary',  node_size='small',  link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary', link_arrow=True).renderSVG()
        self.rt_self.link    (self.df_pl_b1, self.relates_b2, self.pos_i_b2, color_by='co', link_size='vary',  node_size='small',  link_shape='curve',  link_size_min=3, link_size_max=5, link_color='vary', link_arrow=True).renderSVG()

    def test_concatFrames(self):
        self.rt_self.linkNode(self.df_b3,    self.relates_b3, self.pos_b3).renderSVG()
        self.rt_self.linkNode(self.df_pl_b3, self.relates_b3, self.pos_b3).renderSVG()
        # link doesn't support mixed types

    def test_concatFramesWithLabelOnly(self):
        self.rt_self.linkNode(self.df_b3,    self.relates_b3, self.pos_b3, label_only=set(['1','2','3','4','5','6'])).renderSVG()
        self.rt_self.linkNode(self.df_pl_b3, self.relates_b3, self.pos_b3, label_only=set(['1','2','3','4','5','6'])).renderSVG()
        # link doesn't support mixed types

    def test_concatFramesWithConvexHull(self):
        _lu_ = {'[a123]':'#ff0000', '[d456]':'#0000ff'}
        self.rt_self.linkNode(self.df_b3,    self.relates_b3, self.pos_b3, label_only=set(['1','2','3','4','5','6','a','d']), convex_hull_lu=_lu_).renderSVG()
        self.rt_self.linkNode(self.df_pl_b3, self.relates_b3, self.pos_b3, label_only=set(['1','2','3','4','5','6','a','d']), convex_hull_lu=_lu_).renderSVG()
        # link doesn't support mixed types

    def test_multipleRelationships(self):
        self.rt_self.linkNode(self.df1_b4,       self.multi_relates_b4, self.pos_b4, count_by='count', 
                              node_size='vary', node_color='#ff0000', node_shape=self.node_shapes_b4,
                              link_size='medium', link_dash="4 3").renderSVG()
        self.rt_self.linkNode(self.df1_pl_b4, self.multi_relates_b4, self.pos_b4, count_by='count', 
                              node_size='vary', node_color='#ff0000', node_shape=self.node_shapes_b4,
                              link_size='medium', link_dash="4 3").renderSVG()
        # link doesn't support
        
    def test_multipleRelationships2(self):
        self.rt_self.linkNode(self.df1_b4,       self.multi_relates_b4, self.pos_b4, count_by='count', 
                              node_size='vary', node_color='#ff0000', node_shape=self.node_shapes_b4,
                              link_size='medium', link_dash={('fm','to'):"2 4"}).renderSVG()
        self.rt_self.linkNode(self.df1_pl_b4, self.multi_relates_b4, self.pos_b4, count_by='count', 
                              node_size='vary', node_color='#ff0000', node_shape=self.node_shapes_b4,
                              link_size='medium', link_dash={('fm','to'):"2 4"}).renderSVG()
        # link doesn't support

    def test_multipleRelationshipsWithVariableLinkSizes(self):
        self.rt_self.linkNode(self.df1_b4, self.multi_relates_b4, self.pos_b4, count_by='count', 
                              node_size='vary', node_color='#ff0000', node_shape=self.node_shapes_b4,
                              link_size={('fm','to'):'large', ('src','dst'):'nil'}, link_dash={('fm','to'):"5 5"}).renderSVG()
        self.rt_self.linkNode(self.df1_pl_b4, self.multi_relates_b4, self.pos_b4, count_by='count', 
                              node_size='vary', node_color='#ff0000', node_shape=self.node_shapes_b4,
                              link_size={('fm','to'):'large', ('src','dst'):'nil'}, link_dash={('fm','to'):"5 5"}).renderSVG()
        # link doesn't support

    def test_multipleRelationshipsWithArrowFalse(self):
        self.rt_self.linkNode(self.df1_b4,    self.multi_relates_b4, self.pos_b4, count_by='count', link_arrow=False, link_size=10).renderSVG()
        self.rt_self.linkNode(self.df1_pl_b4, self.multi_relates_b4, self.pos_b4, count_by='count', link_arrow=False, link_size=10).renderSVG()
        self.rt_self.link    (self.df1_pl_b4, self.multi_relates_b4, self.pos_b4, count_by='count', link_arrow=False, link_size=10).renderSVG()

    def test_multipleRelationshipsWithColorLookups(self):
        self.rt_self.linkNode(self.df1_b4,            self.multi_relates_b4, self.pos_b4, count_by='count', 
                              color_by='src', node_color={'a':'#00ff00', 2:'#ff0000'}, node_size=15).renderSVG() # Not Correct 2023-09-24
        self.rt_self.linkNode(self.df1_pl_b4,         self.multi_relates_b4, self.pos_b4, count_by='count', 
                              color_by='src', node_color={'a':'#00ff00', 2:'#ff0000'}, node_size=15).renderSVG() # Not Correct 2023-09-24
        self.rt_self.link    (self.df1_pl_b4,         self.multi_relates_b4, self.pos_b4, count_by='count', 
                              color_by='src', node_color={'a':'#00ff00', 2:'#ff0000'}, node_size=15).renderSVG() # Not Correct 2023-09-24

    def test_labels(self):
        self.rt_self.linkNode(self.df_b5,    node_labels=self._labels_b5_, **self.params_b5).renderSVG()
        self.rt_self.linkNode(self.df_pl_b5, node_labels=self._labels_b5_, **self.params_b5).renderSVG()

    def test_nodeLabeler(self):
        self.rt_self.linkNode(self.df_b5,    node_labels=self.rt_self.nodeLabeler(self.df_b5,    'fm', 'fm_i'), **self.params_b5).renderSVG()
        self.rt_self.linkNode(self.df_pl_b5, node_labels=self.rt_self.nodeLabeler(self.df_pl_b5, 'fm', 'fm_i'), **self.params_b5).renderSVG()

    def test_nodeLabeler2(self):
        self.rt_self.linkNode(self.df_b6,    node_labels=self.rt_self.nodeLabeler(self.df_b6,    'fm', 'txt'), **self.params_b6).renderSVG()
        self.rt_self.linkNode(self.df_pl_b6, node_labels=self.rt_self.nodeLabeler(self.df_pl_b6, 'fm', 'txt'), **self.params_b6).renderSVG()

    def test_nodeLabeler3(self):
        self.rt_self.linkNode(self.df_b6,    node_labels=self.rt_self.nodeLabeler(self.df_b6,    'fm', 'txt', max_lines=6, max_line_len=24), **self.params_b6).renderSVG()
        self.rt_self.linkNode(self.df_pl_b6, node_labels=self.rt_self.nodeLabeler(self.df_pl_b6, 'fm', 'txt', max_lines=6, max_line_len=24), **self.params_b6).renderSVG()

    def test_nodeLabeler4(self):
        self.rt_self.linkNode(self.df_b6,    node_labels=self.rt_self.nodeLabeler(self.df_b6,    'fm', 'txt2', node_labels=self.rt_self.nodeLabeler(self.df_b6,    'fm', 'txt')), **self.params_b6).renderSVG()
        self.rt_self.linkNode(self.df_pl_b6, node_labels=self.rt_self.nodeLabeler(self.df_pl_b6, 'fm', 'txt2', node_labels=self.rt_self.nodeLabeler(self.df_pl_b6, 'fm', 'txt')), **self.params_b6).renderSVG()

    def test_nodeLabeler5(self):
        self.rt_self.linkNode(self.df_b6,    node_labels_only=True, node_labels=self.rt_self.nodeLabeler(self.df_b6,    'fm', 'txt2', node_labels=self.rt_self.nodeLabeler(self.df_b6,    'fm', 'txt')), **self.params_b6).renderSVG()
        self.rt_self.linkNode(self.df_pl_b6, node_labels_only=True, node_labels=self.rt_self.nodeLabeler(self.df_pl_b6, 'fm', 'txt2', node_labels=self.rt_self.nodeLabeler(self.df_pl_b6, 'fm', 'txt')), **self.params_b6).renderSVG()

    def test_timingMarks(self):
        self.params_b7['link_shape'] = None
        self.rt_self.linkNode(self.df_b7,    **self.params_b7).renderSVG()
        self.rt_self.linkNode(self.df_pl_b7, **self.params_b7).renderSVG()
        self.rt_self.link    (self.df_pl_b7, **self.params_b7).renderSVG()

    def test_timingMarksCurvedLinks(self):
        self.params_b7['link_shape'] = 'curve'
        self.rt_self.linkNode(self.df_b7,    **self.params_b7).renderSVG()
        self.rt_self.linkNode(self.df_pl_b7, **self.params_b7).renderSVG()
        self.rt_self.link    (self.df_pl_b7, **self.params_b7).renderSVG()

    def test_confirmTimingMarksWithXY(self):
        self.rt_self.xy(self.df_b7,    x_field='ts', y_field=['fm','to']).renderSVG()
        self.rt_self.xy(self.df_pl_b7, x_field='ts', y_field=['fm','to']).renderSVG()

    def test_concatenatedNodes(self):
        self.rt_self.linkNode(self.df_b8, [(('fm','fm2'),('to','to2'))], self.my_pos_b8, link_shape='curve', bounds_percent=0.2).renderSVG()
        self.rt_self.linkNode(pl.DataFrame(self.df_b8), [(('fm','fm2'),('to','to2'))], self.my_pos_b8, link_shape='curve', bounds_percent=0.2).renderSVG()
        self.rt_self.link    (pl.DataFrame(self.df_b8), [(('fm','fm2'),('to','to2'))], self.my_pos_b8, link_shape='curve', bounds_percent=0.2).renderSVG()

    def test_smallMultiples(self):
        self.rt_self.linkNode(self.df_b9,    node_opacity=0.4, **self.params_b9).renderSVG()
        self.rt_self.linkNode(self.df_pl_b9, node_opacity=0.4, **self.params_b9).renderSVG()
        self.rt_self.link    (self.df_pl_b9, node_opacity=0.4, **self.params_b9).renderSVG()

    def test_smallMultiplesWithLinkMode(self):
        self.rt_self.linkNode(self.df_b9,    sm_mode='link', **self.params_b9).renderSVG()
        self.rt_self.linkNode(self.df_pl_b9, sm_mode='link', **self.params_b9).renderSVG()
        self.rt_self.link    (self.df_pl_b9, sm_mode='link', **self.params_b9).renderSVG()

    def test_linkLabeling(self):
        df    = pd.DataFrame({'subject':['dog',  'dog',  'dog',   'cat'], 
                              'verb':   ['ran',  'walk', 'fetch', 'bit the hand of the (because cats suck)'], 
                              'object': ['home', 'home', 'stick', 'owner']})
        df_pl = pl.DataFrame(df)
        pos    = {'dog':(0,0), 'home':(1,0), 'stick':(1,1), 'cat':(0,1), 'owner':(0.5,1)}
        params = {'df':df, 'relationships':[('subject','object')], 'pos':pos, 'color_by':'verb','link_color':'vary', 'link_labels':True, 'bounds_percent':0.2, 'draw_labels':True}

        self.rt_self.linkNode(**params).renderSVG()
        self.rt_self.linkNode(**params, link_size=6).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','dog','stick'}).renderSVG()

        params['df'] = df_pl
        self.rt_self.linkNode(**params).renderSVG()
        self.rt_self.linkNode(**params, link_size=6).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','dog','stick'}).renderSVG()
        self.rt_self.link(**params).renderSVG()
        self.rt_self.link(**params, link_size=6).renderSVG()
        self.rt_self.link(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.link(**params, label_only={'fetch','dog','stick'}).renderSVG()

        params = {'df':df, 'relationships':[('subject', 'object', 'verb')], 'pos':pos, 'color_by':'verb', 'link_labels':True, 'bounds_percent':0.2, 'draw_labels':True}
        self.rt_self.linkNode(**params).renderSVG()
        self.rt_self.linkNode(**params, link_size=6).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','dog','stick'}).renderSVG()

        params['df'] = df_pl
        self.rt_self.linkNode(**params).renderSVG()
        self.rt_self.linkNode(**params, link_size=6).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','dog','stick'}).renderSVG()
        self.rt_self.link(**params).renderSVG()
        self.rt_self.link(**params, link_size=6).renderSVG()
        self.rt_self.link(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.link(**params, label_only={'fetch','dog','stick'}).renderSVG()

        params = {'df':df, 'relationships':[('subject', 'object', 'verb')], 'pos':pos, 'color_by':'subject', 'link_labels':True, 'bounds_percent':0.2, 'draw_labels':True}
        self.rt_self.linkNode(**params).renderSVG()
        self.rt_self.linkNode(**params, link_size=6).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','dog','stick'}).renderSVG()

        params['df'] = df_pl
        self.rt_self.linkNode(**params).renderSVG()
        self.rt_self.linkNode(**params, link_size=6).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','dog','stick'}).renderSVG()
        self.rt_self.link(**params).renderSVG()
        self.rt_self.link(**params, link_size=6).renderSVG()
        self.rt_self.link(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.link(**params, label_only={'fetch','dog','stick'}).renderSVG()

        params = {'df':df, 'relationships':[('subject', 'object', 'verb')], 'pos':pos,'link_labels':True, 'bounds_percent':0.2, 'draw_labels':True}
        self.rt_self.linkNode(**params).renderSVG()
        self.rt_self.linkNode(**params, link_size=6).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','dog','stick'}).renderSVG()

        params['df'] = df_pl
        self.rt_self.linkNode(**params).renderSVG()
        self.rt_self.linkNode(**params, link_size=6).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.linkNode(**params, label_only={'fetch','dog','stick'}).renderSVG()
        self.rt_self.link(**params).renderSVG()
        self.rt_self.link(**params, link_size=6).renderSVG()
        self.rt_self.link(**params, label_only={'fetch','walk'}).renderSVG()
        self.rt_self.link(**params, label_only={'fetch','dog','stick'}).renderSVG()

    def test_convexHullsAsLists(self):
        lu = {'fm':'a b c d e f g h i j'.split(), 
              'to':'b c a e f d h i j g'.split()}
        df      = pl.DataFrame(lu)
        relates = [('fm','to')]
        pos = {"a":(0.58, 0.40), "e":(0.07, 0.31), "i":(0.25, 0.19), "d":(0.25, 0.40), "b":(0.43, 0.40),
               "c":(0.58, 0.31), "g":(0.07, 0.07), "f":(0.07, 0.40), "j":(0.07, 0.19), "h":(0.25, 0.07) }
        convex_hull_lu = {'abc':['a','b','c'], 'def':['d','e','f','x','y','z'], 'nothere':['m','n','o'], 'h_only':['h'], 'empty':[], 'as_set':set(['g','i','j'])}
        self.rt_self.linkNode(df, relates, pos, bounds_percent=0.2, node_size='small', convex_hull_lu=convex_hull_lu).renderSVG()
        self.rt_self.link    (df, relates, pos, bounds_percent=0.2, node_size='small', convex_hull_lu=convex_hull_lu).renderSVG()
        self.rt_self.linkNode(df, relates, pos, bounds_percent=0.2, node_size='small', convex_hull_lu=convex_hull_lu, convex_hull_labels=True).renderSVG()
        self.rt_self.link    (df, relates, pos, bounds_percent=0.2, node_size='small', convex_hull_lu=convex_hull_lu, convex_hull_labels=True).renderSVG()

if __name__ == '__main__':
    unittest.main()
