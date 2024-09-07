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

from rtsvg import *

class Testrt_chord_diagram_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

    #
    # These test are from the test scripts directory (rt_test_chord_diagram.py)
    #
    def test_ch1_batch_1(self):
        _node_gap_ = 40

        df = pd.DataFrame({'fm':['a'], 'to':['b']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()
        df = pd.DataFrame({'fm':['a','a'], 'to':['b','c']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()
        df = pd.DataFrame({'fm':['a','a','a'], 'to':['b','c','d']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()

        df = pd.DataFrame({'fm':['a','c'], 'to':['b','d']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()
        df = pd.DataFrame({'fm':['a','a','d','d'], 'to':['b','c','e','f']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()
        df = pd.DataFrame({'fm':['a','a','a','x','x','x','x'], 'to':['b','c','d','r','s','t','u']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()

        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_, node_color='#ff0000', link_color='#ff0000').renderSVG()
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_, link_arrow=None).renderSVG()
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_, link_arrow='sharp').renderSVG()

    def test_ch1_batch_2(self):
        df = pd.DataFrame({'fm':['a',  'a',  'a',  'a',  'b',  'b',  'b',  'c',  'c',  'd',  'd',  'd', 'd'],
                        'to':['b',  'c',  'd',  'b',  'a',  'b',  'c',  'a',  'b',  'c',  'a',  'b', 'd'],
                        'ct':[10,   20,   5,    1,    20,   3,    5,    10,   15,   5,    10,   50,  20]})
        self.rt_self.dendrogramOrdering(df, 'fm', 'to', None, False)
        self.rt_self.dendrogramOrdering(df, 'fm', 'to', 'ct', False)
        self.rt_self.dendrogramOrdering(df, 'fm', 'to', 'ct', True)
        params = {'df':df, 'relationships':[('fm','to')]}
        self.rt_self.chordDiagram(**params).renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct', count_by_set=True).renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct', count_by_set=True).renderSVG()

    def test_ch1_batch_3(self):
        df2 = pd.DataFrame({'x':['a',  'b',   'c',  'd'],
                            'y':['b',  'a',   'd',  'c'],
                            'z':[200,  200,   5,    8]})
        params = {'df':df2, 'relationships':[('x','y')]}
        self.rt_self.chordDiagram(**params, count_by='z', node_h=10).renderSVG()
        df3 = pd.DataFrame({'sip':['1.2.3.4',  '1.2.3.4',   '5.6.7.8',  '1.1.1.1'],
                            'dpt':[80,         80,          443,        443],
                            'dip':['5.6.7.8',  '1.1.1.1',   '1.1.1.1',  '5.6.7.8'],
                            'pkt':[200,  200,   5,    8]})
        params = {'df':df3, 'relationships':[('sip',['dip','dpt'])]}
        self.rt_self.chordDiagram(**params, count_by='pkt', node_h=10).renderSVG()
        df4 = pd.DataFrame({'x':['a',  'b',   'a',  'b', 'c'],
                            'y':['a',  'a',   'b',  'b', 'c'],
                            'z':[200,  200,   5,    8,   400]})
        params = {'df':df4, 'relationships':[('x','y')]}
        self.rt_self.chordDiagram(**params, count_by='z', node_h=10).renderSVG()

    def test_ch1_batch_4(self):
        df5 = pd.DataFrame({'sip':['1.2.3.4',  '1.2.3.4',   '5.6.7.8',  '1.1.1.1'],
                            'dpt':[80,         80,          443,        443],
                            'dip':['5.6.7.8',  '1.1.1.1',   '1.1.1.1',  '5.6.7.8'],
                            'pkt':[200,  200,   5,    8]})
        params = {'df':df5, 'relationships':[('sip',['dip','dpt'])], 'w':128, 'h':128, 'node_h':5, 'count_by':'pkt' }
        self.rt_self.chordDiagram(**params, color_by=None).renderSVG()
        self.rt_self.chordDiagram(**params, color_by='sip', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='dpt', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='dip', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='pkt', link_color='vary').renderSVG()

    def test_ch1_batch_5(self):
        df = pd.DataFrame({'fm':['a','a','a',        'x','x','x','x',              'm', 'n',          'o'], 
                        'to':['b','c','d',        'r','s','t','u',              'n', 'm',          'p'],
                        'co':['red','red','red',  'blue','blue','blue','blue',  'black','black',   'green']})

        params = {'df':df, 'relationships':[('fm','to')], 'w':192, 'h':192, 'node_h':5, 'link_opacity':0.2}
        self.rt_self.chordDiagram(**params, color_by=None, node_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='fm', node_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='to', node_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='co', node_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by=None, node_color='vary', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='fm', node_color='vary', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='to', node_color='vary', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='co', node_color='vary', link_color='vary').renderSVG()

    def test_ch1_batch_6(self):
        df = pd.DataFrame({'fm': ['a','a','a','a','a',  'b','b','b',    'c','c','c','c'],
                           'to': ['b','c','d','b','b',  'a','c','d',    'a','b','d','e'],
                           'opt':[ 1,  1,  1,  2,  3,    1,  2,  3,      1,  2,  3,  4]})
        params = {'relationships':[('fm','to')]}
        _struct_ = self.rt_self.chordDiagram(df, **params)
        _struct_.renderSVG()
        _op1_ = self.rt_self.chordDiagram(df.query('opt == 1'), **params)
        _op1_.applyViewConfiguration(_struct_)
        _op1_.renderSVG()
        _op2_ = self.rt_self.chordDiagram(df.query('opt == 2'), **params)
        _op2_.applyViewConfiguration(_struct_)
        _op2_.renderSVG()
        _op3_ = self.rt_self.chordDiagram(df.query('opt == 3'), **params)
        _op3_.applyViewConfiguration(_struct_)
        _op3_.renderSVG()
        _op4_ = self.rt_self.chordDiagram(df.query('opt == 4'), **params)
        _op4_.applyViewConfiguration(_struct_)
        _op4_.renderSVG()        

        self.rt_self.chordDiagram(df.query('opt == 1'), **params, structure_template=_struct_).renderSVG()
        self.rt_self.chordDiagram(df.query('opt == 2'), **params, structure_template=_struct_).renderSVG()
        self.rt_self.chordDiagram(df.query('opt == 3'), **params, structure_template=_struct_).renderSVG()
        self.rt_self.chordDiagram(df.query('opt == 4'), **params, structure_template=_struct_).renderSVG()

        self.rt_self.smallMultiples(df, category_by='opt', sm_type='chordDiagram', sm_params={'relationships':[('fm','to')]}, w=300, h=300).renderSVG()

        self.rt_self.smallMultiples(df, category_by='opt', sm_type='chordDiagram', sm_params={'relationships':[('fm','to')]}, w=300, h=300, x_axis_independent=False).renderSVG()
        self.rt_self.smallMultiples(df, category_by='opt', sm_type='chordDiagram', sm_params={'relationships':[('fm','to')], 'link_style':'wide'},  w=300, h=300, x_axis_independent=False).renderSVG()

        params = {'df':df, 'relationships':[('fm','to')], 'node_h':5}

        self.rt_self.chordDiagram(**params, w=200, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=160, h=160).renderSVG()
        self.rt_self.chordDiagram(**params, w=140, h=140).renderSVG()
        self.rt_self.chordDiagram(**params, w=120, h=120).renderSVG()
        self.rt_self.chordDiagram(**params, w=100, h=100).renderSVG()
        self.rt_self.chordDiagram(**params, w= 80, h= 80).renderSVG()

        self.rt_self.chordDiagram(**params, w=200, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h=160).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h=140).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h=120).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h=100).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h= 80).renderSVG()

        self.rt_self.chordDiagram(**params, w=200, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=160, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=140, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=120, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=100, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w= 80, h=200).renderSVG()

        self.rt_self.chordDiagram(**params).renderSVG()
        self.rt_self.chordDiagram(txt_h=16, **params).renderSVG()
        self.rt_self.chordDiagram(txt_h=16, label_only={'a','d'}, **params).renderSVG()


    def test_ch1_batch_7(self):
        df = pd.DataFrame({'fm':['a',  'a',  'a',  'a',  'b',  'b',  'b',  'c',  'c',  'd',  'd',  'd', 'd'],
                        'to':['b',  'c',  'd',  'b',  'a',  'b',  'c',  'a',  'b',  'c',  'a',  'b', 'd'],
                        'ct':[10,   20,   5,    1,    20,   3,    5,    10,   15,   5,    10,   50,  20]})

        params = {'df':df, 'relationships':[('fm','to')], 'equal_size_nodes':True, 'draw_labels':True}
        self.rt_self.chordDiagram(**params).renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct', count_by_set=True).renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct', count_by_set=True).renderSVG()

        params = {'df':df, 'relationships':[('fm','to')], 'equal_size_nodes':False, 'draw_labels':True}
        self.rt_self.chordDiagram(**params).renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct', count_by_set=True).renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct', count_by_set=True).renderSVG()


    #
    # These test are from the test scripts directory (rt_test_chord_diagram_2.py)
    # - identical to above but with polars
    #
    def test_ch2_batch_1(self):
        _node_gap_ = 40

        df = pl.DataFrame({'fm':['a'], 'to':['b']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()
        df = pl.DataFrame({'fm':['a','a'], 'to':['b','c']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()
        df = pl.DataFrame({'fm':['a','a','a'], 'to':['b','c','d']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()

        df = pl.DataFrame({'fm':['a','c'], 'to':['b','d']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()
        df = pl.DataFrame({'fm':['a','a','d','d'], 'to':['b','c','e','f']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()
        df = pl.DataFrame({'fm':['a','a','a','x','x','x','x'], 'to':['b','c','d','r','s','t','u']})
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_).renderSVG()

        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_, node_color='#ff0000', link_color='#ff0000').renderSVG()
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_, link_arrow=None).renderSVG()
        self.rt_self.chordDiagram(df, [('fm','to')], node_gap=_node_gap_, link_arrow='sharp').renderSVG()

    def test_ch2_batch_2(self):
        df = pl.DataFrame({'fm':['a',  'a',  'a',  'a',  'b',  'b',  'b',  'c',  'c',  'd',  'd',  'd', 'd'],
                        'to':['b',  'c',  'd',  'b',  'a',  'b',  'c',  'a',  'b',  'c',  'a',  'b', 'd'],
                        'ct':[10,   20,   5,    1,    20,   3,    5,    10,   15,   5,    10,   50,  20]})
        self.rt_self.dendrogramOrdering(df, 'fm', 'to', None, False)
        self.rt_self.dendrogramOrdering(df, 'fm', 'to', 'ct', False)
        self.rt_self.dendrogramOrdering(df, 'fm', 'to', 'ct', True)
        params = {'df':df, 'relationships':[('fm','to')]}
        self.rt_self.chordDiagram(**params).renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct', count_by_set=True).renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct', count_by_set=True).renderSVG()

    def test_ch2_batch_3(self):
        df2 = pl.DataFrame({'x':['a',  'b',   'c',  'd'],
                            'y':['b',  'a',   'd',  'c'],
                            'z':[200,  200,   5,    8]})
        params = {'df':df2, 'relationships':[('x','y')]}
        self.rt_self.chordDiagram(**params, count_by='z', node_h=10).renderSVG()
        df3 = pl.DataFrame({'sip':['1.2.3.4',  '1.2.3.4',   '5.6.7.8',  '1.1.1.1'],
                            'dpt':[80,         80,          443,        443],
                            'dip':['5.6.7.8',  '1.1.1.1',   '1.1.1.1',  '5.6.7.8'],
                            'pkt':[200,  200,   5,    8]})
        params = {'df':df3, 'relationships':[('sip',['dip','dpt'])]}
        self.rt_self.chordDiagram(**params, count_by='pkt', node_h=10).renderSVG()
        df4 = pl.DataFrame({'x':['a',  'b',   'a',  'b', 'c'],
                            'y':['a',  'a',   'b',  'b', 'c'],
                            'z':[200,  200,   5,    8,   400]})
        params = {'df':df4, 'relationships':[('x','y')]}
        self.rt_self.chordDiagram(**params, count_by='z', node_h=10).renderSVG()

    def test_ch2_batch_4(self):
        df5 = pl.DataFrame({'sip':['1.2.3.4',  '1.2.3.4',   '5.6.7.8',  '1.1.1.1'],
                            'dpt':[80,         80,          443,        443],
                            'dip':['5.6.7.8',  '1.1.1.1',   '1.1.1.1',  '5.6.7.8'],
                            'pkt':[200,  200,   5,    8]})
        params = {'df':df5, 'relationships':[('sip',['dip','dpt'])], 'w':128, 'h':128, 'node_h':5, 'count_by':'pkt' }
        self.rt_self.chordDiagram(**params, color_by=None).renderSVG()
        self.rt_self.chordDiagram(**params, color_by='sip', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='dpt', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='dip', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='pkt', link_color='vary').renderSVG()

    def test_ch2_batch_5(self):
        df = pl.DataFrame({'fm':['a','a','a',        'x','x','x','x',              'm', 'n',          'o'], 
                        'to':['b','c','d',        'r','s','t','u',              'n', 'm',          'p'],
                        'co':['red','red','red',  'blue','blue','blue','blue',  'black','black',   'green']})

        params = {'df':df, 'relationships':[('fm','to')], 'w':192, 'h':192, 'node_h':5, 'link_opacity':0.2}
        self.rt_self.chordDiagram(**params, color_by=None, node_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='fm', node_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='to', node_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='co', node_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by=None, node_color='vary', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='fm', node_color='vary', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='to', node_color='vary', link_color='vary').renderSVG()
        self.rt_self.chordDiagram(**params, color_by='co', node_color='vary', link_color='vary').renderSVG()

    def test_ch2_batch_6(self):
        df = pl.DataFrame({'fm': ['a','a','a','a','a',  'b','b','b',    'c','c','c','c'],
                           'to': ['b','c','d','b','b',  'a','c','d',    'a','b','d','e'],
                           'opt':[ 1,  1,  1,  2,  3,    1,  2,  3,      1,  2,  3,  4]})
        params = {'relationships':[('fm','to')]}
        _struct_ = self.rt_self.chordDiagram(df, **params)
        _struct_.renderSVG()
        _op1_ = self.rt_self.chordDiagram(df.filter(pl.col('opt') == 1), **params)
        _op1_.applyViewConfiguration(_struct_)
        _op1_.renderSVG()
        _op2_ = self.rt_self.chordDiagram(df.filter(pl.col('opt') == 2), **params)
        _op2_.applyViewConfiguration(_struct_)
        _op2_.renderSVG()
        _op3_ = self.rt_self.chordDiagram(df.filter(pl.col('opt') == 3), **params)
        _op3_.applyViewConfiguration(_struct_)
        _op3_.renderSVG()
        _op4_ = self.rt_self.chordDiagram(df.filter(pl.col('opt') == 4), **params)
        _op4_.applyViewConfiguration(_struct_)
        _op4_.renderSVG()        

        self.rt_self.chordDiagram(df.filter(pl.col('opt') == 1), **params, structure_template=_struct_).renderSVG()
        self.rt_self.chordDiagram(df.filter(pl.col('opt') == 2), **params, structure_template=_struct_).renderSVG()
        self.rt_self.chordDiagram(df.filter(pl.col('opt') == 3), **params, structure_template=_struct_).renderSVG()
        self.rt_self.chordDiagram(df.filter(pl.col('opt') == 4), **params, structure_template=_struct_).renderSVG()

        self.rt_self.smallMultiples(df, category_by='opt', sm_type='chordDiagram', sm_params={'relationships':[('fm','to')]}, w=300, h=300).renderSVG()

        self.rt_self.smallMultiples(df, category_by='opt', sm_type='chordDiagram', sm_params={'relationships':[('fm','to')]}, w=300, h=300, x_axis_independent=False).renderSVG()
        self.rt_self.smallMultiples(df, category_by='opt', sm_type='chordDiagram', sm_params={'relationships':[('fm','to')], 'link_style':'wide'},  w=300, h=300, x_axis_independent=False).renderSVG()

        params = {'df':df, 'relationships':[('fm','to')], 'node_h':5}

        self.rt_self.chordDiagram(**params, w=200, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=160, h=160).renderSVG()
        self.rt_self.chordDiagram(**params, w=140, h=140).renderSVG()
        self.rt_self.chordDiagram(**params, w=120, h=120).renderSVG()
        self.rt_self.chordDiagram(**params, w=100, h=100).renderSVG()
        self.rt_self.chordDiagram(**params, w= 80, h= 80).renderSVG()

        self.rt_self.chordDiagram(**params, w=200, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h=160).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h=140).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h=120).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h=100).renderSVG()
        self.rt_self.chordDiagram(**params, w=200, h= 80).renderSVG()

        self.rt_self.chordDiagram(**params, w=200, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=160, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=140, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=120, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w=100, h=200).renderSVG()
        self.rt_self.chordDiagram(**params, w= 80, h=200).renderSVG()

        self.rt_self.chordDiagram(**params).renderSVG()
        self.rt_self.chordDiagram(txt_h=16, **params).renderSVG()
        self.rt_self.chordDiagram(txt_h=16, label_only={'a','d'}, **params).renderSVG()


    def test_ch2_batch_7(self):
        df = pl.DataFrame({'fm':['a',  'a',  'a',  'a',  'b',  'b',  'b',  'c',  'c',  'd',  'd',  'd', 'd'],
                        'to':['b',  'c',  'd',  'b',  'a',  'b',  'c',  'a',  'b',  'c',  'a',  'b', 'd'],
                        'ct':[10,   20,   5,    1,    20,   3,    5,    10,   15,   5,    10,   50,  20]})

        params = {'df':df, 'relationships':[('fm','to')], 'equal_size_nodes':True, 'draw_labels':True}
        self.rt_self.chordDiagram(**params).renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct', count_by_set=True).renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct', count_by_set=True).renderSVG()

        params = {'df':df, 'relationships':[('fm','to')], 'equal_size_nodes':False, 'draw_labels':True}
        self.rt_self.chordDiagram(**params).renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, count_by='ct', count_by_set=True).renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct').renderSVG()
        self.rt_self.chordDiagram(**params, link_style='wide', count_by='ct', count_by_set=True).renderSVG()


if __name__ == '__main__':
    unittest.main()
