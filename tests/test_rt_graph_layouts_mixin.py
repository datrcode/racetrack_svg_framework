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
import random
import string

from math import sin, cos, sqrt, pi

from rtsvg import *

class Testrt_graph_layouts_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()
        _nodes_, _edges_ = 200, 3000
        _node_list_ = []
        def randomString(n): return ''.join(random.choices(string.ascii_uppercase + string.digits, k=n))
        self.pos    = {}
        for i in range(_nodes_):
            _node_name_ = randomString(5)
            _node_list_.append(_node_name_)
            self.pos[_node_name_] = (random.random()*400, random.random()*100)
        _lu_ = {'fm':[], 'to':[], 'ct':[]}
        for i in range(_edges_):
            _fm_ = random.choice(_node_list_)
            _to_ = random.choice(_node_list_)
            _ct_ = float(random.randint(1,10))
            _lu_['fm'].append(_fm_), _lu_['to'].append(_to_), _lu_['ct'].append(_ct_)
        self.df          = pl.DataFrame(_lu_)
        self.relates     = [('fm','to')]        
        self.g           = self.rt_self.createNetworkXGraph(self.df, self.relates)
        self.node_subset = []
        _already_seen_   = set()
        for i in range(20):
            _node_ = random.choice(_node_list_)
            if _node_ not in _already_seen_:
                _already_seen_.add(_node_)
                self.node_subset.append(_node_)

    def test_distanceDictionary(self):
        # Example graph from page 212 of the Cohen paper
        g = nx.DiGraph()
        g.add_edge('a', 'b', weight=2.0), g.add_edge('b','a', weight=2.0)
        g.add_edge('a', 'c', weight=2.0), g.add_edge('c','a', weight=2.0)
        g.add_edge('b', 'c', weight=1.0), g.add_edge('c','b', weight=1.0)
        g.add_edge('b', 'd', weight=3.0), g.add_edge('d','b', weight=3.0)
        _distances_ = self.rt_self.distanceDictionary(g, distance_metric='resistive')
        self.assertAlmostEqual(_distances_['b']['d'], 1.0/3.0)
        self.assertAlmostEqual(_distances_['b']['c'], 1.0/2.0)
        _distances_ = self.rt_self.distanceDictionary(g, distance_metric='dijkstra')
        self.assertAlmostEqual(_distances_['b']['d'], 3.0)
        self.assertAlmostEqual(_distances_['b']['c'], 1.0)

    def test_positionExtents(self):
        self.rt_self.positionExtents(self.pos, self.g)
        self.rt_self.positionExtents(self.pos)

    def test_calculateLevelSet(self):
        _node_info_, _found_time_ = self.rt_self.calculateLevelSet(self.pos)
        self.rt_self.levelSetSVG(_node_info_, _found_time_)

    def test_rectangularArrangement(self):
        self.rt_self.rectangularArrangement(self.g, self.node_subset)

    def test_sunflowerSeedArrangement(self):
        self.rt_self.sunflowerSeedArrangement(self.g, self.node_subset)

    def test_linearOptimizedArrangement(self):
        self.rt_self.linearOptimizedArrangement(self.g, self.node_subset, self.pos)

    def test_circularOptimizedArrangement(self):
        self.rt_self.circularOptimizedArrangement(self.g, self.node_subset, self.pos)

    def test_circularLayout(self):
        self.rt_self.circularLayout(self.g)
        self.rt_self.circularLayout(self.g, self.node_subset)

    def test_hyperTreeLayout(self):
        self.rt_self.hyperTreeLayout(self.g)
        _roots_, _as_set_ = [], set()
        for x in self.g.nodes():
            _roots_.append(x), _as_set_.add(x)
            break
        self.rt_self.hyperTreeLayout(self.g, roots=_roots_)
        self.rt_self.hyperTreeLayout(self.g, roots=_as_set_)

    def test_circlePackGraphComponentPlacement(self):
        df     = pl.DataFrame({'fm':['a','b','c','d'], 'to':['b','a','d','c']})
        g      = self.rt_self.createNetworkXGraph(df, [('fm','to')])
        pos    = {'a':(0,0), 'b':(1,2), 'c':(5,8), 'd':(7,3)}
        def dist(x,y): return ((pos[x][0] - pos[y][0])**2 + (pos[x][1] - pos[y][1])**2)**0.5
        d0, d1 = dist('a','b'), dist('c','d')
        pos_adj, shapes_gen = self.rt_self.circlePackGraphComponentPlacement(g, pos)
        self.rt_self.link(df, [('fm','to')], pos_adj, bg_shape_lu=shapes_gen)._repr_svg_()

    def test_treeMapGraphComponentPlacement(self):
        self.rt_self.treeMapGraphComponentPlacement(self.g, self.pos)

    def test_springLayout(self):
        self.rt_self.springLayout(self.g)
        self.rt_self.springLayout(self.g, self.pos, selection=self.node_subset)

    def test_barycentricLayout(self):
        self.rt_self.barycentricLayout(self.g, self.pos, selection=self.node_subset)

    def test_polarsForceDirectedLayout(self):
        _pfdl_ = PolarsForceDirectedLayout(self.g)
        _pfdl_.results(), _pfdl_.stress(), _pfdl_.stressVector()
        self.rt_self.graphLayoutSVGAnimation(_pfdl_.df_anim, _pfdl_.g_connected)

        _pfdl_ = PolarsForceDirectedLayout(self.g, self.pos, static_nodes=self.node_subset)
        _pfdl_.results(), _pfdl_.stress(), _pfdl_.stressVector()
        self.rt_self.graphLayoutSVGAnimation(_pfdl_.df_anim, _pfdl_.g_connected)

    def test_polarsSpringLayout(self):
        _psl_ = PolarsSpringLayout(self.g)
        _psl_.results()
        self.rt_self.graphLayoutSVGAnimation(_psl_.pos_history, self.g)

        _psl_ = PolarsSpringLayout(self.g, self.pos, static_nodes=self.node_subset)
        _psl_.results()
        self.rt_self.graphLayoutSVGAnimation(_psl_.pos_history, self.g)

    def test_polarsSpringLayoutOpt(self):
        _pslo_ = PolarsSpringLayoutOpt(self.g)
        _pslo_.results()
        self.rt_self.graphLayoutSVGAnimation(_pslo_.pos_history, self.g)

        _pslo_ = PolarsSpringLayoutOpt(self.g, self.pos, static_nodes=self.node_subset)
        _pslo_.results()
        self.rt_self.graphLayoutSVGAnimation(_pslo_.pos_history, self.g)

    def test_ConveyProximityLayout(self):
        _cpl_ = ConveyProximityLayout(self.g)
        _cpl_.results()
        _cpl_.svgOfVertexAdditions(self.rt_self)._repr_svg_()

    def test_landmarkMaxMin(self):
        self.rt_self.landmarkMaxMin(self.g)

    def test_identifyLandmarks(self):
        self.rt_self.identifyLandmarks(self.g)

if __name__ == '__main__':
    unittest.main()
