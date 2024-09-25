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
        self.rt_self.hyperTreeLayout(self.g, touch_up_with_springs=True)

    def test_treeMapGraphComponentPlacement(self):
        self.rt_self.treeMapGraphComponentPlacement(self.g, self.pos)

    def test_springLayout(self):
        self.rt_self.springLayout(self.g)
        self.rt_self.springLayout(self.g, self.pos, selection=self.node_subset)

    def test_barycentricLayout(self):
        self.rt_self.barycentricLayout(self.g, self.pos, selection=self.node_subset)

    def test_springLayoutThreaded(self):
        pass # something broken in the package for any classes outside of the mixins...
        # self.rt_self.springLayoutThreaded(self.g)

if __name__ == '__main__':
    unittest.main()
