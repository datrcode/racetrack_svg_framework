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
import polars as pl
import networkx as nx
import numpy as np
from math import sqrt
import random
import rtsvg

__name__ = 'polars_lazy_force_directed_layout'

#
# PolarsForceDirectedLayout() - modeled after the rt_graph_layouts_mixin.py springLayout() method
# - renamed ForceDirected vs Springs because this implements a broader class of layout algorithms
# - attempting to use lazy dataframes to understand the performance
#
# Implements portions of the following:
#
# Drawing Graphs to Convey Proximity: An Incremental Arrangement Method
# J.D. Cohen
# ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
#
class PolarsLazyForceDirectedLayout(object):
    def __init__(self, g_connected, pos=None, static_nodes=None, k=0, iterations=None, stress_threshold=1e-2, distances=None):
        self.g_connected  = g_connected
        self.pos          = pos
        self.static_nodes = static_nodes
        self.k            = k

        if self.static_nodes is None: self.static_nodes = set()

        all_nodes_had_initial_positions = True
        if self.pos is None: self.pos = {}
        for _node_ in self.g_connected.nodes: 
            if _node_ not in self.pos: 
                self.pos[_node_] = (random.random(), random.random())
                all_nodes_had_initial_positions = False

        self.df_anim          = []
        self.df_dist          = {}
        self.df_results       = None
        self.df_result_bounds = None

        # Create a graph distance dataframe
        _lu_  = {'fm':[],'to':[], 't':[]}
        self.dists = dict(nx.all_pairs_dijkstra_path_length(self.g_connected)) if distances is None else distances
        for _node_ in self.dists.keys():
            for _nbor_ in self.dists[_node_].keys():
                if _node_ == _nbor_: continue
                _lu_['fm'].append(_node_), _lu_['to'].append(_nbor_), _lu_['t'].append(self.dists[_node_][_nbor_])
        self.df_dist = pl.DataFrame(_lu_).with_columns(pl.col('t').cast(pl.Float64)).lazy()

        # Create a node position dataframe
        _lu_ = {'node':[], 'x':[], 'y':[], 's':[]}
        for _node_ in self.g_connected.nodes:
            _xy_ = self.pos[_node_]
            _lu_['node'].append(_node_), _lu_['x'].append(_xy_[0]), _lu_['y'].append(_xy_[1])
            if _node_ in self.static_nodes: _lu_['s'].append(True)
            else:                           _lu_['s'].append(False)
        df_pos         = pl.DataFrame(_lu_).with_columns(pl.col('x').cast(pl.Float64), pl.col('y').cast(pl.Float64))
        x0, y0, x1, y1 = df_pos['x'].min(), df_pos['y'].min(), df_pos['x'].max(), df_pos['y'].max()
        if x0 == x1 and y0 == y1: 
            self.df_results = df_pos
            return # nothing to do... all nodes in the same place
        
        # Determine the number of iterations
        if iterations is None: 
            iterations = 2*len(self.g_connected.nodes())
            if iterations < 64: iterations = 64
        mu = 1.0/(2.0*len(self.g_connected.nodes()))

        # Perform the iterations by shifting the nodes per the spring force
        _stress_last_, stress_ok_times = 1e9, 0
        for _iteration_ in range(iterations):
            if _iteration_ == 0: self.df_anim.append(df_pos)
            __dx__, __dy__ = (pl.col('x') - pl.col('x_right')), (pl.col('y') - pl.col('y_right'))
            df_pos = df_pos.lazy() \
                            .join(df_pos.lazy(), how='cross') \
                            .filter(pl.col('node') != pl.col('node_right')) \
                            .with_columns((__dx__**2 + __dy__**2).sqrt().alias('d')) \
                            .join(self.df_dist, left_on=['node', 'node_right'], right_on=['fm','to']) \
                            .with_columns(pl.col('t').pow(self.k).alias('t_k')) \
                            .with_columns(pl.when(pl.col('d') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('d')).alias('d'),
                                          pl.when(pl.col('t') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('t')).alias('w')) \
                            .with_columns(((2.0*__dx__*(1.0 - pl.col('t')/pl.col('d')))/pl.col('t_k')).alias('xadd'),
                                          ((2.0*__dy__*(1.0 - pl.col('t')/pl.col('d')))/pl.col('t_k')).alias('yadd'),
                                          (((pl.col('t') - pl.col('d'))**2)/pl.col('t_k')).alias('stress')) \
                            .group_by(['node','x','y','s']).agg(pl.col('xadd').sum(), pl.col('yadd').sum(), pl.col('stress').sum()/len(self.g_connected.nodes())) \
                            .with_columns(pl.when(pl.col('s')).then(pl.col('x')).otherwise(pl.col('x') - mu * pl.col('xadd')).alias('x'),
                                          pl.when(pl.col('s')).then(pl.col('y')).otherwise(pl.col('y') - mu * pl.col('yadd')).alias('y')) \
                            .drop(['xadd','yadd']) \
                            .collect()
            # Keep track of the animation sequence
            self.df_anim.append(df_pos)
            _stress_ = df_pos['stress'].sum()
            if stress_threshold is not None and _iteration_ > 32 and abs(_stress_ - _stress_last_) < stress_threshold: 
                stress_ok_times += 1
                if stress_ok_times >= 5: break
            else:
                stress_ok_times  = 0
            _stress_last_ = _stress_
        
            # Save off the normalization coordinates
            self.df_result_bounds = (df_pos['x'].min(), df_pos['y'].min(), df_pos['x'].max(), df_pos['y'].max())

            # Store the results
            self.df_results = df_pos

    #
    # results() - return the results as a dictionary of nodes to xy coordinate tuples
    #
    def results(self): return dict(zip(self.df_results['node'], zip(self.df_results['x'], self.df_results['y'])))
 
    #
    # stressVector() -- produce an array of stress summations
    #
    def stressVector(self):
        _sums_ = []
        for i in range(len(self.df_anim)): _sums_.append(self.stress(animation_step=i))
        return _sums_

    #
    # stress() - calculate the stress per the following paper:
    #
    # Drawing Graphs to Convey Proximity: An Incremental Arrangement Method
    # J.D. Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    #
    # k=0 # absolute stress
    # k=1 # semi-proportional stress
    # k=2 # proportional stress
    #
    def stress(self, animation_step=-1):
        df_pos  = self.df_anim[animation_step]
        df_dist = self.df_dist
        _df_    = df_pos.join(df_pos, how='cross') \
                        .filter(pl.col('node') != pl.col('node_right')) \
                        .join(df_dist, left_on=['node', 'node_right'], right_on=['fm','to']) \
                        .with_columns(((pl.col('x') - pl.col('x_right'))**2 + (pl.col('y') - pl.col('y_right'))**2).sqrt().alias('d')) \
                        .with_columns((pl.col('t')**(2-self.k)).alias('__prod_1__'),
                                     ((pl.col('d') - pl.col('t'))**2 / pl.col('t')**self.k).alias('__prod_2__'))
        return (1.0 / _df_['__prod_1__'].sum()) * _df_['__prod_2__'].sum()
