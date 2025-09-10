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
from locale import normalize
from turtle import pos
import polars as pl
import networkx as nx
from math import sqrt
import random

__name__ = 'polars_spring_layout'

#
# PolarsSpringLayout() - modeled after the rt_graph_layouts_mixin.py springLayout() method
#
class PolarsSpringLayout(object):
    def __init__(self, g, pos=None, static_nodes=None, spring_exp=1.0, iterations=None, stress_threshold=1e-2, normalize_coordinates=False):
        self.g            = g
        self.pos          = pos
        self.static_nodes = static_nodes
        self.spring_exp   = spring_exp

        if self.static_nodes is None: self.static_nodes = set()

        if self.pos is None: self.pos = {}
        for _node_ in self.g.nodes: 
            if _node_ not in self.pos: 
                self.pos[_node_] = (random.random(), random.random())

        self.df_anim          = {}
        self.g_s              = {}
        self.df_results       = []
        self.df_result_bounds = []
        self.df_for_rms       = None

        # For each subgraph
        self.S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        S_i = 0
        for g_s in self.S:
            if len(g_s.nodes()) == 1: continue # skip if there's only one node
            self.df_anim[S_i] = []
            self.g_s    [S_i] = g_s
            # Create a distance dataframe
            _lu_  = {'fm':[],'to':[], 't':[]}
            #dists = dict(nx.all_pairs_shortest_path_length(g_s)) # doesn't consider weighted edges...
            self.dists = dict(nx.all_pairs_dijkstra_path_length(g_s))
            for _node_ in self.dists.keys():
                for _nbor_ in self.dists[_node_].keys():
                    if _node_ == _nbor_: continue
                    _lu_['fm'].append(_node_), _lu_['to'].append(_nbor_), _lu_['t'].append(dists[_node_][_nbor_])
            self.df_dist = pl.DataFrame(_lu_)
            self.df_dist = self.df_dist.with_columns(pl.col('t').cast(pl.Float64))

            # Create a node position dataframe
            _lu_ = {'node':[], 'x':[], 'y':[], 's':[]}
            for _node_ in g_s.nodes:
                _xy_ = self.pos[_node_]
                _lu_['node'].append(_node_), _lu_['x'].append(_xy_[0]), _lu_['y'].append(_xy_[1])
                if _node_ in self.static_nodes: _lu_['s'].append(True)
                else:                           _lu_['s'].append(False)
            df_pos         = pl.DataFrame(_lu_).with_columns(pl.col('x').cast(pl.Float64), pl.col('y').cast(pl.Float64))
            x0, y0, x1, y1 = df_pos['x'].min(), df_pos['y'].min(), df_pos['x'].max(), df_pos['y'].max()
            if x0 == x1 and y0 == y1: continue # skip if there's no positional differentiation
            
            # Determine the number of iterations
            if iterations is None: 
                iterations = len(g_s.nodes())
                if iterations < 64: iterations = 64
            mu = 1.0/len(g_s.nodes())

            # Perform the iterations by shifting the nodes per the spring force
            _stress_last_, stress_ok_times = 1e9, 0
            for _iteration_ in range(iterations):
                if _iteration_ == 0: self.df_anim[S_i].append(df_pos)
                __dx__, __dy__ = (pl.col('x') - pl.col('x_right')), (pl.col('y') - pl.col('y_right'))
                df_pos = df_pos.join(df_pos, how='cross') \
                               .filter(pl.col('node') != pl.col('node_right')) \
                               .with_columns((__dx__**2 + __dy__**2).sqrt().alias('d')) \
                               .join(self.df_dist, left_on=['node', 'node_right'], right_on=['fm','to']) \
                               .with_columns(pl.col('t').pow(self.spring_exp).alias('e')) \
                               .with_columns(pl.when(pl.col('d') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('d')).alias('d'),
                                             pl.when(pl.col('t') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('t')).alias('w')) \
                               .with_columns(((2.0*__dx__*(1.0 - pl.col('t')/pl.col('d')))/pl.col('e')).alias('xadd'),
                                             ((2.0*__dy__*(1.0 - pl.col('t')/pl.col('d')))/pl.col('e')).alias('yadd'),
                                             ((pl.col('t') - pl.col('d'))**2).alias('stress')) \
                               .group_by(['node','x','y','s']).agg(pl.col('xadd').sum(), pl.col('yadd').sum(), pl.col('stress').sum()/len(g_s.nodes())) \
                               .with_columns(pl.when(pl.col('s')).then(pl.col('x')).otherwise(pl.col('x') - mu * pl.col('xadd')).alias('x'),
                                             pl.when(pl.col('s')).then(pl.col('y')).otherwise(pl.col('y') - mu * pl.col('yadd')).alias('y')) \
                               .drop(['xadd','yadd'])
                # Keep track of the animation sequence
                self.df_anim[S_i].append(df_pos)
                _stress_      = df_pos['stress'].sum()
                if stress_threshold is not None and _iteration_ > 32 and abs(_stress_ - _stress_last_) < stress_threshold: 
                    stress_ok_times += 1
                    if stress_ok_times >= 5: break
                else:
                    stress_ok_times  = 0
                _stress_last_ = _stress_
            
            # Save off the normalization coordinates
            self.df_result_bounds.append((df_pos['x'].min(), df_pos['y'].min(), df_pos['x'].max(), df_pos['y'].max()))

            # Store the results
            if normalize_coordinates:
                self.df_results.append(df_pos.with_columns((pl.col('x') - pl.col('x').min())/(pl.col('x').max() - pl.col('x').min()), 
                                                          ((pl.col('y') - pl.col('y').min())/(pl.col('y').max() - pl.col('y').min()))) \
                                             .with_columns((x0 + pl.col('x') * (x1 - x0)).alias('x'), 
                                                           (y0 + pl.col('y') * (y1 - y0)).alias('y')))
            else: self.df_results.append(df_pos)
            S_i += 1

    #
    # results() - return the results as a dictionary of nodes to xy coordinate tuples
    #
    def results(self):
        _pos_ = {}
        for i in range(0, len(self.df_results)): 
            _this_pos_  =  dict(zip(self.df_results[i]['node'], zip(self.df_results[i]['x'], self.df_results[i]['y'])))
            _pos_      |=  _this_pos_
        return _pos_

    #
    # stressSums() -- produce an array of stress summations for the specific subgraph
    #
    def stressSums(self, anim_i=0):
        _sums_ = []
        for i in range(1, len(self.df_anim[0])):
            _stress_sum_ = (self.df_anim[0][i]['stress']).sum()
            _sums_.append(_stress_sum_)
        return _sums_

    #
    # rootMeanSquareError() - determine the root mean square error between the distances and the ideal distances
    # - scale parameters will first be used to modify the distances relative to each other
    # - this only looks at edges ... not nodes that don't share a connection
    # - this method was created for interactively re-springing portions of the layout
    #   (because the positions will need to be adjusted if they were normalized earlier -- see "normalized_coordinates" flag)
    #
    def rootMeanSquareError(self, x_scale=1.0, y_scale=1.0):
        # Create the structure for the calculation
        if self.df_for_rms is None:
            pos = self.results()
            _lu_ = {'n':[], 'nbor':[], 'w':[], 'n_x':[], 'nbor_x':[], 'n_y':[], 'nbor_y':[]}
            for n in self.g.nodes():
                for nbor in self.g.neighbors(n):
                    _w_                  =  1.0 if 'weight' not in self.g[n][nbor] else self.g[n][nbor]['weight']
                    _xy_n_, _xy_nbor_    =  pos[n], pos[nbor]
                    _lu_['n'].append(n),                 _lu_['nbor'].append(nbor),             _lu_['w'].append(_w_)
                    _lu_['n_x'].append(_xy_n_[0]),       _lu_['n_y'].append(_xy_n_[1])
                    _lu_['nbor_x'].append(_xy_nbor_[0]), _lu_['nbor_y'].append(_xy_nbor_[1])
            self.df_for_rms = pl.DataFrame(_lu_)
        # Implementation using Polars Operations
        __n_x__, __n_y__       = pl.col('n_x'), pl.col('n_y')
        __nbor_x__, __nbor_y__ = pl.col('nbor_x'), pl.col('nbor_y')
        _df_ = self.df_for_rms.with_columns(((__n_x__*x_scale - __nbor_x__*x_scale)**2 + (__n_y__*y_scale - __nbor_y__*y_scale)**2).alias('x2'))
        _df_ = _df_.with_columns((pl.col('x2')**0.5).alias('d'))
        _df_ = _df_.with_columns(((_df_['w'] - _df_['d'])**2).alias('rms'))
        return sqrt(_df_['rms'].sum()/len(_df_))

    #
    # svgAnimation() - produce the animation svg for the spring layout
    # - copied from the udist_scatterplots_via_sectors_tile_opt.py method
    #
    def svgAnimation(self, duration='10s', w=256, h=256, r=0.04, anim_i=0, draw_links=True, draw_nodes=True):
        df = self.df_anim[anim_i][0]
        x_cols = [f'x{i}' for i in range(0, len(self.df_anim[anim_i]))]
        y_cols = [f'y{i}' for i in range(0, len(self.df_anim[anim_i]))]
        x_cols.extend(x_cols[::-1]), y_cols.extend(y_cols[::-1])
        for i in range(1, len(self.df_anim[anim_i])): df = df.join(self.df_anim[anim_i][i].drop(['s', 'stress']), on=['node']).rename({'x_right':f'x{i}', 'y_right':f'y{i}'})
        df = df.rename({'x':'x0', 'y':'y0'})
        # Determine the bounds
        x0, y0, x1, y1 = df['x0'].min(), df['y0'].min(), df['x0'].max(), df['y0'].max()
        for i in range(1, len(self.df_anim[anim_i])):
            x0, y0, x1, y1 = min(x0, df[f'x{i}'].min()), min(y0, df[f'y{i}'].min()), max(x1, df[f'x{i}'].max()), max(y1, df[f'y{i}'].max())
        # Produce the values strings for x & y and drop the unneeded columns
        df = df.with_columns(pl.concat_str(x_cols, separator=';').alias('x_values_str'), 
                             pl.concat_str(y_cols, separator=';').alias('y_values_str')).drop(x_cols).drop(y_cols)


        svg = []
        svg.append(f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="{x0} {y0} {x1-x0} {y1-y0}" xmlns="http://www.w3.org/2000/svg">')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="#ffffff" />')

        # Edges
        if draw_links:
            _lu_ = {'fm':[], 'to':[]}
            for _node_ in self.g_s[anim_i].nodes():
                for _nbor_ in self.g_s[anim_i].neighbors(_node_):
                    _lu_['fm'].append(_node_), _lu_['to'].append(_nbor_)
            df_edges = pl.DataFrame(_lu_).join(df, left_on='fm', right_on='node') \
                                         .rename({'x_values_str':'fm_x_values_str', 'y_values_str':'fm_y_values_str'}) \
                                         .join(df, left_on='to', right_on='node') \
                                         .rename({'x_values_str':'to_x_values_str', 'y_values_str':'to_y_values_str'})
            _str_ops_ = [pl.lit(f'<line stroke-width="{r}" stroke="#a0a0a0">'),
                        
                         pl.lit('<animate attributeName="x1" values="'),
                         pl.col('fm_x_values_str'),
                         pl.lit(f'" begin="0s" dur="{duration}" repeatCount="indefinite" />'),

                         pl.lit('<animate attributeName="y1" values="'),
                         pl.col('fm_y_values_str'),
                         pl.lit(f'" begin="0s" dur="{duration}" repeatCount="indefinite" />'),

                         pl.lit('<animate attributeName="x2" values="'),
                         pl.col('to_x_values_str'),
                         pl.lit(f'" begin="0s" dur="{duration}" repeatCount="indefinite" />'),

                         pl.lit('<animate attributeName="y2" values="'),
                         pl.col('to_y_values_str'),
                         pl.lit(f'" begin="0s" dur="{duration}" repeatCount="indefinite" />'),

                         pl.lit('</line>')]
            df_edges = df_edges.with_columns(pl.concat_str(*_str_ops_, separator='').alias('svg'))
            svg.extend(df_edges['svg'])

        # Nodes
        if draw_nodes:
            _str_ops_ = [pl.lit(f'<circle r="{r}" fill="#000000"> <animate attributeName="cx" values="'),
                         pl.col('x_values_str'),
                         pl.lit(f'" begin="0s" dur="{duration}" repeatCount="indefinite" />'),
                         pl.lit('<animate attributeName="cy" values="'),
                         pl.col('y_values_str'),
                         pl.lit(f'" begin="0s" dur="{duration}" repeatCount="indefinite" />'),
                         pl.lit('</circle>')]
            df = df.with_columns(pl.concat_str(*_str_ops_, separator='').alias('svg'))
            svg.extend(df['svg'])

        # Close the SVG
        svg.append('</svg>')
        return ''.join(svg)

