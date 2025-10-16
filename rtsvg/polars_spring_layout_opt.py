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
import random
import time
from math import sqrt


__name__ = 'polars_spring_layout_opt'

#
# Implements the spring layout described in the following paper:
#
# "Graph Drawing by Force-directed Placement"
# Fruchterman & Reingold
# Software -- Practice and Experience, Vol. 21 (1 1), 1129-1164 (November 1991)
#
# Figure 1 (Force-directed Placement)
# - with the optimization to only consider vertices w/in 2*k of the vertex for repulsion
#
class PolarsSpringLayoutOpt(object):
    def __init__(self, 
                 g, 
                 pos          = None, 
                 static_nodes = None, 
                 iterations   = None,
                 t            = 5.0,
                 C            = 0.2,
                 W            = 256,
                 H            = 256):
        # Variables described in the paper
        area = W * H
        k    = C * sqrt(area/len(g.nodes()))

        # State to determine overall performance
        self.time_lu = {'dist_df_create'       : 0.0, 
                        'pos_df_create'        : 0.0,
                        'cross_join_iteration' : 0.0,
                        'repulse_iteration'    : 0.0,
                        'attract_iteration'    : 0.0,
                        'sum_iteration'        : 0.0,
                        'adjust_iteration'     : 0.0,
                        'copy_pos'             : 0.0,}
        self.repulse_rows = []

        # Create the distance dataframe from the graph
        t0 = time.time()
        _lu_    = {'fm':[], 'to':[], 't':[]}
        for node in g.nodes:
            for nbor in g.neighbors(node):
                _w_ = g[node][nbor]['weight'] if 'weight' in g[node][nbor] else 1.0
                _lu_['fm'].append(node), _lu_['to'].append(nbor), _lu_['t'].append(_w_)
        df_dist = pl.DataFrame(_lu_)
        t1 = time.time()
        self.time_lu['dist_df_create'] = t1 - t0

        # Create positional dataframe w/ random values
        t2 = time.time()
        _lu_    = {'node':[], 'x':[], 'y':[], 's':[]}
        for node in g.nodes: 
            _lu_['node'].append(node)
            if pos is not None and node in pos: _lu_['x'].append(pos[node][0]),      _lu_['y'].append(pos[node][1])
            else:                               _lu_['x'].append(W*random.random()), _lu_['y'].append(H*random.random()) 
            if static_nodes is not None and node in static_nodes: _lu_['s'].append(True)
            else:                                                 _lu_['s'].append(False)
        df_pos  = pl.DataFrame(_lu_)
        t3 = time.time()
        self.time_lu['pos_df_create'] = t3 - t2

        self.pos_history = []
        for i in range(2*len(g.nodes)):
            self.pos_history.append(df_pos)
            # Cross join so that all combinations of nodes are considered (except where nod and node_right are the same node)
            t4 = time.time()
            df_cross = df_pos.drop('s')\
                             .join(df_pos.drop('s'), how='cross') \
                             .filter(pl.col('node') != pl.col('node_right'))
            t5 = time.time()
            self.time_lu['cross_join_iteration'] += t5 - t4

            # Repulsive Forces
            t6 = time.time()
            __dx__, __dy__ = (pl.col('x') - pl.col('x_right')), (pl.col('y') - pl.col('y_right'))
            # Dataframe describing one off tiles
            df_around = pl.DataFrame({'xi_offset':[-1,  0,  1, -1, 0, 1, -1, 0, 1],
                                      'yi_offset':[-1, -1, -1,  0, 0, 0,  1, 1, 1]})
            # Next set of commands do the following:
            # - Calculate the tile xi, yi based on the 2*k parameter
            # - Determine which tiles a node needs
            # - Join those tiles in (and remove self-to-self edges)
            # - Remove all the fields related to tiles
            # - Remove edges (these are not repulsed)
            # - Calculate the distance squared
            # - filter out the ones that are too far and calculate distance
            # - Proceed with the original calculation (and using the modified formula)
            df_repulse = df_pos.with_columns(((pl.col('x') - pl.col('x').min())//(2 * k)).cast(pl.Int32).alias('xi'),
                                             ((pl.col('y') - pl.col('y').min())//(2 * k)).cast(pl.Int32).alias('yi')) \
                               .join(df_around, how='cross') \
                               .with_columns((pl.col('xi_offset')+pl.col('xi')).alias('xi_join'),
                                             (pl.col('yi_offset')+pl.col('yi')).alias('yi_join'))            
            df_repulse = df_repulse.join(df_repulse, left_on=['xi_join','yi_join'], right_on=['xi','yi']) \
                                   .filter(pl.col('node') != pl.col('node_right')) \
                                   .drop(['s','xi','yi','xi_offset','yi_offset','xi_join','yi_join',
                                          's_right','xi_offset_right','yi_offset_right','xi_join_right','yi_join_right']) \
                                   .join(df_dist, left_on=['node','node_right'], right_on=['fm','to'], how='anti') \
                                   .with_columns(((pl.col('x_right')-pl.col('x'))**2 + (pl.col('y_right')-pl.col('y'))**2).alias('d_squared')) \
                                   .filter(pl.col('d_squared') < (2*k)**2) \
                                   .with_columns((__dx__**2 + __dy__**2).sqrt().alias('d')) \
                                   .with_columns(pl.when(pl.col('d') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('d')).alias('d')) \
                                   .with_columns((((k**2)/pl.col('d'))*(2.0 * k - pl.col('d'))).alias('f_r')) \
                                   .with_columns(((__dx__/pl.col('d')) * pl.col('f_r')).alias('disp_x'),
                                                 ((__dy__/pl.col('d')) * pl.col('f_r')).alias('disp_y'))
            self.repulse_rows.append(len(df_repulse))
            t7 = time.time()
            self.time_lu['repulse_iteration'] += t7 - t6

            # Attractive Forces
            t8 = time.time()
            df_attract = df_cross.join(df_dist, left_on=['node', 'node_right'], right_on=['fm', 'to']) \
                                 .with_columns((__dx__**2 + __dy__**2).sqrt().alias('d')) \
                                 .with_columns(pl.when(pl.col('d') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('d')).alias('d')) \
                                 .with_columns(((pl.col('d')**2)/k).alias('f_a')) \
                                 .with_columns((-(__dx__)/(pl.col('d')*pl.col('t'))*pl.col('f_a')).alias('disp_x'),
                                               (-(__dy__)/(pl.col('d')*pl.col('t'))*pl.col('f_a')).alias('disp_y'),
                                               (  __dx__ /(pl.col('d')*pl.col('t'))*pl.col('f_a')).alias('disp_x_right'),
                                               (  __dy__ /(pl.col('d')*pl.col('t'))*pl.col('f_a')).alias('disp_y_right')) \
                                 .drop({'x','y','x_right','y_right'})
            t9 = time.time()
            self.time_lu['attract_iteration'] += t9 - t8

            # Sum them up (but prepare the same columns for both dataframes columns)
            t10 = time.time()
            df_repulse = df_repulse.drop(['x','y','node_right','x_right','y_right','d_squared',    'd','f_r'])
            df_attract = df_attract.drop([                                                     't','d','f_a'])
            df_sums = pl.concat([df_repulse,
                                 df_attract.drop({'node_right','disp_x_right','disp_y_right'}),
                                 df_attract.drop({'node',      'disp_x',      'disp_y'}).rename({'node_right':'node','disp_x_right':'disp_x','disp_y_right':'disp_y'})]) \
                        .group_by('node').agg(pl.col('disp_x').sum(), pl.col('disp_y').sum())
            t11 = time.time()
            self.time_lu['sum_iteration'] += t11 - t10

            # Add the forces to the positions & prepare to loop again
            t12 = time.time()
            df_pos = df_pos.join(df_sums, on='node').rename({'disp_x':'dx','disp_y':'dy'}) \
                           .with_columns((pl.col('dx')**2 + pl.col('dy')**2).sqrt().alias('d')) \
                           .with_columns(pl.when(pl.col('d') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('d')).alias('d')) \
                           .with_columns(pl.when(pl.col('dx').abs() > t).then(pl.lit(t)).otherwise(pl.col('dx')).alias('xmag'),
                                         pl.when(pl.col('dy').abs() > t).then(pl.lit(t)).otherwise(pl.col('dy')).alias('ymag')) \
                           .with_columns((pl.when(pl.col('s')).then(pl.col('x')).otherwise(pl.col('x') + pl.col('xmag')*pl.col('dx')/pl.col('d'))).alias('x'),
                                         (pl.when(pl.col('s')).then(pl.col('y')).otherwise(pl.col('y') + pl.col('ymag')*pl.col('dy')/pl.col('d'))).alias('y')) \
                           .drop({'dx','dy','d','xmag','ymag'}) \
                           .with_columns((W*(pl.col('x') - pl.col('x').min())/(pl.col('x').max() - pl.col('x').min())).alias('x'), 
                                         (H*(pl.col('y') - pl.col('y').min())/(pl.col('y').max() - pl.col('y').min())).alias('y'))
            t13 = time.time()
            self.time_lu['adjust_iteration'] += t13 - t12
            
            # Cool the temperature
            t *= 0.999

        self.pos_history.append(df_pos)

        t14 = time.time()
        self.pos = {}
        for i in range(len(df_pos)): self.pos[df_pos['node'][i]] = (df_pos['x'][i], df_pos['y'][i])
        t15 = time.time()
        self.time_lu['copy_pos'] += t15 - t14

    #
    # results() - Return the positions
    #
    def results(self): return self.pos
