import polars as pl
import networkx as nx
from math import sqrt
import random

__name__ = 'polars_spring_layout'

#
# PolarsSpringLayout() - modeled after the rt_graph_layouts_mixin.py springLayout() method
#
class PolarsSpringLayout(object):
    def __init__(self, g, pos=None, static_nodes=None, spring_exp=1.0, iterations=None):
        self.g            = g
        self.pos          = pos
        self.static_nodes = static_nodes
        self.spring_exp   = spring_exp

        if self.pos is None: self.pos = {}
        for _node_ in self.g.nodes: 
            if _node_ not in self.pos: 
                self.pos[_node_] = (random.random(), random.random())

        self.df_anim      = {}
        self.g_s          = {}
        self.df_results   = []

        # For each subgraph
        S   = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        S_i = 0
        for g_s in S:
            if len(g_s.nodes()) == 1: continue # skip if there's only one node
            self.df_anim[S_i] = []
            self.g_s    [S_i] = g_s
            # Create a distance dataframe
            _lu_  = {'fm':[],'to':[], 't':[]}
            dists = dict(nx.all_pairs_shortest_path_length(g_s))
            for _node_ in dists.keys():
                for _nbor_ in dists[_node_].keys():
                    if _node_ == _nbor_: continue
                    _lu_['fm'].append(_node_), _lu_['to'].append(_nbor_), _lu_['t'].append(dists[_node_][_nbor_])
            df_dist = pl.DataFrame(_lu_)
            df_dist = df_dist.with_columns(pl.col('t').cast(pl.Float64))

            # Create a node position dataframe
            _lu_ = {'node':[], 'x':[], 'y':[]}
            for _node_ in g_s.nodes:
                _xy_ = self.pos[_node_]
                _lu_['node'].append(_node_), _lu_['x'].append(_xy_[0]), _lu_['y'].append(_xy_[1])
            df_pos         = pl.DataFrame(_lu_).with_columns(pl.col('x').cast(pl.Float64), pl.col('y').cast(pl.Float64))
            x0, y0, x1, y1 = df_pos['x'].min(), df_pos['y'].min(), df_pos['x'].max(), df_pos['y'].max()
            if x0 == x1 and y0 == y1: continue # skip if there's no positional differentiation
            
            # Calculate distance between all nodes
            if iterations is None: iterations = len(g_s.nodes())
            mu = 1.0/len(g_s.nodes())
            for _iteration_ in range(iterations):
                if _iteration_ == 0: self.df_anim[S_i].append(df_pos)
                df_pos = df_pos.join(df_pos, how='cross') \
                               .filter(pl.col('node') != pl.col('node_right')) \
                               .with_columns((pl.col('x') - pl.col('x_right')).alias('dx'),
                                             (pl.col('y') - pl.col('y_right')).alias('dy')) \
                               .with_columns((pl.col('dx')**2 + pl.col('dy')**2).sqrt().alias('d')) \
                               .join(df_dist, left_on=['node', 'node_right'], right_on=['fm','to']) \
                               .with_columns(pl.col('t').pow(self.spring_exp).alias('e')) \
                               .with_columns(pl.when(pl.col('d') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('d')).alias('d'),
                                             pl.when(pl.col('t') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('t')).alias('w')) \
                               .with_columns(((2.0*pl.col('dx')*(1.0 - pl.col('t')/pl.col('d')))/pl.col('e')).alias('xadd'),
                                             ((2.0*pl.col('dy')*(1.0 - pl.col('t')/pl.col('d')))/pl.col('e')).alias('yadd')) \
                               .group_by(['node','x','y']).agg(pl.col('xadd').sum(), pl.col('yadd').sum()) \
                               .with_columns((pl.col('x') - mu * pl.col('xadd')).alias('x'),
                                             (pl.col('y') - mu * pl.col('yadd')).alias('y')) \
                               .drop(['xadd','yadd'])
                self.df_anim[S_i].append(df_pos)
            
            self.df_results.append(df_pos.with_columns((pl.col('x') - pl.col('x').min())/(pl.col('x').max() - pl.col('x').min()), 
                                                      ((pl.col('y') - pl.col('y').min())/(pl.col('y').max() - pl.col('y').min()))) \
                                         .with_columns((x0 + pl.col('x') * (x1 - x0)).alias('x'), 
                                                       (y0 + pl.col('y') * (y1 - y0)).alias('y')))
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
    # svgAnimation() - produce the animation svg for the spring layout
    # - copied from the udist_scatterplots_via_sectors_tile_opt.py method
    #
    def svgAnimation(self, duration='10s', w=256, h=256, r=0.04, anim_i=0):
        df = self.df_anim[anim_i][0]
        x_cols = [f'x{i}' for i in range(0, len(self.df_anim[anim_i]))]
        y_cols = [f'y{i}' for i in range(0, len(self.df_anim[anim_i]))]
        x_cols.extend(x_cols[::-1]), y_cols.extend(y_cols[::-1])
        for i in range(1, len(self.df_anim[anim_i])): df = df.join(self.df_anim[anim_i][i], on=['node']).rename({'x_right':f'x{i}', 'y_right':f'y{i}'})
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

