import polars as pl
import networkx as nx
from math import sqrt

__name__ = 'polars_spring_layout'

class PolarsSpringLayout(object):
    def __init__(self, g, pos, static_nodes=None, spring_exp=0.1):
        self.g            = g
        self.pos          = pos
        self.static_nodes = static_nodes
        self.spring_exp   = spring_exp

        # Normalize the positions to (0.0, 0.0) -> (1.0, 1.0)
        # ... so that all the scalars & params can be simpler
        _es_, _xs_, _ys_ = [], [], []
        for _entity_ in self.pos:
            _xy_ = self.pos[_entity_]
            _es_.append(_entity_), _xs_.append(_xy_[0]), _ys_.append(_xy_[1])
        _df_ = pl.DataFrame({'node':_es_, 'x':_xs_, 'y':_ys_})
        _df_ = _df_.with_columns(pl.col('x').cast(pl.Float64), pl.col('y').cast(pl.Float64))
        self.x0, self.y0, self.x1, self.y1 = _df_['x'].min(), _df_['y'].min(), _df_['x'].max(), _df_['y'].max()
        _df_ = _df_.with_columns((pl.col('x') - _df_['x'].min())/(pl.col('x').max() - _df_['x'].min()), 
                                 (pl.col('y') - _df_['y'].min())/(pl.col('y').max() - _df_['y'].min()))
        self.pos = dict(zip(_df_['node'], zip(_df_['x'], _df_['y']))) # overwrite pos w/ normalized positions

        # For each subgraph
        S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        for g_s in S:
            # Create a distance dataframe
            _lu_  = {'fm':[],'to':[], 'w':[]}
            dists = dict(nx.all_pairs_shortest_path_length(g_s))
            for _node_ in dists.keys():
                for _nbor_ in dists[_node_].keys():
                    if _node_ == _nbor_: continue
                    _lu_['fm'].append(_node_), _lu_['to'].append(_nbor_), _lu_['w'].append(dists[_node_][_nbor_])
            df_dist = pl.DataFrame(_lu_)
            df_dist = df_dist.with_columns(pl.col('w').cast(pl.Float64))
            df_dist = df_dist.with_columns(pl.col('w')/df_dist['w'].max())

            # Create a node position dataframe
            _lu_ = {'node':[], 'x':[], 'y':[]}
            for _node_ in g_s.nodes:
                _xy_ = self.pos[_node_]
                _lu_['node'].append(_node_), _lu_['x'].append(_xy_[0]), _lu_['y'].append(_xy_[1])
            df_pos = pl.DataFrame(_lu_)

            # Calculate distance between all nodes
            df_pos = df_pos.join(df_pos, how='cross') \
                           .filter(pl.col('node') != pl.col('node_right')) \
                           .with_columns(((pl.col('x') - pl.col('x_right'))**2 + 
                                          (pl.col('y') - pl.col('y_right'))**2).sqrt().alias('d')) \
                           .join(df_dist, left_on=['node', 'node_right'], right_on=['fm','to']) \
                           .with_columns(pl.col('w').pow(self.spring_exp).alias('e'))
            
            self.df_pos  = df_pos
            self.df_dist = df_dist






