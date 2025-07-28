import polars as pl
import networkx as nx

__name__ = 'polars_spring_layout'

class PolarsSpringLayout(object):
    def __init__(self, g, pos, static_nodes=None):
        self.g            = g
        self.pos          = pos
        self.static_nodes = static_nodes

        # Normalize the positions to (0.0, 0.0) -> (1.0, 1.0)
        # ... so that all the scalars & params can be simpler
        _es_, _xs_, _ys_ = [], [], []
        for _entity_ in self.pos:
            _xy_ = self.pos[_entity_]
            _es_.append(_entity_), _xs_.append(_xy_[0]), _ys_.append(_xy_[1])
        _df_ = pl.DataFrame({'entity':_es_, 'x':_xs_, 'y':_ys_})
        _df_ = _df_.with_columns(pl.col('x').cast(pl.Float64), pl.col('y').cast(pl.Float64))
        self.x0, self.y0, self.x1, self.y1 = _df_['x'].min(), _df_['y'].min(), _df_['x'].max(), _df_['y'].max()
        _df_ = _df_.with_columns((pl.col('x') - _df_['x'].min())/(pl.col('x').max() - _df_['x'].min()), 
                                 (pl.col('y') - _df_['y'].min())/(pl.col('y').max() - _df_['y'].min()))
        self.pos = dict(zip(_df_['entity'], zip(_df_['x'], _df_['y'])))

        # For each subgraph
        S = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        for g_s in S:
            self.dists = dict(nx.all_pairs_shortest_path_length(g_s))

