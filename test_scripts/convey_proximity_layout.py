#
# Implementation of the following:
#
# Drawing Graphs to Convey Proximity: An Incremental Arrangement Method
# J.D. Cohen
# ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
#
import polars as pl
import networkx as nx
from math import log10, ceil
import random
import time
import rtsvg

__name__ = 'convey_proximity_layout'

class ConveyProximityLayout(object):
    #
    # Table II of paper
    #
    def __init__(self, g_connected, k=2.0):
        pos         = {}                                                   # Results
        V           = set(g_connected.nodes)
        distances   = self.__getTargetDistances__(g_connected)             # Establish target distances
        Q           = self.__orderVertices__(g_connected, distances)       # Make vector Q of vertices in order of inclusion
        H           = set()                                                # Initialize vertices to arrange

        # Stress information
        self.stress_lu      = {'stress':[], 'i':[], 'i_global':[], 'arrange_round':[]}
        self.arrange_round  = 0
        self.i_global       = 0

        # There needs to be an initial vertex set that is already placed for the while loop to actually work...
        # vvv
        for i in range(self.__numberToAddThisTime__(len(H), len(V))):
            H.add(Q[0])
            pos[Q[0]] = (random.random(), random.random())
            Q = Q[1:]
        pos = self.__arrangeDirect__(H, pos, distances, k=k)
        # ^^^

        while H != V:
            _number_to_add_ = self.__numberToAddThisTime__(len(H), len(V))
            for i in range(_number_to_add_):
                v      = Q[0]                                                   # Get next vertex
                h1, h2 = self.__closestMembers__(H, v, distances)               # Find closest two members of H
                pos[v] = self.__neighborlyLocation__(v, h1, h2, pos, distances) # Put new vertex near them
                Q      = Q[1:]                                                  # This vertex is done
                H.add(v)
            pos = self.__arrangeDirect__(H, pos, distances, k=k)                # Arrange accumulated subset
        
        self.pos = pos

    # results() - returns the results
    def results(self): return self.pos

    # __getTargetDistances__()
    def __getTargetDistances__(self, _g_): return dict(nx.all_pairs_dijkstra_path_length(_g_))

    # Table IV of paper
    def __everyNthMember__(self, Q, n): return [Q[i] for i in range(0, len(Q), n)]
    def __disperseTheseVertices__(self, Q, increment_ratio=1):
        if len(Q) > increment_ratio + 1:
            F = self.__everyNthMember__(Q, 1+increment_ratio)
            B = [item for item in Q if item not in F]     # Q - F
            F = self.__disperseTheseVertices__(F)
            F.extend(B)
            F
        return Q

    def __orderVertices__(self, _g_, _dist_):
        Q = [n for n in nx.traversal.dfs_preorder_nodes(_g_)]
        return self.__disperseTheseVertices__(Q)

    # Table III of paper
    def __numberToAddThisTime__(self, _prev_, _final_, increment_ratio=1, increment_minimum=10):
        if _prev_ > 0: _inc_ = _prev_ * increment_ratio
        else:
            _inc_ = _final_
            while _inc_ > increment_minimum: _inc_ = ceil(_inc_ / (1 + increment_ratio))
        if _inc_ > _final_ - _prev_: _inc_ = _final_ - _prev_
        return _inc_

    # Primitive version of closest members
    def __closestMembers__(self, _H_, _v_, _distances_):
        _h1_, _h1_d_, _h2_, _h2_d_ = None, None, None, None
        for _k_ in _H_:
            if _k_ == _v_: continue
            _d_ = _distances_[_v_][_k_]
            if   _h1_d_ is None: _h1_, _h1_d_ = _k_, _d_
            elif _h2_d_ is None: _h2_, _h2_d_ = _k_, _d_
            elif _d_ < _h1_d_ or _d_ < _h2_d_:
                if   _d_ < _h1_d_ and _d_ < _h2_d_:
                    if _h1_d_ < _h2_d_: _h2_, _h2_d_ = _k_, _d_
                    else:               _h1_, _h1_d_ = _k_, _d_
                elif _d_ < _h1_d_:      _h1_, _h1_d_ = _k_, _d_
                else:                   _h2_, _h2_d_ = _k_, _d_
        return _h1_, _h2_

    # Figure 3 from the paper (and the accompanying formulas)
    # Primitive version of neighborlyLocation()
    def __neighborlyLocation__(self, i, j, k, _pos_, _distances_):
        t_ik, t_ij, t_jk = _distances_[i][k], _distances_[i][j], _distances_[j][k]
        _expr_   = (1.0 / (2 * t_jk**2)) * (t_ik**2 - t_ij**2 - t_jk**2)
        _gamma_  = min(_expr_, 0.5)
        x_j, y_j = _pos_[j]
        x_k, y_k = _pos_[k]
        e_x, e_y = random.random() * 0.01, random.random() * 0.01
        x, y     = x_j + _gamma_ * (x_j - x_k) + e_x, y_j + _gamma_ * (y_j - y_k) + e_y
        return x, y

    #
    # Modified version from the PolarsForceDirectedLayout implementation
    #
    def __arrangeDirect__(self, _nodes_, _pos_, _distances_, k):
        # Construct the df_pos and df_dist dataframes
        _lu_pos_  = {'node':[], 'x':[], 'y':[]}
        _lu_dist_ = {'fm':[],'to':[], 't':[]}
        for _node_ in _nodes_:
            _lu_pos_['node'].append(_node_), _lu_pos_['x'].append(_pos_[_node_][0]), _lu_pos_['y'].append(_pos_[_node_][1])
            for _nbor_ in _nodes_:
                if _nbor_ == _node_: continue
                _lu_dist_['fm'].append(_node_), _lu_dist_['to'].append(_nbor_), _lu_dist_['t'].append(_distances_[_node_][_nbor_])
        df_pos, df_dist = pl.DataFrame(_lu_pos_), pl.DataFrame(_lu_dist_)

        # Iterate using the force directed layout algorithm
        iterations = max(64, len(_nodes_))
        mu         = 1.0/(2.0*len(_nodes_))
        __dx__, __dy__ = (pl.col('x') - pl.col('x_right')), (pl.col('y') - pl.col('y_right'))
        for i in range(iterations):
            df_pos = df_pos.join(df_pos, how='cross') \
                           .filter(pl.col('node') != pl.col('node_right')) \
                           .with_columns((__dx__**2 + __dy__**2).sqrt().alias('d')) \
                           .join(df_dist, left_on=['node', 'node_right'], right_on=['fm','to']) \
                           .with_columns(pl.col('t').pow(k).alias('t_k')) \
                           .with_columns(pl.when(pl.col('d') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('d')).alias('d'),
                                         pl.when(pl.col('t') < 0.001).then(pl.lit(0.001)).otherwise(pl.col('t')).alias('w')) \
                           .with_columns((pl.col('t')**(2-k)).alias('__prod_1__'),
                                         ((2.0*__dx__*(1.0 - pl.col('t')/pl.col('d')))/pl.col('t_k')).alias('xadd'),
                                         ((2.0*__dy__*(1.0 - pl.col('t')/pl.col('d')))/pl.col('t_k')).alias('yadd'),
                                         (((pl.col('t') - pl.col('d'))**2)/(pl.col('t_k'))**k).alias('__prod_2__')) \
                           .group_by(['node','x','y']).agg(pl.col('xadd').sum(), pl.col('yadd').sum(), pl.col('__prod_1__').sum(), pl.col('__prod_2__').sum()) \
                           .with_columns((pl.col('x') - mu * pl.col('xadd')).alias('x'),
                                         (pl.col('y') - mu * pl.col('yadd')).alias('y')) \
                           .drop(['xadd','yadd'])
            # Stress calculation & storage
            stress = (1.0 / df_pos['__prod_1__'].sum()) * df_pos['__prod_2__'].sum()
            self.stress_lu['stress'].append(stress), self.stress_lu['i'].append(i), self.stress_lu['i_global'].append(self.i_global), self.stress_lu['arrange_round'].append(self.arrange_round)
            self.i_global += 1
            # Early termination
            if i > 3 and round(self.stress_lu['stress'][-1],3) == round(self.stress_lu['stress'][-2],3) and round(self.stress_lu['stress'][-2],3) == round(self.stress_lu['stress'][-3],3): break


        self.arrange_round += 1
        _updated_ = {}
        for i in range(len(df_pos)): _updated_[df_pos['node'][i]] = (df_pos['x'][i], df_pos['y'][i])
        return _updated_
