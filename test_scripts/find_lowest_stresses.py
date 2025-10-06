import polars as pl
import rtsvg
import time
from linknode_graph_patterns import LinkNodeGraphPatterns
from convey_proximity_layout import ConveyProximityLayout

_base_filename_ = str(int(time.time()*1_000)) + '_stresses'
print(_base_filename_)

_lu_ = {'stress':[], 'algo':[], 'trial':[], 'time':[], 'k':[], 'pattern':[]}

_patterns_ = LinkNodeGraphPatterns()
for i in range(len(_patterns_)):
    _pattern_ = _patterns_[i]
    print(_pattern_,end=' ')
    _g_       = _patterns_.createPattern(_pattern_)
    _stress_min_, _algo_min_, _algo_min_str_ = None, None, None
    for k in range(2):
        for _trial_ in range(4):
            for _algo_no_ in range(2):
                t0 = time.time()
                if _algo_no_ == 0: _algo_, _algo_str_  = rtsvg.PolarsForceDirectedLayout(_g_, k=k),                      'PFDL'
                else:              _algo_, _algo_str_  = ConveyProximityLayout(_g_, k=k, use_resistive_distances=False), 'CPL'
                t1 = time.time()
                _pos_    = _algo_.results()
                _stress_ = _algo_.stress()
                _lu_['stress'].append(_stress_), _lu_['algo'].append(_algo_str_), _lu_['trial'].append(_trial_), _lu_['time'].append(t1-t0), _lu_['k'].append(k), _lu_['pattern'].append(_pattern_)
                if _stress_min_ is None or _stress_min_ > _stress_: 
                    print(f'{_algo_str_} ',end='')
                    _stress_min_, _algo_min_, _algo_str_min_ = _stress_, _algo_, _algo_str_
                    _lu_pos_ = {'node':[], 'x':[], 'y':[]}
                    for _node_ in _pos_: _lu_pos_['node'].append(_node_), _lu_pos_['x'].append(_pos_[_node_][0]), _lu_pos_['y'].append(_pos_[_node_][1])
    _df_pos_ = pl.DataFrame(_lu_pos_)
    _df_pos_.write_csv(f'{_base_filename_}_{_pattern_}.csv')
    print()
_df_ = pl.DataFrame(_lu_)
_df_.write_csv(f'{_base_filename_}.csv')
