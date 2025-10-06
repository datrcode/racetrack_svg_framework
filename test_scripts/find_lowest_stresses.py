import polars as pl
import rtsvg
import time
from linknode_graph_patterns import LinkNodeGraphPatterns
from convey_proximity_layout import ConveyProximityLayout

_patterns_ = LinkNodeGraphPatterns()
for i in range(len(_patterns_)):
    _pattern_ = _patterns_[i]
    print(_pattern_,end=' ')
    _g_       = _patterns_.createPattern(_pattern_)
    _stress_min_, _algo_min_, _algo_min_str_ = None, None, None
    for k in range(3):
        for _trial_ in range(4):
            for _algo_no_ in range(2):
                t0 = time.time()
                if _algo_no_ == 0: _algo_, _algo_str_  = rtsvg.PolarsForceDirectedLayout(_g_, k=k),                      'PFDL'
                else:              _algo_, _algo_str_  = ConveyProximityLayout(_g_, k=k, use_resistive_distances=False), 'CPL'
                t1 = time.time()
                _stress_ = _algo_.stress()
                if _stress_min_ is None or _stress_min_ > _stress_: 
                    print(f'{_algo_str_} ',end='')
                    _stress_min_, _algo_min_, _algo_str_min_ = _stress_, _algo_, _algo_str_
    print()
