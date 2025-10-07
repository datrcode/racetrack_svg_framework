import networkx as nx
import re

__name__ = 'linknode_graph_patterns'

class LinkNodeGraphPatterns(object):
    def __init__(self):
        self.types = []
        for _str_ in dir(self):
            _match_ = re.match('__pattern_(.*)__', _str_)
            if _match_ is not None: self.types.append(_match_.group(1))
        self.stress_min = {}
        for k in range(2):
            self.stress_min[k] = {}
            for distance_metric in ['dijkstra', 'resistive']: self.stress_min[k][distance_metric] = {}

        _empirical_results_ = [
{"k":0, "distance_metric":'dijkstra', "pattern":'mesh',          "stress_min":0.010369263755955656}, {"k":1, "distance_metric":'dijkstra', "pattern":'mesh',          "stress_min":0.013293017494576767},
{"k":0, "distance_metric":'dijkstra', "pattern":'checker',       "stress_min":0.005664990413741567}, {"k":1, "distance_metric":'dijkstra', "pattern":'checker',       "stress_min":0.008883382169004963},
{"k":0, "distance_metric":'dijkstra', "pattern":'twin_cubes',    "stress_min":0.06221410493122583},  {"k":1, "distance_metric":'dijkstra', "pattern":'twin_cubes',    "stress_min":0.079472668165888},
{"k":0, "distance_metric":'dijkstra', "pattern":'boxinbox',      "stress_min":0.01447779149306904},  {"k":1, "distance_metric":'dijkstra', "pattern":'boxinbox',      "stress_min":0.016053802417961435},
{"k":0, "distance_metric":'dijkstra', "pattern":'cohen_fig_11',  "stress_min":0.0811873537809558},   {"k":1, "distance_metric":'dijkstra', "pattern":'cohen_fig_11',  "stress_min":0.08903415275462864},
{"k":1, "distance_metric":'dijkstra', "pattern":'X',             "stress_min":0.02217642443451038},  {"k":0, "distance_metric":'dijkstra', "pattern":'X',             "stress_min":0.023052634013478644},
{"k":0, "distance_metric":'dijkstra', "pattern":'cohen_fig_14b', "stress_min":0.01621650561726668},  {"k":1, "distance_metric":'dijkstra', "pattern":'cohen_fig_14b', "stress_min":0.020594855267259012},
{"k":0, "distance_metric":'dijkstra', "pattern":'cohen_fig_14a', "stress_min":0.013212881076396985}, {"k":1, "distance_metric":'dijkstra', "pattern":'cohen_fig_14a', "stress_min":0.016426480814406175},
{"k":0, "distance_metric":'dijkstra', "pattern":'trianglestars', "stress_min":0.026355947825847624}, {"k":1, "distance_metric":'dijkstra', "pattern":'trianglestars', "stress_min":0.04820164626380459},
{"k":0, "distance_metric":'dijkstra', "pattern":'dodecahedron',  "stress_min":0.08044964406979078},  {"k":1, "distance_metric":'dijkstra', "pattern":'dodecahedron',  "stress_min":0.0889790908811913},
{"k":0, "distance_metric":'dijkstra', "pattern":'binarytree',    "stress_min":0.07365983175066931},  {"k":1, "distance_metric":'dijkstra', "pattern":'binarytree',    "stress_min":0.08357673128571591},
{"k":0, "distance_metric":'dijkstra', "pattern":'ring',          "stress_min":0.015327039145082866}, {"k":1, "distance_metric":'dijkstra', "pattern":'ring',          "stress_min":0.017092802044942786},
{"k":0, "distance_metric":'dijkstra', "pattern":'Y',             "stress_min":0.0017536710968615503},{"k":1, "distance_metric":'dijkstra', "pattern":'Y',             "stress_min":0.0030471103667565735},
{"k":0, "distance_metric":'dijkstra', "pattern":'cohen_fig_5',   "stress_min":0.018539759586453625}, {"k":1, "distance_metric":'dijkstra', "pattern":'cohen_fig_5',   "stress_min":0.028864567878219248},
]
        for _d_ in _empirical_results_: self.stress_min[_d_['k']][_d_['distance_metric']][_d_['pattern']] = _d_['stress_min']

    def __len__    (self):    return len(self.types)
    def __getitem__(self, i): return self.types[i]

    #
    # minimumStressFound() - minimum stress found so far
    #
    def minimumStressFound(self, 
                           _type_,                     # graph pattern
                           distance_metric='dijkstra', # dijkstra or resistive
                           k=0):
        return self.stress_min[k][distance_metric][_type_]

    def createPattern(self, _type_, prefix='', **kwargs):
        if _type_ not in self.types: raise Exception(f'Unknown pattern type: {_type_}')
        _fn_ = '__pattern_' + _type_ + '__'
        return getattr(self, _fn_)(prefix=prefix,**kwargs)

    def __pattern_binarytree__(self, depth=5, prefix='', **kwargs):
        _g_     = nx.balanced_tree(depth,2)
        _g_str_ = nx.Graph()
        for _node_ in _g_.nodes:
            for _nbor_ in _g_.neighbors(_node_):
                _g_str_.add_edge(prefix+str(_node_), prefix+str(_nbor_))
        return _g_str_

    def __pattern_ring__(self, spokes=20, prefix='', **kwargs):
        g = nx.Graph()
        for i in range(spokes): g.add_edge(prefix+str(i), prefix+str((i+1)%spokes))
        return g

    def __pattern_mesh__(self, xtiles=8, ytiles=8, prefix='', **kwargs):
        g       = nx.Graph()
        _nodes_ = set()
        for _y_ in range(ytiles+1):
            for _x_ in range(xtiles+1):
                _node_ = f'{prefix}node_{_y_}_{_x_}'
                _nodes_.add(_node_)
        for _node_ in _nodes_:
            _y_, _x_ = int(_node_.split('_')[-1]), int(_node_.split('_')[-2])
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if (dx == 0 and dy == 0) or (abs(dx) == 1 and abs(dy) == 1): continue
                    _nbor_ = f'{prefix}node_{_y_+dy}_{_x_+dx}'
                    if _nbor_ in _nodes_: g.add_edge(_node_, _nbor_, weight=1.0)
        return g

    def __pattern_boxinbox__(self, prefix='', **kwargs):
        pos = {f'{prefix}ul': (0.0, 0.0), f'{prefix}um': (0.5, 0.0), f'{prefix}ur': (1.0, 0.0), 
               f'{prefix}ml': (0.0, 0.5),                            f'{prefix}mr': (1.0, 0.5),
               f'{prefix}ll': (0.0, 1.0), f'{prefix}lm': (0.5, 1.0), f'{prefix}lr': (1.0, 1.0),
               f'{prefix}inner_ul': (0.1, 0.1),                      f'{prefix}inner_ur': (0.9, 0.1),
               f'{prefix}inner_ll': (0.1, 0.9),                      f'{prefix}inner_lr': (0.9, 0.9)}

        def d(a, b): return (((pos[a][0]-pos[b][0])**2 + (pos[a][1]-pos[b][1])**2)**0.5)

        g   = nx.Graph()
        g.add_edge(f'{prefix}ul', f'{prefix}um', weight=d(f'{prefix}ul', f'{prefix}um')), g.add_edge(f'{prefix}um', f'{prefix}ur', weight=d(f'{prefix}um', f'{prefix}ur'))
        g.add_edge(f'{prefix}ul', f'{prefix}ml', weight=d(f'{prefix}ul', f'{prefix}ml')), g.add_edge(f'{prefix}ml', f'{prefix}ll', weight=d(f'{prefix}ml', f'{prefix}ll'))
        g.add_edge(f'{prefix}ur', f'{prefix}mr', weight=d(f'{prefix}ur', f'{prefix}mr')), g.add_edge(f'{prefix}mr', f'{prefix}lr', weight=d(f'{prefix}mr', f'{prefix}lr'))
        g.add_edge(f'{prefix}ll', f'{prefix}lm', weight=d(f'{prefix}ll', f'{prefix}lm')), g.add_edge(f'{prefix}lm', f'{prefix}lr', weight=d(f'{prefix}lm', f'{prefix}lr'))

        g.add_edge(f'{prefix}ul', f'{prefix}inner_ul', weight=d(f'{prefix}ul', f'{prefix}inner_ul'))
        g.add_edge(f'{prefix}ur', f'{prefix}inner_ur', weight=d(f'{prefix}ur', f'{prefix}inner_ur'))
        g.add_edge(f'{prefix}lr', f'{prefix}inner_lr', weight=d(f'{prefix}lr', f'{prefix}inner_lr'))
        g.add_edge(f'{prefix}ll', f'{prefix}inner_ll', weight=d(f'{prefix}ll', f'{prefix}inner_ll'))

        g.add_edge(f'{prefix}inner_ul', f'{prefix}inner_ur', weight=d(f'{prefix}inner_ul', f'{prefix}inner_ur'))
        g.add_edge(f'{prefix}inner_ur', f'{prefix}inner_lr', weight=d(f'{prefix}inner_ur', f'{prefix}inner_lr'))
        g.add_edge(f'{prefix}inner_lr', f'{prefix}inner_ll', weight=d(f'{prefix}inner_lr', f'{prefix}inner_ll'))
        g.add_edge(f'{prefix}inner_ll', f'{prefix}inner_ul', weight=d(f'{prefix}inner_ll', f'{prefix}inner_ul'))

        return g

    def __pattern_trianglestars__(self, prefix='',**kwargs):
        g = nx.Graph()
        g.add_edge(f'{prefix}a', f'{prefix}b', weight=2.0)
        g.add_edge(f'{prefix}a', f'{prefix}c', weight=2.0)
        g.add_edge(f'{prefix}b', f'{prefix}c', weight=2.0)
        for i in range(40):
            g.add_edge(f'{prefix}a', f'{prefix}a'+str(i), weight=0.5)
            g.add_edge(f'{prefix}b', f'{prefix}b'+str(i), weight=0.5)
            g.add_edge(f'{prefix}c', f'{prefix}c'+str(i), weight=0.5)
        return g

    def __pattern_X__(self, prefix='', **kwargs):
        g = nx.Graph()
        for i in range(30):
            g.add_edge(f'{prefix}a{i}', f'{prefix}a{i+1}')
            g.add_edge(f'{prefix}b{i}', f'{prefix}b{i+1}')
            g.add_edge(f'{prefix}c{i}', f'{prefix}c{i+1}')
            g.add_edge(f'{prefix}d{i}', f'{prefix}d{i+1}')
        g.add_edge(f'{prefix}center', f'{prefix}a0')
        g.add_edge(f'{prefix}center', f'{prefix}b0')
        g.add_edge(f'{prefix}center', f'{prefix}c0')
        g.add_edge(f'{prefix}center', f'{prefix}d0')
        return g

    def __pattern_Y__(self, prefix='', **kwargs):
        g = nx.Graph()
        for i in range(30):
            g.add_edge(f'{prefix}a{i}', f'{prefix}a{i+1}')
            g.add_edge(f'{prefix}b{i}', f'{prefix}b{i+1}')
            g.add_edge(f'{prefix}c{i}', f'{prefix}c{i+1}')
        g.add_edge(f'{prefix}center', f'{prefix}a0')
        g.add_edge(f'{prefix}center', f'{prefix}b0')
        g.add_edge(f'{prefix}center', f'{prefix}c0')
        return g
    
    def __pattern_checker__(self, prefix='', **kwargs):
        n = 5
        g = nx.Graph()
        for x in range(n):
            for y in range(n):
                for wo in ['ul', 'ur', 'll', 'lr', 'center']:
                    g.add_edge(f'{prefix}{wo}_{x}_{y}',     f'{prefix}{wo}_{x}_{y+1}')
                    g.add_edge(f'{prefix}{wo}_{x}_{y}',     f'{prefix}{wo}_{x+1}_{y}')
                    g.add_edge(f'{prefix}{wo}_{x+1}_{y+1}', f'{prefix}{wo}_{x}_{y+1}') # may add duplicate edges
                    g.add_edge(f'{prefix}{wo}_{x+1}_{y+1}', f'{prefix}{wo}_{x+1}_{y}') # may add duplicate edges

        g.add_edge(f'{prefix}center_0_0',     f'{prefix}ul_{n}_{n}')
        g.add_edge(f'{prefix}center_{n}_{n}', f'{prefix}lr_0_0')
        g.add_edge(f'{prefix}center_{n}_0',   f'{prefix}ur_0_{n}')
        g.add_edge(f'{prefix}center_0_{n}',   f'{prefix}ll_{n}_0')

        return g
    
    # Figure 14(b) of "Drawing Graphs to Convey Proximity" Paper
    # by Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    # Originally from Eades [1984]
    def __pattern_cohen_fig_14a__(self, prefix='', **kwargs):
        g = nx.Graph()
        # triangle 0
        g.add_edge(f'{prefix}t0_a', f'{prefix}t0_b'), g.add_edge(f'{prefix}t0_a', f'{prefix}t0_c'), g.add_edge(f'{prefix}t0_b', f'{prefix}t0_c')
        # diamond
        g.add_edge(f'{prefix}d_a', f'{prefix}d_b'), g.add_edge(f'{prefix}d_b', f'{prefix}d_c'), g.add_edge(f'{prefix}d_c', f'{prefix}d_d'), g.add_edge(f'{prefix}d_d', f'{prefix}d_a')
        # pentagon
        g.add_edge(f'{prefix}p_a', f'{prefix}p_b'), g.add_edge(f'{prefix}p_b', f'{prefix}p_c'), g.add_edge(f'{prefix}p_c', f'{prefix}p_d')
        g.add_edge(f'{prefix}p_d', f'{prefix}p_e'), g.add_edge(f'{prefix}p_e', f'{prefix}p_a')
        # L
        g.add_edge(f'{prefix}l_0', f'{prefix}l_1'), g.add_edge(f'{prefix}l_1', f'{prefix}l_2')
        # connections back to diamond
        g.add_edge(f'{prefix}d_a', f'{prefix}t0_c') # triangle 0
        g.add_edge(f'{prefix}d_b', f'{prefix}x0'), g.add_edge(f'{prefix}d_b', f'{prefix}x1'), g.add_edge(f'{prefix}x0', f'{prefix}x1')
        g.add_edge(f'{prefix}d_c', f'{prefix}l_0'), g.add_edge(f'{prefix}d_d', f'{prefix}l_1') # L
        g.add_edge(f'{prefix}d_d', f'{prefix}p_a') # pentagon
        return g

    # Figure 14(b) of "Drawing Graphs to Convey Proximity" Paper
    # by Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    # Originally from Kamada and Kawai [1989]
    def __pattern_cohen_fig_14b__(self, prefix='', **kwargs):
        g = nx.Graph()
        _fms_ = 'a b c d e f g h h h k k m g g n p g q e e r t t'.split()
        _tos_ = 'c c d e f g h d j k l m l n p o o q f t r s s u'.split()
        for fm, to in zip(_fms_, _tos_): g.add_edge(prefix+fm, prefix+to)
        return g

    # Figure 5 of "Drawing Graphs to Convey Proximity" Paper
    # by Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    def __pattern_cohen_fig_5__(self, prefix='', **kwargs):
        g = nx.Graph()
        for i in range(12): g.add_edge(f'{prefix}{i}', f'{prefix}{(i+1)%12}') # the ring
        for i in range(0, 12, 2):
            b0, b1 = f'{prefix}{i}', f'{prefix}{(i+1)%12}'
            for j in range(4):
                _nbor_ = f'{prefix}{i}_{(i+1)%12}_{j}'
                g.add_edge(b0, _nbor_), g.add_edge(b1, _nbor_)
        return g
    
    # Figure 11 of "Drawing Graphs to Convey Proximity" Paper
    # by Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    # Originally from Davidson and Harel [1990] and Fruchterman and Reingold [1991]
    def __pattern_cohen_fig_11__(self, prefix='', **kwargs):
        g     = nx.Graph()
        _cen_ = 'center'
        for i in range(10): g.add_edge(f'{prefix}{_cen_}', f'{prefix}{i}')
        for i in range(10): 
            n0, n1 = f'{prefix}{i}', f'{prefix}{(i+1)%10}'
            o = f'{prefix}outer_{i}'
            g.add_edge(n0, n1)
            g.add_edge(n0, o), g.add_edge(n1, o)
            g.add_edge(_cen_, o)
        return g

    #
    # The “twin cubes” graph of Fruchterman and Reingold [1991]
    # ... or Figure 12 from Cohen
    #
    def __pattern_twin_cubes__(self, prefix='', **kwargs):
        g = nx.Graph()
        for i in range(4): 
            # top, middle, and bottom square
            g.add_edge(f'{prefix}a{i}', f'{prefix}a{(i+1)%4}'), g.add_edge(f'{prefix}b{i}', f'{prefix}b{(i+1)%4}'), g.add_edge(f'{prefix}c{i}', f'{prefix}c{(i+1)%4}')
            # connections between top, middle, and bottom
            g.add_edge(f'{prefix}a{i}', f'{prefix}b{i}'), g.add_edge(f'{prefix}b{i}', f'{prefix}c{i}')
        return g

    #
    # "Dodecahedron" in Kamada and Kawai [1989] and in Fruchterman and Reingold [1991].
    # ... or Figure 13 from Cohen
    #
    def __pattern_dodecahedron__(self, prefix='', **kwargs):
        g = nx.Graph()
        for i in range(5):
            j = (i+1)%5
            g.add_edge(f'{prefix}top_{i}', f'{prefix}top_{j}')
            g.add_edge(f'{prefix}top_{i}', f'{prefix}mid_{2*i}')
            g.add_edge(f'{prefix}bot_{i}', f'{prefix}bot_{j}')
            g.add_edge(f'{prefix}bot_{i}', f'{prefix}mid_{2*i+1}')
        for i in range(10):
            j = (i+1)%10
            g.add_edge(f'{prefix}mid_{i}', f'{prefix}mid_{j}')
        return g



