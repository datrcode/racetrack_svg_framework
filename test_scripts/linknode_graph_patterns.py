import networkx as nx
import re

__name__ = 'linknode_graph_patterns'

class LinkNodeGraphPatterns(object):
    def __init__(self):
        self.types = []
        for _str_ in dir(self):
            _match_ = re.match('__pattern_(.*)__', _str_)
            if _match_ is not None: self.types.append(_match_.group(1))

    def __len__    (self):    return len(self.types)
    def __getitem__(self, i): return self.types[i]

    #
    # minimumStressFound() - minimum stress found so far
    #
    def minimumStressFound(self, 
                           _type_, 
                           distance_metric, # dijkstra or resistive
                           k=0):
        pass

    def createPattern(self, _type_, prefix='', **kwargs):
        if _type_ not in self.types: raise Exception(f'Unknown pattern type: {_type_}')
        _fn_ = '__pattern_' + _type_ + '__'
        return getattr(self, _fn_)(prefix=prefix,**kwargs)

    def __pattern_binarytree__(self, depth=5, prefix='', **kwargs):
        return nx.balanced_tree(depth,2)

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

    def __pattern_boxinbox__(self, **kwargs):
        pos = {'ul': (0.0, 0.0), 'um': (0.5, 0.0), 'ur': (1.0, 0.0), 
               'ml': (0.0, 0.5),                   'mr': (1.0, 0.5),
               'll': (0.0, 1.0), 'lm': (0.5, 1.0), 'lr': (1.0, 1.0),
               'inner_ul': (0.1, 0.1),             'inner_ur': (0.9, 0.1),
               'inner_ll': (0.1, 0.9),             'inner_lr': (0.9, 0.9)}

        def d(a, b): return (((pos[a][0]-pos[b][0])**2 + (pos[a][1]-pos[b][1])**2)**0.5)

        g   = nx.Graph()
        g.add_edge('ul', 'um', weight=d('ul', 'um')), g.add_edge('um', 'ur', weight=d('um', 'ur'))
        g.add_edge('ul', 'ml', weight=d('ul', 'ml')), g.add_edge('ml', 'll', weight=d('ml', 'll'))
        g.add_edge('ur', 'mr', weight=d('ur', 'mr')), g.add_edge('mr', 'lr', weight=d('mr', 'lr'))
        g.add_edge('ll', 'lm', weight=d('ll', 'lm')), g.add_edge('lm', 'lr', weight=d('lm', 'lr'))

        g.add_edge('ul', 'inner_ul', weight=d('ul', 'inner_ul'))
        g.add_edge('ur', 'inner_ur', weight=d('ur', 'inner_ur'))
        g.add_edge('lr', 'inner_lr', weight=d('lr', 'inner_lr'))
        g.add_edge('ll', 'inner_ll', weight=d('ll', 'inner_ll'))

        g.add_edge('inner_ul', 'inner_ur', weight=d('inner_ul', 'inner_ur'))
        g.add_edge('inner_ur', 'inner_lr', weight=d('inner_ur', 'inner_lr'))
        g.add_edge('inner_lr', 'inner_ll', weight=d('inner_lr', 'inner_ll'))
        g.add_edge('inner_ll', 'inner_ul', weight=d('inner_ll', 'inner_ul'))

        return g

    def __pattern_trianglestars__(self, **kwargs):
        g = nx.Graph()
        g.add_edge('a', 'b', weight=2.0)
        g.add_edge('a', 'c', weight=2.0)
        g.add_edge('b', 'c', weight=2.0)
        for i in range(40):
            g.add_edge('a', 'a'+str(i), weight=0.5)
            g.add_edge('b', 'b'+str(i), weight=0.5)
            g.add_edge('c', 'c'+str(i), weight=0.5)
        return g

    def __pattern_X__(self, **kwargs):
        g = nx.Graph()
        for i in range(30):
            g.add_edge(f'a{i}', f'a{i+1}')
            g.add_edge(f'b{i}', f'b{i+1}')
            g.add_edge(f'c{i}', f'c{i+1}')
            g.add_edge(f'd{i}', f'd{i+1}')
        g.add_edge('center', 'a0')
        g.add_edge('center', 'b0')
        g.add_edge('center', 'c0')
        g.add_edge('center', 'd0')
        return g

    def __pattern_Y__(self, **kwargs):
        g = nx.Graph()
        for i in range(30):
            g.add_edge(f'a{i}', f'a{i+1}')
            g.add_edge(f'b{i}', f'b{i+1}')
            g.add_edge(f'c{i}', f'c{i+1}')
        g.add_edge('center', 'a0')
        g.add_edge('center', 'b0')
        g.add_edge('center', 'c0')
        return g
    
    def __pattern_checker__(self, **kwargs):
        n = 5
        g = nx.Graph()
        for x in range(n):
            for y in range(n):
                for wo in ['ul', 'ur', 'll', 'lr', 'center']:
                    g.add_edge(f'{wo}_{x}_{y}',     f'{wo}_{x}_{y+1}')
                    g.add_edge(f'{wo}_{x}_{y}',     f'{wo}_{x+1}_{y}')
                    g.add_edge(f'{wo}_{x+1}_{y+1}', f'{wo}_{x}_{y+1}') # may add duplicate edges
                    g.add_edge(f'{wo}_{x+1}_{y+1}', f'{wo}_{x+1}_{y}') # may add duplicate edges

        g.add_edge(f'center_0_0',     f'ul_{n}_{n}')
        g.add_edge(f'center_{n}_{n}', f'lr_0_0')
        g.add_edge(f'center_{n}_0',   f'ur_0_{n}')
        g.add_edge(f'center_0_{n}',   f'll_{n}_0')

        return g
    
    # Figure 14(b) of "Drawing Graphs to Convey Proximity" Paper
    # by Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    # Originally from Eades [1984]
    def __pattern_cohen_fig_14a__(self, **kwargs):
        g = nx.Graph()
        # triangle 0
        g.add_edge('t0_a', 't0_b'), g.add_edge('t0_a', 't0_c'), g.add_edge('t0_b', 't0_c')
        # diamond
        g.add_edge('d_a', 'd_b'), g.add_edge('d_b', 'd_c'), g.add_edge('d_c', 'd_d'), g.add_edge('d_d', 'd_a')
        # pentagon
        g.add_edge('p_a', 'p_b'), g.add_edge('p_b', 'p_c'), g.add_edge('p_c', 'p_d')
        g.add_edge('p_d', 'p_e'), g.add_edge('p_e', 'p_a')
        # L
        g.add_edge('l_0', 'l_1'), g.add_edge('l_1', 'l_2')
        # connections back to diamond
        g.add_edge('d_a', 't0_c') # triangle 0
        g.add_edge('d_b', 'x0'), g.add_edge('d_b', 'x1'), g.add_edge('x0', 'x1')
        g.add_edge('d_c', 'l_0'), g.add_edge('d_d', 'l_1') # L
        g.add_edge('d_d', 'p_a') # pentagon
        return g

    # Figure 14(b) of "Drawing Graphs to Convey Proximity" Paper
    # by Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    # Originally from Kamada and Kawai [1989]
    def __pattern_cohen_fig_14b__(self, **kwargs):
        g = nx.Graph()
        _fms_ = 'a b c d e f g h h h k k m g g n p g q e e r t t'.split()
        _tos_ = 'c c d e f g h d j k l m l n p o o q f t r s s u'.split()
        for fm, to in zip(_fms_, _tos_): g.add_edge(fm, to)
        return g

    # Figure 5 of "Drawing Graphs to Convey Proximity" Paper
    # by Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    def __pattern_cohen_fig_5__(self, **kwargs):
        g = nx.Graph()
        for i in range(12): g.add_edge(f'{i}', f'{(i+1)%12}') # the ring
        for i in range(0, 12, 2):
            b0, b1 = f'{i}', f'{(i+1)%12}'
            for j in range(4):
                _nbor_ = f'{i}_{(i+1)%12}_{j}'
                g.add_edge(b0, _nbor_), g.add_edge(b1, _nbor_)
        return g
    
    # Figure 11 of "Drawing Graphs to Convey Proximity" Paper
    # by Cohen
    # ACM Transactions on Computer-Human Interaction, Vol. 4, No. 3, September 1997, Pages 197–229.
    # Originally from Davidson and Harel [1990] and Fruchterman and Reingold [1991]
    def __pattern_cohen_fig_11__(self, **kwargs):
        g     = nx.Graph()
        _cen_ = 'center'
        for i in range(10): g.add_edge(f'{_cen_}', f'{i}')
        for i in range(10): 
            n0, n1 = f'{i}', f'{(i+1)%10}'
            o = f'outer_{i}'
            g.add_edge(n0, n1)
            g.add_edge(n0, o), g.add_edge(n1, o)
            g.add_edge(_cen_, o)
        return g

    #
    # The “twin cubes” graph of Fruchterman and Reingold [1991]
    # ... or Figure 12 from Cohen
    #
    def __pattern_twin_cubes__(self, **kwargs):
        g = nx.Graph()
        for i in range(4): 
            # top, middle, and bottom square
            g.add_edge(f'a{i}', f'a{(i+1)%4}'), g.add_edge(f'b{i}', f'b{(i+1)%4}'), g.add_edge(f'c{i}', f'c{(i+1)%4}')
            # connections between top, middle, and bottom
            g.add_edge(f'a{i}', f'b{i}'), g.add_edge(f'b{i}', f'c{i}')
        return g

    #
    # "Dodecahedron" in Kamada and Kawai [1989] and in Fruchterman and Reingold [1991].
    # ... or Figure 13 from Cohen
    #
    def __pattern_dodecahedron__(self, **kwargs):
        g = nx.Graph()
        for i in range(5):
            j = (i+1)%5
            g.add_edge(f'top_{i}', f'top_{j}')
            g.add_edge(f'top_{i}', f'mid_{2*i}')
            g.add_edge(f'bot_{i}', f'bot_{j}')
            g.add_edge(f'bot_{i}', f'mid_{2*i+1}')
        for i in range(10):
            j = (i+1)%10
            g.add_edge(f'mid_{i}', f'mid_{j}')
        return g



