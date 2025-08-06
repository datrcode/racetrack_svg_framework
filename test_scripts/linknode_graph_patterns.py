import networkx as nx
import re

__name__ = 'linknode_graph_patterns'

class LinkNodeGraphPatterns(object):
    def __init__(self):
        self.types = []
        for _str_ in dir(self):
            _match_ = re.match('__pattern_(.*)__', _str_)
            if _match_ is not None: self.types.append(_match_.group(1))

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

    def __pattern_mesh__(self, xtiles=10, ytiles=10, prefix='', **kwargs):
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
                    if _nbor_ in _nodes_: g.add_edge(_node_, _nbor_)
        return g

    def __pattern_boxinbox__(self, **kwargs):
        pos = {'ul': (0.0, 0.0), 'um': (0.5, 0.0), 'ur': (1.0, 0.0), 
               'ml': (0.0, 0.5),                   'mr': (1.0, 0.5),
               'll': (0.0, 1.0), 'lm': (0.5, 1.0), 'lr': (1.0, 1.0),
               'inner_ul': (0.1, 0.1),             'inner_ur': (0.9, 0.1),
               'inner_ll': (0.1, 0.9),             'inner_lr': (0.9, 0.9)}

        def d(a, b): return 1.0 / (((pos[a][0]-pos[b][0])**2 + (pos[a][1]-pos[b][1])**2)**0.5)

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


