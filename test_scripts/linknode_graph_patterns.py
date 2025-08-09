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
                    if _nbor_ in _nodes_: g.add_edge(_node_, _nbor_)
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

    def __pattern_uscities__(self, **kwargs):

        pos = {
            'new_york': (40.7128, -74.0059),
            'los_angeles': (34.0522, -118.2437),
            'san_francisco': (37.7749, -122.4194),
            'washington_dc': (38.9072, -77.0369),
            'chicago': (41.8781, -87.6298),
            'phoenix': (33.4484, -112.0740),
            'san_diego': (32.7157, -117.1611),
            'san_antonio': (29.4241, -98.4936),
            'dallas': (32.7767, -96.7970),
            'houston': (29.7604, -95.3698),
            'kansas_city': (39.0997, -94.5786),
            'denver': (39.7392, -104.9903),
            'minneapolis': (44.9778, -93.2650),
            'seattle': (47.6062, -122.3321),
            'boston': (42.3601, -71.0589),
            'orlando': (28.5383, -81.3792),
            'miami': (25.7617, -80.1918),
            'atlanta': (33.7490, -84.3880),
            'tampa': (27.9506, -82.4572),
            'tallahassee': (30.4383, -84.2807),
            'salt_lake_city': (40.7608, -111.8910),
            'reno': (39.5296, -119.8135),
            'las_vegas': (36.1699, -115.1398),
            'sacramento': (38.5816, -121.4944),
            'spokane': (47.6782, -117.4260),
            'boise': (43.6189, -116.2146),
            'pittsburgh': (40.4406, -79.9959),
            'philadelphia': (39.9526, -75.1652),
            'baltimore': (39.2904, -76.6122),
            'richmond': (37.54, -77.46),
            'providence': (41.8244, -71.4128),
            'bangor': (44.8378, -68.7798),
            'jacksonville': (30.3322, -81.6557),
        }
        def d(a, b): return (((pos[a][0]-pos[b][0])**2 + (pos[a][1]-pos[b][1])**2)**0.5)

        _connects_ = {
            'new_york': ['boston','philadelphia', 'providence'],
            'boston': ['providence', 'new_york', 'bangor'],
            'philadelphia': ['pittsburgh', 'baltimore'],
            'pittsburgh': ['chicago', 'new_york', 'bangor'],
            'baltimore': ['washington_dc', 'pittsburgh'],
            'washington_dc': ['richmond', 'atlanta', 'chicago'],
            'chicago': ['minneapolis', 'denver', 'boise', 'kansas_city'],
            'atlanta': ['tallahassee', 'orlando', 'kansas_city', 'richmond', 'jacksonville'],
            'orlando': ['tampa', 'miami', 'tallahassee', 'jacksonville'],
            'tampa': ['miami', 'tallahassee'],
            'denver': ['salt_lake_city', 'phoenix', 'kansas_city', 'boise', 'reno', 'las_vegas'],
            'dallas': ['houston', 'san_antonio', 'denver', 'tallahassee', 'kansas_city'],
            'seattle': ['spokane', 'san_francisco'],
            'spokane': ['boise', 'minneapolis'],
            'san_francisco': ['sacramento', 'los_angeles'],
            'los_angeles': ['sacramento', 'san_diego', 'las_vegas', 'phoenix'],
            'reno': ['salt_lake_city', 'las_vegas', 'spokane', 'sacramento'],
            'phoenix':['dallas', 'san_antonio'],
            'houston': ['san_antonio'],
            'boise': ['salt_lake_city', 'reno'],
            'miami': ['jacksonville'],
        }

        _seen_ = set()
        g = nx.Graph()
        for _node_ in pos.keys():
            for _nbor_ in pos.keys():
                if _node_ == _nbor_: continue
                _seen_.add(_node_), _seen_.add(_nbor_)
                g.add_edge(_node_, _nbor_, weight=d(_node_, _nbor_))
        '''
        for _node_ in _connects_.keys():
            for _nbor_ in _connects_[_node_]:
                if _node_ == _nbor_: continue
                _seen_.add(_node_), _seen_.add(_nbor_)
                g.add_edge(_node_, _nbor_, weight=d(_node_, _nbor_))
        '''

        if _seen_ != set(pos.keys()): 
            print('Missing nodes:', set(pos.keys()) - _seen_)
            raise Exception('Graph is not fully connected')
        return g

    def __pattern_trianglestars__(self, **kwargs):
        g = nx.Graph()
        g.add_edge('a', 'b'), g.add_edge('a', 'c'), g.add_edge('b', 'c')
        for i in range(40):
            g.add_edge('a', 'a'+str(i))
            g.add_edge('b', 'b'+str(i))
            g.add_edge('c', 'c'+str(i))
        return g

