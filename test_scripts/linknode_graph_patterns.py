import networkx as nx

__name__ = 'linknode_graph_patterns'

class LinkNodeGraphPatterns(object):
    def __init__(self):
        self.types = ['binarytree', 'ring', 'mesh']    
    def createPattern(self, _type_, **kwargs):
        _fn_ = '__pattern_' + _type_ + '__'
        if hasattr(self, _fn_): return getattr(self, _fn_)(**kwargs)
        else:                   return None

    def __pattern_binarytree__(self, depth=5, **kwargs):
        return nx.balanced_tree(depth,2)

    def __pattern_ring__(self, spokes=20, **kwargs):
        g = nx.Graph()
        for i in range(spokes): g.add_edge(i, (i+1)%spokes)
        return g

    def __pattern_mesh__(self, xtiles=10, ytiles=10, **kwargs):
        g = nx.grid_2d_graph(xtiles,ytiles)
        return g
