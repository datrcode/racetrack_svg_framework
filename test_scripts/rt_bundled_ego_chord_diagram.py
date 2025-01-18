import polars as pl
import pandas as pd
import numpy as np
import networkx as nx
import random

__name__ = "rt_bundled_ego_chord_diagram"

class RTBundledEgoChordDiagram(object):
    def __init__(self,
                 rt_self,
                 df,
                 # Graph Information
                 relationships,
                 pos                    = None,
                 # Render Information
                 color_by               = None,
                 count_by               = None,
                 count_by_set           = False,
                 # Visualization Hints
                 focal_node             = None,  # focus node of the visualization
                 selected_nodes         = None,  # a list of nodes to highlight
                 high_degree_node_count = 5,     # of high degree nodes to remove from communities
                 chord_diagram_points   = 4,     # of entry / exit points for the chord diagrams
                 node_communities       = None,  # a list of sets -- within the sets are the nodes
                 # Geometry
                 chord_diagram_min_r    = 40,    # minimum radius of the chord diagrams
                 chord_diagram_max_r    = 100,   # maximum radius of the chord diagrams
                 node_r                 = 10,    # radius of the individial nodes
                 clouds_r               = 10,    # radius of the clouds
                 widget_id              = None,
                 w                      = 768,
                 h                      = 768):
        # Copy the parameters into local variables
        self.rt_self            = rt_self
        self.relationships_orig = relationships
        self.pos                = pos
        self.color_by           = color_by
        self.count_by           = count_by
        self.count_by_set       = count_by_set
        self.w                  = w
        self.h                  = h
        self.widget_id          = widget_id

        # Make a widget_id if it's not set already
        if self.widget_id is None: self.widget_id = "bundled_ego_chord_diagram_" + str(random.randint(0,65535))

        # Copy the dataframe (columns are going to be added for the rendering)
        self.df = rt_self.copyDataFrame(df)

        #
        # Field Transformations
        #
        # Apply count-by transforms
        if self.count_by is not None and rt_self.isTField(self.count_by): self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)
        # Apply color-by transforms
        if self.color_by is not None and rt_self.isTField(self.color_by): self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)
        # Apply node field transforms
        for _edge_ in self.relationships_orig:
            for _node_ in _edge_:
                if type(_node_) == str:
                    if rt_self.isTField(_node_) and rt_self.tFieldApplicableField(_node_) in self.df.columns:
                        self.df,_throwaway_ = rt_self.applyTransform(self.df, _node_)
                else:
                    for _tup_part_ in _node_:
                        if rt_self.isTField(_tup_part_) and rt_self.tFieldApplicableField(_tup_part_) in self.df.columns:
                            self.df,_throwaway_ = rt_self.applyTransform(self.df, _tup_part_)

        # Create concatenated fields for the tuple nodes
        self.relationships, i = [], 0
        for _edge_ in self.relationships_orig:
            _fm_ = _edge_[0]
            _to_ = _edge_[1]
            if type(_fm_) == tuple or type(_to_) == tuple:
                new_fm, new_to = _fm_, _to_

                if type(_fm_) == tuple:
                    new_fm = f'__fm{i}__'
                    self.df = self.rt_self.createConcatColumn(self.df, _fm_, new_fm)

                if type(_to_) == tuple:
                    new_to = f'__to{i}__'
                    self.df = self.rt_self.createConcatColumn(self.df, _to_, new_to)

                if   len(_edge_) == 2: self.relationships.append((new_fm, new_to))
                elif len(_edge_) == 3: self.relationships.append((new_fm, new_to, _edge_[2]))
                else:                  raise Exception(f'RTBundledEgoChordDiagram(): relationship tuples should have two or three parts "{_edge_}"')
            else:
                if   len(_edge_) == 2: self.relationships.append((_fm_, _to_))
                elif len(_edge_) == 3: self.relationships.append((_fm_, _to_, _edge_[2]))
                else:                  raise Exception(f'RTBundledEgoChordDiagram(): relationship tuples should have two or three parts "{_edge_}"')
            i += 1

        # Align the dataset so that there's only __fm__ and __to__
        # ... if any to / froms are integers, they will be converted to strings (polars has strongly typed columns)
        columns_to_keep = set(['__fm__','__to__'])
        if self.count_by is not None: columns_to_keep.add(self.count_by)
        if self.color_by is not None: columns_to_keep.add(self.color_by)
        _partials_ = []
        for _relationship_ in self.relationships:
            _fm_, _to_ = _relationship_[0], _relationship_[1]
            _df_       = self.df.drop_nulls(subset=[_fm_, _to_]) \
                                .with_columns(pl.col(_fm_).cast(pl.String).alias('__fm__'), 
                                              pl.col(_to_).cast(pl.String).alias('__to__'))
            _partials_.append(_df_.drop(set(_df_.columns) - columns_to_keep))
        self.df_aligned = pl.concat(_partials_)

        # Create the graph representation
        self.g = self.rt_self.createNetworkXGraph(self.df_aligned, [('__fm__','__to__')], count_by=self.count_by, count_by_set=self.count_by_set)

        # Create the layout
        if self.pos is None:
            # Use the supplied communities (or create them via community detection)
            # ... make sure that whichever path is followed, the state is consistent
            if node_communities is None:
                # Find the high degree nodes first
                degree_sorter = []
                for _tuple_ in self.g.degree: degree_sorter.append(_tuple_)
                degree_sorter.sort(key=lambda x: x[1], reverse=True)
                if len(degree_sorter) > high_degree_node_count: self.high_degree_nodes = [ _tuple_[0] for _tuple_ in degree_sorter[:high_degree_node_count] ]
                else:                                           self.high_degree_nodes = []
                # Find the communities (of the remaining nodes)
                self.g_minus_high_degree_nodes = self.g.copy()
                self.g_minus_high_degree_nodes.remove_nodes_from(self.high_degree_nodes)
                self.communities = list(nx.community.louvain_communities(self.g_minus_high_degree_nodes))
            else:
                self.high_degree_nodes         = []
                self.g_minus_high_degree_nodes = self.g.copy()
                self.communities               = node_communities
        else: # pos is supplied -- any nodes with the same position are in the same community
            self.high_degree_nodes         = []
            self.g_minus_high_degree_nodes = self.g.copy()
            # Make sure every node has a position
            for _node_ in self.g.nodes():
                if _node_ not in self.pos: self.pos[_node_] = [random.random(),random.random()]
            # Find the communities (of the remaining nodes)
            xy_to_nodes      = {}
            for _node_ in self.g.nodes():
                _xy_ = self.pos[_node_]
                if _xy_ not in xy_to_nodes: xy_to_nodes[_xy_] = set()
                xy_to_nodes[_xy_].add(_node_)
            self.communities = [ set(xy_to_nodes[_xy_]) for _xy_ in xy_to_nodes ]

        # State
        self.last_render   = None

    #
    # __repr_svg__() - SVG Representation
    #
    def _repr_svg_(self):
        if self.last_render is None: self.renderSVG()
        return self.last_render

    #
    # renderSVG() - render the SVG
    #
    def renderSVG(self):
        bg_color = self.rt_self.co_mgr.getTVColor('background','default')
        svg = [f'<svg id="{self.widget_id}" x="0" y="0" width="{self.w}" height="{self.h}">',
               f'<rect width="{self.w}" height="{self.h}" x="0" y="0" fill="{bg_color}" stroke="{bg_color}" />']

        svg.append('</svg>')
        self.last_render = ''.join(svg)
        return self.last_render