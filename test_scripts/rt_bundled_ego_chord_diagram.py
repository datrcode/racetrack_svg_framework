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
                 min_intra_circle_d     = 10,    # minimum distance between circles
                 chord_diagram_min_r    = 40,    # minimum radius of the chord diagrams
                 chord_diagram_max_r    = 100,   # maximum radius of the chord diagrams
                 chord_diagram_opacity  = 0.25,  # opacity of edges within the chord diagram
                 chord_diagram_node_h   = 4,     # height of the nodes in the chord diagram
                 shrink_circles_by      = 5,     # how much to shrink the circles by after the layout
                 node_r                 = 10,    # radius of the individial nodes
                 clouds_r               = 10,    # radius of the clouds
                 widget_id              = None,
                 x_ins                  = 30,
                 y_ins                  = 30,
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
        self.df_aligned = pl.concat(_partials_) # maybe we should do the counting and coloring here...

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
                if selected_nodes is not None: self.high_degree_nodes.extend(selected_nodes)
                if focal_node is not None:     self.high_degree_nodes.append(focal_node)
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
        
        # Create the community lookup so that we can do the collapse
        self.community_lookup, self.node_to_community, self.community_size_min, self.community_size_max = {}, {}, None, None
        for _community_ in self.communities:
            # Name will be __community_<community_number>_low_high_
            _low_, _high_ = None, None
            for _member_ in _community_:
                if _low_ is None: _low_ = _high_ = _member_
                if _low_  > _member_: _low_  = _member_
                if _high_ < _member_: _high_ = _member_
            _community_name_ = f'__community_{len(_community_)}_{_low_}_{_high_}__'
            self.community_lookup[_community_name_] = _community_
            for _member_ in _community_: self.node_to_community[_member_] = _community_name_
            if self.community_size_min is None: self.community_size_min = self.community_size_max = len(_community_)
            if len(_community_) < self.community_size_min: self.community_size_min = len(_community_)
            if len(_community_) > self.community_size_max: self.community_size_max = len(_community_)
        for _node_ in self.high_degree_nodes:
            self.node_to_community[_node_] = _node_
            self.community_lookup[_node_]  = set([_node_])

        # Collapse the communities
        self.df_communities  = rt_self.collapseDataFrameGraphByClusters(self.df_aligned, [('__fm__','__to__')], self.community_lookup)
        self.g_communities   = rt_self.createNetworkXGraph(self.df_communities, [('__fm__','__to__')])
        
        # Fill in the community positions and the regular positions ... 
        if self.pos is None:
            self.pos_communities = nx.spring_layout(self.g_communities)
        else:
            self.pos_communities = {}
            for _node_ in self.g.nodes():
                if _node_ in self.g_communities.nodes(): # it's a regular node... just copy over the position information
                    self.pos_communities[_node_] = self.pos[_node_]
            for _community_name_ in self.community_lookup:
                if len(self.community_lookup[_community_name_]) > 1:
                    _node_ = list(self.community_lookup[_community_name_])[0]
                    self.pos_communities[_community_name_] = self.pos[_node_]
        
        # Create an ordered list of community names and their associated circles
        self.community_names = list(self.community_lookup.keys())

        # Figure out how to position the nodes on the screen
        no_overlaps, _attempts_, self.circles = False, 1, None
        while no_overlaps is False and _attempts_ < 10:
            _attempts_ += 1
            x_min, y_min, x_max, y_max = self.rt_self.positionExtents(self.pos_communities)
            wxToSx  = lambda wx: (x_ins+chord_diagram_max_r) + (w - 2*(x_ins+chord_diagram_max_r))*(wx-x_min)/(x_max-x_min)
            wyToSy  = lambda wy: (y_ins+chord_diagram_max_r) + h - (h - 2*(y_ins+chord_diagram_max_r))*(wy-y_min)/(y_max-y_min)
            circles = []
            for _name_ in self.community_names:
                sx, sy = wxToSx(self.pos_communities[_name_][0]), wyToSy(self.pos_communities[_name_][1])
                _sz_   = len(self.community_lookup[_name_])
                scaled_r = chord_diagram_min_r + (chord_diagram_max_r-chord_diagram_min_r)*((_sz_ - self.community_size_min) / (self.community_size_max - self.community_size_min))
                if len(self.community_lookup[_name_]) == 1: circles.append((sx, sy, node_r))
                else:                                       circles.append((sx, sy, scaled_r))
            # Crunch the circles
            circles_adjusted = self.rt_self.crunchCircles(circles)
            # Re-adjust w/in screen coordinates
            x_min, y_min = circles_adjusted[0][0] - circles_adjusted[0][2], circles_adjusted[0][1] - circles_adjusted[0][2]
            x_max, y_max = circles_adjusted[0][0] + circles_adjusted[0][2], circles_adjusted[0][1] + circles_adjusted[0][2]
            for i in range(len(circles_adjusted)):
                x_min, y_min = min(x_min, circles_adjusted[i][0] - circles_adjusted[i][2]), min(y_min, circles_adjusted[i][1] - circles_adjusted[i][2])
                x_max, y_max = max(x_max, circles_adjusted[i][0] + circles_adjusted[i][2]), max(y_max, circles_adjusted[i][1] + circles_adjusted[i][2])
            wxToSx  = lambda wx:      x_ins + (w - 2*x_ins)*(wx-x_min)/(x_max-x_min)
            wyToSy  = lambda wy: h - (y_ins + (h - 2*y_ins)*(wy-y_min)/(y_max-y_min))
            _circles_again_ = []
            for i in range(len(circles_adjusted)):
                sx, sy = wxToSx(circles_adjusted[i][0]), wyToSy(circles_adjusted[i][1])
                _circles_again_.append((sx, sy, circles_adjusted[i][2]))
            circles_adjusted = _circles_again_
            # Check for overlaps
            no_overlaps = True
            for i in range(len(circles_adjusted)):
                for j in range(i+1,len(circles_adjusted)):
                    _l_ = self.rt_self.segmentLength((circles_adjusted[i], circles_adjusted[j]))
                    if _l_ < circles_adjusted[i][2]+circles_adjusted[j][2]+min_intra_circle_d: 
                        no_overlaps = False
                        break
                if no_overlaps is False: break
            if no_overlaps: 
                self.circles = []
                for _circle_ in circles_adjusted: self.circles.append((_circle_[0], _circle_[1], _circle_[2] - shrink_circles_by))
            else:
                chord_diagram_max_r  *= 0.9
                if chord_diagram_max_r < chord_diagram_min_r: chord_diagram_max_r = chord_diagram_min_r+5
                self.pos_communities  = nx.spring_layout(self.g_communities)

        if self.circles is None: raise Exception(f'RTBundledEgoChordDiagram(): could not find a layout that didn\'t overlap after {_attempts_} attempts')

        # Create the chord diagrams
        dfs_rendered, self.community_to_chord_diagram = [], {}
        for i in range(len(self.community_names)):
            _name_   = self.community_names[i]
            _circle_ = self.circles[i]
            _nodes_  = self.community_lookup[_name_]
            if len(_nodes_) > 1:
                _df_     = self.df_aligned.filter(pl.col('__fm__').is_in(_nodes_) & pl.col('__to__').is_in(_nodes_))
                cd_r     = _circle_[2]
                _cd_     = self.rt_self.chordDiagram(_df_, [('__fm__', '__to__')], w=cd_r*2, h=cd_r*2, x_ins=0, y_ins=0, link_opacity=chord_diagram_opacity, draw_border=False, node_h=chord_diagram_node_h)
                _cd_svg_ = _cd_._repr_svg_() # force the svg to be rendered
                self.community_to_chord_diagram[_name_] = _cd_
                dfs_rendered.append(_df_)

        # Dataframe of all the data rendered in the chord diagrams
        self.df_cd_rendered = pl.concat(dfs_rendered)
    


        # Calculate the voronoi cells
        self.voronoi_cells = self.rt_self.laguerreVoronoi(self.circles, Box=[(x_ins/2.0,y_ins/2.0),(x_ins/2.0,h-y_ins/2.0),(w-x_ins/2.0,h-y_ins/2.0),(w-x_ins/2.0,y_ins/2.0)])

        # State
        self.last_render   = None

    #
    #
    #
    def outlineSVG(self):
        bg_color = self.rt_self.co_mgr.getTVColor('background','default')
        svg = [f'<svg x="0" y="0" width="{self.w}" height="{self.h}">',
               f'<rect width="{self.w}" height="{self.h}" x="0" y="0" fill="{bg_color}" stroke="{bg_color}" />']
        # Render the circles
        for _circle_ in self.circles: svg.append(f'<circle cx="{_circle_[0]}" cy="{_circle_[1]}" r="{_circle_[2]}" fill="none" stroke="#b0b0b0" />')
        # Render the voronoi cells
        for _poly_ in self.voronoi_cells:
            d = [f'M {_poly_[0][0]} {_poly_[0][1]} ']
            for j in range(1,len(_poly_)): d.append(f'L {_poly_[j][0]} {_poly_[j][1]} ')
            d.append('Z')
            svg.append(f'<path d=\"{" ".join(d)}\" fill="none" stroke="#b0b0b0" stroke-width="0.5" />')

        svg.append('</svg>')
        return ''.join(svg)

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

        # Render the chord diagrams
        for i in range(len(self.community_names)):
            _name_       = self.community_names[i]
            sx, sy, cd_r = self.circles[i]
            if _name_ in self.community_to_chord_diagram.keys():
                _cd_ = self.community_to_chord_diagram[_name_]
                svg.append(f'<g transform="translate({sx-cd_r}, {sy-cd_r})">{_cd_._repr_svg_()}</g>')
            else:
                svg.append(f'<circle cx="{sx}" cy="{sy}" r="{3.0}" fill="#000000" stroke="none" />')

        svg.append('</svg>')
        self.last_render = ''.join(svg)
        return self.last_render