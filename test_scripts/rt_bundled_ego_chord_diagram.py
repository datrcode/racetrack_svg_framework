import polars as pl
import pandas as pd
import numpy as np
import networkx as nx
import random
import time
from math import sin, cos, atan2, pi, sqrt
from shapely.geometry import Polygon

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
                 chord_diagram_kwargs   = None,  # kwargs for the chord diagram
                 # Geometry
                 min_intra_circle_d     = 10,    # minimum distance between circles
                 chord_diagram_min_r    = 40,    # minimum radius of the chord diagrams
                 chord_diagram_max_r    = 100,   # maximum radius of the chord diagrams
                 chord_diagram_pushout  = 3,     # how much to push out the entry/exit points from the chord diagram
                 chord_diagram_node_h   = 3,     # height of the entry/exit point nodes
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
        self.min_intra_circle_d = min_intra_circle_d
        self.w                  = w
        self.h                  = h
        self.widget_id          = widget_id

        # Performance information
        self.time_lu            = {}

        # Make a widget_id if it's not set already
        if self.widget_id is None: self.widget_id = "bundled_ego_chord_diagram_" + str(random.randint(0,65535))

        # Copy the dataframe (columns are going to be added for the rendering)
        self.df = rt_self.copyDataFrame(df)

        #
        # Field Transformations
        #
        t0 = time.time()
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
        self.time_lu['field_transforms'] = time.time() - t0

        # Create concatenated fields for the tuple nodes
        t0 = time.time()
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
        self.time_lu['concat_fields'] = time.time() - t0

        # Align the dataset so that there's only __fm__ and __to__
        # ... if any to / froms are integers, they will be converted to strings (polars has strongly typed columns)
        t0 = time.time()
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
        self.time_lu['df_alignment'] = time.time() - t0

        # Create the graph representation
        t0 = time.time()
        self.g = self.rt_self.createNetworkXGraph(self.df_aligned, [('__fm__','__to__')], count_by=self.count_by, count_by_set=self.count_by_set)
        self.time_lu['graph_init_creation'] = time.time() - t0

        # Create the layout
        t0 = time.time()
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
        self.time_lu['community_detection'] = time.time() - t0

        # Collapse the communities
        t0 = time.time()
        self.df_communities  = rt_self.collapseDataFrameGraphByClusters(self.df_aligned, [('__fm__','__to__')], self.community_lookup)
        self.g_communities   = rt_self.createNetworkXGraph(self.df_communities, [('__fm__','__to__')])
        
        # Fill in the community positions and the regular positions ... 
        if self.pos is None:
            self.pos_communities = nx.spring_layout(self.g_communities)
            self.pos             = {}
            for _what_ in self.g_communities.nodes():
                if _what_ in self.g.nodes(): self.pos[_what_] = self.pos_communities[_what_]
                else:
                    for _node_ in self.community_lookup[_what_]:
                        self.pos[_node_] = self.pos_communities[_what_]
        else:
            self.pos_communities = {}
            for _node_ in self.g.nodes():
                if _node_ in self.g_communities.nodes(): # it's a regular node... just copy over the position information
                    self.pos_communities[_node_] = self.pos[_node_]
            for _community_name_ in self.community_lookup:
                if len(self.community_lookup[_community_name_]) > 1:
                    _node_ = list(self.community_lookup[_community_name_])[0]
                    self.pos_communities[_community_name_] = self.pos[_node_]
        self.time_lu['community_collapse'] = time.time() - t0

        # Create an ordered list of community names and their associated circles
        self.community_names = list(self.community_lookup.keys())

        # Figure out how to position the nodes on the screen
        t0 = time.time()
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
            circles_adjusted = self.rt_self.crunchCircles(circles, min_d=min_intra_circle_d)
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
        self.time_lu['community_layout'] = time.time() - t0

        if self.circles is None: raise Exception(f'RTBundledEgoChordDiagram(): could not find a layout that didn\'t overlap after {_attempts_} attempts')

        # Use a convex hull as a wrapper
        t0 = time.time()
        _pts_, _num_ = {}, 12
        for i in range(len(self.circles)):
            sx, sy, r = self.circles[i]
            for j in range(_num_):
                _angle_ = 2*pi*j/_num_
                _pts_[f'{i}_{j}'] = (sx+(r+2*min_intra_circle_d)*cos(_angle_), sy+(r+2*min_intra_circle_d)*sin(_angle_))
        _graham_scan_results_ = self.rt_self.grahamScan(_pts_)
        _graham_scan_ = []
        for k in _graham_scan_results_: _graham_scan_.append(_pts_[k])
        self.time_lu['graham_scan'] = time.time() - t0

        # Calculate the voronoi cells
        t0 = time.time()
        #self.voronoi_cells = self.rt_self.laguerreVoronoi(self.circles, 
        #                                                  Box=[(x_ins/2.0,y_ins/2.0),(x_ins/2.0,h-y_ins/2.0),(w-x_ins/2.0,h-y_ins/2.0),(w-x_ins/2.0,y_ins/2.0)])
        self.voronoi_cells = self.rt_self.laguerreVoronoi(self.circles, _graham_scan_)
        # ... double check the results of the voronoi output -- no xy's should be duplicated
        for i in range(len(self.voronoi_cells)):
            _poly_ = self.voronoi_cells[i]
            _xys_  = set()
            for j in range(len(_poly_)):
                _xy_ = _poly_[j]
                if _xy_ in _xys_: raise Exception(f'RTBundledEgoChordDiagram(): duplicate xy ({_xy_}) in poly {i}')
                _xys_.add(_xy_)
        self.time_lu['voronoi_cells'] = time.time() - t0

        # Create the chord diagrams
        t0 = time.time()
        if chord_diagram_kwargs is None: my_chord_diagram_kwargs = {}
        else:                            my_chord_diagram_kwargs = chord_diagram_kwargs
        dfs_rendered, self.community_to_chord_diagram = [], {}
        for i in range(len(self.community_names)):
            _name_   = self.community_names[i]
            _circle_ = self.circles[i]
            _nodes_  = self.community_lookup[_name_]
            if len(_nodes_) > 1:
                _df_     = self.df_aligned.filter(pl.col('__fm__').is_in(_nodes_) & pl.col('__to__').is_in(_nodes_))
                cd_r     = _circle_[2]
                _cd_     = self.rt_self.chordDiagram(_df_, [('__fm__', '__to__')], w=cd_r*2, h=cd_r*2, x_ins=0, y_ins=0, 
                                                     draw_border=False, node_h=chord_diagram_node_h, **my_chord_diagram_kwargs)
                _cd_svg_ = _cd_._repr_svg_() # force the svg to be rendered
                self.community_to_chord_diagram[_name_] = _cd_
                dfs_rendered.append(_df_)
        self.time_lu['chord_diagram_creation'] = time.time() - t0

        # Dataframe of all the data rendered in the chord diagrams
        t0 = time.time()
        self.df_cd_rendered = pl.concat(dfs_rendered)
        self.time_lu['df_cd_rendered'] = time.time() - t0

        # Determine the entry / exit points (largely copies from routing_3.ipynb)
        t0 = time.time()
        entity_to_ins, entity_to_outs, entity_to_cdarc = {}, {}, {}
        fm_arcs,       to_arcs                         = {}, {}
        arc_pos_and_vec                                = {}
        arc_str_to_circle_i                            = {}
        for _community_i_ in range(len(self.community_names)):
            _community_ = self.community_names[_community_i_]
            sx, sy, r   = self.circles[_community_i_]
            if len(self.community_lookup[_community_]) == 1:
                _community_fm_str_ = 'fm|'+_community_
                _community_to_str_ = 'to|'+_community_
                arc_str_to_circle_i[_community_fm_str_] = _community_i_
                arc_str_to_circle_i[_community_to_str_] = _community_i_
                fm_arcs[_community_fm_str_] = self.community_lookup[_community_]
                to_arcs[_community_to_str_] = self.community_lookup[_community_]

                # Find the closest edge & connect it to that
                _poly_, d_closest, xy_closest = self.voronoi_cells[_community_i_], None, None
                for i in range(len(_poly_)):
                    _xy0_, _xy1_ = _poly_[i], _poly_[(i+1)%len(_poly_)]
                    _d_, _xy_    = self.rt_self.closestPointOnSegment((_xy0_, _xy1_), (sx, sy))
                    if d_closest is None or _d_ < d_closest: d_closest, xy_closest = _d_, _xy_
                
                if xy_closest is None: # Used fixed positions / doesn't yield good results
                    _angle_ = pi / 2.0
                    _x_in_, _y_in_ = sx + (r + chord_diagram_pushout/2.0) * cos(_angle_), sy + (r + chord_diagram_pushout/2.0) * sin(_angle_)
                    _u_in_, _v_in_ = cos(_angle_), sin(_angle_)
                    _angle_ = 3 * pi / 2.0
                    _x_out_, _y_out_ = sx + (r + chord_diagram_pushout) * cos(_angle_), sy + (r + chord_diagram_pushout) * sin(_angle_)
                    _u_out_, _v_out_ = cos(_angle_), sin(_angle_)
                else:
                    _uv_    = self.rt_self.unitVector(((sx, sy), xy_closest))
                    _angle_, _diff_ = atan2(_uv_[1], _uv_[0]), pi/8.0
                    _x_in_,  _y_in_  = sx + (r + chord_diagram_pushout) * cos(_angle_+_diff_), sy + (r + chord_diagram_pushout) * sin(_angle_+_diff_)
                    _u_in_,  _v_in_  = cos(_angle_+_diff_), sin(_angle_+_diff_)
                    _x_out_, _y_out_ = sx + (r + chord_diagram_pushout) * cos(_angle_-_diff_), sy + (r + chord_diagram_pushout) * sin(_angle_-_diff_)
                    _u_out_, _v_out_ = cos(_angle_-_diff_), sin(_angle_-_diff_)

                arc_pos_and_vec[_community_fm_str_] = ((_x_in_,  _y_in_),  (_u_in_,  _v_in_),  _community_i_)
                arc_pos_and_vec[_community_to_str_] = ((_x_out_, _y_out_), (_u_out_, _v_out_), _community_i_)
            else:
                _cd_        = self.community_to_chord_diagram[_community_]
                _cd_r_      = _cd_.r
                _angle_inc_ = 360.0 / chord_diagram_points
                for i in range(chord_diagram_points):
                    _angle_min_            = _angle_inc_ * i
                    _angle_max_            = _angle_inc_ * (i+1)
                    _community_arc_str_    = _community_ + f'|{i}'
                    _community_arc_fm_str_ = 'fm|'+_community_arc_str_
                    _community_arc_to_str_ = 'to|'+_community_arc_str_
                    arc_str_to_circle_i[_community_arc_fm_str_] = _community_i_
                    arc_str_to_circle_i[_community_arc_to_str_] = _community_i_
                    fm_arcs[_community_arc_fm_str_] = set()
                    to_arcs[_community_arc_to_str_] = set()

                    _angle_in_             = -_angle_inc_ / 8.0 + (_angle_min_ + _angle_max_) / 2.0
                    _angle_out_            =  _angle_inc_ / 8.0 + (_angle_min_ + _angle_max_) / 2.0
                    if i == 0 and chord_diagram_points == 1: _angle_in_, _angle_out_ = 90.0, 270.0
                    _x_in_                 = sx + (_cd_r_ + chord_diagram_pushout) * cos(_angle_in_ * pi / 180.0)
                    _y_in_                 = sy + (_cd_r_ + chord_diagram_pushout) * sin(_angle_in_ * pi / 180.0)
                    arc_pos_and_vec[_community_arc_fm_str_] = ((_x_in_,  _y_in_),  (cos(_angle_in_  * pi / 180.0), sin(_angle_in_  * pi / 180.0)), _community_i_) # (xy, uv, community_i)

                    _x_out_                = sx + (_cd_r_ + chord_diagram_pushout) * cos(_angle_out_ * pi / 180.0)
                    _y_out_                = sy + (_cd_r_ + chord_diagram_pushout) * sin(_angle_out_ * pi / 180.0)
                    arc_pos_and_vec[_community_arc_to_str_] = ((_x_out_, _y_out_), (cos(_angle_out_ * pi / 180.0), sin(_angle_out_ * pi / 180.0)), _community_i_) # (xy, uv, community_i)

                    _entities_             = _cd_.entitiesOnArc(_angle_min_, _angle_max_)
                    for _entity_ in _entities_:
                        entity_to_ins   [_entity_] = ((_x_in_,  _y_in_),  (cos(_angle_in_  * pi / 180.0), sin(_angle_in_  * pi / 180.0))) # (xy, uv)
                        entity_to_outs  [_entity_] = ((_x_out_, _y_out_), (cos(_angle_out_ * pi / 180.0), sin(_angle_out_ * pi / 180.0))) # (xy, uv)
                        entity_to_cdarc [_entity_] = _community_arc_str_
                        fm_arcs[_community_arc_fm_str_].add(_entity_)
                        to_arcs[_community_arc_to_str_].add(_entity_)
        self.entity_to_ins, self.entity_to_outs, self.entity_to_cdarc, self.arc_pos_and_vec, self.arc_str_to_circle_i = entity_to_ins, entity_to_outs, entity_to_cdarc, arc_pos_and_vec, arc_str_to_circle_i
        self.fm_arcs, self.to_arcs = fm_arcs, to_arcs
        self.time_lu['enter_exit_points'] = time.time() - t0

        # Create the inter dataframe
        t0 = time.time()
        self.df_aligned_btwn_communities           = self.df_aligned.join(self.df_cd_rendered, on=['__fm__', '__to__'], how='anti')
        self.df_aligned_btwn_communities_collapsed = self.rt_self.collapseDataFrameGraphByClustersDirectional(self.df_aligned_btwn_communities, [('__fm__','__to__')], self.fm_arcs, self.to_arcs, color_by=self.color_by)
        self.time_lu['inter_dataframe'] = time.time() - t0

        # Create the voronoi graph routing network by connecting the entry / exit points to the voronoi diagram (copying from routing_4.ipynb)
        # (Uniquify) Segment To Poly
        def uniquifySegment(s):
            _xy0_, _xy1_ = s[0], s[1]
            if   _xy0_[0] <  _xy1_[0]: return s
            elif _xy0_[0] >  _xy1_[0]: return (_xy1_, _xy0_)
            elif _xy0_[1] <  _xy1_[1]: return s
            else:                      return (_xy1_, _xy0_)

        # Unique segments to the circle indices
        segment_to_circle_is = {}
        for i in range(len(self.voronoi_cells)):
            _poly_ = self.voronoi_cells[i]
            for j in range(len(_poly_)):
                _segment_    = (_poly_[j], _poly_[(j+1)%len(_poly_)])
                # Do a sanity check
                seg0_count, seg1_count = 0, 0
                for k in range(len(_poly_)):
                    if _poly_[k] == _segment_[0]: seg0_count += 1
                    if _poly_[k] == _segment_[1]: seg1_count += 1
                if seg0_count != 1 or seg1_count != 1: raise Exception(f'Voronoi {i} , Segment Indices {j} | seg0_count == {seg0_count} | seg1_count == {seg1_count} | (both should be 1)')
                _uniquified_ = uniquifySegment(_segment_)
                if _uniquified_ not in segment_to_circle_is: segment_to_circle_is[_uniquified_] = []
                segment_to_circle_is[_uniquified_].append(i)

        # Add a vertex on a segment and update the associated polygons
        def addVertexAndUpdatePolygons(seg, xy, pixel_diff=4.0):
            seg    = uniquifySegment(seg)
            # Try to use an existing vertex
            l      = self.rt_self.segmentLength(seg)
            l0, l1 = self.rt_self.segmentLength((seg[0], xy)), self.rt_self.segmentLength((seg[1], xy))
            if l0 < pixel_diff and l0 < l1: return seg[0]
            if l1 < pixel_diff:             return seg[1]
            # Determine the polys to update & then put the new vertex in the right segment for each poly
            polys_to_update = segment_to_circle_is[seg]
            for i in polys_to_update:
                _poly_ = self.voronoi_cells[i]
                seg0_count, seg1_count = 0, 0
                for j in range(len(_poly_)):
                    if _poly_[j] == seg[0]: seg0_count += 1
                    if _poly_[j] == seg[1]: seg1_count += 1
                if seg0_count != 1 or seg1_count != 1: raise Exception(f'seg0_count == {seg0_count} | seg1_count == {seg1_count} | (both should be 1)')

                j      = 0
                while _poly_[j] != seg[0] and _poly_[j] != seg[1]: j += 1
                k      = (j+1)%len(_poly_)
                if    _poly_[k] != seg[0] and _poly_[k] != seg[1]: k = len(_poly_)-1
                if    _poly_[k] != seg[0] and _poly_[k] != seg[1]: raise Exception('can\'t find k')
                if j == 0 and k == len(_poly_)-1: _poly_.append(xy)
                else:                             _poly_.insert(k,xy)
            del segment_to_circle_is[seg]                                            # remove old segment -- it's now two segments
            seg0, seg1 = uniquifySegment((xy,seg[0])), uniquifySegment((xy,seg[1]))  # uniquify the two new segments
            if seg0 not in segment_to_circle_is: segment_to_circle_is[seg0] = []     # update the lookup tables w/ the new segments
            if seg1 not in segment_to_circle_is: segment_to_circle_is[seg1] = []
            segment_to_circle_is[seg0].extend(polys_to_update)
            segment_to_circle_is[seg1].extend(polys_to_update)
            return xy

        # For every used entry / exit point, determine how to connect that point to the voronoi diagram
        t0 = time.time()
        self.entry_exit_to_xy, self.pos_routing = {}, {}
        _large_distance_      = 1e9
        entry_exits_used = set(self.df_aligned_btwn_communities_collapsed['__fm__']) | set(self.df_aligned_btwn_communities_collapsed['__to__'])
        for entry_exit_pt in self.arc_pos_and_vec:
            if entry_exit_pt not in entry_exits_used: continue
            _xy_, _uv_, _community_i_ = self.arc_pos_and_vec[entry_exit_pt]
            self.pos_routing[entry_exit_pt] = _xy_
            entry_exit_ray = (_xy_, (_xy_[0]+_large_distance_*_uv_[0], _xy_[1]+_large_distance_*_uv_[1]))
            _poly_ = self.voronoi_cells[_community_i_]
            _xy_inter_, _segment_inter_ = None, None
            for i in range(len(_poly_)):
                _segment_            = (_poly_[i], _poly_[(i+1)%len(_poly_)])
                _intersection_tuple_ = self.rt_self.segmentsIntersect(entry_exit_ray, _segment_)
                if _intersection_tuple_[0]:
                    _xy_inter_, _segment_inter_ = (_intersection_tuple_[1], _intersection_tuple_[2]), _segment_
                    break
            if _xy_inter_ is None: raise Exception('xy_inter should not be none...')
            _xy_inter_ = addVertexAndUpdatePolygons(_segment_inter_, _xy_inter_)
            self.entry_exit_to_xy[entry_exit_pt] = _xy_inter_
        self.time_lu['entry_exit_to_xy'] = time.time() - t0

        # Construct the voronoi routing graph
        t0 = time.time()
        _lu_           = {'__fm__':[],'__to__':[],'__dist__':[]}
        _xy_to_name_   = {}
        for _segment_ in segment_to_circle_is: # already uniquified
            if _segment_[0] not in _xy_to_name_: 
                _xy_to_name_[_segment_[0]] = f'V_{len(_xy_to_name_)}'
                self.pos_routing[_xy_to_name_[_segment_[0]]] = _segment_[0]
            if _segment_[1] not in _xy_to_name_: 
                _xy_to_name_[_segment_[1]] = f'V_{len(_xy_to_name_)}'
                self.pos_routing[_xy_to_name_[_segment_[1]]] = _segment_[1]
            _lu_['__fm__']  .append(_xy_to_name_[_segment_[0]])
            _lu_['__to__']  .append(_xy_to_name_[_segment_[1]])
            _lu_['__dist__'].append(self.rt_self.segmentLength(_segment_))
        for entry_exit_pt in self.entry_exit_to_xy:
            _xy_ = self.entry_exit_to_xy[entry_exit_pt]
            if _xy_ not in _xy_to_name_: raise Exception(f'_xy_ not in _xy_to_name_: {_xy_}')
            _lu_['__fm__']  .append(_xy_to_name_[_xy_])
            _lu_['__to__']  .append(entry_exit_pt)
            _lu_['__dist__'].append(self.rt_self.segmentLength((_xy_, self.pos_routing[entry_exit_pt])))

        self.df_routing = pl.DataFrame(_lu_)
        self.g_routing  = self.rt_self.createNetworkXGraph(self.df_routing, [('__fm__','__to__')], count_by='__dist__')
        self.time_lu['routing_graph_construction'] = time.time() - t0

        # Determine the routes
        t0 = time.time()
        self.segment_contains_tree = {}
        self.segment_max_count, self.segment_max_fm = 0, None
        _tos_ = set(self.df_aligned_btwn_communities_collapsed['__to__'])
        for _to_ in _tos_:
            len_lu, path_lu = nx.single_source_bellman_ford(self.g_routing, source=_to_, weight='weight')
            _df_ = self.df_aligned_btwn_communities_collapsed.filter(pl.col('__to__') == _to_)
            for _fm_tuple_, _fm_df_ in _df_.group_by('__fm__'):
                if len(_fm_df_) > 1: raise Exception(f'len(_fm_df_) > 1: {_fm_df_}')
                _fm_   = _fm_tuple_[0]
                _ct_   = _fm_df_['__count__'][0]
                _path_ = path_lu[_fm_]
                for i in range(len(_path_)-1):
                    _v0_, _v1_ = _path_[i], _path_[i+1]
                    _tuple_ = (_v0_, _v1_) # if _v0_ < _v1_ else (_v1_, _v0_) # keep directionality (why the rest is commented out)
                    if _tuple_ not in self.segment_contains_tree:          self.segment_contains_tree[_tuple_]       = {}
                    if _to_    not in self.segment_contains_tree[_tuple_]: self.segment_contains_tree[_tuple_][_fm_] = 0
                    self.segment_contains_tree[_tuple_][_fm_] += _ct_
                    if self.segment_contains_tree[_tuple_][_fm_] > self.segment_max_count:
                        self.segment_max_count = self.segment_contains_tree[_tuple_][_fm_]
                        self.segment_max_fm    = _fm_
        self.time_lu['routing_calculations'] = time.time() - t0

        # State
        self.last_render   = None

    #
    #
    #
    def outlineSVG(self, render_inter_edges=False, view_border=64):
        bg_color = self.rt_self.co_mgr.getTVColor('background','default')
        svg = [f'<svg x="0" y="0" width="{self.w}" height="{self.h}" viewBox="{-view_border} {-view_border} {self.w+2*view_border} {self.h+2*view_border}">',
               f'<rect width="{self.w+2*view_border}" height="{self.h+2*view_border}" x="{-view_border}" y="{-view_border}" fill="#f0f0f0" stroke="#f0f0f0" />'
               f'<rect width="{self.w}"               height="{self.h}"               x="0"              y="0"              fill="{bg_color}" stroke="{bg_color}" />']
        # Render the circles
        for _circle_ in self.circles: svg.append(f'<circle cx="{_circle_[0]}" cy="{_circle_[1]}" r="{_circle_[2]}" fill="none" stroke="#b0b0b0" />')
        # Render the entry / exit points
        for _arc_str_ in self.arc_pos_and_vec:
            _tuple_                   = self.arc_pos_and_vec[_arc_str_]
            _xy_, _uv_, _community_i_ = _tuple_
            if   _arc_str_.startswith('fm|'): _color_ = '#013220'
            elif _arc_str_.startswith('to|'): _color_ = '#ff0000'
            svg.append(f'<circle cx="{_xy_[0]}" cy="{_xy_[1]}" r="2" fill="none" stroke="{_color_}" />')
        # Render the voronoi cells
        _xys_ = set()
        for _poly_ in self.voronoi_cells:
            _xys_.add(_poly_[0])
            d = [f'M {_poly_[0][0]} {_poly_[0][1]} ']
            for j in range(1,len(_poly_)):
                _xys_.add(_poly_[j])
                d.append(f'L {_poly_[j][0]} {_poly_[j][1]} ')
            d.append('Z')
            svg.append(f'<path d=\"{" ".join(d)}\" fill="none" stroke="#b0b0b0" stroke-width="0.5" />')
        for _entry_exit_ in self.entry_exit_to_xy:
            _xy_poly_ = self.entry_exit_to_xy[_entry_exit_]
            _xy_, _uv_, circle_i = self.arc_pos_and_vec[_entry_exit_]
            _xys_.add(_xy_poly_), _xys_.add(_xy_)
            svg.append(f'<line x1="{_xy_poly_[0]}" y1="{_xy_poly_[1]}" x2="{_xy_[0]}" y2="{_xy_[1]}" stroke="#b0b0b0" stroke-width="0.5" />')
        for _xy_ in _xys_: svg.append(f'<circle cx="{_xy_[0]}" cy="{_xy_[1]}" r="{1.0+2.0*random.random()}" fill="none" stroke="#000000" stroke-width="0.1"/>')

        # Inter Edges (In Between Communities)
        if render_inter_edges:
            for i in range(len(self.df_aligned_btwn_communities_collapsed)):
                _fm_,    _to_    = self.df_aligned_btwn_communities_collapsed['__fm__'][i], self.df_aligned_btwn_communities_collapsed['__to__'][i]
                _fm_xy_, _to_xy_ = self.arc_pos_and_vec[_fm_][0], self.arc_pos_and_vec[_to_][0]
                svg.append(f'<line x1="{_fm_xy_[0]}" y1="{_fm_xy_[1]}" x2="{_to_xy_[0]}" y2="{_to_xy_[1]}" stroke="#b0b0b0" stroke-width="0.5" />')

        # Graham Scan
        _pos_ = {}
        for _arc_str_ in self.arc_pos_and_vec:
            _xy_, _uv_, _community_i_ = self.arc_pos_and_vec[_arc_str_]
            _pos_[_arc_str_] = _xy_
        _nodes_ = self.rt_self.grahamScan(_pos_)
        d = self.rt_self.extrudePolyLine(_nodes_, _pos_, r=2*self.min_intra_circle_d, use_curved_caps=False)
        svg.append(f'<path d="{d}" fill="none" stroke="#0000ff" stroke-width="0.1" />')

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

        # Render the routing network
        for _named_vertices_ in self.segment_contains_tree:
            _v0_,  _v1_   = _named_vertices_
            _xy0_, _xy1_  = self.pos_routing[_v0_], self.pos_routing[_v1_]
            _uv_          = self.rt_self.unitVector((_xy0_, _xy1_))
            _perp_        = (-_uv_[1], _uv_[0])
            _d_,   _side_ = 0.0, -1
            for _arc_str_ in self.segment_contains_tree[_named_vertices_]:
                _color_ = self.rt_self.co_mgr.getColor(_arc_str_)
                svg.append(f'<line x1="{_xy0_[0]+_d_*_side_*_perp_[0]}" y1="{_xy0_[1]+_d_*_side_*_perp_[1]}" x2="{_xy1_[0]+_d_*_side_*_perp_[0]}" y2="{_xy1_[1]+_d_*_side_*_perp_[1]}" stroke="{_color_}" stroke-width="1.5" />')
                if   _d_    == 0.0: _d_         =     _d_ + 1.5
                elif _side_ == -1:  _side_      =  1
                else:               _side_, _d_ = -1, _d_ + 1.5

        svg.append('</svg>')
        self.last_render = ''.join(svg)
        return self.last_render