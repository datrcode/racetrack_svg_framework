# Copyright 2023 David Trimm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pandas as pd
import polars as pl
import numpy as np
import networkx as nx
import random
import math
import time
import re

from math import sqrt

from shapely.geometry import Point,Polygon,LineString

from .rt_component import RTComponent
from .rt_entity_position import RTEntityPosition

__name__ = 'rt_link_mixin'

#
# Abstraction for Link ... diferent implementation of linknode that is param-capatible with linknode
#
class RTLinkMixin(object):
    #
    # linkPreferredDimensions()
    # - Return the preferred size
    #
    def linkPreferredDimensions(self, **kwargs):
        return (256,256)

    #
    # linkMinimumDimensions()
    # - Return the minimum size
    #
    def linkMinimumDimensions(self, **kwargs):
        return (32,32)

    #
    # linkSmallMultipleDimensions()
    # - Return the minimum size
    #
    def linkSmallMultipleDimensions(self, **kwargs):
        return (32,32)

    #
    # Identify the required fields in the dataframe from link parameters
    #
    def linkRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('relationships', kwargs, columns_set)
        self.identifyColumnsFromParameters('color_by',      kwargs, columns_set)
        self.identifyColumnsFromParameters('count_by',      kwargs, columns_set)
        if 'timing_marks' in kwargs.keys() and kwargs['timing_marks'] == True:
            self.identifyColumnsFromParameters('ts_field',  kwargs, columns_set)
            
        # Ignoring the small multiples version // for now
        return columns_set

    #
    # link
    #    
    def link(self,
             df,                           # dataframe(s) to render ... unlike other parts, this can be more than one...
             relationships,                # list of tuple pairs... pairs can be single strings or tuples of strings
                                           # [('f0','f1')] // 1 relationship: f0 to f1
                                           # [('f0','f1'),('f1','f2')] // 2 relationships: f0 to f1 and f1 to f2
                                           # [(('f0','f1'),('f2','f3'))] // 1 relationship: 'f0'|'f1' to 'f2'|'f3'
                                           # ... can now also add the relationship field:  ('subject','predicate','object')

             # -----------------------     # everything else is a default...

             pos                 = {},     # networkx style position dictionary pos['node_name'] = 2d array of positions e.g., [[0...1],[0...1]]
             view_window         = None,   # (wx0, wy0, wx1, wy1) // if none, will be derived from pos parameter

             use_pos_for_bounds  = True,   # use the pos values for the boundary of the view
             render_pos_context  = False,  # Render all the pos keys by default...  to provide context for the other nodes
             pos_context_opacity = 0.8,    # opacity of the pos context nodes

             bounds_percent      = .05,    # inset the graph into the view by this percent... so that the nodes aren't right at the edges 

             color_by            = None,   # just the default color or a string for a field
             count_by            = None,   # none means just count rows, otherwise, use a field to sum by
             count_by_set        = False,  # count by summation (by default)... count_by column is checked

             widget_id           = None,   # naming the svg elements                 

             # -----------------------     # linknode visualization
            
             node_color        = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
                                           # ... or a dictionary of the node string to either a string to color hash or a "#xxxxxx"
             node_border_color = None,     # small edge around nodes ... should only be "#xxxxxx"
             node_size         = 'medium', # 'small', 'medium', 'large', 'vary', 'hidden' / None
             node_shape        = None,     # 'square', 'ellipse' / None, 'triangle', 'utriangle', 'diamond', 'plus', 'x', 'small_multiple',
                                           # ... or a dictionary of the field tuples node to a shape name
                                           # ... or a dictionary of the field tuples node to an SVG small multiple
                                           # ... or a function pointer to a shape function
             node_opacity      = 1.0,      # fixed node opacity                 
             node_labels       = None,     # Dictionary of node string to array of strings for additional labeling options
             node_labels_only  = False,    # Only label based on the node_labels dictionary
             label_only        = set(),    # label only set - only label these nodes
             node_label_max_w  = 64,       # max label width for a node in pixels -- None means no limit

             node_size_max     = 4,        # for node vary...
             node_size_min     = 0.3,      # for node vary...

             selected_entities = None,     # list of selected node names

             link_color        = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
             link_size         = 'small',  # 'nil', 'small', 'medium', 'large', 'vary', 'hidden' / None
             link_opacity      = '1.0',    # link opacity
             link_shape        = 'line',   # 'curve','line', or 'arrow'
             link_arrow        = True,     # draw an arrow at the end of the curve...
             link_arrow_style  = 'kite_v3',# 'kite', 'kite_v2', 'kite_v3'
             link_arrow_length = 10,       # length in pixels of the link arrow
             link_dash         = None,     # svg stroke-dash string, callable, or dictionary of relationship tuple to dash string array 

             link_max_curvature_px = 100,  # maximum link curvature outward
             link_parallel_perc    = 0.2,  # percent for control point parallel to the link
             link_ortho_perc       = 0.2,  # percent for control point orthogonal to the link

             link_size_max     = 4,        # for link vary...
             link_size_min     = 0.25,     # for link vary...

             # -----------------------     # label information

             link_labels       = False,    # label links (by color_by... if link_color is 'vary'... and label only is empty or contains the label)

             # -----------------------     # timing information

             timing_marks       = False,   # flag to enable timing marks on links
             ts_field           = None,    # timestamp field
             timing_mark_length = 5,       # corresponds to the length of the timing mark

             # -----------------------     # convex hull annotations

             convex_hull_lu           = None,  # dictionary... regex for node name to convex hull name
             convex_hull_opacity      = 0.3,   # opacity of the convex hulls
             convex_hull_labels       = False, # draw a label for the convex hull in the center of the convex hull
             convex_hull_stroke_width = None,  # Stroke width for the convex hull -- if None, will not be drawn...

             # -----------------------     # background polygons // copied mostly from the xy implementation

             bg_shape_lu              = None,       # lookup for background shapes -- key will be used for varying colors (if bg_shape_label_color == 'vary')
                                                # ['key'] = [(x0,y0),(x1,y1),...] OR
                                                # ['key'] = svg path description
             bg_shape_label_color     = None,       # None = no label, 'vary', lookup to hash color, or a hash color
             bg_shape_opacity         = 1.0,        # None (== 0.0), number, lookup to opacity
             bg_shape_fill            = None,       # None, 'vary', lookup to hash color, or a hash color
             bg_shape_stroke_w        = 1.0,        # None, number, lookup to width
             bg_shape_stroke          = 'default',  # None, 'default', lookup to hash color, or a hash color

             # -----------------------     # small multiple options

             sm_type               = None,   # should be the method name // similar to the smallMultiples method
             sm_w                  = None,   # override the width of the small multiple
             sm_h                  = None,   # override the height of the small multiple
             sm_params             = {},     # dictionary of parameters for the small multiples
             sm_x_axis_independent = True,   # Use independent axis for x (xy, temporal, and linkNode)
             sm_y_axis_independent = True,   # Use independent axis for y (xy, temporal, periodic, pie)
             sm_mode               = 'node', # 'node' or 'link'
             sm_t                  = 0.5,    # location of the small multiple on the link // only applies to sm_mode == 'link'

             # ----------------------------- # visualization geometry / etc.

             track_state           = False,  # track state for interactive filtering
             x_view                = 0,      # x offset for the view
             y_view                = 0,      # y offset for the view
             w                     = 256,    # width of the view
             h                     = 256,    # height of the view
             x_ins                 = 3,
             y_ins                 = 3,
             txt_h                 = 12,     # text height for labeling
             draw_labels           = False,  # draw labels flag # not implemented yet
             draw_performance      = True,   # draw performance information
             draw_border           = True):  # draw a border around the graph
        """
        See documentation for linkNode().

        Notes:
        - link is a syntactic equivalent call as linkNode()
        - However, link does not fully implement all of the functionality of linkNode()
        - Notable differences include:
        -- link() only works with Polars dataframes -- for example, mixed types are not supported due to Polars optimizations
        -- link() requires both relatonship fields to be the same type.
        """
        _params_ = locals().copy()
        _params_.pop('self')
        if self.isPandas(df): return self.RTLinkNode(self, **_params_)
        else:                 return self.RTLink    (self, **_params_)

    #
    # RTLink Class
    #
    class RTLink(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     **kwargs):
            self.parms                      = locals().copy()
            self.rt_self                    = rt_self
            self.relationships_orig         = kwargs['relationships']
            self.pos                        = kwargs['pos']
            self.view_window                = kwargs['view_window']
            self.view_window_orig           = kwargs['view_window'] # Orig will be used for user requests to reset the view
            self.use_pos_for_bounds         = kwargs['use_pos_for_bounds']
            self.render_pos_context         = kwargs['render_pos_context']
            self.pos_context_opacity        = kwargs['pos_context_opacity']
            self.bounds_percent             = kwargs['bounds_percent']
            self.color_by                   = kwargs['color_by']
            self.count_by                   = kwargs['count_by']
            self.count_by_set               = kwargs['count_by_set']
            self.widget_id                  = kwargs['widget_id']
            if self.widget_id is None:      self.widget_id = "link_" + str(random.randint(0,2**24))
            self.node_color                 = kwargs['node_color']
            self.node_border_color          = kwargs['node_border_color']
            self.node_size                  = kwargs['node_size']
            self.node_shape                 = kwargs['node_shape']
            self.node_opacity               = kwargs['node_opacity']
            self.node_labels                = kwargs['node_labels']
            self.node_labels_only           = kwargs['node_labels_only']
            self.node_label_max_w           = kwargs['node_label_max_w']
            self.label_only                 = kwargs['label_only']             # tied with labelOnly() method
            self.node_size_max              = kwargs['node_size_max']
            self.node_size_min              = kwargs['node_size_min']
            self.selected_entities          = kwargs['selected_entities']
            if self.selected_entities is None: self.selected_entities = set()
            self.link_color                 = kwargs['link_color']
            self.link_size                  = kwargs['link_size']
            self.link_opacity               = kwargs['link_opacity']
            self.link_shape                 = kwargs['link_shape']
            self.link_arrow                 = kwargs['link_arrow']
            self.link_arrow_style           = kwargs['link_arrow_style']
            self.link_arrow_length          = kwargs['link_arrow_length']
            self.link_dash                  = kwargs['link_dash']
            self.link_max_curvature_px      = kwargs['link_max_curvature_px']
            self.link_parallel_perc         = kwargs['link_parallel_perc']
            self.link_ortho_perc            = kwargs['link_ortho_perc']
            self.link_size_max              = kwargs['link_size_max']
            self.link_size_min              = kwargs['link_size_min']
            self.link_labels                = kwargs['link_labels']
            self.timing_marks               = kwargs['timing_marks']
            self.ts_field                   = kwargs['ts_field']
            self.timing_mark_length         = kwargs['timing_mark_length']
            self.convex_hull_lu             = kwargs['convex_hull_lu']
            self.convex_hull_opacity        = kwargs['convex_hull_opacity']
            self.convex_hull_labels         = kwargs['convex_hull_labels']
            self.convex_hull_stroke_width   = kwargs['convex_hull_stroke_width']

            self.bg_shape_lu                = kwargs['bg_shape_lu']           # Copied from xy implementation --vvv
            self.bg_shape_label_color       = kwargs['bg_shape_label_color']
            self.bg_shape_opacity           = kwargs['bg_shape_opacity']
            self.bg_shape_fill              = kwargs['bg_shape_fill']
            self.bg_shape_stroke_w          = kwargs['bg_shape_stroke_w']
            self.bg_shape_stroke            = kwargs['bg_shape_stroke']       # Copied from xy implementation --^^^

            self.sm_type                    = kwargs['sm_type']
            self.sm_w                       = kwargs['sm_w']
            self.sm_h                       = kwargs['sm_h']
            self.sm_params                  = kwargs['sm_params']
            self.sm_x_axis_independent      = kwargs['sm_x_axis_independent']
            self.sm_y_axis_independent      = kwargs['sm_y_axis_independent']
            self.sm_mode                    = kwargs['sm_mode']
            self.sm_t                       = kwargs['sm_t']
            self.track_state                = kwargs['track_state']
            self.x_view                     = kwargs['x_view']
            self.y_view                     = kwargs['y_view']
            self.w                          = kwargs['w']
            self.h                          = kwargs['h']
            self.x_ins                      = kwargs['x_ins']
            self.y_ins                      = kwargs['y_ins']
            self.txt_h                      = kwargs['txt_h']
            self.draw_performance           = kwargs['draw_performance']
            self.draw_labels                = kwargs['draw_labels']  # tied with drawLabels() method
            self.draw_border                = kwargs['draw_border']

            # Copy dataframe            
            self.df = rt_self.copyDataFrame(kwargs['df'])

            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)

            # Apply node field transforms
            for _edge in self.relationships_orig:
                for _node in _edge:
                    if type(_node) == str:
                        if rt_self.isTField(_node) and rt_self.tFieldApplicableField(_node) in self.df.columns:
                            self.df,_throwaway = rt_self.applyTransform(self.df, _node)
                    else:
                        for _tup_part in _node:
                            if rt_self.isTField(_tup_part) and rt_self.tFieldApplicableField(_tup_part) in self.df.columns:
                                self.df,_throwaway = rt_self.applyTransform(self.df, _tup_part)

            # vvv
            # vvv -- REMOVABLE (UNTIL WE MODIFIED THE REST OF THE CODE BASE)
            # vvv

            # Create concatenated fields for the tuple nodes
            # ... may be inefficient if there are multiples of the same tuple in different edges...
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

                    if   len(_edge) == 2: self.relationships.append((new_fm, new_to))
                    elif len(_edge) == 3: self.relationships.append((new_fm, new_to, _edge[2]))
                    else:                 raise Exception(f'link(): relationship tuples should have two or three parts "{_edge}"')
                else:
                    if   len(_edge) == 2: self.relationships.append((_fm_, _to_))
                    elif len(_edge) == 3: self.relationships.append((_fm_, _to_, _edge[2]))
                    else:                 raise Exception(f'link(): relationship tuples should have two or three parts "{_edge}"')
                i += 1

            # ^^^
            # ^^^ -- REMOVABLE (UNTIL WE MODIFY THE REST OF THE CODE BASE)
            # ^^^

            # Create the columns for colors
            self.df = self.df.with_columns(pl.lit(self.rt_self.co_mgr.getTVColor('data','default')).alias('__color_default__'))

            # Handle varying colors
            if self.color_by is not None and self.color_by in self.df.columns and (self.node_color == 'vary' or self.link_color == 'vary'):
                if   self.node_color == 'vary' and self.link_color == 'vary':
                    self.df = self.df.with_columns(pl.col(self.color_by).map_elements(lambda x: self.rt_self.co_mgr.getColor(x), return_dtype=pl.String).alias('__color_nodes__'))
                    self.df = self.df.with_columns(pl.col('__color_nodes__').alias('__color_links__')) # make a copy...
                elif self.node_color == 'vary':
                    self.df = self.df.with_columns(pl.col(self.color_by).map_elements(lambda x: self.rt_self.co_mgr.getColor(x), return_dtype=pl.String).alias('__color_nodes__'))
                elif self.link_color == 'vary':
                    self.df = self.df.with_columns(pl.col(self.color_by).map_elements(lambda x: self.rt_self.co_mgr.getColor(x), return_dtype=pl.String).alias('__color_links__'))

            # Handle fixed color for nodes
            if self.node_color is not None and type(self.node_color) is str and self.node_color.startswith('#') and len(self.node_color) == 7:
                self.df = self.df.with_columns(pl.lit(self.node_color).alias('__color_nodes__'))
            
            # Make sure there's something for node colors
            if '__color_nodes__' not in self.df.columns:
                _co_ = self.rt_self.co_mgr.getTVColor('background','default')
                self.df = self.df.with_columns(pl.lit(_co_).alias('__color_nodes__'))

            # Handle fixed color for links
            if self.link_color is not None and type(self.link_color) is str and self.link_color.startswith('#') and len(self.link_color) == 7:
                self.df = self.df.with_columns(pl.lit(self.link_color).alias('__color_links__'))
            
            # Make sures there's something for link colors
            if '__color_links__' not in self.df.columns:
                _co_ = self.rt_self.co_mgr.getTVColor('data','default')
                self.df = self.df.with_columns(pl.lit(_co_).alias('__color_links__'))

            # Check the node information... make sure the parameters are set
            if self.sm_type is not None and self.sm_mode == 'node':                   self.node_shape = 'small_multiple'
            if self.sm_type is not None and (self.sm_w is None or self.sm_h is None): self.sm_w,self.sm_h = getattr(rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
            if callable(self.node_shape) and self.node_size is None:                  self.node_size = 'medium'

            # Check the count_by column across all the df's...  if any of them
            # don't work.. then it's count_by_set
            if self.count_by_set == False: self.count_by_set = rt_self.countBySet(self.df, self.count_by)

            # Configuration lookups
            self.node_size_lu = {'small':3, 'medium':5, 'large':7, 'nil':0.5}
            self.link_size_lu = {'small':1, 'medium':3, 'large':5, 'nil':0.2}

            # Tracking state
            self.geom_to_df  = {}
            self.last_render = None
            self.node_coords = {}

        #
        # labelOnly() - set the label only set
        # - this controls which labels will be shown
        #
        def labelOnly(self,  label_set):
            if label_set is None:        label_set = set()
            if type(label_set) == list:  label_set = set(label_set)
            if type(label_set) == str:   label_set = set([label_set])
            self.label_only  = label_set

        #
        # drawLabels() - set the draw labels flag
        #
        def drawLabels(self, draw_labels): self.draw_labels = draw_labels

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None: self.renderSVG()
            return self.last_render
        
        #
        # applyScrollEvent()
        # - zoom in or out based on the specified coordinate.
        #
        def applyScrollEvent(self, scroll_amount, coordinate=None):
            scroll_amount = scroll_amount / 1000.0
            if coordinate is not None:
                coord_wx = self.xT_inv(coordinate[0])
                coord_wy = self.yT_inv(coordinate[1])
                coordinate = (coord_wx, coord_wy)
            self.setViewWindow(self.rt_self.viewWindowZoom(self.view_window, scroll_amount, coordinate))
            return True

        #
        # applyMiddleClick()
        # - reset the view
        #
        def applyMiddleClick(self, coordinate):
            if self.view_window != self.view_window_orig:
                self.setViewWindow(self.view_window_orig)
                return True
            return False

        #
        # applyMiddleDrag()
        # - draw the view
        #        
        def applyMiddleDrag(self, coordinate, delta):
            if self.view_window is not None:
                wx0,wy0,wx1,wy1 = self.xT_inv(coordinate[0]), self.yT_inv(coordinate[1]),self.xT_inv(coordinate[0]+delta[0]), self.yT_inv(coordinate[1]+delta[1])
                dwx,dwy         = wx1-wx0, wy1-wy0
                self.setViewWindow((self.view_window[0]-dwx, self.view_window[1]-dwy, self.view_window[2]-dwx, self.view_window[3]-dwy))
                return True
            return False

        #
        # applyViewConfiguration()
        # - adjust the view window based on the other view window
        # - return True if the view actually changed (and needs a re-render)
        #
        def applyViewConfiguration(self, other):
             other_view_window = other.getViewWindow()
             if other_view_window != self.getViewWindow():
                 self.setViewWindow(other_view_window)
                 return True
             return False

        #
        # setViewWindow() - Set the view window and set flag to re-render on next call to _repr_svg_()
        # - will force a re-render on next call to _repr_svg_()
        #         
        def setViewWindow(self, view_window):
            self.view_window = view_window
            self.last_render = None
        
        #
        # getViewWindow() - return the current view window
        #
        def getViewWindow(self): return self.view_window

        #
        # nodeSVGID() - return the SVG ID for the specified node
        #
        def nodeSVGID(self, node): return self.rt_self.encSVGID(self.widget_id + '-' + str(node))

        #
        # __calculateGeometry__() - determine geometry and fill in the word coordinates into the dataframe
        # - one limitationis that you can't have mixed types in the columns...
        # -- all strings... all ints... all <whatever> ... but mixed types will fail
        #
        def __calculateGeometry__(self, for_entities=None):
            # Determine all possible nodes
            self.all_nodes = set()
            for _relationship_ in self.relationships:
                self.all_nodes |= set(self.df[_relationship_[0]])
                self.all_nodes |= set(self.df[_relationship_[1]])

            # Fill in any positions that aren't set
            for _node_ in self.all_nodes - self.pos.keys(): self.pos[_node_] = (random.random(),random.random())

            # Use a shadow pos that's forced to floats
            _pos_ = {}
            for k in self.pos.keys(): _pos_[k] = [float(self.pos[k][0]), float(self.pos[k][1])]

            # Create the operations to fill in the world coordinates for all relationship pairs
            _operations_, self.xcols, self.ycols = [], [], []
            for i in range(len(self.relationships)):
                _rel_ = self.relationships[i]
                _fm_  = _rel_[0]
                _fmx_ = f'__rel{i}_fm_wx__'
                _fmy_ = f'__rel{i}_fm_wy__'
                _operations_.append(pl.col(_fm_).replace_strict(_pos_, default=None, return_dtype=pl.List(pl.Float32)).list.get(0).alias(_fmx_))
                _operations_.append(pl.col(_fm_).replace_strict(_pos_, default=None, return_dtype=pl.List(pl.Float32)).list.get(1).alias(_fmy_))
                self.xcols.append(_fmx_), self.ycols.append(_fmy_)
                _to_  = _rel_[1]
                _tox_ = f'__rel{i}_to_wx__'
                _toy_ = f'__rel{i}_to_wy__'
                _operations_.append(pl.col(_to_).replace_strict(_pos_, default=None, return_dtype=pl.List(pl.Float32)).list.get(0).alias(_tox_))
                _operations_.append(pl.col(_to_).replace_strict(_pos_, default=None, return_dtype=pl.List(pl.Float32)).list.get(1).alias(_toy_))
                self.xcols.append(_tox_), self.ycols.append(_toy_)
            
            # Execute the operations
            self.df = self.df.with_columns(*_operations_)

            # Determine the min and max of the world coordinates
            self.wx0 = self.df[self.xcols[0]].min()
            self.wy0 = self.df[self.ycols[0]].min()
            self.wx1 = self.df[self.xcols[0]].max()
            self.wy1 = self.df[self.ycols[0]].max()
            for i in range(1,len(self.xcols)):
                self.wx0 = min(self.wx0, self.df[self.xcols[i]].min())
                self.wy0 = min(self.wy0, self.df[self.ycols[i]].min())
                self.wx1 = max(self.wx1, self.df[self.xcols[i]].max())
                self.wy1 = max(self.wy1, self.df[self.ycols[i]].max())

            # Edge cases
            if   for_entities is not None and len(for_entities) > 0:
                self.wx0 = self.wy0 = self.wx1 = self.wy1 = None
                for _entity_ in for_entities:
                    v = self.pos[_entity_]
                    if self.wx0 is None:
                        self.wx0 = self.wx1 = v[0]
                        self.wy0 = self.wy1 = v[1]
                    else:
                        self.wx0 = min(v[0], self.wx0)
                        self.wy0 = min(v[1], self.wy0)
                        self.wx1 = max(v[0], self.wx1)
                        self.wy1 = max(v[1], self.wy1)

            # This is the default... cost should be O(nodes)
            elif self.use_pos_for_bounds:
                for k in self.pos.keys():
                    v = self.pos[k]
                    self.wx0 = min(v[0], self.wx0)
                    self.wy0 = min(v[1], self.wy0)
                    self.wx1 = max(v[0], self.wx1)
                    self.wy1 = max(v[1], self.wy1)
            
            # Ensure that they aren't equal
            if abs(self.wx0 - self.wx1) < 0.001: self.wx0, self.wx1 = self.wx0 - 0.5, self.wx1 + 0.5
            if abs(self.wy0 - self.wy1) < 0.001: self.wy0, self.wy1 = self.wy0 - 0.5, self.wy1 + 0.5

            # Give some air around the boundaries
            if self.bounds_percent != 0:
                in_x = (self.wx1-self.wx0)*self.bounds_percent
                self.wx0 -= in_x
                self.wx1 += in_x
                in_y = (self.wy1-self.wy0)*self.bounds_percent
                self.wy0 -= in_y
                self.wy1 += in_y
            
            return self.wx0, self.wy0, self.wx1, self.wy1
        
        #
        # __calculateScreenCoordinates__() - add columns for the screen coordinates
        # - make the screen coordinates into integers for dataframe accumulation
        #
        def __calculateScreenCoordinates__(self):
            _operations_ = []
            for i in range(len(self.xcols)):
                _operations_.append(pl.col(self.xcols[i]).map_elements(self.xT, return_dtype=pl.Float32).cast(pl.Int32).alias(self.xcols[i].replace('wx','sx')))
                _operations_.append(pl.col(self.ycols[i]).map_elements(self.yT, return_dtype=pl.Float32).cast(pl.Int32).alias(self.ycols[i].replace('wy','sy')))
            self.df = self.df.with_columns(*_operations_)

        #
        # __renderConvexHull__() - render the convex hull
        #
        def __renderConvexHull__(self):
            # Render the convex hulls
            svg = ''
            if self.convex_hull_lu is not None and len(self.convex_hull_lu) > 0:
                _pt_lu = {} # pt_lu[convex_hull_name][node_str] = [x,y]
                for x in self.convex_hull_lu:
                    _first_value_ = self.convex_hull_lu[x]
                    break

                # Determine the points for each convex hull
                if type(_first_value_) is list or type(_first_value_) is set: # convex_hull_lu[name] = list | set of node names
                    for convex_hull_name in self.convex_hull_lu:
                        possibles = {}
                        for node_str in self.convex_hull_lu[convex_hull_name]:
                            if node_str in self.pos.keys():
                                possibles[node_str] = (self.xT(self.pos[node_str][0]), self.yT(self.pos[node_str][1]))
                        # only if something was found
                        if len(possibles) > 0: _pt_lu[convex_hull_name] = possibles

                else: # regex version
                    for rel_tuple in self.relationships:
                        if len(rel_tuple) < 2 or len(rel_tuple) > 3:
                            raise Exception(f'linkNode(): relationship tuples should have two or three parts "{rel_tuple}"')
                        fm_flds = [rel_tuple[0]]
                        to_flds = [rel_tuple[1]]                
                        gb = self.df.groupby(list(rel_tuple[:2])) if self.rt_self.isPandas(self.df) else self.df.group_by(list(rel_tuple[:2])) if self.rt_self.isPolars(self.df) else None
                        for k,k_df in gb:
                            k_fm   = k[:len(fm_flds)]
                            k_to   = k[len(fm_flds):]

                            fm_str = self.rt_self.nodeStringAndFillPos(k_fm, self.pos)
                            to_str = self.rt_self.nodeStringAndFillPos(k_to, self.pos)

                            x1 = self.xT(self.pos[fm_str][0])
                            x2 = self.xT(self.pos[to_str][0])
                            y1 = self.yT(self.pos[fm_str][1])
                            y2 = self.yT(self.pos[to_str][1])

                            for i in range(0,2):
                                if i == 0:
                                    _str = fm_str
                                    _x   = x1
                                    _y   = y1
                                else:
                                    _str = to_str
                                    _x   = x2
                                    _y   = y2

                                for my_regex in self.convex_hull_lu.keys():
                                    my_regex_name = self.convex_hull_lu[my_regex]
                                    if re.match(my_regex, _str):
                                        if my_regex_name not in _pt_lu.keys():
                                            _pt_lu[my_regex_name] = {}
                                        _pt_lu[my_regex_name][_str] = [_x,_y]

                # Render each convex hull
                for my_regex_name in _pt_lu.keys():
                    _color = self.rt_self.co_mgr.getColor(my_regex_name)
                    _pts   = _pt_lu[my_regex_name] # dictionary of node names to [x,y]
                    #
                    # Single Point
                    #
                    if   len(_pts.keys()) == 1:
                        _pt    = next(iter(_pts))
                        _x,_y  = _pts[_pt][0],_pts[_pt][1]
                        svg += f'<circle cx="{_x}" cy="{_y}" r="8" fill="{_color}" fill-opacity="{self.convex_hull_opacity}" />'
                        if self.convex_hull_stroke_width is not None:
                            _opacity = self.convex_hull_opacity + 0.2
                            if _opacity > 1:
                                _opacity = 1
                            svg += f'<circle cx="{_x}" cy="{_y}" r="8" fill-opacity="0.0" stroke="{_color}" stroke-width="{self.convex_hull_stroke_width}" stroke-opacity="{_opacity}" />'

                        # Defer labels                    
                        if self.convex_hull_labels:
                            svg_text = self.rt_self.svgText(my_regex_name, _x, self.txt_h+_y, self.txt_h, anchor='middle')
                            self.defer_render.append(svg_text)

                    #
                    # Two Points
                    #
                    elif len(_pts.keys()) == 2:
                        _my_iter = iter(_pts)
                        _pt0     = next(_my_iter)
                        _pt1     = next(_my_iter)

                        _x0,_y0  = _pts[_pt0][0],_pts[_pt0][1]
                        _x1,_y1  = _pts[_pt1][0],_pts[_pt1][1]

                        if _x0 == _x1 and _y0 == _y1:
                            svg += f'<circle cx="{_x0}" cx="{_y0}" r="8" fill="{_color}" fill-opacity="{self.convex_hull_opacity}" />'
                            if self.convex_hull_stroke_width is not None:
                                _opacity = self.convex_hull_opacity + 0.2
                                if _opacity > 1:
                                    _opacity = 1
                                svg += f'<circle cx="{_x0}" cy="{_y0}" r="8" fill-opacity="0.0" stroke="{_color}" stroke-width="{self.convex_hull_stroke_width}" stroke-opacity="{_opacity}" />'
                        else:
                            _dx  = _x1 - _x0
                            _dy  = _y1 - _y0
                            _len = sqrt(_dx*_dx+_dy*_dy)
                            if _len < 0.001:
                                _len = 0.001
                            _dx /= _len
                            _dy /= _len
                            _pdx =  _dy
                            _pdy = -_dx

                            # oblong path connecting two semicircles
                            svg_path  = ''
                            svg_path += '<path d="'
                            svg_path += f'M {_x0 + _pdx*8} {_y0 + _pdy*8} '
                            cx0 = _x0+_pdx*8 - _dx*12
                            cy0 = _y0+_pdy*8 - _dy*12
                            cx1 = _x0-_pdx*8 - _dx*12
                            cy1 = _y0-_pdy*8 - _dy*12
                            svg_path += f'C {cx0} {cy0} {cx1} {cy1} {_x0-_pdx*8} {_y0-_pdy*8} '
                            svg_path += f'L {_x1 - _pdx*8} {_y1 - _pdy*8} '
                            cx0 = _x1-_pdx*8 + _dx*12
                            cy0 = _y1-_pdy*8 + _dy*12
                            cx1 = _x1+_pdx*8 + _dx*12
                            cy1 = _y1+_pdy*8 + _dy*12
                            svg_path += f'C {cx0} {cy0} {cx1} {cy1} {_x1+_pdx*8} {_y1+_pdy*8} '
                            svg_path += f'Z" '
                            
                            svg += svg_path + f'fill="{_color}" fill-opacity="{self.convex_hull_opacity}" />'
                            
                            if self.convex_hull_stroke_width is not None:
                                _opacity = self.convex_hull_opacity + 0.2
                                if _opacity > 1:
                                    _opacity = 1
                                svg += svg_path + f'fill-opacity="0.0" stroke="{_color}" stroke-width="{self.convex_hull_stroke_width}" stroke-opacity="{_opacity}" />'

                        # Defer labels                    
                        if self.convex_hull_labels:
                            svg_text  = self.rt_self.svgText(my_regex_name, (_x0+_x1)/2, self.txt_h/2+(_y0+_y1)/2, self.txt_h)
                            self.defer_render.append(svg_text)

                    #
                    # Three or More Points
                    #
                    else:
                        _poly_pts = self.rt_self.grahamScan(_pts)
                        svg_path = ''
                        svg_path += '<path d="'
                        svg_path += self.rt_self.extrudePolyLine(_poly_pts, _pts, r=8) + '"'
                        svg += svg_path + f' fill="{_color}" fill-opacity="{self.convex_hull_opacity}" />'

                        if self.convex_hull_stroke_width is not None:
                            _opacity = self.convex_hull_opacity + 0.2
                            if _opacity > 1:
                                _opacity = 1
                            svg += svg_path + f'fill-opacity="0.0" stroke="{_color}" stroke-width="{self.convex_hull_stroke_width}" stroke-opacity="{_opacity}" />'

                        # Defer labels
                        if self.convex_hull_labels:
                            _chl_x0,_chl_x1,chl_y0,chl_y1 = None,None,None,None
                            for _poly_pt in _poly_pts:
                                _xy = _pts[_poly_pt]
                                _x  = _xy[0]
                                _y  = _xy[1]
                                if _chl_x0 is None:
                                    _chl_x0 = _chl_x1 = _xy[0]
                                    _chl_y0 = _chl_y1 = _xy[1]
                                else:
                                    if  _chl_x0 > _xy[0]:
                                        _chl_x0 = _xy[0]
                                    if  _chl_y0 > _xy[1]:
                                        _chl_y0 = _xy[1]
                                    if  _chl_x1 < _xy[0]:
                                        _chl_x1 = _xy[0]
                                    if  _chl_y1 < _xy[1]:
                                        _chl_y1 = _xy[1]

                            svg_text = self.rt_self.svgText(my_regex_name, (_chl_x0+_chl_x1)/2, self.txt_h/2 + (_chl_y0+_chl_y1)/2, self.txt_h, anchor='middle')
                            self.defer_render.append(svg_text)

            return svg

        #
        # __renderBackgroundShapes__() - render background shapes
        # - mostly a copy of the xy implementation
        #
        def __renderBackgroundShapes__(self):
            _svg_ = []
            # Draw the background shapes
            if self.bg_shape_lu is not None:
                _bg_shape_labels = []
                for k in self.bg_shape_lu.keys():
                    shape_desc = self.bg_shape_lu[k]
                    _shape_svg, _label_svg = self.rt_self.__transformBackgroundShapes__(k,                         shape_desc,
                                                                                        self.xT,                   self.yT,
                                                                                        self.bg_shape_label_color, self.bg_shape_opacity,
                                                                                        self.bg_shape_fill,        self.bg_shape_stroke_w,
                                                                                        self.bg_shape_stroke,      self.txt_h)
                    _svg_.append(_shape_svg)
                    _bg_shape_labels.append(_label_svg) # Defer render
                _svg_.extend(_bg_shape_labels)            
            return ''.join(_svg_)

        #
        # __renderLinksBasic__()
        # - baseline... very primitive
        #
        def __renderLinksBasic__(self):
            _operations_  = []
            self.linkcols = []
            _gb_str_      = []
            for i in range(len(self.relationships)):
                _link_  = f'__rel{i}_link__'
                _fm_sx_, _fm_sy_ = f'__rel{i}_fm_sx__', f'__rel{i}_fm_sy__'
                _to_sx_, _to_sy_ = f'__rel{i}_to_sx__', f'__rel{i}_to_sy__'
                _gb_str_.extend([_fm_sx_, _fm_sy_, _to_sx_, _to_sy_])
                _str_ops_ = [pl.lit('<line x1="'), pl.col(_fm_sx_), pl.lit('" y1="'), pl.col(_fm_sy_), 
                             pl.lit('" x2="'),     pl.col(_to_sx_), pl.lit('" y2="'), pl.col(_to_sy_), 
                             pl.lit('" stroke="#000000" stroke-width="2" />')]
                _operations_.append(pl.concat_str(_str_ops_).alias(_link_))
                self.linkcols.append(_link_)

            # Uniquify the x0,y0 -> x1,y1 coordinates ... then format it into svg lines
            self.df_link = self.df.group_by(_gb_str_).agg(pl.len()).with_columns(*_operations_)

            # Return the list of links
            _set_ = set()
            for i in range(len(self.linkcols)): _set_ |= set(self.df_link.drop_nulls(subset=[self.linkcols[i]])[self.linkcols[i]].unique())

            return list(_set_)

        #
        # __renderLinks__()
        #
        def __renderLinks__(self):
            _sz_ = self.link_size_lu[self.link_size] if self.link_size in self.link_size_lu else 1.0
            if type(self.link_size) == int or type(self.link_size) == float: _sz_ = self.link_size

            _set_ = set() # final set of SVG tags to append to the SVG array of strings

            for i in range(len(self.relationships)):
                _operations_, _color_calcs_, _gb_str_  = [], [], []

                _link_  = '__rel_link__'
                _fm_sx_, _fm_sy_ = f'__rel{i}_fm_sx__', f'__rel{i}_fm_sy__'
                _to_sx_, _to_sy_ = f'__rel{i}_to_sx__', f'__rel{i}_to_sy__'
                _gb_str_.extend([_fm_sx_, _fm_sy_, _to_sx_, _to_sy_])
                if self.link_size == 'vary':
                    _str_ops_ = [pl.lit('<line x1="'), pl.col(_fm_sx_), pl.lit('" y1="'), pl.col(_fm_sy_), 
                                 pl.lit('" x2="'),     pl.col(_to_sx_), pl.lit('" y2="'), pl.col(_to_sy_), 
                                 pl.lit('" stroke="'), pl.col('__color_links_final__'), pl.lit('" stroke-width="'),
                                 self.link_size_min + (self.link_size_max - self.link_size_min)*(pl.col('__count__') - pl.col('__count__').min())/(0.01 + pl.col('__count__').max() - pl.col('__count__').min()),
                                 pl.lit(f'" opacity="{self.link_opacity}" />')]
                else:
                    _str_ops_ = [pl.lit('<line x1="'),  pl.col(_fm_sx_), pl.lit('" y1="'), pl.col(_fm_sy_), 
                                 pl.lit('" x2="'),      pl.col(_to_sx_), pl.lit('" y2="'), pl.col(_to_sy_), 
                                 pl.lit(f'" stroke="'), pl.col('__color_links_final__'), pl.lit(f'" stroke-width="{_sz_}" opacity="{self.link_opacity}" />')]

                _color_calcs_.append((pl.when(pl.col('__color_links_nuniq__')==1).then(pl.col('__color_links_first__')).otherwise(pl.col('__color_default__'))).alias('__color_links_final__'))
                _operations_.append(pl.concat_str(_str_ops_).alias(_link_))

                # Define the color operations
                color_operations = [pl.col('__color_links__').n_unique().alias('__color_links_nuniq__'),
                                    pl.col('__color_default__').first(),
                                    pl.col('__color_links__').first().alias('__color_links_first__')]

                # Uniquify the x0,y0 -> x1,y1 coordinates ... then format it into svg lines
                # count by a field
                if self.link_size == 'vary' and self.count_by_set == False and self.count_by in self.df:
                    self.df_link = self.df.group_by(_gb_str_). \
                                        agg(pl.sum(self.count_by).alias('__count__'), *color_operations). \
                                        with_columns(*_color_calcs_). \
                                        with_columns(*_operations_)
                # count by a field (set-based)
                elif self.link_size == 'vary' and self.count_by_set == True and self.count_by in self.df:
                    self.df_link = self.df.group_by(_gb_str_). \
                                        agg(pl.col(self.count_by).n_unique().alias('__count__'), *color_operations). \
                                        with_columns(*_color_calcs_). \
                                        with_columns(*_operations_)
                # else either no counting or by the number of rows
                else:
                    self.df_link = self.df.group_by(_gb_str_). \
                                           agg(pl.len().alias('__count__'), *color_operations). \
                                           with_columns(*_color_calcs_). \
                                           with_columns(*_operations_)

                # Return the list of links
                if self.link_size is not None: _set_ |= set(self.df_link.drop_nulls(subset=[_link_])[_link_].unique())

            return list(_set_)
        
        #
        # __renderNodesBasic__()
        # - baseline... very primitive
        #
        def __renderNodesBasic__(self):
            # Create the node df by concatenating all fm/to columns into a common column
            _dfs_ = []
            for i in range(len(self.relationships)):
                for j in range(2):
                    if j == 0:
                        _sxfld_, _syfld_, _nmfld_ = f'__rel{i}_fm_sx__', f'__rel{i}_fm_sy__', self.relationships[i][0]
                    else:
                        _sxfld_, _syfld_, _nmfld_ = f'__rel{i}_to_sx__', f'__rel{i}_to_sy__', self.relationships[i][1]
                    _operations_ = [pl.col(_sxfld_).alias('__sx__'),
                                    pl.col(_syfld_).alias('__sy__'),
                                    pl.col(_nmfld_).alias('__nm__')]
                    _dfs_.append(self.df.with_columns(*_operations_))
            self.df_node = pl.concat(_dfs_).group_by(['__sx__','__sy__']).agg(pl.len()/2.0, pl.col('__nm__').unique())

            # Create a simple svg node via concatenation
            _str_op_ = [pl.lit('<circle cx="'), pl.col('__sx__'),
                        pl.lit('" cy="'),       pl.col('__sy__'),
                        pl.lit('" r="5" fill="#ffffff" stroke="#000000" stroke-width="1" />')]
            self.df_node = self.df_node.with_columns(pl.concat_str(_str_op_).alias('__node_svg__'))

            return list(set(self.df_node.drop_nulls(subset=['__node_svg__'])['__node_svg__'].unique()))

        #
        # __renderNodes__()
        # - expands capability to determine performance impact
        #
        def __renderNodes__(self):
            # Create the node df by concatenating all fm/to columns into a common column
            _dfs_ = []
            for i in range(len(self.relationships)):
                for j in range(2):
                    if j == 0: _sxfld_, _syfld_, _nmfld_ = f'__rel{i}_fm_sx__', f'__rel{i}_fm_sy__', self.relationships[i][0]
                    else:      _sxfld_, _syfld_, _nmfld_ = f'__rel{i}_to_sx__', f'__rel{i}_to_sy__', self.relationships[i][1]
                    _operations_ = [pl.col(_sxfld_).alias('__sx__'), 
                                    pl.col(_syfld_).alias('__sy__'), 
                                    pl.col(_nmfld_).alias('__nm__')]
                    _dfs_.append(self.df.with_columns(*_operations_).drop_nulls(subset=['__sx__','__sy__','__nm__']))
            self.df_node = pl.concat(_dfs_).group_by(['__sx__','__sy__']). \
                                            agg((pl.len()/2.0).alias('__count__'), pl.col('__nm__').unique(),
                                                 pl.col('__color_nodes__').n_unique().alias('__color_nodes_nuniq__'),
                                                 pl.col('__color_default__').first(),
                                                 pl.col('__color_nodes__').first().alias('__color_nodes_first__'))
            self.df_node = self.df_node.with_columns(pl.col('__nm__').list.len().alias('__nodes__'),
                                                     pl.col('__nm__').list.get(0).alias('__first__'),
                                                     (pl.when(pl.col('__color_nodes_nuniq__')==1).then(pl.col('__color_nodes_first__')).otherwise(pl.col('__color_default__'))).alias('__color_nodes_final__'))

            # Create the node SVG
            if    self.node_size is None: _svg_strs_ = []
            elif  self.node_size in self.node_size_lu or type(self.node_size) == int or type(self.node_size) == float:
                _sz_         = self.node_size_lu[self.node_size] if self.node_size in self.node_size_lu else self.node_size
                stroke_width = 1.0 if _sz_ > 3 else _sz_/2.0
                self.df_node = self.df_node.with_columns(pl.lit(_sz_).alias('__sz__'))
                # Single nodes
                _str_op_ = [pl.lit('<circle cx="'), pl.col('__sx__'), pl.lit('" cy="'),       pl.col('__sy__'),
                            pl.lit(f'" r="{_sz_}" fill="'),  pl.col('__color_nodes_final__'), pl.lit('" stroke="#000000" stroke-width="{stroke_width}" />')]
                df_node_singles = self.df_node.filter(pl.col('__nodes__')==1).with_columns(pl.concat_str(_str_op_).alias('__node_svg__'))
                _svg_strs_ = list(set(df_node_singles.drop_nulls(subset=['__node_svg__'])['__node_svg__'].unique()))
                # Multi nodes // nodes that are collapsed into a single pixel
                _str_op_ = [pl.lit('<use href="#cloud" x="'), pl.col('__sx__'), pl.lit('" y="'), pl.col('__sy__'), pl.lit('" />')]
                df_node_multis  = self.df_node.filter(pl.col('__nodes__')>1).with_columns(pl.concat_str(_str_op_).alias('__node_svg__'))
                _svg_strs_.extend(list(set(df_node_multis.drop_nulls(subset=['__node_svg__'])['__node_svg__'].unique())))
            elif self.node_size == 'vary':
                self.df_node = self.df_node.with_columns((self.node_size_min + (self.node_size_max - self.node_size_min) * (pl.col('__count__') - pl.col('__count__').min()) / (0.01 + pl.col('__count__').max() - pl.col('__count__').min())).alias('__sz__'))
                _str_op_ = [pl.lit('<circle cx="'), pl.col('__sx__'), pl.lit('" cy="'),       pl.col('__sy__'),
                            pl.lit('" r="'), pl.col('__sz__'), pl.lit(f'" fill="'), pl.col('__color_nodes_final__'), pl.lit('" stroke="#000000" stroke-width="{stroke_width}" />')]
                self.df_node = self.df_node.with_columns(pl.concat_str(_str_op_).alias('__node_svg__'))
                _svg_strs_ = list(set(self.df_node.drop_nulls(subset=['__node_svg__'])['__node_svg__'].unique()))
            else: _svg_strs_ = []

            # Add labels
            if self.draw_labels and len(_svg_strs_) > 0:
                # Uncollapse Nodes
                df_node_labels = self.df_node.filter(pl.col('__nodes__') == 1)
                if self.label_only  is not None and len(self.label_only)  > 0: df_node_labels = self.df_node.filter(pl.col('__first__').is_in(self.label_only)) # Filter
                df_node_labels = df_node_labels.with_columns(pl.col('__first__').cast(pl.Utf8).alias('__label__'))
                if self.node_labels is not None and len(self.node_labels) > 0: 
                    df_node_labels = df_node_labels.with_columns(pl.col('__first__').map_elements(lambda x: self.node_labels[x] if x in self.node_labels else None, return_dtype=pl.Utf8).alias('__label__'))
                df_node_labels = df_node_labels.with_columns(pl.col('__label__').str.replace_all('&','&amp').str.replace_all('<','&lt;').str.replace_all('>','&gt;'))
                _str_op_ = [pl.lit('<text x="'), pl.col('__sx__'), pl.lit('" y="'), pl.col('__sy__') + pl.col('__sz__') + self.txt_h,
                            pl.lit(f'" font-size="{self.txt_h}px" text-anchor="middle">'), pl.col('__label__'),
                            pl.lit('</text>')]
                df_node_labels = df_node_labels.with_columns(pl.concat_str(_str_op_).alias('__label_svg__'))
                df_node_labels = df_node_labels.filter(pl.col('__label_svg__').is_not_null())
                _svg_strs_.extend(list(set(df_node_labels['__label_svg__'].unique())))

                # Collapsed nodes
                df_node_labels = self.df_node.filter(pl.col('__nodes__') > 1)
                _str_op_ = [pl.lit('<text x="'), pl.col('__sx__'), pl.lit('" y="'), pl.col('__sy__') + self.txt_h + 5,
                            pl.lit(f'" font-size="{self.txt_h}px" text-anchor="middle">['), pl.col('__nodes__'),
                            pl.lit(']</text>')]
                df_node_labels = df_node_labels.with_columns(pl.concat_str(_str_op_).alias('__label_svg__'))                
                _svg_strs_.extend(list(set(df_node_labels['__label_svg__'].unique())))
            return _svg_strs_

        #
        # renderSVG() - render as SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state: self.geom_to_df = {}
            self.node_to_svg_markup = {}
            t_svg_0 = time.time()

            # Determine geometry and fill in the word coordinates into the dataframe
            self.__calculateGeometry__()
            if self.view_window is None: self.view_window = self.view_window_orig = (self.wx0, self.wy0, self.wx1, self.wy1)
            else:                        self.wx0, self.wy0, self.wx1, self.wy1 = self.view_window
                
            # Coordinate transform lambdas (and inverse lambdas)
            self.xT     = lambda __wx__:          self.x_ins + (self.w - 2*self.x_ins) * (__wx__ - self.wx0)/(self.wx1-self.wx0)
            self.yT     = lambda __wy__: self.h - self.y_ins - (self.h - 2*self.y_ins) * (__wy__ - self.wy0)/(self.wy1-self.wy0)

            self.xT_inv = lambda __sx__: self.wx0 + ((__sx__ - self.x_ins) * (self.wx1 - self.wx0))/(self.w - 2*self.x_ins)
            self.yT_inv = lambda __sy__: self.wy0 + (self.h - self.y_ins - __sy__) * (self.wy1 - self.wy0) / (self.h - 2 * self.y_ins) 

            # Add columns for the screen coordinates
            self.__calculateScreenCoordinates__()

            # Start the SVG Frame
            svg = [f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">']
            svg.append(self.rt_self.iconCloud(id="cloud")) # behind the background...
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg.append(f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />')
            self.defer_render = [] # Any deferred rendering (compatibility with linkNode)

            # Add background shapes & convex hulls
            svg.append(self.__renderBackgroundShapes__())
            svg.append(self.__renderConvexHull__())

            # Links and Nodes
            svg.extend(self.__renderLinks__())
            svg.extend(self.__renderNodes__())

            # Add any deferred rendering
            svg.extend(self.defer_render)

            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg.append(f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill-opacity="0.0" stroke="{border_color}" />')

            t_svg_1 = time.time()
            if self.draw_performance:
                t_svg   = t_svg_1 - t_svg_0
                _color_ = self.rt_self.co_mgr.getTVColor('performance','warning') if t_svg > 0.2 else self.rt_self.co_mgr.getTVColor('performance','default')
                svg.append(self.rt_self.svgText(f'{t_svg:.2f}s', self.w-3, self.h-3, 10, color=_color_, anchor='end'))
            svg.append('</svg>')
            self.last_render = ''.join(svg)
            return self.last_render

        #
        # overlappingDataFrames() - Determine which dataframe geometries overlap with a specific region
        # - to_intersect should be a shapely shape
        # - return value is a pandas dataframe or None
        #
        def overlappingDataFrames(self, to_intersect):
            return None

        #
        # overlappingEntities() - Determine which entity geometrics overlap with a specific region
        # - to_intersect should be a shapely shape
        # - return value is a list of entities (possibly an empty list) or None
        #
        def overlappingEntities(self, to_intersect):
            _point_check_ = lambda d: Point(d['__sx__'],d['__sy__']).within(to_intersect)
            _series_      = self.df_node.filter([pl.struct(['__sx__','__sy__']).map_elements(_point_check_, return_dtype=pl.Boolean)])['__nm__']
            _set_ = set()
            for x in _series_: _set_ = _set_ | set(x)
            return list(_set_)

        #
        # entitiesAtPoint() - Determine all the entities under a specific point
        #
        def entitiesAtPoint(self, xy):
            _poly_ = Polygon([(xy[0]-5,xy[1]-5),(xy[0]-5,xy[1]+5),(xy[0]+5,xy[1]+5),(xy[0]+5,xy[1]-5)])
            return self.overlappingEntities(_poly_)

        #
        # __createPathDescriptionOfSelectedEntities__() - create an svg path description of the selected entities
        # - for prototyping the graph interact panel application
        #
        def __createPathDescriptionOfSelectedEntities__(self, my_selection=None):
            if my_selection is None: my_selection = self.selected_entities
            _df_ = self.df_node.explode('__nm__').filter(pl.col('__nm__').is_in(my_selection))
            if len(_df_) == 0: return ''
            _str_ops_ = [pl.lit('M '), pl.col('__sx__')-5, 
                         pl.lit(' '),  pl.col('__sy__')-5,
                         pl.lit(' l 10 0 l 0 10 l -10 0 z')]
            _df_ = _df_.unique(['__sx__','__sy__']).with_columns(pl.concat_str(*_str_ops_).alias('__svg__'))
            return ' '.join(_df_['__svg__'])

        #
        # __createPathDescriptionForAllEntities__() - create an svg path description of all entities
        # - for prototyping the graph interact panel application
        #
        def __createPathDescriptionForAllEntities__(self):
            _str_ops_ = [pl.lit('M '), pl.col('__sx__')-5, 
                         pl.lit(' '),  pl.col('__sy__')-5,
                         pl.lit(' l 10 0 l 0 10 l -10 0 z')]
            _df_ = self.df_node.with_columns(pl.concat_str(*_str_ops_).alias('__svg__'))
            return ' '.join(_df_['__svg__'])

        #
        #  __adjustSelectedEntities__() - adjust the selected entities
        # - for prototyping the graph interact panel application
        #
        def __moveSelectedEntities__(self, dxy, my_selection=None):
            if my_selection is None: my_selection = self.selected_entities
            if my_selection is None or len(my_selection) == 0: return
            _df_ = self.df_node.explode('__nm__').filter(pl.col('__nm__').is_in(my_selection))
            for i in range(len(_df_)):
                __nm__ = _df_['__nm__'][i]
                xy     = (_df_['__sx__'][i], _df_['__sy__'][i])
                xy_new = (self.xT_inv(xy[0] + dxy[0]), self.yT_inv(xy[1] + dxy[1]))
                self.pos[__nm__] = xy_new
            self.last_render = None # force a re-render

        #
        # selectedEntities() - return the set of selected entities
        #
        def selectedEntities(self):
            return self.selected_entities

        #
        # selectEntities() - set the set of selected entities
        #
        def selectEntities(self, selection):
            self.selected_entities = selection
            self.last_render = None # invalidate the render

        #
        # entityPositions() - return information about the entity geometry for rendering
        # - Empty list means either not implemented... or entity not in view...
        # - return the positions of the entity ... rendering had to have happened first
        def __entityPositions__(self, entity):
             return []
        
        def entityPositions(self, entity_or_label):
            return []
