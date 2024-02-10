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
import re

from math import sqrt

from shapely.geometry import Polygon,LineString

from rt_component import RTComponent

__name__ = 'rt_linknode_mixin'

#
# Abstraction for LinkNode
#
class RTLinkNodeMixin(object):
    #
    # viewWindowCenter()
    # - return the center coordinate of a view window
    #
    def viewWindowCenter(self, view_window):
        _dx_ = view_window[2] - view_window[0]
        _dy_ = view_window[3] - view_window[1]
        return view_window[0] + _dx_/2, view_window[1] + _dy_/2
    
    #
    # viewWindowDimensions()
    # - return width, height of the view window
    #
    def viewWindowDimensions(self, view_window):
        return view_window[2] - view_window[0], view_window[3] - view_window[1]
    
    #
    # viewWindowZoom()
    # - zoom a view window by the specified amount
    # -- zoom > 0 == zoom_in
    # -- zoom < 0 == zoom_out
    # -- ideally, zoom in by 2.0 and zoom out by -2.0 should result in the original transform
    #
    def viewWindowZoom(self, view_window, 
                             zoom_amount=1.0,   # > 0.0 == zoom_in , < 0.0 == zoom_out...
                             zoom_center=None): # None means that the current view center will be used
        if zoom_center is None:
            zoom_center = self.viewWindowCenter(view_window)
        w,h = self.viewWindowDimensions(view_window)
        if zoom_amount > 0.0:
            exp = 1.5**zoom_amount
            w_n, h_n = w/exp, h/exp
        else:
            exp = 1.5**(-zoom_amount)
            w_n, h_n = w*exp, h*exp
        x_perc = (zoom_center[0] - view_window[0]) / w
        y_perc = (zoom_center[1] - view_window[1]) / h
        wx0_n  = zoom_center[0] - x_perc * w_n
        wy0_n  = zoom_center[1] - y_perc * h_n
        return wx0_n, wy0_n, wx0_n + w_n, wy0_n + h_n

    #
    # viewWIndowNodeFocus()
    # - construct a view window that retains all specified nodes.
    #
    def viewWindowNodeFocus(self, pos, nodes, x_perc=0.1, y_perc=0.1):
        x0,y0,x1,y1 = None,None,None,None
        for _node_ in nodes:
            if _node_ in pos.keys():
                x,y = pos[_node_][0], pos[_node_][1]
                if x0 is None:
                    x0 = x1 = x
                    y0 = y1 = y
                else:
                    x0,x1 = min(x0, x), max(x1, x)
                    y0,y1 = min(y0, y), max(y1, y)
        if x0 is None:
            raise Exception('viewWindowNodeFocus() - no nodes with coordinates')
        if x0 == x1: # make sure it's not the same value
            x = x0
            x0,x1 = x - 0.5, x + 0.5
        if y0 == y1: # make sure it's not the same value
            y = y0
            y0,y1 = y - 0.5, y + 0.5
        if x_perc > 0.0: # add percentage of view back in
            d = x1 - x0
            x0 -= d * x_perc
            x1 += d * x_perc
        if y_perc > 0.0: # add percentage of view back in
            d = y1 - y0
            y0 -= d * y_perc
            y1 += d * y_perc
        return x0, y0, x1, y1

    #
    # nodeLabeler()
    # - Create the dictionary for the node_labels parameter
    # - if node_labels is passed, will be added to / not replaced
    #
    def nodeLabeler(self, df, node_field, label_field, node_labels=None, word_wrap=True, max_line_len=32, max_lines=4):
        if node_labels is None:
            node_labels = {}
        gb = df.groupby(node_field) if self.isPandas(df) else df.group_by(node_field) if self.isPolars(df) else None
        for k,k_df in gb:
            node_str = self.nodeString(k)
            label_array = node_labels[node_str] if node_str in node_labels.keys() else [] # maybe just adding to?
            field_set   = set(k_df[label_field])
            _str_       = ''
            if len(field_set) == 1:
                _str_ = str(list(field_set)[0])
            else:
                _as_list_ = list(field_set)
                _str_     = str(_as_list_[0])
                for i in range(1,len(_as_list_)):
                    _str_ += ' ' + str(_as_list_[i])
            
            # Split the string into lines if it's greater than max_line_len and word_wrap is True
            if len(_str_) >= max_line_len and word_wrap == True:
                _lines_           = _str_.split('\n')
                dot_dot_dot_added = False
                _line_no_         = 0
                for _line_ in _lines_:
                    _parts_ = _line_.split() # splits by whitespace w/out any params...
                    if len(_parts_) > 0:
                        _line_ = _parts_[0]
                        for i in range(1,len(_parts_)):
                            if len(_line_ + ' ' + _parts_[i]) < max_line_len:
                                _line_ += ' ' + _parts_[i]
                            else:
                                if _line_no_ < max_lines:
                                    label_array.append(_line_)
                                    _line_no_ += 1
                                elif dot_dot_dot_added == False:
                                    label_array.append('...')
                                    dot_dot_dot_added = True                                
                                _line_ = _parts_[i]
                        if len(_line_) > 0 and _line_no_ < max_lines:
                            label_array.append(_line_)
                            _line_no_ += 1
                        elif dot_dot_dot_added == False:
                            label_array.append('...')
                            dot_dot_dot_added = True
            elif len(_str_) >= max_line_len:
                label_array.append(_str_[:max_line_len] + '...')
            else:
                label_array.append(_str_)

            node_labels[node_str] = label_array
        return node_labels

    #
    # nodeString()
    # - like nodeStringAndFill() but without the pos parameter
    #
    def nodeString(self, k):        
        # Figure out the actual string (or integer)
        if type(k) == tuple or type(k) == list:
            if len(k) == 1:
                node_str = k[0]
            else:
                node_str = str(k[0])
                for i in range(1,len(k)):
                    node_str = node_str + '|' + str(k[i])
        else:
            node_str = k
        # Make sure it's a string
        if type(node_str) != str:
            node_str = str(node_str)
        return node_str

    #
    # nodeStringAndFillPos()
    # - create a node string... complicated due to possible occurence of ints...
    #
    def nodeStringAndFillPos(self, k, pos=None):
        # Figure out the actual string (or integer)
        if type(k) == tuple or type(k) == list:
            if len(k) == 1:
                node_str = k[0]
            else:
                node_str = str(k[0])
                for i in range(1,len(k)):
                    node_str = node_str + '|' + str(k[i])
        else:
            node_str = k

        # Get or make the node's position
        if pos is not None:
            if type(node_str) == str:
                if node_str not in pos.keys():
                    pos[node_str] = [random.random(),random.random()]
            else:
                if node_str in pos.keys():
                    pos[str(node_str)] = pos[node_str]
                    node_str = str(node_str)
                else:
                    node_str = str(node_str)
                    if node_str not in pos.keys():
                        pos[node_str] = [random.random(),random.random()]
        return node_str

    #
    # Calculate Information About the Nodes
    # ... mostly a copy of the node render loop... should probably be refactored
    #
    def calculateNodeInformation(self, df, relationships, pos, count_by, count_by_set):
        # Boundary
        wx0 = math.inf
        wy0 = math.inf
        wx1 = -math.inf
        wy1 = -math.inf

        # Maximum node value
        max_node_value = 0

        # Iterate over the relationships
        for rel_tuple in relationships:
            # Make sure it's the right number of tuples
            if len(rel_tuple) != 2:
                raise Exception(f'linkNode(): relationship tuples should have two parts "{rel_tuple}"')

            # Flatten out into the groupby array, the fm_flds array, and the to_flds array            
            flat    = flattenTuple(rel_tuple)

            fm_flds = flattenTuple(rel_tuple[0])
            if type(fm_flds) != list:
                fm_flds = [fm_flds]

            to_flds = flattenTuple(rel_tuple[1])                
            if type(to_flds) != list:
                to_flds = [to_flds]

            # Do the from and to fields separately
            for flds_i in range(0,2):
                if flds_i == 0:
                    flds = fm_flds
                else:
                    flds = to_flds

                # Iterate over the dfs
                for _df in df:

                    # if the _df has all of the columns
                    if len(set(_df.columns) & set(flds)) == len(set(flds)):

                        # create the edge table
                        if len(flds) == 1:
                            if self.isPandas(_df):
                                gb = _df.groupby(flds[0])
                            elif self.isPolars(_df):
                                gb = _df.group_by(flds[0])
                        else:
                            if self.isPandas(_df):
                                gb = _df.groupby(flds)
                            elif self.isPolars(_df):
                                gb = _df.group_by(flds)

                        # iterate over the edges
                        for k,k_df in gb:
                            node_str = self.nodeStringAndFillPos(k, pos)

                            # Perform the comparison for the bounds
                            v = pos[node_str]
                            wx1 = max(v[0], wx1)
                            wy1 = max(v[1], wy1)
                            wx0 = min(v[0], wx0)
                            wy0 = min(v[1], wy0)

                            # Determine the maximum node size
                            if count_by is None:
                                if max_node_value < len(k_df):
                                    max_node_value = len(k_df)
                            elif count_by in _df.columns and count_by_set:
                                set_size = len(k_df[count_by])
                                if max_node_value < set_size:
                                    max_node_value = set_size
                            elif count_by in _df.columns:
                                summation = k_df[count_by].sum()
                                if max_node_value < summation:
                                    max_node_value = summation

        # Make sure the max node value is not zero
        if max_node_value == 0:
            max_node_value = 1

        return max_node_value, wx0, wy0, wx1, wy1

    #
    # linkNodePreferredDimensions()
    # - Return the preferred size
    #
    def linkNodePreferredDimensions(self, **kwargs):
        return (256,256)

    #
    # linkNodeMinimumDimensions()
    # - Return the minimum size
    #
    def linkNodeMinimumDimensions(self, **kwargs):
        return (32,32)

    #
    # linkNodeSmallMultipleDimensions()
    # - Return the minimum size
    #
    def linkNodeSmallMultipleDimensions(self, **kwargs):
        return (32,32)

    #
    # Identify the required fields in the dataframe from linknode parameters
    #
    def linkNodeRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('relationships', kwargs, columns_set)
        self.identifyColumnsFromParameters('color_by',      kwargs, columns_set)
        self.identifyColumnsFromParameters('count_by',      kwargs, columns_set)
        if 'timing_marks' in kwargs.keys() and kwargs['timing_marks'] == True:
            self.identifyColumnsFromParameters('ts_field',  kwargs, columns_set)
            
        # Ignoring the small multiples version // for now
        return columns_set

    #
    # linkNode
    #
    # Make the SVG for a link node from a set of dataframes
    #    
    def linkNode(self,
                 df,                           # dataframe(s) to render ... unlike other parts, this can be more than one...
                 relationships,                # list of tuple pairs... pairs can be single strings or tuples of strings
                                               # [('f0','f1')] // 1 relationship: f0 to f1
                                               # [('f0','f1'),('f1','f2')] // 2 relationships: f0 to f1 and f1 to f2
                                               # [(('f0','f1'),('f2','f3'))] // 1 relationship: 'f0'|'f1' to 'f2'|'f3'

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

                 max_node_size     = 4,        # for node vary...
                 min_node_size     = 0.3,      # for node vary...

                 link_color        = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
                 link_size         = 'small',  # 'nil', 'small', 'medium', 'large', 'vary', 'hidden' / None
                 link_opacity      = '1.0',    # link opacity
                 link_shape        = 'line',   # 'curve','line'
                 link_arrow        = True,     # draw an arrow at the end of the curve...
                 link_arrow_length = 10,       # length in pixels of the link arrow
                 link_dash         = None,     # svg stroke-dash string, callable, or dictionary of relationship tuple to dash string array 

                 link_max_curvature_px = 100,  # maximum link curvature outward
                 link_parallel_perc    = 0.2,  # percent for control point parallel to the link
                 link_ortho_perc       = 0.2,  # percent for control point orthogonal to the link

                 max_link_size     = 4,        # for link vary...
                 min_link_size     = 0.25,     # for link vary...

                 label_only        = set(),    # label only set

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
                 draw_labels           = True,   # draw labels flag # not implemented yet
                 draw_border           = True):  # draw a border around the graph
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTLinkNode(self, **_params_)

    #
    # __minAndMaxLinkSize__()
    # ... copy of the next method but only determines the min and max link size
    #
    def __minAndMaxLinkSize__(self, df, relationships, count_by=None):
        _min_,_max_ = None,None
        # Make the df into a list
        if type(df) != list:
            df = [df]
        # Check the count_by column across all the df's...  if any of them
        # don't work.. then it's count_by_set
        count_by_set = False
        if count_by is not None:
            for _df in df:
                if self.fieldIsArithmetic(_df, count_by) == False:
                    count_by_set = True
        # Iterate over the relationships
        for rel_tuple in relationships:
            # Flatten out into the groupby array, the fm_flds array, and the to_flds array            
            flat    = flattenTuple(rel_tuple)
            fm_flds = flattenTuple(rel_tuple[0])
            if type(fm_flds) != list:
                fm_flds = [fm_flds]
            to_flds = flattenTuple(rel_tuple[1])                
            if type(to_flds) != list:
                to_flds = [to_flds]
            # Iterate over the dfs
            for _df in df:
                # if the _df has all of the columns
                if len(set(_df.columns) & set(flat)) == len(set(flat)):
                    if self.isPandas(_df):
                        gb = _df.groupby(flat)
                        if count_by is None: 
                            gb_sz = gb.size()
                        elif count_by_set:
                            gb_sz = gb[count_by].nunique()
                        else:
                            gb_sz = gb[count_by].sum()
                        for i in range(0,len(gb)):
                            _weight_ = gb_sz.iloc[i]
                            _min_ = _weight_ if _min_ is None else min(_min_, _weight_)
                            _max_ = _weight_ if _max_ is None else max(_max_, _weight_)
                    elif self.isPolars(_df):
                        counter = self.polarsCounter(_df, flat, count_by, count_by_set)
                        _min_ = counter['__count__'].min()
                        _max_ = counter['__count__'].max()
                    else:
                        raise Exception('RTLinkNode.minAndMaxLinkSize() - only pandas and polars supported')
        if _min_ == _max_:
            _max_ = _min_ + 1
        return _min_,_max_

    #
    # createNetworkXGraph()
    #
    # Use the same construction technique as linkNode but make a networkx graph instead.
    #    
    def createNetworkXGraph(self,
                            df,              # dataframe(s) to render ... unlike other parts, this can be more than one...
                            relationships,   # list of tuple pairs... pairs can be single strings or tuples of strings
                            count_by=None):  # edge weight field
        # Make the df into a list
        if type(df) != list:
            df = [df]

        # Check the count_by column across all the df's...  if any of them
        # don't work.. then it's count_by_set
        count_by_set = False
        if count_by is not None:
            for _df in df:
                if self.fieldIsArithmetic(_df, count_by) == False:
                    count_by_set = True

        # Create the return graph structure
        nx_g = nx.Graph()

        # Iterate over the relationships
        for rel_tuple in relationships:
            # Make sure it's the right number of tuples
            if len(rel_tuple) != 2:
                raise Exception(f'createNetworkXGraph(): relationship tuples should have two parts "{rel_tuple}"')

            # Flatten out into the groupby array, the fm_flds array, and the to_flds array            
            flat    = flattenTuple(rel_tuple)

            fm_flds = flattenTuple(rel_tuple[0])
            if type(fm_flds) != list:
                fm_flds = [fm_flds]

            to_flds = flattenTuple(rel_tuple[1])                
            if type(to_flds) != list:
                to_flds = [to_flds]

            # Iterate over the dfs
            for _df in df:

                def stringify(_list_):
                    _str_ = str(_list_[0])
                    for _x_ in _list_[1:]:
                        _str_ += '|' + str(_x_)
                    return _str_

                # if the _df has all of the columns
                if len(set(_df.columns) & set(flat)) == len(set(flat)):
                    if self.isPandas(_df):
                        if count_by is None or count_by_set: # count_by_set not implemented...
                            gb = _df.groupby(flat).size()
                        else:
                            gb = _df.groupby(flat)[count_by].sum()
                        for i in range(0,len(gb)):
                            k = gb.index[i]
                            k_fm   = k[:len(fm_flds)]
                            k_to   = k[len(fm_flds):]
                            _fm_   = stringify(k_fm)
                            _to_   = stringify(k_to)
                            nx_g.add_edge(_fm_,_to_,weight=gb.iloc[i])
                    elif self.isPolars(_df):
                        counter = self.polarsCounter(_df, flat, count_by, count_by_set)
                        for i in range(len(counter)):
                            _row_   = counter[i]
                            fm_list = []
                            for _fm_fld_ in fm_flds:
                                fm_list.append(_row_[_fm_fld_][0])
                            to_list = []
                            for _to_fld_ in to_flds:
                                to_list.append(_row_[_to_fld_][0])
                            _fm_ = stringify(fm_list)
                            _to_ = stringify(to_list)
                            nx_g.add_edge(_fm_,_to_,weight=_row_['__count__'][0])
                    else:
                        raise Exception('RTLinkNode.createNetworkXGraph() - only pandas and polars is supported')
        return nx_g

    #
    # RTLinkNode Class
    #
    class RTLinkNode(RTComponent):
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

            # Make a widget_id if it's not set already
            if self.widget_id is None:
                self.widget_id = "linknode_" + str(random.randint(0,65535))

            self.node_color                 = kwargs['node_color']
            self.node_border_color          = kwargs['node_border_color']
            self.node_size                  = kwargs['node_size']
            self.node_shape                 = kwargs['node_shape']
            self.node_opacity               = kwargs['node_opacity']
            self.node_labels                = kwargs['node_labels']
            self.node_labels_only           = kwargs['node_labels_only']
            self.max_node_size              = kwargs['max_node_size']
            self.min_node_size              = kwargs['min_node_size']
            self.link_color                 = kwargs['link_color']
            self.link_size                  = kwargs['link_size']
            self.link_opacity               = kwargs['link_opacity']
            self.link_shape                 = kwargs['link_shape']
            self.link_arrow                 = kwargs['link_arrow']
            self.link_arrow_length          = kwargs['link_arrow_length']
            self.link_dash                  = kwargs['link_dash']
            self.link_max_curvature_px      = kwargs['link_max_curvature_px']
            self.link_parallel_perc         = kwargs['link_parallel_perc']
            self.link_ortho_perc            = kwargs['link_ortho_perc']
            self.max_link_size              = kwargs['max_link_size']
            self.min_link_size              = kwargs['min_link_size']
            self.label_only                 = kwargs['label_only']
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
            self.draw_labels                = kwargs['draw_labels']
            self.draw_border                = kwargs['draw_border']

            # Make sure it's a list... and prevent the added columns from corrupting original dataframe
            my_df_list = []
            if type(kwargs['df']) != list:
                my_df_list.append(rt_self.copyDataFrame(kwargs['df']))
            else:
                for _df in kwargs['df']:
                    my_df_list.append(rt_self.copyDataFrame(_df))
            self.df = my_df_list
            
            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)

            # Apply node field transforms across all of the dataframes
            for _df in self.df:
                for _edge in self.relationships_orig:
                    for _node in _edge:
                        if type(_node) == str:
                            if rt_self.isTField(_node) and rt_self.tFieldApplicableField(_node) in _df.columns:
                                _df,_throwaway = rt_self.applyTransform(_df, _node)
                        else:
                            for _tup_part in _node:
                                if rt_self.isTField(_tup_part) and rt_self.tFieldApplicableField(_tup_part) in _df.columns:
                                    _df,_throwaway = rt_self.applyTransform(_df, _tup_part)

            # vvv
            # vvv -- REMOVABLE (UNTIL WE MODIFIED THE REST OF THE CODE BASE)
            # vvv
            # Determine if all columns are in the df
            def allColumnsInDF(_df_, _cols_):
                for x in _cols_:
                    if x not in _df_.columns:
                        return False
                return True

            # Create concatenated fields for the tuple nodes
            self.relationships, i = [], 0
            for _edge_ in self.relationships_orig:
                _fm_ = _edge_[0]
                _to_ = _edge_[1]
                if type(_fm_) == tuple or type(_to_) == tuple:
                    new_fm, new_to = _fm_, _to_
                    new_dfs = []
                    for _df_ in self.df:
                        if type(_fm_) == tuple and allColumnsInDF(_df_, _fm_):
                            new_fm = f'__fm{i}__'
                            _df_ = self.rt_self.createConcatColumn(_df_, _fm_, new_fm)

                        if type(_to_) == tuple and allColumnsInDF(_df_, _fm_):
                            new_to = f'__to{i}__'
                            _df_ = self.rt_self.createConcatColumn(_df_, _to_, new_to)

                        new_dfs.append(_df_)
                    self.relationships.append((new_fm, new_to))
                    self.df = new_dfs
                else:
                    self.relationships.append((_fm_, _to_))
                i += 1
            # ^^^
            # ^^^ -- REMOVABLE (UNTIL WE MODIFY THE REST OF THE CODE BASE)
            # ^^^

            # Check the node information... make sure the parameters are set
            if self.sm_type is not None and self.sm_mode == 'node':
                self.node_shape = 'small_multiple'
            if self.sm_type is not None and (self.sm_w is None or self.sm_h is None):
                    self.sm_w,self.sm_h = getattr(rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
            if callable(self.node_shape) and self.node_size is None:
                self.node_size = 'medium'

            # Check the count_by column across all the df's...  if any of them
            # don't work.. then it's count_by_set
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)
            
            # Tracking state
            self.geom_to_df  = {}
            self.last_render = None
            self.node_coords = {}

        #
        # __calculateGeometry__() - determine the geometry for the view
        #
        def __calculateGeometry__(self):
            # Calculate world coordinates
            self.wx0 =  math.inf
            self.wy0 =  math.inf
            self.wx1 = -math.inf
            self.wy1 = -math.inf

            # And possibly the max node size
            self.max_node_value = 1

            if self.use_pos_for_bounds:
                for k in self.pos.keys():
                    v = self.pos[k]
                    self.wx0 = min(v[0], self.wx0)
                    self.wy0 = min(v[1], self.wy0)
                    self.wx1 = max(v[0], self.wx1)
                    self.wy1 = max(v[1], self.wy1)
                if self.node_size == 'vary':
                    self.max_node_value,ignore0,ignore1,ignore2,ignore3 = self.rt_self.calculateNodeInformation(self.df, self.relationships, self.pos, self.count_by, self.count_by_set)
            else:
                self.max_node_value,self.wx0,self.wy0,self.wx1,self.wy1 = self.rt_self.calculateNodeInformation(self.df, self.relationships, self.pos, self.count_by, self.count_by_set)

            # Make it sane
            if math.isinf(self.wx0):
                self.wx0 = 0.0
                self.wx1 = 1.0
            if math.isinf(self.wy0):
                self.wy0 = 0.0
                self.wy1 = 1.0

            # Make it sane some more
            if self.wx0 == self.wx1:
                self.wx0 -= 0.5
                self.wx1 += 0.5
            if self.wy0 == self.wy1:
                self.wy0 -= 0.5
                self.wy1 += 0.5

            # Give some air around the boundaries
            if self.bounds_percent != 0:
                in_x = (self.wx1-self.wx0)*self.bounds_percent
                self.wx0 -= in_x
                self.wx1 += in_x
                in_y = (self.wy1-self.wy0)*self.bounds_percent
                self.wy0 -= in_y
                self.wy1 += in_y

        #
        # __renderConvexHull__() - render the convex hull
        #
        def __renderConvexHull__(self):
            # Render the convex hulls
            svg = ''
            if self.convex_hull_lu is not None:
                _pt_lu = {} # pt_lu[convex_hull_name][node_str][x,y]

                # Determine the points for each convex hull
                for rel_tuple in self.relationships:
                    if len(rel_tuple) != 2:
                        raise Exception(f'linkNode(): relationship tuples should have two parts "{rel_tuple}"')
                    flat    = flattenTuple(rel_tuple)
                    fm_flds = flattenTuple(rel_tuple[0])
                    if type(fm_flds) != list:
                        fm_flds = [fm_flds]
                    to_flds = flattenTuple(rel_tuple[1])                
                    if type(to_flds) != list:
                        to_flds = [to_flds]                    
                    for _df in self.df:
                        if len(set(_df.columns) & set(flat)) == len(set(flat)):
                            gb = _df.groupby(flat) if self.rt_self.isPandas(_df) else _df.group_by(flat) if self.rt_self.isPolars(_df) else None
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
        # __renderLinks__() - return links
        #
        def __renderLinks__(self):
            # Render links
            svg          = ''
            count_by_set = True
            if self.link_size is not None and self.link_size != 'hidden':
                link_to_dfs, link_to_xy = {}, {} # For small multiples (if enabled)
                # Set the link size
                if   type(self.link_size) == dict:
                    _sz = 1
                elif type(self.link_size) == int or type(self.link_size) == float:
                    _sz = self.link_size
                elif self.link_size == 'small':
                    _sz = 1
                elif self.link_size == 'medium':
                    _sz = 3
                elif self.link_size == 'large':
                    _sz = 5
                elif self.link_size == 'nil':
                    _sz = 0.2
                else: # Vary
                    # Check the count_by column across all the df's...  if any of them
                    # don't work.. then it's count_by_set
                    count_by_set = False
                    if self.count_by is not None:
                        for _df in self.df:
                            if self.rt_self.fieldIsArithmetic(_df, self.count_by) == False:
                                count_by_set = True
                    _sz_min, _sz_max = self.rt_self.__minAndMaxLinkSize__(self.df, self.relationships, self.count_by)
                    _sz = None

                # Iterate over the relationships
                for rel_tuple in self.relationships:
                    # Make sure it's the right number of tuples
                    if len(rel_tuple) != 2:
                        raise Exception(f'linkNode(): relationship tuples should have two parts "{rel_tuple}"')

                    # Flatten out into the groupby array, the fm_flds array, and the to_flds array            
                    flat    = flattenTuple(rel_tuple)

                    fm_flds = flattenTuple(rel_tuple[0])
                    if type(fm_flds) != list:
                        fm_flds = [fm_flds]

                    to_flds = flattenTuple(rel_tuple[1])                
                    if type(to_flds) != list:
                        to_flds = [to_flds]

                    # Iterate over the dfs
                    for _df in self.df:

                        # if the _df has all of the columns
                        if len(set(_df.columns) & set(flat)) == len(set(flat)):
                            gb = _df.groupby(flat) if self.rt_self.isPandas(_df) else _df.group_by(flat) if self.rt_self.isPolars(_df) else None

                            if self.rt_self.isPandas(_df):
                                if self.count_by is None: 
                                    gb_sz = gb.size()
                                elif count_by_set:
                                    gb_sz = gb[self.count_by].nunique()
                                else:
                                    gb_sz = gb[self.count_by].sum()
                            else:
                                counter = self.rt_self.polarsCounter(_df, flat, self.count_by, self.count_by_set)

                            gb_sz_i = 0
                            for k,k_df in gb:
                                if self.rt_self.isPandas(_df):
                                    _weight_ =  gb_sz.iloc[gb_sz_i]
                                    gb_sz_i  += 1
                                else:
                                    if self.count_by is None:
                                        _weight_ = len(k_df)
                                    elif count_by_set:
                                        _weight_ = k_df[self.count_by].n_unique()
                                    else:
                                        _weight_ = k_df[self.count_by].sum()

                                k_fm   = k[:len(fm_flds)]
                                k_to   = k[len(fm_flds):]

                                fm_str = self.rt_self.nodeStringAndFillPos(k_fm, self.pos)
                                to_str = self.rt_self.nodeStringAndFillPos(k_to, self.pos)
                                
                                # Transform the coordinates
                                x1 = self.xT(self.pos[fm_str][0])
                                x2 = self.xT(self.pos[to_str][0])
                                y1 = self.yT(self.pos[fm_str][1])
                                y2 = self.yT(self.pos[to_str][1])
                                                            
                                # Determine the color
                                if   self.link_color == 'vary' and self.color_by is not None and self.color_by in _df.columns:
                                    _co_set = set(k_df[self.color_by])
                                    if len(_co_set) == 1:
                                        _co = self.rt_self.co_mgr.getColor(_co_set.pop())
                                    else:
                                        _co = self.rt_self.co_mgr.getTVColor('data','default')
                                elif self.link_color is not None and self.link_color.startswith('#'):
                                    _co = self.link_color
                                else:
                                    _co = self.rt_self.co_mgr.getTVColor('data','default')

                                # Capture the state
                                if self.track_state:
                                    _line = LineString([[x1,y1],[x2,y2]])
                                    if _line not in self.geom_to_df.keys():
                                        self.geom_to_df[_line] = []
                                    self.geom_to_df[_line].append(k_df)
                                
                                # Determine the size
                                if _sz is None:
                                    _this_sz = self.min_link_size + self.max_link_size * (_weight_ - _sz_min) / (_sz_max - _sz_min)
                                else:
                                    if type(self.link_size) == dict:
                                        if rel_tuple in self.link_size.keys():
                                            _str_ = self.link_size[rel_tuple]
                                            if   type(_str_) == int or type(_str_) == float:
                                                _this_sz = _str_
                                            elif _str_ == 'small':
                                                _this_sz = 1
                                            elif _str_ == 'medium':
                                                _this_sz = 3
                                            elif _str_ == 'large':
                                                _this_sz = 5
                                            elif _str_ == 'nil':
                                                _this_sz = 0.2
                                            else:
                                                _this_sz = 0.0
                                        else:
                                            _this_sz = 0.0
                                    else:
                                        _this_sz = _sz
                                
                                # Determine stroke dash
                                stroke_dash = ''
                                if self.link_dash is not None:
                                    if   type(self.link_dash) == str:
                                        stroke_dash = f'stroke-dasharray="{self.link_dash}"'
                                    elif type(self.link_dash) == dict and rel_tuple in self.link_dash:
                                        stroke_dash = f'stroke-dasharray="{self.link_dash[rel_tuple]}"'
                                    elif callable(self.link_dash):
                                        _return_value_ = self.link_dash(fm_str, to_str, (x1,y1), (x2,y2))
                                        if _return_value_ is not None:
                                            stroke_dash = f'stroke-dasharray="{_return_value_}"'

                                # Determine the link style
                                if    self.link_shape == 'line':
                                    svg += f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                                    svg += f'stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" {stroke_dash} />'

                                    def _xyLink_(t):
                                        return x1+(x2-x1)*t, y1+(y2-y1)*t
                                    if fm_str < to_str:
                                        _xyLinkDir_ = _xyLink_
                                    else:
                                        def _xyLinkDir_(t):
                                            return x2+(x1-x2)*t, y2+(y1-y2)*t

                                    if self.link_arrow:
                                        dx, dy = x2 - x1, y2 - y1
                                        l = sqrt((dx*dx)+(dy*dy))
                                        if l <= 0.01:
                                            l = 1
                                        dx /= l
                                        dy /= l

                                        x3 = x2 - dx*self.link_arrow_length - dy*3*self.link_arrow_length/4
                                        y3 = y2 - dy*self.link_arrow_length + dx*3*self.link_arrow_length/4
                                        x4 = x2 - dx*self.link_arrow_length + dy*3*self.link_arrow_length/4
                                        y4 = y2 - dy*self.link_arrow_length - dx*3*self.link_arrow_length/4

                                        svg += f'<path d="M {x3} {y3} L {x2} {y2} L {x4} {y4}" '
                                        svg += f'fill-opacity="0.0" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" />'
                                elif self.link_shape == 'curve':
                                    dx = x2 - x1
                                    dy = y2 - y1
                                    # vector length
                                    l  = sqrt((dx*dx)+(dy*dy))
                                    if l <= 0.01:
                                        l = 1

                                    # normalize the vector
                                    dx /= l
                                    dy /= l

                                    # calculate the perpendicular vector
                                    pdx =  dy
                                    pdy = -dx

                                    # bound the link curvature
                                    _link_curve_ = self.link_max_curvature_px if l > self.link_max_curvature_px else l

                                    # calculate the control points
                                    x1p = x1 + self.link_parallel_perc*_link_curve_*dx + self.link_ortho_perc*_link_curve_*pdx
                                    y1p = y1 + self.link_parallel_perc*_link_curve_*dy + self.link_ortho_perc*_link_curve_*pdy

                                    x2p = x2 - self.link_parallel_perc*_link_curve_*dx + self.link_ortho_perc*_link_curve_*pdx
                                    y2p = y2 - self.link_parallel_perc*_link_curve_*dy + self.link_ortho_perc*_link_curve_*pdy

                                    def _xyLink_(t): # Bezier Curve Formula from Wikipedia
                                        return (1-t)**3*x1+3*(1-t)**2*t*x1p+3*(1-t)*t**2*x2p+t**3*x2,(1-t)**3*y1+3*(1-t)**2*t*y1p+3*(1-t)*t**2*y2p+t**3*y2
                                    if fm_str < to_str:
                                        _xyLinkDir_ = _xyLink_
                                    else:
                                        def _xyLinkDir_(t):
                                            return (1-t)**3*x2+3*(1-t)**2*t*x2p+3*(1-t)*t**2*x1p+t**3*x1,(1-t)**3*y2+3*(1-t)**2*t*y2p+3*(1-t)*t**2*y1p+t**3*y1

                                    edx, edy = _xyLink_(1.0 - 0.05) # Calculate the endpoint derivative
                                    edx, edy = x2 - edx, y2 - edy
                                    l = sqrt((edx*edx)+(edy*edy))
                                    l = 1.0 if l < 0.01 else l
                                    edx, edy = edx/l, edy/l        # As a unit vector

                                    x3  = x2 - self.link_arrow_length*edx - (self.link_arrow_length/2) * (-edy)
                                    y3  = y2 - self.link_arrow_length*edy - (self.link_arrow_length/2) * ( edx)

                                    x4  = x2 - self.link_arrow_length*edx + (self.link_arrow_length/2) * (-edy)
                                    y4  = y2 - self.link_arrow_length*edy + (self.link_arrow_length/2) * ( edx)

                                    if self.link_arrow:
                                        svg += f'<path d="M {x1} {y1} C {x1p} {y1p} {x2p} {y2p} {x2} {y2} M {x3} {y3} L {x2} {y2} L {x4} {y4}" '
                                        svg += f'fill-opacity="0.0" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" {stroke_dash} />'
                                    else:
                                        svg += f'<path d="M {x1} {y1} C {x1p} {y1p} {x2p} {y2p} {x2} {y2}" '
                                        svg += f'fill-opacity="0.0" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" {stroke_dash} />'
                                else:
                                    raise Exception(f'Unknown link_shape "{self.link_shape}"')
                                
                                # Small multiples
                                if self.sm_mode == 'link' and self.sm_type is not None:
                                    link_to_dfs[fm_str + '->' + to_str] = k_df
                                    _x_, _y_ = _xyLink_(self.sm_t)
                                    if self.link_shape == 'line': # For linear version, offset one side a little to make it visible
                                        if fm_str < to_str:
                                            _x_ += 2
                                            _y_ += 2
                                        else:
                                            _x_ -= 2
                                            _y_ -= 2
                                    link_to_xy[fm_str + '->' + to_str]  = (_x_, _y_)
                                
                                # Timing marks
                                if self.timing_marks and self.ts_field is not None and self.ts_field in k_df.columns and self.rt_self.isPandas(k_df):
                                    _tfield_, _tml_ = '_linknode_tms_', self.timing_mark_length
                                    _side_ = 1.0 if fm_str < to_str else -1.0
                                    k_df[_tfield_] = (k_df[self.ts_field] - _df[self.ts_field].min()) / (_df[self.ts_field].max() - _df[self.ts_field].min())
                                    for row_i, row in k_df.iterrows():
                                        _color_ = self.rt_self.co_mgr.spectrumAbridged(row[_tfield_], 0.0, 1.0)
                                        _t_box_     = 0.1 + 0.8 * row[_tfield_]
                                        _x_  , _y_  = _xyLinkDir_(_t_box_)
                                        _xp_ , _yp_ = _xyLinkDir_(_t_box_+0.01)  # slight offset point
                                        _dx_ , _dy_ = _xp_ - _x_ , _yp_ - _y_          # slope at this location
                                        _l_         = sqrt(_dx_*_dx_ + _dy_*_dy_)
                                        _l_         = 1.0 if _l_ < 0.001 else _l_
                                        _dx_ , _dy_ = _dx_ / _l_ , _dy_ / _l_          # unitize the vector
                                        _xe_ , _ye_ = _x_ - _side_ * _dx_ * _tml_/2 + _side_ * _dy_ * _tml_, _y_ - _side_ * _dy_ * _tml_/2 - _side_ * _dx_ * _tml_
                                        svg += f'<line x1="{_x_}" y1="{_y_}" x2="{_xe_}" y2="{_ye_}" stroke="{_color_}" stroke-width="1.5" />'
                                elif self.timing_marks and self.ts_field is not None and self.ts_field in k_df.columns and self.rt_self.isPolars(k_df):
                                    _tfield_, _tml_ = '_linknode_tms_', self.timing_mark_length
                                    _side_ = 1.0 if fm_str < to_str else -1.0
                                    my_min, my_max = _df[self.ts_field].min(), _df[self.ts_field].max()
                                    k_df = k_df.with_columns(((pl.col(self.ts_field)-my_min)/(my_max-my_min)).alias(_tfield_))
                                    for row_i in range(len(k_df)):
                                        row = k_df[row_i]
                                        _color_ = self.rt_self.co_mgr.spectrumAbridged(row[_tfield_][0], 0.0, 1.0)
                                        _t_box_     = 0.1 + 0.8 * row[_tfield_][0]
                                        _x_  , _y_  = _xyLinkDir_(_t_box_)
                                        _xp_ , _yp_ = _xyLinkDir_(_t_box_+0.01)  # slight offset point
                                        _dx_ , _dy_ = _xp_ - _x_ , _yp_ - _y_          # slope at this location
                                        _l_         = sqrt(_dx_*_dx_ + _dy_*_dy_)
                                        _l_         = 1.0 if _l_ < 0.001 else _l_
                                        _dx_ , _dy_ = _dx_ / _l_ , _dy_ / _l_          # unitize the vector
                                        _xe_ , _ye_ = _x_ - _side_ * _dx_ * _tml_/2 + _side_ * _dy_ * _tml_, _y_ - _side_ * _dy_ * _tml_/2 - _side_ * _dx_ * _tml_
                                        svg += f'<line x1="{_x_}" y1="{_y_}" x2="{_xe_}" y2="{_ye_}" stroke="{_color_}" stroke-width="1.5" />'

                # Handle the small multiples
                if self.sm_mode == 'link' and self.sm_type is not None:
                    sm_lu = self.rt_self.createSmallMultiples(self.df, link_to_dfs, link_to_xy,
                                                              self.count_by, self.count_by_set, self.color_by, None, self.widget_id,
                                                              self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent,
                                                              self.sm_w, self.sm_h)
                    for node_str in sm_lu.keys():
                        svg += sm_lu[node_str]

            return svg

        #
        # __renderNodes__() - render the nodes
        #
        def __renderNodes__(self):
            svg = ''
            node_already_rendered = set()

            # Small multiple structures
            node_to_dfs = {}
            node_to_xy  = {}

            # Render nodes
            if self.node_size is not None and self.node_size != 'hidden':
                # Set the node size
                if   type(self.node_size) == int or type(self.node_size) == float:
                    _sz = self.node_size
                elif self.node_size == 'small':
                    _sz = 2
                elif self.node_size == 'medium':
                    _sz = 5
                elif self.node_size == 'large':
                    _sz = 8
                else: # Vary
                    _sz = 1

                # Render position context (if selected)
                if self.render_pos_context:
                    _co = self.rt_self.co_mgr.getTVColor('context','text')
                    for node_str in self.pos.keys():
                        x = self.xT(self.pos[node_str][0])
                        y = self.yT(self.pos[node_str][1])
                        if x >= -5 and x <= self.w+5 and y >= -5 and y <= self.h+5:
                            svg += f'<circle cx="{x}" cy="{y}" r="{2}" fill="{_co}" stroke="{_co}" stroke-opacity="{self.pos_context_opacity}" fill-opacity="{self.pos_context_opacity}" />'

                # Iterate over the relationships
                for rel_tuple in self.relationships:
                    # Make sure it's the right number of tuples
                    if len(rel_tuple) != 2:
                        raise Exception(f'linkNode(): relationship tuples should have two parts "{rel_tuple}"')

                    # Flatten out into the groupby array, the fm_flds array, and the to_flds array            
                    flat    = flattenTuple(rel_tuple)

                    fm_flds = flattenTuple(rel_tuple[0])
                    if type(fm_flds) != list:
                        fm_flds = [fm_flds]

                    to_flds = flattenTuple(rel_tuple[1])                
                    if type(to_flds) != list:
                        to_flds = [to_flds]

                    # Do the from and to fields separately
                    for flds_i in range(0,2):
                        if flds_i == 0:
                            flds = fm_flds
                        else:
                            flds = to_flds

                        if flds_i == 1 and fm_flds == to_flds:
                            continue
                        
                        # Iterate over the dfs
                        for _df in self.df:
                            # if the _df has all of the columns
                            if len(set(_df.columns) & set(flds)) == len(set(flds)):
                                # create the node table
                                if   self.rt_self.isPandas(_df):
                                    gb = _df.groupby(flds[0]) if len(flds) == 1 else _df.groupby(flds)
                                elif self.rt_self.isPolars(_df):
                                    gb = _df.group_by(flds[0]) if len(flds) == 1 else _df.group_by(flds)
                                else:
                                    raise Exception('RTLinkNode.__renderNodes__() - only pandas and polars supported')

                                # iterate over the nodes
                                for k,k_df in gb:
                                    node_str = self.rt_self.nodeStringAndFillPos(k, self.pos)

                                    # Prevents duplicate renderings
                                    if node_str in node_already_rendered:
                                        continue
                                    else:
                                        node_already_rendered.add(node_str)
                                    
                                    # Transform the coordinates
                                    x = self.xT(self.pos[node_str][0])
                                    y = self.yT(self.pos[node_str][1])
                                    self.node_coords[node_str] = (x,y)

                                    if self.node_shape == 'small_multiple':
                                        if k not in node_to_dfs.keys():
                                            node_to_dfs[k] = []

                                        node_to_dfs[k].append(k_df)
                                        node_to_xy[k] = (x,y)

                                        if self.track_state:
                                            _poly = Polygon([[x-self.sm_w/2,y-self.sm_h/2],
                                                             [x+self.sm_w/2,y-self.sm_h/2],
                                                             [x+self.sm_w/2,y+self.sm_h/2],
                                                             [x-self.sm_w/2,y+self.sm_h/2]])
                                            if _poly not in self.geom_to_df.keys():
                                                self.geom_to_df[_poly] = []
                                            self.geom_to_df[_poly].append(k_df)

                                    else:
                                        # Determine the color
                                        if   type(self.node_color) == dict:
                                            if node_str in self.node_color.keys():
                                                _lu_co = self.node_color[node_str]

                                                # It's a hash RGB hex string
                                                if len(_lu_co) == 7 and _lu_co.startswith('#'):
                                                    _co        = _lu_co
                                                    _co_border = _lu_co
        
                                                # The string needs to be converted at the global level
                                                else:
                                                    _co        = self.rt_self.co_mgr.getColor(_lu_co)
                                                    _co_border = _co
                                            else:
                                                _co        = self.rt_self.co_mgr.getTVColor('data','default')
                                                _co_border = self.rt_self.co_mgr.getTVColor('data','default_border')
                                        elif self.node_color == 'vary' and self.color_by is not None and self.color_by in _df.columns:
                                            _co_set = set(k_df[self.color_by])
                                            if len(_co_set) == 1:
                                                _co        = self.rt_self.co_mgr.getColor(_co_set.pop())
                                                _co_border = _co
                                            else:
                                                _co        = self.rt_self.co_mgr.getTVColor('data','default')
                                                _co_border = _co
                                        elif self.node_color is not None and self.node_color.startswith('#'):
                                            _co        = self.node_color
                                            if self.node_border_color is not None:
                                                _co_border = self.node_border_color
                                            else:
                                                _co_border = self.node_color
                                        else:
                                            _co        = self.rt_self.co_mgr.getTVColor('data','default')
                                            _co_border = self.rt_self.co_mgr.getTVColor('data','default_border')
                                                                                
                                        # Determine the size (if it varies)
                                        if self.node_size == 'vary':
                                            if self.count_by is None:
                                                _sz = self.max_node_size * len(k_df) / self.max_node_value
                                            elif self.count_by in _df.columns and self.count_by_set:
                                                _sz = self.max_node_size * len(set(k_df[self.count_by])) / self.max_node_value
                                            elif self.count_by in _df.columns:
                                                _sz = self.max_node_size * k_df[self.count_by].sum() / self.max_node_value
                                            else:
                                                _sz = 1
                                            if _sz < self.min_node_size:
                                                _sz = self.min_node_size
                                        
                                        # Determine the node shape
                                        # ... by dictionary... into either a shape string... or into an SVG string
                                        if type(self.node_shape) == dict:
                                            # Create the Node Shape Key ... complicated by tuples... // field (column) version
                                            _node_shape_key = flds
                                            if type(_node_shape_key) == list and len(_node_shape_key) == 1:
                                                _node_shape_key = _node_shape_key[0]
                                            if type(_node_shape_key) == list and len(_node_shape_key) > 1:
                                                _node_shape_key = tuple(_node_shape_key)

                                            # Retrieve the node shape key
                                            if _node_shape_key in self.node_shape.keys():
                                                _shape = self.node_shape[_node_shape_key]
                                            else:
                                                # Otherwise, see if there's a direct key lookup...
                                                if k in self.node_shape.keys():
                                                    _shape = self.node_shape[k]
                                                else:
                                                    _shape = 'ellipse'
                                                    _sz    = 5

                                        # Functional node shapes...
                                        elif callable(self.node_shape):
                                            _shape = self.node_shape(k_df, k, x, y, _sz, _co, self.node_opacity)
                                        
                                        # Just a simple node shape
                                        else:
                                            _shape = self.node_shape

                                        # Shape render...  if it's SVG, the rewrite coordinates into the right place...
                                        if _shape is not None and _shape.startswith('<svg'):
                                            _svg_w,_svg_h  = self.rt_self.__extractSVGWidthAndHeight__(_shape)
                                            svg           += self.rt_self.__overwriteSVGOriginPosition__(_shape, (x,y), _svg_w, _svg_h)
                                            _sz            = _svg_h/2

                                        # Otherwise, call the super class shape renderer...
                                        else:
                                            svg += self.rt_self.renderShape(_shape, x, y, _sz, _co, _co_border, self.node_opacity)

                                        # Track state
                                        if self.track_state:
                                            _poly = Polygon([[x-_sz,y-_sz],
                                                             [x+_sz,y-_sz],
                                                             [x+_sz,y+_sz],
                                                             [x-_sz,y+_sz]])
                                            if _poly not in self.geom_to_df.keys():
                                                self.geom_to_df[_poly] = []
                                            self.geom_to_df[_poly].append(k_df)

                                        # Prepare the label
                                        k_str = node_str

                                        # Check for if the conditions are met to render the label
                                        if self.draw_labels and self.node_shape != 'small_multiple' and ((len(self.label_only) == 0) or (k_str in self.label_only)):
                                            if len(k_str) > 16:
                                                k_str = k_str[:16] + '...'

                                            if self.node_labels_only == False:
                                                svg_text = self.rt_self.svgText(str(k_str), x, y+_sz+self.txt_h, self.txt_h, anchor='middle')                                            
                                                self.defer_render.append(svg_text) # Defer render

                                            if self.node_labels is not None and k_str in self.node_labels.keys():
                                                if self.node_labels_only:
                                                    y_label = y + _sz + 1*self.txt_h
                                                else:
                                                    y_label = y + _sz + 2*self.txt_h
                                                _strs_  = self.node_labels[k_str]
                                                if type(_strs_) == str:
                                                    svg_text = self.rt_self.svgText(_strs_, x, y_label, self.txt_h, anchor='middle')
                                                    self.defer_render.append(svg_text) # Defer render
                                                else:
                                                    for _str_ in _strs_:
                                                        svg_text = self.rt_self.svgText(_str_, x, y_label, self.txt_h, anchor='middle')
                                                        self.defer_render.append(svg_text) # Defer render
                                                        y_label += self.txt_h
                                                

            # Handle the small multiples
            if self.node_shape == 'small_multiple':
                sm_lu = self.rt_self.createSmallMultiples(self.df, node_to_dfs, node_to_xy,
                                                          self.count_by, self.count_by_set, self.color_by, None, self.widget_id,
                                                          self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent,
                                                          self.sm_w, self.sm_h)
                                                
                for k in sm_lu.keys():
                    _small_multiple_svg_ = sm_lu[k]
                    if self.node_opacity != 1.0:
                        _svg_index_ = _small_multiple_svg_.index('<svg')
                        _small_multiple_svg_ = _small_multiple_svg_[:(_svg_index_+4)] + \
                                               f' opacity="{self.node_opacity}" '     + \
                                               _small_multiple_svg_[(_svg_index_+4):]
                    svg += _small_multiple_svg_

                    # Copy of the draw labels portion a few lines up...
                    if self.draw_labels:
                        node_str = self.rt_self.nodeStringAndFillPos(k)
                        if len(node_str) > 16:
                            node_str = node_str[:16] + '...'
                        if k not in node_to_xy.keys(): # polars hack 2023-02-09
                            k = k[0]
                        x, y = node_to_xy[k]
                        svg_text = self.rt_self.svgText(node_str, x, y+self.sm_h/2+self.txt_h, self.txt_h, anchor='middle')
                        self.defer_render.append(svg_text)

                # Possible that some nodes may not have been rendered due to the nature of the multi-dataframe structure
                if self.draw_labels:
                    for k in node_to_xy.keys():
                        if k not in sm_lu.keys():
                            node_str = self.rt_self.nodeStringAndFillPos(k)
                            if node_str not in node_already_rendered:
                                node_already_rendered.add(node_str)
                                if len(node_str) > 16:
                                    node_str = node_str[:16] + '...'
                                if k not in node_to_xy.keys(): # polars hack 2023-02-09
                                    k = k[0]
                                x, y = node_to_xy[k]
                                svg_text = self.rt_self.svgText(node_str, x, y+self.txt_h/2, self.txt_h, anchor='middle')
                                self.defer_render.append(svg_text)
            return svg

        #
        # __renderBackgroundShapes__() - render background shapes
        # - mostly a copy of the xy implementation
        #
        def __renderBackgroundShapes__(self):
            _svg_ = ''
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
                    _svg_ += _shape_svg
                    _bg_shape_labels.append(_label_svg) # Defer render

                # Render the labels
                for _label_svg in _bg_shape_labels:
                    _svg_ += _label_svg
            
            return _svg_

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
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
                self.setViewWindow((self.view_window[0]+dwx, self.view_window[1]+dwy, self.view_window[2]+dwx, self.view_window[3].dwy))
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
        def getViewWindow(self):
            return self.view_window

        #
        # renderSVG() - render as SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state:
                self.geom_to_df = {}

            # Determine geometry
            if self.view_window is None:
                self.__calculateGeometry__()
                self.view_window = self.view_window_orig = (self.wx0, self.wy0, self.wx1, self.wy1)
            else:
                self.wx0, self.wy0, self.wx1, self.wy1 = self.view_window
                
            # Coordinate transform lambdas (and inverse lambdas)
            self.xT     = lambda __wx__:          self.w * (__wx__ - self.wx0)/(self.wx1-self.wx0)
            self.yT     = lambda __wy__: self.h - self.h * (__wy__ - self.wy0)/(self.wy1-self.wy0)
            self.xT_inv = lambda __sx__: self.wx0 + ((__sx__ * (self.wx1 - self.wx0))/self.w)
            self.yT_inv = lambda __sy__: self.wy0 + ((self.h - __sy__) * (self.wy1 - self.wy0))/self.h

            # 2023-12-22 19:50
            #self.xT     = lambda __x__: self.x_ins + (self.w - 2*self.x_ins) * (__x__ - self.wx0)/(self.wx1-self.wx0)
            #self.yT     = lambda __y__: (self.h + self.y_ins) - (self.h - 2*self.y_ins) * (__y__ - self.wy0)/(self.wy1-self.wy0)
            #self.xT_inv = lambda __sx__: self.wx0 + (self.wx1 - self.wx0) * (__sx__ - self.x_ins)/(self.w - 2*self.x_ins)
            #self.yT_inv = lambda __sy__: self.wy0 + (self.wy1 - self.wy0) * ((self.h + self.y_ins) - __sy__)/(self.h - 2*self.y_ins)

            # Start the SVG Frame
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'

            # Elements to render after nodes (labels, in this case)
            self.defer_render = []

            # Render background shapes, convex hulls, links, and then nodes
            svg += self.__renderBackgroundShapes__()
            svg += self.__renderConvexHull__()
            svg += self.__renderLinks__()
            svg += self.__renderNodes__()

            # Defer render
            for x in self.defer_render:
                svg += x

            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill-opacity="0.0" stroke="{border_color}" />'

            svg += '</svg>'
            self.last_render = svg
            return svg

        #
        # Determine which dataframe geometries overlap with a specific
        #
        def overlappingDataFrames(self, to_intersect):
            _dfs = []
            for _poly in self.geom_to_df.keys():
                if _poly.intersects(to_intersect):
                    _dfs.extend(self.geom_to_df[_poly]) # <== SLIGHTLY DIFFERENT THAN ALL OF THE OTHERS...
            if len(_dfs) > 0:
                return self.rt_self.concatDataFrames(_dfs)
            else:
                return None

# ---------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------

#
# Flatten a Tuple into a one dimensional list
#
def flattenTuple(x):
    _list = list()
    if type(x) == str:
        return x
    else:
        for i in range(0,len(x)):
            _recurse = flattenTuple(x[i])
            if type(_recurse) == str:
                _list.append(_recurse)
            else:
                _list.extend(_recurse)
    return _list
