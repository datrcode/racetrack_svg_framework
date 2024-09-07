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

from shapely.geometry import Point,Polygon,LineString

from .rt_component import RTComponent
from .rt_entity_position import RTEntityPosition

__name__ = 'rt_linknode_mixin'

#
# Abstraction for LinkNode
#
class RTLinkNodeMixin(object):
    #
    # concatDisparateDataFrames() - concatenate disparate dataframes together into a single dataframe.
    # - replaces the multiple dataframes used before...
    #
    def concatDisparateDataFrames(self, dfs):
        if   self.isPandas(dfs[0]): return pd.concat(dfs)
        elif self.isPolars(dfs[0]): return pl.concat(dfs, how='diagonal')
        else:                       raise Exception('concatDisparateDataFrame() - only supports pandas and polars')

    #
    # filterDataFrameByGraph() - keep only the rows that are in the graph
    #
    def filterDataFrameByGraph(self, df, relationships, g):
        # Fix up the relationships so that it's one column to one column
        new_relationships = []
        for i in range(len(relationships)):
            _relationship_ = relationships[i]
            _fm_ , _to_ = _relationship_[0], _relationship_[1]
            if type(_relationship_[0]) == tuple:
                if len(_relationship_[0]) == 1: _fm_ = _relationship_[0][0]
                else:
                    _fm_ = f'__fm{i}__'
                    df = self.createConcatColumn(df, _relationship_[0], _fm_)
            if type(_relationship_[1]) == tuple:
                if len(_relationship_[1]) == 1: _to_ = _relationship_[1][0]
                else:
                    _to_ = f'__to{i}__'
                    df = self.createConcatColumn(df, _relationship_[1], _to_)                
            new_relationships.append((_fm_, _to_))

        # Set of all the edges in the graph
        edges = set()
        for _node_ in g:
            for _nbor_ in g.neighbors(_node_):
                edges.add((_node_, _nbor_))

        # For each relationship, check for the existence of that edge
        _dfs_ = []
        for _relationship_ in new_relationships:
            if   self.isPandas(df): gb = df.groupby(list(_relationship_))
            elif self.isPolars(df): gb = df.group_by(_relationship_)
            else: raise Exception('filterDataFrameByGraph() - only supports pandas and polars')
            
            for k, k_df in gb:
                if k in edges: 
                    _dfs_.append(k_df)
        
        # Concatenate them together # may have duplicates...
        if   len(_dfs_)   == 0:
            if   self.isPandas(df): return pd.DataFrame(columns=df.columns)
            elif self.isPolars(df): return pl.DataFrame(schema=df.schema)
            else:                   raise Exception('filterDataFrameByGraph() - only supports pandas and polars [empty dataframe]')
        elif self.isPandas(df): return pd.concat(_dfs_)
        elif self.isPolars(df): return pl.concat(_dfs_)
        else:                   raise Exception('filterDataFrameByGraph() - only supports pandas and polars')
        
    #
    # filterDataFrameByGraph() - keep only the rows that are in the graph
    #
    def REALLYSLOW__filterDataFrameByGraph(self, df, relationships, g):
        if len(relationships) != 1: raise Exception('graphKeepRowsInGraph() - only supports single relationship')
        if self.isPandas(df):
            return self.__filterDataFrameByGraph_pandas__(df, relationships, g)
        elif self.isPolars(df):
            return self.__filterDataFrameByGraph_polars__(df, relationships, g)
        else:
            raise Exception('graphKeepRowsInGraph() - only supports pandas and polars')

    # __filterDataFrameByGraph_pandas__()
    def __filterDataFrameByGraph_pandas__(self, df, relationships, g):
        q_strs = []
        for _node_ in g:
            for _nbor_ in g.neighbors(_node_):
                q0 = '"' if "'" in _node_ else "'"
                q1 = '"' if "'" in _nbor_ else "'"
                q_strs.append(f'({relationships[0][0]} == {q0}{_node_}{q0} and {relationships[0][1]} == {q1}{_nbor_}{q1})')
        return df.query(' or '.join(q_strs))

    # __filterDataFrameByGraph_polars__()
    def __filterDataFrameByGraph_polars__(self, df, relationships, g):
        _filter_ = None
        for _node_ in g:
            for _nbor_ in g.neighbors(_node_):
                _expr_   = ((pl.col(relationships[0][0]) == _node_) & (pl.col(relationships[0][1]) == _nbor_))
                _filter_ = _expr_ if _filter_ is None else _filter_.or_(_expr_)
        return df.filter(_filter_)
                
    #
    # graphDictToDataFrame() - converts a dictionary into a dataframe.
    # - d[fm][to] = ct
    # - fields will be "fm", "to", and "ct"
    #
    def graphDictToDataFrame(self, d):
        fms, tos, cts = [], [], []
        for fm in d.keys():
            fm_str = fm if (type(fm) == int or type(fm) == float) else str(fm)
            if   type(d[fm]) == set:
                for to in d[fm]:
                    to_str = to if (type(to) == int or type(to) == float) else str(to)
                    fms.append(fm_str), tos.append(to_str), cts.append(1)
            elif type(d[fm]) == dict:
                for to in d[fm].keys():
                    to_str = to if (type(to) == int or type(to) == float) else str(to)
                    fms.append(fm_str), tos.append(to_str), cts.append(d[fm][to])
            elif type(d[fm]) == list:
                for i in range(len(d[fm])):
                    to_str = str(d[fm][i])
                    fms.append(fm_str), tos.append(to_str), cts.append(1)
            else:
                raise Exception('RTLinkNode.graphDictToDataFrame() - only supports dictionary or set keys')
        return pd.DataFrame({'fm':fms,'to':tos,'ct':cts})

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
    # calculateNodeInformation() - calculate information about the nodes from the specified dataframe
    # ... needed to know the overall info about the nodes for rendering
    # ... mostly a copy of the node render loop... should probably be refactored
    #
    def calculateNodeInformation(self, df, relationships, pos, count_by, count_by_set):
        # Boundary
        wx0 =  math.inf
        wy0 =  math.inf
        wx1 = -math.inf
        wy1 = -math.inf

        # Maximum node value
        max_node_value = 0

        # Nodes found
        nodes_in_df = set()

        # Iterate over the relationships
        for rel_tuple in relationships:
            # Make sure it's the right number of tuples
            if len(rel_tuple) != 2:
                raise Exception(f'linkNode(): relationship tuples should have two parts "{rel_tuple}"')

            # Flatten out into the groupby array, the fm_flds array, and the to_flds array            
            fm_flds = (rel_tuple[0])
            to_flds = (rel_tuple[1])                

            # Do the from and to fields separately
            for flds_i in range(0,2):
                flds = fm_flds if flds_i == 0 else to_flds

                # create the edge table
                if len(flds) == 1:
                    if   self.isPandas(df): gb = df.groupby(flds[0])
                    elif self.isPolars(df): gb = df.group_by(flds[0])
                else:
                    if   self.isPandas(df): gb = df.groupby(flds)
                    elif self.isPolars(df): gb = df.group_by(flds)

                # iterate over the edges
                for k,k_df in gb:
                    node_str = self.nodeStringAndFillPos(k, pos)
                    nodes_in_df.add(node_str)

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
                    elif count_by in df.columns and count_by_set:
                        set_size = len(k_df[count_by])
                        if max_node_value < set_size:
                            max_node_value = set_size
                    elif count_by in df.columns:
                        summation = k_df[count_by].sum()
                        if max_node_value < summation:
                            max_node_value = summation

        # Make sure the max node value is not zero
        if max_node_value == 0: max_node_value = 1

        return max_node_value, wx0, wy0, wx1, wy1, nodes_in_df

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
                 draw_performance      = True,   # draw performance information (not implemented yet)
                 draw_labels           = False,  # draw labels flag # not implemented yet
                 draw_border           = True):  # draw a border around the graph
        """Implementation of a link node diagram in SVG.

        Required Parameters
        -------------------

        df : pandas.DataFrame | polars.DataFrame
            Dataframe to render.

        relationships : list[tuple]
            The list of relationships to be drawn.  Examples include:
            [('fm','to')]                # single relationship
            [('fm','to'), ('src','dst')] # multiple relationships
            [(('fm','fm_sub'),'to')]     # multi-field from relationship with single field to relationship

        Useful Parameters
        -----------------

        pos : dict[key:tuple(float,float)]
            The positions for the nodes.  Examples include:
            {'node1':(0.0,0.0), 'node2':(0.0,100.0)}
            Note:  any nodes not in the position list will be assigned a random position.
            Note:  for link(), the coordinates must be float values.

        color_by : str | None
            The field to be used to color the nodes or links.  node_color and/or link_color must be set to "vary" to use this.

        count_by : str | None
            The field to be used to count the nodes or links.  node_size and/or link_size must be set to "vary" to use this.

        Node Parameters
        ---------------

        node_color : None | 'vary' | hex-color-string | dict[str:hex-color-string]

        node_border_color : None | hex-color-string

        node_size : None | str | int | float
            'small', 'medium' (default), 'large', 'vary', 'hidden' | None

        node_shape : None | str | dict[str:str] | function
            Shapes to use for nodes.
        
        node_opacity : float

        node_labels : None | dict[str:str]
            Alternative labels to use for nodes.

        label_only : set[str]
            If set and not None, only the specified nodes will be labeled.

        node_label_max_w : int | float
            Maximum label width for node labels.

        node_size_max: int | float
            Maximum size of a node for variable size nodes.
        
        node_size_min: int | float
            Minimum size of a node for variable size nodes.
        
        selected_entities : None | set[str] # should be deprecated

        Link Parameters
        ---------------

        link_color : None | str | hex-color-string
        
        link_size : None | str
            'nil', 'small' (default), 'medium', 'large', 'vary', 'hidden' | None

        link_opacity : float

        link_shape: str
            'line' (default), 'curve', 'arrow'

        link_arrow: bool

        link_arrow_style: str
            'kite', 'kite_v2', 'kite_v3' (default)

        link_arrow_length: int | float

        link_dash: None | str | dict[relationship-tuple:str] | function
            SVG dash patterns for links

        link_max_curvature_px: int | float

        link_parallel_perc: float

        link_ortho_perc: float

        link_size_max: int | float
            Maximum size of a link for variable size links.

        link_size_min: int | float
            Minimum size of a link for variable size links.

        timing_marks: bool
            Render timing marks on the links
        
        ts_field: str | None
            The field to use for timing marks

        timing_mark_length: int | float
            Size of the timing mark in pixels

        Standard Parameters
        -------------------

        track_state : bool
            Track state for interactive filtering operations
        
        x_view, y_view : int
            The x and y offset for the SVG view

        w, h : int
            The width and height of the SVG frame

        x_ins, y_ins : int
            The x and y spacing to inset the visualization

        txt_h : int
            The height of the text for node and link labels

        draw_performance : bool
            Draw the performance information

        draw_labels : bool
            Draw the node labels (link labels are enabled by "link_labels" parameter)

        draw_border : bool
            Draw a border around the visualization

        Background Shapes
        -----------------

        bg_shape_lu : None | dict[str:list[tuple(float,float)]] | dict[str:str]
            The lookup table for background shapes.
            - key is the shape name
            - value is a list of (x,y) tuples
            - value is an svg-compliant path string

        bg_shape_label_color : None | 'vary' | hex-color-string | dict[str:hex-color-string]
            - color to use for background shape labels

        bg_shape_fill : None | 'vary' | hex-color-string | dict[str:hex-color-string]
            The background shape fill color

        bg_shape_stroke : None | 'default' | hex-color-string | dict[str:hex-color-string]
            The background shape stroke color

        bg_shape_stroke_w : None | int | float | dict[str:int | float]
            The background shape stroke width
        
        Convey Hulls
        ------------

        convex_hull_lu : dict[str:str] | dict[str:list[str]]
            The keys are the regex strings to match against node names.  The values
            are the names to use for the convex hulls.

            OR

            The keys are the convex hull names and the values are the nodes names part of
            the convex hull.

        convex_hull_opacity : float

        convex_hull_labels : bool

        convex_hull_stroke_width : None | int | float

        Small Multiples
        ---------------

        """
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTLinkNode(self, **_params_)

    #
    # __minAndMaxLinkSize__()
    # ... copy of the next method but only determines the min and max link size
    #
    def __minAndMaxLinkSize__(self, df, relationships, count_by=None):
        _min_,_max_ = None,None
        # Check the count_by column across all the df's...  if any of them
        # don't work.. then it's count_by_set
        count_by_set = False
        if count_by is not None:
            if self.fieldIsArithmetic(df, count_by) == False:
                count_by_set = True
        # Iterate over the relationships
        for rel_tuple in relationships:
            # Flatten out into the groupby array, the fm_flds array, and the to_flds array            
            fm_flds = (rel_tuple[0])
            to_flds = (rel_tuple[1])                
            if self.isPandas(df):
                gb = df.groupby(list(rel_tuple[:2]))
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
            elif self.isPolars(df):
                counter = self.polarsCounter(df, list(rel_tuple[:2]), count_by, count_by_set)
                _min_ = counter['__count__'].min()
                _max_ = counter['__count__'].max()
            else:
                raise Exception('RTLinkNode.minAndMaxLinkSize() - only pandas and polars supported')
        if _min_ == _max_:
            _max_ = _min_ + 1
        return _min_,_max_

    #
    # createNetworkXGraph() - construct networkx graph as this class would construct the graph
    # - relationship examples:
    #   [('fm','to')]
    #   [('fm','to), ('src','dst')]
    #   [('fm',('to1','to2'))]
    #   [('subj','obj','verb')]
    #    
    def createNetworkXGraph(self,
                            df,                        # dataframe for graph creation
                            relationships,             # list of tuple pairs... pairs can be single strings or tuples of strings
                            use_digraph     = False,   # use directed graph
                            count_by        = None,    # edge weight field
                            count_by_set    = False):  # count this via set operation
        # Determine the count by
        if count_by is not None and count_by_set:
            if self.fieldIsArithmetic(df, count_by) == False:
                count_by_set = True

        # Create the return graph structure
        nx_g = nx.DiGraph() if use_digraph else nx.Graph()

        # Create concatenated fields for the tuple nodes
        # ... may be inefficient if there are multiples of the same tuple in different edges...
        df = self.copyDataFrame(df)
        new_relationships, i = [], 0
        for _edge_ in relationships:
            _fm_ = _edge_[0]
            _to_ = _edge_[1]
            if type(_fm_) == tuple or type(_to_) == tuple:
                new_fm, new_to = _fm_, _to_
                if type(_fm_) == tuple:
                    new_fm = f'__fm{i}__'
                    df = self.createConcatColumn(df, _fm_, new_fm)

                if type(_to_) == tuple:
                    new_to = f'__to{i}__'
                    df = self.createConcatColumn(df, _to_, new_to)

                if   len(_edge_) == 2:
                    new_relationships.append((new_fm, new_to))
                elif len(_edge_) == 3:
                    new_relationships.append((new_fm, new_to, _edge_[2]))
                else:
                    raise Exception(f'createNetworkXGraph(): relationship tuples should have two or three parts "{_edge_}"')
            else:
                if   len(_edge_) == 2:
                    new_relationships.append((_fm_, _to_))
                elif len(_edge_) == 3:
                    new_relationships.append((_fm_, _to_, _edge_[2]))
                else:
                    raise Exception(f'createNetworkXGraph(): relationship tuples should have two or three parts "{_edge_}"')
            i += 1

        # Iterate over the relationships
        for rel_tuple in new_relationships:
            if self.isPandas(df):
                if count_by is None: # count_by_set not implemented...
                    gb = df.groupby(list(rel_tuple)).size()
                elif count_by_set:
                    gb = df.groupby(list(rel_tuple))[count_by].nunique()
                else:
                    gb = df.groupby(list(rel_tuple))[count_by].sum()
                for i in range(0,len(gb)):
                    k      = gb.index[i]
                    k_fm   = k[0]
                    k_to   = k[1]
                    params = {}
                    if len(rel_tuple) == 3:
                        params[rel_tuple[2]] = k[2]
                    nx_g.add_edge(k_fm,k_to,weight=gb.iloc[i], **params)
            elif self.isPolars(df):
                df_filtered = self.polarsFilterColumnsWithNaNs(df, self.flattenTuple(rel_tuple))
                counter = self.polarsCounter(df_filtered, list(rel_tuple), count_by, count_by_set)
                for i in range(len(counter)):
                    _row_   = counter[i]
                    params = {}
                    if len(rel_tuple) == 3:
                        params[rel_tuple[2]] = _row_[rel_tuple[2]][0]
                    nx_g.add_edge(_row_[rel_tuple[0]][0],_row_[rel_tuple[1]][0],weight=_row_['__count__'][0], **params)
            else:
                raise Exception('RTLinkNode.createNetworkXGraph() - only pandas and polars supported')

        return nx_g
    
    #
    # createNetworkXGraph()
    #
    # Use the same construction technique as linkNode but make a networkx graph instead.
    #    
    def createNetworkXGraph_OLD_20240506(self,
                            df,                       # dataframe for graph creation
                            relationships,            # list of tuple pairs... pairs can be single strings or tuples of strings
                            use_digraph     = False,  # use directed graph
                            count_by        = None):  # edge weight field
        # Check the count_by column across all the df's...  if any of them
        # don't work.. then it's count_by_set
        count_by_set = False
        if count_by is not None:
            if self.fieldIsArithmetic(df, count_by) == False:
                count_by_set = True

        # Create the return graph structure
        nx_g = nx.DiGraph() if use_digraph else nx.Graph()

        # Iterate over the relationships
        for rel_tuple in relationships:
            # Make sure it's the right number of tuples
            if len(rel_tuple) != 2:
                raise Exception(f'createNetworkXGraph(): relationship tuples should have two parts "{rel_tuple}"')

            fm_flds, to_flds = (rel_tuple[0]), (rel_tuple[1])

            def stringify(_list_):
                if type(_list_) == str:
                    return _list_
                _str_ = str(_list_[0])
                for _x_ in _list_[1:]:
                    _str_ += '|' + str(_x_)
                return _str_

            if self.isPandas(df):
                if count_by is None or count_by_set: # count_by_set not implemented...
                    gb = df.groupby(list(rel_tuple[:2])).size()
                else:
                    gb = df.groupby(list(rel_tuple[:2]))[count_by].sum()
                for i in range(0,len(gb)):
                    k      = gb.index[i]
                    k_fm   = k[0]
                    k_to   = k[1]
                    _fm_   = stringify(k_fm)
                    _to_   = stringify(k_to)
                    nx_g.add_edge(_fm_,_to_,weight=gb.iloc[i])
            elif self.isPolars(df):
                counter = self.polarsCounter(df, list(rel_tuple[:2]), count_by, count_by_set)
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
            if self.link_shape is None: self.link_shape = 'line'
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

                    if len(_edge) == 2:
                        self.relationships.append((new_fm, new_to))
                    elif len(_edge) == 3:
                        self.relationships.append((new_fm, new_to, _edge[2]))
                    else:
                        raise Exception(f'linkNode(): relationship tuples should have two or three parts "{_edge}"')
                else:
                    if len(_edge) == 2:
                        self.relationships.append((_fm_, _to_))
                    elif len(_edge) == 3:
                        self.relationships.append((_fm_, _to_, _edge[2]))
                    else:
                        raise Exception(f'linkNode(): relationship tuples should have two or three parts "{_edge}"')
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
        def __calculateGeometry__(self, for_entities=None):
            # Calculate world coordinates
            self.wx0 =  math.inf
            self.wy0 =  math.inf
            self.wx1 = -math.inf
            self.wy1 = -math.inf

            # And possibly the max node size
            self.max_node_value = 1

            if   for_entities is not None and len(for_entities) > 0:
                for k in for_entities:
                    v = self.pos[k]
                    self.wx0 = min(v[0], self.wx0)
                    self.wy0 = min(v[1], self.wy0)
                    self.wx1 = max(v[0], self.wx1)
                    self.wy1 = max(v[1], self.wy1)
            elif self.use_pos_for_bounds:
                for k in self.pos.keys():
                    v = self.pos[k]
                    self.wx0 = min(v[0], self.wx0)
                    self.wy0 = min(v[1], self.wy0)
                    self.wx1 = max(v[0], self.wx1)
                    self.wy1 = max(v[1], self.wy1)
                if self.node_size == 'vary':
                    self.max_node_value,ignore0,ignore1,ignore2,ignore3,nodes_in_df = self.rt_self.calculateNodeInformation(self.df, self.relationships, self.pos, self.count_by, self.count_by_set)
            else:
                self.max_node_value,self.wx0,self.wy0,self.wx1,self.wy1,nodes_in_df = self.rt_self.calculateNodeInformation(self.df, self.relationships, self.pos, self.count_by, self.count_by_set)

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
            
            return self.wx0, self.wy0, self.wx1, self.wy1

        #
        # labelOnly() - set the label only set
        # - this controls which labels will be shown
        #
        def labelOnly(self,  label_set):   self.label_only  = label_set

        #
        # drawLabels() - set the draw labels flag
        #
        def drawLabels(self, draw_labels): self.draw_labels = draw_labels

        #
        # __renderConvexHull__() - render the convex hull
        # (copied directly into the link() code -- any mods should be replicated there)
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
        # __renderLinks__() - return links
        #
        def __renderLinks__(self):
            # Render links
            svg          = []
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
                        if self.rt_self.fieldIsArithmetic(self.df, self.count_by) == False:
                            count_by_set = True
                    _sz_min, _sz_max = self.rt_self.__minAndMaxLinkSize__(self.df, self.relationships, self.count_by)
                    _sz = None

                # Iterate over the relationships
                for rel_tuple in self.relationships:
                    # Make sure it's the right number of tuples
                    if len(rel_tuple) < 2 or len(rel_tuple) > 3:
                        raise Exception(f'linkNode(): relationship tuples should have two or three parts "{rel_tuple}"')

                    fm_flds, to_flds = [rel_tuple[0]], [rel_tuple[1]]

                    if   self.rt_self.isPandas(self.df):
                        gb = self.df.groupby(list(rel_tuple[:2]))
                    elif self.rt_self.isPolars(self.df):
                        df_filtered = self.rt_self.polarsFilterColumnsWithNaNs(self.df, self.rt_self.flattenTuple(rel_tuple)[:2])
                        gb = df_filtered.group_by(list(rel_tuple[:2]))
                    else:
                        raise Exception('RTLinkNodeMixin.__renderLinks__() - only pandas and polars supported')

                    if self.rt_self.isPandas(self.df):
                        if self.count_by is None: 
                            gb_sz = gb.size()
                        elif count_by_set:
                            gb_sz = gb[self.count_by].nunique()
                        else:
                            gb_sz = gb[self.count_by].sum()
                    else:
                        counter = self.rt_self.polarsCounter(df_filtered, list(rel_tuple[:2]), self.count_by, self.count_by_set)

                    gb_sz_i = 0
                    for k,k_df in gb:
                        if self.rt_self.isPandas(self.df):
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
                        l  = self.rt_self.segmentLength(((x1,y1),(x2,y2)))

                        # Adjust the coordinates based on the shape and size information information
                        if self.node_size is not None and self.node_size != 'hidden' and (self.link_arrow == True or self.link_shape == 'arrow'):
                            node_sz = self.__nodeSize__()
                            if node_sz > 1 and l > 5+2*node_sz: # node has to be larger than 1... and the distance between the two also needs to be large
                                if   self.node_shape is None:
                                    shape1 = 'circle'
                                elif type(self.node_shape) == str:
                                    shape1 = self.node_shape
                                elif type(self.node_shape) == dict and fm_str in self.node_shape.keys():
                                    shape1 = self.node_shape[fm_str]
                                else:
                                    shape1 = None

                                x1_orig, y1_orig = x1, y1
                                if shape1 is not None:
                                    x1, y1 = self.rt_self.shapeAttachmentPoint(shape1, x1, y1, node_sz, x2, y2)

                                if   self.node_shape is None:
                                    shape2 = 'circle'
                                elif type(self.node_shape) == str:
                                    shape2 = self.node_shape
                                elif type(self.node_shape) == dict and to_str in self.node_shape.keys():
                                    shape2 = self.node_shape[to_str]
                                else:
                                    shape2 = None

                                if shape2 is not None:
                                    x2, y2 = self.rt_self.shapeAttachmentPoint(shape2, x2, y2, node_sz, x1_orig, y1_orig)

                        # Determine the size
                        if _sz is None:
                            _this_sz = self.link_size_min + (self.link_size_max - self.link_size_min) * (_weight_ - _sz_min) / (_sz_max - _sz_min)
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

                        # Vector info
                        dx, dy = x2 - x1, y2 - y1
                        l = sqrt((dx*dx)+(dy*dy))
                        l = 1 if l <= 0.01 else l
                        dx,  dy  =  dx/l,  dy/l
                        pdx, pdy =  dy,   -dx

                        # Determine the color
                        if   self.link_color == 'vary' and self.color_by is not None and self.color_by in self.df.columns:
                            _co_set = set(k_df[self.color_by])
                            if len(_co_set) == 1:
                                _co = self.rt_self.co_mgr.getColor(_co_set.pop())
                            else:
                                _co = self.rt_self.co_mgr.getTVColor('data','default')
                        elif self.link_color is not None and self.link_color.startswith('#'):
                            _co = self.link_color
                        else:
                            _co = self.rt_self.co_mgr.getTVColor('data','default')

                        # Draw the link labels (third part of the relationship tuple or the color_by value)
                        if self.link_labels and ((len(rel_tuple) == 3) or (self.color_by is not None)):
                            _label_field_ = rel_tuple[2] if len(rel_tuple) == 3 else self.color_by
                            _label_set_   = set(k_df[_label_field_])
                            _link_str = _label_set_.pop() if len(_label_set_) == 1 else '*'
                            # no label set, str is in the label set, or set overlaps with the label set
                            if (len(self.label_only) == 0)     or \
                                (_link_str in self.label_only) or \
                                (_link_str == '*' and len(_label_set_.intersection(self.label_only)) > 0):                                
                                _l_shorter    = (l-10) if l > 15 else l
                                _cropped      = self.rt_self.cropText(_link_str, self.txt_h, _l_shorter)
                                _label_color_ = _co if _label_field_ == self.color_by else self.rt_self.co_mgr.getTVColor('label','defaultfg')
                                _label_svg_   = self.rt_self.svgLabelOnLine((x1,y1,x2,y2), _cropped, _label_color_, 2+_this_sz/2, self.txt_h)
                                svg.append(_label_svg_)

                        # Capture the state
                        if self.track_state:
                            _line = LineString([[x1,y1],[x2,y2]])
                            if _line not in self.geom_to_df.keys():
                                self.geom_to_df[_line] = []
                            self.geom_to_df[_line].append(k_df)
                                                    
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
                            svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" ')
                            svg.append(f'stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" {stroke_dash} />')

                            def _xyLink_(t):
                                return x1+(x2-x1)*t, y1+(y2-y1)*t
                            if fm_str < to_str:
                                _xyLinkDir_ = _xyLink_
                            else:
                                def _xyLinkDir_(t):
                                    return x2+(x1-x2)*t, y2+(y1-y2)*t

                            if self.link_arrow:
                                x3 = x2 - dx*self.link_arrow_length - dy*3*self.link_arrow_length/8
                                y3 = y2 - dy*self.link_arrow_length + dx*3*self.link_arrow_length/8
                                x4 = x2 - dx*self.link_arrow_length + dy*3*self.link_arrow_length/8
                                y4 = y2 - dy*self.link_arrow_length - dx*3*self.link_arrow_length/8

                                svg.append(f'<path d="M {x3} {y3} L {x2} {y2} L {x4} {y4}" ')
                                svg.append(f'fill-opacity="0.0" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" />')
                        elif self.link_shape == 'arrow':
                            x3 = x2 - dx*2*self.link_arrow_length - dy*2*self.link_arrow_length/4
                            y3 = y2 - dy*2*self.link_arrow_length + dx*2*self.link_arrow_length/4
                            x4 = x2 - dx*2*self.link_arrow_length + dy*2*self.link_arrow_length/4
                            y4 = y2 - dy*2*self.link_arrow_length - dx*2*self.link_arrow_length/4

                            x5 = x2 - dx*2*self.link_arrow_length - dy*1.8*self.link_arrow_length/4
                            y5 = y2 - dy*2*self.link_arrow_length + dx*1.8*self.link_arrow_length/4
                            x6 = x2 - dx*2*self.link_arrow_length + dy*1.8*self.link_arrow_length/4
                            y6 = y2 - dy*2*self.link_arrow_length - dx*1.8*self.link_arrow_length/4

                            if   self.link_arrow_style ==  'kite':
                                svg.append(f'<path d="M {x1} {y1} L {x3} {y3} L {x2} {y2} L {x4} {y4} Z" ')
                            elif self.link_arrow_style ==  'kite_v2': 
                                svg.append(f'<path d="M {x1} {y1} L {x3} {y3} L {x2} {y2} L {x5} {y5} L {x6} {y6} L {x2} {y2} L {x4} {y4} Z" ') # arrow head is not filled
                            elif self.link_arrow_style ==  'kite_v3':
                                svg.append(f'<path d="M {x1} {y1} L {x6} {y6} L {x5} {y5} L {x1} {y1} L {x3} {y3} L {x2} {y2} L {x4} {y4} Z" ') # arrow body is not filled
                            else:
                                raise Exception('link_arrow_style must be "kite" or "kite_v2" or "kite_v3"')

                            svg.append(f'fill-opacity="{self.link_opacity}" fill="{_co}" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" />')
                        elif self.link_shape == 'curve':
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
                                svg.append(f'<path d="M {x1} {y1} C {x1p} {y1p} {x2p} {y2p} {x2} {y2} M {x3} {y3} L {x2} {y2} L {x4} {y4}" ')
                                svg.append(f'fill-opacity="0.0" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" {stroke_dash} />')
                            else:
                                svg.append(f'<path d="M {x1} {y1} C {x1p} {y1p} {x2p} {y2p} {x2} {y2}" ')
                                svg.append(f'fill-opacity="0.0" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" {stroke_dash} />')
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
                            k_df[_tfield_] = (k_df[self.ts_field] - self.df[self.ts_field].min()) / (self.df[self.ts_field].max() - self.df[self.ts_field].min())
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
                                svg.append(f'<line x1="{_x_}" y1="{_y_}" x2="{_xe_}" y2="{_ye_}" stroke="{_color_}" stroke-width="1.5" />')
                        elif self.timing_marks and self.ts_field is not None and self.ts_field in k_df.columns and self.rt_self.isPolars(k_df):
                            _tfield_, _tml_ = '_linknode_tms_', self.timing_mark_length
                            _side_ = 1.0 if fm_str < to_str else -1.0
                            my_min, my_max = self.df[self.ts_field].min(), self.df[self.ts_field].max()
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
                                svg.append(f'<line x1="{_x_}" y1="{_y_}" x2="{_xe_}" y2="{_ye_}" stroke="{_color_}" stroke-width="1.5" />')

                # Handle the small multiples
                if self.sm_mode == 'link' and self.sm_type is not None:
                    sm_lu = self.rt_self.createSmallMultiples(self.df, link_to_dfs, link_to_xy,
                                                              self.count_by, self.count_by_set, self.color_by, None, self.widget_id,
                                                              self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent,
                                                              self.sm_w, self.sm_h)
                    for node_str in sm_lu.keys(): svg.append(sm_lu[node_str])

            return ''.join(svg)

        #
        # __nodeSize__() - return the node size
        #
        def __nodeSize__(self):
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
            return _sz

        #
        # __renderNodes__() - render the nodes
        #
        def __renderNodes__(self):
            svg, selected_svg = [], []
            node_already_rendered = set()

            # Small multiple structures
            node_to_dfs = {}
            node_to_xy  = {}

            # Render nodes
            if self.node_size is not None and self.node_size != 'hidden':
                # Set the node size
                _sz = self.__nodeSize__()

                # Render position context (if selected) [nodes are rendered in the background]
                if self.render_pos_context:
                    _co = self.rt_self.co_mgr.getTVColor('context','text')
                    for node_str in self.pos.keys():
                        x = self.xT(self.pos[node_str][0])
                        y = self.yT(self.pos[node_str][1])
                        if x >= -5 and x <= self.w+5 and y >= -5 and y <= self.h+5:
                            svg.append(f'<circle cx="{x}" cy="{y}" r="{2}" fill="{_co}" stroke="{_co}" stroke-opacity="{self.pos_context_opacity}" fill-opacity="{self.pos_context_opacity}" />')

                # Iterate over the relationships
                for rel_tuple in self.relationships:
                    # Make sure it's the right number of tuples
                    if len(rel_tuple) < 2 or len(rel_tuple) > 3:
                        raise Exception(f'linkNode(): relationship tuples should have two or three parts "{rel_tuple}"')

                    # Flatten out into the groupby array, the fm_flds array, and the to_flds array
                    # ... not necessary anymore since the columns are getting concat'ed in the constructor
                    fm_flds = [rel_tuple[0]]
                    to_flds = [rel_tuple[1]]

                    # Do the from and to fields separately
                    for flds_i in range(0,2):
                        if flds_i == 0:  flds = fm_flds
                        else:            flds = to_flds
                        if flds_i == 1 and fm_flds == to_flds: continue # if they are the same thing, don't re-render
                        
                        # if the df has all of the columns
                        if len(set(self.df.columns) & set(flds)) == len(set(flds)):
                            # create the node table
                            if   self.rt_self.isPandas(self.df):
                                gb = self.df.groupby(flds[0]) if len(flds)  == 1 else self.df.groupby(flds)
                            elif self.rt_self.isPolars(self.df):
                                df_filtered = self.rt_self.polarsFilterColumnsWithNaNs(self.df, self.rt_self.flattenTuple(flds)) # stranded nodes are still possible... 
                                gb = df_filtered.group_by(flds[0]) if len(flds) == 1 else self.df.group_by(flds)
                            else: raise Exception('RTLinkNode.__renderNodes__() - only pandas and polars supported')

                            # iterate over the nodes
                            for k,k_df in gb:
                                node_str = self.rt_self.nodeStringAndFillPos(k, self.pos)

                                # Prevents duplicate renderings
                                if node_str in node_already_rendered: continue
                                node_already_rendered.add(node_str)
                                
                                # Transform the coordinates
                                x = self.xT(self.pos[node_str][0])
                                y = self.yT(self.pos[node_str][1])
                                self.node_coords[node_str] = (x,y)

                                if self.node_shape == 'small_multiple':
                                    if k not in node_to_dfs.keys(): node_to_dfs[k] = []

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
                                    elif self.node_color == 'vary' and self.color_by is not None and self.color_by in self.df.columns:
                                        _co_set = set(k_df[self.color_by])
                                        if len(_co_set) == 1:
                                            _co        = self.rt_self.co_mgr.getColor(_co_set.pop())
                                            _co_border = _co
                                        else:
                                            _co        = self.rt_self.co_mgr.getTVColor('data','default')
                                            _co_border = _co
                                    elif self.node_color is not None and self.node_color.startswith('#'):
                                        _co        = self.node_color
                                        if self.node_border_color is not None:  _co_border = self.node_border_color
                                        else:                                   _co_border = self.node_color
                                    else:
                                        _co        = self.rt_self.co_mgr.getTVColor('data','default')
                                        _co_border = self.rt_self.co_mgr.getTVColor('data','default_border')
                                                                            
                                    # Determine the size (if it varies)
                                    if self.node_size == 'vary':
                                        if self.count_by is None:
                                            _sz = self.node_size_max * len(k_df) / self.max_node_value
                                        elif self.count_by in self.df.columns and self.count_by_set:
                                            _sz = self.node_size_max * len(set(k_df[self.count_by])) / self.max_node_value
                                        elif self.count_by in self.df.columns:
                                            _sz = self.node_size_max * k_df[self.count_by].sum() / self.max_node_value
                                        else:
                                            _sz = 1
                                        if _sz < self.node_size_min:
                                            _sz = self.node_size_min
                                    
                                    # Determine the node shape
                                    # ... by dictionary... into either a shape string... or into an SVG string
                                    if type(self.node_shape) == dict:
                                        # Create the Node Shape Key ... complicated by tuples... // field (column) version
                                        _node_shape_key = flds
                                        if type(_node_shape_key) == list and len(_node_shape_key) == 1: _node_shape_key = _node_shape_key[0]
                                        if type(_node_shape_key) == list and len(_node_shape_key) >  1: _node_shape_key = tuple(_node_shape_key)

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
                                        svg_markup = self.rt_self.__overwriteSVGOriginPosition__(_shape, (x,y), _svg_w, _svg_h)
                                        svg_markup = self.rt_self.__overwriteSVGID__(_shape, self.nodeSVGID(k))
                                        svg.append(svg_markup)
                                        self.node_to_svg_markup[str(k)] = svg_markup
                                        _sz = _svg_h/2 # probably for the label?

                                    # Otherwise, call the super class shape renderer...
                                    else:
                                        svg_markup = self.rt_self.renderShape(_shape, x, y, _sz, _co, _co_border, self.node_opacity, self.nodeSVGID(k))
                                        svg.append(svg_markup)
                                        self.node_to_svg_markup[str(k)] = self.rt_self.renderShape(_shape, x, y, _sz) # unadorned
                                        if self.selected_entities is not None and node_str in self.selected_entities:
                                            selected_svg.append(self.node_to_svg_markup[str(k)])
                                        
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
                                        k_render_str = k_str
                                        if self.node_label_max_w is not None:
                                            k_render_str = self.rt_self.cropText(k_str, self.txt_h, self.node_label_max_w)
                                        if self.node_labels_only == False: # flag to indicate that the actual node string is hidden
                                            svg_text = self.rt_self.svgText(str(k_render_str), x, y+_sz+self.txt_h, self.txt_h, anchor='middle')                                            
                                            self.defer_render.append(svg_text) # Defer render

                                        if self.node_labels is not None and k_str in self.node_labels.keys():
                                            if self.node_labels_only: y_label = y + _sz + 1*self.txt_h
                                            else:                     y_label = y + _sz + 2*self.txt_h
                                            _strs_  = self.node_labels[k_str]
                                            if type(_strs_) == str:
                                                _str_render_ = _strs_
                                                if self.node_label_max_w is not None:
                                                    _str_render_ = self.rt_self.cropText(_strs_, self.txt_h, self.node_label_max_w)
                                                svg_text = self.rt_self.svgText(_str_render_, x, y_label, self.txt_h, anchor='middle')
                                                self.defer_render.append(svg_text) # Defer render
                                            else:
                                                for _str_ in _strs_:
                                                    _str_render_ = _str_
                                                    if self.node_label_max_w is not None:
                                                        _str_render_ = self.rt_self.cropText(_str_, self.txt_h, self.node_label_max_w)
                                                    svg_text = self.rt_self.svgText(_str_render_, x, y_label, self.txt_h, anchor='middle')
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
                    svg.append(_small_multiple_svg_)

                    # Copy of the draw labels portion a few lines up... 2024-05-15 // NEEDS UPDATING
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
                if self.draw_labels: # Copy of the above... 2024-05-15 // NEEDS UPDATING
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
            return ''.join(svg) + '<g stroke="#ff0000" stroke-width="2.5" fill="none">' + ''.join(selected_svg) + '</g>'

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
        # print() version of class
        #
        def __repr__(self):
            _s_ = []
            for k in self.relationships: _s_.append(str(k[0]) + '->' + str(k[1]))
            _relates_ = ' | '.join(_s_)
            return f'linkNode(df.len={len(self.df)}, relationships={_relates_}, {self.w}x{self.h})'

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
        def getViewWindow(self):
            return self.view_window

        #
        # nodeSVGID() - return the SVG ID for the specified node
        #
        def nodeSVGID(self, node):
            return self.rt_self.encSVGID(self.widget_id + '-' + str(node))

        #
        # renderSVG() - render as SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state: self.geom_to_df = {}
            self.node_to_svg_markup = {}

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
        # overlappingDataFrames() - Determine which dataframe geometries overlap with a specific region
        # - to_intersect should be a shapely shape
        # - return value is a pandas dataframe or None
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

        #
        # overlappingEntities() - Determine which entity geometrics overlap with a specific region
        # - to_intersect should be a shapely shape
        # - return value is a list of entities (possibly an empty list) or None
        #
        def overlappingEntities(self, to_intersect):
            node_strs = set()
            for node_str in self.node_coords:
                xy = self.node_coords[node_str]
                if to_intersect.contains(Point(xy[0],xy[1])):
                    node_strs.add(node_str)
            return node_strs

        #
        # entitiesAtPoint() - Determine all the entities under a specific point
        #
        def entitiesAtPoint(self, xy):
            node_strs = set()
            for node_str in self.node_coords:
                node_xy = self.node_coords[node_str]
                if (xy[0] >= node_xy[0]-5) and (xy[0] <= node_xy[0]+5) and \
                   (xy[1] >= node_xy[1]-5) and (xy[1] <= node_xy[1]+5):
                    node_strs.add(node_str)
            return node_strs

        #
        # __createPathDescriptionOfSelectedEntities__() - create an svg path description of the selected entities
        # - for prototyping the graph interact panel application
        #
        def __createPathDescriptionOfSelectedEntities__(self, my_selection=None):
            if my_selection is None: my_selection = self.selected_entities
            if my_selection is None or len(my_selection) == 0: return ''
            d = []
            for node_str in my_selection:
                if node_str in self.node_coords:
                    xy = self.node_coords[node_str]
                    sx = xy[0]
                    sy = xy[1]
                    d.append(f'M {sx-5} {sy-5} l 10 0 l 0 10 l -10 0 l 0 -10 z')
            return ' '.join(d)

        #
        # __createPathDescriptionForAllEntities__() - create an svg path description of all entities
        # - for prototyping the graph interact panel application
        #
        def __createPathDescriptionForAllEntities__(self):
            d = []
            for node_str in self.node_coords:
                if node_str in self.node_coords:
                    xy = self.node_coords[node_str]
                    sx = xy[0]
                    sy = xy[1]
                    d.append(f'M {sx-5} {sy-5} l 10 0 l 0 10 l -10 0 l 0 -10 z')
            return ' '.join(d)

        #
        #  __adjustSelectedEntities__() - adjust the selected entities
        # - for prototyping the graph interact panel application
        #
        def __moveSelectedEntities__(self, dxy, my_selection=None):
            if my_selection is None: my_selection = self.selected_entities
            if my_selection is None or len(my_selection) == 0: return
            for node_str in my_selection:
                if node_str in self.node_coords:
                    xy                 = self.node_coords[node_str]
                    xy_new             = (self.xT_inv(xy[0] + dxy[0]), self.yT_inv(xy[1] + dxy[1]))
                    self.pos[node_str] = xy_new
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
             entity_str = str(entity)
             if entity_str in self.node_coords:
                 xy   = self.node_coords[entity_str]
                 rtep = RTEntityPosition(entity_str,
                                         self.rt_self,
                                         self,
                                         xy,
                                         (xy[0], xy[1], 0.0, 1.0),
                                         self.nodeSVGID(entity_str),
                                         self.node_to_svg_markup[entity_str],
                                         self.widget_id)
                 return [rtep]
             return []
        
        def entityPositions(self, entity_or_label):
            if entity_or_label in self.node_coords:
                return self.__entityPositions__(entity_or_label)
            elif len(self.node_labels) > 0:
                rteps = []
                for entity in self.node_labels:
                    if self.node_labels[entity] == entity_or_label:
                        _results_ = self.__entityPositions__(entity)
                        for rtep in _results_:
                            rtep.entity = entity_or_label
                            rteps.append(rtep)
                return rteps
            return []
