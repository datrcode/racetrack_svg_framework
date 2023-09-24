# Copyright 2022 David Trimm
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
    # nodeLabeler()
    # - Create the dictionary for the node_labels parameter
    # - if node_labels is passed, will be added to / not replaced
    #
    def nodeLabeler(self, df, node_field, label_field, node_labels=None, word_wrap=True, max_line_len=32, max_lines=4):
        if node_labels is None:
            node_labels = {}
        gb = df.groupby(node_field)
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
    def nodeStringAndFillPos(self, k, pos):
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
                            gb = _df.groupby(flds[0])
                        else:
                            gb = _df.groupby(flds)

                        # iterate over the edges
                        for k,k_df in gb:
                            node_str = self.nodeStringAndFillPos(k, pos)

                            # Perform the comparison for the bounds
                            v = pos[node_str]
                            if v[0] > wx1:
                                wx1 = v[0]
                            if v[0] < wx0:
                                wx0 = v[0]
                            if v[1] > wy1:
                                wy1 = v[1]
                            if v[1] < wy0:
                                wy0 = v[1]
                            
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
        self.identifyColumnsFromParameters('color_by',kwargs,columns_set)
        self.identifyColumnsFromParameters('count_by',kwargs,columns_set)
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
                 link_dash         = None,     # svg stroke-dash string, callable, or dictionary of relationship tuple to dash string array 

                 max_link_size     = 4,        # for link vary...
                 min_link_size     = 0.25,     # for link vary...

                 label_only        = set(),    # label only set

                 # -----------------------     # convex hull annotations

                 convex_hull_lu           = None,  # dictionary... regex for node name to convex hull name
                 convex_hull_opacity      = 0.3,   # opacity of the convex hulls
                 convex_hull_labels       = False, # draw a label for the convex hull in the center of the convex hull
                 convex_hull_stroke_width = None,  # Stroke width for the convex hull -- if None, will not be drawn...

                 # -----------------------     # small multiple options

                 sm_type               = None, # should be the method name // similar to the smallMultiples method
                 sm_w                  = None, # override the width of the small multiple
                 sm_h                  = None, # override the height of the small multiple
                 sm_params             = {},   # dictionary of parameters for the small multiples
                 sm_x_axis_independent = True, # Use independent axis for x (xy, temporal, and linkNode)
                 sm_y_axis_independent = True, # Use independent axis for y (xy, temporal, periodic, pie)

                 # -----------------------     # visualization geometry / etc.

                 x_view=0,                     # x offset for the view
                 y_view=0,                     # y offset for the view
                 w=256,                        # width of the view
                 h=256,                        # height of the view
                 x_ins=3,
                 y_ins=3,
                 txt_h=12,                     # text height for labeling
                 draw_labels=True,             # draw labels flag # not implemented yet
                 draw_border=True):            # draw a border around the graph
        rt_linknode = self.RTLinkNode(self,df,relationships,pos=pos,use_pos_for_bounds=use_pos_for_bounds,render_pos_context=render_pos_context,
                                      pos_context_opacity=pos_context_opacity,bounds_percent=bounds_percent,color_by=color_by,count_by=count_by,
                                      count_by_set=count_by_set,widget_id=widget_id,node_color=node_color,node_border_color=node_border_color,
                                      node_size=node_size,node_shape=node_shape,node_opacity=node_opacity,node_labels=node_labels,node_labels_only=node_labels_only,
                                      max_node_size=max_node_size,min_node_size=min_node_size,
                                      link_color=link_color,link_size=link_size,link_opacity=link_opacity,link_shape=link_shape,link_arrow=link_arrow, link_dash=link_dash,
                                      max_link_size=max_link_size,min_link_size=min_link_size,
                                      label_only=label_only,convex_hull_lu=convex_hull_lu,convex_hull_opacity=convex_hull_opacity,
                                      convex_hull_labels=convex_hull_labels,convex_hull_stroke_width=convex_hull_stroke_width,
                                      sm_type=sm_type,sm_w=sm_w,sm_h=sm_h,sm_params=sm_params,sm_x_axis_independent=sm_x_axis_independent,
                                      sm_y_axis_independent=sm_y_axis_independent,x_view=x_view,y_view=y_view,w=w,h=h,x_ins=x_ins,y_ins=y_ins,
                                      txt_h=txt_h,draw_labels=draw_labels,draw_border=draw_border)
        return rt_linknode.renderSVG()

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
                if  _df[count_by].dtypes != np.int64    and \
                    _df[count_by].dtypes != np.int32    and \
                    _df[count_by].dtypes != np.float64  and \
                    _df[count_by].dtypes != np.float32:
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
                if  _df[count_by].dtypes != np.int64    and \
                    _df[count_by].dtypes != np.int32    and \
                    _df[count_by].dtypes != np.float64  and \
                    _df[count_by].dtypes != np.float32:
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

                # if the _df has all of the columns
                if len(set(_df.columns) & set(flat)) == len(set(flat)):

                    if count_by is None or count_by_set: # count_by_set not implemented...
                        gb = _df.groupby(flat).size()
                    else:
                        gb = _df.groupby(flat)[count_by].sum()

                    for i in range(0,len(gb)):
                        k = gb.index[i]

                        k_fm   = k[:len(fm_flds)]
                        k_to   = k[len(fm_flds):]
                        _fm_ = '|'.join(k_fm) if len(k_fm) > 1 else str(k_fm[0])
                        _to_ = '|'.join(k_to) if len(k_to) > 1 else str(k_to[0])
                        nx_g.add_edge(_fm_,_to_,weight=gb.iloc[i])

        return nx_g

    #
    # linkNodeInstance() - create a RTLinkNode instance
    #    
    def linkNodeInstance(self,
                         df,                                  # dataframe(s) to render ... unlike other parts, this can be more than one...
                         relationships,                       # list of tuple pairs... pairs can be single strings or tuples of strings
                                                              # [('f0','f1')] // 1 relationship: f0 to f1
                                                              # [('f0','f1'),('f1','f2')] // 2 relationships: f0 to f1 and f1 to f2
                                                              # [(('f0','f1'),('f2','f3'))] // 1 relationship: 'f0'|'f1' to 'f2'|'f3'
                         # ---------------------------------- # everything else is a default...
                         pos                      = {},       # networkx style position dictionary pos['node_name'] = 2d array of positions e.g., [[0...1],[0...1]]
                         use_pos_for_bounds       = True,     # use the pos values for the boundary of the view
                         render_pos_context       = False,    # Render all the pos keys by default...  to provide context for the other nodes
                         pos_context_opacity      = 0.8,      # opacity of the pos context nodes
                         bounds_percent           = .05,      # inset the graph into the view by this percent... so that the nodes aren't right at the edges 
                         color_by                 = None,     # just the default color or a string for a field
                         count_by                 = None,     # none means just count rows, otherwise, use a field to sum by
                         count_by_set             = False,    # count by summation (by default)... count_by column is checked
                         widget_id                = None,     # naming the svg elements                 
                         # ---------------------------------- # linknode visualization
                         node_color               = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
                                                              # ... or a dictionary of the node string to either a string to color hash or a "#xxxxxx"
                         node_border_color        = None,     # small edge around nodes ... should only be "#xxxxxx"
                         node_size                = 'medium', # 'small', 'medium', 'large', 'vary', 'hidden' / None
                         node_shape               = None,     # 'square', 'ellipse' / None, 'triangle', 'utriangle', 'diamond', 'plus', 'x', 'small_multiple',
                                                              # ... or a dictionary of the field tuples node to a shape name
                                                              # ... or a dictionary of the field tuples node to an SVG small multiple
                                                              # ... or a function pointer to a shape function
                         node_opacity             = 1.0,      # fixed node opacity
                         node_labels              = None,     # Dictionary of node string to array of strings for additional labeling options
                         node_labels_only         = False,    # Only label based on the node_labels dictionary                         
                         max_node_size            = 4,        # for node vary...
                         min_node_size            = 0.3,      # for node vary...
                         link_color               = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
                         link_size                = 'small',  # 'nil', 'small', 'medium', 'large', 'vary', 'hidden' / None
                         link_opacity             = '1.0',    # link opacity
                         link_shape               = 'line',   # 'curve','line'
                         link_arrow               = True,     # draw an arrow at the end of the curve...
                         link_dash                = None,     # string for svg stroke-dash, callable, or dictionary of the relationship tuple to stroke dash string
                         max_link_size            = 4,        # for link vary...
                         min_link_size            = 0.25,     # for link vary...
                         label_only               = set(),    # label only set
                         # ---------------------------------- # convex hull annotations
                         convex_hull_lu           = None,     # dictionary... regex for node name to convex hull name
                         convex_hull_opacity      = 0.3,      # opacity of the convex hulls
                         convex_hull_labels       = False,    # draw a label for the convex hull in the center of the convex hull
                         convex_hull_stroke_width = None,     # Stroke width for the convex hull -- if None, will not be drawn...
                         # ---------------------------------- # small multiple options
                         sm_type                  = None,     # should be the method name // similar to the smallMultiples method
                         sm_w                     = None,     # override the width of the small multiple
                         sm_h                     = None,     # override the height of the small multiple
                         sm_params                = {},       # dictionary of parameters for the small multiples
                         sm_x_axis_independent    = True,     # Use independent axis for x (xy, temporal, and linkNode)
                         sm_y_axis_independent    = True,     # Use independent axis for y (xy, temporal, periodic, pie)
                         # ---------------------------------- # visualization geometry / etc.
                         x_view                   = 0,        # x offset for the view
                         y_view                   = 0,        # y offset for the view
                         w                        = 256,      # width of the view
                         h                        = 256,      # height of the view
                         x_ins                    = 3,
                         y_ins                    = 3,
                         txt_h                    = 12,       # text height for labeling
                         draw_labels              = True,     # draw labels flag # not implemented yet
                         draw_border              = True):    # draw a border around the graph
        return self.RTLinkNode(self,df,relationships,pos=pos,use_pos_for_bounds=use_pos_for_bounds,render_pos_context=render_pos_context,
                               pos_context_opacity=pos_context_opacity,bounds_percent=bounds_percent,color_by=color_by,count_by=count_by,
                               count_by_set=count_by_set,widget_id=widget_id,node_color=node_color,node_border_color=node_border_color,
                               node_size=node_size,node_shape=node_shape,node_opacity=node_opacity,node_labels=node_labels,node_labels_only=node_labels_only,
                               max_node_size=max_node_size,min_node_size=min_node_size,
                               link_color=link_color,link_size=link_size,link_opacity=link_opacity,link_shape=link_shape,link_arrow=link_arrow, link_dash=link_dash,
                               max_link_size=max_link_size,min_link_size=min_link_size,
                               label_only=label_only,convex_hull_lu=convex_hull_lu,convex_hull_opacity=convex_hull_opacity,
                               convex_hull_labels=convex_hull_labels,convex_hull_stroke_width=convex_hull_stroke_width,
                               sm_type=sm_type,sm_w=sm_w,sm_h=sm_h,sm_params=sm_params,sm_x_axis_independent=sm_x_axis_independent,
                               sm_y_axis_independent=sm_y_axis_independent,x_view=x_view,y_view=y_view,w=w,h=h,x_ins=x_ins,y_ins=y_ins,
                               txt_h=txt_h,draw_labels=draw_labels,draw_border=draw_border)

    #
    # RTLinkNode Class
    #
    class RTLinkNode(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,                             # outer class reference 
                     df,                                  # dataframe(s) to render ... unlike other parts, this can be more than one...
                     relationships,                       # list of tuple pairs... pairs can be single strings or tuples of strings
                                                          # [('f0','f1')] // 1 relationship: f0 to f1
                                                          # [('f0','f1'),('f1','f2')] // 2 relationships: f0 to f1 and f1 to f2
                                                          # [(('f0','f1'),('f2','f3'))] // 1 relationship: 'f0'|'f1' to 'f2'|'f3'
                     # ---------------------------------- # everything else is a default...
                     pos                      = {},       # networkx style position dictionary pos['node_name'] = 2d array of positions e.g., [[0...1],[0...1]]
                     use_pos_for_bounds       = True,     # use the pos values for the boundary of the view
                     render_pos_context       = False,    # Render all the pos keys by default...  to provide context for the other nodes
                     pos_context_opacity      = 0.8,      # opacity of the pos context nodes
                     bounds_percent           = .05,      # inset the graph into the view by this percent... so that the nodes aren't right at the edges 
                     color_by                 = None,     # just the default color or a string for a field
                     count_by                 = None,     # none means just count rows, otherwise, use a field to sum by
                     count_by_set             = False,    # count by summation (by default)... count_by column is checked
                     widget_id                = None,     # naming the svg elements                 
                     # ---------------------------------- # linknode visualization
                     node_color               = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
                                                          # ... or a dictionary of the node string to either a string to color hash or a "#xxxxxx"
                     node_border_color        = None,     # small edge around nodes ... should only be "#xxxxxx"
                     node_size                = 'medium', # 'small', 'medium', 'large', 'vary', 'hidden' / None
                     node_shape               = None,     # 'square', 'ellipse' / None, 'triangle', 'utriangle', 'diamond', 'plus', 'x', 'small_multiple',
                                                          # ... or a dictionary of the field tuples node to a shape name
                                                          # ... or a dictionary of the field tuples node to an SVG small multiple
                                                          # ... or a function pointer to a shape function
                     node_opacity             = 1.0,      # fixed node opacity
                     node_labels              = None,     # Dictionary of node string to array of strings for additional labeling options
                     node_labels_only         = False,    # Only label based on the node_labels dictionary                     
                     max_node_size            = 4,        # for node vary...
                     min_node_size            = 0.3,      # for node vary...
                     link_color               = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
                     link_size                = 'small',  # 'nil', 'small', 'medium', 'large', 'vary', 'hidden' / None
                     link_opacity             = '1.0',    # link opacity
                     link_shape               = 'line',   # 'curve','line'
                     link_arrow               = True,     # draw an arrow at the end of the curve...
                     link_dash                = None,     # String for the stroke-dash, callable, or dictionary of relationship tuple to the stroke-dash string
                     max_link_size            = 4,        # for link vary...
                     min_link_size            = 0.25,     # for link vary...
                     label_only               = set(),    # label only set
                     # ---------------------------------- # convex hull annotations
                     convex_hull_lu           = None,     # dictionary... regex for node name to convex hull name
                     convex_hull_opacity      = 0.3,      # opacity of the convex hulls
                     convex_hull_labels       = False,    # draw a label for the convex hull in the center of the convex hull
                     convex_hull_stroke_width = None,     # Stroke width for the convex hull -- if None, will not be drawn...
                     # ---------------------------------- # small multiple options
                     sm_type                  = None,     # should be the method name // similar to the smallMultiples method
                     sm_w                     = None,     # override the width of the small multiple
                     sm_h                     = None,     # override the height of the small multiple
                     sm_params                = {},       # dictionary of parameters for the small multiples
                     sm_x_axis_independent    = True,     # Use independent axis for x (xy, temporal, and linkNode)
                     sm_y_axis_independent    = True,     # Use independent axis for y (xy, temporal, periodic, pie)
                     # ---------------------------------- # visualization geometry / etc.
                     x_view                   = 0,        # x offset for the view
                     y_view                   = 0,        # y offset for the view
                     w                        = 256,      # width of the view
                     h                        = 256,      # height of the view
                     x_ins                    = 3,
                     y_ins                    = 3,
                     txt_h                    = 12,       # text height for labeling
                     draw_labels              = True,     # draw labels flag # not implemented yet
                     draw_border              = True):    # draw a border around the graph

            self.parms                      = locals().copy()

            self.rt_self                    = rt_self
            self.relationships              = relationships
            self.pos                        = pos
            self.use_pos_for_bounds         = use_pos_for_bounds
            self.render_pos_context         = render_pos_context
            self.pos_context_opacity        = pos_context_opacity
            self.bounds_percent             = bounds_percent
            self.color_by                   = color_by
            self.count_by                   = count_by
            self.count_by_set               = count_by_set
            self.widget_id                  = widget_id

            # Make a widget_id if it's not set already
            if self.widget_id is None:
                self.widget_id = "linknode_" + str(random.randint(0,65535))

            self.node_color                 = node_color
            self.node_border_color          = node_border_color
            self.node_size                  = node_size
            self.node_shape                 = node_shape
            self.node_opacity               = node_opacity
            self.node_labels                = node_labels
            self.node_labels_only           = node_labels_only
            self.max_node_size              = max_node_size
            self.min_node_size              = min_node_size
            self.link_color                 = link_color
            self.link_size                  = link_size
            self.link_opacity               = link_opacity
            self.link_shape                 = link_shape
            self.link_arrow                 = link_arrow
            self.link_dash                  = link_dash
            self.max_link_size              = max_link_size
            self.min_link_size              = min_link_size
            self.label_only                 = label_only
            self.convex_hull_lu             = convex_hull_lu
            self.convex_hull_opacity        = convex_hull_opacity
            self.convex_hull_labels         = convex_hull_labels
            self.convex_hull_stroke_width   = convex_hull_stroke_width
            self.sm_type                    = sm_type
            self.sm_w                       = sm_w
            self.sm_h                       = sm_h
            self.sm_params                  = sm_params
            self.sm_x_axis_independent      = sm_x_axis_independent
            self.sm_y_axis_independent      = sm_y_axis_independent
            self.x_view                     = x_view
            self.y_view                     = y_view
            self.w                          = w
            self.h                          = h
            self.x_ins                      = x_ins
            self.y_ins                      = y_ins
            self.txt_h                      = txt_h
            self.draw_labels                = draw_labels
            self.draw_border                = draw_border

            # Make sure it's a list... and prevent the added columns from corrupting original dataframe
            my_df_list = []
            if type(df) != list:
                my_df_list.append(df.copy())
            else:
                for _df in df:
                    my_df_list.append(_df.copy())
            self.df = my_df_list
            
            # Make a widget_id if it's not set already
            if widget_id == None:
                widget_id = "linknode_" + str(random.randint(0,65535))

            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)

            # Apply node field transforms across all of the dataframes
            for _df in self.df:
                for _edge in self.relationships:
                    for _node in _edge:
                        if type(_node) == str:
                            if rt_self.isTField(_node) and rt_self.tFieldApplicableField(_node) in _df.columns:
                                _df,_throwaway = rt_self.applyTransform(_df, _node)
                        else:
                            for _tup_part in _node:
                                if rt_self.isTField(_tup_part) and rt_self.tFieldApplicableField(_tup_part) in _df.columns:
                                    _df,_throwaway = rt_self.applyTransform(_df, _tup_part)

            # Check the node information... make sure the parameters are set
            if self.sm_type is not None:
                self.node_shape = 'small_multiple'             
            if self.node_shape == 'small_multiple':
                if self.sm_type is None:        # sm_type must be set to the widget type... else default back to small node size
                    self.node_shape = 'ellipse'
                    self.node_size  = 'small'
                elif self.sm_w is None or self.sm_h is None:
                    self.sm_w,self.sm_h = getattr(rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
            elif callable(self.node_shape) and self.node_size is None:
                self.node_size = 'medium'

            # Check the count_by column across all the df's...  if any of them
            # don't work.. then it's count_by_set
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)
            
            # Tracking state
            self.geom_to_df = {}
            self.last_render = None

        #
        # __calculateGeometry__() - determine the geometry for the view
        #
        def __calculateGeometry__(self):
            # Calculate world coordinates
            self.wx0 = math.inf
            self.wy0 = math.inf
            self.wx1 = -math.inf
            self.wy1 = -math.inf

            # And possibly the max node size
            self.max_node_value = 1

            if self.use_pos_for_bounds:
                for k in self.pos.keys():
                    v = self.pos[k]
                    if v[0] > self.wx1:
                        self.wx1 = v[0]
                    if v[0] < self.wx0:
                        self.wx0 = v[0]
                    if v[1] > self.wy1:
                        self.wy1 = v[1]
                    if v[1] < self.wy0:
                        self.wy0 = v[1]
                if self.node_size == 'vary':
                    self.max_node_value,ignore0,ignore1,ignore2,ignore3 = self.rt_self.calculateNodeInformation(self.df, self.relationships, self.pos, self.count_by, self.count_by_set)
            else:
                self.max_node_value,self.wx0,self.wy0,self.wx1,self.wy1 = self.rt_self.calculateNodeInformation(self.df, self.relationships, self.pos, self.count_by, self.count_by_set)

            # Make it sane
            if math.isinf(self.wx0):
                self.wx0 = 0.0
                self.wy0 = 0.0
                self.wx1 = 1.0
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

            # Coordinate transform lambdas
            self.xT = lambda __x__: self.x_ins + (self.w - 2*self.x_ins) * (__x__ - self.wx0)/(self.wx1-self.wx0)
            self.yT = lambda __y__: self.y_ins + (self.h - 2*self.y_ins) * (__y__ - self.wy0)/(self.wy1-self.wy0)

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
                            gb = _df.groupby(flat)                            
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
        def __renderLinks__(self,track_state=False):
            # Render links
            svg          = ''
            count_by_set = True
            if self.link_size is not None and self.link_size != 'hidden':
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
                            if  _df[self.count_by].dtypes != np.int64    and \
                                _df[self.count_by].dtypes != np.int32    and \
                                _df[self.count_by].dtypes != np.float64  and \
                                _df[self.count_by].dtypes != np.float32:
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
                            gb = _df.groupby(flat)
                            if self.count_by is None: 
                                gb_sz = gb.size()
                            elif count_by_set:
                                gb_sz = gb[self.count_by].nunique()
                            else:
                                gb_sz = gb[self.count_by].sum()

                            gb_sz_i = 0
                            for k,k_df in gb:
                                _weight_ =  gb_sz.iloc[gb_sz_i]
                                gb_sz_i  += 1

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
                                if track_state:
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
                                elif self.link_shape == 'curve':
                                    dx = x2 - x1
                                    dy = y2 - y1
                                    # vector length
                                    l  = sqrt((dx*dx)+(dy*dy))
                                    if l == 0:
                                        l = 1

                                    # normalize the vector
                                    dx /= l
                                    dy /= l

                                    # calculate the perpendicular vector
                                    pdx =  dy
                                    pdy = -dx

                                    # calculate the control points
                                    x1p = x1 + 0.2*l*dx + 0.2*l*pdx
                                    y1p = y1 + 0.2*l*dy + 0.2*l*pdy

                                    x2p = x2 - 0.2*l*dx + 0.2*l*pdx
                                    y2p = y2 - 0.2*l*dy + 0.2*l*pdy

                                    x3  = x2 - 0.1*l*dx
                                    y3  = y2 - 0.1*l*dy

                                    if self.link_arrow:
                                        svg += f'<path d="M {x1} {y1} C {x1p} {y1p} {x2p} {y2p} {x2} {y2} L {x3} {y3}" '
                                        svg += f'fill-opacity="0.0" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" {stroke_dash} />'
                                    else:
                                        svg += f'<path d="M {x1} {y1} C {x1p} {y1p} {x2p} {y2p} {x2} {y2}" '
                                        svg += f'fill-opacity="0.0" stroke-width="{_this_sz}" stroke="{_co}" stroke-opacity="{self.link_opacity}" {stroke_dash} />'
                                else:
                                    raise Exception(f'Unknown link_shape "{self.link_shape}"')

            return svg

        #
        # __renderNodes__() - render the nodes
        #
        def __renderNodes__(self, track_state=False):
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
                                if len(flds) == 1:
                                    gb = _df.groupby(flds[0])
                                else:
                                    gb = _df.groupby(flds)

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

                                    if self.node_shape == 'small_multiple':
                                        if node_str not in node_to_dfs.keys():
                                            node_to_dfs[node_str] = []

                                        node_to_dfs[node_str].append(k_df)
                                        node_to_xy[node_str] = (x,y)

                                        if track_state:
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
                                        if track_state:
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
                                        if self.draw_labels and ((len(self.label_only) == 0) or (k_str in self.label_only)):
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
                                                
                for node_str in sm_lu.keys():
                    svg += sm_lu[node_str]

                    # Copy of the draw labels portion a few lines up...
                    if self.draw_labels:
                        if len(node_str) > 16:
                            node_str = node_str[:16] + '...'
                        x = node_to_xy[node_str][0]
                        y = node_to_xy[node_str][1]
                        svg_text = self.rt_self.svgText(node_str, x, y+self.sm_h/2+self.txt_h, self.txt_h, anchor='middle')
                        self.defer_render.append(svg_text)

                # Possible that some nodes may not have been rendered due to the nature of the multi-dataframe structure
                if self.draw_labels:
                    for node_str in node_to_xy.keys():
                        if node_str not in sm_lu.keys():
                            if len(node_str) > 16:
                                node_str = node_str[:16] + '...'
                            x = node_to_xy[node_str][0]
                            y = node_to_xy[node_str][1]
                            svg_text = self.rt_self.svgText(node_str, x, y+self.txt_h/2, self.txt_h, anchor='middle')
                            self.defer_render.append(svg_text)
            return svg

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
            return self.last_render

        #
        # renderSVG() - render as SVG
        #
        def renderSVG(self, just_calc_max=False, track_state=False):
            # Determine geometry
            self.__calculateGeometry__()

            # Start the SVG Frame
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'

            # Elements to render after nodes (labels, in this case)
            self.defer_render = []

            # Render convex hulls, links, and then nodes
            svg += self.__renderConvexHull__()
            svg += self.__renderLinks__(track_state)
            svg += self.__renderNodes__(track_state)

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
                return pd.concat(_dfs)
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
