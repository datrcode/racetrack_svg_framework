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

import re
import inspect
import random

import pandas as pd
import polars as pl
import numpy as np

from shapely.affinity import translate
from shapely.geometry import Polygon, Point

from .rt_component import RTComponent

from math import ceil

__name__ = 'rt_layouts_mixin'

#
# Abstraction for colorscale // by default this is the dark theme
#
class RTLayoutsMixin(object):
    #
    # Place a node within the tree based on the layout path
    #
    def __recursivePlace(self, root, pos, leaf):
        if '|' in pos:
            first = pos[:pos.index('|')]
            rest  = pos[pos.index('|')+1:]
            if first not in root.keys():
                root[first] = {}
            self.__recursivePlace(root[first], rest, leaf)
        else:
            if pos in root.keys():
                print('error ... pos in {root.keys()}')
            else:
                root[pos] = leaf

    #
    # Return the layout type
    #
    def __layoutType(self, x):
        if x in ['c','n','e','w','s']: # BorderLayout
            return 'borderLayout'
        elif ',' in x: # GridLayout
            return 'gridLayout'
        elif re.match(r"[-]{0,1}[0-9]+",x) and int(x) < 0: # FlowLayout Horizontal
            return 'flowLayoutHorizontal'
        elif re.match(r"[-]{0,1}[0-9]+",x): # FlowLayout Vertical
            return 'flowLayoutVertical'
        else: # Unknown -- throw exception
            raise Exception(f'unknown layout mnemonic "{x}"')

    #
    # Flow Layout (Vertical) Calculation
    #
    def __flowLayoutVerticalCalculation(self, my_node, leaf_dims, so_far):
        min_dim  = None
        pref_dim = None
        for key in my_node.keys():
            my_min_dim  = leaf_dims[so_far + '|' + key][0]
            my_pref_dim = leaf_dims[so_far + '|' + key][1]
            if min_dim is None:
                min_dim  = my_min_dim
                pref_dim = my_pref_dim
            else:
                min_dim  = (max(min_dim[0],  my_min_dim[0]),  min_dim[1]  + my_min_dim[1])
                pref_dim = (max(pref_dim[0], my_pref_dim[0]), pref_dim[1] + my_pref_dim[1])
        return (min_dim, pref_dim)

    #
    # Flow Layout (Horizontal) Calculation
    #
    def __flowLayoutHorizontalCalculation(self, my_node, leaf_dims, so_far):
        min_dim  = None
        pref_dim = None
        for key in my_node.keys():
            my_min_dim  = leaf_dims[so_far + '|' + key][0]
            my_pref_dim = leaf_dims[so_far + '|' + key][1]
            if min_dim is None:
                min_dim  = my_min_dim
                pref_dim = my_pref_dim
            else:
                min_dim  = (min_dim[0]  + my_min_dim[0],  max(min_dim[1],  my_min_dim[1]))
                pref_dim = (pref_dim[0] + my_pref_dim[0], max(pref_dim[1], my_pref_dim[1]))
        return (min_dim, pref_dim)

    #
    # Grid Layout Calculation
    #
    def __gridLayoutCalculation(self, my_node, leaf_dims, so_far):
        min_dim  = None
        pref_dim = None
        max_x    = 1
        max_y    = 1
        for key in my_node.keys():
            x = int(key.split(',')[0])
            y = int(key.split(',')[1])
            max_x = max(x, max_x)
            max_y = max(y, max_y)

            my_min_dim  = leaf_dims[so_far + '|' + key][0]
            my_pref_dim = leaf_dims[so_far + '|' + key][1]
            if min_dim is None:
                min_dim  = my_min_dim
                pref_dim = my_pref_dim
            else:
                min_dim  = (max(min_dim[0],  my_min_dim[0]),  max(min_dim[1],  my_min_dim[1]))
                pref_dim = (max(pref_dim[0], my_pref_dim[0]), max(pref_dim[1], my_pref_dim[1]))


        return ((min_dim[0] * max_x, min_dim[1] * max_y), (pref_dim[0] * max_x, pref_dim[1] * max_y))

    #
    # Border Layout Calculation
    #
    def __borderLayoutCalculation(self, my_node, leaf_dims, so_far):
        s_dim = n_dim = w_dim = e_dim = c_dim = ((0,0),(0,0))
        if 'e' in my_node.keys():
            e_dim = leaf_dims[so_far+'|'+'e']
        if 'w' in my_node.keys():
            w_dim = leaf_dims[so_far+'|'+'w']
        if 'n' in my_node.keys():
            n_dim = leaf_dims[so_far+'|'+'n']
        if 's' in my_node.keys():
            s_dim = leaf_dims[so_far+'|'+'s']
        if 'c' in my_node.keys():
            c_dim = leaf_dims[so_far+'|'+'c']

        x_min  = max(e_dim[0][0] + c_dim[0][0] + w_dim[0][0], n_dim[0][0], s_dim[0][0])
        x_pref = max(e_dim[1][0] + c_dim[1][0] + w_dim[1][0], n_dim[1][0], s_dim[1][0])

        y_min  = max(e_dim[0][1],  c_dim[0][1]  ,w_dim[0][1]) + n_dim[0][1] + s_dim[0][1]
        y_pref = max(e_dim[1][1],  c_dim[1][1]  ,w_dim[1][1]) + n_dim[1][1] + s_dim[1][1]

        return ((x_min,y_min),(x_pref,y_pref))

    #
    # Fill in the interior node layout dimensions
    #
    def __fillInteriorNodeDimensions(self, my_node, leaf_dims, dims, so_far='root'):        
        #
        # Leaf Node -- store the dimensions
        #
        if type(my_node) == tuple:
            widget_type       = str(my_node)
            leaf_dims[so_far] = (dims[widget_type]['min'],dims[widget_type]['pref'])

        #
        # Leaf Node -- that doesn't really have a tuple...
        #
        elif type(my_node) == str:
            widget_type       = my_node
            leaf_dims[so_far] = (dims[widget_type]['min'],dims[widget_type]['pref'])
            
        #
        # Interior Node -- have the leaves execute first... then do the layout calculation here
        #
        else:
            # Visit leaves first... and then process this node
            for key in my_node.keys():
                self.__fillInteriorNodeDimensions(my_node[key], leaf_dims, dims, so_far = so_far+'|'+str(key))

            # Figure out the dimensions based on the layout schema
            # - make sure there's only one layout type for all the children
            child_layouts = set()
            for child in my_node.keys():
                child_layouts.add(self.__layoutType(child))
            if len(child_layouts) != 1:
                raise Exception(f'unknown layout mnemonic "{child_layouts}"')

            # - convert any of the children to the layout type
            child_layout = self.__layoutType(child)

            # - call the appropriate method to convert that layout to dimensions
            if   child_layout == 'borderLayout':
                leaf_dims[so_far] = self.__borderLayoutCalculation(my_node, leaf_dims, so_far)
            elif child_layout == 'gridLayout':
                leaf_dims[so_far] = self.__gridLayoutCalculation(my_node, leaf_dims, so_far)
            elif child_layout == 'flowLayoutHorizontal':
                leaf_dims[so_far] = self.__flowLayoutHorizontalCalculation(my_node, leaf_dims, so_far)
            elif child_layout == 'flowLayoutVertical':
                leaf_dims[so_far] = self.__flowLayoutVerticalCalculation(my_node, leaf_dims, so_far)
            else: # Unknown -- throw exception
                raise Exception(f'unknown layout mnemonic "{child}" / "{child_layout}"')

    #
    # Finalizize the widget placement
    #
    def __finalizeWidgetPlacement(self, my_node, leaf_dims, placement, x, y, w, h, h_gap, v_gap, widget_h_gap, widget_v_gap, so_far='root'):
        # Purely just the placement information
        placement[so_far] = (x,y,w,h)
        
        # Leaf node... which should be a widget
        if type(my_node) == tuple:
            placement[so_far+'|'+my_node[0]] = (x+widget_h_gap,y+widget_v_gap,w-2*widget_h_gap,h-2*widget_v_gap)

        # Leaf node... which is a tuple with only one element.. i.e., just the element
        elif type(my_node) == str:
            placement[so_far+'|'+my_node] = (x+widget_h_gap,y+widget_v_gap,w-2*widget_h_gap,h-2*widget_v_gap)
            
        # Interior node - requires processing the layout type and allocating spacing basd on the layout type 
        else:
            child_layout = self.__layoutType(next(iter(my_node.keys())))

            #
            # Border Layout
            #
            if   child_layout == 'borderLayout':
                dim_i = 0 # Use minimum dimensions by default
                if w < leaf_dims[so_far][1][0] and h < leaf_dims[so_far][1][1]:
                    dim_i = 1 # Okay to use preferred dimensions

                # Get the minimum (or preferred dimensions) for each cardinal tile
                c_dim = n_dim = s_dim = e_dim = w_dim = (0,0)
                if 'c' in my_node.keys():
                    c_dim = leaf_dims[so_far+'|c'][dim_i]
                if 'n' in my_node.keys():
                    n_dim = leaf_dims[so_far+'|n'][dim_i]
                if 's' in my_node.keys():
                    s_dim = leaf_dims[so_far+'|s'][dim_i]
                if 'e' in my_node.keys():
                    e_dim = leaf_dims[so_far+'|e'][dim_i]
                if 'w' in my_node.keys():
                    w_dim = leaf_dims[so_far+'|w'][dim_i]

                # The center gets any left overs... (if there's a center)
                if 'c' in my_node.keys():
                    # Give the center the left overs
                    c_dim = (w - (e_dim[0] + w_dim[0]), h - (s_dim[1] + n_dim[1]))

                    # Adjust everyone around the center...
                    if n_dim[0] > 0:
                        n_dim = (w, n_dim[1])
                    if s_dim[0] > 0:
                        s_dim = (w, s_dim[1])
                    if e_dim[1] > 0:
                        e_dim = (e_dim[0], c_dim[1])
                    if w_dim[1] > 0:
                        w_dim = (w_dim[0], c_dim[1])

                # Just north, just south, or just north and south? // no east, no west
                elif ('n' in my_node.keys() or 's' in my_node.keys()) and ('e' not in my_node.keys()) and ('w' not in my_node.keys()):
                    if    'n' in my_node.keys() and 's' in my_node.keys():
                        sum_h = leaf_dims[so_far + '|n'][dim_i][1] + leaf_dims[so_far + '|s'][dim_i][1]
                        n_dim = (w, h * n_dim[1]/sum_h)
                        s_dim = (w, h * s_dim[1]/sum_h)
                    elif  'n' in my_node.keys():
                        n_dim = (w, h)
                    else:
                        s_dim = (w, h)

                # Just east, just west, or just east and west? // no north, no south
                elif ('w' in my_node.keys() or 'e' in my_node.keys()) and ('n' not in my_node.keys()) and ('s' not in my_node.keys()):
                    if    'w' in my_node.keys() and 'e' in my_node.keys():
                        sum_w = leaf_dims[so_far + '|w'][dim_i][0] + leaf_dims[so_far + '|e'][dim_i][0]
                        w_dim = (w * w_dim[0]/sum_w, h)
                        e_dim = (w * e_dim[0]/sum_w, h)
                    elif  'w' in my_node.keys():
                        w_dim = (w, h)
                    else:
                        e_dim = (w, h)

                # have at least one in both directions...  everyone gets their minimum or preferred
                # -- left overs in height get allocated to the east and west
                # -- east and west (if they are both in existence, have to split the ratio
                else:
                    sum_w = 0
                    if 'w' in my_node.keys():
                        sum_w += leaf_dims[so_far + '|w'][dim_i][0]
                    if 'e' in my_node.keys():
                        sum_w += leaf_dims[so_far + '|e'][dim_i][0]

                    sum_h = 0
                    if 'n' in my_node.keys():
                        sum_h += leaf_dims[so_far + '|n'][dim_i][1]
                    if 's' in my_node.keys():
                        sum_h += leaf_dims[so_far + '|s'][dim_i][1]


                    if    'n' in my_node.keys() and 's' in my_node.keys():
                        n_dim = (w, n_dim[1])
                        s_dim = (w, s_dim[1])
                    elif  'n' in my_node.keys():
                        n_dim = (w, sum_h)
                    else:
                        s_dim = (w, sum_h)

                    if    'w' in my_node.keys() and 'e' in my_node.keys():
                        w_dim = (w * w_dim[0]/sum_w, h - sum_h)
                        e_dim = (w * e_dim[0]/sum_w, h - sum_h)
                    elif  'w' in my_node.keys():
                        w_dim = (w, h - sum_h)
                    else:
                        e_dim = (w, h - sum_h)


                if 'n' in my_node.keys():
                    my_x = x+h_gap
                    my_y = y+v_gap
                    self.__finalizeWidgetPlacement(my_node['n'], leaf_dims, placement, my_x, my_y, n_dim[0] - 2*h_gap, n_dim[1] - 2*v_gap, h_gap, v_gap, widget_h_gap, widget_v_gap, so_far + "|n")

                if 'w' in my_node.keys():
                    my_x = x + h_gap
                    my_y = y + n_dim[1] + v_gap
                    self.__finalizeWidgetPlacement(my_node['w'], leaf_dims, placement, my_x, my_y, w_dim[0] - 2*h_gap, w_dim[1] - 2*v_gap, h_gap, v_gap, widget_h_gap, widget_v_gap, so_far + "|w")

                if 'c' in my_node.keys():
                    my_x = x + w_dim[0] + h_gap
                    my_y = y + n_dim[1] + v_gap
                    self.__finalizeWidgetPlacement(my_node['c'], leaf_dims, placement, my_x, my_y, c_dim[0] - 2*h_gap, c_dim[1] - 2*v_gap, h_gap, v_gap, widget_h_gap, widget_v_gap, so_far + "|c")

                if 'e' in my_node.keys():
                    my_x = x + w_dim[0] + c_dim[0] + h_gap
                    my_y = y + n_dim[1] + v_gap
                    self.__finalizeWidgetPlacement(my_node['e'], leaf_dims, placement, my_x, my_y, e_dim[0] - 2*h_gap, e_dim[1] - 2*v_gap, h_gap, v_gap, widget_h_gap, widget_v_gap, so_far + "|e")

                if 's' in my_node.keys():
                    my_x = x
                    my_y = y + n_dim[1] + max(e_dim[1],c_dim[1],w_dim[1]) + v_gap
                    self.__finalizeWidgetPlacement(my_node['s'], leaf_dims, placement, my_x, my_y, s_dim[0] - 2*h_gap, s_dim[1] - 2*v_gap, h_gap, v_gap, widget_h_gap, widget_v_gap, so_far + "|s")

            #
            # Grid Layout
            #
            elif child_layout == 'gridLayout':
                # Determine the number of tiles in both x and y
                x_tiles = y_tiles = 1
                for key in my_node.keys():
                    if int(key.split(',')[0]) > x_tiles:
                        x_tiles = int(key.split(',')[0])
                    if int(key.split(',')[1]) > y_tiles:
                        y_tiles = int(key.split(',')[1])

                # Check the min dimensions... make it at least the mins...
                if w  < leaf_dims[so_far][0][0]:
                    w  = leaf_dims[so_far][0][0]
                if h < leaf_dims[so_far][0][1]:
                    h = leaf_dims[so_far][0][1]

                # Give each tile the same amount of space...
                tile_width  = w/x_tiles
                tile_height = h/y_tiles

                # Place the tiles and the subtree
                for x_tile in range(0,x_tiles):
                    for y_tile in range(0,y_tiles):
                        key = str(x_tile+1) + ',' + str(y_tile+1)
                        if key in my_node.keys():
                            self.__finalizeWidgetPlacement(my_node[key], leaf_dims, 
                                                           placement, 
                                                           x + x_tile*tile_width + h_gap, y + y_tile*tile_height + v_gap, 
                                                           tile_width-2*h_gap, tile_height-2*v_gap, 
                                                           h_gap, v_gap, widget_h_gap, widget_v_gap, so_far + "|" + key)

            #
            # Flow Layout (Horizontal)
            #
            elif child_layout == 'flowLayoutHorizontal':
                dim_i = 0 # use minimum dimensions
                if leaf_dims[so_far][1][0] < w:
                    dim_i = 1 # use preferred dimensions

                # determine the sum of dimension needed (for ratio)
                sum_w   = 0
                to_sort = []
                for key in my_node.keys():
                    to_sort.append(int(key))
                    sum_w += leaf_dims[so_far + "|" + key][dim_i][0]

                to_sort = sorted(to_sort)
                x_inc = x
                for i in range(len(to_sort)-1,-1,-1):
                    key = so_far + "|" + str(to_sort[i])
                    w_ratio = leaf_dims[key][dim_i][0] / sum_w
                    self.__finalizeWidgetPlacement(my_node[str(to_sort[i])], leaf_dims, placement, x_inc + h_gap, y + v_gap, w_ratio*w - 2*h_gap, h - 2*v_gap, 
                                                   h_gap, v_gap, widget_h_gap, widget_v_gap, key)
                    x_inc += w_ratio*w

            #
            # Flow Layout (Vertical)
            # - MOSTLY A COPY OF FLOW LAYOUT (HORIZONTAL) WITH TRANSPOSE
            #
            elif child_layout == 'flowLayoutVertical':
                dim_i = 0 # use minimum dimensions
                if leaf_dims[so_far][1][1] < h:
                    dim_i = 1 # use preferred dimensions

                # determine the sum of dimension needed (for ratio)
                sum_h   = 0
                to_sort = []
                for key in my_node.keys():
                    to_sort.append(int(key))
                    sum_h += leaf_dims[so_far + "|" + key][dim_i][1]

                to_sort = sorted(to_sort)
                y_inc = y
                for i in range(0,len(to_sort)):
                    key = so_far + "|" + str(to_sort[i])
                    h_ratio = leaf_dims[key][dim_i][1] / sum_h
                    self.__finalizeWidgetPlacement(my_node[str(to_sort[i])], leaf_dims, placement, x + h_gap, y_inc + v_gap, w - 2*h_gap, h_ratio*h - 2*v_gap, 
                                                   h_gap, v_gap, widget_h_gap, widget_v_gap, key)
                    y_inc += h_ratio*h

            #
            # Error...
            #
            else:
                raise Exception(f'unknown layout mnemonic "{my_node}" / "{child_layout}"')

    #
    # Create a spatial dimension lookup based on a dashboard specification (as a dictionary)
    #
    def dictionaryLayoutToSpatialDimensions(self, spec, w, h, h_gap=0, v_gap=0, widget_h_gap=1, widget_v_gap=1):
        # Obtain the minimum and preferred dimensions dynamically
        dims = {}
        for widget in self.widgets:
            dims[widget] = {}
            dims[widget]['pref'] = eval(f'self.{widget}PreferredDimensions()')
            dims[widget]['min']  = eval(f'self.{widget}MinimumDimensions()')
        
        # Obtain the minimum and preferred dimensions dynamically // new version passes params to the preferred dimension methods
        for k,v in spec.items():
            widget = v
            if len(v) == 2:
                widget        = v[0]
            widget_params = {}
            if len(v) == 2:
                widget_params = v[1]
            dims[str(v)]         = {}
            dims[str(v)]['pref'] = eval(f'self.{widget}PreferredDimensions(**widget_params)')
            dims[str(v)]['min']  = eval(f'self.{widget}MinimumDimensions(**widget_params)')
        
        # Create the layout tree -- fill in the leaves with the widget dimensions
        # - layout_tree[layout_mnemonic] -> [child_layout_mnemonic] -> ... -> widget description
        layout_tree = {}
        for key in spec.keys():
            value = spec[key]
            self.__recursivePlace(layout_tree, key, value)

        # Calculate the minimum and preferred size for the interior nodes of the tree
        # - leaf_dims[<original_keys_from_the_spec>] = ((minimum dimensions - x,y),(preferred dimensions - x,y))
        leaf_dims = {}
        self.__fillInteriorNodeDimensions(layout_tree,leaf_dims,dims)

        # Finalize the placement -- reconcile either the preferred (requested) with the minimum widget size
        placement = {}
        self.__finalizeWidgetPlacement(layout_tree, leaf_dims, placement, 0, 0, w, h, h_gap, v_gap, widget_h_gap, widget_v_gap)

        # return placement,layout_tree,leaf_dims # Used for debugging the internal datastructures
        return placement

    #
    # Create SVG for the placement data structure
    # - To debug the placement really...
    #
    def placementSVGDebug(self, placement):
        # Start the svg object
        svg =  f'<svg width="{placement["root"][2]+1}" height="{placement["root"][3]+1}" xmlns="http://www.w3.org/2000/svg">'

        for key in placement.keys():
            # Ignore the widgets themselves... i.e., just show the placement positions
            render = True
            for widget in self.widgets:
                if widget in key:
                    render = False
            if render:
                coords = placement[key]
                svg += f'<rect width="{coords[2]}" height="{coords[3]}" x="{coords[0]}" y="{coords[1]}" stroke="#000000" fill-opacity="0.0" />'
                svg += f'<text x="{coords[0] + coords[2]/2}" y="{coords[1] + coords[3]/2}" text-anchor="middle">{key}</text>'

        # Close out the svg object
        svg += '</svg>'
        return svg
    
    #
    # Generic layout method -- will choose proper layout method...
    #
    def layout(self,
               spec,                                # Multiwidget specification
               df,                                  # Dataframe to render
               #------------------------------------#
               widget_id        = None,             # Widget ID
               #------------------------------------#
               w                = 1024,             # Width of the multi-widget panel
               h                = 1024,             # Height of the multi-widget panel
               h_gap            = 0,                # Horizontal left/right gap
               v_gap            = 0,                # Verticate top/bottom gap
               widget_h_gap     = 1,                # Horizontal gap between widgets
               widget_v_gap     = 1,                # Vertical gap between widgets
               track_state      = False,            # Track state of geometry to data frame // for Panel
               rt_reactive_html = None,             # Reference to Reactive HTML Panel // for Panel
               **kwargs):
        # Determine type of specification...
        str_count,tup_count,unk_count = 0,0,0
        for k in spec.keys():
            if type(k) == str:
                str_count += 1
            elif type(k) == tuple:
                tup_count += 1
            else:
                unk_count += 1
        if    str_count >  0 and tup_count == 0 and unk_count == 0:
            return self.multiWidgetPanel(df,spec,widget_id,w,h,h_gap,v_gap,widget_h_gap,widget_v_gap,track_state,rt_reactive_html,**kwargs)
        elif  str_count == 0 and tup_count >  0 and unk_count == 0:
            return self.gridBagLayout   (df,spec,widget_id,w,h,h_gap,v_gap,widget_h_gap,widget_v_gap,track_state,rt_reactive_html,**kwargs)
        else:
            raise Exception(f'rt.layout() failed to recognize specification type {str_count}/{tup_count}/{unk_count}')

    #
    # Create the SVG multipanel widget based on the spec, the dataframe, and the dynamic variables
    #
    def multiWidgetPanel(self,
                         df,                                  # Dataframe to render
                         spec,                                # Multiwidget specification
                         #------------------------------------#
                         widget_id        = None,             # Widget ID
                         #------------------------------------#
                         w                = 1024,             # Width of the multi-widget panel
                         h                = 1024,             # Height of the multi-widget panel
                         h_gap            = 0,                # Horizontal left/right gap
                         v_gap            = 0,                # Verticate top/bottom gap
                         widget_h_gap     = 1,                # Horizontal gap between widgets
                         widget_v_gap     = 1,                # Vertical gap between widgets
                         track_state      = False,            # Track state of geometry to data frame // for Panel 
                         rt_reactive_html = None,             # Reference to Reactive HTML Panel // for Panel
                         **kwargs):
        # Widget ID
        if widget_id is None:
            widget_id = 'multiwidget_panel_' + str(random.randint(0,65535))

        # Calculate the placement
        placement = self.dictionaryLayoutToSpatialDimensions(spec, w, h, h_gap, v_gap, widget_h_gap, widget_v_gap)

        # General application calculations
        # ... time granularity
        widgets_set = set()
        for k,v in spec.items():
            if type(v) == tuple:
                widgets_set.add(v[0])
            else:
                widgets_set.add(v)

        if ('temporal_granularity' not in kwargs.keys()) and 'temporalBarChart' in widgets_set:
            if ('ts_field' not in kwargs.keys()) or kwargs['ts_field'] is None:
                ts_field = self.guessTimestampField(df)
            else:
                ts_field = kwargs['ts_field']
            temporal_granularity = self.temporalGranularity(df, ts_field)

        # Go through the placement, identify the placement keys that are widgets... fill them in... then render them...
        _instance_lu = {}
        for place in placement.keys():
            # Only keep the widgets
            render = False
            for widget in self.widgets:
                if widget in place:
                    render = True
                    
            # For the widgets... create the custom svg call and add to the svg element
            if render:
                spec_key   = place[place.index('|')+1:place.rindex('|')]
                spec_tuple = spec[spec_key]
                widget_loc = placement[place]
                
                # Create the custom params for the widget creation
                my_params = kwargs.copy()
                my_params['df']          = df
                my_params['x_view']      = widget_loc[0]
                my_params['y_view']      = widget_loc[1]
                my_params['w']           = widget_loc[2]
                my_params['h']           = widget_loc[3]
                my_params['track_state'] = track_state
                
                # If the spec_tuple is actually a tuple, they copy those parts into the params as well
                # - these should override any passed from this method (the kwargs.copy())
                if type(spec_tuple) == tuple:
                    widget_method = spec_tuple[0]
                    for k in spec_tuple[1].keys():
                        v = spec_tuple[1][k]
                        my_params[k] = v
                else:
                    widget_method = spec_tuple

                # Need to remove any args that aren't applicable to this widget
                accepted_args = set(inspect.getfullargspec(getattr(self, widget_method)).args)
                to_remove = set()
                for k in my_params.keys():
                    if k not in accepted_args:
                        to_remove.add(k)
                for x in to_remove:
                    my_params.pop(x)

                # General application parameters
                if 'temporal_granularity' in accepted_args and 'temporal_granularity' not in my_params:
                    my_params['temporal_granularity'] = temporal_granularity
                if 'rt_reactive_html' in accepted_args:
                    my_params['rt_reactive_html'] = rt_reactive_html

                # Resolve the method name and invoke it adding to the svg string
                _func      = getattr(self, widget_method)
                _instance  = _func(**my_params)

                _px0,_py0,_px1,_py1 = my_params['x_view'], my_params['y_view'], my_params['x_view']+my_params['w'], my_params['y_view']+my_params['h']
                _instance_bounds = Polygon([[_px0,_py0],[_px0,_py1],[_px1,_py1],[_px1,_py0]])
                _instance_lu[_instance_bounds] = _instance

        return RTComponentsLayout(self, widget_id, _instance_lu, w, h)

    #
    # Create the SVG multipanel widget using a gridbag-like layout method.
    # ... spec --> spec[x,y,w,h - tuple] -> ('component',{'param1':'value1', ... })
    #
    def gridBagLayout(self,
                      df,                                    # Dataframe to render
                      spec,                                  # Multiwidget specification
                      #--------------------------------------#
                      widget_id        = None,               # Widget ID
                      #--------------------------------------#
                      w                = 1024,               # Width of the multi-widget panel
                      h                = 1024,               # Height of the multi-widget panel
                      h_gap            = 0,                  # Horizontal left/right gap
                      v_gap            = 0,                  # Verticate top/bottom gap
                      widget_h_gap     = 1,                  # Horizontal gap between widgets
                      widget_v_gap     = 1,                  # Vertical gap between widgets
                      track_state      = False,              # Track the state of the dataframe to geometry // for Panel
                      rt_reactive_html = None,               # Reference to Reactive HTML Panel // for Panel
                      **kwargs):
        # Widget ID
        if widget_id is None:
            widget_id = 'panel_' + str(random.randint(0,65535))
        
        # Validate the layout first
        # - Determine the max coordinate
        widgets_set = set()
        _tile_x_max,_tile_y_max = 1,1
        for xywh_tuple in spec.keys():
            _x0,_y0,_tile_w,_tile_h = xywh_tuple
            if (_x0 + _tile_w) > _tile_x_max:
                _tile_x_max = _x0 + _tile_w
            if (_y0 + _tile_h) > _tile_y_max:
                _tile_y_max = _y0 + _tile_h
            if type(spec[xywh_tuple]) == tuple:
                widgets_set.add(spec[xywh_tuple][0])
            else:
                widgets_set.add(spec[xywh_tuple])
        # - Allocate a representation
        cells = [[0 for i in range(_tile_x_max)] for j in range(_tile_y_max)] # cells[y][x]
        for _y in range(_tile_y_max):
            for _x in range(_tile_x_max):
                cells[_y][_x] = -1
        # - Place tuples into representation - checking to make sure there isn't overlap
        for xywh_tuple in spec.keys():
            _x0,_y0,_tile_w,_tile_h = xywh_tuple
            for _x in range(_x0,_x0+_tile_w):
                for _y in range(_y0,_y0+_tile_h):
                    if cells[_y][_x] != -1:
                        raise Exception(f'gridBagLayout() - overlapping coordinate @ {_x},{_y}')
                    cells[_y][_x] = xywh_tuple
        
        # Determine the tile size -- make it into actual pixels... x-pixels-per-tile (xppt)
        _xppt,_yppt = ceil((w-2*h_gap)/_tile_x_max),ceil((h-2*v_gap)/_tile_y_max)
        w,h         = _xppt * _tile_x_max, _yppt * _tile_y_max

        # Determine temporal granularity if there's a temporalBarChart...
        if ('temporal_granularity' not in kwargs.keys()) and 'temporalBarChart' in widgets_set:
            if ('ts_field' not in kwargs.keys()) or kwargs['ts_field'] is None:
                ts_field = self.guessTimestampField(df)                
            else:
                ts_field = kwargs['ts_field']
            temporal_granularity = self.temporalGranularity(df, ts_field)
        else:
            temporal_granularity = None

        # Start the SVG
        _instance_lu = {}
        for xywh_tuple in spec.keys():
            _x0,_y0,_tile_w,_tile_h = xywh_tuple
            spec_tuple = spec[xywh_tuple]
                
            # Create the custom params for the widget creation
            my_params = kwargs.copy()
            my_params['df']          = df
            my_params['x_view']      = h_gap + _x0 * _xppt + widget_h_gap
            my_params['y_view']      = v_gap + _y0 * _yppt + widget_v_gap
            my_params['w']           = _tile_w * _xppt - 2 * widget_h_gap
            my_params['h']           = _tile_h * _yppt - 2 * widget_v_gap
            my_params['track_state'] = track_state
                
            # If the spec_tuple is actually a tuple, then copy those parts into the params as well
            # - these should override any passed from this method (the kwargs.copy())
            if type(spec_tuple) == tuple:
                widget_method = spec_tuple[0]
                for k in spec_tuple[1].keys():
                    v = spec_tuple[1][k]
                    my_params[k] = v
            else:
                widget_method = spec_tuple

            # Need to remove any args that aren't applicable to this widget
            accepted_args = set(inspect.getfullargspec(getattr(self, widget_method)).args)
            to_remove = set()
            for k in my_params.keys():
                if k not in accepted_args:
                    to_remove.add(k)
            for x in to_remove:
                my_params.pop(x)

            # General application parameters
            if 'temporal_granularity' in accepted_args and 'temporal_granularity' not in my_params:
                my_params['temporal_granularity'] = temporal_granularity
            if 'rt_reactive_html' in accepted_args:
                my_params['rt_reactive_html'] = rt_reactive_html

            # Resolve the method name and invoke it adding to the svg string
            _func      = getattr(self, widget_method)
            _instance  = _func(**my_params)
            
            _px0,_py0,_px1,_py1 = my_params['x_view'], my_params['y_view'], my_params['x_view']+my_params['w'], my_params['y_view']+my_params['h']
            _instance_bounds = Polygon([[_px0,_py0],[_px0,_py1],[_px1,_py1],[_px1,_py0]])
            _instance_lu[_instance_bounds] = _instance

        return RTComponentsLayout(self, widget_id, _instance_lu, w, h)

    #
    # panelControlPreferredDimensions() - preferred dimensions for the panel control panel
    #
    def panelControlPreferredDimensions(self, **kwargs):
        return (256, 24)

    #
    # panelControlMinimumDimensions() - minimum dimensions for the panel control panel
    #
    def panelControlMinimumDimensions(self, **kwargs):
        return (64, 20)

    #
    # panelControl() - return instance of the panel control panel
    #
    def panelControl(self,
                     df=None,
                     rt_reactive_html=None,
                     txt_h=None,
                     x_view=0,
                     y_view=0,
                     x_ins=2,
                     y_ins=2,
                     w=128,
                     h=20):
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTPanelControl(self, **_params_)

    #
    # RTPanelControl() - return instance of the panel control panel
    #
    class RTPanelControl(RTComponent):
        #
        # Constructor
        #
        def __init__(self, rt_self, **kwargs):
            self.rt_self          = rt_self
            self.df               = kwargs['df']
            self.x_view           = kwargs['x_view']
            self.y_view           = kwargs['y_view']
            self.w                = kwargs['w']
            self.h                = kwargs['h']
            self.x_ins            = kwargs['x_ins']
            self.y_ins            = kwargs['y_ins']
            self.rt_reactive_html = kwargs['rt_reactive_html']
            self.last_render      = None                         # For Compatibility w/ RTComponent Usage
            self.geom_to_df       = {}

            if 'txt_h' in kwargs.keys() and kwargs['txt_h'] is not None:
                self.txt_h = kwargs['txt_h']
            else:
                self.txt_h = self.h - 2 * self.y_ins
                if self.txt_h > 24:
                    self.txt_h = 24

        #
        # _repr_svg_() - SVG Representation
        # - always re-render (do not keep the last render around)
        # - this method is unique across the RTComponent class since last_render is never used
        #
        def _repr_svg_(self):
            return self.renderSVG()
        
        #
        # renderSVG() - render the SVG representation of the panel control
        #
        def renderSVG(self):
            self.geom_to_df = {}
            _svg_ = f'<svg id="panel_control" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            bg_color = self.rt_self.co_mgr.getTVColor('background','default')
            _svg_ += f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="{bg_color}" stroke="{bg_color}" stroke-width="1"/>'
            _svg_ += self.rt_self.svgText('RT', self.x_ins, self.y_ins + self.txt_h, self.txt_h)
            rt_l   = self.rt_self.textLength('RT', self.txt_h) + 2 * self.x_ins
            # If an RT Reactive HTML component was specified, draw information about the state of the panel
            if self.rt_reactive_html is not None:
                # Blocks to indicate the depth
                _depth_     = self.rt_reactive_html.df_level + 1
                _blocks_    = len(self.rt_reactive_html.dfs)
                block_color = self.rt_self.co_mgr.getTVColor('data','default')
                block_gap   = 2
                block_w     = (self.h - 2 * self.y_ins) if self.h < 24 else 20
                for xi in range(_blocks_):
                    _fill_   = block_color if xi < _depth_ else 'none'
                    x = self.x_ins + rt_l + xi * (block_w + block_gap)
                    _svg_ += f'<rect x="{x}" y="{self.y_ins}" width="{block_w}" height="{block_w}" fill="{_fill_}" stroke="{block_color}" stroke-width="1"/>'
                    _poly_ = Polygon([[x, self.y_ins],[x, self.y_ins+block_w],[x+block_w, self.y_ins+block_w],[x+block_w, self.y_ins]])
                    self.geom_to_df[_poly_] = self.rt_reactive_html.dfs[xi]
                # Text string to indicate the size of the dataframe
                if self.df is not None:
                    _readable_ = self.rt_self.readable(len(self.df))
                    _svg_ += self.rt_self.svgText(_readable_, self.x_ins + rt_l + _blocks_*(block_w+block_gap) + block_gap, self.y_ins + self.txt_h, self.txt_h)
            _svg_ += '</svg>'
            return _svg_

        #
        # overlappingDataFrames() - Determine which dataframe geometris overlap with a specific one
        #
        def overlappingDataFrames(self, to_intersect):
            _dfs = []
            for _poly in self.geom_to_df.keys():
                if _poly.intersects(to_intersect):
                    _dfs.append(self.geom_to_df[_poly])
            if len(_dfs) > 0:
                return _dfs[0]
            else:
                return None

#
# RT Layout Instance (new version)
#
class RTComponentsLayout(RTComponent):
    #
    # Constructor
    #
    def __init__(self, rt_self, widget_id, instance_lu, w, h):
        self.rt_self     = rt_self
        self.widget_id   = widget_id
        self.instance_lu = instance_lu
        self.w           = w
        self.h           = h
        self.last_render = None
    
    #
    # SVG Representation
    #
    def _repr_svg_(self):
        if self.last_render is None:
            self.renderSVG()
        return self.last_render
    
    #
    # renderSVG() - render the SVG representation
    #
    def renderSVG(self):
        _svg_ = [f'<svg id="{self.widget_id}" width="{self.w+1}" height="{self.h+1}" xmlns="http://www.w3.org/2000/svg">']
        for _poly in self.instance_lu.keys():
            _svg_.append(self.instance_lu[_poly]._repr_svg_())
        _svg_.append('</svg>')
        self.last_render = ''.join(_svg_)
        return self.last_render

    #
    # applyViewConfigurations() - apply view configurations based on a reference layout.
    #
    def applyViewConfigurations(self, ref_layout):
        for _poly_ in self.instance_lu.keys():
            self.instance_lu[_poly_].applyViewConfiguration(ref_layout.instance_lu[_poly_])

    #
    # identifyComponent() - identify the component that the coordinate falls within
    # - return the component, the key for this component, and the modified/relative coordinate adjusted to the component
    #
    def identifyComponent(self, coordinate):
        for _poly_ in self.instance_lu.keys():
            bnds = _poly_.bounds
            if coordinate[0] >= bnds[0] and coordinate[0] <= bnds[2] and coordinate[1] >= bnds[1] and coordinate[1] <= bnds[3]:
                return self.instance_lu[_poly_], _poly_, (coordinate[0] - bnds[0], coordinate[1] - bnds[1])
        return None, None, None

    #
    # componentInstance() - return the component instance for the specified key
    # - used in conjuction with the key returned in identifyComponent()
    # - and should work across layers in the interactive stack
    #
    def componentInstance(self, k):
        return self.instance_lu[k] if k in self.instance_lu.keys() else None

    #
    # entityPositions() - return information about the entity geometry for rendering
    def entityPositions(self, entity):
        all_positions = []
        for _poly_ in self.instance_lu.keys():
            _instance_  = self.instance_lu[_poly_]
            _positions_ = _instance_.entityPositions(entity)
            if _positions_ is not None and len(_positions_) > 0:
                for _pos_ in _positions_:
                    _pos_.xyOffset((_poly_.bounds[0], _poly_.bounds[1]))
                    all_positions.append(_pos_)
        return all_positions

    #
    # overlappingDataFrames() - Return Overlapping Dataframes
    #
    def overlappingDataFrames(self, to_intersect):
        if type(to_intersect) == tuple: # assume it's a bounding box (x0,y0,x1,y1)
            _x0,_y0,_x1,_y1 = to_intersect
            if _x0 > _x1:
                _x0,_x1 = _x1,_x0
            if _y0 > _y1:
                _y0,_y1 = _y1,_y0
            to_intersect = Polygon([[_x0,_y0],[_x0,_y1],[_x1,_y1],[_x1,_y0]])
        _dfs = []
        for _poly in self.instance_lu.keys():
            if _poly.intersects(to_intersect):
                adj_to_intersect = translate(to_intersect, -_poly.bounds[0], -_poly.bounds[1])
                _df = self.instance_lu[_poly].overlappingDataFrames(adj_to_intersect)
                if _df is not None and len(_df) > 0:
                    _dfs.append(_df)
        if len(_dfs) > 0:
            return self.rt_self.concatDataFrames(_dfs)
        else:
            return None
