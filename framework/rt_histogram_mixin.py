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
import numpy as np
import random

from shapely.geometry import Polygon
from math import sqrt

from rt_component import RTComponent

__name__ = 'rt_histogram_mixin'

#
# Histogram Mixin
#
class RTHistogramMixin(object):
    #
    # histogramPreferredSize()
    # - Return the preferred size
    #
    def histogramPreferredDimensions(self, **kwargs):
        return (160,256)

    #
    # histogramMinimumSize()
    # - Return the minimum size
    #
    def histogramMinimumDimensions(self, **kwargs):
        return (64,64)

    #
    # histogramSmallMultipleSize()
    # - Return the minimum size
    #
    def histogramSmallMultipleDimensions(self, **kwargs):
        return (32,32)

    #
    # histogramRequiredFields()
    # - Return the required fields for a histogram configuration
    #
    def histogramRequiredFields(self, **kwargs):
        columns = set()
        self.identifyColumnsFromParameters('bin_by',   kwargs, columns)
        self.identifyColumnsFromParameters('color_by', kwargs, columns)
        self.identifyColumnsFromParameters('count_by', kwargs, columns)
        return columns

    #
    # histogram
    #
    # Make the SVG for a histogram from a dataframe
    #    
    def histogram(self,
                  df,                         # dataframe to render
                  bin_by,                     # string or an array of strings                  
                  # ------------------------- # everything else is a default...                  
                  color_by           = None,   # just the default color or a string for a field
                  global_color_order = None,   # color by ordering... if none (default), will be created and filled in...                  
                  count_by           = None,   # none means just count rows, otherwise, use a field to sum by
                  count_by_set       = False,  # count by using a set operation                  
                  widget_id          = None,   # naming the svg elements
                  # -------------------------- # global rendering params
                  first_line_i       = 0,      # first line index to render
                  global_max         = None,   # maximum to use for the bar length calculation
                  just_calc_max      = False,  # forces return of the maximum for this render config...
                                               # ... which will then be used for the global max across bar charts...                        
                  # ------------------------- # rendering specific params                  
                  x_view=0,                   # x offset for the view
                  y_view=0,                   # y offset for the view
                  w=128,                      # width of the view
                  h=256,                      # height of the view
                  bar_h=14,                   # bar height
                  v_gap=0,                    # gap between bars
                  draw_labels=True,           # draw labels flag
                  draw_border=True            # draw a border around the histogram
                 ):
        rt_histogram = self.RTHistogram(self, df, bin_by, color_by=color_by, global_color_order=global_color_order,
                                        count_by=count_by,count_by_set=count_by_set,widget_id=widget_id, first_line_i=first_line_i,
                                        global_max=global_max,x_view=x_view,y_view=y_view,w=w,h=h,bar_h=bar_h,v_gap=v_gap,
                                        draw_labels=draw_labels,draw_border=draw_border)
        # Calculate max
        if just_calc_max:
            return rt_histogram.renderSVG(True)
        
        # Render SVG
        else:
            return rt_histogram.renderSVG()

    #
    # histogramInstance()
    # - create a RTHistogram object
    #    
    def histogramInstance(self,
                          df,                          # dataframe to render
                          bin_by,                      # string or an array of strings
                          # -------------------------  #  everything else is a default...
                          color_by           = None,   # just the default color or a string for a field
                          global_color_order = None,   # color by ordering... if none (default), will be created and filled in...
                          count_by           = None,   # none means just count rows, otherwise, use a field to sum by
                          count_by_set       = False,  # count by using a set operation
                          widget_id          = None,   # naming the svg elements
                          # -------------------------  # global rendering params
                          first_line_i       = 0,      # first line index to render
                          global_max         = None,   # maximum to use for the bar length calculation
                          # -------------------------  # rendering specific params
                          x_view             = 0,      # x offset for the view
                          y_view             = 0,      # y offset for the view
                          w                  = 128,    # width of the view
                          h                  = 256,    # height of the view
                          bar_h              = 14,     # bar height
                          v_gap              = 0,      # gap between bars
                          draw_labels        = True,   # draw labels flag
                          draw_border        = True):  # draw a border around the histogram
        return self.RTHistogram(self, df, bin_by, color_by=color_by, global_color_order=global_color_order,
                                 count_by=count_by,count_by_set=count_by_set,widget_id=widget_id, first_line_i=first_line_i,
                                 global_max=global_max,x_view=x_view,y_view=y_view,w=w,h=h,bar_h=bar_h,v_gap=v_gap,
                                 draw_labels=draw_labels,draw_border=draw_border)

    #
    # RTHistogram Class
    #
    class RTHistogram(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     df,                         # dataframe to render
                     bin_by,                     # string or an array of strings                  
                     # ------------------------- # everything else is a default...
                     color_by           = None,  # just the default color or a string for a field
                     global_color_order = None,  # color by ordering... if none (default), will be created and filled in...
                     count_by           = None,  # none means just count rows, otherwise, use a field to sum by
                     count_by_set       = False, # count by using a set operation
                     widget_id          = None,  # naming the svg elements
                     # ------------------------- # global rendering params
                     first_line_i       = 0,     # first line index to render
                     global_max         = None,  # maximum to use for the bar length calculation
                     # ------------------------- # rendering specific params
                     x_view             = 0,     # x offset for the view
                     y_view             = 0,     # y offset for the view
                     w                  = 128,   # width of the view
                     h                  = 256,   # height of the view
                     bar_h              = 14,    # bar height
                     v_gap              = 0,     # gap between bars
                     draw_labels        = True,  # draw labels flag
                     draw_border        = True): # draw a border around the histogram
            self.parms              = locals().copy()
            self.rt_self            = rt_self
            self.df                 = df.copy()

            # Make sure the bin_by is a list...
            if type(bin_by) != list: # Make it into a list for consistency
                self.bin_by = [bin_by]
            else:
                self.bin_by = bin_by

            self.color_by           = color_by
            self.global_color_order = global_color_order
            self.count_by           = count_by
            self.count_by_set       = count_by_set

            # Make a histogram_id if it's not set already
            if widget_id is None:
                self.widget_id = "histogram_" + str(random.randint(0,65535))
            else:
                self.widget_id          = widget_id

            self.first_line_i       = first_line_i
            self.global_max         = global_max
            self.x_view             = x_view
            self.y_view             = y_view
            self.w                  = w
            self.h                  = h
            self.bar_h              = bar_h
            self.v_gap              = v_gap
            self.draw_labels        = draw_labels
            self.draw_border        = draw_border

            # Apply bin-by transforms
            self.df, self.bin_by = rt_self.transformFieldListAndDataFrame(self.df, self.bin_by)
        
            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)

            # Check the count_by column
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)
            
            # Geometry lookup for tracking state
            self.geom_to_df = {}
            self.last_render = None
    
        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
            return self.last_render

        #
        # scrollView()
        # - scroll the list by the specified amount
        # - coordinate included to make it similar to other view functionality
        # ... looks like we don't know the list length... so we can't bound the calc by that...
        #
        def scrollView(self, scroll_amount, coordinate=None):
            if (self.first_line_i+scroll_amount) >= 0:
                self.first_line_i += scroll_amount
            else:
                self.first_line_i =  0
            self.last_render = None

        #
        # renderSVG() - create the SVG
        #
        def renderSVG(self, just_calc_max=False, track_state=False):
            # Leave space for a label
            max_bar_w = self.w - self.bar_h
                                            
            # Determine the color order (for each bar)
            if self.global_color_order is None:
                self.global_color_order = self.rt_self.colorRenderOrder(self.df, self.color_by, self.count_by, self.count_by_set)

            # Determine the bin order (for the bars)
            if    self.count_by == None:
                order = self.df.groupby(by=self.bin_by).size().sort_values(ascending=False)
            elif  self.count_by_set:
                if self.count_by in self.bin_by:
                    _df = self.df.groupby(by=self.bin_by).size().reset_index()
                    order = _df.groupby(by=self.bin_by).size().sort_values(ascending=False)
                else:
                    _combined =  self.bin_by.copy()
                    _combined.append(self.count_by)
                    _df = self.df.groupby(by=_combined).size().reset_index()
                    order = _df.groupby(by=self.bin_by).size().sort_values(ascending=False)
            else:
                if self.count_by in self.bin_by:
                    self.df['__count_by_copy__'] = self.df[self.count_by]
                    self.count_by_field = '__count_by_copy__'
                else:
                    self.count_by_field = self.count_by
                order = self.df.groupby(by=self.bin_by)[self.count_by_field].sum().sort_values(ascending=False)

            gb = self.df.groupby(self.bin_by)

            # If the height/width are less than the minimums, turn off labeling... and make the min_bar_w = 1
            # ... for small multiples
            params_orig_minus_self = self.parms.copy()
            params_orig_minus_self.pop('self')
            min_dims = self.rt_self.histogramSmallMultipleDimensions(**params_orig_minus_self)
            if self.w < min_dims[0] or self.h < min_dims[1]:
                self.draw_labels = False
                self.bar_h       = 2
                self.x_ins       = 1
                self.y_ins       = 1
                self.v_gap       = 0

            # Create the SVG ... render the background
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'
            
            textfg = self.rt_self.co_mgr.getTVColor('label','defaultfg')
            defer_labels = []

            # Determine the max bin size... make sure it isn't zero
            if self.global_max is None:
                max_group_by = order.iloc[0]
                if just_calc_max:
                    return 0,max_group_by
                if max_group_by == 0:
                    max_group_by = 1
            else:
                max_group_by = self.global_max
            
            #
            # Render each bin ... only do the visible ones...
            # ... make sure the first line isn't more than the number of lines...
            #
            i = self.first_line_i
            if i >= len(order):
                self.first_line_i = i = len(order) - 1
            if i <  0:
                self.first_line_i = i = 0
            y = 0
            while y < (self.h - 1.9*self.bar_h) and i < len(order):
                # Width of the bar in pixels
                px = max_bar_w * order.iloc[i] / max_group_by

                # Bin label... used for the id... and used for the labeling (if draw_labels is true)
                if type(order.index[i]) != list and type(order.index[i]) != tuple:
                    bin_text = str(order.index[i])
                else:
                    bin_text = ' | '.join([str(x) for x in order.index[i]])
                
                # Make a safe id to reference this element later
                element_id = self.widget_id + "_" + self.rt_self.encSVGID(bin_text)

                # Make the bar // even if we're going to color it in later
                color = self.rt_self.co_mgr.getTVColor('data','default')

                # Render the bar ... next section does the color... but this makes sure it's at least filled in...
                svg += f'<rect id="{element_id}" width="{px}" height="{self.bar_h}" x="0" y="{y}" fill="{color}" stroke="{color}"/>'
                if track_state:
                    self.geom_to_df[Polygon([[0,y],[px,y],[px,y+self.bar_h],[0,y+self.bar_h]])] = gb.get_group(order.index[i])

                # 'Color By' options
                if self.color_by is not None:
                    row_df = gb.get_group(order.index[i])
                    svg += self.rt_self.colorizeBar(row_df, self.global_color_order, self.color_by, self.count_by, self.count_by_set, 0, y, px, self.bar_h, True)

                # Render the label
                if self.draw_labels:
                    cropped_bin_text = self.rt_self.cropText(str(bin_text), self.bar_h-2, max_bar_w)
                    defer_labels.append(self.rt_self.svgText(cropped_bin_text, 2, y+self.bar_h-1, self.bar_h-2))
                
                i += 1
                y += self.bar_h+1+self.v_gap
            
            # Indicate how many more we are missing
            if self.draw_labels and i != len(order):
                svg += self.rt_self.svgText(f'{len(order)-i} more', 2, self.h-3, self.bar_h-1, color=self.rt_self.co_mgr.getTVColor('label','error'))
            
            # Draws the maximum amount of the histogram
            if self.draw_labels:
                # Draw deferred labels
                for _label in defer_labels:
                    svg += _label

                # Draw axes
                axis_co = self.rt_self.co_mgr.getTVColor('axis', 'default')
                svg += self.rt_self.svgText(str(max_group_by), max_bar_w-5, self.h-3, self.bar_h-2, anchor='end')
                svg += f'<line x1="{max_bar_w}" y1="{2}" x2="{max_bar_w}" y2="{self.h}" stroke="{axis_co}" stroke-width="1" stroke-dasharray="3 2" />'
                bin_by_str = '|'.join(self.bin_by)
                svg += self.rt_self.svgText(bin_by_str, max_bar_w+4, self.h/2, self.bar_h-2, anchor='middle', rotation=90)
            
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill-opacity="0.0" stroke="{border_color}" />'
            
            svg += '</svg>'
            self.last_render = svg
            return svg

        #
        # smallMultipleFeatureVector()
        # ... feature vector for comparison with other small multiple instances of this class
        # ... pretty much a copy of the render code above...
        #
        def smallMultipleFeatureVector(self):
            # Determine the bin order (for the bars)
            if    self.count_by == None:
                order = self.df.groupby(by=self.bin_by).size().sort_values(ascending=False)
            elif  self.count_by_set:
                if self.count_by in self.bin_by:
                    _df = self.df.groupby(by=self.bin_by).size()
                    order = _df.groupby(by=self.bin_by).size().sort_values(ascending=False)
                else:
                    _combined =  self.bin_by.copy()
                    _combined.append(self.count_by)
                    _df = self.df.groupby(by=_combined).size()
                    order = _df.groupby(by=self.bin_by).size().sort_values(ascending=False)
            else:
                if self.count_by in self.bin_by:
                    self.df['__count_by_copy__'] = self.df[self.count_by]
                    self.count_by_field = '__count_by_copy__'
                else:
                    self.count_by_field = self.count_by
                order = self.df.groupby(by=self.bin_by)[self.count_by_field].sum().sort_values(ascending=False)

            gb = self.df.groupby(self.bin_by)

            # Determine the max bin size... make sure it isn't zero
            if self.global_max is None:
                max_group_by = order.iloc[0]
                if max_group_by == 0:
                    max_group_by = 1
            else:
                max_group_by = self.global_max
            
            # Calculate each bar width
            max_bar_w,fv,i = 1.0,{},0
            while i < len(order):
                # Width of the bar in pixels
                px = max_bar_w * order.iloc[i] / max_group_by

                if type(order.index[i]) != list and type(order.index[i]) != tuple:
                    bin_text = str(order.index[i])
                else:
                    bin_text = ' | '.join([str(x) for x in order.index[i]])

                fv[bin_text] = px
                
                i += 1

            # Make it into a unit vector
            sq_sum = 0
            for k in fv.keys():
                sq_sum += fv[k]*fv[k]
            sq_sum = sqrt(sq_sum)
            if sq_sum < 0.001:
                sq_sum = 0.001
            fv_norm = {}
            for k in fv.keys():
                fv_norm[k] = fv[k]/sq_sum

            return fv_norm
        
        #
        # Determine which dataframe geometries overlap with a specific
        #
        def overlappingDataFrames(self, to_intersect):
            _dfs = []
            for _poly in self.geom_to_df.keys():
                if _poly.intersects(to_intersect):
                    _dfs.append(self.geom_to_df[_poly])
            if len(_dfs) > 0:
                return pd.concat(_dfs)
            else:
                return None
