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

import math
from math import sqrt

import random

__name__ = 'rt_piechart_mixin'

#
# Abstraction for Pie Chart
#
class RTPieChartMixin(object):
    #
    # pieChartPreferredDimensions()
    # - Return the preferred size
    #
    def pieChartPreferredDimensions(self, **kwargs):
        return (96,96)

    #
    # pieChartMinimumDimensions()
    # - Return the minimum size
    #
    def pieChartMinimumDimensions(self, **kwargs):
        return (32,32)

    #
    # pieChartSmallMultipleDimensions()
    #
    def pieChartSmallMultipleDimensions(self, **kwargs):
        return (24,24)

    #
    # Identify the required fields in the dataframe
    #
    def pieChartRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('color_by', kwargs, columns_set)
        return columns_set

    #
    # pieChart
    #
    # Make the SVG for a piechart
    #    
    def pieChart(self,
                 df,                              # dataframe to render
                 # ------------------------------ # everything else is a default...
                color_by             = None,      # just the default color or a string for a field
                global_color_order   = None,      # color by ordering... if none (default), will be created and filled in...
                count_by             = None,      # none means just count rows, otherwise, use a field to sum by # Not Implemented
                count_by_set         = False,     # count by summation (by default)... column is checked
                widget_id            = None,      # naming the svg elements
                # ------------------------------- # custom render for this component
                style                = 'pie',     # 'pie' or 'waffle'
                min_render_angle_deg = 5,         # minimum render angle
                # ------------------------------- # visualization geometry / etc.
                x_view               = 0,         # x offset for the view
                y_view               = 0,         # y offset for the view
                x_ins                = 3,         # side inserts
                y_ins                = 3,         # top & bottom inserts
                w                    = 256,       # width of the view
                h                    = 256,       # height of the view
                draw_border          = True,      # draw a border around the histogram
                draw_background      = False):    # useful to turn off in small multiples settings
        rt_piechart = self.RTPieChart(self, df, color_by=color_by, global_color_order=global_color_order, count_by=count_by, 
                                      count_by_set=count_by_set, widget_id=widget_id, style=style, min_render_angle_deg=min_render_angle_deg, 
                                      x_view=x_view, y_view=y_view, x_ins=x_ins, y_ins=y_ins, w=w, h=h, draw_border=draw_border, draw_background=draw_background)
        return rt_piechart.renderSVG()

    #
    # pieChart
    #
    # Make the SVG for a piechart
    #    
    def pieChartInstance(self,
                         df,                               # dataframe to render
                         # ------------------------------- # everything else is a default...
                         color_by             = None,      # just the default color or a string for a field
                         global_color_order   = None,      # color by ordering... if none (default), will be created and filled in...
                         count_by             = None,      # none means just count rows, otherwise, use a field to sum by # Not Implemented
                         count_by_set         = False,     # count by summation (by default)... column is checked
                         widget_id            = None,      # naming the svg elements
                         # ------------------------------- # custom render for this component
                         style                = 'pie',     # 'pie' or 'waffle'
                         min_render_angle_deg = 5,         # minimum render angle
                         # ------------------------------- # visualization geometry / etc.
                         x_view               = 0,         # x offset for the view
                         y_view               = 0,         # y offset for the view
                         x_ins                = 3,         # side inserts
                         y_ins                = 3,         # top & bottom inserts
                         w                    = 256,       # width of the view
                         h                    = 256,       # height of the view
                         draw_border          = True,      # draw a border around the histogram
                         draw_background      = False):     # useful to turn off in small multiples settings
        return self.RTPieChart(self, df, color_by=color_by, global_color_order=global_color_order, count_by=count_by, 
                               count_by_set=count_by_set, widget_id=widget_id, style=style, min_render_angle_deg=min_render_angle_deg, 
                               x_view=x_view, y_view=y_view, x_ins=x_ins, y_ins=y_ins, w=w, h=h, draw_border=draw_border, 
                               draw_background=draw_background)

    #
    # RTPieChart
    #
    class RTPieChart(object):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     df,                               # dataframe to render
                     # ------------------------------- # everything else is a default...
                     color_by             = None,      # just the default color or a string for a field
                     global_color_order   = None,      # color by ordering... if none (default), will be created and filled in...
                     count_by             = None,      # none means just count rows, otherwise, use a field to sum by # Not Implemented
                     count_by_set         = False,     # count by summation (by default)... column is checked
                     widget_id            = None,      # naming the svg elements
                     # ------------------------------- # custom render for this component
                     style                = 'pie',     # 'pie' or 'waffle'
                     min_render_angle_deg = 5,         # minimum render angle
                     # ------------------------------- # visualization geometry / etc.
                     x_view               = 0,         # x offset for the view
                     y_view               = 0,         # y offset for the view
                     x_ins                = 3,         # side inserts
                     y_ins                = 3,         # top & bottom inserts
                     w                    = 256,       # width of the view
                     h                    = 256,       # height of the view
                     draw_border          = True,      # draw a border around the histogram
                     draw_background      = False):    # useful to turn off in small multiples settings

            self.parms     = locals().copy()
            self.rt_self   = rt_self
            self.df        = df.copy()
            self.widget_id = widget_id

            # Make a widget_id if it's not set already
            if self.widget_id is None:
                self.widget_id = "piechart_" + str(random.randint(0,65535))
            
            self.color_by             = color_by 
            self.global_color_order   = global_color_order
            self.count_by             = count_by
            self.count_by_set         = count_by_set
            self.style                = style
            self.min_render_angle_deg = min_render_angle_deg
            self.x_view               = x_view
            self.y_view               = y_view
            self.x_ins                = x_ins
            self.y_ins                = y_ins
            self.w                    = w
            self.h                    = h
            self.draw_border          = draw_border
            self.draw_background      = draw_background

            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)

            # Simple solution to color_by == count_by problem // unsure of the performance penalty
            if self.color_by == self.count_by:
                new_col = 'color_by_' + str(random.randint(0,65535))
                self.df[new_col] = self.df[self.color_by]
                self.color_by = new_col

            # Check the count_by column
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)

        #
        # renderSVG() - create the SVG
        #
        def renderSVG(self):
            # Color ordering
            if self.global_color_order is None:
                self.global_color_order = self.rt_self.colorRenderOrder(self.df, self.color_by, self.count_by, self.count_by_set)
           
            #
            # Geometry
            #
            params_orig_minus_self = self.parms.copy()
            params_orig_minus_self.pop('self')
            min_dims = self.rt_self.pieChartSmallMultipleDimensions(**params_orig_minus_self)
            if self.w < min_dims[0] or self.h < min_dims[1]:
                self.x_ins        = 1
                self.y_ins        = 1

            w_usable = self.w - 2*self.x_ins
            h_usable = self.h - 2*self.y_ins

            # Create the SVG ... render the background
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            if self.draw_background:
                background_color = self.rt_self.co_mgr.getTVColor('background','default')
                svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'
            
            # Render the different styles
            if self.style   == 'pie':
                svg += self.__renderPieStyle__(w_usable, h_usable)
            elif self.style == 'waffle':
                svg += self.__renderWaffleStyle__(w_usable, h_usable)
            else:
                raise Exception(f'RTPieChart() - do not under style "{self.style}"')

            svg += '</svg>'
            return svg

        #
        # Render the standard pie chart style
        #
        def __renderWaffleStyle__(self, w_usable, h_usable):
            svg = ''

            # Waffle square dimensions
            w_intra = w_usable * 0.1/10
            h_intra = h_usable * 0.1/10
            tile_w  = (w_usable - w_intra)/11
            tile_h  = (h_usable - h_intra)/11
            xT = lambda x: self.x_ins + w_intra/2 + (tile_w+w_intra)*x
            yT = lambda y: self.y_ins + h_intra/2 + (tile_h+h_intra)*y

            # Make default squares for whatever doesn't get filled in
            default_color = self.rt_self.co_mgr.getTVColor('data','default')
            x_tile,y_tile = 0,0
            for i in range(0,100):
                svg += f'<rect x="{xT(x_tile)}" y="{yT(y_tile)}" width="{tile_w}" height="{tile_h}" fill="{default_color}" />'
                x_tile +=1
                if x_tile == 10:
                    x_tile = 0
                    y_tile += 1

            # Colorized version
            if self.color_by is not None:
                # Count By Rows
                if   self.count_by is None:
                    totals = len(self.df) # total number of rows
                    tmp_df = pd.DataFrame(self.df.groupby(self.color_by).size())
                    total_i = 0
                # Count By Set
                elif self.count_by_set:
                    tmp_df = pd.DataFrame(self.df.groupby([self.color_by,self.count_by]).size()).reset_index()
                    totals = len(tmp_df)
                    tmp_df = pd.DataFrame(tmp_df.groupby(self.color_by).size())
                    total_i = 0
                # Count By Numbers
                else:
                    tmp_df = pd.DataFrame(self.df.groupby(self.color_by)[self.count_by].sum())
                    totals = tmp_df[self.count_by].sum()
                    total_i = self.count_by

                # Common render code
                x_tile,y_tile = 0,0
                my_intersection = self.rt_self.__myIntersection__(self.global_color_order.index,tmp_df.index)
                for cb_bin in my_intersection:
                    my_color          = cb_bin
                    _co               = self.rt_self.co_mgr.getColor(my_color)
                    my_total          = tmp_df.loc[cb_bin][total_i]
                    squares_to_render = int(100.0*my_total/totals)
                    for i in range(0,squares_to_render):
                        svg += f'<rect x="{xT(x_tile)}" y="{yT(y_tile)}" width="{tile_w}" height="{tile_h}" fill="{_co}" />'
                        x_tile +=1
                        if x_tile == 10:
                            x_tile = 0
                            y_tile += 1

            return svg

        #
        # Render the standard pie chart style
        #
        def __renderPieStyle__(self, w_usable, h_usable):
            cx = self.x_ins + w_usable/2
            cy = self.y_ins + h_usable/2
            if w_usable < h_usable:
                r = w_usable/2
            else:
                r = h_usable/2
            default_color = self.rt_self.co_mgr.getTVColor('data','default')

            # Draw the default data color circle...
            svg = f'<ellipse rx="{r}" ry="{r}" cx="{cx}" cy="{cy}" fill="{default_color}" stroke-opacity="0.0" />'
            
            # Otherwise, break the cases down by how we're counting...
            if self.color_by is not None:
                # Count By Rows
                if   self.count_by is None:
                    totals = len(self.df) # total number of rows
                    tmp_df = pd.DataFrame(self.df.groupby(self.color_by).size())
                    total_i = 0
                # Count By Set
                elif self.count_by_set:
                    tmp_df = pd.DataFrame(self.df.groupby([self.color_by,self.count_by]).size()).reset_index()
                    totals = len(tmp_df)
                    tmp_df = pd.DataFrame(tmp_df.groupby(self.color_by).size())
                    total_i = 0
                # Count By Numbers
                else:
                    tmp_df = pd.DataFrame(self.df.groupby(self.color_by)[self.count_by].sum())
                    totals = tmp_df[self.count_by].sum()
                    total_i = self.count_by

                # Common render code
                deg = 0
                my_intersection = self.rt_self.__myIntersection__(self.global_color_order.index,tmp_df.index)
                for cb_bin in my_intersection:
                    my_color = cb_bin
                    my_total = tmp_df.loc[cb_bin][total_i]
                    # Replicated arc code
                    degrees_to_render = 360.0*my_total/totals
                    if degrees_to_render > self.min_render_angle_deg:
                        _co = self.rt_self.co_mgr.getColor(my_color)
                        deg_end = deg + degrees_to_render
                        if degrees_to_render >= 360.0:
                            svg += f'<ellipse rx="{r}" ry="{r}" cx="{cx}" cy="{cy}" fill="{_co}" stroke-opacity="0.0" />'
                        else:
                            svg += f'<path d="{arcPath(cx,cy,r,deg,deg_end)}" fill="{_co}" stroke-opacity="0.0" />'
                        deg = deg_end

            return svg
        
        #
        # smallMultipleFeatureVector()
        # ... feature vector for comparison with other small multiple instances of this class
        # ... pretty much a copy of the render code above...
        #
        def smallMultipleFeatureVector(self):
            # Otherwise, break the cases down by how we're counting...
            fv,fv_norm = {},{}
            if self.color_by is not None:
                # Count By Rows
                if   self.count_by is None:
                    totals = len(self.df) # total number of rows
                    tmp_df = pd.DataFrame(self.df.groupby(self.color_by).size())
                    total_i = 0
                # Count By Set
                elif self.count_by_set:
                    tmp_df = pd.DataFrame(self.df.groupby([self.color_by,self.count_by]).size()).reset_index()
                    totals = len(tmp_df)
                    tmp_df = pd.DataFrame(tmp_df.groupby(self.color_by).size())
                    total_i = 0
                # Count By Numbers
                else:
                    tmp_df = pd.DataFrame(self.df.groupby(self.color_by)[self.count_by].sum())
                    totals = tmp_df[self.count_by].sum()
                    total_i = self.count_by

                # Common render code
                for cb_bin in tmp_df.index:
                    my_total = tmp_df.loc[cb_bin][total_i]
                    fv[cb_bin] = my_total/totals

                # Make it into a unit vector
                sq_sum = 0
                for k in fv.keys():
                    sq_sum += fv[k]*fv[k]
                sq_sum = sqrt(sq_sum)
                if sq_sum < 0.001:
                    sq_sum = 0.001
                for k in fv.keys():
                    fv_norm[k] = fv[k]/sq_sum

            return fv_norm

#
# Converted from the following description
# https://stackoverflow.com/questions/5736398/how-to-calculate-the-svg-path-for-an-arc-of-a-circle
#
def polarToCartesian(cx, cy, r, deg):
    rads = (deg-90) * math.pi / 180.0
    return cx + (r*math.cos(rads)), cy + (r*math.sin(rads))
def arcPath(cx, cy, r, deg0, deg1):
    x0,y0 = polarToCartesian(cx,cy,r,deg1)
    x1,y1 = polarToCartesian(cx,cy,r,deg0)
    if (deg1 - deg0) <= 180.0:
        flag = "0"
    else:
        flag = "1"
    return f'M {cx} {cy} L {x0} {y0} A {r} {r} 0 {flag} 0 {x1} {y1}'
