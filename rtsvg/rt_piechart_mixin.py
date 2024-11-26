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
import polars as pl
import numpy as np

from shapely.geometry import Polygon

import math
from math import sqrt, pi, cos, sin, floor, ceil

import random

from .rt_component import RTComponent

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
                 df,                                                 # dataframe to render
                 # ------------------------------------------------- # everything else is a default...
                 color_by             : str             = None,      # just the default color or a string for a field
                 global_color_order   : list            = None,      # color by ordering... if none (default), will be created and filled in...
                 count_by             : str             = None,      # none means just count rows, otherwise, use a field to sum by # Not Implemented
                 count_by_set         : bool            = False,     # count by summation (by default)... column is checked
                 widget_id            : str             = None,      # naming the svg elements
                 # ------------------------------------------------- # custom render for this component
                 style                : str             = 'pie',     # 'pie' or 'waffle'
                 min_render_angle_deg : int             = 5,         # minimum render angle
                 # ------------------------------------------------- # visualization geometry / etc.
                 track_state          : bool            = False,     # track state for interactive filtering
                 x_view               : int             = 0,         # x offset for the view
                 y_view               : int             = 0,         # y offset for the view
                 x_ins                : int             = 3,         # side inserts
                 y_ins                : int             = 3,         # top & bottom inserts
                 w                    : int             = 256,       # width of the view
                 h                    : int             = 256,       # height of the view
                 draw_border          : bool            = True,      # draw a border around the histogram
                 draw_background      : bool            = False):    # useful to turn off in small multiples settings
        """
        Required Parameters
        -------------------

        df : pandas.DataFrame | polars.DataFrame
            Dataframe to render.

        Useful Parameters
        -----------------

        color_by : str | None
            The field to be used to individualize the pie slices.

        count_by : str | None
            The field to be used to determine the size of the pie slices.

        count_by_set : bool
            Use a set operation to determine the size of the pie slices in combination with "count_by".

        style : str
            The style of the chart.  Either 'pie' or 'waffle'.  Pie is the default and handles more cases.

        min_render_angle_deg : int
            The minimum angle in degrees to render a slice.  The default is 5 degrees.
        
        
        Other Parameters
        -----------------

        global_color_order : list
            The list of colors to use for the pie slices.  If not specified, will be created from the dataframe.

            This parameter is primarily used for small multiple renders.
            
        Standard Parameters
        -------------------

        widget_id : str | None
            The id of the widget.

        track_state : bool
            Track state for interactive filtering operations
        
        x_view, y_view : int
            The x and y offset for the SVG view

        w, h : int
            The width and height of the SVG frame

        x_ins, y_ins : int
            The x and y spacing to inset the visualization

        draw_labels : bool
            Draw the node labels (link labels are enabled by "link_labels" parameter)

        draw_border : bool
            Draw a border around the visualization

        """
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTPieChart(self, **_params_)

    #
    # RTPieChart
    #
    class RTPieChart(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     **kwargs):
            self.parms     = locals().copy()
            self.rt_self   = rt_self
            self.df        = rt_self.copyDataFrame(kwargs['df'])
            self.widget_id = kwargs['widget_id']

            # Make a widget_id if it's not set already
            if self.widget_id is None:
                self.widget_id = "piechart_" + str(random.randint(0,65535))
            
            self.color_by             = kwargs['color_by'] 
            self.global_color_order   = kwargs['global_color_order']
            self.count_by             = kwargs['count_by']
            self.count_by_set         = kwargs['count_by_set']
            self.style                = kwargs['style']
            self.min_render_angle_deg = kwargs['min_render_angle_deg']
            self.track_state          = kwargs['track_state']
            self.x_view               = kwargs['x_view']
            self.y_view               = kwargs['y_view']
            self.x_ins                = kwargs['x_ins']
            self.y_ins                = kwargs['y_ins']
            self.w                    = kwargs['w']
            self.h                    = kwargs['h']
            self.draw_border          = kwargs['draw_border']
            self.draw_background      = kwargs['draw_background']

            # Color by must not be none
            if self.color_by is None:
                raise Exception('RTPieChart.__init__() - color_by cannot be None')

            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)

            # Simple solution to color_by == count_by problem // unsure of the performance penalty
            if rt_self.isPandas(self.df) and self.color_by == self.count_by:
                new_col = 'color_by_' + str(random.randint(0,65535))
                self.df[new_col] = self.df[self.color_by]
                self.color_by = new_col

            # Check the count_by column
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)

            # Stateful tracking of geometry to dataframe
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
        # renderSVG() - create the SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state:
                self.geom_to_df = {}

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
            
            if len(self.df) > 0:
                # Render the different styles
                if self.style   == 'pie':
                    if   self.rt_self.isPandas(self.df):
                        svg += self.__renderPieStyle_pandas__(w_usable, h_usable)
                    elif self.rt_self.isPolars(self.df):
                        svg += self.__renderPieStyle_polars__(w_usable, h_usable)
                    else:
                        raise Exception("RTPieChart() - only pandas and polars supported")
                elif self.style == 'waffle':
                    if   self.rt_self.isPandas(self.df):                
                        svg += self.__renderWaffleStyle__(w_usable, h_usable)
                    else:
                        raise Exception('RTPieChart() - only pandas is supported for waffle style')
                else:
                    raise Exception(f'RTPieChart() - do not under style "{self.style}"')

            svg += '</svg>'
            self.last_render = svg
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
        def __renderPieStyle_polars__(self, w_usable, h_usable):
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
                counter = self.rt_self.polarsCounter(self.df, self.color_by, self.count_by, self.count_by_set)
                totals  = counter['__count__'].sum()
                tracking_gb = self.df.partition_by(self.color_by, as_dict=True)
                # Common render code
                deg, not_rendered = 0, []
                my_intersection = self.rt_self.__myIntersection__(self.global_color_order['index'], counter[self.color_by])
                for cb_bin in my_intersection:
                    my_color = cb_bin
                    my_total = counter.filter(pl.col(self.color_by) == cb_bin)['__count__'][0]
                    # Replicated arc code
                    degrees_to_render = 360.0*my_total/totals
                    if degrees_to_render > self.min_render_angle_deg:
                        _co = self.rt_self.co_mgr.getColor(my_color)
                        deg_end = deg + degrees_to_render
                        if degrees_to_render >= 360.0:
                            svg += f'<ellipse rx="{r}" ry="{r}" cx="{cx}" cy="{cy}" fill="{_co}" stroke-opacity="0.0" />'
                        else:
                            svg += f'<path d="{arcPath(cx,cy,r,deg,deg_end)}" fill="{_co}" stroke-opacity="0.0" />'

                            if self.track_state:
                                _poly_points = [[cx,cy]]
                                for _poly_degree in range(floor(deg),ceil(deg_end)+1,1):
                                    _poly_angle = pi * (_poly_degree-90) / 180.0 
                                    _poly_points.append([cx + cos(_poly_angle)*r, cy + sin(_poly_angle)*r])
                                self.geom_to_df[Polygon(_poly_points)] = tracking_gb[cb_bin]
                        deg = deg_end
                    elif self.track_state:
                        not_rendered.append(tracking_gb[cb_bin])
                
                # For any arcs that weren't long enough allocate them to the end
                if len(not_rendered) > 0 and self.track_state:
                    deg_end = 359
                    _poly_points = [[cx,cy]]
                    for _poly_degree in range(floor(deg),ceil(deg_end)+1,1):
                        _poly_angle = pi * (_poly_degree-90) / 180.0 
                        _poly_points.append([cx + cos(_poly_angle)*r, cy + sin(_poly_angle)*r])
                    self.geom_to_df[Polygon(_poly_points)] = not_rendered

            return svg

        #
        # Render the standard pie chart style
        #
        def __renderPieStyle_pandas__(self, w_usable, h_usable):
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

                # Create a groupby just for state tracking
                if self.track_state:
                    tracking_gb = self.df.groupby(self.color_by)
                    
                # Common render code
                deg, not_rendered = 0, []
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

                            if self.track_state:
                                _poly_points = [[cx,cy]]
                                for _poly_degree in range(floor(deg),ceil(deg_end)+1,1):
                                    _poly_angle = pi * (_poly_degree-90) / 180.0 
                                    _poly_points.append([cx + cos(_poly_angle)*r, cy + sin(_poly_angle)*r])
                                self.geom_to_df[Polygon(_poly_points)] = tracking_gb.get_group(cb_bin)
                        deg = deg_end
                    elif self.track_state:
                        not_rendered.append(tracking_gb.get_group(cb_bin))
                
                # For any arcs that weren't long enough allocate them to the end
                if len(not_rendered) > 0 and self.track_state:
                    deg_end = 359
                    _poly_points = [[cx,cy]]
                    for _poly_degree in range(floor(deg),ceil(deg_end)+1,1):
                        _poly_angle = pi * (_poly_degree-90) / 180.0 
                        _poly_points.append([cx + cos(_poly_angle)*r, cy + sin(_poly_angle)*r])
                    self.geom_to_df[Polygon(_poly_points)] = not_rendered

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
                if self.rt_self.isPandas(self.df):
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
                elif self.rt_self.isPolars(self.df):
                    counter = self.rt_self.polarsCounter(self.df, self.color_by, self.count_by, self.count_by_set)
                    totals  = counter['__count__'].sum()
                    for i in range(len(counter)):
                        fv[counter[self.color_by][i]] = counter['__count__'][i] / totals
                else:
                    raise Exception('RTPieChart.smallMultipleFeatureVector() - only pandas and polars supported')

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
        # Determine which dataframe geometries overlap with a specific
        #
        def overlappingDataFrames(self, to_intersect):
            _dfs = []
            for _poly in self.geom_to_df.keys():
                if _poly.intersects(to_intersect):
                    if type(self.geom_to_df[_poly]) == list:
                        _dfs.extend(self.geom_to_df[_poly])
                    else:
                        _dfs.append(self.geom_to_df[_poly])
            if len(_dfs) > 0:
                return self.rt_self.concatDataFrames(_dfs)
            else:
                return None

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
