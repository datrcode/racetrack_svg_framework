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

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from datetime import timedelta
from dateutil.relativedelta import relativedelta

from math import log10

import random

__name__ = 'rt_xy_mixin'

#
# Abstraction for XY Scatterplot
#
class RTXYMixin(object):
    #
    # Draw the background temporal context for the x-axis.
    # ... attempted rewrite... doesn't work as well as the original...
    #
    def SUBOPTIMALdrawXYTemporalContext(self, x, y, w, h, txt_h, ts_min, ts_max, draw_labels):
        svg = ''
        fill_co       = self.co_mgr.getTVColor('context','default')
        txt_co        = self.co_mgr.getTVColor('context','text')
        axis_major_co = self.co_mgr.getTVColor('axis',   'major')
        axis_minor_co = self.co_mgr.getTVColor('axis',   'minor')

        hashmark_interval = 0 # timedelta between hashmarks
        mult              = 1
        format_zero       = 2 # zeroized the format to find the start hashmark
        format_render     = 3 # how to render the major hashmarks
        minor_time_test   = 4 # lambda for true/false for minor labels
        major_time_test   = 5 # lambda for true/false on label render

        fmt_lu = {
            #        hashmark_interval            mult  format_zero       format_render   minor_time_test                        major_time_test
            '1s':   (relativedelta(seconds=1),    1,    '%Y-%m-%d %H:%M', '%M:%s',        lambda x: True,                        lambda x: x.second%15   == 0),
            '10s':  (relativedelta(seconds=1),    10,   '%Y-%m-%d %H:%M', '%M:%s',        lambda x: True,                        lambda x: x.second%30   == 0),
            '15s':  (relativedelta(seconds=1),    15,   '%Y-%m-%d %H:%M', '%M:%s',        lambda x: x.second%5    == 0,          lambda x: x.second      == 0),
            '30s':  (relativedelta(seconds=1),    30,   '%Y-%m-%d %H:%M', '%M:%s',        lambda x: x.second%30   == 0,          lambda x: x.second      == 0),

            '1Mi':  (relativedelta(minutes=1),    1,    '%Y-%m-%d %H:%M', '%H%M',         lambda x: x.minute%10   == 0,          lambda x: x.minute%10   == 0),
            '5Mi':  (relativedelta(minutes=1),    5,    '%Y-%m-%d %H:%M', '%H%M',         lambda x: x.minute%15   == 0,          lambda x: x.minute%15   == 0),
            '10Mi': (relativedelta(minutes=1),    10,   '%Y-%m-%d %H:%M', '%H%M',         lambda x: x.minute%30   == 0,          lambda x: x.minute%30   == 0),
            '15Mi': (relativedelta(minutes=1),    15,   '%Y-%m-%d %H:%M', '%H%M',         lambda x: x.minute      == 0,          lambda x: x.minute      == 0),
            '30Mi': (relativedelta(minutes=1),    30,   '%Y-%m-%d %H:%M', '%H%M',         lambda x: x.minute      == 0,          lambda x: x.minute      == 0),

            '1h':   (relativedelta(hours=1),      1,    '%Y-%m-%d %H',    '%d %H',        lambda x: x.hour        == 0,          lambda x: x.hour        == 0),
            '3h':   (relativedelta(hours=1),      3,    '%Y-%m-%d %H',    '%d %H',        lambda x: x.hour        == 0,          lambda x: x.hour        == 0),
            '6h':   (relativedelta(hours=1),      6,    '%Y-%m-%d %H',    '%d %H',        lambda x: x.hour        == 0,          lambda x: x.hour        == 0),
            '12h':  (relativedelta(hours=1),      12,   '%Y-%m-%d %H',    '%d %H',        lambda x: x.hour        == 0,          lambda x: x.hour        == 0),

            '1d':   (relativedelta(days=1),       1,    '%Y-%m',          '%m/%d',        lambda x: True,                        lambda x: (x.day-1)%9   == 0),
            '5d':   (relativedelta(days=1),       5,    '%Y-%m',          '%m/%d',        lambda x: (x.day-1)%2 == 0,            lambda x: x.day         == 1),
            '7d':   (relativedelta(days=1),       7,    '%Y-%m',          '%m/%d',        lambda x: x.day == 1 or x.day == 15,   lambda x: x.day         == 1),
            '15d':  (relativedelta(days=1),       15,   '%Y-%m',          '%m/%d',        lambda x: (x.day-1)     == 0,          lambda x: (x.day-1)     == 0 and (x.month-1)%3 == 0),

            '1Mo':  (relativedelta(months=1),     1,    '%Y-%m',          '%Y-%m',        lambda x: x.month-1     == 0,          lambda x: x.month-1     == 0),
            '3Mo':  (relativedelta(months=1),     3,    '%Y-%m',          '%Y-%m',        lambda x: x.month-1     == 0,          lambda x: x.month-1     == 0),
            '6Mo':  (relativedelta(months=1),     6,    '%Y-%m',          '%Y-%m',        lambda x: x.month-1     == 0,          lambda x: x.month-1     == 0),

            '1y':   (relativedelta(years=1),      1,    '%Y',             '%Y',           lambda x: x.year%10     == 0,          lambda x: x.year%10     == 0),
            '5y':   (relativedelta(years=1),      5,    '%Y',             '%Y',           lambda x: x.year%20     == 0,          lambda x: x.year%20     == 0),
            '10y':  (relativedelta(years=1),      10,   '%Y',             '%Y',           lambda x: x.year%50     == 0,          lambda x: x.year%50     == 0),
            '25y':  (relativedelta(years=1),      25,   '%Y',             '%Y',           lambda x: x.year%50     == 0,          lambda x: x.year%50     == 0),
            '50y':  (relativedelta(years=1),      50,   '%Y',             '%Y',           lambda x: x.year%100    == 0,          lambda x: x.year%100    == 0)
        }

        # Transform for the x position
        xT = lambda _ts_: x + w * ((_ts_ - ts_min)/(ts_max - ts_min)) 

        # Find the correct resolution
        for k in fmt_lu.keys():
            px_dist = xT(ts_min + fmt_lu[k][mult]*fmt_lu[k][hashmark_interval]) - xT(ts_min)
            if px_dist > 6:
                break
            k = None

        if k is not None:                                                     #DEBUG
            svg += self.svgText(k, w-3-self.textLength(k, txt_h), h-4, txt_h) #DEBUG

        # If we found a resolution...
        if k is not None:
            tup = fmt_lu[k]
            ts_zero = ts_min.strftime(tup[format_zero])  # start at the earliest timestamp in the view
            ts      = pd.to_datetime(ts_zero)            # zeroize the parts that don't matter

            # Do the minor hashmarks...
            while ts < ts_max:
                px =  xT(ts)
                if px >= x and px <= x+w and tup[minor_time_test](ts):
                    svg += f'<line x1="{px}" y1="{y}" x2="{px}" y2="{y+5}" stroke="{axis_minor_co}" stroke-width="0.8" />'
                ts += tup[hashmark_interval]

            # Do the major hashmarks + labels
            ts      = pd.to_datetime(ts_zero)            # zeroize the parts that don't matter
            while ts < ts_max:
                px = xT(ts)
                if px >= x and px <= x+w and tup[major_time_test](ts):
                    ts_str = ts.strftime(tup[format_render])
                    svg += f'<line x1="{px}" y1="{y}" x2="{px}" y2="{y+h}" stroke="{axis_major_co}" stroke-width="0.8" />'
                    svg += self.svgText(ts_str, px+2, y+txt_h+1, 3*txt_h/4, color=txt_co)
                ts += tup[hashmark_interval]

        return svg

    #
    # Draw the background temporal context for the x-axis.
    #
    def drawXYTemporalContext(self, x, y, w, h, txt_h, ts_min, ts_max, draw_labels):
        svg = ''

        fill_co       = self.co_mgr.getTVColor('context','default')
        txt_co        = self.co_mgr.getTVColor('context','text')
        axis_major_co = self.co_mgr.getTVColor('axis',   'major')
        axis_minor_co = self.co_mgr.getTVColor('axis',   'minor')

        tdiv       = 0
        fmt_render = 1
        fmt_zero   = 2
        tpart      = 3
        tmod       = 4
        tinc       = 5
        tsearch    = 6
        tinc_div2  = 7

        #          tdiv                     fmt_render  fmt_zero          tpart                      tmod  tinc                        tsearch                   tinc_div2
        fmt_lu = {
            '50y': (timedelta(days=365*50), '%Y',       '%Y-01-01',       lambda x: x.year,          50,   relativedelta(years=+100),  relativedelta(years=+1),  relativedelta(years=+50)),
            '25y': (timedelta(days=365*25), '%Y',       '%Y-01-01',       lambda x: x.year,          25,   relativedelta(years=+50),   relativedelta(years=+1),  relativedelta(years=+25)),
            '10y': (timedelta(days=365*10), '%Y',       '%Y-01-01',       lambda x: x.year,          10,   relativedelta(years=+20),   relativedelta(years=+1),  relativedelta(years=+10)),
            '5y':  (timedelta(days=365*5),  '%Y',       '%Y-01-01',       lambda x: x.year,           5,   relativedelta(years=+10),   relativedelta(years=+1),  relativedelta(years=+5)),
            '1y':  (timedelta(days=365),    '%Y',       '%Y-01-01',       lambda x: x.year,           2,   relativedelta(years=+2),    relativedelta(years=+1),  relativedelta(years=+1)),
            '6m':  (timedelta(days=182),    '%Y-%m',    '%Y-%m-01',       lambda x: x.month-1,        6,   relativedelta(years=+1),    relativedelta(months=+1), relativedelta(months=+6)),

            '3m':  (timedelta(days=91),     '%Y-%m',    '%Y-%m-01',       lambda x: x.month-1,        3,   relativedelta(months=+2),   relativedelta(months=+1), relativedelta(months=+3)),
            '2m':  (timedelta(days=61),     '%Y-%m',    '%Y-%m-01',       lambda x: x.month-1,        2,   relativedelta(months=+1),   relativedelta(days=+1),   relativedelta(days=+15)),
            '1m':  (timedelta(days=30),     '%Y-%m',    '%Y-%m-01',       lambda x: x.month-1,        2,   relativedelta(months=+1),   relativedelta(days=+1),   relativedelta(days=+15)),

            '15d': (timedelta(days=15),     '%Y-%m-%d', '%Y-%m-01',       lambda x: x.day-1,         45,   relativedelta(months=+1),   relativedelta(days=+1),   relativedelta(days=+15)),
            '7d':  (timedelta(days=7),      '%a',       '%Y-%m-%d',       lambda x: x.day_of_week,   45,   relativedelta(days=+7),     relativedelta(days=+1),   relativedelta(days=+5)),
            '1d':  (timedelta(days=1),      '%m-%d',    '%Y-%m-%d',       lambda x: x.day-1,          2,   relativedelta(days=+2),     relativedelta(days=+1),   relativedelta(days=+1)),
            '6h':  (timedelta(hours=6),     '%H:00',    '%Y-%m-%d %H',    lambda x: x.hour,           6,   relativedelta(hours=+12),   relativedelta(hours=+1),  relativedelta(hours=+6)),
            '3h':  (timedelta(hours=3),     '%H:00',    '%Y-%m-%d %H',    lambda x: x.hour,           3,   relativedelta(hours=+6),    relativedelta(hours=+1),  relativedelta(hours=+3)),
            '1h':  (timedelta(hours=1),     '%H:00',    '%Y-%m-%d %H',    lambda x: x.hour,           2,   relativedelta(hours=+2),    relativedelta(hours=+1),  relativedelta(hours=+1)),
            '15m': (timedelta(minutes=15),  '%H:%M',    '%Y-%m-%d %H:%M', lambda x: x.minute,        15,   relativedelta(minutes=+30), relativedelta(minutes=+1),relativedelta(minutes=+15)),
        }

        px_annotation = 200 # desired annotation size in pixels

        # Find the right resolution
        for k in fmt_lu.keys():
            tup    = fmt_lu[k]
            if ((ts_max - ts_min)/tup[tdiv]) > w/px_annotation:
                break
            tup    = None
        
        #if tup is not None: #DEBUG
        #    svg += self.svgText(k, w-3-self.textLength(k, txt_h), h-4, txt_h) #DEBUG

        # Render at that resolution
        if tup is not None:
            # Put minor markings down (same as next loop)
            ts_zero = ts_min.strftime(tup[fmt_zero])  # start at the earliest timestamp in the view
            ts      = pd.to_datetime(ts_zero)         # zeroize the parts that don't matter
            ts_part = tup[tpart](ts)                  # extract out the time part to compare
            x0_ratio = 0.0
            while x0_ratio < 1.0:           # while we haven't found the first location of this time interval...
                x0_ratio     = (ts                  - ts_min)/(ts_max - ts_min)
                x0_highlight = x + w * x0_ratio
                if x0_highlight >= 0:
                    svg += f'<line x1="{x0_highlight}" y1="{y}" x2="{x0_highlight}" y2="{y+5}" stroke="{axis_major_co}" stroke-width="0.8" />'
                ts += tup[tsearch]                    # ... increment by the time search parameter
                ts_part = tup[tpart](ts)              # ... extract out the time part to compare

            # Search for first mark
            ts_zero = ts_min.strftime(tup[fmt_zero])  # start at the earliest timestamp in the view
            ts      = pd.to_datetime(ts_zero)         # zeroize the parts that don't matter
            ts_part = tup[tpart](ts)                  # extract out the time part to compare
            while (ts_part%tup[tmod]) != 0:           # while we haven't found the first location of this time interval...
                ts += tup[tsearch]                    # ... increment by the time search parameter
                ts_part = tup[tpart](ts)              # ... extract out the time part to compare
            
            # Iterate over the range of dates and render
            while ts < ts_max:
                x0_ratio     = (ts                  - ts_min)/(ts_max - ts_min)
                x0_highlight = x + w * x0_ratio
                x1_ratio     = ((ts+tup[tinc_div2]) - ts_min)/(ts_max - ts_min)
                x1_highlight = x + w * x1_ratio
                w_highlight  = x1_highlight - x0_highlight

                ts_str = ts.strftime(tup[fmt_render])
                if x0_highlight >= 0:
                    svg += f'<line x1="{x0_highlight}" y1="{y}" x2="{x0_highlight}" y2="{y+h}" stroke="{axis_major_co}" stroke-width="0.8" />'
                    svg += self.svgText(ts_str, x0_highlight+2, y+txt_h+1, 3*txt_h/4, color=txt_co)

                ts += tup[tinc]
        
        return svg

    #
    # xyPreferredDimensions()
    # - Return the preferred size
    #
    def xyPreferredDimensions(self, **kwargs):
        if 'x_is_time' in kwargs.keys() and kwargs['x_is_time']:
            return (256, 128)
        return (160,160)

    #
    # xyMinimumDimensions()
    # - Return the minimum size
    #
    def xyMinimumDimensions(self, **kwargs):
        if 'x_is_time' in kwargs.keys() and kwargs['x_is_time']:
            return (160,96)
        return (96,96)

    #
    # xySmallMultipleDimensions()
    #
    def xySmallMultipleDimensions(self, **kwargs):
        if 'x_is_time' in kwargs.keys() and kwargs['x_is_time']:
            return (200,20)        
        return (64,64)

    #
    # Identify the required fields in the dataframe from linknode parameters
    #
    def xyRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('x_field',  kwargs, columns_set)
        self.identifyColumnsFromParameters('y_field',  kwargs, columns_set)
        self.identifyColumnsFromParameters('color_by', kwargs, columns_set)
        self.identifyColumnsFromParameters('count_by', kwargs, columns_set)
        self.identifyColumnsFromParameters('line_groupby_field',  kwargs, columns_set)
        self.identifyColumnsFromParameters('df2_ts_field',        kwargs, columns_set)
        self.identifyColumnsFromParameters('y2_field',            kwargs, columns_set)
        self.identifyColumnsFromParameters('line2_groupby_field', kwargs, columns_set)
        return columns_set

    #
    # histogram
    #
    # Make the SVG for a histogram from a dataframe
    #    
    def xy(self,
           df,                            # dataframe to render
           x_field,                       # string or an array of strings
           y_field,                       # string or an array of strings

           # -----------------------      # everything else is a default...

           x_field_is_scalar = True,      # default... logic will check in the method to determine if this is true
           y_field_is_scalar = True,      # default... logic will check in the method to determine if this is true
           color_by          = None,      # just the default color or a string for a field
           color_magnitude   = None,      # Only applies when color_by is None, options: None / 'linear' / 'log' / 'stretch'
           count_by          = None,      # none means just count rows, otherwise, use a field to sum by # Not Implemented
           count_by_set      = False,     # count by summation (by default)... column is checked
           dot_size          = 'medium',  # Dot size - ['small', 'medium', 'large', 'vary', 'hidden'/None]
           dot_shape         = 'ellipse', # Dot shape - ['square', 'ellipse', 'triangle, 'utriangle', 'diamond', 'plus', x', 'small_multiple', function_pointer]
           max_dot_size      = 5,         # Max dot size (used when the dot sz varies)
           opacity           = 1.0,       # Opacity of the dots
           vary_opacity      = False,     # If true, vary opacity by the count_by # Not Implemented
           align_pixels      = True,      # Align pixels to integer values
           widget_id         = None,      # naming the svg elements

           # ------------------------     # used for globally making the same scale/etc

           x_axis_col        = None,      # x axis column name
           x_is_time         = False,     # x is time flag
           x_label_min       = None,      # min label on the x axis
           x_label_max       = None,      # max label on the x axis
           x_trans_func      = None,      # lambda transform function for x axis
           x_order           = None,      # order of categorical values on the x axis
           x_fill_transforms = True,      # for t-fields, fill in all the values to properly space out data

           y_axis_col        = None,      # y axis column name
           y_is_time         = False,     # y is time flag
           y_label_min       = None,      # min label on the y axis
           y_label_max       = None,      # max label on the y axis
           y_trans_func      = None,      # lambeda transform function for y axis
           y_order           = None,      # order of categorical values on the y axis
           y_fill_transforms = True,      # for t-fields, fill in all the values to properly space out data

           # ------------------------     # x = timestamp options // Only applies if x-axis is time

           line_groupby_field  = None,    # will use a field to perform a groupby for a line chart
                                          # calling app should make sure that all timeslots are filled in...
           line_groupby_w      = 1,       # width of the line for the line chart

           # ------------------------     # secondary axis settings # probably not small multiple safe...

           df2                     = None,       # secondary axis dataframe ... if not set but y2_field is, then this will be set to df field
           df2_ts_field            = None,       # secondary axis x timestamp field ... if not set but the y2_field is, then this be set to the x_field
           y2_field                = None,       # secondary axis field ... if this is set, then df2 will be set to df // only required field really...
           y2_field_is_scalar      = True,       # default... logic will check in the method to determine if this is true
           y2_axis_col             = None,       # y2 axis column name
           line2_groupby_field     = None,       # secondary line field ... will NOT be set
           line2_groupby_w         = 0.75,       # secondary line field width
           line2_groupby_color     = None,       # line2 color... if none, pulls from the color_by field
           line2_groupby_dasharray = "4 2",      # line2 dasharray
           dot2_size               = 'inherit',  # dot2 size... 'inherit' means take from the dot_size...

           # -----------------------      # small multiple options

           sm_type               = None,  # should be the method name // similar to the smallMultiples method
           sm_w                  = None,  # override the width of the small multiple
           sm_h                  = None,  # override the height of the small multiple
           sm_params             = {},    # dictionary of parameters for the small multiples
           sm_x_axis_independent = True,  # Use independent axis for x (xy, temporal, and linkNode)
           sm_y_axis_independent = True,  # Use independent axis for y (xy, temporal, periodic, pie)

           # -----------------------      # background information

           bg_shape_lu           = None,       # lookup for background shapes -- key will be used for varying colors (if bg_shape_label_color == 'vary')
                                               # ['key'] = [(x0,y0),(x1,y1),...] OR
                                               # ['key'] = svg path description
           bg_shape_label_color  = None,       # None = no label, 'vary', lookup to hash color, or a hash color
           bg_shape_opacity      = 1.0,        # None (== 0.0), number, lookup to opacity
           bg_shape_fill         = None,       # None, 'vary', lookup to hash color, or a hash color
           bg_shape_stroke_w     = 1.0,        # None, number, lookup to width
           bg_shape_stroke       = 'default',  # None, 'default', lookup to hash color, or a hash color

           # ----------------------------------- # Distributions
           render_x_distribution       = None,       # number of x distribution buckets ... None == don't render
           render_y_distribution       = None,       # number of x distribution buckets ... None == don't render
           render_distribution_opacity = 0.5,        # Opacity of the distribution render
           distribution_h_perc         = 0.33,       # height of the distribution as a function of the overall h of xy chart     
           distribution_style          = 'outside',  # 'outside' - outside of xy... 'inside' - inside of xy

           # ---------------------------------------  # visualization geometry / etc.

           x_view                    = 0,             # x offset for the view
           y_view                    = 0,             # y offset for the view
           x_ins                     = 3,             # side inserts
           y_ins                     = 3,             # top & bottom inserts
           w                         = 256,           # width of the view
           h                         = 256,           # height of the view
           txt_h                     = 12,            # text height for labeling
           background_opacity        = 1.0,
           background_override       = None,          # override the background color // hex value
           plot_background_override  = None,          # override the background for the plot area // hex value
           draw_x_gridlines          = False,         # draw the x gridlines for scalar values
           draw_y_gridlines          = False,         # draw the y gridlines for scalar values
           draw_labels               = True,          # draw labels flag
           draw_border               = True,          # draw a border around the histogram
           draw_context              = True):         # draw temporal context information if (and only if) x_axis is time
        rt_xy = self.RTXy(self, df, x_field, y_field, x_field_is_scalar=x_field_is_scalar, y_field_is_scalar=y_field_is_scalar, color_by=color_by, color_magnitude=color_magnitude,                         
                          count_by=count_by, count_by_set=count_by_set, dot_size=dot_size, dot_shape=dot_shape, max_dot_size=max_dot_size, opacity=opacity, vary_opacity=vary_opacity, align_pixels=align_pixels,
                          widget_id=widget_id, x_axis_col=x_axis_col, x_is_time=x_is_time, x_label_min=x_label_min, x_label_max=x_label_max, x_trans_func=x_trans_func, y_axis_col=y_axis_col,
                          x_order=x_order,y_order=y_order, x_fill_transforms=x_fill_transforms, y_fill_transforms=y_fill_transforms,
                          y_is_time=y_is_time, y_label_min=y_label_min, y_label_max=y_label_max, y_trans_func=y_trans_func, line_groupby_field=line_groupby_field, line_groupby_w=line_groupby_w,
                          df2=df2, df2_ts_field=df2_ts_field, y2_field=y2_field, y2_field_is_scalar=y2_field_is_scalar, y2_axis_col=y2_axis_col, line2_groupby_field=line2_groupby_field,
                          line2_groupby_w=line2_groupby_w, line2_groupby_color=line2_groupby_color, line2_groupby_dasharray=line2_groupby_dasharray, dot2_size=dot2_size,
                          sm_type=sm_type, sm_w=sm_w, sm_h=sm_h, sm_params=sm_params, sm_x_axis_independent=sm_x_axis_independent, sm_y_axis_independent=sm_y_axis_independent,
                          render_x_distribution=render_x_distribution,render_y_distribution=render_y_distribution,render_distribution_opacity=render_distribution_opacity,
                          distribution_h_perc=distribution_h_perc, distribution_style=distribution_style,
                          bg_shape_lu=bg_shape_lu, bg_shape_label_color=bg_shape_label_color, bg_shape_opacity=bg_shape_opacity, bg_shape_fill=bg_shape_fill,
                          bg_shape_stroke_w=bg_shape_stroke_w, bg_shape_stroke=bg_shape_stroke, x_view=x_view, y_view=y_view, x_ins=x_ins, y_ins=y_ins, w=w, h=h, txt_h=txt_h,
                          background_opacity=background_opacity, background_override=background_override, plot_background_override=plot_background_override,
                          draw_x_gridlines=draw_x_gridlines, draw_y_gridlines=draw_y_gridlines,
                          draw_labels=draw_labels, draw_border=draw_border, draw_context=draw_context)
        return rt_xy.renderSVG()

    #
    # Create a column on the 0..1 scale for an axis
    # - can be used externally to make consistent scales across small multiples
    #
    def xyCreateAxisColumn(self, 
                           df, 
                           field, 
                           is_scalar, 
                           new_axis_field,
                           order          = None,   # Order of the values on the axis
                           fill_transform = True,   # Fill in missing transform values
                           timestamp_min  = None,   # Minimum timestamp field
                           timestamp_max  = None):  # Maximum timestamp field
        if type(field) != list:
            field = [field]
        is_time = False
        field_countable = (df[field[0]].dtypes == np.int64   or df[field[0]].dtypes == np.int32 or \
                           df[field[0]].dtypes == np.float64 or df[field[0]].dtypes == np.float32)
        f0 = field[0]

        transFunc = None

        # Numeric scaling
        if field_countable and is_scalar and len(field) == 1:
            my_min = df[f0].min()
            my_max = df[f0].max()
            if my_min == my_max:
                my_min -= 0.5
                my_max += 0.5
            df[new_axis_field] = ((df[f0] - my_min)/(my_max - my_min))
            label_min = str(my_min)
            label_max = str(my_max)

            transFunc = lambda x: ((x - my_min)/(my_max - my_min))

        # Timestamp scaling
        elif len(field) == 1 and is_datetime(df[field[0]]):
            # Use dataframe for min... or the parameter version if set
            if timestamp_min is None:
                my_min = df[f0].min()
            else:
                my_min = timestamp_min

            # Use dataframe for min... or the parameter version if set
            if timestamp_max is None:
                my_max = df[f0].max()
            else:
                my_max = timestamp_max

            if my_min == my_max:
                my_max += timedelta(seconds=1)
            df[new_axis_field] = ((df[f0] - my_min)/(my_max - my_min))
            label_min = df[f0].min()
            label_max = df[f0].max()
            is_time = True

            transFunc = lambda x: ((x - my_min)/(my_max - my_min))
        
        # Equal scaling
        else:
            # This fills in the natural ordering of the data if the fill_transform is enabled (it's true by default)
            # ... unclear what should be done if this is multifield and one or more transforms exists
            if fill_transform and order is None and len(field) == 1 and self.isTField(f0):
                order = self.transformNaturalOrder(df, f0)
                order_filled_by_transform = True
            else:
                if fill_transform and order is None and len(field) > 1:
                    for _field in field:
                        if self.isTField(_field):
                            raise Exception('xy - fill_transform is specified but there are multiple fields with a least one transform... create your own order...')
                order_filled_by_transform = False
                
            gb = df.groupby(field)
            if order is None:
                # 0...1 assignment
                if len(gb) >= 2:
                    my_inc = self.XYInc(1.0/(len(gb)-1))
                else:
                    my_inc = self.XYInc(1.0/len(gb))
                df[new_axis_field] = gb[field[0]].transform(lambda x: my_inc.nextValue(x))
                # Labeling
                gb_df = gb.size().reset_index() # is this the most optimal?
                label_min = str(gb_df.iloc[ 0][f0])
                label_max = str(gb_df.iloc[-1][f0])
                for i in range(1,len(field)):
                    label_min += '|'+str(gb_df.iloc[ 0][field[i]])
                    label_max += '|'+str(gb_df.iloc[-1][field[i]])
            else:
                # 0...1 assignment
                gb_set    = set(gb.groups.keys())
                order_set = set(order)
                order_is_complete = (len(gb_set) == len(order_set)) and (len(gb_set & order_set) == len(order_set))
                my_inc = self.OrderInc(order, order_is_complete == False)
                df[new_axis_field] = gb[field[0]].transform(lambda x: my_inc.nextValue(x))
                # Labeling
                label_min = order[0]
                if order_is_complete or order_filled_by_transform:
                    label_max = order[-1]
                else:
                    label_max = 'ee' # everthing else
                
        return is_time, label_min, label_max, transFunc, order

    #
    # XYInc... simple incrememter to handle non-numeric coordinate axes
    #
    class XYInc():
        def __init__(self, inc_amount):
            self.my_var = 0
            self.inc    = inc_amount
        def nextValue(self, x):
            my_var_copy = self.my_var
            self.my_var += self.inc
            return my_var_copy

    #
    # OrderInc... order by a specified array of values or tuples
    # ... if the value is not in the array, will be mapped to 1.0
    # ... _reserve_na == reserve space for elements that aren't in the order list...
    #
    class OrderInc():
        def __init__(self, _order, _reserve_na):
            self._order      = _order
            self._reserve_na = _reserve_na
        def nextValue(self, x):
            if x.name in self._order:
                if self._reserve_na:
                    return self._order.index(x.name)/len(self._order)
                else:
                    return self._order.index(x.name)/(len(self._order)-1)
            return 1.0

    #
    # For background context, transform an existing path description using the transforms and return as an SVG path.
    #
    def __transformPathDescription__(self,
                                     name,
                                     shape_desc,
                                     x_trans_func,
                                     y_trans_func,
                                     bg_shape_label_color,
                                     bg_shape_opacity,
                                     bg_shape_fill,
                                     bg_shape_stroke_w,
                                     bg_shape_stroke,
                                     txt_h):
        svg = '<path d="'
        x0,y0,x1,y1 = None,None,None,None
        shape_desc = " ".join(shape_desc.split()) # make sure there's no extra whitespaces
        tokens = shape_desc.lower().split(' ')
        i = 0
        while i < len(tokens):
            if   tokens[i] == 'm':
                _x,_y = x_trans_func(float(tokens[i+1])),y_trans_func(float(tokens[i+2]))
                svg += f' M {_x} {_y}'
                x0,y0,x1,y1 = self.__minsAndMaxes__(_x,_y,x0,y0,x1,y1)
                i += 3
            elif tokens[i] == 'l':
                _x,_y = x_trans_func(float(tokens[i+1])),y_trans_func(float(tokens[i+2]))
                svg += f' L {x_trans_func(float(tokens[i+1]))} {y_trans_func(float(tokens[i+2]))}'
                x0,y0,x1,y1 = self.__minsAndMaxes__(_x,_y,x0,y0,x1,y1)
                i += 3
            elif tokens[i] == 'c':
                _x_cp1,_y_cp1 = x_trans_func(float(tokens[i+1])),y_trans_func(float(tokens[i+2]))
                _x_cp2,_y_cp2 = x_trans_func(float(tokens[i+3])),y_trans_func(float(tokens[i+4]))
                _x,_y         = x_trans_func(float(tokens[i+5])),y_trans_func(float(tokens[i+6]))
                svg += f' C {_x_cp1} {_y_cp1} {_x_cp2} {_y_cp2} {_x} {_y}'
                x0,y0,x1,y1 = self.__minsAndMaxes__(_x,    _y,    x0,y0,x1,y1)
                x0,y0,x1,y1 = self.__minsAndMaxes__(_x_cp1,_y_cp1,x0,y0,x1,y1)
                x0,y0,x1,y1 = self.__minsAndMaxes__(_x_cp2,_y_cp2,x0,y0,x1,y1)
                i += 7
            elif tokens[i] == 'z':
                svg += ' Z'
                i += 1
            else:
                raise Exception(f'__transformPathDescription__() - does not handle path description "{tokens[i]}"')
        svg += '"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill, 
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>',self.__backgroundShapeLabel__(name, x0, y0, x1, y1, bg_shape_label_color, txt_h)
    
    #
    # For background context, transform a points list of coordinate and return as an SVG path.
    #
    def __transformPointsList__(self,
                                name,
                                points_list,
                                x_trans_func,
                                y_trans_func,
                                bg_shape_label_color,    # None = no label, 'vary', lookup to hash color, or a hash color
                                bg_shape_opacity,        # None (== 0.0), number, lookup to opacity
                                bg_shape_fill,           # None, 'vary', lookup to hash color, or a hash color
                                bg_shape_stroke_w,       # None, number, lookup to width
                                bg_shape_stroke,         # None, 'default', lookup to hash color, or a hash color
                                txt_h):
        _x,_y = x_trans_func(points_list[0][0]),y_trans_func(points_list[0][1])
        svg = f'<path d="M {_x} {_y}'
        x0,y0,x1,y1 = _x,_y,_x,_y
        for i in range(1, len(points_list)):
            _x,_y = x_trans_func(points_list[i][0]),y_trans_func(points_list[i][1])
            svg += f' L {_x} {_y}'
            x0,y0,x1,y1 = self.__minsAndMaxes__(_x,_y,x0,y0,x1,y1)

        svg += ' Z"'
        svg += self.__backgroundShapeRenderDetails__(name, bg_shape_opacity, bg_shape_fill, 
                                                     bg_shape_stroke_w, bg_shape_stroke)
        return svg + '/>',self.__backgroundShapeLabel__(name, x0, y0, x1, y1, bg_shape_label_color, txt_h)
    
    #
    # Simplify Min and Max Calculation For Bounding Box
    #
    def __minsAndMaxes__(self, x, y, x0, y0, x1, y1):
        if x0 is None:
            return x,y,x,y
        else:
            if x < x0:
                x0 = x
            if x > x1:
                x1 = x
            if y < y0:
                y0 = y
            if y > y1:
                y1 = y
            return x0,y0,x1,y1

    #
    # Add the render details for the background string...
    #
    def __backgroundShapeRenderDetails__(self,
                                         name, 
                                         bg_shape_opacity,       # None (== 0.0), number, lookup to opacity
                                         bg_shape_fill,          # None, 'vary', lookup to hash color, or a hash color
                                         bg_shape_stroke_w,      # None, number, lookup to width
                                         bg_shape_stroke):       # None, 'default', lookup to hash color, or a hash color
        svg =''
        # Fill
        if bg_shape_fill is not None and bg_shape_opacity is not None:
            # Handle opacity
            _opacity = 1.0
            if type(bg_shape_opacity) == dict:
                if name in bg_shape_opacity.keys():
                    _opacity = bg_shape_opacity[name]
                else:
                    _opacity = 1.0
            else:
                _opacity = bg_shape_opacity

            svg += f' fill-opacity="{_opacity}"'

            # Handle fill
            if    type(bg_shape_fill) == dict and name in bg_shape_fill.keys():
                _co = bg_shape_fill[name]
            elif  bg_shape_fill == 'vary':
                _co = self.co_mgr.getColor(name)
            elif  type(bg_shape_fill) == str and bg_shape_fill.startswith('#') and len(bg_shape_fill) == 7:
                _co = bg_shape_fill
            else:
                _co = self.co_mgr.getTVColor('context','default')

            svg += f' fill="{_co}"'
        else:
            svg += f' fill-opacity="0.0"'

        # Outline stroke
        if bg_shape_stroke_w is not None and bg_shape_stroke is not None:
            if   bg_shape_stroke == 'vary':
                _co = self.co_mgr.getColor(name)
            elif type(bg_shape_stroke) == str and bg_shape_stroke.startswith('#') and len(bg_shape_stroke) == 7:
                _co = bg_shape_stroke
            elif type(bg_shape_stroke) == dict and name in bg_shape_stroke.keys():
                _co = bg_shape_stroke[name] 
            else:
                _co =self.co_mgr.getTVColor('context','text')

            _wi = 1.0
            if type(bg_shape_stroke_w) == dict and name in bg_shape_stroke_w.keys():
                _wi = bg_shape_stroke_w[name]
            else:
                _wi = bg_shape_stroke_w

            svg += f' stroke="{_co}" stroke-width="{_wi}"'

        return svg


    #
    # Label for Background Shapes
    #
    def __backgroundShapeLabel__(self,
                                 name, 
                                 x0, y0, x1, y1, 
                                 bg_shape_label_color,       # None = no label, 'vary', lookup to hash color, or a hash color 
                                 txt_h):
        if bg_shape_label_color is not None:

            if    type(bg_shape_label_color) == dict and name in bg_shape_label_color.keys():
                _co = bg_shape_label_color[name]
            elif  bg_shape_label_color == 'vary':
                _co = self.co_mgr.getColor(name)
            elif  type(bg_shape_label_color) == str and bg_shape_label_color.startswith('#') and len(bg_shape_label_color) == 7:
                _co = bg_shape_label_color
            else:
                _co = self.co_mgr.getTVColor('context','text')

            # svg = f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill-opacity="0.0", stroke="#ff0000" />'
            svg = ''
            svg += f'<text x="{(x0+x1)/2}" y="{txt_h/2 + (y0+y1)/2}" text-anchor="middle" '
            svg +=   f'font-family="{self.default_font}" fill="{_co}" font-size="{txt_h}px">'
            svg +=   f'{name}</text>'
            return svg
        else:
            return ''
    
    #
    # xyInstance() - create an RTXy Instance
    #
    def xyInstance(self,
                   df,                                   # dataframe to render
                   x_field,                              # string or an array of strings
                   y_field,                              # string or an array of strings
                   # ----------------------------------- # everything else is a default...
                   x_field_is_scalar       = True,       # default... logic will check in the method to determine if this is true
                   y_field_is_scalar       = True,       # default... logic will check in the method to determine if this is true
                   color_by                = None,       # just the default color or a string for a field
                   color_magnitude         = None,       # Only applies when color_by is None, options: None / 'linear' / 'log' / 'stretch'
                   count_by                = None,       # none means just count rows, otherwise, use a field to sum by # Not Implemented
                   count_by_set            = False,      # count by summation (by default)... column is checked
                   dot_size                = 'medium',   # Dot size - ['small', 'medium', 'large', 'vary', 'hidden'/None]
                   dot_shape               = 'ellipse',  # Dot shape - ['square', 'ellipse', 'triangle, 'utriangle', 'diamond', 'plus', 'x', 'small_multiple', function pointer]
                   max_dot_size            = 5,          # Max dot size (used when the dot sz varies)
                   opacity                 = 1.0,        # Opacity of the dots
                   vary_opacity            = False,      # If true, vary opacity by the count_by # Not Implemented
                   align_pixels            = True,       # Align pixels to integer values
                   widget_id               = None,       # naming the svg elements
                   # ----------------------------------- # used for globally making the same scale/etc
                   x_axis_col              = None,       # x axis column name
                   x_is_time               = False,      # x is time flag
                   x_label_min             = None,       # min label on the x axis
                   x_label_max             = None,       # max label on the x axis
                   x_trans_func            = None,       # lambda transform function for x axis
                   x_order                 = None,       # order of x axis for categorical values
                   x_fill_transforms       = True,       # for t-fields, fill in all the values to properly space out data
                   y_axis_col              = None,       # y axis column name
                   y_is_time               = False,      # y is time flag
                   y_label_min             = None,       # min label on the y axis
                   y_label_max             = None,       # max label on the y axis
                   y_trans_func            = None,       # lambeda transform function for y axis
                   y_order                 = None,       # order of y axis for categorical values
                   y_fill_transforms       = True,       # for t-fields, fill in all the values to properly space out data
                   # ----------------------------------- # x = timestamp options // Only applies if x-axis is time
                   line_groupby_field      = None,       # will use a field to perform a groupby for a line chart
                                                         # calling app should make sure that all timeslots are filled in...
                   line_groupby_w          = 1,          # width of the line for the line chart
                   # ----------------------------------- # secondary axis settings # probably not small multiple safe...
                   df2                     = None,       # secondary axis dataframe ... if not set but y2_field is, then this will be set to df field
                   df2_ts_field            = None,       # secondary axis x timestamp field ... if not set but the y2_field is, then this be set to the x_field
                   y2_field                = None,       # secondary axis field ... if this is set, then df2 will be set to df // only required field really...
                   y2_field_is_scalar      = True,       # default... logic will check in the method to determine if this is true
                   y2_axis_col             = None,       # y2 axis column name
                   line2_groupby_field     = None,       # secondary line field ... will NOT be set
                   line2_groupby_w         = 0.75,       # secondary line field width
                   line2_groupby_color     = None,       # line2 color... if none, pulls from the color_by field
                   line2_groupby_dasharray = "4 2",      # line2 dasharray
                   dot2_size               = 'inherit',  # dot2 size... 'inherit' means to take from dot_size
                   # ----------------------------------- # small multiple options
                   sm_type                 = None,       # should be the method name // similar to the smallMultiples method
                   sm_w                    = None,       # override the width of the small multiple
                   sm_h                    = None,       # override the height of the small multiple
                   sm_params               = {},         # dictionary of parameters for the small multiples
                   sm_x_axis_independent   = True,       # Use independent axis for x (xy, temporal, and linkNode)
                   sm_y_axis_independent   = True,       # Use independent axis for y (xy, temporal, periodic, pie)
                   # ----------------------------------- # background information
                   bg_shape_lu             = None,       # lookup for background shapes -- key will be used for varying colors
                                                         # ['key'] = [(x0,y0),(x1,y1),...] OR
                                                         # ['key'] = svg path description
                   bg_shape_label_color  = None,         # None = no label, 'vary', lookup to hash color, or a hash color
                   bg_shape_opacity      = 1.0,          # None (== 0.0), number, lookup to opacity
                   bg_shape_fill         = None,         # None, 'vary', lookup to hash color, or a hash color
                   bg_shape_stroke_w     = 1.0,          # None, number, lookup to width
                   bg_shape_stroke       = 'default',    # None, 'default', lookup to hash color, or a hash color
                   # ----------------------------------- # Distributions
                   render_x_distribution       = None,       # number of x distribution buckets ... None == don't render
                   render_y_distribution       = None,       # number of x distribution buckets ... None == don't render
                   render_distribution_opacity = 0.5,        # Opacity of the distribution render
                   distribution_h_perc         = 0.33,       # height of the distribution as a function of the overall h of xy chart     
                   distribution_style          = 'outside',  # 'outside' - outside of xy... 'inside' - inside of xy
                   # ------------------------------------ # visualization geometry / etc.
                   x_view                   = 0,          # x offset for the view
                   y_view                   = 0,          # y offset for the view
                   x_ins                    = 3,          # side inserts
                   y_ins                    = 3,          # top & bottom inserts
                   w                        = 256,        # width of the view
                   h                        = 256,        # height of the view
                   txt_h                    = 12,         # text height for labeling
                   background_opacity       = 1.0,        # background opacity
                   background_override      = None,       # override the background color // hex value
                   plot_background_override = None,       # plot background override // hex value
                   draw_x_gridlines         = False,      # draw x gridlines for scalar values
                   draw_y_gridlines         = False,      # draw y gridlines for scalar values
                   draw_labels              = True,       # draw labels flag
                   draw_border              = True,       # draw a border around the histogram
                   draw_context             = True):      # draw temporal context information if (and only if) x_axis is time)
        return self.RTXy(self, df, x_field, y_field, x_field_is_scalar=x_field_is_scalar, y_field_is_scalar=y_field_is_scalar, color_by=color_by, color_magnitude=color_magnitude,                         
                         count_by=count_by, count_by_set=count_by_set, dot_size=dot_size, dot_shape=dot_shape, max_dot_size=max_dot_size, opacity=opacity, vary_opacity=vary_opacity, align_pixels=align_pixels,
                         widget_id=widget_id, x_axis_col=x_axis_col, x_is_time=x_is_time, x_label_min=x_label_min, x_label_max=x_label_max, x_trans_func=x_trans_func, y_axis_col=y_axis_col,
                         y_is_time=y_is_time, y_label_min=y_label_min, y_label_max=y_label_max, y_trans_func=y_trans_func, x_order=x_order, y_order=y_order, 
                         x_fill_transforms=x_fill_transforms, y_fill_transforms=y_fill_transforms,
                         line_groupby_field=line_groupby_field, line_groupby_w=line_groupby_w,
                         df2=df2, df2_ts_field=df2_ts_field, y2_field=y2_field, y2_field_is_scalar=y2_field_is_scalar, y2_axis_col=y2_axis_col, line2_groupby_field=line2_groupby_field,
                         line2_groupby_w=line2_groupby_w, line2_groupby_color=line2_groupby_color, line2_groupby_dasharray=line2_groupby_dasharray, dot2_size=dot2_size,
                         sm_type=sm_type, sm_w=sm_w, sm_h=sm_h, sm_params=sm_params, sm_x_axis_independent=sm_x_axis_independent, sm_y_axis_independent=sm_y_axis_independent,
                         render_x_distribution=render_x_distribution, render_y_distribution=render_y_distribution, render_distribution_opacity=render_distribution_opacity,
                         distribution_h_perc=distribution_h_perc, distribution_style=distribution_style,
                         bg_shape_lu=bg_shape_lu, bg_shape_label_color=bg_shape_label_color, bg_shape_opacity=bg_shape_opacity, bg_shape_fill=bg_shape_fill,
                         bg_shape_stroke_w=bg_shape_stroke_w, bg_shape_stroke=bg_shape_stroke, x_view=x_view, y_view=y_view, x_ins=x_ins, y_ins=y_ins, w=w, h=h, txt_h=txt_h,
                         background_opacity=background_opacity, background_override=background_override, plot_background_override=plot_background_override,
                         draw_x_gridlines=draw_x_gridlines, draw_y_gridlines=draw_y_gridlines,
                         draw_labels=draw_labels, draw_border=draw_border, draw_context=draw_context)

    #
    # RTXy
    #
    class RTXy(object):
        def __init__(self,
                     rt_self,                              # outer class
                     df,                                   # dataframe to render
                     x_field,                              # string or an array of strings
                     y_field,                              # string or an array of strings
                     # ----------------------------------- # everything else is a default...
                     x_field_is_scalar       = True,       # default... logic will check in the method to determine if this is true
                     y_field_is_scalar       = True,       # default... logic will check in the method to determine if this is true
                     color_by                = None,       # just the default color or a string for a field
                     color_magnitude         = None,       # Only applies when color_by is None, options: None / 'linear' / 'log' / 'stretch'
                     count_by                = None,       # none means just count rows, otherwise, use a field to sum by # Not Implemented
                     count_by_set            = False,      # count by summation (by default)... column is checked
                     dot_size                = 'medium',   # Dot size - ['small', 'medium', 'large', 'vary', 'hidden'/None]
                     dot_shape               = 'ellipse',  # Dot shape - ['square', 'ellipse', 'triangle, 'utriangle', 'diamond', 'plus', 'x', 'small_multiple', function pointer]
                     max_dot_size            = 5,          # Max dot size (used when the dot sz varies)
                     opacity                 = 1.0,        # Opacity of the dots
                     vary_opacity            = False,      # If true, vary opacity by the count_by # Not Implemented
                     align_pixels            = True,       # Align pixels to integer values
                     widget_id               = None,       # naming the svg elements
                     # ----------------------------------- # used for globally making the same scale/etc
                     x_axis_col              = None,       # x axis column name
                     x_is_time               = False,      # x is time flag
                     x_label_min             = None,       # min label on the x axis
                     x_label_max             = None,       # max label on the x axis
                     x_trans_func            = None,       # lambda transform function for x axis
                     x_order                 = None,       # order of categorical values on x axis
                     x_fill_transforms       = True,       # for t-fields, fill in all the values to properly space out data
                     y_axis_col              = None,       # y axis column name
                     y_is_time               = False,      # y is time flag
                     y_label_min             = None,       # min label on the y axis
                     y_label_max             = None,       # max label on the y axis
                     y_trans_func            = None,       # lambeda transform function for y axis
                     y_order                 = None,       # order of categorical values on y axis
                     y_fill_transforms       = True,       # for t-fields, fill in all the values to properly space out data
                     # ----------------------------------- # x = timestamp options // Only applies if x-axis is time
                     line_groupby_field      = None,       # will use a field to perform a groupby for a line chart
                                                           # calling app should make sure that all timeslots are filled in...
                     line_groupby_w          = 1,          # width of the line for the line chart
                     # ----------------------------------- # secondary axis settings # probably not small multiple safe...
                     df2                     = None,       # secondary axis dataframe ... if not set but y2_field is, then this will be set to df field
                     df2_ts_field            = None,       # secondary axis x timestamp field ... if not set but the y2_field is, then this be set to the x_field
                     y2_field                = None,       # secondary axis field ... if this is set, then df2 will be set to df // only required field really...
                     y2_field_is_scalar      = True,       # default... logic will check in the method to determine if this is true
                     y2_axis_col             = None,       # y2 axis column name
                     line2_groupby_field     = None,       # secondary line field ... will NOT be set // wish i had given this "not be set" a description
                     line2_groupby_w         = 0.75,       # width of the line2 groupby
                     line2_groupby_color     = None,       # line2 color... if none, pulls from the color_by field
                     line2_groupby_dasharray = "4 2",      # line2 dasharray
                     dot2_size               = 'inherit',  # dot2 size -- 'inherit' means to take from the dot_size
                     # ----------------------------------- # small multiple options
                     sm_type                 = None,       # should be the method name // similar to the smallMultiples method
                     sm_w                    = None,       # override the width of the small multiple
                     sm_h                    = None,       # override the height of the small multiple
                     sm_params               = {},         # dictionary of parameters for the small multiples
                     sm_x_axis_independent   = True,       # Use independent axis for x (xy, temporal, and linkNode)
                     sm_y_axis_independent   = True,       # Use independent axis for y (xy, temporal, periodic, pie)
                     # ----------------------------------- # background information
                     bg_shape_lu             = None,       # lookup for background shapes -- key will be used for varying colors
                                                           # ['key'] = [(x0,y0),(x1,y1),...] OR
                                                           # ['key'] = svg path description

                     bg_shape_label_color  = None,         # None = no label, 'vary', lookup to hash color, or a hash color
                     bg_shape_opacity      = 1.0,          # None (== 0.0), number, lookup to opacity
                     bg_shape_fill         = None,         # None, 'vary', lookup to hash color, or a hash color
                     bg_shape_stroke_w     = 1.0,          # None, number, lookup to width
                     bg_shape_stroke       = 'default',    # None, 'default', lookup to hash color, or a hash color
                     # ----------------------------------- # Distributions
                     render_x_distribution       = None,      # number of x distribution buckets ... None == don't render
                     render_y_distribution       = None,      # number of x distribution buckets ... None == don't render
                     render_distribution_opacity = 0.5,       # Opacity of the distribution render
                     distribution_h_perc         = 0.33,      # height of the distribution as a function of the overall h of xy chart     
                     distribution_style          = 'outside', # 'outside' - outside of xy... 'inside' - inside of xy
                     # ------------------------------------ # visualization geometry / etc.
                     x_view                   = 0,          # x offset for the view
                     y_view                   = 0,          # y offset for the view
                     x_ins                    = 3,          # side inserts
                     y_ins                    = 3,          # top & bottom inserts
                     w                        = 256,        # width of the view
                     h                        = 256,        # height of the view
                     txt_h                    = 12,         # text height for labeling
                     background_opacity       = 1.0,        # background opacity
                     background_override      = None,       # override the background color // hex value
                     plot_background_override = None,       # override the plot bckground // hex value
                     draw_x_gridlines         = False,      # draw x gridlines for scalar values
                     draw_y_gridlines         = False,      # draw y gridlines for scalar values
                     draw_labels              = True,       # draw labels flag
                     draw_border              = True,       # draw a border around the histogram
                     draw_context             = True):      # draw temporal context information if (and only if) x_axis is time):
            self.parms                   = locals().copy()
            self.rt_self                 = rt_self
            self.df                      = df.copy()
            self.x_field                 = x_field
            self.y_field                 = y_field
            self.x_field_is_scalar       = x_field_is_scalar 
            self.y_field_is_scalar       = y_field_is_scalar 
            self.color_by                = color_by 
            self.color_magnitude         = color_magnitude                         
            self.count_by                = count_by
            self.count_by_set            = count_by_set
            self.dot_size                = dot_size
            self.dot_shape               = dot_shape
            self.max_dot_size            = max_dot_size
            self.opacity                 = opacity
            self.vary_opacity            = vary_opacity
            self.align_pixels            = align_pixels
            self.widget_id               = widget_id

            # Make a widget_id if it's not set already
            if self.widget_id is None:
                self.widget_id = "xy_" + str(random.randint(0,65535))

            self.x_axis_col              = x_axis_col
            self.x_is_time               = x_is_time
            self.x_label_min             = x_label_min
            self.x_label_max             = x_label_max
            self.x_trans_func            = x_trans_func
            self.x_order                 = x_order
            self.x_fill_transforms       = x_fill_transforms
            self.y_axis_col              = y_axis_col
            self.y_is_time               = y_is_time
            self.y_label_min             = y_label_min
            self.y_label_max             = y_label_max
            self.y_trans_func            = y_trans_func
            self.y_order                 = y_order
            self.y_fill_transforms       = y_fill_transforms
            self.line_groupby_field      = line_groupby_field
            self.line_groupby_w          = line_groupby_w

            if df2 is not None:
                self.df2                 = df2.copy()
            elif y2_field is not None:
                self.df2                 = df.copy()
                self.df2_ts_field        = x_field
            else:
                self.df2                 = None

            self.df2_ts_field            = df2_ts_field
            self.y2_field                = y2_field
            self.y2_field_is_scalar      = y2_field_is_scalar
            self.y2_axis_col             = y2_axis_col
            self.line2_groupby_field     = line2_groupby_field
            self.line2_groupby_w         = line2_groupby_w
            self.line2_groupby_color     = line2_groupby_color
            self.line2_groupby_dasharray = line2_groupby_dasharray
            self.dot2_size               = dot2_size
            if dot2_size == 'inherit':
                self.dot2_size = dot_size

            self.sm_type                 = sm_type
            self.sm_w                    = sm_w
            self.sm_h                    = sm_h
            self.sm_params               = sm_params
            self.sm_x_axis_independent   = sm_x_axis_independent
            self.sm_y_axis_independent   = sm_y_axis_independent

            self.bg_shape_lu             = bg_shape_lu
            self.bg_shape_label_color    = bg_shape_label_color
            self.bg_shape_opacity        = bg_shape_opacity
            self.bg_shape_fill           = bg_shape_fill
            self.bg_shape_stroke_w       = bg_shape_stroke_w
            self.bg_shape_stroke         = bg_shape_stroke

            self.render_x_distribution       = render_x_distribution
            self.render_y_distribution       = render_y_distribution
            self.render_distribution_opacity = render_distribution_opacity
            self.distribution_h_perc         = distribution_h_perc
            self.distribution_style          = distribution_style

            self.x_view                   = x_view
            self.y_view                   = y_view
            self.x_ins                    = x_ins
            self.y_ins                    = y_ins
            self.w                        = w
            self.h                        = h
            self.txt_h                    = txt_h
            self.background_opacity       = background_opacity
            self.background_override      = background_override
            self.plot_background_override = plot_background_override
            self.draw_x_gridlines         = draw_x_gridlines
            self.draw_y_gridlines         = draw_y_gridlines
            self.draw_labels              = draw_labels
            self.draw_border              = draw_border
            self.draw_context             = draw_context

            # Check the dot information
            if self.dot_shape == 'small_multiple':
                self.dot_size = 'medium' # put a valid value in here
                if self.sm_type is None:
                    self.dot_shape = 'ellipse'
                    self.dot_size  = 'small'
                elif self.sm_w is None or self.sm_h is None:
                    self.sm_w,self.sm_h = getattr(rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
            elif callable(self.dot_shape) and self.dot_size is None:
                self.dot_size = 'medium'

            # Make sure x_field and y_field are lists...
            if type(self.x_field) != list: # Make it into a list for consistency
                self.x_field = [self.x_field]
            if type(self.y_field) != list: # Make it into a list for consistency
                self.y_field = [self.y_field]
            if self.y2_field is not None and type(self.y2_field) != list: # Make it into a list for consistency
                self.y2_field = [self.y2_field]

            #
            # Transforms section
            #
                
            # Apply bin-by transforms
            self.df, self.x_field   = rt_self.transformFieldListAndDataFrame(self.df, self.x_field)
            self.df, self.y_field   = rt_self.transformFieldListAndDataFrame(self.df, self.y_field)
            if self.y2_field: # just the y2_field here... not the corresponding x2_field
                self.df2, self.y2_field = rt_self.transformFieldListAndDataFrame(self.df2, self.y2_field)

            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)
            if self.y2_field is not None and self.count_by is not None and rt_self.isTField(self.count_by):
                self.df2,self.count_by = rt_self.applyTransform(self.df2, self.count_by) # should be the same field name... i.e., count_by column needs to be in both df and df2

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)
            if self.y2_field is not None and self.color_by is not None and rt_self.isTField(self.color_by):
                self.df2,self.color_by = rt_self.applyTransform(self.df2, self.color_by) # should be the same field name

            # Check the count_by column
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)

            # Setup the y2 info (if the y2_field is set)
            self.timestamp_min = None
            self.timestamp_max = None
            if len(self.x_field) == 1 and is_datetime(self.df[self.x_field[0]]):
                self.timestamp_min = self.df[self.x_field[0]].min()
                self.timestamp_max = self.df[self.x_field[0]].max()
                if self.y2_field is not None:
                    if self.df2 is None:
                        self.df2 = self.df
                        self.df2_is_df = True
                    else:
                        self.df2_is_df = False
                    if self.df2_ts_field is None:
                        self.df2_ts_field = self.x_field
                    if type(self.df2_ts_field) != list:
                        self.df2_ts_field = [self.df2_ts_field]
                    
                    # Determine actual timestamp min and max 
                    if self.df2[self.df2_ts_field[0]].min() < self.timestamp_min:
                        self.timestamp_min = df2[df2_ts_field[0]].min()
                    if self.df2[self.df2_ts_field[0]].max() > self.timestamp_max:
                        self.timestamp_max = self.df2[self.df2_ts_field[0]].max()
                else:
                    self.y2_field = self.df2 = self.line2_groupby_field = self.df2_ts_field = None

        #
        # renderSVG() - render as SVG
        #
        def renderSVG(self):
            #
            # Geometry
            #
            params_orig_minus_self = self.parms.copy()
            params_orig_minus_self.pop('self')
            min_dims = self.rt_self.xySmallMultipleDimensions(**params_orig_minus_self)
            if self.w < min_dims[0] or self.h < min_dims[1]:
                self.draw_labels  = False
                self.draw_context = False
                self.x_ins        = 1
                self.y_ins        = 1

            # Turn off labels if they are proportionally too large
            if (6*self.txt_h) > self.w or (6*self.txt_h) > self.h:
                self.draw_labels = False

            # Actual geometry...
            if self.draw_labels:
                if self.y2_field is None:
                    self.w_usable = self.w - (2*self.x_ins + self.txt_h    + 4)
                else:
                    self.w_usable = self.w - (2*self.x_ins + 2*self.txt_h  + 4) # give space for other y axis on the right side

                self.x_left   =             self.x_ins + self.txt_h + 2
                self.y_bottom = self.h -    self.y_ins - self.txt_h - 2
                self.h_usable = self.h - (2*self.y_ins + self.txt_h + 4)
            else:
                self.x_left   = self.x_ins
                self.y_bottom = self.h - self.y_ins
                self.w_usable = self.w - (2*self.x_ins)
                self.h_usable = self.h - (2*self.y_ins)

            # Give the distribution renders a third of the space
            if self.distribution_style == 'outside':
                if self.render_x_distribution:
                    self.x_distribution_h = self.h_usable * self.distribution_h_perc # DIST_GEOM // search for DIST_GEOM to find related calcs
                    self.h_usable         = self.h_usable - self.x_distribution_h
                    self.y_bottom         = self.y_ins + self.h_usable
                if self.render_y_distribution:
                    self.y_distribution_h = self.w_usable * self.distribution_h_perc # DIST_GEOM // search for DIST_GEOM to find related calcs
                    self.w_usable         = self.w_usable - self.y_distribution_h
            else:
                self.x_distribution_h = self.h_usable * self.distribution_h_perc
                self.y_distribution_h = self.w_usable * self.distribution_h_perc

            # dot_w will be used for the actual geometry
            if self.dot_size is None or self.dot_size == 'hidden':
                dot_w =  None
            elif self.dot_size == 'medium':
                dot_w =  2
            elif self.dot_size == 'small':
                dot_w =  1
            elif self.dot_size == 'large':
                dot_w =  3
            else:
                dot_w = -1

            # dot2_w ... should refactor...
            if self.dot2_size is None or self.dot2_size == 'hidden':
                dot2_w =  None
            elif self.dot2_size == 'medium':
                dot2_w =  2
            elif self.dot2_size == 'small':
                dot2_w =  1
            elif self.dot2_size == 'large':
                dot2_w =  3
            else:
                dot2_w = -1

            # Create the extra columns for the x and y coordinates
            if self.x_axis_col is None:
                self.x_axis_col = 'my_x_' + self.widget_id
                self.x_is_time, self.x_label_min, self.x_label_max, self.x_trans_func, self.x_order = self.rt_self.xyCreateAxisColumn(self.df, self.x_field, self.x_field_is_scalar, self.x_axis_col, self.x_order, self.x_fill_transforms, self.timestamp_min, self.timestamp_max)
            if self.y_axis_col is None:
                self.y_axis_col = 'my_y_' + self.widget_id
                self.y_is_time, self.y_label_min, self.y_label_max, self.y_trans_func, self.y_order = self.rt_self.xyCreateAxisColumn(self.df, self.y_field, self.y_field_is_scalar, self.y_axis_col, self.y_order, self.y_fill_transforms)

            # Secondary axis settings
            self.y2_label_min, self.y2_label_max = None,None
            if self.y2_field is not None and self.y2_axis_col is None:
                self.y2_axis_col = 'my_y2_' + self.widget_id
                self.y2_is_time, self.y2_label_min, self.y2_label_max, _throwaway_func = self.rt_self.xyCreateAxisColumn(self.df2, self.y2_field, self.y2_field_is_scalar, self.y2_axis_col)
            if self.y2_field is not None:
                if self.df2_is_df:
                    self.x2_axis_col = self.x_axis_col
                else:
                    self.x2_axis_col = 'my_x2_' + self.widget_id
                    self.x2_is_time, self.x2_label_min, self.x2_label_max, _throwaway_func = self.rt_self.xyCreateAxisColumn(self.df2, self.df2_ts_field, False, self.x2_axis_col, None, self.timestamp_min, self.timestamp_max)

            # Create the pixel-level columns
            self.df[self.x_axis_col+"_px"] = self.x_left                + self.df[self.x_axis_col]*self.w_usable
            self.df[self.y_axis_col+"_px"] = self.y_ins + self.h_usable - self.df[self.y_axis_col]*self.h_usable
            if self.align_pixels:
                self.df[self.x_axis_col+"_px"] = self.df[self.x_axis_col+"_px"].astype(np.int32)
                self.df[self.y_axis_col+"_px"] = self.df[self.y_axis_col+"_px"].astype(np.int32)

            self.x_trans_norm_func = None
            if self.x_trans_func is not None:
                self.x_trans_norm_func = lambda x: self.x_left + self.x_trans_func(x) * self.w_usable
            self.y_trans_norm_func = None
            if self.y_trans_func is not None:
                self.y_trans_norm_func = lambda x: self.y_ins + self.h_usable - self.y_trans_func(x) * self.h_usable

            # Secondary axis pixel-level columns
            if self.y2_field:
                if self.df2_is_df == False:
                    self.df2[self.x2_axis_col+"_px"] = self.x_left  + self.df2[self.x2_axis_col]*self.w_usable
                    if self.align_pixels:
                        self.df2[self.x2_axis_col+"_px"] =                self.df2[self.x2_axis_col+"_px"].astype(np.int32)

                self.df2[self.y2_axis_col+"_px"] = self.y_ins + self.h_usable - self.df2[self.y2_axis_col]*self.h_usable
                if self.align_pixels:
                    self.df2[self.y2_axis_col+"_px"] =                              self.df2[self.y2_axis_col+"_px"].astype(np.int32)

            # Create the SVG ... render the background
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            if self.background_override is None:
                background_color = self.rt_self.co_mgr.getTVColor('background','default')
            else:
                background_color = self.background_override                
            svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" fill-opacity="{self.background_opacity}" stroke="{background_color}" stroke-opacity="{self.background_opacity}" />'

            if self.plot_background_override is not None:
                _co = self.plot_background_override
                svg += f'<rect x="{self.x_left}" y="{self.y_ins}" width="{self.w_usable}" height="{self.h_usable+1}" fill="{_co}" stroke="{_co}" />'

            # Draw the temporal context
            if self.x_is_time and self.draw_context:
                if self.x_label_min is not None and self.x_label_max is not None:
                    _ts_min,_ts_max = pd.to_datetime(self.x_label_min),pd.to_datetime(self.x_label_max)
                    svg += self.rt_self.drawXYTemporalContext(self.x_left, self.y_ins, self.w_usable, self.h_usable, self.txt_h, _ts_min,            _ts_max,            self.draw_labels)
                else:
                    svg += self.rt_self.drawXYTemporalContext(self.x_left, self.y_ins, self.w_usable, self.h_usable, self.txt_h, self.timestamp_min, self.timestamp_max, self.draw_labels)

            # Draw grid lines (if enabled)
            if self.draw_x_gridlines and self.x_field_is_scalar:
                svg += self.__drawGridlines__(True,  self.x_label_min, self.x_label_max, self.x_trans_norm_func, self.w_usable, self.y_ins,  self.y_ins  + self.h_usable)
            if self.draw_y_gridlines and self.y_field_is_scalar:
                svg += self.__drawGridlines__(False, self.y_label_min, self.y_label_max, self.y_trans_norm_func, self.h_usable, self.x_left, self.x_left + self.w_usable)
                
            # Draw the background shapes
            if self.bg_shape_lu is not None and self.x_trans_func is not None and self.y_trans_func is not None:
                _bg_shape_labels = []
                for k in self.bg_shape_lu.keys():
                    shape_desc = self.bg_shape_lu[k]
                    if   type(shape_desc) == str:   # path description
                        _shape_svg, _label_svg = self.rt_self.__transformPathDescription__(k,
                                                                                           shape_desc,
                                                                                           self.x_trans_norm_func,
                                                                                           self.y_trans_norm_func,
                                                                                           self.bg_shape_label_color,
                                                                                           self.bg_shape_opacity,
                                                                                           self.bg_shape_fill,
                                                                                           self.bg_shape_stroke_w,
                                                                                           self.bg_shape_stroke, self.txt_h)
                    elif type(shape_desc) == list:  # list of tuple pairs
                        _shape_svg, _label_svg = self.rt_self.__transformPointsList__(k,
                                                                                      shape_desc,
                                                                                      self.x_trans_norm_func,
                                                                                      self.y_trans_norm_func,
                                                                                      self.bg_shape_label_color,
                                                                                      self.bg_shape_opacity,
                                                                                      self.bg_shape_fill,
                                                                                      self.bg_shape_stroke_w,
                                                                                      self.bg_shape_stroke, self.txt_h)
                    else:
                        raise Exception(f'RTXy.renderSVG() - type "{type(shape_desc)}" as background lookup')

                    svg += _shape_svg
                    _bg_shape_labels.append(_label_svg) # Defer render

                # Render the labels
                for _label_svg in _bg_shape_labels:
                    svg += _label_svg

            # Draw the distributions (if selected)
            if self.render_x_distribution is not None:
                svg += self.__renderBackgroundDistribution__(True,  self.x_left, self.y_bottom, self.x_left + self.w_usable, self.y_bottom, self.x_left, self.y_ins)
            if self.render_y_distribution is not None:
                svg += self.__renderBackgroundDistribution__(False, self.x_left, self.y_bottom, self.x_left + self.w_usable, self.y_bottom, self.x_left, self.y_ins)

            # Axis
            axis_co = self.rt_self.co_mgr.getTVColor('axis',  'default')
            svg += f'<line x1="{self.x_left}" y1="{self.y_bottom}" x2="{self.x_left}"                 y2="{self.y_ins}"      stroke="{axis_co}" stroke-width=".6" />'
            svg += f'<line x1="{self.x_left}" y1="{self.y_bottom}" x2="{self.x_left + self.w_usable}" y2="{self.y_bottom}"   stroke="{axis_co}" stroke-width=".6" />'

            # Handle the line option... this needs to be rendered before the dots so that the lines are behind the dots
            #
            # ... first version handles timestamped vector data...
            #
            if self.line_groupby_field is not None     and \
               type(self.line_groupby_field) == list   and \
               is_datetime(self.df[self.line_groupby_field[-1]]):
                
                color = self.rt_self.co_mgr.getTVColor('data','default')
                _gb_fields = self.line_groupby_field[:-1]
                if len(_gb_fields) == 1:
                    _gb_fields = _gb_fields[0]

                _ts_field = self.line_groupby_field[-1]

                gb = self.df.groupby(_gb_fields)
                for k,k_df in gb:
                    gbxy = k_df.groupby([_ts_field, self.x_axis_col+"_px",self.y_axis_col+"_px"])
                    points = ''
                    for xy,xy_df in gbxy:
                        points += f'{xy[1]},{xy[2]} '
                    if self.color_by:
                        color_set = set(k_df[self.color_by])
                        if len(color_set) == 1:
                            color = self.rt_self.co_mgr.getColor(color_set.pop())
                        else:
                            color = self.rt_self.co_mgr.getTVColor('data','default')                            
                    if len(points) > 0:
                        svg += f'<polyline points="{points}" stroke="{color}" stroke-width="{self.line_groupby_w}" fill="none" />'
            #
            # ... second version handles the normal use cases...
            #
            elif self.line_groupby_field:
                color = self.rt_self.co_mgr.getTVColor('data','default')
                gb = self.df.groupby(self.line_groupby_field)
                for k,k_df in gb:
                    gbxy = k_df.groupby([self.x_axis_col+"_px",self.y_axis_col+"_px"])
                    points = ''
                    for xy,xy_df in gbxy:
                        points += f'{xy[0]},{xy[1]} '
                    if self.color_by:
                        color_set = set(k_df[self.color_by])
                        if len(color_set) == 1:
                            color = self.rt_self.co_mgr.getColor(color_set.pop())
                        else:
                            color = self.rt_self.co_mgr.getTVColor('data','default')                            
                    if len(points) > 0:
                        svg += f'<polyline points="{points}" stroke="{color}" stroke-width="{self.line_groupby_w}" fill="none" />'

            # Handle the line 2 option // like the first one... but some additional options, reassignments
            if self.line2_groupby_field:
                gb = self.df2.groupby(self.line2_groupby_field)
                for k,k_df in gb:
                    gbxy = k_df.groupby([self.x2_axis_col+"_px",self.y2_axis_col+"_px"])
                    points = ''
                    for xy,xy_df in gbxy:
                        points += f'{xy[0]},{xy[1]} '

                    if   self.line2_groupby_color:
                        if   self.line2_groupby_color.startswith('#'):
                            color = self.line2_groupby_color
                        elif self.line2_groupby_color in k_df.columns:
                            color_set = set(k_df[self.line2_groupby_color])
                            if len(color_set) == 1:
                                color = self.rt_self.co_mgr.getColor(color_set.pop())
                            else:
                                color = self.rt_self.co_mgr.getTVColor('data','default')                            
                        else:
                            color = self.rt_self.co_mgr.getTVColor('data','default')                            
                    elif self.color_by and self.color_by in k_df.columns:
                        color_set = set(k_df[self.color_by])
                        if len(color_set) == 1:
                            color = self.rt_self.co_mgr.getColor(color_set.pop())
                        else:
                            color = self.rt_self.co_mgr.getTVColor('data','default')                            
                    else:
                        color = self.rt_self.co_mgr.getTVColor('data','default')                            

                    if len(points) > 0:
                        if self.line2_groupby_dasharray:
                            svg += f'<polyline points="{points}" stroke="{color}" stroke-width="{self.line2_groupby_w}" fill="none" stroke-dasharray="{self.line2_groupby_dasharray}" />'
                        else:
                            svg += f'<polyline points="{points}" stroke="{color}" stroke-width="{self.line2_groupby_w}" fill="none" />'

            #
            # If we're going to draw dots...
            #
            node_to_xy  = {} # for small multiples
            node_to_dfs = {} # for small multiples

            if dot_w is not None or dot2_w is not None:
                #
                # Repeat for both axes
                #
                for y_axis_i in range(0,2):
                    if y_axis_i == 0:
                        if dot_w is not None:
                            _df,_x_axis_col,_y_axis_col,_local_color_by,_local_dot_w = self.df,self.x_axis_col,self.y_axis_col,self.color_by,dot_w
                        else:
                            continue
                    else:
                        if self.y2_field and dot2_w is not None:
                            _df,_x_axis_col,_y_axis_col,_local_color_by,_local_dot_w = self.df2,self.x2_axis_col,self.y2_axis_col,self.line2_groupby_color,dot2_w
                            if _local_color_by is None:
                                _local_color_by = self.color_by
                        else:
                            continue

                    #
                    # Group by x,y for the render
                    #
                    gb = _df.groupby([_x_axis_col+"_px",_y_axis_col+"_px"])

                    # Determine the min and max counts for the dot width / for contrast stretching, track counts
                    max_xy,self.stretch_histogram,self.stretch_total = 0,{},0
                    if _local_dot_w <= 0 or self.vary_opacity or self.color_magnitude is not None:
                        for k,k_df in gb:
                            # count by rows
                            if   self.count_by is None:
                                my_count = len(k_df)
                            # count by set
                            elif self.count_by_set:
                                my_count = len(set(k_df[self.count_by]))
                            # count by summation
                            else:
                                my_count = k_df[self.count_by].sum()
                            
                            if self.color_magnitude == 'stretch':
                                self.stretch_total += my_count
                                if my_count not in self.stretch_histogram.keys():
                                    self.stretch_histogram[my_count] =  1
                                else:
                                    self.stretch_histogram[my_count] += 1 
                                
                            if max_xy < my_count:
                                max_xy = my_count

                    # Make sure the max is not zero
                    if max_xy == 0:
                        max_xy = 1
                    
                    # Contrast stretch calculation
                    self.contrast_stretch = {}
                    if self.color_magnitude == 'stretch':
                        _total_so_far = 0
                        _sorted       = sorted(list(self.stretch_histogram.keys()))
                        for x in _sorted:
                            _perc = _total_so_far / self.stretch_total
                            self.contrast_stretch[x] = _perc
                            _total_so_far += self.stretch_histogram[x] * x

                    #
                    # Render loop
                    #
                    for k,k_df in gb:
                        x,y = k

                        # Small Multiple Version // mirrors the linknode version
                        if self.dot_shape == 'small_multiple':
                            xy_as_str = str(x) + ',' + str(y)
                            if xy_as_str not in node_to_dfs.keys():
                                node_to_xy [xy_as_str] = (x,y)
                                node_to_dfs[xy_as_str] = []
                            node_to_dfs[xy_as_str].append(k_df)

                        # Regular Version
                        else:                    
                            # Determine coloring options
                            if _local_color_by is None:
                                if self.color_magnitude is None:
                                    color = self.rt_self.co_mgr.getTVColor('data','default')
                                else:
                                    # count by rows
                                    if   self.count_by is None:
                                        my_count = len(k_df)
                                    # count by set
                                    elif self.count_by_set:
                                        my_count = len(set(k_df[self.count_by]))
                                    # count by summation
                                    else:
                                        my_count = k_df[self.count_by].sum()

                                    if self.color_magnitude == 'stretch':
                                        color = self.rt_self.co_mgr.spectrum(self.contrast_stretch[my_count], 0, 1.0,    True)
                                    else:
                                        color = self.rt_self.co_mgr.spectrum(my_count, 0, max_xy, self.color_magnitude)
                            elif _local_color_by in k_df.columns and pd.core.dtypes.common.is_datetime_or_timedelta_dtype(k_df[_local_color_by]):
                                _scaled_time = (k_df[_local_color_by].min() - self.df[_local_color_by].min())/(self.df[_local_color_by].max() - self.df[_local_color_by].min())
                                color        = self.rt_self.co_mgr.spectrum(_scaled_time, 0.0, 1.0)
                            elif _local_color_by in k_df.columns:
                                color_set = set(k_df[_local_color_by])
                                if len(color_set) == 1:
                                    color = self.rt_self.co_mgr.getColor(color_set.pop())
                                else:
                                    color = self.rt_self.co_mgr.getTVColor('data','default')
                            elif _local_color_by.startswith('#'):
                                color = _local_color_by
                            else:
                                color = self.rt_self.co_mgr.getTVColor('data','default')

                            
                            # Render the dot
                            # - Simple Render
                            if _local_dot_w > 0 and self.vary_opacity == False:
                                _my_dot_shape = self.dot_shape
                                if callable(self.dot_shape):
                                    _my_dot_shape = self.dot_shape(k_df, k, x, y, _local_dot_w, color, self.opacity)
                                svg += self.rt_self.renderShape(_my_dot_shape, x, y, _local_dot_w, color, None, self.opacity)
                                
                            # - Complex Render
                            else:
                                # count by rows
                                if   self.count_by is None:
                                    my_count = len(k_df)
                                # count by set
                                elif self.count_by_set:
                                    my_count = len(set(k_df[self.count_by]))
                                # count by summation
                                else:
                                    my_count = k_df[self.count_by].sum()
                                
                                var_w = _local_dot_w
                                var_o = 1.0 
                                if _local_dot_w <= 0 and self.vary_opacity:                    
                                    var_w = 0.2 + self.max_dot_size  * my_count/max_xy
                                    var_o = 0.2 + 0.8           * my_count/max_xy                    
                                elif _local_dot_w <= 0:
                                    var_w = 0.2 + self.max_dot_size  * my_count/max_xy                    
                                else:
                                    var_o = 0.2 + 0.8           * my_count/max_xy
                                
                                _my_dot_shape = self.dot_shape
                                if callable(self.dot_shape):
                                    _my_dot_shape = self.dot_shape(k_df, k, x, y, var_w, color, var_o)

                                svg += self.rt_self.renderShape(_my_dot_shape, x, y, var_w, color, None, var_o)

                # Handle the small multiples // mostly a copy from the linknode version
                if self.dot_shape == 'small_multiple':
                    _ts_field = None
                    if self.x_is_time:
                        _ts_field = self.x_field[0]

                    sm_lu = self.rt_self.createSmallMultiples(self.df, node_to_dfs, node_to_xy,
                                                              self.count_by, self.count_by_set, self.color_by, _ts_field, self.widget_id,
                                                              self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent,
                                                              self.sm_w, self.sm_h)
                    for node_str in sm_lu.keys():
                        svg += sm_lu[node_str]

            # Draw labels
            if self.draw_labels:
                #
                # X Axis
                #
                if self.x_is_time:
                    self.x_label_min,self.x_label_max = self.rt_self.condenseTimeLabels(self.x_label_min,self.x_label_max)

                _x0_lab,     _x1_lab     = self.format(self.x_label_min),                self.format(self.x_label_max)
                _x0_lab_len, _x1_lab_len = self.rt_self.textLength(_x0_lab, self.txt_h), self.rt_self.textLength(_x1_lab, self.txt_h)
                x_field_str = '|'.join(self.x_field)
                x_field_str_len = self.rt_self.textLength(x_field_str, self.txt_h)

                if (_x0_lab_len + _x1_lab_len) < (self.w_usable * 0.8):
                    svg += self.rt_self.svgText(_x0_lab, self.x_left,               self.h-self.y_ins, self.txt_h)
                    svg += self.rt_self.svgText(_x1_lab, self.x_left+self.w_usable, self.h-self.y_ins, self.txt_h, anchor='end')
                    
                    # See if we can fit the x_field string in the middle
                    if (_x0_lab_len + x_field_str_len + _x1_lab_len) < (self.w_usable * 0.8):
                        svg += self.rt_self.svgText(x_field_str, self.x_left + self.w_usable/2, self.h - self.y_ins, self.txt_h, anchor='middle')
                elif x_field_str_len < (self.w_usable * 0.8):
                    svg += self.rt_self.svgText(x_field_str, self.x_left + self.w_usable/2, self.h - self.y_ins, self.txt_h, anchor='middle')

                #
                # Y Axis (Copy of last code block)
                #
                if self.y_is_time:
                    self.y_label_min,self.y_label_max = self.rt_self.condenseTimeLabels(self.y_label_min,self.y_label_max)

                _y0_lab,     _y1_lab     = self.format(self.y_label_min),                self.format(self.y_label_max)
                _y0_lab_len, _y1_lab_len = self.rt_self.textLength(_y0_lab, self.txt_h), self.rt_self.textLength(_y1_lab, self.txt_h)
                y_field_str = '|'.join(self.y_field)
                y_field_str_len = self.rt_self.textLength(y_field_str, self.txt_h)

                if (_y0_lab_len + _y1_lab_len) < (self.h_usable * 0.8):
                    svg += self.rt_self.svgText(_y0_lab, self.x_left-4, self.y_ins+self.h_usable, self.txt_h,               rotation=-90)
                    svg += self.rt_self.svgText(_y1_lab, self.x_left-4, self.y_ins,               self.txt_h, anchor='end', rotation=-90)
                    
                    # See if we can fit the x_field string in the middle
                    if (_y0_lab_len + y_field_str_len + _y1_lab_len) < (self.h_usable * 0.8):
                        svg += self.rt_self.svgText(y_field_str, self.x_left-4, self.y_ins + self.h_usable/2, self.txt_h, anchor='middle', rotation=-90)
                elif y_field_str_len < (self.h_usable * 0.8):
                    svg +=     self.rt_self.svgText(y_field_str, self.x_left-4, self.y_ins + self.h_usable/2, self.txt_h, anchor='middle', rotation=-90)

                #
                # Y2 Axis
                #
                if self.y2_label_min is not None:
                    _y0_lab,     _y1_lab     = self.format(self.y2_label_min),               self.format(self.y2_label_max)
                    _y0_lab_len, _y1_lab_len = self.rt_self.textLength(_y0_lab, self.txt_h), self.rt_self.textLength(_y1_lab, self.txt_h)
                    y_field_str = '|'.join(self.y2_field)
                    y_field_str_len = self.rt_self.textLength(y_field_str, self.txt_h)

                    if (_y0_lab_len + _y1_lab_len) < (self.h_usable * 0.8):
                        svg += self.rt_self.svgText(_y0_lab, self.w - 5, self.y_ins + self.h_usable, self.txt_h,               rotation=-90)
                        svg += self.rt_self.svgText(_y1_lab, self.w - 5, self.y_ins,                 self.txt_h, anchor='end', rotation=-90)
                        
                        # See if we can fit the x_field string in the middle
                        if (_y0_lab_len + y_field_str_len + _y1_lab_len) < (self.h_usable * 0.8):
                            svg += self.rt_self.svgText(y_field_str, self.w - 5, self.y_ins + self.h_usable/2, self.txt_h, anchor='middle', rotation=-90)
                    elif y_field_str_len < (self.h_usable * 0.8):
                        svg +=     self.rt_self.svgText(y_field_str, self.w - 5, self.y_ins + self.h_usable/2, self.txt_h, anchor='middle', rotation=-90)
                        
            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg += f'<rect width="{self.w-1}" height="{self.h}" x="0" y="0" fill-opacity="0.0" stroke="{border_color}" />'
            
            svg += '</svg>'
                    
            return svg
        #
        # Format min/max labels
        #
        def format(self, x):
            as_str = str(x)
            if   self.rt_self.strIsInt(as_str):
                return as_str
            elif self.rt_self.strIsFloat(as_str):
                return f'{float(as_str):{self.rt_self.fformat}}'
            else:
                return as_str
        
        #
        # Draw gridlines
        #
        def __drawGridlines__(self,
                              x_axis_flag,        # true if this is for the x-axis
                              _label_min,         # min realworld coordinate // should be convertible to a float
                              _label_max,         # max realworld coordinate // should be convertible to a float
                              _trans_norm_func,   # transform into pixel space for axis specified by the x_axis_flag
                              _pixel_dims,        # total pixels across (or vertically)
                              _base_coord,        # base pixel coordinate in the other dimension
                              _max_coord):        # max pixel coordinate in the other dimension
            _label_max,_label_min = float(_label_max),float(_label_min)
            _delta = _label_max - _label_min
            if   _delta > 100000:
                _inc = None
            elif _delta > 10000:
                _inc   = 2500
                _inc_m = 500
                _start = int(_label_min/1000)*1000
            elif _delta > 1000:
                _inc   = 1000
                _inc_m = 100
                _start = int(_label_min/1000)*1000
            elif _delta > 500:
                _inc   = 100
                _inc_m = 25
                _start = int(_label_min/500)*500
            elif _delta > 100:
                _inc   = 25
                _inc_m = 5
                _start = int(_label_min/100)*100
            elif _delta > 10:
                _inc   = 5
                _inc_m = 1
                _start = int(_label_min/10)*10
            elif _delta > 1:
                _inc   = 1
                _inc_m = 0.25
                _start = int(_label_min)
            elif _delta > .1:
                _inc   = .1
                _inc_m = 0.02
                _start = int(_label_min*10)/10.0
            else:
                _inc = None

            _svg = ''
            if _inc is not None:
                axis_major_co = self.rt_self.co_mgr.getTVColor('axis',   'major')
                axis_minor_co = self.rt_self.co_mgr.getTVColor('axis',   'minor')
                _txt_co       = self.rt_self.co_mgr.getTVColor('context','text')
                _txt_h        = self.txt_h - 2

                _drawn = set()
                i = _start
                while i < _label_max:
                    if x_axis_flag:
                        x = _trans_norm_func(i)
                        _svg += f'<line x1="{x}" y1="{_base_coord+_txt_h}" x2="{x}" y2="{_max_coord}" stroke="{axis_major_co}" stroke-width="1" />'
                        _svg += self.rt_self.svgText(str(i), x, _base_coord+_txt_h-2, _txt_h, _txt_co, anchor='middle')
                    else:
                        y = _trans_norm_func(i)
                        _svg += f'<line x1="{_base_coord}" y1="{y}" x2="{_max_coord}" y2="{y}" stroke="{axis_major_co}" stroke-width="1" />'
                        _svg += self.rt_self.svgText(str(i), _base_coord+3, y-1, _txt_h, _txt_co)
                    _drawn.add(i)
                    i += _inc

                hashmarks = 8
                i = _start
                while i < _label_max:
                    if x_axis_flag:
                        if i not in _drawn:
                            x = _trans_norm_func(i)
                            _svg += f'<line x1="{x}" y1="{_base_coord+hashmarks}" x2="{x}" y2="{_base_coord}" stroke="{axis_minor_co}" stroke-width="1" />'
                            _svg += f'<line x1="{x}" y1="{_max_coord-hashmarks}"  x2="{x}" y2="{_max_coord}"  stroke="{axis_minor_co}" stroke-width="1" />'
                    else:
                        if i not in _drawn:
                            y = _trans_norm_func(i)
                            _svg += f'<line x1="{_max_coord-hashmarks}"  y1="{y}" x2="{_max_coord}"  y2="{y}" stroke="{axis_minor_co}" stroke-width="1" />'
                            _svg += f'<line x1="{_base_coord+hashmarks}" y1="{y}" x2="{_base_coord}" y2="{y}" stroke="{axis_minor_co}" stroke-width="1" />'
                    i += _inc_m

            return _svg

        #
        # Render background distributions
        #
        def __renderBackgroundDistribution__(self,
                                             x_axis_flag,  # True for x-axis, False for y-axis  
                                             x_orig,       # origin x coordinate
                                             y_orig,       # origin y coordinate
                                             x_xa,         # far x coordinate (on the x-axis)
                                             y_xa,         # far y coordinate (on the x-axis)
                                             x_ya,         # far x coordinate (on the y-axis)
                                             y_ya):        # far y coordinate (on the y-axis)
            if x_axis_flag:
                _col = self.x_axis_col
                N    = self.render_x_distribution
            else:
                _col = self.y_axis_col
                N    = self.render_y_distribution
            
            # Determine the max
            v_max,v_lu = 0,{}
            for n in range(1,N+1):
                if n < N:
                    _df = self.df.query(f'`{_col}` >= {(n-1)/N} and `{_col}` <  {n/N}')
                else:
                    _df = self.df.query(f'`{_col}` >= {(n-1)/N} and `{_col}` <= {n/N}')

                # Use the count-by to determine how to sum    
                if self.count_by is None:
                    v = len(_df)
                elif self.count_by_set:
                    v = len(set(_df[self.count_by]))
                else:
                    v = _df[self.count_by].sum()
                
                # Track the max
                if v > v_max:
                    v_max = v
                
                # Save the value for the render
                v_lu[n] = v
                
            
            # Ensure v_max is not zero
            if v_max == 0:
                v_max = 1

            # Determine the colors
            if x_axis_flag:
                _color = self.rt_self.co_mgr.getTVColor('data','default')
            else:
                _color = self.rt_self.co_mgr.getTVColor('data','default')

            # Perform the render
            svg = ''
            for n in range(1,N+1):
                if v_lu[n] > 0:
                    perc    = v_lu[n]/v_max

                    if x_axis_flag:
                        x0 = x_orig + (x_xa-x_orig)*(n-1)/N
                        x1 = x_orig + (x_xa-x_orig)*(n)  /N
                        if self.distribution_style == 'outside':
                            y0 = y_orig
                            y1 = y_orig + self.x_distribution_h*perc  # DIST_GEOM // search for DIST_GEOM to find related calcs
                        else:
                            y0 = y_orig - self.x_distribution_h*perc  # DIST_GEOM // search for DIST_GEOM to find related calcs
                            y1 = y_orig 
                    else:
                        y1 = y_orig - (y_orig-y_ya)*(n-1)/N
                        y0 = y_orig - (y_orig-y_ya)*(n)  /N
                        if self.distribution_style == 'outside':
                            x0 = x_xa
                            x1 = x_xa   + self.y_distribution_h*perc    # DIST_GEOM // search for DIST_GEOM to find related calcs
                        else:
                            x0 = x_orig
                            x1 = x_orig + self.y_distribution_h*perc    # DIST_GEOM // search for DIST_GEOM to find related calcs

                    svg += f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{_color}" ' 
                    svg += f'fill-opacity="{self.render_distribution_opacity*0.8}" stroke="{_color}" stroke-opacity="{self.render_distribution_opacity}" />'

            return svg

        #
        # timestampXCoord() 
        # - calculate the x coordinate for a specific timestamp value
        # - negative values indicate that the timestamp fell before the earliest or after the latest
        # - ... the magnitude of the negative value is equivalent to the positive position 
        # - none result means that the x-axis isn't time...
        #
        def timestampXCoord(self, 
                            _timestamp):
            if type(_timestamp) == str:
                _ts = pd.to_datetime(_timestamp)
            else:
                _ts = _timestamp

            _ts0,_ts1 = self.timestamp_min,self.timestamp_max
            if type(_ts0) == str:
                _ts0 = pd.to_datetime(_ts0)
            if type(_ts1) == str:
                _ts1 = pd.to_datetime(_ts1)

            if self.x_is_time:
                if _ts   < _ts0:
                    return -self.x_left
                elif _ts > _ts1:
                    return -(self.x_left + self.w_usable)
                else:
                    return self.x_left + self.w_usable*((_ts - _ts0)/(_ts1 - _ts0))
            else:
                return None
        
        #
        # timestampExtents()
        # - return the minimum and maximum timestamps as a pandas tuple
        #
        def timestampExtents(self):
            _ts0,_ts1 = self.timestamp_min,self.timestamp_max
            if type(_ts0) == str:
                _ts0 = pd.to_datetime(_ts0)
            if type(_ts1) == str:
                _ts1 = pd.to_datetime(_ts1)
            return _ts0,_ts1

        #
        # contrastStretchLegend()
        #
        def contrastStretchLegend(self, txt_h=12):
            # Number of pixels at each percent... scaled to the pixels in the legend
            legend_px         = {}
            legend_px_perc    = {}
            max_legend_px     = 0
            px_to_value_min   = {}
            px_to_value_max   = {}
            px_to_value_count = {}

            _total_so_far = 0
            _sorted       = sorted(list(self.stretch_histogram.keys()))
            for x in _sorted:
                _perc = _total_so_far / self.stretch_total
                _total_so_far += self.stretch_histogram[x] * x
                _y        = int(12 + _perc*500)
                if _y not in legend_px.keys():
                    legend_px[_y]         = 0
                    legend_px_perc[_y]    = _perc
                    px_to_value_min[_y]   = x
                    px_to_value_max[_y]   = x
                    px_to_value_count[_y] = 0
                legend_px[_y] += self.stretch_histogram[x]
                px_to_value_count[_y] += self.stretch_histogram[x]
                if legend_px[_y] > max_legend_px:
                    max_legend_px = legend_px[_y]
                if x < px_to_value_min[_y]:
                    px_to_value_min[_y] = x
                if x > px_to_value_max[_y]:
                    px_to_value_max[_y] = x

            svg =   '<svg width="512" height="524">'
            svg += f'<rect width="512" height="524" fill="{self.rt_self.co_mgr.getTVColor("background","default")}" />'
            _last_txt_y = -100
            for _y in legend_px.keys():
                _count =  legend_px[_y]
                _perc  =  legend_px_perc[_y]
                _color =  self.rt_self.co_mgr.spectrum(_perc, 0, 1.0, True)
                svg    += f'<rect x="{5}" y="{_y}" width="{100*log10(_count+1)/log10(max_legend_px+1)}" height="{1.5}" fill="{_color}" />'

                if _last_txt_y < _y:
                    _str = str(px_to_value_min[_y])
                    if px_to_value_min[_y] != px_to_value_max[_y]:
                        _str += ' - ' + str(px_to_value_max[_y])
                    svg += self.rt_self.svgText(_str, 150, _y + txt_h/2, txt_h, color=_color)
                    _last_txt_y = _y + txt_h
                    _str = str(px_to_value_count[_y]) + ' pixels'
                    svg += self.rt_self.svgText(_str, 250, _y + txt_h/2, txt_h, color=_color)

            svg += '</svg>'
            return svg

            # Contrast stretch calculation
            #self.contrast_stretch = {}
            #if self.color_magnitude == 'stretch':
            #    _total_so_far = 0
            #    _sorted       = sorted(list(self.stretch_histogram.keys()))
            #    for x in _sorted:
            #        _perc = _total_so_far / self.stretch_total
            #        self.contrast_stretch[x] = _perc
            #        _total_so_far += self.stretch_histogram[x] * x
            #color = self.rt_self.co_mgr.spectrum(self.contrast_stretch[my_count], 0, 1.0,    True)

    #
    # Condense a Time Label Down To The Minimum...
    # ... uses simple rules... should really be using the time granularity...
    #
    def condenseTimeLabels(self, x, y):
        x,y = str(x),str(y)    
        if x.endswith(':00') and y.endswith(':00'): # Remove empty seconds
            x = x[:-3]
            y = y[:-3]
        if x.endswith(':00') and y.endswith(':00'): # Remove empty minutes
            x = x[:-3]
            y = y[:-3]
        if x.endswith(' 00') and y.endswith(' 00'): # Remove empty hours
            x = x[:-3]
            y = y[:-3]
        if x.endswith('-01') and y.endswith('-01'): # Remove day
            x = x[:-3]
            y = y[:-3]
        if x.endswith('-01') and y.endswith('-01'): # Remove month
            x = x[:-3]
            y = y[:-3]
        return x,y
