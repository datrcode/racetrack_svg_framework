# Copyright 2024 David Trimm
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
import time

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from datetime import timedelta
from dateutil.relativedelta import relativedelta

from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, MultiLineString

from math import log10

import random

from .rt_component import RTComponent

__name__ = 'rt_xy_mixin'

#
# Abstraction for XY Scatterplot
#
class RTXYMixin(object):
    #
    # __distributionSingle__() - single variable distribution
    #
    def __distributionSingle__(self, df, field, k='a', bins=40):
        counts, _ranges_ = np.histogram(df[field], bins=bins)
        xs, counts_norm, splits = [], [], []
        _counts_max_ = max(counts)
        r0, r1 = _ranges_[0], _ranges_[-1]
        for i in range(len(_ranges_)-1):
            x = r0 + ((r1 - r0) * i) / (len(_ranges_)-2)
            xs.append(x)
            splits.append(k)
            counts_norm.append(counts[i]/_counts_max_)
        return counts, splits, xs, counts_norm

    #
    # __distributionSplit__() - multiple variable distribution
    #
    def __distributionSplit__(self, df, field, split_by=None, bins=40):
        counts, splits, xs, counts_norm = [], [], [], []
        gb = df.groupby(split_by)
        for k, k_df in gb:
            _counts_, _splits_, _xs_, _counts_norm_ = self.__distributionSingle__(k_df, field, k, bins)
            counts.extend(_counts_)
            splits.extend(_splits_)
            xs.extend(_xs_)
            counts_norm.extend(_counts_norm_)
        return counts, splits, xs, counts_norm

    #
    # distroXY() - create a distribution to deliver to the xy component.
    #
    def distroXY(self, df, field, split_by=None, bins=20, use_norm=False, **params):
        if split_by is not None:
            counts, splits, xs, counts_norm = self.__distributionSplit__(df, field, split_by, bins)
        else:
            counts, splits, xs, counts_norm = self.__distributionSingle__(df, field, bins=bins)
                
        _df_ = pd.DataFrame({'x':xs,'split':splits,'count':counts, 'count_norm':counts_norm})
        params['df']                 = _df_
        params['x_field']            = 'x'    
        params['y_field']            = 'count_norm' if use_norm else 'count'
        params['line_groupby_field'] = 'split'
        params['color_by']           = 'split'
        params['dot_size']           = None
        return params

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
    # Identify the required fields for this component
    #
    def xyRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('x_field',  kwargs, columns_set)
        self.identifyColumnsFromParameters('y_field',  kwargs, columns_set)
        self.identifyColumnsFromParameters('color_by', kwargs, columns_set)
        self.identifyColumnsFromParameters('count_by', kwargs, columns_set)
        self.identifyColumnsFromParameters('line_groupby_field',  kwargs, columns_set)
        self.identifyColumnsFromParameters('x2_field',            kwargs, columns_set)
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
           dot_size          = 'medium',  # Dot size - ['tiny', 'small', 'medium', 'large', 'huge', 'vary', 'hidden'/None]
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

           # ------------------------     # secondary axis override for polynomial best fit

           poly_fit_degree     = None,    # integer value for polynomial degree -- will overrride all of the other df2 settings if set...

           # ------------------------     # secondary axis settings # probably not small multiple safe...

           df2                     = None,       # secondary axis dataframe ... if not set but y2_field is, then this will be set to df field
           x2_field                = None,       # x2 field ... if not set but the y2_field is, then this be set to the x_field
           x2_field_is_scalar      = True,       # x2 field is scalar
           x2_axis_col             = None,       # x2 axis column name
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

           track_state               = False,         # state tracking for interactive filtering
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
        ''' xy() - xy scatter plot

        Parameters
        ----------

        df                      : pandas or polars dataframe
        x,y_field               : x and y axes fields

        x,y_field_is_scalar     : treat x and y axes field as a scalar

        color_by                : color by field
        color_magnitude         : Only applies when color_by is None, options: None / 'linear' / 'log' / 'stretch'
        opacity                 : Opacity of the dots
        vary_opacity            : If true, vary opacity by the count_by # Not Implemented

        count_by                : none means just count rows, otherwise, use a field to sum by # Not Implemented
        count_by_set            : count by set operation

        dot_size                : Dot size - ['tiny', 'small', 'medium', 'large', 'huge', 'vary', 'hidden'/None]
        dot_shape               : Dot shape - ['square', 'ellipse', 'triangle, 'utriangle', 'diamond', 'plus', x', 'small_multiple', function_pointer]
        max_dot_size            : Max dot size (used when the dot sz varies)

        align_pixels            : Align pixels to integer values
        widget_id               : svg element id

        line_groupby_field      : will use a field to perform a groupby for a line chart
        line_groupby_w          : width of the line for the line chart

        draw_context            : draw temporal context information if (and only if) x_axis is time

        Parameters (Distributions)
        --------------------------

        render_x_distribution       : number of x distribution buckets ... None == don't render
        render_y_distribution       : number of x distribution buckets ... None == don't render
        render_distribution_opacity : Opacity of the distribution render
        distribution_h_perc         : height of the distribution as a function of the overall h of xy chart     
        distribution_style          : 'outside' - outside of xy... 'inside' - inside of xy

        Secondary Axis Parameters
        -------------------------

        df2                     : secondary axis dataframe ... if not set but y2_field is, then this will be set to df field
        x2_field                : x2 field ... if not set but the y2_field is, then this be set to the x_field
        x2_field_is_scalar      : x2 field is scalar
        x2_axis_col             : x2 axis column name
        y2_field                : secondary axis field ... if this is set, then df2 will be set to df // only required field really...
        y2_field_is_scalar      : default... logic will check in the method to determine if this is true
        y2_axis_col             : y2 axis column name
        line2_groupby_field     : secondary line field ... will NOT be set
        line2_groupby_w         : secondary line field width
        line2_groupby_color     : line2 color... if none, pulls from the color_by field
        line2_groupby_dasharray : line2 dasharray
        dot2_size               : 'inherit',  # dot2 size... 'inherit' means take from the dot_size...

        poly_fit_degree         : integer value for polynomial degree -- will overrride all of the other df2 settings if set...

        Parameters (Views That Share Parameters)
        ----------------------------------------

        x,y_axis_col            : axis column name
        x,y_is_time             : is time flag
        x,y_label_min           : min label on the x axis
        x,y_label_max           : max label on the x axis
        x,y_trans_func          : lambda transform function for x axis
        x,y_order               : order of categorical values on the x axis
        x,y_fill_transforms     : for t-fields, fill in all the values to properly space out data

        Parameters (Small Multiples Options)
        ------------------------------------

        sm_type                 : should be the method name // similar to the smallMultiples method
        sm_w,h                  : override the width/height of the small multiple
        sm_params               : dictionary of parameters for the small multiples
        sm_x,y_axis_independent : use independent axis for x/y

        Parameters (Background Options)
        -------------------------------

        bg_shape_lu           : lookup for background shapes -- key will be used for varying colors (if bg_shape_label_color == 'vary')
                                ['key'] = [(x0,y0),(x1,y1),...] OR
                                ['key'] = svg path description
        bg_shape_label_color  : None = no label, 'vary', lookup to hash color, or a hash color
        bg_shape_opacity      : None (== 0.0), number, lookup to opacity
        bg_shape_fill         : None, 'vary', lookup to hash color, or a hash color
        bg_shape_stroke_w     : None, number, lookup to width
        bg_shape_stroke       : None, 'default', lookup to hash color, or a hash color


        Parameters (Generic)
        --------------------

        track_state              : state tracking for interactive filtering
        x_view                   : x offset for the view
        y_view                   : y offset for the view
        x_ins                    : side inserts
        y_ins                    : top & bottom inserts
        w                        : width of the view
        h                        : height of the view
        txt_h                    : text height for labeling
        background_opacity       : opacity for entire plot
        background_override      : override the background color // hex value
        plot_background_override : override the background for the plot area // hex value
        draw_x_gridlines         : draw the x gridlines for scalar values
        draw_y_gridlines         : draw the y gridlines for scalar values
        draw_labels              : draw labels flag
        draw_border              : draw a border around the xy chart
        '''
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTXy(self, **_params_)

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
                           timestamp_max  = None,   # Maximum timestamp field
                           _min           = None,   # Minimum for scalar axis
                           _max           = None):  # Maximum for scalar axis
        if self.isPandas(df):
            return self.__xyCreateAxisColumn_pandas__(df, field, is_scalar, new_axis_field, order, fill_transform, timestamp_min, timestamp_max, _min, _max)
        elif self.isPolars(df):
            return self.__xyCreateAxisColumn_polars__(df, field, is_scalar, new_axis_field, order, fill_transform, timestamp_min, timestamp_max, _min, _max)
        else:
            raise Exception('RTXY.xyCreateAxisColumn() - only pandas and polars is supported')

    # Pandas version
    def __xyCreateAxisColumn_pandas__(self, 
                                      df, 
                                      field, 
                                      is_scalar, 
                                      new_axis_field,
                                      order          = None,   # Order of the values on the axis
                                      fill_transform = True,   # Fill in missing transform values
                                      timestamp_min  = None,   # Minimum timestamp field
                                      timestamp_max  = None,   # Maximum timestamp field
                                      _min           = None,   # Minimum for scalar axis
                                      _max           = None):  # Maximum for scalar axis
        if type(field) != list:
            field = [field]
        is_time = False        
        field_countable = self.fieldIsArithmetic(df, field[0])
        f0 = field[0]
        transFunc = None
        # Numeric scaling
        if field_countable and is_scalar and len(field) == 1:
            my_min = df[f0].min() if _min is None else _min
            my_max = df[f0].max() if _max is None else _max
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
            my_min = df[f0].min() if timestamp_min is None else timestamp_min
            my_max = df[f0].max() if timestamp_max is None else timestamp_max
            if my_min == my_max:
                my_max += timedelta(seconds=1)
            df[new_axis_field] = ((df[f0] - my_min)/(my_max - my_min))
            label_min = timestamp_min # df[f0].min()
            label_max = timestamp_max # df[f0].max()
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
                            raise Exception('xy - fill_transform is specified but there are multiple fields with at least one transform... create your own order...')
                order_filled_by_transform = False                
            gb = df.groupby(field)
            if order is None:
                # 0...1 assignment              
                my_inc = self.XYInc(1.0/(len(gb)-1)) if len(gb) >= 2 else self.XYInc(1.0/len(gb))
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
                if order is not None and len(order) > 0:
                    label_min = order[0]                
                    label_max = order[-1] if (order_is_complete or order_filled_by_transform) else 'ee'
                else:
                    label_min = label_max = 'zero len order'
        return df, is_time, label_min, label_max, transFunc, order

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
            self._div        = len(_order)
            if self._reserve_na:
                pass
            else:
                self._div -= 1
            if self._div <= 0:
                self._div = 1

        def nextValue(self, x):
            if x.name in self._order:
                if self._reserve_na:
                    return self._order.index(x.name)/self._div # len(self._order)
                else:
                    return self._order.index(x.name)/self._div # (len(self._order)-1)
            return 1.0

    # Polars version
    def __xyCreateAxisColumn_polars__(self, 
                                    df, 
                                    field, 
                                    is_scalar, 
                                    new_axis_field,
                                    order          = None,   # Order of the values on the axis
                                    fill_transform = True,   # Fill in missing transform values
                                    timestamp_min  = None,   # Minimum timestamp field
                                    timestamp_max  = None,   # Maximum timestamp field
                                    _min           = None,   # Minimum for scalar axis
                                    _max           = None):  # Maximum for scalar axis
        if type(field) != list:
            field = [field]
        is_time = False    
        field_countable = self.fieldIsArithmetic(df, field[0])
        f0 = field[0]
        transFunc = None
        # Numeric scaling // DONE!
        if field_countable and is_scalar and len(field) == 1:
            my_min = df[f0].min() if _min is None else _min
            my_max = df[f0].max() if _max is None else _max
            if my_min == my_max:
                my_min -= 0.5
                my_max += 0.5
            df = df.with_columns(((pl.col(f0)-my_min)/(my_max-my_min)).alias(new_axis_field))
            label_min = str(my_min)
            label_max = str(my_max)
            transFunc = lambda x: ((x - my_min)/(my_max - my_min))
        # Timestamp scaling // DONE!
        elif len(field) == 1 and df[field[0]].dtype == pl.Datetime:
            # Use dataframe for min... or the parameter version if set
            my_min = df[f0].min() if timestamp_min is None else timestamp_min
            my_max = df[f0].max() if timestamp_max is None else timestamp_max
            if my_min == my_max:
                my_max += timedelta(seconds=1)
            df = df.with_columns(((pl.col(f0)-my_min)/(my_max-my_min)).alias(new_axis_field))
            label_min = timestamp_min
            label_max = timestamp_max
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
                            raise Exception('xy - fill_transform is specified but there are multiple fields with at least one transform... create your own order...')
                order_filled_by_transform = False

            # Determine all the possibilities in the dataframe
            if len(field) == 1:
                all_combos = sorted(list(set(df[field[0]])))
            else:
                df         = df.sort(field)
                group_by   = df.group_by(field, maintain_order=True)
                all_combos    = []
                for k, k_df in group_by:
                    all_combos.append(k)

            # Determine the order & create the dictionary
            if order is None:
                order = all_combos
                # Create the dictionary
                _order_len_ = (len(order)-1) if len(order) > 1 else 1
                _dict_, i = {}, 0
                for x in order:
                    _dict_[x] = i/_order_len_
                    i += 1
            else:
                gb_set, order_set = set(all_combos), set(order)
                order_is_complete = (len(gb_set) == len(order_set)) and (len(gb_set & order_set) == len(order_set))
                if order_is_complete:
                    # Create the dictionary
                    _order_len_ = (len(order)-1) if len(order) > 1 else 1
                    _dict_, i = {}, 0
                    for x in order:
                        _dict_[x] = i/_order_len_
                        i += 1
                else:
                    order.append('ee') # last order is the 'everything else' category...
                    # Create the dictionary
                    _order_len_ = (len(order)-1) if len(order) > 1 else 1
                    _dict_, i = {}, 0
                    for x in order:
                        _dict_[x] = i/_order_len_
                        i += 1
                    for x in (gb_set - order_set):
                        _dict_[x] = 1.0

            # Create the new column from the dictionary
            if len(field) == 1:
                df = df.with_columns(pl.col(field[0]).replace_strict(_dict_, return_dtype=pl.Float32).alias(new_axis_field))
            else:
                def myMapRows(k):
                    return _dict_[k]
                axis_series = df.drop(set(df.columns) - set(field)).select(field).map_rows(myMapRows)['map']
                df = df.with_columns(pl.Series(new_axis_field, axis_series))

            # Compute the min and max labels
            def concatAsStrs(x):
                if type(x) == list:
                    s = str(x[0])
                    for i in range(1,len(x)):
                        s += '|' + str(x[i])
                else:
                    return str(x)
            label_min, label_max = concatAsStrs(order[0]), concatAsStrs(order[-1])

        return df, is_time, label_min, label_max, transFunc, order

    #
    # __transformBackgroundShapes__() - refactored to split for other methods
    #
    def __transformBackgroundShapes__(self, 
                                      shape_name,
                                      shape_desc,
                                      x_trans_norm_func,
                                      y_trans_norm_func,
                                      bg_shape_label_color,
                                      bg_shape_opacity,
                                      bg_shape_fill,
                                      bg_shape_stroke_w,
                                      bg_shape_stroke, 
                                      txt_h):
        # Transform to a path description if necessary ... Polygons & MultiPolygons can be handled in the same way
        if type(shape_desc) == Polygon or type(shape_desc) == MultiPolygon:
            shape_desc = self.shapelyPolygonToSVGPathDescription(shape_desc)

        # Multiline Strings should have have their fill type set to 'none'
        if type(shape_desc) == MultiLineString or type(shape_desc) == LineString:
            shape_desc = self.shapelyPolygonToSVGPathDescription(shape_desc)
            bg_shape_fill = 'none'

        # Geometry Collections are usually empty... maybe the results of clipping?
        if type(shape_desc) == GeometryCollection:
            if len(shape_desc.geoms) > 0: # Haven't seen this... so unsure of how to process
                raise Exception('RTXYMixin.__transformBackgroundShapes__() - geometrycollection not empty')    
            return '', ''
            
        # Handle intermediate types
        if   type(shape_desc) == str:   # path description
            _shape_svg, _label_svg = self.__transformPathDescription__(shape_name,           shape_desc,
                                                                       x_trans_norm_func,    y_trans_norm_func,
                                                                       bg_shape_label_color, bg_shape_opacity,
                                                                       bg_shape_fill,        bg_shape_stroke_w,
                                                                       bg_shape_stroke,      txt_h)
        elif type(shape_desc) == list:  # list of tuple pairs
            _shape_svg, _label_svg = self.__transformPointsList__(shape_name,            shape_desc,
                                                                  x_trans_norm_func,     y_trans_norm_func,
                                                                  bg_shape_label_color,  bg_shape_opacity,
                                                                  bg_shape_fill,         bg_shape_stroke_w,
                                                                  bg_shape_stroke,       txt_h)
        else:
            raise Exception(f'RTXYMixin.__transformBackgroundShapes__() - type "{type(shape_desc)}" as background lookup')
        
        return _shape_svg, _label_svg

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
    # RTXy
    #
    class RTXy(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     **kwargs):
            self.parms                   = locals().copy()
            self.rt_self                 = rt_self
            self.df                      = rt_self.copyDataFrame(kwargs['df'])
            self.x_field                 = kwargs['x_field']
            self.y_field                 = kwargs['y_field']
            self.x_field_is_scalar       = kwargs['x_field_is_scalar'] 
            self.y_field_is_scalar       = kwargs['y_field_is_scalar']
            self.color_by                = kwargs['color_by']
            self.color_magnitude         = kwargs['color_magnitude']                         
            self.count_by                = kwargs['count_by']
            self.count_by_set            = kwargs['count_by_set']
            self.dot_size                = kwargs['dot_size']
            self.dot_shape               = kwargs['dot_shape']
            self.max_dot_size            = kwargs['max_dot_size']
            self.opacity                 = kwargs['opacity']
            self.vary_opacity            = kwargs['vary_opacity']
            self.align_pixels            = kwargs['align_pixels']
            self.widget_id               = kwargs['widget_id']

            # Make a widget_id if it's not set already
            if self.widget_id is None:
                self.widget_id = "xy_" + str(random.randint(0,2**24))

            self.x_axis_col              = kwargs['x_axis_col']
            self.x_is_time               = kwargs['x_is_time']
            self.x_label_min             = kwargs['x_label_min']
            self.x_label_max             = kwargs['x_label_max']
            self.x_trans_func            = kwargs['x_trans_func']
            self.x_order                 = kwargs['x_order']
            self.x_fill_transforms       = kwargs['x_fill_transforms']
            self.y_axis_col              = kwargs['y_axis_col']
            self.y_is_time               = kwargs['y_is_time']
            self.y_label_min             = kwargs['y_label_min']
            self.y_label_max             = kwargs['y_label_max']
            self.y_trans_func            = kwargs['y_trans_func']
            self.y_order                 = kwargs['y_order']
            self.y_fill_transforms       = kwargs['y_fill_transforms']
            self.line_groupby_field      = kwargs['line_groupby_field']
            self.line_groupby_w          = kwargs['line_groupby_w']
            self.line2_groupby_field     = kwargs['line2_groupby_field']

            self.time_lu                 = {} # Performance Analysis

            #
            # Are we fitting to something?  Or just diplaying what the user wants to show as a secondary?
            #
            if kwargs['poly_fit_degree'] is None or self.rt_self.isPolars(self.df): # Polars only supported on this path...
                self.x2_field                = kwargs['x2_field']
                self.x2_field_is_scalar      = kwargs['x2_field_is_scalar']
                self.x2_axis_col             = kwargs['x2_axis_col']
                self.y2_field                = kwargs['y2_field']
                self.y2_field_is_scalar      = kwargs['y2_field_is_scalar']
                self.y2_axis_col             = kwargs['y2_axis_col']

                # y2_field is really the only required param to make the df2 work...
                if kwargs['y2_field'] is not None:
                    if kwargs['df2'] is not None:
                        self.df2 = rt_self.copyDataFrame(kwargs['df2'])
                        self.df2_is_df = False
                    else:
                        self.df2 = self.df
                        self.df2_is_df = True

                    if kwargs['x2_field'] is None:
                        self.x2_field           = self.x_field
                        self.x2_field_is_scalar = self.x_field_is_scalar
                        self.x2_axis_col        = self.x_axis_col
                else:
                    self.df2 = None

                if self.x2_field is not None and type(self.x2_field) != list:
                    self.x2_field = [self.x2_field]
                if self.y2_field is not None and type(self.y2_field) != list:
                    self.y2_field = [self.y2_field]

            #
            # Fitting ... which overrides df2, df2_is_df, x2_field, (x2_field specifics), y2_field, (y2_field specifics)
            # ... doesn't work with timestamps...
            # ... xmins and xmaxes are messed up ... i.e., not aligned with the primary y-axis column...
            # ... ... pros and cons with that approach... visually more accurate to align it... but you may not see the whole shape...
            #
            else:
                t0_polyfit = time.time()
                # Figure out the x-range
                _min,_max = self.df[self.x_field].min(),self.df[self.x_field].max()
                if _min == _max:
                    _max = _min + 1
                _x,_x_inc,_x_values = _min, (_max - _min)/kwargs['w'], []
                while _x <= _max:
                    _x_values.append(_x)
                    _x += _x_inc

                # Differentiate between groupby version and non-groupby version
                if kwargs['line_groupby_field'] is None:
                    _fit      = np.polyfit(self.df[self.x_field], self.df[self.y_field], kwargs['poly_fit_degree'])
                    _y_values = np.polyval(_fit, _x_values)
                    self.df2            = pd.DataFrame({'x':_x_values,'y':_y_values})
                    self.df2['groupby'] = 'all'
                else:
                    _dfs = []
                    for k,k_df in self.df.groupby(kwargs['line_groupby_field']):
                        _fit           = np.polyfit(k_df[self.x_field], k_df[self.y_field], kwargs['poly_fit_degree'])
                        _y_values      = np.polyval(_fit, _x_values)
                        _df            = pd.DataFrame({'x':_x_values,'y':_y_values})
                        _df['groupby'] = k
                        _dfs.append(_df)
                    self.df2 = self.rt_self.concatDataFrames(_dfs)

                t1_polyfit = time.time()
                self.time_lu['polyfit'] = t1_polyfit - t0_polyfit

                # Common settings once the df2 is created
                self.line2_groupby_field = 'groupby'
                self.df2_is_df           = False
                self.x2_field,self.y2_field,self.x2_field_is_scalar,self.y2_field_is_scalar = ['x'],['y'],True,True
                self.x2_axis_col,self.y2_axis_col = None,None

            self.line2_groupby_w         = kwargs['line2_groupby_w']
            self.line2_groupby_color     = kwargs['line2_groupby_color']
            self.line2_groupby_dasharray = kwargs['line2_groupby_dasharray']
            self.dot2_size               = kwargs['dot2_size']
            if kwargs['dot2_size'] == 'inherit':
                self.dot2_size = kwargs['dot_size']

            self.sm_type                 = kwargs['sm_type']
            self.sm_w                    = kwargs['sm_w']
            self.sm_h                    = kwargs['sm_h']
            self.sm_params               = kwargs['sm_params']
            self.sm_x_axis_independent   = kwargs['sm_x_axis_independent']
            self.sm_y_axis_independent   = kwargs['sm_y_axis_independent']

            self.bg_shape_lu             = kwargs['bg_shape_lu']
            self.bg_shape_label_color    = kwargs['bg_shape_label_color']
            self.bg_shape_opacity        = kwargs['bg_shape_opacity']
            self.bg_shape_fill           = kwargs['bg_shape_fill']
            self.bg_shape_stroke_w       = kwargs['bg_shape_stroke_w']
            self.bg_shape_stroke         = kwargs['bg_shape_stroke']

            self.render_x_distribution       = kwargs['render_x_distribution']
            self.render_y_distribution       = kwargs['render_y_distribution']
            self.render_distribution_opacity = kwargs['render_distribution_opacity']
            self.distribution_h_perc         = kwargs['distribution_h_perc']
            self.distribution_style          = kwargs['distribution_style']

            self.track_state              = kwargs['track_state']
            self.x_view                   = kwargs['x_view']
            self.y_view                   = kwargs['y_view']
            self.x_ins                    = kwargs['x_ins']
            self.y_ins                    = kwargs['y_ins']
            self.w                        = kwargs['w']
            self.h                        = kwargs['h']
            self.txt_h                    = kwargs['txt_h']
            self.background_opacity       = kwargs['background_opacity']
            self.background_override      = kwargs['background_override']
            self.plot_background_override = kwargs['plot_background_override']
            self.draw_x_gridlines         = kwargs['draw_x_gridlines']
            self.draw_y_gridlines         = kwargs['draw_y_gridlines']
            self.draw_labels              = kwargs['draw_labels']
            self.draw_border              = kwargs['draw_border']
            self.draw_context             = kwargs['draw_context']

            # Check the dot information
            if self.sm_type is not None:
                self.dot_shape = 'small_multiple'            
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
            if self.x2_field is not None and type(self.x2_field) != list: # Make it into a list for consistency
                self.x2_field = [self.x2_field]
            if self.y2_field is not None and type(self.y2_field) != list: # Make it into a list for consistency
                self.y2_field = [self.y2_field]

            #
            # Transforms section
            #

            # Apply axes transforms
            t0_transforms = time.time()
            self.df, self.x_field   = rt_self.transformFieldListAndDataFrame(self.df, self.x_field)
            self.df, self.y_field   = rt_self.transformFieldListAndDataFrame(self.df, self.y_field)
            if self.x2_field is not None:
                self.df2, self.x2_field = rt_self.transformFieldListAndDataFrame(self.df2, self.x2_field)
            if self.y2_field is not None:
                self.df2, self.y2_field = rt_self.transformFieldListAndDataFrame(self.df2, self.y2_field)

            # Make sure we understand what's scalar... and what's not...
            if self.x_field_is_scalar:
                self.x_field_is_scalar = len(self.x_field)   == 1 and self.rt_self.fieldIsArithmetic(self.df,  self.x_field[0])
            if self.y_field_is_scalar:
                self.y_field_is_scalar = len(self.y_field)   == 1 and self.rt_self.fieldIsArithmetic(self.df,  self.y_field[0])
            if self.x2_field is not None and self.x2_field_is_scalar:
                self.x2_field_is_scalar = len(self.x2_field) == 1 and self.rt_self.fieldIsArithmetic(self.df2, self.x2_field[0])
            if self.y2_field is not None and self.y2_field_is_scalar:
                self.y2_field_is_scalar = len(self.y2_field) == 1 and self.rt_self.fieldIsArithmetic(self.df2, self.y2_field[0])

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

            t1_transforms = time.time()
            self.time_lu['transforms'] = t1_transforms - t0_transforms

            # Setup the y2 info (if the y2_field is set)
            t0_y2_setup = time.time()
            self.timestamp_min, self.timestamp_max, self.x_min, self.x_max = None,None,None,None
            if len(self.x_field) == 1 and \
                ( (self.rt_self.isPandas(self.df) and is_datetime(self.df[self.x_field[0]])) or
                  (self.rt_self.isPolars(self.df) and self.df[self.x_field[0]].dtype == pl.Datetime)): # TIME
                self.timestamp_min = self.df[self.x_field[0]].min()
                self.timestamp_max = self.df[self.x_field[0]].max()
                if self.y2_field is not None:                    
                    # Determine actual timestamp min and max 
                    if self.df2[self.x2_field[0]].min() < self.timestamp_min:
                        self.timestamp_min = self.df2[self.x2_field[0]].min()
                    if self.df2[self.x2_field[0]].max() > self.timestamp_max:
                        self.timestamp_max = self.df2[self.x2_field[0]].max()
            elif len(self.x_field) == 1 and self.x_field_is_scalar:              # SCALARS
                self.x_min = self.df[self.x_field[0]].min()
                self.x_max = self.df[self.x_field[0]].max()

                if self.y2_field is not None:
                    # Check some assumptions
                    if self.x2_field is not None and len(self.x2_field) != 1:
                        raise Exception('xy() - x2_field must be a single field')
                    if self.x2_field_is_scalar == False:
                        raise Exception('xy() - x2_field must be a scalar field')
                    
                    # Perform the comparisons
                    if self.df2[self.x2_field[0]].min() < self.x_min:
                        self.x_min = self.df2[self.x2_field[0]].min()
                    if self.df2[self.x2_field[0]].max() > self.x_max:
                        self.x_max = self.df2[self.x2_field[0]].max()
            elif self.y2_field is not None and self.df2_is_df == False and self.x_order is None: # CATEGORICALS
                if   self.rt_self.isPandas(self.df):
                    _set0 = set(self.df. groupby(self.x_field). groups.keys())
                    _set1 = set(self.df2.groupby(self.x2_field).groups.keys())
                elif self.rt_self.isPolars(self.df):
                    _set0, _set1 = set(), set()
                    for k, k_df in self.df.group_by(self.x_field):
                        _set0.add(k)
                    for k, k_df in self.df2.group_by(self.x2_field):
                        _set1.add(k)
                self.x_order = sorted(list(_set0 | _set1))

            t1_y2_setup = time.time()
            self.time_lu['y2_setup'] = t1_y2_setup - t0_y2_setup

            # Geometry lookup for tracking state
            self.geom_to_df = {}
            self.last_render = None

        #
        # print() version of class
        #
        def __repr__(self):
            return f'xy(df.len={len(self.df)}, x_field={self.x_field}, y_field={self.y_field}, {self.w}x{self.h})'
            
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
        def renderSVG(self, just_calc_max=False):
            if self.track_state:
                self.geom_to_df = {}

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
            def dotSizeNumber(_str_):
                if _str_ is None or _str_ == 'hidden':           return  0
                elif type(_str_) == int or type(_str_) == float: return  _str_
                elif _str_ == 'medium':                          return  2
                elif _str_ == 'small':                           return  1
                elif _str_ == 'large':                           return  3
                elif _str_ == 'huge':                            return  8
                elif _str_ == 'tiny':                            return  0.3
                else:                                            return -1
            dot_w  = dotSizeNumber(self.dot_size)
            dot2_w = dotSizeNumber(self.dot2_size)

            # Create the extra columns for the x and y coordinates
            t0_x_create = time.time()
            if self.x_axis_col is None:
                self.x_axis_col = 'my_x_' + self.widget_id
                self.df, self.x_is_time, self.x_label_min, self.x_label_max, self.x_trans_func, self.x_order = self.rt_self.xyCreateAxisColumn(self.df, self.x_field, self.x_field_is_scalar, self.x_axis_col, self.x_order, self.x_fill_transforms, 
                                                                                                                                               self.timestamp_min, self.timestamp_max, self.x_min, self.x_max)
            t1_x_create = time.time()
            self.time_lu['x_create'] = t1_x_create - t0_x_create

            t0_y_create = time.time()
            if self.y_axis_col is None:
                self.y_axis_col = 'my_y_' + self.widget_id
                self.df, self.y_is_time, self.y_label_min, self.y_label_max, self.y_trans_func, self.y_order = self.rt_self.xyCreateAxisColumn(self.df, self.y_field, self.y_field_is_scalar, self.y_axis_col, self.y_order, self.y_fill_transforms)
            
            t1_y_create = time.time()
            self.time_lu['y_create'] = t1_y_create - t0_y_create

            # Secondary axis settings
            t0_y2_create = time.time()
            self.y2_label_min, self.y2_label_max = None,None
            if self.y2_field is not None and self.y2_axis_col is None:
                self.y2_axis_col = 'my_y2_' + self.widget_id
                self.df2, self.y2_is_time, self.y2_label_min, self.y2_label_max, _throwaway_func, _throwaway_order = self.rt_self.xyCreateAxisColumn(self.df2, self.y2_field, self.y2_field_is_scalar, self.y2_axis_col)
                if self.df2_is_df:
                    self.x2_axis_col = self.x_axis_col
                else:
                    self.x2_axis_col = 'my_x2_' + self.widget_id
                    self.df2, self.x2_is_time, self.x2_label_min, self.x2_label_max, _throwaway_func, _throwaway_order = self.rt_self.xyCreateAxisColumn(self.df2, self.x2_field, self.x2_field_is_scalar, self.x2_axis_col, self.x_order, self.x_fill_transforms, 
                                                                                                                                                         self.timestamp_min, self.timestamp_max, self.x_min, self.x_max)

            t1_y2_create = time.time()
            self.time_lu['y2_create'] = t1_y2_create - t0_y2_create

            # Create the pixel-level columns
            t0_pixel_calc_and_align = time.time()
            if self.rt_self.isPandas(self.df):
                self.df[self.x_axis_col+"_px"] = self.x_left                + self.df[self.x_axis_col]*self.w_usable
                self.df[self.y_axis_col+"_px"] = self.y_ins + self.h_usable - self.df[self.y_axis_col]*self.h_usable
                if self.align_pixels:
                    self.df[self.x_axis_col+"_px"] = self.df[self.x_axis_col+"_px"].astype(np.int32)
                    self.df[self.y_axis_col+"_px"] = self.df[self.y_axis_col+"_px"].astype(np.int32)
            elif self.rt_self.isPolars(self.df):
                self.df = self.df.with_columns((self.x_left                + pl.col(self.x_axis_col)*self.w_usable).alias(self.x_axis_col+"_px"))
                self.df = self.df.with_columns((self.y_ins + self.h_usable - pl.col(self.y_axis_col)*self.h_usable).alias(self.y_axis_col+"_px"))
                if self.align_pixels:
                    self.df = self.df.with_columns([pl.col(self.x_axis_col+'_px').cast(pl.Int32), pl.col(self.y_axis_col+'_px').cast(pl.Int32)])

            self.x_trans_norm_func = None
            if self.x_trans_func is not None:
                self.x_trans_norm_func = lambda x: self.x_left + self.x_trans_func(x) * self.w_usable
            self.y_trans_norm_func = None
            if self.y_trans_func is not None:
                self.y_trans_norm_func = lambda x: self.y_ins + self.h_usable - self.y_trans_func(x) * self.h_usable

            # Secondary axis pixel-level columns
            if self.y2_field and self.rt_self.isPandas(self.df2):
                if self.df2_is_df == False:
                    self.df2[self.x2_axis_col+"_px"] = self.x_left  + self.df2[self.x2_axis_col]*self.w_usable
                    if self.align_pixels:
                        self.df2[self.x2_axis_col+"_px"] =                self.df2[self.x2_axis_col+"_px"].astype(np.int32)
                self.df2[self.y2_axis_col+"_px"] = self.y_ins + self.h_usable - self.df2[self.y2_axis_col]*self.h_usable
                if self.align_pixels:
                    self.df2[self.y2_axis_col+"_px"] =                              self.df2[self.y2_axis_col+"_px"].astype(np.int32)
            elif self.y2_field and self.rt_self.isPolars(self.df2):
                if self.df2_is_df == False:
                    self.df2 = self.df2.with_columns((self.x_left                + pl.col(self.x2_axis_col)*self.w_usable).alias(self.x2_axis_col+"_px"))
                    if self.align_pixels:
                        self.df2 = self.df2.with_columns([pl.col(self.x2_axis_col+'_px').cast(pl.Int32)])
                self.df2 = self.df2.with_columns((self.y_ins + self.h_usable - pl.col(self.y2_axis_col)*self.h_usable).alias(self.y2_axis_col+"_px"))
                if self.align_pixels:
                    self.df2 = self.df2.with_columns([pl.col(self.y2_axis_col+'_px').cast(pl.Int32)])

            t1_pixel_calc_and_align = time.time()
            self.time_lu['pixel_calc_and_align'] = t1_pixel_calc_and_align - t0_pixel_calc_and_align

            # Create the SVG ... render the background
            svg_strs = []
            svg_strs.append(f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">')
            if self.background_override is None:
                background_color = self.rt_self.co_mgr.getTVColor('background','default')
            else:
                background_color = self.background_override                
            svg_strs.append(f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" fill-opacity="{self.background_opacity}" stroke="{background_color}" stroke-opacity="{self.background_opacity}" />')

            if self.plot_background_override is not None:
                _co = self.plot_background_override
                svg_strs.append(f'<rect x="{self.x_left}" y="{self.y_ins}" width="{self.w_usable}" height="{self.h_usable+1}" fill="{_co}" stroke="{_co}" />')

            # Draw the temporal context
            if self.x_is_time and self.draw_context:
                if self.x_label_min is not None and self.x_label_max is not None:
                    _ts_min,_ts_max = pd.to_datetime(self.x_label_min),pd.to_datetime(self.x_label_max)
                    svg_strs.append(self.rt_self.drawXYTemporalContext(self.x_left, self.y_ins, self.w_usable, self.h_usable, self.txt_h, _ts_min,            _ts_max,            self.draw_labels))
                else:
                    svg_strs.append(self.rt_self.drawXYTemporalContext(self.x_left, self.y_ins, self.w_usable, self.h_usable, self.txt_h, self.timestamp_min, self.timestamp_max, self.draw_labels))

            # Draw grid lines (if enabled)
            if self.draw_x_gridlines and self.x_field_is_scalar:
                svg_strs.append(self.__drawGridlines__(True,  self.x_label_min, self.x_label_max, self.x_trans_norm_func, self.w_usable, self.y_ins,  self.y_ins  + self.h_usable))
            if self.draw_y_gridlines and self.y_field_is_scalar:
                svg_strs.append(self.__drawGridlines__(False, self.y_label_min, self.y_label_max, self.y_trans_norm_func, self.h_usable, self.x_left, self.x_left + self.w_usable))
                
            # Draw the background shapes
            t0_bg_shapes = time.time()
            if self.bg_shape_lu is not None and self.x_trans_func is not None and self.y_trans_func is not None:
                _bg_shape_labels = []
                for k in self.bg_shape_lu.keys():
                    shape_desc = self.bg_shape_lu[k]
                    _shape_svg, _label_svg = self.rt_self.__transformBackgroundShapes__(k,                         shape_desc,
                                                                                        self.x_trans_norm_func,    self.y_trans_norm_func,
                                                                                        self.bg_shape_label_color, self.bg_shape_opacity,
                                                                                        self.bg_shape_fill,        self.bg_shape_stroke_w,
                                                                                        self.bg_shape_stroke,      self.txt_h)
                    svg_strs.append(_shape_svg)
                    _bg_shape_labels.append(_label_svg) # Defer render

                # Render the labels
                for _label_svg in _bg_shape_labels:
                    svg_strs.append(_label_svg)

            t1_bg_shapes = time.time()
            self.time_lu['render_bg_shapes'] = t1_bg_shapes - t0_bg_shapes

            # Draw the distributions (if selected)
            t0_render_distributions = time.time()
            if self.render_x_distribution is not None:
                svg_strs.append(self.__renderBackgroundDistribution__(True,  self.x_left, self.y_bottom, self.x_left + self.w_usable, self.y_bottom, self.x_left, self.y_ins))
            if self.render_y_distribution is not None:
                svg_strs.append(self.__renderBackgroundDistribution__(False, self.x_left, self.y_bottom, self.x_left + self.w_usable, self.y_bottom, self.x_left, self.y_ins))
            t1_render_distributions = time.time()
            self.time_lu['render_distributions'] = t1_render_distributions - t0_render_distributions

            # Axis
            axis_co = self.rt_self.co_mgr.getTVColor('axis',  'default')
            #svg_strs.append(f'<line x1="{self.x_left}" y1="{self.y_bottom}" x2="{self.x_left}"                 y2="{self.y_ins}"      stroke="{axis_co}" stroke-width=".6" />')
            #svg_strs.append(f'<line x1="{self.x_left}" y1="{self.y_bottom}" x2="{self.x_left + self.w_usable}" y2="{self.y_bottom}"   stroke="{axis_co}" stroke-width=".6" />')
            axis_path = f'M {self.x_left} {self.y_bottom} L {self.x_left} {self.y_ins} L {self.x_left + self.w_usable} {self.y_ins} L {self.x_left + self.w_usable} {self.y_bottom} Z'
            svg_strs.append(f'<path d="{axis_path}" stroke="{axis_co}" stroke-width=".4" fill="none" />')
                
            # Handle the line option... this needs to be rendered before the dots so that the lines are behind the dots
            # ... first version handles vector data...
            t0_render_lines = time.time()
            if self.line_groupby_field is not None and type(self.line_groupby_field) == list:
                svg_strs.append(self.__rendersvg_line_groupby_timestamped__())
            # ... second version handles the normal use cases...
            elif self.line_groupby_field:
                svg_strs.append(self.__rendersvg_line_groupby__())
            # Handle the line 2 option // like the first one... but some additional options, reassignments
            if self.line2_groupby_field:
                svg_strs.append(self.__rendersvg_line2_groupby__())

            t1_render_lines = time.time()
            self.time_lu['render_lines'] = t1_render_lines - t0_render_lines

            #
            # Dot Render Loops
            #
            # Small Multiples First -- can only be on the primary axis...
            if self.dot_shape == 'small_multiple':
                t0_render_small_multiples = time.time()
                node_to_xy  = {} # for small multiples
                node_to_dfs = {} # for small multiples
                if   self.rt_self.isPandas(self.df): gb = self.df.groupby([self.x_axis_col+"_px",self.y_axis_col+"_px"])
                elif self.rt_self.isPolars(self.df): gb = self.df.group_by([self.x_axis_col+"_px",self.y_axis_col+"_px"])
                else:                                raise Exception('Unknown dataframe type')
                for k, k_df in gb:
                    x,y = k
                    xy_as_str = str(x) + ',' + str(y)
                    if xy_as_str not in node_to_dfs.keys():
                        node_to_xy [xy_as_str] = (x,y)
                        node_to_dfs[xy_as_str] = []
                    node_to_dfs[xy_as_str].append(k_df)
                    if self.track_state:
                        _poly = Polygon([[x-self.sm_w/2,y-self.sm_h/2],
                                            [x-self.sm_w/2,y+self.sm_h/2],
                                            [x+self.sm_w/2,y+self.sm_h/2],
                                            [x+self.sm_w/2,y-self.sm_h/2]])
                        self.geom_to_df[_poly] = k_df
                _ts_field = self.x_field[0] if self.x_is_time else None
                sm_lu = self.rt_self.createSmallMultiples(self.df, node_to_dfs, node_to_xy,
                                                          self.count_by, self.count_by_set, self.color_by, _ts_field, self.widget_id,
                                                          self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent,
                                                          self.sm_w, self.sm_h)
                for node_str in sm_lu.keys():
                    svg_strs.append(sm_lu[node_str])
                
                t1_render_small_multiples = time.time()
                self.time_lu['render_small_multiples'] = t1_render_small_multiples - t0_render_small_multiples

            # Dots / Primary Axis
            elif dot_w is not None and dot_w != 0:
                t0_render_dots = time.time()
                if self.rt_self.isPandas(self.df):
                    svg_strs.append(self.__rendersvg_dots_pandas__(self.df,   self.x_axis_col,   self.y_axis_col,   self.color_by,    dot_w))
                elif self.rt_self.isPolars(self.df):
                    svg_strs.append(self.__rendersvg_dots_polars__(self.df,   self.x_axis_col,   self.y_axis_col,   self.color_by,    dot_w))
                t1_render_dots = time.time()
                self.time_lu['render_dots'] = t1_render_dots - t0_render_dots

            # Dots / Secondary Axis
            if dot2_w is not None and self.df2 is not None and dot2_w != 0:
                t0_render_dots2 = time.time()
                _local_color_by = self.line2_groupby_color if self.line2_groupby_color is not None else self.color_by
                if self.rt_self.isPandas(self.df):
                    svg_strs.append(self.__rendersvg_dots_pandas__(self.df2,  self.x2_axis_col,  self.y2_axis_col,  _local_color_by,  dot2_w))
                elif self.rt_self.isPolars(self.df):
                    svg_strs.append(self.__rendersvg_dots_polars__(self.df2,  self.x2_axis_col,  self.y2_axis_col,  _local_color_by,  dot2_w))
                t1_render_dots2 = time.time()
                self.time_lu['render_dots2'] = t1_render_dots2 - t0_render_dots2

            # Draw labels
            if self.draw_labels:
                svg_strs.append(self.__rendersvg_drawlabels__())
                        
            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg_strs.append(f'<rect width="{self.w-1}" height="{self.h}" x="0" y="0" fill-opacity="0.0" fill="none" stroke="{border_color}" />')
            
            svg_strs.append('</svg>')
            self.last_render = ''.join(svg_strs)
            return self.last_render
        
        #
        # Determine which dataframe geometries overlap with a specific
        #
        def overlappingDataFrames(self, to_intersect):
            _dfs = []
            for _poly in self.geom_to_df.keys():
                if _poly.intersects(to_intersect):
                    _dfs.append(self.geom_to_df[_poly])
            if len(_dfs) > 0:
                return self.rt_self.concatDataFrames(_dfs)
            else:
                return None

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
        # __rendersvg_drawlabels__() - draw the axis labels
        #
        def __rendersvg_drawlabels__(self):
            svg = ''
            #
            # X Axis
            # ... don't draw if we are rendering the distribution only (those numbers don't make sense for distributions)
            #
            if self.x_is_time:
                self.x_label_min,self.x_label_max = self.rt_self.condenseTimeLabels(self.x_label_min,self.x_label_max)

            if self.render_y_distribution is not None and (self.dot_size is None or self.dot_size == 'hidden'):
                pass
            else:
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
            # ... don't draw if we are rendering the distribution only (those numbers don't make sense for distributions)                    
            #
            if self.y_is_time:
                self.y_label_min,self.y_label_max = self.rt_self.condenseTimeLabels(self.y_label_min,self.y_label_max)

            if self.render_x_distribution is not None and (self.dot_size is None or self.dot_size == 'hidden'):
                pass
            else:
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
            return svg

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
                if   self.rt_self.isPandas(self.df):
                    if n < N:
                        _df = self.df.query(f'`{_col}` >= {(n-1)/N} and `{_col}` <  {n/N}')
                    else:
                        _df = self.df.query(f'`{_col}` >= {(n-1)/N} and `{_col}` <= {n/N}')
                elif self.rt_self.isPolars(self.df):
                    if n < N:
                        _df = self.df.filter((pl.col(_col) >= (n-1)/N) & (pl.col(_col) <  n/N))
                    else:
                        _df = self.df.filter((pl.col(_col) >= (n-1)/N) & (pl.col(_col) <= n/N))

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
        # __rendersvg_line_groupby_timestamped__(): render the line for groupby field w/ timestamp
        #
        def __rendersvg_line_groupby_timestamped__(self):
            svg   = ''
            color = self.rt_self.co_mgr.getTVColor('data','default')
            _gb_fields = self.line_groupby_field[:-1]
            if len(_gb_fields) == 1:
                _gb_fields = _gb_fields[0]

            _ts_field = self.line_groupby_field[-1]

            if   self.rt_self.isPandas(self.df):
                gb = self.df.groupby(_gb_fields)
            elif self.rt_self.isPolars(self.df):
                gb = self.df.group_by(_gb_fields)

            for k,k_df in gb:
                if   self.rt_self.isPandas(self.df):
                    gbxy = k_df.groupby([_ts_field, self.x_axis_col+"_px",self.y_axis_col+"_px"])
                elif self.rt_self.isPolars(self.df):
                    k_df = k_df.sort(_ts_field)
                    gbxy = k_df.group_by([_ts_field, self.x_axis_col+"_px",self.y_axis_col+"_px"], maintain_order=True)

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
            return svg

        #
        # __rendersvg_line_groupby__(): render the line for groupby field
        #
        def __rendersvg_line_groupby__(self):
            svg = ''
            color = self.rt_self.co_mgr.getTVColor('data','default')

            if self.rt_self.isPandas(self.df):
                gb = self.df.groupby(self.line_groupby_field)
            else:
                gb = self.df.group_by(self.line_groupby_field)

            for k,k_df in gb:
                if   self.rt_self.isPandas(self.df):
                    gbxy = k_df.groupby([self.x_axis_col+"_px",self.y_axis_col+"_px"])
                elif self.rt_self.isPolars(self.df):
                    k_df = k_df.sort(self.x_axis_col+"_px")
                    gbxy = k_df.group_by([self.x_axis_col+"_px",self.y_axis_col+"_px"], maintain_order=True)

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
            return svg

        #
        # __rendersvg_line2_groupby__(): render the line for groupby 2 field
        #
        def __rendersvg_line2_groupby__(self):
            svg = ''
            if   self.rt_self.isPandas(self.df2):
                gb = self.df2.groupby(self.line2_groupby_field)
            elif self.rt_self.isPolars(self.df2):
                gb = self.df2.group_by(self.line2_groupby_field)

            for k,k_df in gb:
                if   self.rt_self.isPandas(self.df2):
                    gbxy = k_df.groupby([self.x2_axis_col+"_px",self.y2_axis_col+"_px"])
                elif self.rt_self.isPolars(self.df2):
                    k_df = k_df.sort(self.x2_axis_col+'_px')
                    gbxy = k_df.group_by([self.x2_axis_col+"_px",self.y2_axis_col+"_px"], maintain_order=True)
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
            return svg

        #
        # __rendersvg_dots_polars__() - render dots for polars
        #
        def __rendersvg_dots_polars__(self, _df, _x_axis_col, _y_axis_col, _local_color_by, _local_dot_w):
            svg_strs = []
            proc     = []

            # Count By Calc
            if _local_dot_w < 0 or self.vary_opacity or self.color_magnitude is not None:
                if   self.count_by is None: proc.append(pl.len().alias('__countby__'))
                elif self.count_by_set:     proc.append(pl.n_unique(self.count_by).alias('__countby__'))
                else:                       proc.append(pl.sum(self.count_by).alias('__countby__'))
            
            # Color By Calc
            color, color_by_field = None, '__colorby__'
            if _local_color_by in _df.columns:
                if _df[_local_color_by].dtype == pl.Datetime:
                    _tsmin_, _tsmax_ = _df[_local_color_by].min(), _df[_local_color_by].max()
                    proc.append(((pl.col(_local_color_by).min() - _tsmin_)/(_tsmax_ - _tsmin_)).alias('__colorby__'))
                elif self.color_magnitude is None:
                    proc.append(pl.col(_local_color_by).alias('__colorby__'))
                else:
                    color_by_field = '__countby__'
            elif _local_color_by is not None and _local_color_by.startswith('#') and len(_local_color_by) == 7:
                color = _local_color_by
            else:
                color = self.rt_self.co_mgr.getTVColor('data','default')
            
            # Compute per pixel values
            gb = _df.group_by([_x_axis_col+"_px",_y_axis_col+"_px"]).agg(*proc)

            if _local_color_by in gb.columns:
                _global_min_ = gb[_local_color_by].min()
                _global_max_ = gb[_local_color_by].max()

            # Determine the min and max counts for the dot width / for contrast stretching, track counts
            if _local_dot_w < 0 or self.vary_opacity or self.color_magnitude is not None:
                max_xy = gb['__countby__'].max()
                if self.color_magnitude == 'stretch':
                    self.contrast_stretch = {}
                    _sorted_ = sorted(list(set(gb['__countby__'])))
                    for x in range(len(_sorted_)):
                        self.contrast_stretch[_sorted_[x]] = x/len(_sorted_)

            # State Tracking
            if self.track_state or callable(self.dot_shape):
                pb = _df.partition_by([_x_axis_col+"_px",_y_axis_col+"_px"], as_dict=True)

            # Loop Over The Pixels
            for _index_ in range(len(gb)):
                # Pixel Coordinates
                x, y = gb[_x_axis_col+"_px"][_index_], gb[_y_axis_col+"_px"][_index_]

                # Color Options
                if _local_color_by in _df.columns:
                    if _df[_local_color_by].dtype == pl.Datetime:
                        color        = self.rt_self.co_mgr.spectrum(gb[color_by_field][_index_], 0.0, 1.0)
                    elif self.color_magnitude is None:
                        _set_ = set(gb[color_by_field][_index_])
                        if len(_set_) > 1:
                            color = self.rt_self.co_mgr.getTVColor('data','default')
                        else:
                            color = self.rt_self.co_mgr.getColor(_set_.pop())
                    elif self.color_magnitude == 'stretch':
                        color = self.rt_self.co_mgr.spectrum(self.contrast_stretch[gb[color_by_field][_index_]], 0, 1.0, True)
                    else:
                        color = self.rt_self.co_mgr.spectrum(gb[color_by_field][_index_], 0, max_xy, self.color_magnitude)

                # Two Versions of Rendering
                if _local_dot_w > 0 and self.vary_opacity == False: # Simple Render
                    _my_dot_shape = self.dot_shape
                    if callable(self.dot_shape):
                        k_df = pb[(x,y)]
                        _my_dot_shape = self.dot_shape(k_df, (x,y), x, y, _local_dot_w, color, self.opacity)
                    svg_strs.append(self.rt_self.renderShape(_my_dot_shape, x, y, _local_dot_w, color, color, self.opacity))
                else: # Complex Render
                    my_count = gb['__countby__'][_index_]
                    
                    var_w = _local_dot_w
                    var_o = 1.0 
                    if _local_dot_w < 0 and self.vary_opacity:                    
                        var_w = 0.2 + self.max_dot_size  * my_count/max_xy
                        var_o = 0.2 + 0.8                * my_count/max_xy                    
                    elif _local_dot_w < 0:
                        var_w = 0.2 + self.max_dot_size  * my_count/max_xy                    
                    else:
                        var_o = 0.2 + 0.8                * my_count/max_xy
                    
                    _my_dot_shape = self.dot_shape
                    if callable(self.dot_shape):
                        k_df = pb[(x,y)]
                        _my_dot_shape = self.dot_shape(k_df, (x,y), x, y, var_w, color, var_o)

                    svg_strs.append(self.rt_self.renderShape(_my_dot_shape, x, y, var_w, color, color, var_o))

                # Track state (if requested)
                if self.track_state:
                    k_df = pb[(x,y)]
                    _poly = Polygon([[x-_local_dot_w,y-_local_dot_w],[x-_local_dot_w,y+_local_dot_w],
                                     [x+_local_dot_w,y+_local_dot_w],[x+_local_dot_w,y-_local_dot_w]])
                    self.geom_to_df[_poly] = k_df

            return ''.join(svg_strs)

        #
        # __rendersvg_dots_pandas__(): render the dots
        #
        def __rendersvg_dots_pandas__(self, _df,_x_axis_col,_y_axis_col,_local_color_by,_local_dot_w):
            svg_strs = []
            # Group by x,y for the render
            gb = _df.groupby([_x_axis_col+"_px",_y_axis_col+"_px"])

            # Determine the min and max counts for the dot width / for contrast stretching, track counts
            max_xy,self.stretch_histogram,self.stretch_total = 0,{},0
            if _local_dot_w < 0 or self.vary_opacity or self.color_magnitude is not None:
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

                # Determine coloring options                            
                if   _local_color_by is None:
                    color = self.rt_self.co_mgr.getTVColor('data','default')
                elif _local_color_by in k_df.columns:
                    if is_datetime(k_df[_local_color_by]):
                        _global_min_ = self.df[_local_color_by].min()
                        _global_max_ = self.df[_local_color_by].max()
                        if _global_min_ == _global_max_:
                            _global_min_ -= 0.5
                            _global_max_ += 0.5
                        _scaled_time = (k_df[_local_color_by].min() - _global_min_)/(_global_max_ - _global_min_)
                        color        = self.rt_self.co_mgr.spectrum(_scaled_time, 0.0, 1.0)
                    elif self.color_magnitude is None:
                        color_set = set(k_df[_local_color_by])
                        if len(color_set) == 1:
                            color = self.rt_self.co_mgr.getColor(color_set.pop())
                        else:
                            color = self.rt_self.co_mgr.getTVColor('data','default')
                    else:                                    
                        if self.count_by_set:                              # count by set
                            my_count = len(set(k_df[self.count_by]))                                    
                        elif self.count_by is not None:                    # count by summation
                            my_count = k_df[self.count_by].sum()
                        else:                                              # count by rows
                            my_count = len(k_df)
                        if self.color_magnitude == 'stretch':
                            color = self.rt_self.co_mgr.spectrum(self.contrast_stretch[my_count], 0, 1.0,    True)
                        else:
                            color = self.rt_self.co_mgr.spectrum(my_count, 0, max_xy, self.color_magnitude)
                elif _local_color_by.startswith('#') and len(_local_color_by) == 7:
                    color = _local_color_by
                else:
                    color = self.rt_self.co_mgr.getTVColor('data','default')

                # Render the dot
                # - Simple Render
                if _local_dot_w > 0 and self.vary_opacity == False:
                    _my_dot_shape = self.dot_shape
                    if callable(self.dot_shape):
                        _my_dot_shape = self.dot_shape(k_df, k, x, y, _local_dot_w, color, self.opacity)
                    svg_strs.append(self.rt_self.renderShape(_my_dot_shape, x, y, _local_dot_w, color, color, self.opacity))

                    # Track state (if requested)
                    if self.track_state:
                        _poly = Polygon([[x-_local_dot_w,y-_local_dot_w],
                                            [x-_local_dot_w,y+_local_dot_w],
                                            [x+_local_dot_w,y+_local_dot_w],
                                            [x+_local_dot_w,y-_local_dot_w]])
                        self.geom_to_df[_poly] = k_df

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
                    if _local_dot_w < 0 and self.vary_opacity:                    
                        var_w = 0.2 + self.max_dot_size  * my_count/max_xy
                        var_o = 0.2 + 0.8                * my_count/max_xy                    
                    elif _local_dot_w < 0:
                        var_w = 0.2 + self.max_dot_size  * my_count/max_xy                    
                    else:
                        var_o = 0.2 + 0.8                * my_count/max_xy
                    
                    _my_dot_shape = self.dot_shape
                    if callable(self.dot_shape):
                        _my_dot_shape = self.dot_shape(k_df, k, x, y, var_w, color, var_o)

                    svg_strs.append(self.rt_self.renderShape(_my_dot_shape, x, y, var_w, color, color, var_o))

                    # Track state (if requested)
                    if self.track_state:
                        _poly = Polygon([[x-var_w,y-var_w],
                                         [x-var_w,y+var_w],
                                         [x+var_w,y+var_w],
                                         [x+var_w,y-var_w]])
                        self.geom_to_df[_poly] = k_df

            return ''.join(svg_strs)

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

