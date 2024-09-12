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
import random
import re

from shapely.geometry import Polygon

from math import sqrt

from .rt_component import RTComponent

__name__ = 'rt_periodic_barchart_mixin'

#
# Periodic BarChart Mixin
# - Most of this code is lifted from the TemporalBarChart mixin
#
class RTPeriodicBarChartMixin(object):
    #
    # Init / Constructor for this mixin
    #
    def __periodic_barchart_mixin_init__(self):
        # Only periods allowed
        self.time_periods      = ['quarter', 'month', 'day', 'day_of_week', 'day_of_week_hour', 'hour', 'minute', 'second']
        
        # Number of bins in each period
        self.time_periods_bins = [4,         12,      31,             7,             7*24,               24,     60,       60]
        
        # Ordering of the periods
        self.time_periods_strs  = []
        for period_i in range(0,len(self.time_periods)):
            period = self.time_periods[period_i]
            if   period == 'quarter':
                self.time_periods_strs.append(['Q1',  'Q2',  'Q3',  'Q4'])
            elif period == 'month':
                self.time_periods_strs.append(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
            elif period == 'day': # day of month
                my_arr = []
                for i in range(1,32):
                    my_arr.append(f'{i:02}')
                self.time_periods_strs.append(my_arr)
            elif period == 'day_of_week':
                self.time_periods_strs.append(['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
            elif period == 'day_of_week_hour':
                my_arr = []
                for dow in ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']:
                    for hr in range(0,24):
                        my_arr.append(dow + f'-{hr:02}')
                self.time_periods_strs.append(my_arr)
            elif period == 'hour' or period == 'minute' or period == 'second':
                my_arr = []
                for i in range(0,self.time_periods_bins[period_i]):
                    my_arr.append(f'{i:02}')
                self.time_periods_strs.append(my_arr)
                
        # Double check the bin counts...
        for period_i in range(0,len(self.time_periods)):
            if self.time_periods_bins[period_i] != len(self.time_periods_strs[period_i]):
                raise Exception(f'time period {self.time_periods[period_i]} not set correctly')

    #
    # periodicBarChartPreferredSize()
    # - Return the preferred size
    #
    def periodicBarChartPreferredDimensions(self, time_period=None, min_bar_w=2, **kwargs):
        if time_period is not None and time_period in self.time_periods:
            i    = self.time_periods.index(time_period)
            bins = self.time_periods_bins[i]
            w    = bins * min_bar_w + 15 # give each bar 3 pixels... add a little for inserts and labeling
            h    = int(0.33*w)
            if h < 48:
                h = 48
                w = 3*h
            return (w,h)
        return (384,128)

    #
    # periodicBarChartMinimumSize()
    # - Return the minimum size
    #
    def periodicBarChartMinimumDimensions(self, time_period=None, min_bar_w=2, **kwargs):
        if time_period is not None and time_period in self.time_periods:
            i    = self.time_periods.index(time_period)
            bins = self.time_periods_bins[i]
            w    = bins * min_bar_w + 15 # give each bar 2 pixels... add a little for inserts and labeling
            h    = int(0.33*w)
            if h < 24:
                h = 24
                w = 3*h
            return (w,h)
        return (256,96)

    #
    # periodicBarChartSmallMultipleSize()
    #
    def periodicBarChartSmallMultipleDimensions(self, time_period=None, min_bar_w=2, **kwargs):
        return (32,24)

    #
    # periodicBarChartRequiredFields()
    # - Required fields for this configuration of the periodic barchart
    #
    def periodicBarChartRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('ts_field', kwargs, columns_set) # may need additional work if "None"
        self.identifyColumnsFromParameters('color_by', kwargs, columns_set)
        self.identifyColumnsFromParameters('count_by', kwargs, columns_set)
        return columns_set

    #
    # periodicBarChart
    # 
    def periodicBarChart(self,
                         df,                                    # dataframe to render
                         # ------------------------------------ # everything else is a default...
                         time_period           = 'day_of_week', # periodicity to render
                         ts_field              = None,          # timestamp field // needs to be a np.datetime64 column...                         
                         color_by              = None,          # just the default color or a string for a field
                         global_color_order    = None,          # color by ordering... if none (default), will be created and filled in...                         
                         count_by              = None,          # none means just count rows, otherwise, use a field to sum by
                         count_by_set          = False,         # count by using a set operation                         
                         widget_id             = None,          # naming the svg elements                         
                         # ------------------------------------ #                         
                         global_max            = None,          # maximum to use for the bar heights
                         global_min            = None,          # minimum (only used for the boxplot style(s))                         
                         just_calc_max         = False,         # forces return of the maximum for this render config
                                                                # ... which will then be used for the global max across bar charts
                         # ------------------------------------ #
                         style                 = 'barchart',    # 'barchart' or 'boxplot' or 'boxplot_w_swarm'
                         cap_swarm_at          = 200,           # cap the swarm plot at the specified number... if set to None, then no caps
                         # ------------------------------------ # small multiple options
                         sm_type               = None,          # should be the method name // similar to the smallMultiples method
                         sm_w                  = None,          # override the width of the small multiple
                         sm_h                  = None,          # override the height of the small multiple
                         sm_params             = {},            # dictionary of parameters for the small multiples
                         sm_x_axis_independent = True,          # Use independent axis for x (xy, temporal, and linkNode)
                         sm_y_axis_independent = True,          # Use independent axis for y (xy, temporal, periodic, pie)                         
                         # ------------------------------------ #                
                         track_state           = False,         # state tracking for interactive filtering                         
                         x_view                = 0,             # x offset for the view
                         y_view                = 0,             # y offset for the view
                         w                     = 512,           # width of the view
                         h                     = 128,           # height of the view
                         h_gap                 = 0,             # gap between bars.. should be a zero or a one...
                         min_bar_w             = 2,             # minimum bar width
                         txt_h                 = 14,            # text height for the labels
                         x_ins                 = 3,             # x insert (on both sides of the drawing)
                         y_ins                 = 3,
                         draw_labels           = True,          # draw labels flag
                         draw_border           = True,          # draw a border around the bar chart
                         draw_context          = True):         # draw background hints about the years, months, days, etc.
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTPeriodicBarChart(self, **_params_)

    #
    # RTPeriodicBarChart()
    #
    class RTPeriodicBarChart(RTComponent):
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
                self.widget_id = "periodicbarchart_" + str(random.randint(0,65535))

            self.time_period           = kwargs['time_period']
            self.ts_field              = kwargs['ts_field']
            self.color_by              = kwargs['color_by']
            self.global_color_order    = kwargs['global_color_order']
            self.count_by              = kwargs['count_by']
            self.count_by_set          = kwargs['count_by_set']
            self.global_max            = kwargs['global_max']
            self.global_min            = kwargs['global_min']
            self.style                 = kwargs['style']
            self.cap_swarm_at          = kwargs['cap_swarm_at']
            self.sm_type               = kwargs['sm_type']
            self.sm_w                  = kwargs['sm_w']
            self.sm_h                  = kwargs['sm_h']
            self.sm_params             = kwargs['sm_params']
            self.sm_x_axis_independent = kwargs['sm_x_axis_independent']
            self.sm_y_axis_independent = kwargs['sm_y_axis_independent']
            self.track_state           = kwargs['track_state']
            self.x_view                = kwargs['x_view']
            self.y_view                = kwargs['y_view']
            self.w                     = kwargs['w']
            self.h                     = kwargs['h']
            self.h_gap                 = kwargs['h_gap']
            self.min_bar_w             = kwargs['min_bar_w']
            self.txt_h                 = kwargs['txt_h']
            self.x_ins                 = kwargs['x_ins']
            self.y_ins                 = kwargs['y_ins']
            self.draw_labels           = kwargs['draw_labels']
            self.draw_border           = kwargs['draw_border']
            self.draw_context          = kwargs['draw_context']

            # Determine the timestamp field
            if self.ts_field is None:
                self.ts_field = rt_self.guessTimestampField(self.df)
            
            # Determine the periodicity index
            self.time_period = kwargs['time_period']
            self.period_i = rt_self.time_periods.index(kwargs['time_period'])

            # Perform the transforms
            # Apply count-by transofmrs
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
        # dataFrameRanges() - calculate the max and min of the bars
        #
        def dataFrameRanges(self, period_field):
            if   self.rt_self.isPandas(self.df):
                return self.__dataFrameRanges_pandas__(period_field)
            elif self.rt_self.isPolars(self.df):
                return self.__dataFrameRanges_polars__(period_field)
            else:
                raise Exception('RTPeriodicBarChart.dataFrameRanges() -- only pandas and polars supported')

        #
        def __dataFrameRanges_pandas__(self, period_field):
            group_by = self.df.groupby(period_field)
            if   self.count_by is None:
                group_by_min = 0
                group_by_max = group_by.size().max()
            elif self.count_by_set:
                _df         = self.df.groupby([period_field,self.count_by]).size().reset_index()
                _df_for_max = _df.groupby(period_field)
                group_by_min = 0
                group_by_max = _df_for_max.size().max()
            elif self.style.startswith('boxplot'):
                group_by_min = self.df[self.count_by].min()                    
                group_by_max = self.df[self.count_by].max()
            else:
                group_by_min = 0
                group_by_max = group_by[self.count_by].sum().max()
            return group_by, group_by_min, group_by_max

        #
        def __dataFrameRanges_polars__(self, period_field):
            group_by_min = 0
            if   self.count_by is None:
                tmp_df = self.df.drop(set(self.df.columns) - set([period_field])) \
                                .group_by(period_field)                           \
                                .agg(pl.len()                                     \
                                       .alias('__count__'))
                group_by_max = tmp_df['__count__'].max()
            elif self.count_by_set:
                tmp_df = self.df.drop(set(self.df.columns) - set([period_field, self.count_by])) \
                                .group_by(period_field)                                          \
                                .n_unique()
                group_by_max = tmp_df[self.count_by].max()
            elif self.style.startswith('boxplot'):
                group_by_min = self.df[self.count_by].min()                    
                group_by_max = self.df[self.count_by].max()
            else:
                tmp_df = self.df.drop(set(self.df.columns) - set([period_field, self.count_by])) \
                                .group_by(period_field)                                          \
                                .agg(pl.sum(self.count_by)                                       \
                                       .alias('__count__'))
                group_by_max = tmp_df['__count__'].max()
            return self.df.group_by(period_field), group_by_min, group_by_max

        #
        # print() version of class
        #
        def __repr__(self):
            def tQontQ(t): return 'None' if t is None else "'" + str(t) + "'"
            return f'periodicBarChart(df.len={len(self.df)}, ts_field=\'{self.ts_field}\', period=\'{self.time_period}\', count_by={tQontQ(self.count_by)}, color_by={tQontQ(self.color_by)}, {self.w}x{self.h})'

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None: self.renderSVG()
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

            # If the height/width are less than the minimums, turn off labeling... and make the min_bar_w = 1
            # ... for small multiples
            params_orig_minus_self = self.parms.copy()
            params_orig_minus_self.pop('self')
            min_dims = self.rt_self.periodicBarChartSmallMultipleDimensions(**params_orig_minus_self)
            if self.w <= min_dims[0] or self.h <= min_dims[1]:
                self.draw_labels = False
                self.min_bar_w   = 1
                self.x_ins       = 1
                self.y_ins       = 1
                self.h_gap       = 0

            # Limited range of pixels are allowed...
            if   self.h_gap > 4:
                self.h_gap = 4
            elif self.h_gap < 0:
                self.h_gap = 0
            
            # Calculate the usable width
            w_usable = self.w - 2*self.x_ins
            x_left   = self.x_ins
            if self.draw_labels:
                x_left    =           2*self.y_ins + self.txt_h
                w_usable  = self.w - (3*self.y_ins + self.txt_h)

            # Create a new field with the periodic value there
            t_field = self.rt_self.createTField(self.ts_field, self.time_period)
            self.df, period_field = self.rt_self.applyTransform(self.df, t_field)

            # Total number of bins
            bins    = self.rt_self.time_periods_bins[self.period_i]

            # Adjust the min_bar_w if this is small multiples are to be included
            if self.sm_type is not None:
                if self.sm_w is None or self.sm_h is None:
                    self.sm_w,self.sm_h = getattr(self.rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
                min_bar_w = self.sm_w # do we even use min_bar_w anywhere?
                bar_w   = ((w_usable - (bins*self.h_gap))/bins)
                if   bar_w <  self.sm_w:
                    bar_w = self.sm_w                
                    if self.draw_labels:
                        self.w   =           3*self.y_ins + self.txt_h + bins*bar_w
                        w_usable = self.w - (3*self.y_ins + self.txt_h)
                    else:
                        w        =           2*self.y_ins              + bins*bar_w
                        w_usable = self.w - (2*self.y_ins)
                elif bar_w == self.sm_w:
                    pass
                else:
                    sm_prop   = self.sm_h/self.sm_w
                    self.sm_w = bar_w
                    self.sm_h = bar_w * sm_prop
            else:
                # Finalize the bar width
                bar_w   = ((w_usable - (bins*self.h_gap))/bins)

            # Height geometry
            sm_cy = 0
            if self.sm_type is not None:
                max_bar_h = self.h - 2*self.y_ins             - self.sm_h
                sm_cy     = self.h -   self.y_ins - max_bar_h - self.sm_h/2
                if self.draw_labels:
                    max_bar_h = self.h - 2*self.y_ins - self.sm_h - self.txt_h - 2
            else:
                max_bar_h = self.h - 2*self.y_ins
                if self.draw_labels:
                    max_bar_h = self.h - 2*self.y_ins - self.txt_h - 2

            # Determine the max
            group_by, group_by_min, group_by_max = self.dataFrameRanges(period_field)
            if self.global_max is not None:
                group_by_min, group_by_max = self.global_min, self.global_max
            if just_calc_max:
                return group_by_min,group_by_max

            # Start the SVG Frame
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'
            
            # Draw the background for the temporal chart
            textfg = self.rt_self.co_mgr.getTVColor('label','defaultfg')
            y_baseline = self.h - self.y_ins - 1
            if self.draw_labels:
                y_baseline = self.h - self.y_ins - self.txt_h - 1

            # Draw temporal context
            if self.draw_context:
                pass
            
            sm_adj = 0
            if self.sm_type is not None:
                sm_adj = self.sm_h

            # Draw the axes
            axis_co = self.rt_self.co_mgr.getTVColor('axis','default')
            svg += f'<line x1="{x_left}" y1="{y_baseline+1}" x2="{x_left}"            y2="{self.y_ins+sm_adj}" stroke="{axis_co}" stroke-width="1" />'
            svg += f'<line x1="{x_left}" y1="{y_baseline+1}" x2="{x_left + w_usable}" y2="{y_baseline+1}"      stroke="{axis_co}" stroke-width="1" />'

            # Draw the bars
            if self.style == 'barchart':
                for k,k_df in group_by:
                    if type(k) == tuple and len(k) == 1: k = k[0] # polars fixes on 2024-07-19
                    x = x_left + 1 + (bar_w+self.h_gap)*self.rt_self.time_periods_strs[self.period_i].index(k)
                    if   self.count_by is None:
                        px = max_bar_h * len(k_df) / group_by_max
                    elif self.count_by_set:
                        px = max_bar_h * len(set(k_df[self.count_by])) / group_by_max
                    else:
                        px = max_bar_h * k_df[self.count_by].sum() / group_by_max
                            
                    svg += self.rt_self.colorizeBar(k_df, self.global_color_order, self.color_by, self.count_by, self.count_by_set, x, y_baseline, px, bar_w, False)

                    if self.track_state:
                        _poly = Polygon([[x,y_baseline],[x+bar_w,y_baseline],[x+bar_w,y_baseline-px],[x,y_baseline-px]])
                        self.geom_to_df[_poly] = k_df

            elif self.style.startswith('boxplot'):
                # Adjust the bar width to something reasonable
                _bar_w = bar_w
                if _bar_w > 16:
                    _bar_w = 16
                
                # Make a y-transform lambda expression
                if group_by_max == 0:
                    group_by_max = 1
                yT = lambda __y__: (y_baseline - max_bar_h * (__y__ - group_by_min) / (group_by_max - group_by_min))

                # Render the boxplot columns
                for k,k_df in group_by:
                    x = x_left + 1 + (bar_w+self.h_gap)*self.rt_self.time_periods_strs[self.period_i].index(k)
                    _cx = x + bar_w/2

                    svg += self.rt_self.renderBoxPlotColumn(self.style, k_df, _cx, yT, group_by_max, group_by_min, _bar_w, self.count_by, self.color_by, self.cap_swarm_at)

                    if self.track_state:
                        _poly = Polygon([[x,y_baseline],[x+bar_w,y_baseline],[x+bar_w,y_baseline-max_bar_h],[x,y_baseline-max_bar_h]])
                        self.geom_to_df[_poly] = k_df

            else:
                raise Exception(f'RTPeriodicBarChart() - unknown style "{self.style}"')


            # Handle the small multiple renders
            if self.sm_type is not None:
                if   self.rt_self.isPandas(self.df): group_by = self.df.groupby(period_field)
                elif self.rt_self.isPolars(self.df): group_by = self.df.group_by(period_field)

                node_to_xy  = {}
                node_to_dfs = {}

                for key,key_df in group_by:
                    if type(key) == tuple and len(key) == 1: key = key[0] # polars fixes on 2024-07-19
                    x = x_left + 1 + (bar_w+self.h_gap)*self.rt_self.time_periods_strs[self.period_i].index(key)
                    if len(key_df) != 0:
                        node_to_xy[key]  = [x + bar_w/2, sm_cy]
                        node_to_dfs[key] = key_df
                
                sm_lu = self.rt_self.createSmallMultiples(self.df, node_to_dfs, node_to_xy,
                                                          self.count_by, self.count_by_set, self.color_by, self.ts_field, self.widget_id,
                                                          self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent,
                                                          self.sm_w, self.sm_h)

                for node_str in sm_lu.keys():
                    svg += sm_lu[node_str]

            # Draw the labels // mirrors the rt_temporal_barchart_mixin codeblock
            if self.draw_labels:
                _label_x_min_  = self.rt_self.time_periods_strs[self.period_i][0]
                _label_x_max_  = self.rt_self.time_periods_strs[self.period_i][-1]
                _label_x_axis_ = self.rt_self.time_periods[self.period_i]
                _len_x_min_, _len_x_max_ = self.rt_self.textLength(_label_x_min_,  self.txt_h), self.rt_self.textLength(_label_x_max_, self.txt_h)
                _len_x_axis_             = self.rt_self.textLength(_label_x_axis_, self.txt_h)

                if (_len_x_min_ + _len_x_max_ + _len_x_axis_ + 20) < w_usable:
                    svg += self.rt_self.svgText(_label_x_min_,   x_left,              self.h-3, self.txt_h)
                    svg += self.rt_self.svgText(_label_x_max_,   self.w - self.x_ins, self.h-3, self.txt_h, anchor='end')
                    svg += self.rt_self.svgText(_label_x_axis_,  x_left + w_usable/2, self.h-3, self.txt_h, anchor='middle')
                elif (_len_x_axis_ + 10) < w_usable:
                    svg += self.rt_self.svgText(_label_x_axis_,  x_left + w_usable/2, self.h-3, self.txt_h, anchor='middle')

                # Max Label
                _str_max,_str_min = f'{group_by_max:{self.rt_self.fformat}}',''
                if re.match(r'.*\.0*',_str_max):
                    _str_max = _str_max[:_str_max.index('.')]
                svg += self.rt_self.svgText(_str_max,     self.x_ins+self.txt_h, self.y_ins + sm_adj,             self.txt_h, anchor='end',    rotation=-90)

                # Min Label (for boxplot only)
                if self.style.startswith('boxplot'):
                    _str_min = f'{group_by_min:{self.rt_self.fformat}}'
                    if re.match(r'.*\.0*',_str_min):
                        _str_min = _str_min[:_str_min.index('.')]
                    svg += self.rt_self.svgText(_str_min, self.x_ins+self.txt_h, self.y_ins + sm_adj + max_bar_h, self.txt_h, anchor='start',  rotation=-90)

                # Count By Label
                _count_by_str = 'rows'
                if self.count_by:
                    _count_by_str = self.count_by
                if self.rt_self.textLength(_str_max, self.txt_h) + self.rt_self.textLength(_str_min, self.txt_h) + self.rt_self.textLength(_count_by_str, self.txt_h) + 10 < max_bar_h:
                    if self.style.startswith('boxplot'):
                        _mid_y = self.rt_self.textLength(_str_max, self.txt_h) + (max_bar_h - self.rt_self.textLength(_str_max, self.txt_h) - self.rt_self.textLength(_str_min, self.txt_h))/2
                        svg += self.rt_self.svgText(_count_by_str, self.x_ins+self.txt_h, 
                                                    self.y_ins + sm_adj + _mid_y, 
                                                    self.txt_h, anchor='middle',  rotation=-90)
                    else:
                        svg += self.rt_self.svgText(_count_by_str, self.x_ins+self.txt_h, self.y_ins + sm_adj + max_bar_h,   self.txt_h, anchor='start',  rotation=-90)

            # Draw the border
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
            # Create a new field with the periodic value there
            t_field = self.rt_self.createTField(self.ts_field, self.time_period)
            self.df, period_field = self.rt_self.applyTransform(self.df, t_field)

            # Total number of bins
            bins    = self.rt_self.time_periods_bins[self.period_i]

            # Determine the max
            group_by, group_by_min, group_by_max = self.dataFrameRanges(period_field)
            if self.global_max is not None:
                self.global_min, self.global_max = group_by_min, group_by_max
            
            # Iterate over the keys
            max_bar_h,fv = 1.0,{}
            for k,k_df in group_by:
                if   self.count_by is None:
                    px = max_bar_h * len(k_df) / group_by_max
                elif self.count_by_set:
                    px = max_bar_h * len(set(k_df[self.count_by])) / group_by_max
                else:
                    px = max_bar_h * k_df[self.count_by].sum() / group_by_max
                fv[k] = px
            
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
                return self.rt_self.concatDataFrames(_dfs)
            else:
                return None
