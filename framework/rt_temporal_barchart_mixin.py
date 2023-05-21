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
import time
import random
import re

from datetime import datetime

from math import sqrt

__name__ = 'rt_temporal_barchart_mixin'

#
# Temporal BarChart Mixin
#
class RTTemporalBarChartMixin(object):
    #
    # Init / Constructor for this mixin
    #
    def __temporal_barchart_mixin_init__(self):
        _tups = [('100YS',  '%Y',                'Centuries'),
                 ('10YS',   '%Y',                'Decades'),
                 ('YS',     '%Y',                'Years'),
                 ('QS',     '%Y-%m',             'Quarters'),
                 ('MS',     '%Y-%m',             'Months'),
                 ('1W',     '%Y-%m-%d',          'Weeks'),
                 ('D',      '%Y-%m-%d',          'Days'),
                 ('4H',     '%Y-%m-%d %H',       '4 Hours'),
                 ('H',      '%Y-%m-%d %H',       'Hours'),
                 ('15MIN',  '%Y-%m-%d %H:%M',    '15 Mins'),
                 ('10MIN',  '%Y-%m-%d %H:%M',    '10 Mins'),
                 ('5MIN',   '%Y-%m-%d %H:%M',    '5 Mins'),
                 ('1MIN',   '%Y-%m-%d %H:%M',    'Minutes'),
                 ('15S',    '%Y-%m-%d %H:%M:%S', '15 Secs'),
                 ('S',      '%Y-%m-%d %H:%M:%S', 'Seconds')]

        self.time_rezes     = []
        self.time_rezes_fmt = []
        self.time_rezes_str = []
        for _tup in _tups:
            self.time_rezes.    append(_tup[0])
            self.time_rezes_fmt.append(_tup[1])
            self.time_rezes_str.append(_tup[2])

    #
    # Return the time granularity found in an array of timestamps...
    # ... granularity is depicted as the strftime() value that has variation
    #
    # Faster version... testing on 648K rows == 5s vs 47s for the _SLOW version
    #
    def temporalGranularity(self, my_df, ts_field):
        # Very simple test to determine if it should be second level...
        if my_df.iloc[0][ts_field].second != 0:
            return 'S'
        
        # Harder tests... starting from the bottom and working up...
        my_set = set(my_df[ts_field].apply(lambda x: pd.to_datetime(x).second))
        if len(my_set) > 1 or 0 not in my_set:
            return 'S'

        my_set = set(my_df[ts_field].apply(lambda x: pd.to_datetime(x).minute))
        if len(my_set) > 1 or 0 not in my_set:
            return 'M'

        my_set = set(my_df[ts_field].apply(lambda x: pd.to_datetime(x).hour))
        if len(my_set) > 1 or 0 not in my_set:
            return 'H'

        my_set = set(my_df[ts_field].apply(lambda x: pd.to_datetime(x).day))
        if len(my_set) > 1 or 1 not in my_set:
            return 'd'

        my_set = set(my_df[ts_field].apply(lambda x: pd.to_datetime(x).month))
        if len(my_set) > 1 or 1 not in my_set:    
            return 'm'

        return 'Y'
        
    #
    # Determine if the time granularity exceeds the resolution setting...
    #
    def granularityExceedsResolution(self, granularity, time_rez):
        if   granularity == 'f' or \
             granularity == 'S':      # Seconds or less than seconds
            return False
        elif granularity == 'M':      # Minutes
            return time_rez in [                                                                                        '15S','S']
        elif granularity == 'H':      # Hour
            return time_rez in [                                                        '15MIN','10MIN','5MIN','1MIN',  '15S','S']
        elif granularity == 'd':      # Days
            return time_rez in [                                             '4H','H',  '15MIN','10MIN','5MIN','1MIN',  '15S','S']
        elif granularity == 'm':      # Months
            return time_rez in [                                  '1W','D',  '4H','H',  '15MIN','10MIN','5MIN','1MIN',  '15S','S']
        else:                         # Years
            return time_rez in [                      'QS','MS',  '1W','D',  '4H','H',  '15MIN','10MIN','5MIN','1MIN',  '15S','S']

    #
    # Render context information for the specified time_rez
    #
    def drawTemporalContext(self, time_rez, groupby, bar_w, h_gap, x, y, w, h, txt_h):
        svg  = ''
        fill_co       = self.co_mgr.getTVColor('context','default')
        txt_co        = self.co_mgr.getTVColor('context','text')
        axis_major_co = self.co_mgr.getTVColor('axis','major')
        axis_minor_co = self.co_mgr.getTVColor('axis','minor')

        #         time_rez     fmt         t_part_func         mod  e  ##  secondary
        fmt_lu = {'100YS':    ('%Y',       lambda x: x.year,   100, 0, 10, False),
                  '10YS':     ('%Y',       lambda x: x.year,   20,  0, 1,  False),
                  'YS':       ('%Y',       lambda x: x.year,   10,  0, 5,  False),
                  'QS':       ('%Y',       lambda x: x.year,   2,   0, 4,  True, lambda x: x.month,  1),
                  'MS':       ('%Y',       lambda x: x.year,   1,   0, 12, True, lambda x: x.month,  1),
                  '1W':       ('%m/%d',    lambda x: x.week,   4,   0, 4,  False),
                  'D':        ('%m/%d',    lambda x: x.day,    40,  1, 15, False), # INCLUDES KLUGE BELOW
                  '4H':       ('%m/%d',    lambda x: x.day,    2,   0, 6,  True, lambda x: x.hour,   0),
                  'H':        ('%H:00',    lambda x: x.hour,   12,  0, 6,  False),
                  '15MIN':    ('%H',       lambda x: x.hour,   2,   0, 2,  True, lambda x: x.minute, 0),
                  '10MIN':    ('%H:%M',    lambda x: x.hour,   2,   0, 2,  True, lambda x: x.minute, 0),
                  '5MIN':     ('%H:%M',    lambda x: x.minute, 15,  0, 5,  False),
                  '1MIN':     ('%H:%M',    lambda x: x.minute, 10,  0, 5,  False),
                  '15S':      ('%H:%M:%S', lambda x: x.minute, 10,  0, 20, True, lambda x: x.second, 0),
                  'S':        ('%H:%M:%S', lambda x: x.minute, 1,   0, 30, True, lambda x: x.second, 0) }

        if time_rez not in fmt_lu.keys():
            raise Exception(f'drawTemporalContext() - {time_rez} not a valid time resolution')

        fmt = fmt_lu[time_rez]

        my_x = x
        for ts_tuple in groupby:
            str_formatter     = fmt[0]
            time_part_func    = fmt[1]
            mod_value         = fmt[2]
            compare_to        = fmt[3]
            cover_how_many    = fmt[4] # not currently used
            secondary_matters = fmt[5]

            ts      =  ts_tuple[0]
            ts_part =  time_part_func(ts)
            if (ts_part%mod_value) == compare_to or (time_rez == 'D' and ts_part == 15): # <== KLUGE HERE
                plot_context = False
                if secondary_matters:
                    secondary = (fmt[6])(ts)
                    plot_context = (secondary == fmt[7])
                else:
                    plot_context = True

                # Plot the context
                if plot_context:
                    ts_str  =  pd.to_datetime(ts).strftime(str_formatter)
                    # svg += f'<rect width="{cover_how_many*(bar_w+h_gap)}" height="{h}" x="{my_x}" y="{y}" fill="{fill_co}" stroke-opacity="0.0" />'
                    svg += f'<line x1="{my_x}" y1="{y}" x2="{my_x}" y2="{y+h}" stroke="{axis_major_co}" stroke_width="0.5" />'
                    svg += f'<text x="{my_x+2}" y="{y+txt_h+1}" font-family="{self.default_font}" fill="{txt_co}" font-size="{3*txt_h/4}px">'
                    svg += f'{ts_str}</text>'
                else:
                    svg += f'<line x1="{my_x}" y1="{y}" x2="{my_x}" y2="{y+5}" stroke="{axis_minor_co}" stroke_width="0.5" />'
            else:
                svg += f'<line x1="{my_x}" y1="{y}" x2="{my_x}" y2="{y+5}" stroke="{axis_minor_co}" stroke_width="0.5" />'
            my_x   += bar_w+h_gap
            
        return svg

    #
    # Make the SVG for a multiTemporalBarChart from a dataframe
    #
    def multiTemporalBarCharts(self,
                        df,                                # dataframe to render
                        # ------------------------------   # everything else is a default
                        ts_field                 = None,   # timestamp field                               
                        color_bys                = None,   # just the default color or a string for a field
                        count_bys                = None,   # none means just count rows, otherwise, use a field to sum by
                        widget_id                = None,   # naming the svg elements
                        x_view                   = 0,      # y offset for the view
                        y_view                   = 0,      # x offset for the view
                        w                        = 256,    # width a single temporal barchart
                        h                        = 96 ,    # height of a single temporal barchart
                        heading_h                = 18,     # heading per section height (only applies if draw_labels is set)
                        temporal_granularity     = None,   # minimum temporal granularity to use -- based on strftime, use 'f' for most rez
                        h_gap                    = 0,      # gap between bars
                        draw_labels              = True,   # draw labels flag
                        draw_context             = True,   # render the temporal context
                        draw_perf_stats          = False   # display information about render times
                        # log_scale=True                   # doesn't work yet
                       ):
        # Preserve original
        df = df.copy()
        
        # Keep a copy for passing to the individual bar charts
        my_args = locals()
        
        # Make a widget_id if it's not set already
        if widget_id == None:
            widget_id = "multitemporalbarchart_" + str(random.randint(0,65535))

        # Determine the timestamp field
        if ts_field is None:
            choices = df.select_dtypes(np.datetime64).columns
            if len(choices) == 1:
                ts_field = choices[0]
            elif len(choices) > 1:
                print('multiple timestamp fields... choosing the first (multiTemporalBarChart)')
                ts_field = choices[0]
            else:
                raise Exception('no timestamp field supplied to multiTemporalBarChart(), cannot automatically determine field')
            
        # Make sure count_by and color_by are both in list format..
        if type(color_bys) != list:
            color_bys = [color_bys]
        if type(count_bys) != list:
            count_bys = [count_bys]

        # Perform the transforms
        df, color_bys = self.transformFieldListAndDataFrame(df, color_bys)
        df, count_bys = self.transformFieldListAndDataFrame(df, count_bys)

        # Determine the temporal granularity of the data ... should preclude finer resolution renders...
        if temporal_granularity is None:
            temporal_granularity = self.temporalGranularity(df, ts_field) # Too expensive as written...
            
        # Overall height of each row
        h_plus = h
        if draw_labels:
            h_plus = h + heading_h
            
        # Count By's in Y
        if len(count_bys) > len(color_bys):
            w_overall = len(color_bys) * w
            h_overall = len(count_bys) * h_plus
            
        # Color By's in Y
        else:
            h_overall = len(color_bys) * h_plus
            w_overall = len(count_bys) * w
    
        # Start the SVG
        svg = f'<svg x="{x_view}" y="{y_view}" width="{w_overall}" height="{h_overall}" xmlns="http://www.w3.org/2000/svg">'
        background_color = self.co_mgr.getTVColor('background','default')
        svg += f'<rect width="{w_overall-1}" height="{h_overall-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'
        
        # Count By Labels
        if draw_labels:
            textfg = self.co_mgr.getTVColor('label','defaultfg')
            if len(count_bys) > len(color_bys):
                for i in range(0,len(count_bys)):
                    svg += self.svgText(f'Count By "{count_bys[i]}"', 2, heading_h+i*h_plus-2, heading_h-4)
            # Color By Labels
            else:
                for i in range(0,len(color_bys)):
                    svg += self.svgText(f'Color By "{color_bys[i]}"', 2, heading_h+i*h_plus-2, heading_h-4)
        
        # Iterate over color_bys
        my_iter = 0
        for color_by_i in range(0,len(color_bys)):
            color_by = color_bys[color_by_i]
            
            # Iterate over count_bys
            for count_by_i in range(0,len(count_bys)):
                count_by = count_bys[count_by_i]
                
                # Count By's in Y
                if len(count_bys) > len(color_bys):
                    y_view_tbc = count_by_i * h_plus
                    if draw_labels:
                        y_view_tbc += heading_h                    
                    x_view_tbc = color_by_i * w

                # Color By's in Y
                else:
                    y_view_tbc = color_by_i * h_plus
                    if draw_labels:
                        y_view_tbc += heading_h
                    x_view_tbc = count_by_i * w
                                                
                child_widget_id = widget_id + "_" + str(my_iter)
                my_iter += 1
                
                child_args = my_args.copy()
                child_args.pop('self')
                child_args.pop('count_bys')
                child_args.pop('color_bys')
                child_args.pop('heading_h')
                child_args['x_view']               = x_view_tbc
                child_args['y_view']               = y_view_tbc
                child_args['color_by']             = color_by
                child_args['count_by']             = count_by
                child_args['widget_id']            = child_widget_id
                child_args['temporal_granularity'] = temporal_granularity
                
                svg += self.temporalBarChart(**child_args)
                
        svg += f'</svg>'
        return svg

    #
    # temporalBarChartPreferredDimensions()
    # - Need to check the temporal granularity if passed in
    # - Return the preferred size
    #
    def temporalBarChartPreferredDimensions(self, **kwargs):
        return (384,128)

    #
    # temporalBarChartMinimumDimensions()
    # - Need to check the temporal granularity if passed in
    # - Return the minimum size
    #
    def temporalBarChartMinimumDimensions(self, **kwargs):
        return (256,96)

    #
    # temporalBarChartSmallMultipleDimensions()
    #
    def temporalBarChartSmallMultipleDimensions(self, **kwargs):
        return (48,32)

    #
    # Identify the required fields...
    #
    def temporalBarChartRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('ts_field', kwargs, columns_set) # Needs additional work if set to "None"
        self.identifyColumnsFromParameters('color_by', kwargs, columns_set)
        self.identifyColumnsFromParameters('count_by', kwargs, columns_set)
        return columns_set

    #
    # temporalBarChart
    # 
    # Make the SVG for a temporal barchart from a dataframe
    # Ref:  https://jakevdp.github.io/PythonDataScienceHandbook/03.11-working-with-time-series.html
    #
    def temporalBarChart(self,
                         df,                           # dataframe to render
                         
                         # --------------------------- # everything else is a default...
                         
                         ts_field             = None,  # timestamp field // needs to be a np.datetime64 column...
                         ts_min               = None,  # Render ranges... if they need to be overwritten
                         ts_max               = None,  # Render ranges... if they need to be overwritten
                         temporal_granularity = None,  # minimum temporal granularity to use -- based on strftime, use 'f' for most rez
                         
                         color_by             = None,  # just the default color or a string for a field
                         global_color_order   = None,  # color by ordering... if none (default), will be created and filled in...
                         
                         count_by             = None,  # none means just count rows, otherwise, use a field to sum by
                         count_by_set         = False, # count by using a set operation
                         
                         widget_id            = None,  # naming the svg elements

                         ignore_unintuitive   = True,  # ignore timeframes that are unintuative
                                                       # ... these unintuitive timeframes make the transition points smoother...
                         
                         # --------------------------- # global rendering params
                         
                         global_max           = None,  # maximum to use for bar heights
                         global_min           = None,  # minimum for boxplot style(s)                         
                         just_calc_max        = False, # forces return of the maximum for this render config ...
                                                       # ... which will then be used for the global max across bar charts...
                        
                         # --------------------------- # style

                         style                 = 'barchart', # 'barchart' or 'boxplot' or 'boxplot_w_swarm'
                         cap_swarm_at          = 200,        # cap the swarm plot at the specified number... if set to None, then no caps

                         # ----------------------------------- # secondary axis settings # probably not small multiple safe...
                         df2                     = None,       # secondary axis dataframe ... if not set but y2_field is, then this will be set to df field
                         df2_fade                = 0.1,        # amount to fade the background prior to the line renders / None == no fade
                         x2_field                = None,       # x2 field ... if not set but the y2_field is, then this be set to the ts_field
                         x2_field_is_scalar      = True,       # x2 field is scalar // doesn't make sense for this view... but leaving it in for consistency
                         x2_axis_col             = None,       # x2 axis column name
                         y2_field                = None,       # secondary axis field ... if this is set, then df2 will be set to df // only required field really...
                         y2_field_is_scalar      = True,       # default... logic will check in the method to determine if this is true
                         y2_axis_col             = None,       # y2 axis column name
                         line2_groupby_field     = None,       # secondary line field ... will NOT be set
                         line2_groupby_w         = 1.5,        # secondary line field width
                         line2_groupby_color     = None,       # line2 color... if none, pulls from the color_by field
                         line2_groupby_dasharray = "4 2",      # line2 dasharray
                         dot2_size               = 'medium',   # dot2 size ... 'small', 'medium', 'large', 'vary'

                         # -----------------------     # small multiple options

                         sm_type               = None, # should be the method name // similar to the smallMultiples method
                         sm_w                  = None, # override the width of the small multiple
                         sm_h                  = None, # override the height of the small multiple
                         sm_params             = {},   # dictionary of parameters for the small multiples
                         sm_x_axis_independent = True, # Use independent axis for x (xy, temporal, and linkNode)
                         sm_y_axis_independent = True, # Use independent axis for y (xy, temporal, periodic, pie)

                         # --------------------------- # rendering specific params
                         
                         x_view               = 0,     # x offset for the view
                         y_view               = 0,     # y offset for the view
                         w                    = 512,   # width of the view
                         h                    = 128,   # height of the view

                         h_gap                = 0,     # gap between bars.. should be a zero or a one...
                         min_bar_w            = 3,     # minimum bar width
                         txt_h                = 14,    # text height for the labels
                         x_ins                = 3,     # x insert (on both sides of the drawing)
                         y_ins                = 3,
                         background_opacity   = 1.0,   # background opacity
                         draw_labels          = True,  # draw labels flag
                         draw_border          = True,  # draw a border around the bar chart
                         draw_context         = True,  # draw background hints about the years, months, days, etc.
                         draw_perf_stats      = False):# draw performance stats
        rt_temporal_barchart = self.RTTemporalBarChart(self, df, ts_field=ts_field, ts_min=ts_min, ts_max=ts_max,
                                                       temporal_granularity=temporal_granularity, color_by=color_by, global_color_order=global_color_order,
                                                       count_by=count_by, count_by_set=count_by_set, widget_id=widget_id, ignore_unintuitive=ignore_unintuitive,
                                                       global_max=global_max, global_min=global_min, style=style, cap_swarm_at=cap_swarm_at, 
                                                       df2=df2, df2_fade=df2_fade, x2_field=x2_field, x2_field_is_scalar=x2_field_is_scalar, x2_axis_col=x2_axis_col, 
                                                       y2_field=y2_field, y2_field_is_scalar=y2_field_is_scalar, y2_axis_col=y2_axis_col,
                                                       line2_groupby_field=line2_groupby_field, line2_groupby_w=line2_groupby_w,
                                                       line2_groupby_color=line2_groupby_color, line2_groupby_dasharray=line2_groupby_dasharray,
                                                       dot2_size=dot2_size,
                                                       sm_type=sm_type, sm_w=sm_w,
                                                       sm_h=sm_h, sm_params=sm_params, sm_x_axis_independent=sm_x_axis_independent, sm_y_axis_independent=sm_y_axis_independent,
                                                       x_view=x_view, y_view=y_view, w=w, h=h, h_gap=h_gap, min_bar_w=min_bar_w, txt_h=txt_h, x_ins=x_ins, y_ins=y_ins,
                                                       background_opacity=background_opacity,draw_labels=draw_labels, draw_border=draw_border, draw_context=draw_context)
        return rt_temporal_barchart.renderSVG(just_calc_max)

    #
    # Proper Label Based On Time Resolution
    # time_rezes     = ['100YS',     '10YS',    'YS',    'QS',       'MS',     '1W',    'D',    '4H',      'H',     '15MIN',   '1MIN',     '15S',     'S']
    #
    def relevantTimeLabel(self, dt, time_rez_i):
        return dt.strftime(self.time_rezes_fmt[time_rez_i])

    #
    # temporalBarChartInstance() - return an instance of RTTemporalBarChart
    #
    def temporalBarChartInstance(self,
                                 df,                           # dataframe to render
                                 # --------------------------- # everything else is a default... 
                                 ts_field             = None,  # timestamp field // needs to be a np.datetime64 column...
                                 ts_min               = None,  # Render ranges... if they need to be overwritten
                                 ts_max               = None,  # Render ranges... if they need to be overwritten
                                 temporal_granularity = None,  # minimum temporal granularity to use -- based on strftime, use 'f' for most rez
                                 color_by             = None,  # just the default color or a string for a field
                                 global_color_order   = None,  # color by ordering... if none (default), will be created and filled in...
                                 count_by             = None,  # none means just count rows, otherwise, use a field to sum by
                                 count_by_set         = False, # count by using a set operation
                                 widget_id            = None,  # naming the svg elements
                                 ignore_unintuitive   = True,  # ignore timeframes that are unintuative
                                                               # ... these unintuitive timeframes make the transition points smoother...
                                 # --------------------------- # global rendering params
                                 global_max           = None,  # maximum to use for bar heights   
                                 global_min           = None,  # minimum for boxplot style(s)                                 
                                 # --------------------------- # style
                                 style                 = 'barchart', # 'barchart' or 'boxplot' or 'boxplot_w_swarm'
                                 cap_swarm_at          = 200,        # cap the swarm plot at the specified number... if set to None, then no caps

                                 # ----------------------------------- # secondary axis settings # probably not small multiple safe...
                                 df2                     = None,       # secondary axis dataframe ... if not set but y2_field is, then this will be set to df field
                                 df2_fade                = 0.1,        # amount to fade the background prior to the line renders / None == no fade
                                 x2_field                = None,       # x2 field ... if not set but the y2_field is, then this be set to the ts_field
                                 x2_field_is_scalar      = True,       # x2 field is scalar // doesn't make sense for this view... but leaving it in for consistency
                                 x2_axis_col             = None,       # x2 axis column name
                                 y2_field                = None,       # secondary axis field ... if this is set, then df2 will be set to df // only required field really...
                                 y2_field_is_scalar      = True,       # default... logic will check in the method to determine if this is true
                                 y2_axis_col             = None,       # y2 axis column name
                                 line2_groupby_field     = None,       # secondary line field ... will NOT be set
                                 line2_groupby_w         = 1.5,        # secondary line field width
                                 line2_groupby_color     = None,       # line2 color... if none, pulls from the color_by field
                                 line2_groupby_dasharray = "4 2",      # line2 dasharray
                                 dot2_size               = 'medium',   # dot2 size ... 'small', 'medium', 'large', 'vary'

                                 # -----------------------     # small multiple options
                                 sm_type               = None, # should be the method name // similar to the smallMultiples method
                                 sm_w                  = None, # override the width of the small multiple
                                 sm_h                  = None, # override the height of the small multiple
                                 sm_params             = {},   # dictionary of parameters for the small multiples
                                 sm_x_axis_independent = True, # Use independent axis for x (xy, temporal, and linkNode)
                                 sm_y_axis_independent = True, # Use independent axis for y (xy, temporal, periodic, pie)
                                 # --------------------------- # rendering specific params
                                 x_view               = 0,     # x offset for the view
                                 y_view               = 0,     # y offset for the view
                                 w                    = 512,   # width of the view
                                 h                    = 128,   # height of the view
                                 h_gap                = 0,     # gap between bars.. should be a zero or a one...
                                 min_bar_w            = 3,     # minimum bar width
                                 txt_h                = 14,    # text height for the labels
                                 x_ins                = 3,     # x insert (on both sides of the drawing)
                                 y_ins                = 3,
                                 background_opacity   = 1.0,   # background opacity
                                 draw_labels          = True,  # draw labels flag
                                 draw_border          = True,  # draw a border around the bar chart
                                 draw_context         = True): # draw background hints about the years, months, days, etc.
        return self.RTTemporalBarChart(self, df, ts_field=ts_field, ts_min=ts_min, ts_max=ts_max,
                                       temporal_granularity=temporal_granularity, color_by=color_by, global_color_order=global_color_order,
                                       count_by=count_by, count_by_set=count_by_set, widget_id=widget_id, ignore_unintuitive=ignore_unintuitive,
                                       global_max=global_max, global_min=global_min, style=style, cap_swarm_at=cap_swarm_at, 
                                       df2=df2, df2_fade=df2_fade, x2_field=x2_field, x2_field_is_scalar=x2_field_is_scalar, x2_axis_col=x2_axis_col, 
                                       y2_field=y2_field, y2_field_is_scalar=y2_field_is_scalar, y2_axis_col=y2_axis_col,
                                       line2_groupby_field=line2_groupby_field, line2_groupby_w=line2_groupby_w,
                                       line2_groupby_color=line2_groupby_color, line2_groupby_dasharray=line2_groupby_dasharray,
                                       dot2_size=dot2_size,
                                       sm_type=sm_type, sm_w=sm_w,
                                       sm_h=sm_h, sm_params=sm_params, sm_x_axis_independent=sm_x_axis_independent, sm_y_axis_independent=sm_y_axis_independent,
                                       x_view=x_view, y_view=y_view, w=w, h=h, h_gap=h_gap, min_bar_w=min_bar_w, txt_h=txt_h, x_ins=x_ins, y_ins=y_ins,
                                       background_opacity=background_opacity, draw_labels=draw_labels, draw_border=draw_border, draw_context=draw_context)

    #
    # RTTemporalBarChart Class
    #
    class RTTemporalBarChart(object):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     df,                           # dataframe to render
                     # --------------------------- # everything else is a default... 
                     ts_field             = None,  # timestamp field // needs to be a np.datetime64 column...
                     ts_min               = None,  # Render ranges... if they need to be overwritten
                     ts_max               = None,  # Render ranges... if they need to be overwritten
                     temporal_granularity = None,  # minimum temporal granularity to use -- based on strftime, use 'f' for most rez
                     color_by             = None,  # just the default color or a string for a field
                     global_color_order   = None,  # color by ordering... if none (default), will be created and filled in...
                     count_by             = None,  # none means just count rows, otherwise, use a field to sum by
                     count_by_set         = False, # count by using a set operation
                     widget_id            = None,  # naming the svg elements
                     ignore_unintuitive   = True,  # ignore timeframes that are unintuative
                                                   # ... these unintuitive timeframes make the transition points smoother...
                     # --------------------------- # global rendering params
                     global_max           = None,  # maximum to use for bar heights   
                     global_min           = None,  # minimum for boxplot style(s)
                     # --------------------------- # style
                     style                 = 'barchart', # 'barchart' or 'boxplot' or 'boxplot_w_swarm'
                     cap_swarm_at          = 200,        # cap the swarm plot at the specified number... if set to None, then no caps

                     # ----------------------------------- # secondary axis settings # probably not small multiple safe...
                     df2                     = None,       # secondary axis dataframe ... if not set but y2_field is, then this will be set to df field
                     df2_fade                = 0.1,        # amount to fade the background prior to the line renders / None == no fade
                     x2_field                = None,       # x2 field ... if not set but the y2_field is, then this be set to the ts_field
                     x2_field_is_scalar      = True,       # x2 field is scalar // doesn't make sense for this view... but leaving it in for consistency
                     x2_axis_col             = None,       # x2 axis column name
                     y2_field                = None,       # secondary axis field ... if this is set, then df2 will be set to df // only required field really...
                     y2_field_is_scalar      = True,       # default... logic will check in the method to determine if this is true
                     y2_axis_col             = None,       # y2 axis column name
                     line2_groupby_field     = None,       # secondary line field ... will NOT be set
                     line2_groupby_w         = 1.5,        # secondary line field width
                     line2_groupby_color     = None,       # line2 color... if none, pulls from the color_by field
                     line2_groupby_dasharray = "4 2",      # line2 dasharray
                     dot2_size               = 'medium',   # dot2 size ... 'small', 'medium', 'large', 'vary'

                     # -----------------------     # small multiple options
                     sm_type               = None, # should be the method name // similar to the smallMultiples method
                     sm_w                  = None, # override the width of the small multiple
                     sm_h                  = None, # override the height of the small multiple
                     sm_params             = {},   # dictionary of parameters for the small multiples
                     sm_x_axis_independent = True, # Use independent axis for x (xy, temporal, and linkNode)
                     sm_y_axis_independent = True, # Use independent axis for y (xy, temporal, periodic, pie)
                     # --------------------------- # rendering specific params
                     x_view               = 0,     # x offset for the view
                     y_view               = 0,     # y offset for the view
                     w                    = 512,   # width of the view
                     h                    = 128,   # height of the view
                     h_gap                = 0,     # gap between bars.. should be a zero or a one...
                     min_bar_w            = 3,     # minimum bar width
                     txt_h                = 14,    # text height for the labels
                     x_ins                = 3,     # x insert (on both sides of the drawing)
                     y_ins                = 3,
                     background_opacity   = 1.0,   # background opacity
                     draw_labels          = True,  # draw labels flag
                     draw_border          = True,  # draw a border around the bar chart
                     draw_context         = True): # draw background hints about the years, months, days, etc.
            self.parms     = locals().copy()
            self.rt_self   = rt_self
            self.df        = df.copy()
            self.widget_id = widget_id

            # Make a widget_id if it's not set already
            if self.widget_id is None:
                self.widget_id = "temporalbarchart_" + str(random.randint(0,65535))

            self.ts_field              = ts_field
            self.ts_min                = ts_min
            self.ts_max                = ts_max
            self.temporal_granularity  = temporal_granularity
            self.color_by              = color_by
            self.global_color_order    = global_color_order
            self.count_by              = count_by
            self.count_by_set          = count_by_set
            self.ignore_unintuitive    = ignore_unintuitive
            self.global_max            = global_max
            self.global_min            = global_min
            self.style                 = style
            self.cap_swarm_at          = cap_swarm_at

            self.df2                        = df2
            self.df2_fade                   = df2_fade
            self.x2_field                   = x2_field
            self.x2_field_is_scalar         = x2_field_is_scalar
            self.x2_axis_col                = x2_axis_col
            self.y2_field                   = y2_field
            self.y2_field_is_scalar         = y2_field_is_scalar
            self.y2_axis_col                = y2_axis_col
            self.line2_groupby_field        = line2_groupby_field
            self.line2_groupby_w            = line2_groupby_w
            self.line2_groupby_color        = line2_groupby_color
            self.line2_groupby_dasharray    = line2_groupby_dasharray
            self.dot2_size                  = dot2_size

            self.sm_type               = sm_type
            self.sm_w                  = sm_w
            self.sm_h                  = sm_h
            self.sm_params             = sm_params.copy()
            self.sm_x_axis_independent = sm_x_axis_independent
            self.sm_y_axis_independent = sm_y_axis_independent
            self.x_view                = x_view
            self.y_view                = y_view
            self.w                     = w
            self.h                     = h
            self.h_gap                 = h_gap
            self.min_bar_w             = min_bar_w
            self.txt_h                 = txt_h
            self.x_ins                 = x_ins
            self.y_ins                 = y_ins
            self.background_opacity    = background_opacity,
            self.draw_labels           = draw_labels
            self.draw_border           = draw_border
            self.draw_context          = draw_context

            # Class members thatt are filled in upon render
            self.ts_to_x               = {}    # For calculating timestamp positions
            self.ts_sort               = None  # Sort for the keys in the ts_to_x // only created upon first request

            # Determine the timestamp field
            if self.ts_field is None:
                choices = self.df.select_dtypes(np.datetime64).columns
                if len(choices) == 1:
                    self.ts_field = choices[0]
                elif len(choices) > 1:
                    print('multiple timestamp fields... choosing the first (RTTemporalBarChart)')
                    self.ts_field = choices[0]
                else:
                    raise Exception('no timestamp field supplied to RTTemporalBarChart(), cannot automatically determine field')

            # Figure out the y2_field settings
            if self.y2_field is not None:
                if self.df2 is None:
                    self.df2 = self.df
                    self.df2_is_df = True
                else:
                    self.df2 = self.df2.copy()
                    self.df2_is_df = False

                if self.x2_field is None:
                    if self.df2_is_df:
                        self.x2_field = self.ts_field
                    else:
                        choices = self.df2.select_dtypes(np.datetime64).columns
                        if len(choices) == 1:
                            self.x2_field = choices[0]
                        elif len(choices) > 1:
                            print('multiple timestamp fields [df2]... choosing the first (RTTemporalBarChart)')
                            self.x2_field = choices[0]
                        else:
                            raise Exception('no timestamp field supplied to RTTemporalBarChart() for df2, cannot automatically determine field')

                if type(self.y2_field) != list:
                    self.y2_field = [self.y2_field]
                
                if self.y2_field_is_scalar:
                    if len(self.y2_field) == 1:
                        self.y2_field_is_scalar = self.rt_self.fieldIsArithmetic(self.df2, self.y2_field[0])
                    else:
                        self.y2_field_is_scalar = False

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

            if self.y2_field is not None and rt_self.isTField(self.y2_field):
                self.df2,self.y2_field = rt_self.applyTransform(self.df2, self.y2_field)

        #
        # renderSVG() - create the SVG
        #
        def renderSVG(self,just_calc_max=False):
            # Determine the color order (for each bar) // Copied in whole from histogram method
            if self.global_color_order is None:
                self.global_color_order = self.rt_self.colorRenderOrder(self.df, self.color_by, self.count_by, self.count_by_set)

            # Limited range of pixels are allowed
            if   self.h_gap > 4:
                 self.h_gap = 4
            elif self.h_gap < 0:
                 self.h_gap = 0
        
            # Determine the mins and maxes and ensure it's a datetime64
            if self.ts_min is None:
                self.ts_min = self.df[self.ts_field].min()
            if self.ts_max is None:
                self.ts_max = self.df[self.ts_field].max()        
            if type(self.ts_min) != np.datetime64:
                self.ts_min = np.datetime64(self.ts_min)
            if type(self.ts_max) != np.datetime64:
                self.ts_max = np.datetime64(self.ts_max)
        
            # If the height/width are less than the minimums, turn off labeling... and make the min_bar_w = 1
            # ... for this component as a small multiples
            params_orig_minus_self = self.parms.copy()
            params_orig_minus_self.pop('self')
            min_dims = self.rt_self.temporalBarChartSmallMultipleDimensions(**params_orig_minus_self)
            if self.w < min_dims[0] or self.h < min_dims[1]:
                self.draw_labels  = False
                self.draw_context = False
                self.min_bar_w    = 1
                self.x_ins        = 1
                self.y_ins        = 1
                self.h_gap        = 0

            # Calculate the usable width
            w_usable = self.w - 2*self.x_ins
            x_left   = self.x_ins
            if self.draw_labels:
                x_left    = 2*self.y_ins + self.txt_h
                w_usable  = self.w - (3*self.y_ins + self.txt_h)
                if self.y2_field is not None:
                    w_usable  = self.w - (3*self.y_ins + 2*self.txt_h)

            # Determine the temporal granularity of the data ... should preclude finer resolution renders...
            if self.temporal_granularity is None:
                self.temporal_granularity = self.rt_self.temporalGranularity(self.df, self.ts_field) # Too expensive as written...

            # Adjust the min_bar_w if this is small multiples are to be included
            if self.sm_type is not None:
                if self.sm_w is None or self.sm_h is None:
                    self.sm_w,self.sm_h = getattr(self.rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
                self.min_bar_w = self.sm_w

            # Determine the render resolution
            tmp_df  = pd.DataFrame({'timefield':[self.ts_min,self.ts_max]})
            tmp_df['timefield'] = np.array(tmp_df['timefield'],dtype=np.datetime64)
            time_rez_i     = 0
            for i in range(0,len(self.rt_self.time_rezes)):
                # Disable some resolutions... if ignore unintuitive is set...
                if self.ignore_unintuitive and (self.rt_self.time_rezes[i] == '1W'    or \
                                                self.rt_self.time_rezes[i] == '4H'    or \
                                                self.rt_self.time_rezes[i] == '15MIN' or \
                                                self.rt_self.time_rezes[i] == '10MIN' or \
                                                self.rt_self.time_rezes[i] == '5MIN'):
                    continue

                bins  = len(tmp_df.groupby(pd.Grouper(key='timefield',freq=self.rt_self.time_rezes[i])))
                bar_w = ((w_usable - (bins*self.h_gap))/bins)
                if self.rt_self.granularityExceedsResolution(self.temporal_granularity,self.rt_self.time_rezes[i]):
                    break
                elif bar_w >= self.min_bar_w:
                    time_rez_i = i
                elif bar_w <= 1:
                    break
        
            # Finalize the bar width
            groupby = tmp_df.groupby(pd.Grouper(key='timefield',freq=self.rt_self.time_rezes[time_rez_i]))
            bins    = len(groupby)
            bar_w   = ((w_usable - (bins*self.h_gap))/bins)

            # Create the lookup to the x index per key in the groupby
            xi_lu,xi = {},0
            for k,k_df in groupby:
                xi_lu[k] = xi
                xi += 1

            # Height geometry
            if self.sm_type is None:
                if self.draw_labels:
                    max_bar_h  = self.h - 2*self.y_ins - self.txt_h - 2
                    y_baseline = self.h -   self.y_ins - self.txt_h - 1
                else:
                    max_bar_h  = self.h - 2*self.y_ins
                    y_baseline = self.h -   self.y_ins - 1
            else:
                # re-adjust the small multiple dimensions based on the bar width
                sm_prop    = self.sm_h/self.sm_w
                self.sm_w  = bar_w
                self.sm_h  = bar_w * sm_prop

                # prevent the small multiple from exceeding a quarter of the height
                if self.sm_h > self.h/4:
                    self.sm_h = self.h/4
                    self.sm_w = self.sm_h/sm_prop

                if self.draw_labels:
                    max_bar_h  = self.h - self.sm_h  - 2*self.y_ins - self.txt_h - 2
                    y_baseline = self.h              -   self.y_ins - self.txt_h - 1
                    sm_cy      = y_baseline - max_bar_h - self.sm_h/2 # small multiple center y
                else:
                    max_bar_h  = self.h - self.sm_h  - 2*self.y_ins
                    y_baseline = self.h              -   self.y_ins - 1
                    sm_cy      = y_baseline - max_bar_h - self.sm_h/2 # small multiple center y

            # General adjustment for the small multiple height
            adj_sm_h = 0
            if self.sm_type is not None:
                adj_sm_h = self.sm_h

            # Start the SVG Frame
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" fill-opacity="{self.background_opacity}" stroke="{background_color}" stroke-opacity="{self.background_opacity}" />'
            
            # Draw the background for the temporal chart
            axis_color = self.rt_self.co_mgr.getTVColor('axis','default')
            textfg     = self.rt_self.co_mgr.getTVColor('label','defaultfg')

            # Draw temporal context
            if self.draw_context:
                svg += self.rt_self.drawTemporalContext(self.rt_self.time_rezes[time_rez_i], 
                                                        tmp_df.groupby(pd.Grouper(key='timefield',freq=self.rt_self.time_rezes[time_rez_i])), 
                                                        bar_w, self.h_gap, x_left, self.y_ins, w_usable, y_baseline-self.y_ins,self.txt_h-2)
            svg += f'<line x1="{x_left}" y1="{y_baseline+1}" x2="{x_left}"            y2="{self.y_ins + adj_sm_h}" stroke="{axis_color}" stroke-width="1" />'
            svg += f'<line x1="{x_left}" y1="{y_baseline+1}" x2="{x_left + w_usable}" y2="{y_baseline+1}"          stroke="{axis_color}" stroke-width="1" />'

            # Group and determine the maximum
            group_by_max,group_by_min = self.global_max,self.global_min
            if self.count_by is None or self.count_by_set == False:
                order = self.df.groupby(pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]))
                if self.global_max is None:
                    # Count by Rows
                    if self.count_by is None:
                        group_by_min = 0
                        group_by_max = order.size().max()
                    # Boxplot
                    elif self.style.startswith('boxplot'):
                        group_by_min = self.df[self.count_by].min()                        
                        group_by_max = self.df[self.count_by].max()                        
                    # Numeric summation
                    else:
                        group_by_min = 0
                        group_by_max = order[self.count_by].sum().max()

            elif self.count_by_set:
                _df         = self.df.groupby([pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]),self.count_by]).size().reset_index()
                _df_for_max = _df.groupby(pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]))
                if self.global_max is None:
                    group_by_min = 0
                    group_by_max = _df_for_max.size().max()
                order = self.df.groupby(pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]))

            else:
                raise Exception(f'RTTemporalBarChart -- unknown count_by state "{self.count_by}" / by_set = "{self.count_by_set}"')

            # For just calculating max....
            if just_calc_max:
                return group_by_min,group_by_max

            # Iterate over the order and render each bar
            if self.style == 'barchart':    
                for k,k_df in order:
                    if   self.count_by is None:
                        px = max_bar_h * len(k_df) / group_by_max
                    elif self.count_by_set:
                        px = max_bar_h * len(set(k_df[self.count_by])) / group_by_max
                    else:
                        px = max_bar_h * k_df[self.count_by].sum() / group_by_max
                    
                    x = x_left + 1 + xi_lu[k] * (bar_w + self.h_gap)
                    svg += self.rt_self.colorizeBar(k_df, self.global_color_order, self.color_by, self.count_by, self.count_by_set, x, y_baseline, px, bar_w, False)
                    self.ts_to_x[k] = (x,bar_w)

            elif self.style.startswith('boxplot'):
                # Adjust the bar width so that it's not excessively long
                if bar_w > 16:
                    _bar_w = 16
                else:
                    _bar_w = bar_w

                # Make a y-transform lambda
                if group_by_max == 0:
                    group_by_max = 1
                yT = lambda __y__: (y_baseline - max_bar_h * (__y__ - group_by_min) / (group_by_max - group_by_min))

                # Render the boxplot columns
                for k,k_df in order:
                    x = x_left + 1 + xi_lu[k] * (bar_w + self.h_gap)
                    _cx = x + bar_w/2
                    svg += self.rt_self.renderBoxPlotColumn(self.style, k_df, _cx, yT, group_by_max, group_by_min, _bar_w, self.count_by, self.color_by, self.cap_swarm_at)
                    self.ts_to_x[k] = (x,bar_w)

            # Handle the small multiple renders
            if self.sm_type is not None:
                group_by = self.df.groupby(pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]))

                node_to_xy  = {}
                node_to_dfs = {}

                for key,key_df in group_by:
                    x = x_left + 1 + xi_lu[key] * (bar_w + self.h_gap)
                    if len(key_df) != 0:
                        node_to_xy[key]  = [x + bar_w/2, sm_cy]
                        node_to_dfs[key] = key_df
                
                sm_lu = self.rt_self.createSmallMultiples(self.df, node_to_dfs, node_to_xy, self.count_by, self.count_by_set, self.color_by, self.ts_field, self.widget_id,
                                                          self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent, self.sm_w, self.sm_h)

                for node_str in sm_lu.keys():
                    svg += sm_lu[node_str]
            
            # ***********************************************************************************************************************************************************
            # ***********************************************************************************************************************************************************
            # ***********************************************************************************************************************************************************

            # Handle the xy overlay
            self.df2_extends_beyond = False
            if self.y2_field is not None:
                # Apply the fade (if set)
                if self.df2_fade is not None:
                    _co  =  self.rt_self.co_mgr.getTVColor('background', 'default')
                    svg  += f'<rect x="{x_left}" y="{y_baseline-max_bar_h}" width="{w_usable}" height="{max_bar_h}" fill="{_co}" fill-opacity="{self.df2_fade}" />'

                # Find the end timestamp
                _df   = pd.DataFrame({self.ts_field:[self.ts_min, self.ts_max]}) # tricky because the param extents may be more than the df ones...
                _gb   = _df.groupby(pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]))
                _bins = len(_gb) 
                _ts0,_ts1 = None,None
                for k,k_df in _gb:
                    if _ts0 is None:
                        _ts0 = k
                        break
                for _ts in pd.date_range(start=_ts0, periods=_bins+1, freq=self.rt_self.time_rezes[time_rez_i]):
                    _ts1 = _ts
                # Determine if any of the frame falls outside of this range / clamp it if so...
                if self.df2[self.x2_field].min() < _ts0 or self.df2[self.x2_field].max() > _ts1:
                    self.df2_extends_beyond = True
                    self.df2 = self.df2.query(f"`{self.x2_field}` >= @_ts0 and `{self.x2_field}` <= @_ts1")
                # Scale the x coordinates
                if self.x2_axis_col is None:
                    self.x2_axis_col = f'my_x2_{self.widget_id}'
                    self.df2[self.x2_axis_col] = (self.df2[self.x2_field] - _ts0)/(_ts1 - _ts0)
                # Scale the y coordinates
                if self.y2_axis_col is None:
                    self.y2_axis_col = f'my_y2_{self.widget_id}'
                    if self.y2_field_is_scalar:
                        _min,_max = self.df2[self.y2_field].min().iloc[0],self.df2[self.y2_field].max().iloc[0]
                        print(_min)
                        print(_max)
                        if _min == _max:
                            _min -= 0.5
                            _max += 0.5
                        self.df2[self.y2_axis_col] = (self.df2[self.y2_field] - _min)/(_max - _min)
                    else:
                        raise Exception('temporalBarChart() - y2_field in non-scalar modes not implemented yet')
                # Create the pixels columns
                self.df2[self.x2_axis_col+'_px'] = x_left     + self.df2[self.x2_axis_col]*w_usable
                self.df2[self.y2_axis_col+'_px'] = y_baseline - self.df2[self.y2_axis_col]*max_bar_h
                # Pixelize it...
                self.df2[self.x2_axis_col+'_px'] = self.df2[self.x2_axis_col+'_px'].astype(np.int32)
                self.df2[self.y2_axis_col+'_px'] = self.df2[self.y2_axis_col+'_px'].astype(np.int32)
                # Draw the lines (if configured)
                if self.line2_groupby_field is not None:
                    _gb = self.df2.groupby(by=self.line2_groupby_field)
                    for k,k_df in _gb:
                        _points,gbxy = '',k_df.groupby([self.x2_axis_col+'_px',self.y2_axis_col+'_px'])
                        for xy,xy_df in gbxy:
                            _points += f'{xy[0]},{xy[1]} '
                        _co = self.rt_self.co_mgr.getTVColor('data','default')

                        if self.line2_groupby_color is None and self.color_by is not None and self.color_by in k_df.columns:
                            _set = set(k_df[self.color_by])
                            if len(_set) == 1:
                                _co = self.rt_self.co_mgr.getColor(_set.pop())
                        elif self.line2_groupby_color is not None and self.line2_groupby_color in k_df.columns:
                            _set = set(k_df[self.line2_groupby_color])
                            if len(_set) == 1:
                                _co = self.rt_self.co_mgr.getColor(_set.pop())
                        elif self.line2_groupby_color is not None and len(self.line2_groupby_color) == 7 and self.line2_groupby_color[0] == '#':
                            _co = self.line2_groupby_color

                        if self.line2_groupby_dasharray:
                            svg += f'<polyline points="{_points}" fill-opacity="0.0" fill="None" stroke="{_co}" stroke-width="{self.line2_groupby_w}" stroke-dasharray="{self.line2_groupby_dasharray}" />'
                        else:
                            svg += f'<polyline points="{_points}" fill-opacity="0.0" fill="None" stroke="{_co}" stroke-width="{self.line2_groupby_w}" />'

                # Draw the points
                if self.dot2_size is not None and self.dot2_size != 'hidden':
                    _dot_r = 2
                    if   self.dot2_size == 'small':
                        _dot_r = 1
                    elif self.dot2_size == 'large':
                        _dot_r = 3
                    _co = self.rt_self.co_mgr.getTVColor('data','default')
                    _gb = self.df2.groupby(by=[self.x2_axis_col+'_px', self.y2_axis_col+'_px'])
                    for k,k_df in _gb:
                        _x  = k[0]
                        _y  = k[1]

                        # Exact copy of above... should be re-factored... <Starting Here>
                        if self.line2_groupby_color is None and self.color_by is not None and self.color_by in k_df.columns:
                            _set = set(k_df[self.color_by])
                            if len(_set) == 1:
                                _co = self.rt_self.co_mgr.getColor(_set.pop())
                        elif self.line2_groupby_color is not None and self.line2_groupby_color in k_df.columns:
                            _set = set(k_df[self.line2_groupby_color])
                            if len(_set) == 1:
                                _co = self.rt_self.co_mgr.getColor(_set.pop())
                        elif self.line2_groupby_color is not None and len(self.line2_groupby_color) == 7 and self.line2_groupby_color[0] == '#':
                            _co = self.line2_groupby_color
                        # <Ending Here>

                        svg += f'<circle cx="{_x}" cy="{_y}" r="{_dot_r}" fill="{_co}"/>'

            # ***********************************************************************************************************************************************************
            # ***********************************************************************************************************************************************************
            # ***********************************************************************************************************************************************************

            # Draw the labels // mirrors the rt_periodic_barchart_mixin codeblock
            if self.draw_labels:
                svg += self.rt_self.svgText(self.rt_self.relevantTimeLabel(groupby.size().index[0], time_rez_i),              x_left,              self.h-3, self.txt_h)
                svg += self.rt_self.svgText(self.rt_self.relevantTimeLabel(groupby.size().index[len(groupby)-1], time_rez_i), x_left + w_usable,   self.h-3, self.txt_h, anchor='end')
                svg += self.rt_self.svgText(self.rt_self.time_rezes_str[time_rez_i],                                          x_left + w_usable/2, self.h-3, self.txt_h, anchor='middle')

                # Max Label
                _str_max,_str_min = f'{group_by_max:{self.rt_self.fformat}}',''
                if re.match('.*\.0*',_str_max):
                    _str_max = _str_max[:_str_max.index('.')]
                svg += self.rt_self.svgText(_str_max,     self.x_ins+self.txt_h, self.y_ins + adj_sm_h,             self.txt_h, anchor='end',    rotation=-90)

                # Min Label (for boxplot only)
                if self.style.startswith('boxplot'):
                    _str_min = f'{group_by_min:{self.rt_self.fformat}}'
                    if re.match('.*\.0*',_str_min):
                        _str_min = _str_min[:_str_min.index('.')]
                    svg += self.rt_self.svgText(_str_min, self.x_ins+self.txt_h, self.y_ins + adj_sm_h + max_bar_h, self.txt_h, anchor='start',  rotation=-90)

                # Count By Label
                _count_by_str = 'rows'
                if self.count_by:
                    _count_by_str = self.count_by
                if self.rt_self.textLength(_str_max, self.txt_h) + self.rt_self.textLength(_str_min, self.txt_h) + self.rt_self.textLength(_count_by_str, self.txt_h) + 10 < max_bar_h:
                    if self.style.startswith('boxplot'):
                        _mid_y = self.rt_self.textLength(_str_max, self.txt_h) + (max_bar_h - self.rt_self.textLength(_str_max, self.txt_h) - self.rt_self.textLength(_str_min, self.txt_h))/2
                        svg += self.rt_self.svgText(_count_by_str, self.x_ins+self.txt_h, 
                                                    self.y_ins + adj_sm_h + _mid_y, 
                                                    self.txt_h, anchor='middle',  rotation=-90)
                    else:
                        svg += self.rt_self.svgText(_count_by_str, self.x_ins+self.txt_h, self.y_ins + adj_sm_h + max_bar_h,   self.txt_h, anchor='start',  rotation=-90)

            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill-opacity="0.0" stroke="{border_color}" />'
            
            svg += '</svg>'

            return svg
    
        #
        # timestampXCoord() 
        # - calculate the x coordinate for a specific timestamp value
        # - not performant... because the expectation is it will be called infrequently
        # - negative values indicate that the timestamp fell before the earliest or after the latest
        # - ... the magnitude of the negative value is equivalent to the positive position 
        #
        def timestampXCoord(self, 
                            _timestamp):
            if self.ts_sort is None:
                self.ts_sort = sorted(list(self.ts_to_x.keys()))
            
            ts_dt = pd.to_datetime(_timestamp)
            if ts_dt < self.ts_sort[0]:
                return -(self.ts_to_x[self.ts_sort[0]][0])

            for i in range(0,len(self.ts_sort)-1):
                if ts_dt >= self.ts_sort[i] and ts_dt < self.ts_sort[i+1]:
                    _ratio = (ts_dt - self.ts_sort[i])/(self.ts_sort[i+1] - self.ts_sort[i])
                    _k_ts  = self.ts_sort[i]
                    return self.ts_to_x[_k_ts][0] + _ratio * self.ts_to_x[_k_ts][1]

            i = len(self.ts_sort)
            if ts_dt < self.ts_sort[i-1] + (self.ts_sort[i-1] - self.ts_sort[i-2]):
                _ratio = (ts_dt - self.ts_sort[i-1])/(self.ts_sort[i-1] - self.ts_sort[i-2]) 
                _k_ts  = self.ts_sort[i-1]
                return self.ts_to_x[_k_ts][0] + _ratio * self.ts_to_x[_k_ts][1]

            _k_ts = self.ts_sort[i-1]
            return -(self.ts_to_x[_k_ts][0] + self.ts_to_x[_k_ts][1])

        #
        # timestampExtents()
        # - return the minimum and maximum timestamps as a pandas tuple
        #
        def timestampExtents(self):
            if self.ts_sort is None:
                self.ts_sort = sorted(list(self.ts_to_x.keys()))
            return self.ts_sort[0],self.ts_sort[-1] + (self.ts_sort[-1] - self.ts_sort[-2])

        #
        # smallMultipleFeatureVector()
        # ... feature vector for comparison with other small multiple instances of this class
        # ... pretty much a copy of the render code above...
        #
        def smallMultipleFeatureVector(self):
            # Determine the mins and maxes and ensure it's a datetime64
            if self.ts_min is None:
                self.ts_min = self.df[self.ts_field].min()
            if self.ts_max is None:
                self.ts_max = self.df[self.ts_field].max()        
            if type(self.ts_min) != np.datetime64:
                self.ts_min = np.datetime64(self.ts_min)
            if type(self.ts_max) != np.datetime64:
                self.ts_max = np.datetime64(self.ts_max)
        
            # If the height/width are less than the minimums, turn off labeling... and make the min_bar_w = 1
            # ... for this component as a small multiples
            params_orig_minus_self = self.parms.copy()
            params_orig_minus_self.pop('self')
            min_dims = self.rt_self.temporalBarChartSmallMultipleDimensions(**params_orig_minus_self)
            if self.w < min_dims[0] or self.h < min_dims[1]:
                self.draw_labels  = False
                self.draw_context = False
                self.min_bar_w    = 1
                self.x_ins        = 1
                self.y_ins        = 1
                self.h_gap        = 0

            # Calculate the usable width
            w_usable = self.w - 2*self.x_ins
            x_left   = self.x_ins
            if self.draw_labels:
                x_left    = 2*self.y_ins + self.txt_h
                w_usable  = self.w - (3*self.y_ins + self.txt_h)
        
            # Determine the temporal granularity of the data ... should preclude finer resolution renders...
            if self.temporal_granularity is None:
                self.temporal_granularity = self.rt_self.temporalGranularity(self.df, self.ts_field) # Too expensive as written...

            # Adjust the min_bar_w if this is small multiples are to be included
            if self.sm_type is not None:
                if self.sm_w is None or self.sm_h is None:
                    self.sm_w,self.sm_h = getattr(self.rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
                self.min_bar_w = self.sm_w

            # Determine the render resolution
            tmp_df  = pd.DataFrame({'timefield':[self.ts_min,self.ts_max]})
            tmp_df['timefield'] = np.array(tmp_df['timefield'],dtype=np.datetime64)
            time_rez_i     = 0
            for i in range(0,len(self.rt_self.time_rezes)):
                # Disable some resolutions... if ignore unintuitive is set...
                if self.ignore_unintuitive and (self.rt_self.time_rezes[i] == '1W'    or \
                                                self.rt_self.time_rezes[i] == '4H'    or \
                                                self.rt_self.time_rezes[i] == '15MIN' or \
                                                self.rt_self.time_rezes[i] == '10MIN' or \
                                                self.rt_self.time_rezes[i] == '5MIN'):
                    continue

                bins  = len(tmp_df.groupby(pd.Grouper(key='timefield',freq=self.rt_self.time_rezes[i])))
                bar_w = ((w_usable - (bins*self.h_gap))/bins)
                if self.rt_self.granularityExceedsResolution(self.temporal_granularity,self.rt_self.time_rezes[i]):
                    break
                elif bar_w >= self.min_bar_w:
                    time_rez_i = i
                elif bar_w <= 1:
                    break
        
            # Finalize the bar width
            groupby = tmp_df.groupby(pd.Grouper(key='timefield',freq=self.rt_self.time_rezes[time_rez_i]))
            bins    = len(groupby)
            bar_w   = ((w_usable - (bins*self.h_gap))/bins)
        
            # Height geometry
            if self.sm_type is None:
                if self.draw_labels:
                    max_bar_h  = self.h - 2*self.y_ins - self.txt_h - 2
                    y_baseline = self.h -   self.y_ins - self.txt_h - 1
                else:
                    max_bar_h  = self.h - 2*self.y_ins
                    y_baseline = self.h -   self.y_ins - 1
            else:
                # re-adjust the small multiple dimensions based on the bar width
                sm_prop    = self.sm_h/self.sm_w
                self.sm_w  = bar_w
                self.sm_h  = bar_w * sm_prop

                # prevent the small multiple from exceeding a quarter of the height
                if self.sm_h > self.h/4:
                    self.sm_h = self.h/4
                    self.sm_w = self.sm_h/sm_prop

                if self.draw_labels:
                    max_bar_h  = self.h - self.sm_h  - 2*self.y_ins - self.txt_h - 2
                    y_baseline = self.h              -   self.y_ins - self.txt_h - 1
                    sm_cy      = y_baseline - max_bar_h - self.sm_h/2 # small multiple center y
                else:
                    max_bar_h  = self.h - self.sm_h  - 2*self.y_ins
                    y_baseline = self.h              -   self.y_ins - 1
                    sm_cy      = y_baseline - max_bar_h - self.sm_h/2 # small multiple center y

            # General adjustment for the small multiple height
            adj_sm_h = 0
            if self.sm_type is not None:
                adj_sm_h = self.sm_h

            # Group and determine the maximum
            group_by_max,group_by_min = self.global_max,self.global_min
            if self.count_by is None or self.count_by_set == False:
                order = self.df.groupby(pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]))
                if self.global_max is None:
                    # Count by Rows
                    if self.count_by is None:
                        group_by_min = 0
                        group_by_max = order.size().max()
                    # Boxplot
                    elif self.style.startswith('boxplot'):
                        group_by_min = self.df[self.count_by].min()                        
                        group_by_max = self.df[self.count_by].max()                        
                    # Numeric summation
                    else:
                        group_by_min = 0
                        group_by_max = order[self.count_by].sum().max()

            elif self.count_by_set:
                _df         = self.df.groupby([pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]),self.count_by]).size().reset_index()
                _df_for_max = _df.groupby(pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]))
                if self.global_max is None:
                    group_by_min = 0
                    group_by_max = _df_for_max.size().max()
                order = self.df.groupby(pd.Grouper(key=self.ts_field,freq=self.rt_self.time_rezes[time_rez_i]))

            else:
                raise Exception(f'RTTemporalBarChart -- unknown count_by state "{self.count_by}" / by_set = "{self.count_by_set}"')

            # Iterate over the order and render each bar
            max_bar_h,fv = 1.0,{}
            for k,k_df in order:
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
            