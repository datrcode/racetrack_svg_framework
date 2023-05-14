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

from pandas.tseries.offsets import MonthEnd

from datetime import datetime

__name__ = 'rt_calendarheatmap_mixin'

#
# Calendar Heatmap Mixin
#
class RTCalendarHeatmapMixin(object):

    #
    # calendarHeatmapPreferredSize()
    # - Return the preferred size
    #
    def calendarHeatmapPreferredDimensions(self, **kwargs):
        # print('pref_dim',kwargs)
        return (160,256)

    #
    # calendarHeatmapMinimumSize()
    # - Return the minimum size
    #
    def calendarHeatmapMinimumDimensions(self, **kwargs):
        # print('min_dim',kwargs)
        return (48,64)

    #
    # calendarHeatmapSmallMultipleSize()
    # - Return the minimum size
    #
    def calendarHeatmapSmallMultipleDimensions(self, **kwargs):
        # print('sm_dim', kwargs)
        return (32,32)

    #
    # calendarHeatmapRequiredFields()
    # - Return the required fields for a histogram configuration
    #
    def calendarHeatmapRequiredFields(self, **kwargs):
        columns = set()
        self.identifyColumnsFromParameters('ts_field', kwargs, columns)
        self.identifyColumnsFromParameters('count_by', kwargs, columns)
        return columns

    #
    # calendarHeatmap()
    #
    def calendarHeatmap(self,
                        df,                                 # Dataframe to render
                        # --------------------------------- #
                        ts_field           = None,          # timestamp field -- if None, will be guessed
                        ts_min             = None,          # minimum timestamp -- if None, calculated from df
                        ts_max             = None,          # maximum timestamp -- if None, calculated from df
                        count_by           = None,          # count by field -- if None, count by rows in the df
                        count_by_set       = False,         # use a set operation vs numerical summation
                        color_by           = None,          # not implemented yet
                        color_magnitude    = 'linear',      # 'linear' or 'log'
                        widget_id          = None,          # widget id for embedding in the svg output
                        month_stroke_width = 1.0,           # stroke width for month outlines
                        # --------------------------------- # global options
                        global_max            = None,       # maximum to use for the daily cells
                        global_min            = None,       # minimum value for the daily cells
                        just_calc_max         = False,      # forces return of the maximum for this render config ...
                                                            # ... which will then be used for the global max across bar charts...
                        # --------------------------------- # small multiple options
                        sm_type               = None,       # should be the method name // similar to the smallMultiples method
                        sm_w                  = None,       # override the width of the small multiple
                        sm_h                  = None,       # override the height of the small multiple
                        sm_params             = {},         # dictionary of parameters for the small multiples
                        sm_x_axis_independent = True,       # Use independent axis for x (xy, temporal, and linkNode)
                        sm_y_axis_independent = True,       # Use independent axis for y (xy, temporal, periodic, pie)
                        # --------------------------------- #
                        x_view          = 0,                # coordinates of the svg frame
                        y_view          = 0,
                        w               = None,             # overall width of the view / None -- will calculate best possible
                        h               = None,             # overall height of the view / None -- will calculate best possible
                        txt_h           = 14,               # maximum text height 
                        x_ins           = 3,                # border insets
                        y_ins           = 3,
                        h_gap           = None,             # between each year
                        month_gap       = None,             # gap between months
                        cell_framing    = True,             # frame each day
                        draw_outlines   = True,             # draw month outlines
                        draw_day_labels = False,            # draw day labels
                        draw_labels     = True):            # draw labels
        rt_calendar_heatmap = self.calendarHeatmapInstance(df=df, ts_field=ts_field, ts_min=ts_min, ts_max=ts_max, count_by=count_by, count_by_set=count_by_set,
                                                           color_by=color_by, color_magnitude=color_magnitude, widget_id=widget_id, month_stroke_width=month_stroke_width,
                                                           global_max=global_max, global_min=global_min, sm_type=sm_type, sm_w=sm_w, sm_h=sm_h, sm_params=sm_params,
                                                           sm_x_axis_independent=sm_x_axis_independent, sm_y_axis_independent=sm_y_axis_independent, x_view=x_view, y_view=y_view,
                                                           w=w, h=h, txt_h=txt_h, x_ins=x_ins, y_ins=y_ins, h_gap=h_gap, month_gap=month_gap, cell_framing=cell_framing,
                                                           draw_outlines=draw_outlines, draw_day_labels=draw_day_labels, draw_labels=draw_labels)
        return rt_calendar_heatmap.renderSVG(just_calc_max)

    #
    # calendarHeatmapInstance() - return an instance of an RTCalendarHeatmap
    #
    def calendarHeatmapInstance(self,
                                df,                                 # Dataframe to render
                                # --------------------------------- #
                                ts_field           = None,          # timestamp field -- if None, will be guessed
                                ts_min             = None,          # minimum timestamp -- if None, calculated from df
                                ts_max             = None,          # maximum timestamp -- if None, calculated from df
                                count_by           = None,          # count by field -- if None, count by rows in the df
                                count_by_set       = False,         # use a set operation vs numerical summation
                                color_by           = None,          # not implemented yet
                                color_magnitude    = 'linear',      # 'linear' or 'log'
                                widget_id          = None,          # widget id for embedding in the svg output
                                month_stroke_width = 1.0,           # stroke width for month outlines
                                # --------------------------------- # global options
                                global_max            = None,       # maximum to use for the daily cells
                                global_min            = None,       # minimum value for the daily cells
                                # --------------------------------- # small multiple options
                                sm_type               = None,       # should be the method name // similar to the smallMultiples method
                                sm_w                  = None,       # override the width of the small multiple
                                sm_h                  = None,       # override the height of the small multiple
                                sm_params             = {},         # dictionary of parameters for the small multiples
                                sm_x_axis_independent = True,       # Use independent axis for x (xy, temporal, and linkNode)
                                sm_y_axis_independent = True,       # Use independent axis for y (xy, temporal, periodic, pie)
                                # --------------------------------- #
                                x_view          = 0,                # coordinates of the svg frame
                                y_view          = 0,
                                w               = None,             # overall width of the view / None -- will calculate best possible
                                h               = None,             # overall height of the view / None -- will calculate best possible
                                txt_h           = 14,               # maximum text height 
                                x_ins           = 3,                # border insets
                                y_ins           = 3,
                                h_gap           = None,             # between each year
                                month_gap       = None,             # gap between months
                                cell_framing    = True,             # frame each day
                                draw_outlines   = True,             # draw month outlines
                                draw_day_labels = False,            # draw day labels
                                draw_labels     = True):            # draw labels
        return self.RTCalendarHeatmap(self, df=df, ts_field=ts_field, ts_min=ts_min, ts_max=ts_max, count_by=count_by, count_by_set=count_by_set,
                                      color_by=color_by, color_magnitude=color_magnitude, widget_id=widget_id, month_stroke_width=month_stroke_width,
                                      global_max=global_max, global_min=global_min, sm_type=sm_type, sm_w=sm_w, sm_h=sm_h, sm_params=sm_params,
                                      sm_x_axis_independent=sm_x_axis_independent, sm_y_axis_independent=sm_y_axis_independent, x_view=x_view, y_view=y_view,
                                      w=w, h=h, txt_h=txt_h, x_ins=x_ins, y_ins=y_ins, h_gap=h_gap, month_gap=month_gap, cell_framing=cell_framing,
                                      draw_outlines=draw_outlines, draw_day_labels=draw_day_labels, draw_labels=draw_labels)

    #
    # RTCalendarHeatmap() - Inner Class for Calendar Heatmap
    #
    class RTCalendarHeatmap(object):
        #
        #
        #
        def __init__(self,
                     rt_self,
                     df,                                 # Dataframe to render
                     # --------------------------------- #
                     ts_field           = None,          # timestamp field -- if None, will be guessed
                     ts_min             = None,          # minimum timestamp -- if None, calculated from df
                     ts_max             = None,          # maximum timestamp -- if None, calculated from df
                     count_by           = None,          # count by field -- if None, count by rows in the df
                     count_by_set       = False,         # use a set operation vs numerical summation
                     color_by           = None,          # not implemented yet
                     color_magnitude    = 'linear',      # 'linear' or 'log'
                     widget_id          = None,          # widget id for embedding in the svg output
                     month_stroke_width = 1.0,           # stroke width for month outlines
                     # --------------------------------- # global options
                     global_max            = None,       # maximum to use for the daily cells
                     global_min            = None,       # minimum value for the daily cells
                     just_calc_max         = False,      # forces return of the maximum for this render config ...
                                                         # ... which will then be used for the global max across bar charts...
                     # --------------------------------- # small multiple options
                     sm_type               = None,       # should be the method name // similar to the smallMultiples method
                     sm_w                  = None,       # override the width of the small multiple
                     sm_h                  = None,       # override the height of the small multiple
                     sm_params             = {},         # dictionary of parameters for the small multiples
                     sm_x_axis_independent = True,       # Use independent axis for x (xy, temporal, and linkNode)
                     sm_y_axis_independent = True,       # Use independent axis for y (xy, temporal, periodic, pie)
                     # --------------------------------- #
                     x_view          = 0,                # coordinates of the svg frame
                     y_view          = 0,
                     w               = None,             # overall width of the view / None -- will calculate best possible
                     h               = None,             # overall height of the view / None -- will calculate best possible
                     txt_h           = 14,               # maximum text height 
                     x_ins           = 3,                # border insets
                     y_ins           = 3,
                     h_gap           = None,             # between each year
                     month_gap       = None,             # gap between months
                     cell_framing    = True,             # frame each day
                     draw_outlines   = True,             # draw month outlines
                     draw_day_labels = False,            # draw day labels
                     draw_labels     = True):            # draw labels
            self.parms                  = locals().copy()
            self.rt_self                = rt_self
            self.df                     = df.copy()
            self.ts_field               = ts_field
            self.ts_min                 = ts_min
            self.ts_max                 = ts_max
            self.count_by               = count_by
            self.count_by_set           = count_by_set
            self.color_by               = color_by
            self.color_magnitude        =  color_magnitude
            self.widget_id              =  widget_id

            # Make a widget_id if it's not set already
            if self.widget_id is None:
                self.widget_id = "calendar_heatmap_" + str(random.randint(0,65535))

            self.month_stroke_width     = month_stroke_width
            self.global_max             = global_max
            self.global_min             = global_min
            self.sm_type                = sm_type
            self.sm_w                   = sm_w
            self.sm_h                   = sm_h
            self.sm_params              = sm_params
            self.sm_x_axis_independent  = sm_x_axis_independent
            self.sm_y_axis_independent  = sm_y_axis_independent
            self.x_view                 = x_view
            self.y_view                 = y_view
            self.w                      = w
            self.h                      = h
            self.txt_h                  = txt_h
            self.x_ins                  = x_ins
            self.y_ins                  = y_ins
            self.h_gap                  = h_gap
            self.month_gap              = month_gap
            self.cell_framing           = cell_framing
            self.draw_outlines          = draw_outlines
            self.draw_day_labels        = draw_day_labels
            self.draw_labels            = draw_labels

            # Determine the timestamp field
            if self.ts_field is None:
                choices = self.df.select_dtypes(np.datetime64).columns
                if len(choices) == 1:
                    self.ts_field = choices[0]
                elif len(choices) > 1:
                    print('RTCalendarHeatmap: multiple timestamp fields... choosing the first (RTCalendarHeatmap.__init__)')
                    self.ts_field = choices[0]
                else:
                    raise Exception('no timestamp field supplied to RTCalendarHeatmap(), cannot automatically determine field')

            # Determine the mininum and maximum
            if self.ts_min is None:
                self.ts_min = self.df[self.ts_field].min()
            if self.ts_max is None:
                self.ts_max = self.df[self.ts_field].max()

            # Perform the transforms
            # Apply count-by transofmrs
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Check the count_by column
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)
        
            # Determine the geometry
            self.num_years = 1 + self.ts_max.year - self.ts_min.year
            if self.num_years > 15:
                raise Exception('calendarHeatmap only handles up to fifteen consecutive years')

        #
        # __calculateGeometry__() - calculate the geometry for the render
        #
        def __calculateGeometry__(self):
            _recalc = False

            if self.sm_type is None:
                if self.w is None:
                    self.w = 96 * self.num_years
                if self.h is None:
                    self.h = 512
                if self.month_gap is None:
                    self.month_gap = 1
                if self.h_gap is None:
                    self.h_gap = 1.2*(self.w/self.num_years)/7  # 1.2 days
            
                if (42*self.num_years) > self.w: # year label needs at least 42 pixels... probably should be a function of txt_h...
                    self.draw_labels = False

                if self.draw_labels:
                    self.cell_h = (self.h  - 3*self.y_ins - 1.5*self.txt_h - 11*self.month_gap)/54 # 52 weeks in a year... give it one more for wrap around
                    self.cell_w = ((self.w - 2*self.x_ins - self.txt_h)    - (self.num_years - 1)*self.h_gap)/(7*self.num_years)
                else:
                    self.cell_h = (self.h  - 2*self.y_ins                  - 11*self.month_gap)/54 # 52 weeks in a year... give it one more for wrap around
                    self.cell_w = ((self.w - 2*self.x_ins)                 - (self.num_years - 1)*self.h_gap)/(7*self.num_years)
            else:
                if self.sm_w is None or self.sm_h is None:
                    self.sm_w,self.sm_h = getattr(self.rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
                self.cell_w = self.sm_w
                self.cell_h = self.sm_h
                if self.month_gap is None:
                    self.month_gap = 0
                if self.h_gap is None:
                    self.h_gap = self.sm_w*0.5
                if self.w is None or self.h is None:
                    _recalc = True

            # If we fall under the amount of space needed, recalculation the bounds
            # ... this also occurs with the small multiples...
            if self.cell_w < 2:
                self.x_ins       = 1
                self.cell_w      = 2
                self.h_gap       = 2
                self.draw_outlines = self.draw_labels = self.cell_framing = False
                _recalc     = True

            if self.cell_h < 2:
                self.y_ins       = 1
                self.cell_h      = 2
                self.month_gap   = 0
                self.draw_outlines = self.draw_labels = self.cell_framing = False
                _recalc     = True
            
            if _recalc: # asumes no labels... all the recalc paths turned off labels
                if self.draw_labels:
                    self.w = 2*self.x_ins +     self.txt_h + 7* self.cell_w*self.num_years + self.h_gap*(self.num_years-1)
                    self.h = 3*self.y_ins + 1.5*self.txt_h + 54*self.cell_h                + 11*self.month_gap
                else:
                    self.w = 2*self.x_ins                  + 7* self.cell_w*self.num_years + self.h_gap*(self.num_years-1)
                    self.h = 2*self.y_ins                  + 54*self.cell_h                + 11*self.month_gap

        #
        # renderSVG() - render the SVG for the view
        #
        def renderSVG(self, just_calc_max):
            # Calculate the geometry
            self.__calculateGeometry__()

            # Start the SVG
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            textfg           = self.rt_self.co_mgr.getTVColor('label',     'defaultfg')
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'

            # render the base views - store a lookup for the timestamp strings
            cell_coords        = {}    # date string to x,y tuple
            month_y0,month_y1  = {},{} # min and max y coords -- e.g., month_y0[1] = 20
            deferred_rendering = []    # svg strings to append at the end of this method
            node_to_xy         = {}    # for small multiple rendering

            major_axis_co      = self.rt_self.co_mgr.getTVColor('axis','major')
            context_default_co = self.rt_self.co_mgr.getTVColor('context','default') 
            for year_i in range(0,self.num_years):
                if self.draw_labels:
                    x_base = self.x_ins + self.txt_h + (7 * self.cell_w + self.h_gap) * year_i
                else:
                    x_base = self.x_ins              + (7 * self.cell_w + self.h_gap) * year_i
                if self.draw_labels:
                    y_base = 2 * self.y_ins + self.txt_h
                else:
                    y_base = self.y_ins 
                year = self.ts_min.year + year_i
                d_range = pd.date_range(str(year)+'-01-01',str(year)+'-12-31',freq='D')

                last_dow = None
                last_mon = None

                outline  = []    # for month outlines

                y = y_base
                for _date in d_range:
                    dow = (_date.day_of_week+1)%7 # Sunday should be index zero...
                    mon = _date.month

                    if last_dow is not None and dow == 0:
                        y += self.cell_h
                    if last_mon is not None and last_mon != mon:
                        y += self.month_gap
                    
                    x = x_base + dow*self.cell_w
                    
                    cell_coords[_date] = (x,y)
                    if self.sm_type is not None:
                        node_to_xy[_date] = (x+self.cell_w/2,y+self.cell_h/2)

                    # Track the min and max y coordinates for a month -- for labeling
                    if _date.day == 1:
                        month_y0[mon] = y+self.cell_h/2
                    month_y1[mon] = y+self.cell_h/2

                    # Outline calculation
                    if self.draw_outlines:
                        if _date.day == 1:
                            if dow != 0:
                                outline.append((x_base,y + self.cell_h))
                                outline.append((x,     y + self.cell_h))
                            outline.append((x,y))
                            outline.append((x_base + 7*self.cell_w,y))
                            end_of_month_date = _date + MonthEnd(0)
                        if _date == end_of_month_date:
                            if dow != 6:
                                outline.append((x_base + 7*self.cell_w,y))
                                outline.append((x+self.cell_w,         y))
                            outline.append((x + self.cell_w,y + self.cell_h))
                            outline.append((x_base,         y + self.cell_h))

                            defer_svg = f'<path d="M {outline[0][0]} {outline[0][1]}'
                            for i in range(1,len(outline)):
                                defer_svg += f' L {outline[i][0]} {outline[i][1]}'
                            defer_svg += f' Z" stroke-width="{self.month_stroke_width}" stroke="{textfg}" fill-opacity="0.0" s/>'
                            deferred_rendering.append(defer_svg)

                            outline = []

                    # Render
                    if self.cell_framing:
                        svg += f'<rect width="{self.cell_w}" height="{self.cell_h}" x="{x}" y="{y}" '
                        svg += f'stroke-width="0.8" stroke="{major_axis_co}" fill="{context_default_co}" />'
                    else:
                        svg += f'<rect width="{self.cell_w}" height="{self.cell_h}" x="{x}" y="{y}" '
                        svg += f'stroke-opacity="0.0" fill="{context_default_co}" />'

                    last_dow,last_mon = dow,mon

            # Group the dataframes appropriately
            if self.sm_type is not None or self.count_by is None:  # By rows
                _gb = self.df.groupby(pd.Grouper(key=self.ts_field,freq='D')).size()
            elif self.count_by_set:    # By set value
                tmp_df = pd.DataFrame(self.df.groupby([pd.Grouper(key=self.ts_field,freq='D'),self.count_by]).size()).reset_index()
                _gb    = tmp_df.groupby(pd.Grouper(key=self.ts_field,freq='D')).size()
            else:                 # By numerical summation
                _gb = self.df.groupby(pd.Grouper(key=self.ts_field,freq='D'))[self.count_by].sum()

            # Find the min and max values
            if self.global_max is None or self.global_min is None:
                group_by_max = _gb.max()
                group_by_min = _gb.max()
                for i in range(len(_gb)): # There's got to be a better way to do this...
                    _value  = _gb[i]
                    if _value > 0 and _value < group_by_min:
                        group_by_min = _value
                    if group_by_max == group_by_min:
                        if group_by_max == 0:
                            group_by_max += 1
                        else:
                            group_by_min -= 1
                if just_calc_max:
                    return group_by_max,group_by_min
            else:
                group_by_max,group_by_min = self.global_max,self.global_min

            #
            # Render cells that have data
            #
            if self.sm_type is None:
                for i in range(len(_gb)):
                    _value  = _gb[i]
                    if _value > 0:
                        _ts_str = _gb.index[i]
                        x,y = cell_coords[_ts_str]
                        _co = self.rt_self.co_mgr.spectrum(_value, group_by_min, group_by_max, self.color_magnitude)
                        if self.cell_framing:
                            svg += f'<rect width="{self.cell_w}" height="{self.cell_h}" x="{x}" y="{y}" '
                            svg += f'stroke-width="0.8" stroke="{major_axis_co}" fill="{_co}" />'
                        else:
                            svg += f'<rect width="{self.cell_w}" height="{self.cell_h}" x="{x}" y="{y}" stroke-opacity="0.0" fill="{_co}" />'

            #
            # Render by small multiples
            #
            else:
                node_to_dfs = {}
                for k,k_df in self.df.groupby(pd.Grouper(key=self.ts_field,freq='D')):
                    node_to_dfs[k] = k_df

                sm_lu = self.rt_self.createSmallMultiples(self.df, node_to_dfs, node_to_xy,
                                                          self.count_by, self.count_by_set, self.color_by, self.ts_field, self.widget_id,
                                                          self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent,
                                                          self.sm_w, self.sm_h)
                for node_str in sm_lu.keys():
                    svg += sm_lu[node_str]

            # Any deferred renderings
            for _svg in deferred_rendering:
                svg += _svg

            # Draw the labels
            if self.draw_labels:
                # Draw the months on the side
                if self.cell_h > 10:
                    _mons = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
                else:
                    _mons = ['J','F','M','A','M','J','J','A','S','O','N','D']
                for _mon_i in range(len(_mons)):
                    x = self.txt_h
                    y = (month_y0[_mon_i+1] + month_y1[_mon_i+1])/2
                    svg += self.rt_self.svgText(str(_mons[_mon_i]), x, y, self.txt_h-2, anchor='middle', rotation=-90)

                # Draw the year at the top and the days of the week at the bottom
                for year_i in range(0,self.num_years):
                    x_base = self.x_ins + self.txt_h + (7 * self.cell_w + self.h_gap) * year_i
                    y_base = 2 * self.y_ins + self.txt_h 
                    year = self.ts_min.year + year_i
                    svg += self.rt_self.svgText(str(year), x_base + 3.5*self.cell_w, y_base-5, self.txt_h, anchor='middle')
                
                    _dows = ['S','M','T','W','T','F','S']
                    for _dow_i in range(len(_dows)):
                        _dow = _dows[_dow_i]
                        svg += self.rt_self.svgText(str(_dow), x_base + _dow_i*self.cell_w + self.cell_w/2, self.h-5, self.txt_h-4, anchor='middle')
                
                if self.draw_day_labels:
                    for _date in cell_coords:
                        x,y = cell_coords[_date]
                        svg += self.rt_self.svgText(str(_date.day), x+2, y+self.txt_h, self.txt_h-4)

            svg += '</svg>'

            return svg
            