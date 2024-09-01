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
import polars as pl
import numpy as np
import random
import re

from shapely.geometry import Polygon

from .rt_component import RTComponent

__name__ = 'rt_boxplot_mixin'

#
# Boxplot Mixin
#
class RTBoxplotMixin(object):
    #
    # boxplotPreferredSize()
    # - Return the preferred size
    #
    def boxplotPreferredDimensions(self, **kwargs):
        return (256,160)

    #
    # boxplotMinimumSize()
    # - Return the minimum size
    #
    def boxplotMinimumDimensions(self, **kwargs):
        return (160,128)

    #
    # boxplotSmallMultipleSize()
    # - Return the minimum size
    #
    def boxplotSmallMultipleDimensions(self, **kwargs):
        return (96,64)

    #
    # boxplotRequiredFields()
    # - Return the required fields for a boxplot configuration
    #
    def boxplotRequiredFields(self, **kwargs):
        columns = set()
        self.identifyColumnsFromParameters('bin_by',   kwargs, columns)
        self.identifyColumnsFromParameters('color_by', kwargs, columns)
        self.identifyColumnsFromParameters('count_by', kwargs, columns)
        return columns

    #
    # boxplot
    #
    # Make the SVG for a histogram from a dataframe
    #    
    def boxplot(self,
                df,                             # dataframe to render
                bin_by,                         # string or an array of strings                  
                # ----------------------------- # everything else is a default...
                style              = 'boxplot', # 'boxplot', 'boxplot_w_swarm', 'barchart'
                cap_swarm_at       = 200,       # cap the swarm plot at the specified number... if set to None, then no caps
                order_by           = 'sum',     # 'sum', 'max', 'median', 'average', 'min', or a list of index values
                ascending          = False,     # order by ascending
                # ----------------------------- #
                color_by           = None,      # just the default color or a string for a field
                global_color_order = None,      # color by ordering... if none (default), will be created and filled in...                  
                count_by           = None,      # none means just count rows, otherwise, use a field to sum by
                count_by_set       = False,     # count by using a set operation                  
                widget_id          = None,      # naming the svg elements
                # ----------------------------- # global rendering params
                global_max         = None,      # maximum to use for the bar length calculation
                global_min         = None,      # maximum to use for the bar length calculation
                just_calc_max      = False,     # forces return of the maximum for this render config...
                                                # ... which will then be used for the global max across bar charts...
                # ----------------------------- # small multiple options
                sm_type               = None,   # should be the method name // similar to the smallMultiples method
                sm_w                  = None,   # override the width of the small multiple
                sm_h                  = None,   # override the height of the small multiple
                sm_params             = {},     # dictionary of parameters for the small multiples
                sm_x_axis_independent = True,   # Use independent axis for x (xy, temporal, and linkNode)
                sm_y_axis_independent = True,   # Use independent axis for y (xy, temporal, periodic, pie)
                # ----------------------------- # rendering specific params                  
                track_state        = False,     # track state (for interactive filtering)
                x_view             = 0,         # x offset for the view
                y_view             = 0,         # y offset for the view
                x_ins              = 3,         # x insert
                y_ins              = 4,         # y_insert
                w                  = 256,       # width of the view
                h                  = 128,       # height of the view
                max_bar_w          = 14,        # max bar width
                min_bar_w          = 6,         # min bar width
                h_gap              = 0,         # gap between bars
                txt_h              = 14,        # text height
                label_rotation     = 45,        # label rotation
                extra_label_space  = 0,         # extra label space
                draw_labels        = True,      # draw labels flag
                draw_border        = True):     # draw a border around the histogram        
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTBoxplot(self, **_params_)

    #
    # RTBoxplot Class
    #
    class RTBoxplot(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     **kwargs):
            self.parms              = locals().copy()
            self.rt_self            = rt_self
            self.df                 = rt_self.copyDataFrame(kwargs['df'])

            # Make sure the bin_by is a list...
            bin_by = kwargs['bin_by']
            self.bin_by = [bin_by] if type(bin_by) != list else bin_by

            self.style              = kwargs['style']
            self.cap_swarm_at       = kwargs['cap_swarm_at']
            self.order_by           = kwargs['order_by']
            self.ascending          = kwargs['ascending']

            self.color_by           = kwargs['color_by']
            self.global_color_order = kwargs['global_color_order']
            self.count_by           = kwargs['count_by']
            self.count_by_set       = kwargs['count_by_set']

            # Make a histogram_id if it's not set already
            if kwargs['widget_id'] is None:
                self.widget_id = "boxplot_" + str(random.randint(0,65535))
            else:
                self.widget_id = kwargs['widget_id']

            self.global_max              = kwargs['global_max']
            self.global_min              = kwargs['global_min']

            self.sm_type                 = kwargs['sm_type']
            self.sm_w                    = kwargs['sm_w']
            self.sm_h                    = kwargs['sm_h']
            self.sm_params               = kwargs['sm_params']
            self.sm_x_axis_independent   = kwargs['sm_x_axis_independent']
            self.sm_y_axis_independent   = kwargs['sm_y_axis_independent']

            self.track_state             = kwargs['track_state']
            self.x_view                  = kwargs['x_view']
            self.y_view                  = kwargs['y_view']
            self.x_ins                   = kwargs['x_ins']
            self.y_ins                   = kwargs['y_ins']
            self.w                       = kwargs['w']
            self.h                       = kwargs['h']
            self.max_bar_w               = kwargs['max_bar_w']
            self.min_bar_w               = kwargs['min_bar_w']
            self.h_gap                   = kwargs['h_gap']
            self.txt_h                   = kwargs['txt_h']
            self.label_rotation          = kwargs['label_rotation']
            self.extra_label_space       = kwargs['extra_label_space']
            self.draw_labels             = kwargs['draw_labels']
            self.draw_border             = kwargs['draw_border']

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
            
            # For boxplot mode, we have to have a numerical count-by -- verify the style works with the parameters
            if (self.count_by is None or self.count_by_set) and (self.style == 'boxplot' or self.style == 'boxplot_w_swarm'):
                raise Exception("RTBoxplot - boxplot render style must use a scalar count_by")

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
        #
        #
        def __determineOrder__(self):
            if self.rt_self.isPandas(self.df):
                return self.__determineOrder_pandas__()
            elif self.rt_self.isPolars(self.df):
                return self.__determineOrder_polars__()
            else:
                raise Exception('RTBoxPlot.__determineOrder__()')

        #
        def __determineOrder_pandas__(self):
            # Aggregate into groups
            gb = self.df.groupby(by=self.bin_by)

            # Determine the order // assumption is that we've already checked & verified the style in __init__()
            # Counting by records... order by number of records
            if self.count_by is None: # barchart
                order = gb.size().sort_values(ascending=self.ascending)                
            # Counting by set... order by either number of records or by the magnitude of the set
            elif self.count_by_set:   # barchart
                if self.order is None:
                    order = gb.size().sort_values(ascending=self.ascending)
                else:
                    w_count_by = self.bin_by.copy()
                    _df        = self.df.groupby(by=w_count_by).size()
                    order      = _df.groupby(by=self.bin_by).size().sort_values(ascending=self.ascending)
            elif type(self.order_by) == list: # custom ordering... convert the order index into a categorical... remove missing... and sort by that
                order = gb[self.count_by].max().sort_values(ascending=self.ascending)
                order.index = pd.Categorical(order.index, categories=self.order_by)
                order = order[order.index.notnull()]
                order = order.sort_index()
            else: # barchart or boxplot ... mathematical version...
                if   self.order_by is None:    # Order by number of records
                    order = gb.size().sort_values(ascending=self.ascending)
                elif self.order_by == 'sum':
                    order = gb[self.count_by].sum().sort_values(ascending=self.ascending)
                elif self.order_by == 'max':
                    order = gb[self.count_by].max().sort_values(ascending=self.ascending)
                elif self.order_by == 'median':
                    order = gb[self.count_by].median().sort_values(ascending=self.ascending)
                elif self.order_by == 'average' or self.order_by == 'mean':
                    order = gb[self.count_by].mean().sort_values(ascending=self.ascending)
                elif self.order_by == 'min':
                    order = gb[self.count_by].min().sort_values(ascending=self.ascending)
                else:
                    raise Exception(f'RTBoxplot - do not understand order_by "{self.order_by}"')

            return gb, order

        #
        def __determineOrder_polars__(self):
            pb = self.df.partition_by(self.bin_by, as_dict=True)
            if    self.count_by is None or self.count_by_set: # barchart...
                order = self.rt_self.polarsCounter(self.df, self.bin_by, self.count_by, self.count_by_set)
            elif  type(self.order_by) == list: # user supplied a list... descend by that value
                counts = []
                for i in range(len(self.order_by)):
                    counts.append(len(self.order_by) - i)
                order = pl.DataFrame({'order':self.order_by, '__count__':counts})
            else:
                if self.order_by is None: # number of records
                    order = self.rt_self.polarsCounter(self.df, self.bin_by, None)
                elif self.order_by == 'sum':
                    order = self.rt_self.polarsCounter(self.df, self.bin_by, self.count_by)
                elif self.order_by == 'max':
                    _df_min_ = self.df.drop(set(self.df.columns) - set(self.bin_by) - set([self.count_by]))
                    order    = _df_min_.group_by(self.bin_by).max().rename({self.count_by:'__count__'})
                elif self.order_by == 'median':
                    _df_min_ = self.df.drop(set(self.df.columns) - set(self.bin_by) - set([self.count_by]))
                    order    = _df_min_.group_by(self.bin_by).median().rename({self.count_by:'__count__'})
                elif self.order_by == 'average' or self.order_by == 'mean':
                    _df_min_ = self.df.drop(set(self.df.columns) - set(self.bin_by) - set([self.count_by]))
                    order    = _df_min_.group_by(self.bin_by).mean().rename({self.count_by:'__count__'})
                elif self.order_by == 'min':
                    _df_min_ = self.df.drop(set(self.df.columns) - set(self.bin_by) - set([self.count_by]))
                    order    = _df_min_.group_by(self.bin_by).min().rename({self.count_by:'__count__'})
                else:
                    raise Exception(f'RTBoxplot -- do not understand order_by "{self.order_by}"')

            order = order.sort('__count__', descending=(not self.ascending))
            return pb, order

        #
        # renderSVG() - create the SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state:
                self.geom_to_df = {}
                
            # Determine the color order (for each bar)
            if self.global_color_order is None:
                self.global_color_order = self.rt_self.colorRenderOrder(self.df, self.color_by, self.count_by, self.count_by_set)

            # Adjust the min_bar_w if this is small multiples are to be included
            if self.sm_type is not None:
                if self.sm_w is None or self.sm_h is None:
                    self.sm_w,self.sm_h = getattr(self.rt_self, f'{self.sm_type}SmallMultipleDimensions')(**self.sm_params)
            else:
                self.sm_h,self.sm_w = 0,0 # Eliminate the dimensions from future calculations

            # Determine the ordering
            gb, order     = self.__determineOrder__()
            self.gb_count = len(order)

            # From there, calculate the bar width and check to make sure it's within bounds
            w_usable    = self.w - 2*self.x_ins
            total_bar_w = w_usable/self.gb_count
            x_left      = self.x_ins
            y_baseline  = self.h - self.y_ins
            max_bar_h   = self.h - 2*self.y_ins - self.sm_h
            if self.draw_labels:
                w_usable    = self.w - 2*self.x_ins - self.txt_h
                total_bar_w = w_usable/self.gb_count
                x_left      = self.x_ins + self.txt_h
                y_baseline  = self.h -   self.y_ins - self.txt_h 
                max_bar_h   = self.h - 2*self.y_ins - self.txt_h - self.sm_h

            actual_h_gap = self.h_gap
            bar_w        = total_bar_w - self.h_gap
            x_start      = x_left + 1

            if bar_w < self.min_bar_w:
                bar_w = self.min_bar_w
            if bar_w > self.max_bar_w:
                bar_w = self.max_bar_w

            # If we have more space then bars, then allocate the start and gap differently
            if self.sm_type is None:
                if (self.gb_count * (bar_w + self.h_gap)) < (w_usable - bar_w):
                    if    self.gb_count == 1:
                        x_start       += w_usable/2 - bar_w/2
                    else:
                        x_start       += w_usable/(self.gb_count+1) - bar_w/2
                        actual_h_gap   = w_usable/(self.gb_count+2)
            else:
                if (self.gb_count * self.sm_w) < (w_usable - self.sm_w):
                    if    self.gb_count == 1:
                        x_start       += w_usable/2 - bar_w/2
                    else:
                        x_start       += w_usable/(self.gb_count+1) - self.sm_w/2
                        actual_h_gap   = w_usable/(self.gb_count+2)
                else:
                    if self.max_bar_w < self.sm_w:
                        bar_w = self.max_bar_w
                    elif self.min_bar_w < self.sm_w:
                        bar_w = self.sm_w
                    else:
                        bar_w = self.min_bar_w

                    how_many_fit  = int(w_usable/self.sm_w)
                    w_of_each     = w_usable/how_many_fit
                    x_start      += w_of_each/2 - bar_w/2

                    actual_h_gap = w_of_each - bar_w
                
            # Get the min, max values ... have to do this on what gets rendered due to custom ordering parameters
            group_by_max,group_by_min = self.global_max,self.global_min
            if group_by_max is None: # assumption is that group_by_min won't be None exclusively...
                x,i = x_start,0
                while x < (self.w - self.x_ins) and i < len(order):
                    if self.rt_self.isPandas(self.df):
                        _index    = order.index[i]
                        _value    = order.iloc[i]
                        _as_tuple = (_index,) if type(_index) is not tuple else _index
                        _df    = gb.get_group(_as_tuple)
                    elif self.rt_self.isPolars(self.df):
                        _index = order[i].rows()[0][:len(self.bin_by)]
                        if len(self.bin_by) == 1:
                            _index = _index[0]
                        _value = order['__count__'][i]
                        _df    = gb[_index]

                    if self.style == 'boxplot' or self.style == 'boxplot_w_swarm':
                        if group_by_max is None:
                            group_by_max,group_by_min = _df[self.count_by].max(), _df[self.count_by].min()
                        else:
                            if group_by_max  < _df[self.count_by].max():
                                group_by_max = _df[self.count_by].max()
                            if group_by_min  > _df[self.count_by].min():
                                group_by_min = _df[self.count_by].min()                            
                    else:
                        group_by_min = 0
                        if   group_by_max is None:
                            group_by_max = _value
                        elif group_by_max < _value:
                            group_by_max = _value

                    x += bar_w + actual_h_gap
                    i += 1

            # Return the max if that's the request
            if just_calc_max:
                return group_by_min,group_by_max

            # Y-Transform Function
            yT = lambda __y__: (y_baseline - max_bar_h * (__y__ - group_by_min) / (group_by_max - group_by_min))

            # Start the SVG Frame
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w + self.extra_label_space}" height="{self.h + self.extra_label_space}" xmlns="http://www.w3.org/2000/svg">'
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg += f'<rect width="{self.w + self.extra_label_space - 1}" height="{self.h + self.extra_label_space - 1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'
            
            # Draw the background for the temporal chart
            axis_color = self.rt_self.co_mgr.getTVColor('axis','default')
            textfg     = self.rt_self.co_mgr.getTVColor('label','defaultfg')

            # Draw the axes
            svg += f'<line x1="{x_left}" y1="{y_baseline+1}" x2="{x_left}"            y2="{self.y_ins + self.sm_h}" stroke="{axis_color}" stroke-width="1" />'
            svg += f'<line x1="{x_left}" y1="{y_baseline+1}" x2="{x_left + w_usable}" y2="{y_baseline+1}"           stroke="{axis_color}" stroke-width="1" />'

            # Small multiples variables... even if we don't really need them...
            node_to_xy  = {}
            node_to_dfs = {}
            label_to_x  = {}

            # Render loop
            x,i = x_start,0
            while x < (self.w - self.x_ins) and i < len(order):
                if self.rt_self.isPandas(self.df):
                    _index    = order.index[i]
                    _value    = order.iloc[i]
                    _as_tuple = (_index,) if type(_index) is not tuple else _index
                    _df    = gb.get_group(_as_tuple)
                elif self.rt_self.isPolars(self.df):
                    _index = order[i].rows()[0][:len(self.bin_by)]
                    if len(self.bin_by) == 1:
                        _index = _index[0]
                    _value = order['__count__'][i]
                    _df    = gb[_index]

                px     = max_bar_h * _value / group_by_max

                if self.track_state:
                    _poly = Polygon([[x,y_baseline],[x+bar_w,y_baseline],[x+bar_w,y_baseline-px],[x,y_baseline-px]])
                    self.geom_to_df[_poly] = _df

                node_to_xy  [_index] = [x + bar_w/2, self.y_ins/2 + self.sm_h/2]
                node_to_dfs [_index] = _df
                label_to_x  [_index] = x + bar_w/2

                if   self.style == 'barchart':
                    svg += self.rt_self.colorizeBar(_df,
                                                    self.global_color_order, self.color_by,
                                                    self.count_by, self.count_by_set,
                                                    x, y_baseline, px, bar_w, False)
                elif self.style.startswith('boxplot'):
                    svg += self.rt_self.renderBoxPlotColumn(self.style, 
                                                     _df,
                                                     x + bar_w/2,
                                                     yT,
                                                     group_by_max,
                                                     group_by_min,
                                                     bar_w,
                                                     self.count_by,
                                                     self.color_by,
                                                     self.cap_swarm_at)

                x += bar_w + actual_h_gap
                i += 1

            # Draw indicator that more data existed than that could be rendered
            if i != len(order):
                error_co = self.rt_self.co_mgr.getTVColor('label','error')                
                svg += f'<line x1="{self.w-5}" y1="{self.h-9}" x2="{self.w-2}" y2="{self.h-5}" stroke="{error_co}" stroke-width="1" />'
                svg += f'<line x1="{self.w-5}" y1="{self.h-1}" x2="{self.w-2}" y2="{self.h-5}" stroke="{error_co}" stroke-width="1" />'

            # Small multiples
            if self.sm_type is not None:
                sm_lu = self.rt_self.createSmallMultiples(self.df, node_to_dfs, node_to_xy,
                                                          self.count_by, self.count_by_set, self.color_by, None, self.widget_id,
                                                          self.sm_type, self.sm_params, self.sm_x_axis_independent, self.sm_y_axis_independent,
                                                          self.sm_w, self.sm_h)
                for node_str in sm_lu.keys():
                    svg += sm_lu[node_str]

            # Draw labeling
            if self.draw_labels:
                # Max Label
                _str_max,_str_min = f'{group_by_max:{self.rt_self.fformat}}',''
                if re.match(r'.*\.0*',_str_max):
                    _str_max = _str_max[:_str_max.index('.')]
                svg += self.rt_self.svgText(_str_max,     self.x_ins+self.txt_h-2, self.y_ins + self.sm_h,             self.txt_h, anchor='end',    rotation=-90)

                # Min Label (for boxplot only)
                if self.style.startswith('boxplot'):
                    _str_min = f'{group_by_min:{self.rt_self.fformat}}'
                    if re.match(r'.*\.0*',_str_min):
                        _str_min = _str_min[:_str_min.index('.')]
                    svg += self.rt_self.svgText(_str_min, self.x_ins+self.txt_h-2, self.y_ins + self.sm_h + max_bar_h, self.txt_h, anchor='start',  rotation=-90)

                # Count By Label
                _count_by_str = 'rows'
                if self.count_by:
                    _count_by_str = self.count_by
                if self.rt_self.textLength(_str_max, self.txt_h) + self.rt_self.textLength(_str_min, self.txt_h) + self.rt_self.textLength(_count_by_str, self.txt_h) + 10 < max_bar_h:
                    if self.style.startswith('boxplot'):
                        _mid_y = self.rt_self.textLength(_str_max, self.txt_h) + (max_bar_h - self.rt_self.textLength(_str_max, self.txt_h) - self.rt_self.textLength(_str_min, self.txt_h))/2
                        svg += self.rt_self.svgText(_count_by_str, self.x_ins+self.txt_h-2, 
                                                    self.y_ins + self.sm_h + _mid_y, 
                                                    self.txt_h, anchor='middle',  rotation=-90)
                    else:
                        svg += self.rt_self.svgText(_count_by_str, self.x_ins+self.txt_h-2, self.y_ins + self.sm_h + max_bar_h,   self.txt_h, anchor='start',  rotation=-90)

                # Draw the labels for the y-axis ranges // OLD VERSION OF Y-AXIS LABELING
                #svg += self.rt_self.svgText(f'{group_by_max:{self.rt_self.fformat}}', self.x_ins+self.txt_h-2, self.y_ins + self.sm_h,             self.txt_h, anchor='end', rotation=-90)
                #if self.style.startswith('boxplot'):
                #    svg += self.rt_self.svgText(f'{group_by_min:{self.rt_self.fformat}}', self.x_ins+self.txt_h-2, self.y_ins + self.sm_h + max_bar_h, self.txt_h, anchor='start', rotation=-90)
                
                # x-axis labels are tricky...  can it just fit horizontally?
                horz_fits = True
                if len(label_to_x.keys()) > 1: # by default... length of 1 == horizontal...
                    for _label in label_to_x.keys():
                        if type(_label) is tuple:
                            as_str = '|'.join(_label)
                        else:
                            as_str = str(_label)
                        if self.rt_self.textLength(as_str, self.txt_h) > bar_w+actual_h_gap:
                            horz_fits = False
                
                # Horizontal label rendering
                if horz_fits:
                    for _label in label_to_x.keys():
                        x = label_to_x[_label]
                        if type(_label) is tuple:
                            as_str = '|'.join(_label)
                        else:
                            as_str = str(_label)
                        
                        svg += self.rt_self.svgText(as_str, x, self.y_ins + self.sm_h + max_bar_h + self.txt_h , self.txt_h, anchor='middle')

                # Angled label rendering...
                else:
                    usable_txt_h = self.txt_h
                    if usable_txt_h > bar_w+actual_h_gap:
                        usable_txt_h = bar_w+actual_h_gap

                    _angle = self.rt_self.bestAngleForRotatedLabels(bar_w+actual_h_gap,usable_txt_h)
                    for _label in label_to_x.keys():
                        x = label_to_x[_label]
                        if type(_label) is tuple:
                            as_str = '|'.join(_label)
                        else:
                            as_str = str(_label)
                        tpos,bpos = self.rt_self.calculateAngledLabelTopAndBottomPosition(x-bar_w, self.y_ins+self.sm_h+max_bar_h, bar_w+actual_h_gap, usable_txt_h, _angle)
                        svg += self.rt_self.svgText(as_str, bpos[0], bpos[1] - usable_txt_h/3, usable_txt_h, rotation=_angle)

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
                    _dfs.append(self.geom_to_df[_poly])
            if len(_dfs) > 0:
                return self.rt_self.concatDataFrames(_dfs)
            else:
                return None
