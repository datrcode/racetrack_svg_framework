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

from shapely.geometry import Polygon
from math import sqrt, floor, ceil

from .rt_component import RTComponent
from .rt_entity_position import RTEntityPosition

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
                  df,                          # dataframe to render
                  bin_by,                      # string or an array of strings                  
                  # -------------------------- # everything else is a default...                  
                  color_by           = None,   # just the default color or a string for a field
                  global_color_order = None,   # color by ordering... if none (default), will be created and filled in...                  
                  count_by           = None,   # none means just count rows, otherwise, use a field to sum by
                  count_by_set       = False,  # count by using a set operation                  
                  widget_id          = None,   # naming the svg elements
                  # -------------------------- # global rendering params
                  first_line_i       = 0,      # first line index to render
                  global_max         = None,   # maximum to use for the bar length calculation
                  global_min         = None,   # mininum to use for the bar length calculation -- which is treated as zero by histograms...
                  just_calc_max      = False,  # forces return of the maximum for this render config...
                                               # ... which will then be used for the global max across bar charts...
                  draw_distribution  = False,  # draw the distribution
                  # -------------------------- # rendering specific params
                  labels             = None,   # labels to be used in place of the dataframe cell values     
                  track_state        = False,  # track state for interactive filtering             
                  x_view             = 0,      # x offset for the view
                  y_view             = 0,      # y offset for the view
                  w                  = 128,    # width of the view
                  h                  = 256,    # height of the view
                  bar_h              = 14,     # bar height
                  v_gap              = 0,      # gap between bars
                  draw_labels        = True,   # draw labels flag
                  draw_border        = True    # draw a border around the histogram
                 ):
        """Implementation of a histogram in SVG.

        Required Parameters
        -------------------
        df : pandas.DataFrame | polars.DataFrame
            Dataframe to render

        bin_by : str | list[str]
            Field(s) to bin by

        Useful Parameters
        -----------------

        color_by : str | None
            The field to be used to color the bars

        count_by : str | None
            The field to be used to count the bins
        
        count_by_set : bool
            If True, use a set operation to count the items in a bin
        
        first_line_i : int
            The first bar index to render

        draw_distribution : bool
            If True, draw the distribution of the bars

        labels : dict
            A dictionary of labels to be used in place of the dataframe cell values

        Globalization Parameters
        ------------------------

        global_color_order : list
            The order to use for the color palette within each bar

        global_max : int
            The maximum to use for the bar length calculation

        global_min : int
            The minimum to use for the bar length calculation

        just_calc_max : bool
            Forces return of the maximum for this render config... which will then be used for the global max across bar charts

        Standard Parameters
        -------------------

        widget_id : str
            The id of the SVG widget.  If set to None, a random id will be generated.

        track_state : bool
            Track state for interactive filtering operations
        
        x_view, y_view : int
            The x and y offset for the SVG view

        w, h : int
            The width and height of the SVG frame

        bar_h : int
            The height of individual bars and the height of the text

        v_gap : int
            The vertical gap between bars

        draw_labels : bool
            Draw the node labels (link labels are enabled by "link_labels" parameter)

        draw_border : bool
            Draw a border around the visualization

        """
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTHistogram(self, **_params_)

    #
    # RTHistogram Class
    #
    class RTHistogram(RTComponent):
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
            if type(bin_by) != list: # Make it into a list for consistency
                self.bin_by = [bin_by]
            else:
                self.bin_by = bin_by

            self.color_by           = kwargs['color_by']
            self.global_color_order = kwargs['global_color_order']
            self.count_by           = kwargs['count_by']
            self.count_by_set       = kwargs['count_by_set']

            # Make a histogram_id if it's not set already
            if kwargs['widget_id'] is None:
                self.widget_id = "histogram_" + str(random.randint(0,2**24))
            else:
                self.widget_id          = kwargs['widget_id']

            self.first_line_i       = kwargs['first_line_i']
            self.global_max         = kwargs['global_max']
            self.global_min         = kwargs['global_min']
            self.draw_distribution  = kwargs['draw_distribution']
            self.labels             = kwargs['labels']
            if self.labels is None: self.labels = {}
            self.track_state        = kwargs['track_state']
            self.x_view             = kwargs['x_view']
            self.y_view             = kwargs['y_view']
            self.w                  = kwargs['w']
            self.h                  = kwargs['h']
            self.bar_h              = kwargs['bar_h']
            self.v_gap              = kwargs['v_gap']
            self.draw_labels        = kwargs['draw_labels']
            self.draw_border        = kwargs['draw_border']

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
        # print() version of class
        #
        def __repr__(self):
            def tQontQ(t): return 'None' if t is None else "'" + str(t) + "'"
            return f'histogram(df.len={len(self.df)}, bin_by={self.bin_by}, count_by={tQontQ(self.count_by)}, color_by={tQontQ(self.color_by)}, {self.w}x{self.h})'

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
            return self.last_render

        #
        # applyScrollEvent()
        # - scroll the list by the specified amount
        # - coordinate included to make it similar to other view functionality
        # ... looks like we don't know the list length... so we can't bound the calc by that...
        #
        # 2024-04-26 // continues to throw exceptions -- removed for now
        #
        def BROKEN_applyScrollEvent(self, scroll_amount, coordinate=None):
            scroll_amount /= 100.0
            if (self.first_line_i+scroll_amount) >= 0:
                self.first_line_i += scroll_amount
            else:
                self.first_line_i =  0
            self.last_render = None
            return True

        #
        # binOrder() - determine the bin order (pandas)
        # - side effects include modification of dataframe and changes to the count_by_field member variable
        #
        def __binOrder_pandas__(self):
            if    self.count_by is None:
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
            return order, gb
        
        #
        # binOrder() - determine the bin order (polars)
        #
        def __binOrder_polars__(self):
            self.df = self.df.sort(self.bin_by) # not necessary... but makes the bins that are equal ordered alphabetically/naturally
            if   self.count_by is None:
                order = self.df.group_by(self.bin_by, maintain_order=True).agg(pl.len().alias('__count__')).sort('__count__', descending=True)
            elif self.count_by_set:
                if self.count_by in self.bin_by:
                    df_min  = self.df.drop(set(self.df.columns) - set(self.bin_by))
                    df_dupe = df_min.with_columns(pl.col(self.count_by).alias('__count__'))
                    order   = df_dupe.group_by(self.bin_by, maintain_order=True).n_unique().sort('__count__', descending=True)
                else:
                    df_min  = self.df.drop(set(self.df.columns) - set(self.bin_by) - set([self.count_by]))
                    order   = df_min.group_by(self.bin_by, maintain_order=True).n_unique()
                    order   = order.rename({self.count_by:'__count__'}).sort('__count__', descending=True)
            else:
                if self.count_by in self.bin_by:
                    df_min  = self.df.drop(set(self.df.columns) - set(self.bin_by))
                    df_dupe = df_min.with_columns(pl.col(self.count_by).alias('__count__'))
                    order   = df_dupe.group_by(self.bin_by, maintain_order=True).agg(pl.sum('__count__')).sort('__count__', descending=True)
                else:
                    df_min  = self.df.drop(set(self.df.columns) - set(self.bin_by) - set([self.count_by]))
                    order   = df_min.group_by(self.bin_by, maintain_order=True).agg(pl.sum(self.count_by).alias('__count__')).sort('__count__', descending=True)
            return order, self.df.partition_by(self.bin_by, as_dict=True)

        #
        # renderSVG() - create the SVG
        #
        def renderSVG(self, just_calc_max=False):
            self.entity_pos = {}
            if self.track_state: self.geom_to_df = {}

            # Leave space for a label
            max_bar_w = self.w - self.bar_h
                                            
            # Determine the color order (for each bar)
            if self.global_color_order is None:
                self.global_color_order = self.rt_self.colorRenderOrder(self.df, self.color_by, self.count_by, self.count_by_set)

            # Determine the bin order (for the bars)
            if   self.rt_self.isPandas(self.df): order, gb = self.__binOrder_pandas__()
            elif self.rt_self.isPolars(self.df): order, gb = self.__binOrder_polars__()
            else: raise Exception('RTHistogram.renderSVG() - only pandas and polars dataframes supported')

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
                if   self.rt_self.isPandas(self.df): max_group_by = order.iloc[0]
                elif self.rt_self.isPolars(self.df): max_group_by = order['__count__'][0]
                else: raise Exception('RTHistogram.renderSVG() - only pandas or polars dataframes supported [2]')
                
                if just_calc_max:     return 0,max_group_by
                if max_group_by == 0: max_group_by = 1
            else: max_group_by = self.global_max
            
            #
            # Render each bin ... only do the visible ones...
            # ... make sure the first line isn't more than the number of lines...
            #
            i = self.first_line_i
            if i >= len(order): self.first_line_i = i = len(order) - 1
            if i <  0:          self.first_line_i = i = 0
            y = 0
            while y < (self.h - 1.9*self.bar_h) and i < len(order):
                # Bin label... used for the id... and used for the labeling (if draw_labels is true)
                if self.rt_self.isPandas(self.df):
                    px = max_bar_w * order.iloc[i] / max_group_by
                    if type(order.index[i]) != list and type(order.index[i]) != tuple:
                        bin_text = str(order.index[i])
                    else:
                        bin_text = ' | '.join([str(x) for x in order.index[i]])

                    _tuple_ = order.index[i]
                    if type(_tuple_) != tuple: _tuple_ = (_tuple_, )
                    k_df = gb.get_group(_tuple_)

                    # k_df = gb.get_group(order.index[i]) # 2024-02-07 -- changed due to pandas warning...
                elif self.rt_self.isPolars(self.df):
                    px = max_bar_w * order['__count__'][i] / max_group_by
                    _list_ = []
                    for _bin_ in self.bin_by: _list_.append(order[_bin_][i])
                    _tuple_ = tuple(_list_)
                    bin_text = ' | '.join([str(x) for x in _tuple_])
                    k_df = gb[_tuple_]

                    # 2024-02-07 -- believe polars changed this behavior...
                    #
                    #if len(_tuple_) == 1:
                    #    k_df = gb[_tuple_[0]]
                    #else:
                    #    k_df = gb[_tuple_]

                # Make a safe id to reference this element later
                element_id = self.widget_id + "_" + self.rt_self.encSVGID(bin_text)

                # Make the bar // even if we're going to color it in later
                color = self.rt_self.co_mgr.getTVColor('data','default')

                # Render the bar ... next section does the color... but this makes sure it's at least filled in...
                svg += f'<rect id="{element_id}" width="{px}" height="{self.bar_h}" x="0" y="{y}" fill="{color}" stroke="{color}"/>'
                self.entity_pos[bin_text] = (y, px, element_id) # for entity positions
                if self.track_state: self.geom_to_df[Polygon([[0,y],[px,y],[px,y+self.bar_h],[0,y+self.bar_h]])] = k_df

                # 'Color By' options
                if self.color_by is not None:
                    svg += self.rt_self.colorizeBar(k_df, self.global_color_order, self.color_by, self.count_by, self.count_by_set, 0, y, px, self.bar_h, True)

                # Render the label
                if self.draw_labels:
                    _label_ = str(bin_text)
                    if bin_text      is self.labels: _label_ = self.labels[bin_text]
                    if str(bin_text) in self.labels: _label_ = self.labels[str(bin_text)]
                    cropped_bin_text = self.rt_self.cropText(_label_, self.bar_h-2, max_bar_w)
                    defer_labels.append(self.rt_self.svgText(cropped_bin_text, 2, y+self.bar_h-1, self.bar_h-2))
                
                i += 1
                y += self.bar_h+1+self.v_gap
            
            # Draw the distribution
            if self.draw_distribution:
                svg += self.renderDistribution(self.df, order, max_bar_w)
            
            # Indicate how many more we are missing
            if self.draw_labels and i != len(order):
                svg += self.rt_self.svgText(f'{len(order)-i} more', 2, self.h-3, self.bar_h-1, color=self.rt_self.co_mgr.getTVColor('label','error'))
            
            # Draws the maximum amount of the histogram
            if self.draw_labels:
                # Draw deferred labels
                for _label in defer_labels: svg += _label

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
        # renderDistribution()
        #
        def renderDistribution(self, df, order, max_bar_w, bucket_pixels=8):
            svg     = ''
            hist_h = self.h/3
            if hist_h >= 100:
                hist_h = 100
            counts  = []

            if self.rt_self.isPandas(df):
                # Pandas
                percs = order / order.max()
                buckets = floor(max_bar_w/bucket_pixels)
                for i in range(buckets):
                    t_or_f = ((percs > (i/buckets)) & (percs <= ((i+1)/buckets))).value_counts()
                    if True in t_or_f:
                        counts.append(t_or_f.loc[True])
                    else:
                        counts.append(0)
            elif self.rt_self.isPolars(df):
                # Polars
                percs   = order.with_columns((pl.col('__count__')/pl.col('__count__').max()).alias('perc'))
                buckets = floor(max_bar_w/bucket_pixels)
                for i in range(buckets):
                    counts.append(len(percs.filter((pl.col('perc') > (i/buckets)) & (pl.col('perc') <= ((i+1)/buckets)))))

            ybase = self.h-self.bar_h-2
            _max_ = max(counts)
            _color_ = self.rt_self.co_mgr.getTVColor('axis','major')
            for i in range(buckets):
                h = hist_h * counts[i]/_max_
                if h > 1:
                    svg += f'<rect width="{bucket_pixels-0.5}" height="{h}" x="{2+i*bucket_pixels}" y="{ybase-h}" fill="{_color_}" stroke="none" stroke-width="0.5" />'
                else:
                    svg += f'<line x1="{2+i*bucket_pixels}" y1="{ybase-h}" x2="{2+(i+1)*bucket_pixels-0.5}" y2="{ybase-h}" stroke="{_color_}" stroke-width="0.5" />'
            return svg

        # <copied from RTComponent.py>
        # entityPositions() - return information about the entity geometry for rendering
        # - Empty list means either not implemented... or entity not in view...
        #
        # (originally developed in the RTChordDiagram component... probably overkill here //2024-03-31)
        #
        # - return the positions of the entity ... rendering had to have happened first
        def __entityPositions__(self, entity):
            _results_ = []
            if entity in self.entity_pos:
                _tuple_      = self.entity_pos[entity]
                x,y          = _tuple_[1]/2.0, _tuple_[0] + self.bar_h/2.0 # point to location
                xa,ya        = _tuple_[1],     _tuple_[0] + self.bar_h/2.0 # attachment point
                _svg_markup_ = '<rect x="0" y="{_tuple_[0]}" width="{_tuple_[1]}" height="{self.bar_h}" />'
                _ep_         = RTEntityPosition(entity, self.rt_self, self, (x,y), (xa,ya,1.0,0.0), _tuple_[2], _svg_markup_, self.widget_id)
                _ep_.addAttachmentPointVec((0.0,y,-1.0,0.0))
                _results_.append(_ep_)
            return _results_
        
        def entityPositions(self, entity_or_label):
            if entity_or_label in self.entity_pos:
                return self.__entityPositions__(entity_or_label)
            elif len(self.labels) > 0:
                rteps = []
                for entity in self.labels:
                    if self.labels[entity] == entity_or_label:
                        _results_ = self.__entityPositions__(entity)
                        for rtep in _results_:
                            rtep.entity = entity_or_label
                            rteps.append(rtep)
                return rteps
            return []

        #
        # smallMultipleFeatureVector()
        # ... feature vector for comparison with other small multiple instances of this class
        # ... pretty much a copy of the render code above...
        #
        def smallMultipleFeatureVector(self):
            # Determine the bin order (for the bars)
            if self.rt_self.isPandas(self.df):
                order, gb = self.__binOrder_pandas__()
            elif self.rt_self.isPolars(self.df):
                order, gb = self.__binOrder_polars__()
            else:
                raise Exception('RTHistogram.renderSVG() - only pandas and polars dataframes supported')

            # Determine the max bin size... make sure it isn't zero
            if self.global_max is None:
                if self.rt_self.isPandas(self.df):
                    max_group_by = order.iloc[0]
                elif self.rt_self.isPolars(self.df):
                    max_group_by = order['__count__'][0]

                if max_group_by == 0:
                    max_group_by = 1
            else:
                max_group_by = self.global_max
            
            # Calculate each bar width
            max_bar_w,fv,i = 1.0,{},0
            while i < len(order):
                # Width of the bar in pixels
                if self.rt_self.isPandas(self.df):
                    px = max_bar_w * order.iloc[i] / max_group_by
                    if type(order.index[i]) != list and type(order.index[i]) != tuple:
                        bin_text = str(order.index[i])
                    else:
                        bin_text = ' | '.join([str(x) for x in order.index[i]])
                elif self.rt_self.isPolars(self.df):
                    px = max_bar_w * order['__count__'][i] / max_group_by
                    _list_ = []
                    for _bin_ in self.bin_by:
                        _list_.append(order[_bin_][i])
                    _tuple_ = tuple(_list_)
                    bin_text = ' | '.join([str(x) for x in _tuple_])
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
                return self.rt_self.concatDataFrames(_dfs)
            else:
                return None
