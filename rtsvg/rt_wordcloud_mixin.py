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

from wordcloud import WordCloud, STOPWORDS

from shapely.geometry import Polygon

import random

from .rt_component import RTComponent

__name__ = 'rt_wordcloud_mixin'

#
# Abstraction for Pie Chart
#
class RTWordCloudMixin(object):
    #
    # pieChartPreferredDimensions()
    # - Return the preferred size
    #
    def wordCloudPreferredDimensions(self, **kwargs):
        return (128,96)

    #
    # pieChartMinimumDimensions()
    # - Return the minimum size
    #
    def wordCloudMinimumDimensions(self, **kwargs):
        return (80,64)

    #
    # pieChartSmallMultipleDimensions()
    #
    def wordCloudSmallMultipleDimensions(self, **kwargs):
        return (128,96)

    #
    # Identify the required fields in the dataframe
    #
    def wordCloudRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('text_fields', kwargs, columns_set)
        return columns_set

    #
    # pieChart
    #
    # Make the SVG for a piechart
    #    
    def wordCloud(self,
                  df,                               # dataframe to render
                  text_fields,                      # Single field name or list of field names
                  # ------------------------------- # everything else is a default...
                  widget_id            = None,      # naming the svg elements
                  # ------------------------------- # visualization geometry / etc.
                  count_by             = None,      # for compatibility... no use... 
                  color_by             = None,      # for compatibility... no use...
                  track_state          = False,     # for compatibility... no use...
                  x_view               = 0,         # x offset for the view
                  y_view               = 0,         # y offset for the view
                  x_ins                = 3,         # side inserts
                  y_ins                = 3,         # top & bottom inserts
                  w                    = 256,       # width of the view
                  h                    = 224,       # height of the view
                  draw_border          = True,      # draw a border around the word cloud
                  draw_background      = False):    # probably doesn't apply... keeping for consistency
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTWordCloud(self, **_params_)

    #
    # RTWordCloud
    #
    class RTWordCloud(RTComponent):
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
                self.widget_id = "wordcloud_" + str(random.randint(0,65535))

            if type(kwargs['text_fields']) == list:
                self.text_fields = kwargs['text_fields']
            else:
                self.text_fields = [kwargs['text_fields']]

            self.x_view               = kwargs['x_view']
            self.y_view               = kwargs['y_view']
            self.x_ins                = kwargs['x_ins']
            self.y_ins                = kwargs['y_ins']
            self.w                    = int(kwargs['w'])
            self.h                    = int(kwargs['h'])
            self.draw_border          = kwargs['draw_border']
            self.draw_background      = kwargs['draw_background']

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
            if just_calc_max:
                return float('inf') # Bad idea?  probably... shouldn't be used really...

            #
            # Geometry
            #
            params_orig_minus_self = self.parms.copy()
            params_orig_minus_self.pop('self')
            min_dims = self.rt_self.wordCloudSmallMultipleDimensions(**params_orig_minus_self)
            if self.w < min_dims[0] or self.h < min_dims[1]:
                self.x_ins        = 1
                self.y_ins        = 1

            w_usable = self.w - 2*self.x_ins
            h_usable = self.h - 2*self.y_ins

            # Create the outer SVG
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'

            # Put the text together
            if   self.rt_self.isPandas(self.df) or self.rt_self.isPolars(self.df):
                _txt_ = ''
                for _field_ in self.text_fields:
                    _txt_ += ' '.join(self.df[_field_].tolist())
                    _txt_ += '\n'
            else:
                raise Exception('RTWordCloud.renderSVG() - only pandas and polars supported')

            # Create the word cloud
            wc = WordCloud(width=w_usable,height=h_usable).generate_from_text(_txt_)

            # Inner SVG
            svg += f'<svg x="{self.x_ins}" y="{self.y_ins}" width="{w_usable}" height="{h_usable}">'
            svg += wc.to_svg(embed_font=True)
            svg += '</svg>'

            svg += '</svg>'
            self.last_render = svg
            return svg
        
        #
        # smallMultipleFeatureVector() ... maybe should be the size of each 
        #
        def smallMultipleFeatureVector(self):
            return {}

        #
        # overlappingDataFrames() ... just return none...
        #
        def overlappingDataFrames(self, to_intersect):
            return None
