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

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from datetime import timedelta
from dateutil.relativedelta import relativedelta

import random

from .rt_component import RTComponent

__name__ = 'rt_timeline_mixin'

#
# Abstraction for Timeline
#
class RTTimelineMixin(object):
    #
    # timeLinePreferredDimensions()
    # - Return the preferred size
    #
    def timelinePreferredDimensions(self, **kwargs):
        return (160,24)

    #
    # timelineMinimumDimensions()
    # - Return the minimum size
    #
    def timelineMinimumDimensions(self, **kwargs):
        return (96,16)

    #
    # timelineSmallMultipleDimensions()
    #
    def timelineSmallMultipleDimensions(self, **kwargs):
        return (96,16)

    #
    # Identify the required fields in the dataframe for this visualization
    #
    def timelineRequiredFields(self, **kwargs):
        columns_set = set()
        return columns_set

    #
    # timeline
    # - make just a timeline
    #    
    def timeline(self,
                 df                    = None,  # dataframe to render // if none, pull from the timestamp, timestamp_end field
                 ts_field              = None,  # timestamp field     // if df supplied, ts_field will be guessed
                 timestamp             = None,  # start timestamp     // can supply just this and the end... 
                 timestamp_end         = None,  # end timestamp
                 # ---------------------------- # small multiple options
                 sm_type               = None,  # should be the method name // similar to the smallMultiples method
                 sm_w                  = None,  # override the width of the small multiple
                 sm_h                  = None,  # override the height of the small multiple
                 sm_params             = {},    # dictionary of parameters for the small multiples
                 sm_x_axis_independent = True,  # Use independent axis for x (xy, temporal, and linkNode)
                 sm_y_axis_independent = True,  # Use independent axis for y (xy, temporal, periodic, pie)
                 # ---------------------------- # visualization geometry / etc.
                 line_width            = 4,     # pixels for the timeline
                 x_view                = 0,     # x offset for the view
                 y_view                = 0,     # y offset for the view
                 x_ins                 = 3,     # side inserts
                 y_ins                 = 3,     # top & bottom inserts
                 w                     = 512,   # width of the view
                 h                     = 32,    # height of the view
                 txt_h                 = 14,    # text height for labeling
                 background_opacity    = 1.0,
                 draw_labels           = True,  # draw labels flag
                 draw_border           = True,  # draw a border around the histogram
                 draw_context          = True): # draw temporal context information if (and only if) x_axis is time
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTTimeline(self, **_params_)

    #
    # RTTimeline
    #
    class RTTimeline(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     **kwargs):
            self.parms                   = locals().copy()
            self.rt_self                 = rt_self
            self.df                      = None
            self.ts_field                = kwargs['ts_field']
            self.timestamp               = kwargs['timestamp']
            self.timestamp_end           = kwargs['timestamp_end']
            self.sm_type                 = kwargs['sm_type']
            self.sm_w                    = kwargs['sm_w']
            self.sm_h                    = kwargs['sm_h']
            self.sm_params               = kwargs['sm_params']
            self.sm_x_axis_independent   = kwargs['sm_x_axis_independent']
            self.sm_y_axis_independent   = kwargs['sm_y_axis_independent']
            self.line_width              = kwargs['line_width']
            self.x_view                  = kwargs['x_view']
            self.y_view                  = kwargs['y_view']
            self.x_ins                   = kwargs['x_ins']
            self.y_ins                   = kwargs['y_ins']
            self.w                       = kwargs['w']
            self.h                       = kwargs['h']
            self.txt_h                   = kwargs['txt_h']
            self.background_opacity      = kwargs['background_opacity']
            self.draw_labels             = kwargs['draw_labels']
            self.draw_border             = kwargs['draw_border']
            self.widget_id               = 'rt_timeline_' + str(random.randint(1,65556))

            if kwargs['df'] is not None:
                self.df                  = kwargs['df'].copy()
                if self.ts_field is None:
                    choices = self.df.select_dtypes(np.datetime64).columns
                    if   len(choices) == 1:
                        self.ts_field = choices[0]
                    elif len(choices) >  1:
                        print('multiple timestamp fields... choosing the first (RTTimeline)')
                        self.ts_field = choices[0]
                    else:
                        raise Exception('no timestamp field supplied to RTTimeline(), cannot automatically determine field')
                self.timestamp     = self.df[self.ts_field].min()
                self.timestamp_end = self.df[self.ts_field].max()
            elif kwargs['df'] is None and self.timestamp is not None and self.timestamp_end is not None:
                if type(self.timestamp)     == str:
                    self.timestamp     = pd.to_datetime(self.rt_self.minTimeForStringPrecision(self.timestamp))
                if type(self.timestamp_end) == str:
                    self.timestamp_end = pd.to_datetime(self.rt_self.maxTimeForStringPrecision(self.timestamp_end))
            elif kwargs['df'] is None and self.timestamp is not None and type(self.timestamp) == str:
                _str = self.timestamp
                self.timestamp     = pd.to_datetime(self.rt_self.minTimeForStringPrecision(_str))
                self.timestamp_end = pd.to_datetime(self.rt_self.maxTimeForStringPrecision(_str))
            else:
                raise Exception('either need a df an ts_field... of a timestamp, timestamp_end in RTTimeline()')
            
            self.last_render = None

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
        def renderSVG(self):
            # Create the SVG
            svg = f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">'
            
            # Render the background
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg += f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" fill-opacity="{self.background_opacity}" stroke="{background_color}" stroke-opacity="{self.background_opacity}" />'
            
            _color = self.rt_self.co_mgr.getTVColor('data','default')
            svg += f'<line x1="{self.x_ins}" y1="{self.y_ins}" x2="{self.w-self.x_ins}" y2="{self.y_ins}" stroke-width="{self.line_width}" stroke="{_color}" />'

            # Draw labels if requested
            if self.draw_labels:
                svg += self.rt_self.drawXYTemporalContext(self.x_ins, self.y_ins + self.line_width, self.w, self.h, self.txt_h, self.timestamp, self.timestamp_end, self.draw_labels)

            _min_str,_max_str         = self.rt_self.condenseTimeLabels(self.timestamp, self.timestamp_end)
            _min_str_len,_max_str_len = self.rt_self.textLength(_min_str, self.txt_h), self.rt_self.textLength(_max_str, self.txt_h) 

            _x0 = 5
            svg += f'<rect x="{_x0-3}" y="{self.line_width+1}" width="{_min_str_len+6}" height="{self.txt_h+6}" fill="{background_color}" fill-opacity="0.5" />'
            svg += self.rt_self.svgText(_min_str, _x0, self.line_width + self.txt_h + 3, self.txt_h)

            _x1 = self.w - 5 - _max_str_len
            svg += f'<rect x="{_x1-3}" y="{self.line_width+1}" width="{_min_str_len+6}" height="{self.txt_h+6}" fill="{background_color}" fill-opacity="0.5" />'
            svg += self.rt_self.svgText(_max_str, _x1, self.line_width + self.txt_h + 3, self.txt_h)

            svg += '</svg>'
            self.last_render = svg
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

            if    _ts < self.timestamp:
                return -self.x_ins
            elif  _ts > self.timestamp_end:
                return -(self.w - self.x_ins)
            else:
                w_usable = self.w - 2*self.x_ins
                return self.x_ins + w_usable*((_ts - self.timestamp)/(self.timestamp_end - self.timestamp))
        
        #
        # timestampExtents()
        # - return the minimum and maximum timestamps as a pandas tuple
        #
        def timestampExtents(self):
            return self.timestamp,self.timestamp_end
