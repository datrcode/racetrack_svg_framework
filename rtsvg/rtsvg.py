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
import hashlib
import random
import urllib
import html
import os

from datetime import datetime

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import io

from math import cos,sin,pi

from IPython.core import display as ipc_display

from IPython.display import Javascript, HTML, display

from .rt_annotations_mixin      import RTAnnotationsMixin
from .rt_art_mixin               import RTArtMixin
from .rt_boxplot_mixin           import RTBoxplotMixin
from .rt_calendarheatmap_mixin   import RTCalendarHeatmapMixin
from .rt_chord_diagram_mixin     import RTChordDiagramMixin
from .rt_choroplethmap_mixin     import RTChoroplethMapMixin
from .rt_color_manager           import RTColorManager
from .rt_datamanip_mixin         import RTDataManipMixin
from .rt_geomaps_mixin           import RTGeoMapsMixin
from .rt_geometry_mixin          import RTGeometryMixin
from .rt_graph_layouts_mixin     import RTGraphLayoutsMixin
from .rt_histogram_mixin         import RTHistogramMixin
from .rt_json_mixin              import RTJSONMixin
from .rt_layouts_mixin           import RTLayoutsMixin
from .rt_link_mixin              import RTLinkMixin
from .rt_linknode_mixin          import RTLinkNodeMixin
from .rt_linknodeshortest_mixin  import RTLinkNodeShortestMixin
from .rt_ontologies_mixin        import RTOntologiesMixin
from .rt_panel_mixin             import RTPanelMixin
from .rt_periodic_barchart_mixin import RTPeriodicBarChartMixin
from .rt_piechart_mixin          import RTPieChartMixin
from .rt_shapes_mixin            import RTShapesMixin
from .rt_small_multiples_mixin   import RTSmallMultiplesMixin
from .rt_temporal_barchart_mixin import RTTemporalBarChartMixin
from .rt_text_mixin              import RTTextMixin
from .rt_timeline_mixin          import RTTimelineMixin
from .rt_wordcloud_mixin         import RTWordCloudMixin
from .rt_xy_mixin                import RTXYMixin

__name__ = 'rtsvg'

class RACETrack(RTAnnotationsMixin,
                RTArtMixin,
                RTBoxplotMixin,
                RTCalendarHeatmapMixin,
                RTChordDiagramMixin,
                RTChoroplethMapMixin,
                RTDataManipMixin,
                RTGeoMapsMixin,
                RTGeometryMixin,
                RTGraphLayoutsMixin,
                RTHistogramMixin,
                RTJSONMixin,
                RTLayoutsMixin,   
                RTLinkMixin,             
                RTLinkNodeMixin,
                RTLinkNodeShortestMixin,
                RTOntologiesMixin,
                RTPanelMixin,
                RTPeriodicBarChartMixin,
                RTPieChartMixin,
                RTShapesMixin,
                RTSmallMultiplesMixin,
                RTTemporalBarChartMixin,
                RTTextMixin,
                RTTimelineMixin,
                RTWordCloudMixin,
                RTXYMixin):
    #
    # Constructor (or whatever this is called in Python)
    #
    def __init__(self):
        # Visualization globals
        self.co_mgr            = RTColorManager(self)
        self.default_font      = "Times"
        self.fformat           = '0.2f' # label formatting
        
        # Field transformations
                                  #
                                  # Time-based transformations
                                  #
        self.transforms        = ['day_of_week',          # day of the week
                                  'day_of_week_hour',     # day of the week plus the hour of the day
                                  'year',                 # year
                                  'quarter',              # quarter
                                  'year_quarter',         # year and quarter
                                  'month',                # month
                                  'year_month',           # year and month
                                  'year_month_day',       # year, month, and day
                                  'year_month_day_hour',  # year, month, day, and hour
                                  'day',                  # day (of the month)
                                  'day_of_year',          # day of the year
                                  'day_of_year_hour',     # day of the year w/ hour
                                  'hour',                 # hour (of the day)
                                  'minute',               # minute (of the hour)
                                  'second',               # second (of the minute)
                                  #
                                  # Numeric transformations
                                  #
                                  'log_bins',             # log-based binning
                                  #
                                  # IP-based transformations
                                  #
                                  'ipv4_cidr_24',         # ipv4_cidr_24
                                  'ipv4_cidr_16',         # ipv4_cidr_16
                                  'ipv4_cidr_08'          # ipv4_cidr_08
                                  ]

        # Used for reflections
        self.widgets           = ['boxplot',
                                  'calendarHeatmap',
                                  'chordDiagram',
                                  'choroplethMap',
                                  'histogram',
                                  'linkNode',
                                  'panelControl',       # Control Panel for Panel Impl
                                  'periodicBarChart',
                                  'pieChart',
                                  'temporalBarChart',
                                  'xy']
        
        # Cache for converting strings to integers
        RACETrack.hashcode_lu  = {}
        
        # Inits for mixins...  probably a better way to do this...
        self.__annotations_mixin_init__()
        self.__graph_layouts_mixin_init__()
        self.__panel_mixin_init__()
        self.__periodic_barchart_mixin_init__()
        self.__temporal_barchart_mixin_init__()
        self.__text_mixin_init__()

    #
    # Render the SVG as HTML and display it within a notebook
    #
    def displaySVG(self,_svg):
        if type(_svg) is not str: _svg = _svg._repr_svg_()
        return display(HTML(_svg))

    #
    # Render the SVG as an Image and display it within a notebook
    # - Uses an in memory image buffer
    # - Image form should save processing power for complicated SVGs
    #
    def displaySVGAsImage(self, _svg):
        if type(_svg) is not str: _svg = _svg._repr_svg_()
        b = io.BytesIO()
        renderPM.drawToFile(svg2rlg(io.StringIO(_svg)), b, 'PNG')
        return ipc_display.Image(data=b.getvalue(),format='png',embed=True)

    #
    # isPandas() - is this a pandas dataframe?
    #
    def isPandas(self, df):
        return type(df) is pd.core.frame.DataFrame

    #
    # isPolars() - is this a polars dataframe?
    #
    def isPolars(self, df):
        return type(df) is pl.dataframe.frame.DataFrame

    #
    # copyDataFrame() - copy/clone a dataframe
    #
    def copyDataFrame(self, df):
        if   self.isPandas(df):
            return df.copy()
        elif self.isPolars(df):
            return df.clone()
        else:
            raise Exception('copyDataFrame() - accepts only pandas or polars dataframes')

    #
    # flattenTuple() - flatten a tuple
    #   flattenTuple(('fm','to'))                ==> ('fm', 'to')
    #   flattenTuple(('fm','to','other'))        ==> ('fm', 'to', 'other')
    #   flattenTuple(('a', ('b','c',('d','e')))) ==> ('a', 'b', 'c', 'd', 'e')
    #
    def flattenTuple(self, _tuple_):
        _ls_ = []
        for x in _tuple_:
            if type(x) is tuple:
                _ls_.extend(self.flattenTuple(x))
            else:
                _ls_.append(x)
        return tuple(_ls_)

    #
    # concatDataFrames() - concatenate dataframes
    #
    def concatDataFrames(self, dfs):
        if   self.isPandas(dfs[0]): return pd.concat(dfs)
        elif self.isPolars(dfs[0]): return pl.concat(dfs, how='diagonal_relaxed')
        else: raise Exception('concatDataFrames() - accepts only pandas or polars dataframes')

    #
    # createConcatColumn() - concatenate multiple columns together into a single column
    #
    def createConcatColumn(self, df, columns, new_column):
        def catFields(x, flds):
            s = str(x[flds[0]])
            for i in range(1,len(flds)):
                s += '|' + str(x[flds[i]])
            return s
        if self.isPandas(df):
            df[new_column] = df.apply(lambda x: catFields(x, columns), axis=1)
        elif self.isPolars(df):
            to_concat_new, str_casts = [], []
            for x in columns:
                if df[x].dtype != pl.String:
                    str_casts.append(pl.col(x).cast(str).alias('__' + x + '_as_str__'))
                    to_concat_new.append(pl.col('__' + x + '_as_str__'))
                else:
                    to_concat_new.append(pl.col(x))
            df = df.with_columns(*str_casts).with_columns(pl.concat_str(to_concat_new, separator='|').alias(new_column))
        else:
            raise Exception('createConcatColumn() - only pandas and polars supported')
        return df

    #
    # columnsAreTimestamps()
    # - Helper method to make a column into a suitable timestamp column
    # - ... because apparently the way to make a column into a type continues to evolve :(
    # - "format" parameter only applies to the polars typing method
    #
    def columnsAreTimestamps(self, df, columns, format=None):
        if type(columns) is not list:
            columns = [columns]
        for _column_ in columns:
            if   self.isPandas(df):
                try:
                    df[_column_] = df[_column_].astype('datetime64[ms]')
                except:
                    print("columnsAreTimestamps() - fail over conversion for datetime (pandas)")
                    df[_column_] = df[_column_].apply(lambda x: pd.to_datetime(x, utc=True).tz_convert(None))
            elif self.isPolars(df):
                try:
                    _format_ = format
                    if _format_ is None:
                        _format_ = self.guessTimestampFormat(str(df[_column_][0]))
                    if '.%f' in _format_: _format_ = _format_.replace('.%f', '%.f')
                    df = df.with_columns(pl.col(_column_).str.strptime(pl.Datetime, format=_format_).cast(pl.Datetime))
                except:
                    print(f"columnsAreTimestamps() - fail over conversion for datetime (polars) - example '{str(df[_column_][0])}'")
                    df = df.with_columns(pl.col(_column_).map_elements(lambda x: pd.to_datetime(x, utc=True).tz_convert(None), return_dtype=pl.Datetime))
                    #as_series = df[_column_].map_elements(lambda x: pd.to_datetime(x, utc=True).tz_convert(None), return_dtype=pl.Datetime)
                    #_format_  = self.guessTimestampFormat(str(as_series[0]))
                    #df = df.drop(_column_)
                    #df = df.with_columns(as_series.str.strptime(pl.Datetime, format=_format_).cast(pl.Datetime))
            else:
                raise Exception('columnsAreTimestamps() - not a pandas or polars dataframe')
        return df

    #
    # guessTimestampFormat()
    # ... slightly better code... but
    # ... i [still] hate timestamps
    #
    def guessTimestampFormat(self, sample):
        def endsWithOffset(s):
            def isNumber(c): return c >= '0' and c <= '9'
            if len(s) < 8: return False
            return isNumber(s[-1]) and \
                isNumber(s[-2]) and \
                s[-3] == ':'    and \
                isNumber(s[-4]) and \
                isNumber(s[-5]) and \
                (s[-6] == '-' or s[-6] == '+')
        def formatIsValid(format):
            try:
                datetime.strptime(sample, format)
                return True
            except:
                return False
        
        bases = [
            "%Y-%m-%dT%H:%M:%S.%f",
            "%m-%d-%YT%H:%M:%S", # will actually be with slashes...
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
            "%Y-%m-%dT%H",
            "%Y-%m-%d",
            "%Y-%m",
            "%Y"
        ]

        suffix = ''
        if   'z'   in sample: suffix = 'z'
        elif 'Z'   in sample: suffix = 'Z'
        elif 'UTC' in sample: suffix = ' UTC'
        elif 'utc' in sample: suffix = ' utc'
        elif 'GMT' in sample: suffix = ' GMT'
        elif 'gmt' in sample: suffix = ' gmt'
        elif endsWithOffset(sample): suffix = '%z'

        for base in bases:
            if '/' in sample:      base = base.replace('-','/')
            if ' ' in sample[:12]: base = base.replace('T',' ') # could be a space before timezone, so truncating it
            if (':' in base) ^ (':' in sample): continue       # both have to have a colon ... or neither
            base += suffix
            if formatIsValid(base): return base

        raise Exception(f'guessTimestampFormat() - no format specified for sample "{sample}"')

    #
    # guessTimestampField() - guess the timestamp field
    #
    def guessTimestampField(self, df):
        if   self.isPandas(df):
            return self.__guessTimestampField_pandas__(df)
        elif self.isPolars(df):
            return self.__guessTimestampField_polars__(df)
        else:
            raise Exception(f'guessTimestampField() - only handles pandas and polars')
    def __guessTimestampField_pandas__(self, df):
        choices = df.select_dtypes(np.datetime64).columns
        if len(choices) == 1:
            return choices[0]
        elif len(choices) > 1:
            print('multiple timestamp fields... choosing the first (__guessTimestampField_pandas__)')
            return choices[0]
        else:
            raise Exception('no timestamp field supplied, cannot automatically determine field (__guessTimestampField_pandas__)')
    def __guessTimestampField_polars__(self, df):
        just_dt_columns = df.select(pl.col(pl.Datetime('us')))
        if len(just_dt_columns) == 0:
            just_dt_columns = df.select(pl.col(pl.Datetime('ns')))
        if len(just_dt_columns) == 0:
            just_dt_columns = df.select(pl.col(pl.Datetime('ms')))
        if   len(just_dt_columns.columns) > 1:
            print('multiple timestamp fields... choosing the first (__guessTimestampField_polars__)')
            return just_dt_columns.columns[0]
        elif len(just_dt_columns.columns) == 1:
            return just_dt_columns.columns[0]
        else:
            raise Exception('no timestamp field supplied, cannot automatically determine field (__guessTimestampField_polars__)')

    #
    # Return a consistent hashcode for a string
    #
    def hashcode(self,s):
        if type(s) is not str: # Force non-strings to be strings
            s = str(s)
        if s not in RACETrack.hashcode_lu.keys(): # Cache the results so that we don't have to redo the calculation
            my_bytes = hashlib.sha256(s.encode('utf-8')).digest()
            value = ((my_bytes[0]<<24)&0x00ff000000) | ((my_bytes[1]<<16)&0x0000ff0000) | \
                    ((my_bytes[2]<< 8)&0x000000ff00) | ((my_bytes[3]<< 0)&0x00000000ff)
            RACETrack.hashcode_lu[s] = value
        return RACETrack.hashcode_lu[s]

    #
    # Encode a string into something safe for racetrack
    # ... in general, this code base uses pipes to separate strings... so it needs to be safe for that at least...
    #
    def stringEncode(self,s):
        return urllib.parse.quote_plus(s)
    
    #
    # Decode a string that was encoded with stringEncode()
    #
    def stringDecode(self,s):
        return urllib.parse.unquote_plus(s)

    #
    # Encode a string to make a valid SVG ID.
    # ... uses a colon escape sequence to encode any non [a-zA-Z0-9 ]
    #
    # From:  "https://www.dofactory.com/html/svg/id":
    #  "A unique alphanumeric string. The id value must begin with a letter ([A-Za-z]) and may be followed by 
    #   any number of letters, digits ([0-9]), hyphens (-), underscores (_), colons (:), and periods (.)."
    #
    def encSVGID(self, s):
        _enc = 'encsvgid_'
        if   type(s) is int:
            _enc += f'i_{s}'
            
        elif type(s) is str:        
            _enc += 's_'
            for c in s:
                if (c >= 'a' and c <= 'z') or \
                   (c >= 'A' and c <= 'Z') or \
                   (c >= '0' and c <= '9'):
                   _enc += c
                elif c == ' ':
                    _enc += '_'
                else:
                    as_int = ord(c)
                    _enc += ':'+str(as_int)+':'
        else:
            raise Exception('rtsvg.encSVGID() -- only strings and ints supported')
        return _enc

    #
    # Decode a string that was created by the encSVGID() method.
    #
    def decSVGID(self, s):
        if   s.startswith('encsvgid_i_'):
            s_prime = s[len('encsvgid_s_'):]
            return int(s_prime)
        elif s.startswith('encsvgid_s_'):
            s_prime = s[len('encsvgid_s_'):]
            _dec    = ''
            i = 0
            while i < len(s_prime):
                c = s_prime[i]
                if (c >= 'a' and c <= 'z') or \
                   (c >= 'A' and c <= 'Z') or \
                   (c >= '0' and c <= '9'):
                    _dec += c
                    i += 1
                elif c == '_':
                    _dec += ' '
                    i += 1
                elif c == ':':
                    i += 1
                    int_str = ''
                    while i < len(s_prime) and s_prime[i] != ':':
                        int_str += s_prime[i]
                        i += 1
                    _dec += chr(int(int_str))
                    i += 1
                else:
                    raise Exception(f'decSVGID() - failed to decode "{s}"')
            return _dec
        else:
            raise Exception(f'decSVGID() - unknown encoding type ... should be "_i_" or "_s_" "{s}"')

    # ****************************************************************************************************************
    #
    # Transformation Section
    #
    # ****************************************************************************************************************

    #
    # Transform a list of fields
    # - only handles one level of nesting for lists
    #
    def transformFieldListAndDataFrame(self, df, field_list):
        # Perform the transforms
        new_field_list = []
        for x in field_list:
            if type(x) is list:
                new_list = []
                for y in x:
                    if self.isTField(y) and y not in df.columns:
                        df,new_y = self.applyTransform(df,y)
                        new_list.append(new_y)
                    else:
                        new_list.append(y)
                new_field_list.append(new_list)
            else:
                if self.isTField(x) and x not in df.columns:
                    df,new_x = self.applyTransform(df, x)
                    new_field_list.append(new_x)
                else:
                    new_field_list.append(x)
        return df, new_field_list

    #
    # Determine if a field is a tfield
    #
    def isTField(self,tfield):
        return tfield is not None and type(tfield) is str and tfield.startswith('|tr|')      
    
    #
    # Return the applicable field for this transformation field (tfiled)
    #
    def tFieldApplicableField(self,tfield):
        if self.isTField(tfield):
            return '|'.join(tfield.split('|')[3:])
        return None
        
    #
    # Apply a tranformation field (tfield) to a dataframe and return the new dataframe and the calculated new field
    # ... we'll want set-based counting -- so we'll make sure it's never just a number
    #
    def applyTransform(self, df, tfield):
        if tfield is not None and tfield.startswith('|tr|') and tfield not in df.columns:
            transform = tfield.split('|')[2]
            field     = '|'.join(tfield.split('|')[3:])

            ipv4CIDR08 = lambda ipv4: ipv4.split('.')[0]
            ipv4CIDR16 = lambda ipv4: ipv4.split('.')[0] + '.' + ipv4.split('.')[1]
            ipv4CIDR24 = lambda ipv4: ipv4.split('.')[0] + '.' + ipv4.split('.')[1] + '.' + ipv4.split('.')[2]

            #
            # PANDAS Version
            #
            if self.isPandas(df):            
                if   transform == 'day_of_week':
                    df[tfield] = df[field].apply(lambda x: str(x.day_name()[:3]))
                elif transform == 'day_of_week_hour':
                    df[tfield] = df[field].apply(lambda x: f'{x.day_name()[:3]}-{x.hour:02}')
                elif transform == 'year':
                    df[tfield] = df[field].apply(lambda x: str(x.year))
                elif transform == 'year_quarter':
                    df[tfield] = df[field].apply(lambda x: f'{x.year}Q{x.quarter}')
                elif transform == 'quarter':
                    df[tfield] = df[field].apply(lambda x: f'Q{x.quarter}')
                elif transform == 'month':
                    df[tfield] = df[field].apply(lambda x: x.month_name()[:3])
                elif transform == 'year_month':
                    df[tfield] = df[field].apply(lambda x: f'{x.year}-{x.month:02}')
                elif transform == 'year_month_day':
                    df[tfield] = df[field].apply(lambda x: f'{x.year}-{x.month:02}-{x.day:02}')
                elif transform == 'year_month_day_hour':
                    df[tfield] = df[field].apply(lambda x: f'{x.year}-{x.month:02}-{x.day:02} {x.hour:02}')
                elif transform == 'day':
                    df[tfield] = df[field].apply(lambda x: f'{x.day:02}')
                elif transform == 'day_of_year':
                    df[tfield] = df[field].apply(lambda x: f'{x.day_of_year:03}')
                elif transform == 'day_of_year_hour':
                    df[tfield] = df[field].apply(lambda x: f'{x.day_of_year:03}_{x.hour:02}')
                elif transform == 'hour':
                    df[tfield] = df[field].apply(lambda x: f'{x.hour:02}')
                elif transform == 'minute':
                    df[tfield] = df[field].apply(lambda x: f'{x.minute:02}')
                elif transform == 'second':
                    df[tfield] = df[field].apply(lambda x: f'{x.second:02}')
                elif transform == 'log_bins':
                    df[tfield] = df[field].apply(lambda x: self.transformLogBins(x))
                elif transform == 'ipv4_cidr_24': # ipv4_cidr_24
                    df[tfield] = df[field].apply(lambda x: ipv4CIDR24(x))
                elif transform == 'ipv4_cidr_16': # ipv4_cidr_16
                    df[tfield] = df[field].apply(lambda x: ipv4CIDR16(x))
                elif transform == 'ipv4_cidr_08': # ipv4_cidr_08
                    df[tfield] = df[field].apply(lambda x: ipv4CIDR08(x))

            #
            # POLARS Version
            #            
            elif self.isPolars(df):
                if   transform == 'day_of_week':
                    df = df.with_columns(pl.col(field).dt.strftime('%A').str.slice(0,3).alias(tfield))
                elif transform == 'day_of_week_hour':
                    _intermediate_dow_  = '__day_of_week_hour_dow__'
                    df = df.with_columns(pl.col(field).dt.strftime('%A').str.slice(0,3).alias(_intermediate_dow_))
                    _intermediate_hour_ = '__day_of_week_hour_hour__'
                    df = df.with_columns(pl.col(field).dt.strftime('-%H').alias(_intermediate_hour_))
                    df = df.with_columns(pl.concat_str(_intermediate_dow_, _intermediate_hour_).alias(tfield))
                elif transform == 'year':
                    # df = df.with_columns(pl.col(field).dt.strftime('%Y').cast(pl.Int64).alias(tfield))
                    df = df.with_columns(pl.col(field).dt.strftime('%Y').cast(pl.Int64).alias(tfield))
                elif transform == 'year_quarter':
                    _intermediate_quarter_ = '__quarter__'                    
                    df = df.with_columns(pl.col(field).dt.quarter().cast(str).alias(_intermediate_quarter_))
                    _intermediate_year_    = '__year__'
                    df = df.with_columns(pl.col(field).dt.strftime('%YQ').alias(_intermediate_year_))
                    df = df.with_columns(pl.concat_str(_intermediate_year_, _intermediate_quarter_).alias(tfield))
                elif transform == 'quarter':
                    _intermediate_quarter_ = '__quarter__'
                    df = df.with_columns(pl.col(field).dt.quarter().cast(str).alias(_intermediate_quarter_))
                    df = df.with_columns(pl.format('Q{}', _intermediate_quarter_).alias(tfield))
                elif transform == 'month':
                    df = df.with_columns(pl.col(field).dt.strftime('%h').alias(tfield))
                elif transform == 'year_month':
                    df = df.with_columns(pl.col(field).dt.strftime('%Y-%m').alias(tfield))
                elif transform == 'year_month_day':
                    df = df.with_columns(pl.col(field).dt.strftime('%Y-%m-%d').alias(tfield))
                elif transform == 'year_month_day_hour':
                    df = df.with_columns(pl.col(field).dt.strftime('%Y-%m-%d %H').alias(tfield))
                elif transform == 'day':
                    # df = df.with_columns(pl.col(field).dt.strftime('%d').cast(pl.Int64).alias(tfield))
                    df = df.with_columns(pl.col(field).dt.strftime('%d').alias(tfield))
                elif transform == 'day_of_year':
                    # df = df.with_columns(pl.col(field).dt.strftime('%j').cast(pl.Int64).alias(tfield))
                    df = df.with_columns(pl.col(field).dt.strftime('%j').alias(tfield))
                elif transform == 'day_of_year_hour':
                    df = df.with_columns(pl.col(field).dt.strftime('%j_%H').alias(tfield))
                elif transform == 'hour':
                    # df = df.with_columns(pl.col(field).dt.strftime('%H').cast(pl.Int64).alias(tfield))
                    df = df.with_columns(pl.col(field).dt.strftime('%H').alias(tfield))
                elif transform == 'minute':
                    # df = df.with_columns(pl.col(field).dt.strftime('%M').cast(pl.Int64).alias(tfield))
                    df = df.with_columns(pl.col(field).dt.strftime('%M').alias(tfield))
                elif transform == 'second':
                    # df = df.with_columns(pl.col(field).dt.strftime('%S').cast(pl.Int64).alias(tfield))
                    df = df.with_columns(pl.col(field).dt.strftime('%S').alias(tfield))
                elif transform == 'log_bins':
                    df = df.with_columns(pl.col(field).apply(lambda x: self.transformLogBins(x)).alias(tfield))
                elif transform == 'ipv4_cidr_24': # ipv4_cidr_24
                    df = df.with_columns(pl.col(field).map_elements(lambda x: ipv4CIDR24(x), return_dtype=pl.String).alias(tfield))
                elif transform == 'ipv4_cidr_16': # ipv4_cidr_16
                    df = df.with_columns(pl.col(field).map_elements(lambda x: ipv4CIDR16(x), return_dtype=pl.String).alias(tfield))
                elif transform == 'ipv4_cidr_08': # ipv4_cidr_08
                    df = df.with_columns(pl.col(field).map_elements(lambda x: ipv4CIDR08(x), return_dtype=pl.String).alias(tfield))
            else:
                raise Exception('applyTransform() - df is neither a pandas nor a polars dataframe')

        return df,tfield

    #
    # Define the natural order for the elements returned by a tfield
    # - a few degenerate cases exist -- for example, year_month_day if give many centuries...
    #
    def transformNaturalOrder(self, df, tfield):
        _order = []
        _order_dow     = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        _order_quarter = ['Q1','Q2','Q3','Q4']
        _order_month   = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        if tfield is not None and tfield.startswith('|tr|'):
            transform = tfield.split('|')[2]
            field     = '|'.join(tfield.split('|')[3:])
            if   transform == 'day_of_week':
                _order = _order_dow
            elif transform == 'day_of_week_hour':
                for _dow in _order_dow:
                    for _hour in range(0,24):
                        _order.append(f'{_dow}-{_hour:02}')
            elif transform == 'year':
                for _year in range(df[field].min().year, df[field].max().year+1):
                    _order.append(f'{_year}')
            elif transform == 'year_quarter':
                for _year in range(df[field].min().year, df[field].max().year+1):
                    for _quarter in range(1,5):
                        _order.append(f'{_year}Q{_quarter}')
            elif transform == 'quarter':
                _order = _order_quarter
            elif transform == 'month':
                _order = _order_month
            elif transform == 'year_month':
                for _date in pd.date_range(start=df[field].min(), end=df[field].max(), freq='M'):
                    _order.append(f'{_date.year}-{_date.month:02}')
            elif transform == 'year_month_day':
                for _date in pd.date_range(start=df[field].min(), end=df[field].max(), freq='D'):
                    _order.append(f'{_date.year}-{_date.month:02}-{_date.day:02}')
            elif transform == 'year_month_day_hour':
                for _date in pd.date_range(start=df[field].min(), end=df[field].max(), freq='H'):
                    _order.append(f'{_date.year}-{_date.month:02}-{_date.day:02} {_date.hour:02}')
            elif transform == 'day':
                for _day in range(1,32):
                    _order.append(f'{_day:02}')
            elif transform == 'day_of_year':
                if df[field].min().year == df[field].max().year:
                    _day_min, _day_max = pd.to_datetime(df[field].min()).day_of_year, pd.to_datetime(df[field].max()).day_of_year
                    for _day in range(_day_min,_day_max+1):
                        _order.append(f'{_day:03}')
                else:
                    for _day in range(1,366):
                        _order.append(f'{_day:03}')
            elif transform == 'day_of_year_hour':
                if df[field].min().year == df[field].max().year:
                    _day_min, _day_max = pd.to_datetime(df[field].min()).day_of_year, pd.to_datetime(df[field].max()).day_of_year                    
                    for _day in range(_day_min,_day_max+1):
                        for _hour in range(0,24):
                            _order.append(f'{_day:03}_{_hour:02}')
                else:
                    for _day in range(1,366):
                        for _hour in range(0,24):
                            _order.append(f'{_day:03}_{_hour:02}')
            elif transform == 'hour':
                for _hour in range(0,24):
                    _order.append(f'{_hour:02}')
            elif transform == 'minute' or transform == 'second':
                for _minute in range(0,60):
                    _order.append(f'{_minute:02}')
            elif transform == 'log_bins':
                _order = ['< 0', '= 0', '<= 1', '<= 10','<= 100', '<= 1K', '<= 100K', '<= 1M', '> 1M']
            elif transform == 'ipv4_cidr_24':
                _pad_ = lambda x: f'{int(x.split(".")[0]):03}.{int(x.split(".")[1]):03}.{int(x.split(".")[2]):03}'
                _order = list(set(df[tfield]))
                _order.sort(key=_pad_)
            elif transform == 'ipv4_cidr_16':
                _pad_ = lambda x: f'{int(x.split(".")[0]):03}.{int(x.split(".")[1]):03}'
                _order = list(set(df[tfield]))
                _order.sort(key=_pad_)
            elif transform == 'ipv4_cidr_08':
                _pad_ = lambda x: f'{int(x.split(".")[0]):03}'
                _order = list(set(df[tfield]))
                _order.sort(key=_pad_)
        return _order

    #
    # Create a tranformation field (tfield)
    #
    def createTField(self,field,trans):
        ''' Create a Transformation Field (tfield)  -- print rt.transforms to see option.

            Parameters
            ----------
            field : str
                The field to be transformed
            trans : str
                The transformation to be applied to the field
                See rt.transforms
        '''
        if trans in self.transforms:
            tfield = '|tr|'+trans+'|'+field
        else:
            raise Exception(f'Transform "{trans}" is not defined')
        return tfield

    #
    # Make simple log-based bins
    # - strings are equivalent to a color scheme in RTColorManager class.
    #
    def transformLogBins(self, x):
        x = float(x)
        if   x < 0:
            return '< 0'
        elif x == 0:
            return '= 0'
        elif x <= 1:
            return '<= 1'
        elif x <= 10:
            return '<= 10'
        elif x <= 100:
            return '<= 100'
        elif x <= 1000:
            return '<= 1K'
        elif x <= 100000:
            return '<= 100K'
        elif x <= 1000000:
            return '<= 1M'
        else:
            return '> 1M'

    #
    # Identify columns needed from widget parameters
    #
    def identifyColumnsFromParameters(self, param_name, param_lu, columns_set):
        # print(f'identifyColumnsFromParameters(,"{param_name}","{param_lu}","{columns_set}")') # DEBUG
        if param_name in param_lu.keys() and param_lu[param_name] is not None:
            v = param_lu[param_name]
            self.__recursiveDecompose__(v, columns_set)

    def __recursiveDecompose__(self, something, columns_set):
        if   type(something) is str:
            columns_set.add(something)
        elif type(something) is bool: # unclear why None may be converted to False // is that what's happening?
            pass # do nothing
        elif type(something) is list or type(something) is tuple:
            for x in something:
                self.__recursiveDecompose__(x, columns_set)
        elif type(something) is dict:
            pass # do nothing
        else:
            raise Exception(f'Unknown type ("{type(something)}") for ("{something}") encountered in identifyColumnsFromParameters()')

    #
    # polarsFilterColumnsWithNaNs() -- filter specified columns with NaN values
    #
    def polarsFilterColumnsWithNaNs(self, df, cols):
        _eval_ = []
        for col in cols:
            _eval_.append(f'(pl.col("{col}").is_not_null())')
        df = df.filter(eval('&'.join(_eval_)))
        return df

    #
    # polarsCounter() -- return a dataframe with fields and an __count__ column.
    #
    def polarsCounter(self, df, fields, count_by=None, count_by_set=False):
        fields = [fields] if type(fields) is not list else fields
        if count_by is not None and count_by_set == False:
            if self.fieldIsArithmetic(df, count_by) == False:
                count_by_set = True
        if count_by is None:
            return df.group_by(fields).agg(pl.len().alias('__count__'))
        elif count_by_set and count_by in fields:
            df_min = df.drop(set(df.columns) - set(fields) - set([count_by]))
            df_dupe = df_min.with_columns(pl.col(count_by).alias('__count__'))
            return df_dupe.group_by(fields).n_unique()
        elif count_by_set:
            df_min = df.drop(set(df.columns) - set(fields) - set([count_by])).rename({count_by:'__count__'})
            return df_min.group_by(fields).n_unique()
        elif count_by in fields:
            df_min = df.drop(set(df.columns) - set(fields) - set([count_by]))
            df_dupe = df_min.with_columns(pl.col(count_by).alias('__count__'))
            return df_dupe.group_by(fields).sum()
        else:
            df_min = df.drop(set(df.columns) - set(fields) - set([count_by])).rename({count_by:'__count__'})
            return df_min.group_by(fields).sum()

    #
    # Determine If A Column Has To Be Counted By Set Operation
    #
    def countBySet(self, 
                   df,         # dataframe
                   count_by):  # field to check
        if count_by is None:
            return False
        if type(df) is list:
            for _df in df:
                if self.isPandas(_df):
                    if count_by in _df.columns:
                        if _df[count_by].dtypes != np.int64   and \
                           _df[count_by].dtypes != np.int32   and \
                           _df[count_by].dtypes != np.float64 and \
                           _df[count_by].dtypes != np.float32:
                                return True
                elif self.isPolars(_df):
                    if _df[count_by].is_float() == False and _df[count_by].is_integer() == False:
                        return True
                else:
                    raise Exception('countBySet() - not a pandas or polars dataframe')
            return False
        else:
            if self.isPandas(df):
                return df[count_by].dtypes != np.int64   and \
                       df[count_by].dtypes != np.int32   and \
                       df[count_by].dtypes != np.float64 and \
                       df[count_by].dtypes != np.float32
            elif self.isPolars(df):
                return df[count_by].dtype.is_float() == False and df[count_by].dtype.is_integer() == False
            else:
                raise Exception('countBySet() - not a pandas or polars dataframe')
    
    #
    # fieldIsArithmetic()
    # ... determine if a field can be operated on by arithmetic
    # ... maybe the oposite of the above?
    #
    def fieldIsArithmetic(self, df, field):
        if self.isPandas(df):
            return df[field].dtypes == np.int64   or \
                   df[field].dtypes == np.int32   or \
                   df[field].dtypes == np.float64 or \
                   df[field].dtypes == np.float32
        elif self.isPolars(df):
            return df[field].dtype.is_float() or df[field].dtype.is_integer()
        else:
            raise Exception('fieldIsArithmetic() - not a pandas or polars dataframe')

    #
    # Determine color ordering based on quantity
    #
    def colorRenderOrder(self, 
                         df,                     # dataframe
                         color_by,               # color_by field
                         count_by     = None,    # count_by field (None is equivalent to counting by rows)
                         count_by_set = False):  # for the field set, count by set operation
        '''Produce a structure indicating the order in which to colorize a visualization.'''
        # If no color, then return none...
        if color_by is None: return None
        
        # Make sure we can count by numeric summation
        if count_by_set == False: count_by_set = self.countBySet(df, count_by)
        
        # Do the counting based on the dataframe type
        if   self.isPandas(df): return self.__colorQuantities_pandas__(df, color_by, count_by, count_by_set)
        elif self.isPolars(df): return self.__colorQuantities_polars__(df, color_by, count_by, count_by_set)
        else:                   raise Exception('colorRenderOrder() - not a pandas or polars dataframe')

    #
    # Determine color quantities
    #
    def __colorQuantities_pandas__(self, 
                                   df,            # dataframe 
                                   color_by,      # color_by field
                                   count_by,      # count_by field
                                   count_by_set): # for the field set, count by set operation
        # For count by set... when count_by == color by... then we'll count by rows
        if count_by is not None and count_by_set and count_by == color_by:
            count_by = None

        if count_by is None:
            return df.groupby(color_by).size().sort_values(ascending=False)
        elif count_by_set:
            _df = pd.DataFrame(df.groupby([color_by,count_by]).size()).reset_index()
            return _df.groupby(color_by).size().sort_values(ascending=False)
        elif count_by == color_by:
            _df = df.groupby(color_by).size().reset_index()
            _df['__mult__'] = _df.apply(lambda x: x[color_by]*x[0],axis=1)
            return _df.groupby(color_by)['__mult__'].sum().sort_values(ascending=False)
        else:
            return df.groupby(color_by)[count_by].sum().sort_values(ascending=False)

    #
    # Determine color quantities
    # ... polars has no concept of index... so this isn't going to be a 1-for-1 method...
    # ... i hope there isn't a columns named 'count_by' or 'color_by'
    #
    def __colorQuantities_polars__(self, 
                                   df,            # dataframe 
                                   color_by,      # color_by field
                                   count_by,      # count_by field
                                   count_by_set): # for the field set, count by set operation
        # For count by set... when count_by == color by... then we'll count by rows
        if count_by is not None and count_by_set and count_by == color_by:
            count_by = None

        # Remove all the columns that aren't used for the operation...
        # ... and rename them to 'count_by' and 'color_by'
        if count_by is not None:
            _to_drop_ = list(df.columns)
            _to_drop_.remove(color_by)
            if count_by in _to_drop_:
                _to_drop_.remove(count_by)
            df = df.drop(_to_drop_)
            if color_by != count_by:
                df = df.rename({color_by:'color_by', count_by:'count_by'})
                color_by = 'color_by'
                count_by = 'count_by'
            else:
                df = df.rename({color_by:'color_by'})
                color_by = count_by = 'color_by'
        else:
            _to_drop_ = list(df.columns)
            _to_drop_.remove(color_by)
            df = df.drop(_to_drop_)
            df = df.rename({color_by:'color_by'})
            color_by = 'color_by'

        if count_by is None:            
            _df_ = df.group_by(color_by).agg(pl.len().alias('count')) # .sort('count', descending=True)
        elif count_by_set:
            _df_ = df.group_by(color_by).n_unique() # .sort(count_by,descending=True)
        elif count_by == color_by:
            _df_ = df.group_by(color_by).agg(pl.len().alias('count'))
            _df_ = _df_.with_columns(pl.col(color_by).mul(pl.col('count')).alias('__mult__'))
            _df_ = _df_.sort('__mult__',descending=True)
            _df_ = _df_.drop('count')
            _df_ = _df_.rename({'__mult__':'count'})
        else:
            _df_ = df.group_by(color_by).agg(pl.sum(count_by)) # .sort(count_by, descending=True)

        if 'count_by' in set(_df_.columns):
            _df_ = _df_.rename({'count_by':'count'})
        _df_ = _df_.rename({'color_by':'index'})

        _df_ = _df_.sort('index').sort('count', descending=True)
        
        return _df_
    
    #
    # Colorize Bar
    #
    def colorizeBar(self,
                    df,                  # dataframe
                    global_color_order,  # global color ordering - returned from colorRenderOrder()
                    color_by,            # color_by field
                    count_by,            # count_by field
                    count_by_set,        # for the field set, count by set operation
                    x,                   # x coordinate of the bar base
                    y,                   # y coordinate of the bar base
                    bar_len,             # total bar length -- for vertical, this is the height
                    bar_sz,              # size of bar -- for vertical, this is the width
                    horz):               # true for horizontal bars (histogram), false for vertical bars
        if   self.isPandas(df):
            return self.__colorizeBar_pandas__(df, global_color_order, color_by, count_by, count_by_set, x, y, bar_len, bar_sz, horz)
        elif self.isPolars(df):
            return self.__colorizeBar_polars__(df, global_color_order, color_by, count_by, count_by_set, x, y, bar_len, bar_sz, horz)
        else:
            raise Exception('colorizeBar() - not a pandas or polars dataframe')

    def __colorizeBar_pandas__(self,
                               df,                  # dataframe
                               global_color_order,  # global color ordering - returned from colorRenderOrder()
                               color_by,            # color_by field
                               count_by,            # count_by field
                               count_by_set,        # for the field set, count by set operation
                               x,                   # x coordinate of the bar base
                               y,                   # y coordinate of the bar base
                               bar_len,             # total bar length -- for vertical, this is the height
                               bar_sz,              # size of bar -- for vertical, this is the width
                               horz):               # true for horizontal bars (histogram), false for vertical bars
        svg = []
        if bar_len > 0:
            _co = self.co_mgr.getTVColor('data','default')
            # Default bar w/out color
            if horz:
                svg.append(f'<rect x="{x}" y="{y}" width="{bar_len}" height="{bar_sz}" fill="{_co}" />')
            else:
                svg.append(f'<rect x="{x}" y="{y-bar_len}" width="{bar_sz}" height="{bar_len}" fill="{_co}" />')
            # Colorize it
            if color_by is not None:
                quantities   = self.colorRenderOrder(df, color_by, count_by, count_by_set)
                value        = quantities.sum()
                quantities   = quantities[quantities > value/bar_len]
                intersection = self.__myIntersection__(global_color_order.index, quantities.index)
                d = x if horz else y
                for cb_bin in intersection:
                    v = quantities[cb_bin]
                    l = bar_len * v / value
                    if l >= 1.0:
                        _co = self.co_mgr.getColor(cb_bin)
                        if horz:
                            svg.append(f'<rect x="{d}" y="{y}" width="{l}" height="{bar_sz}" fill="{_co}" />')
                            d += l
                        else:
                            svg.append(f'<rect x="{x}" y="{d-l}" width="{bar_sz}" height="{l}" fill="{_co}" />')
                            d -= l
        return ''.join(svg)

    def __colorizeBar_polars__(self,
                               df,                  # dataframe
                               global_color_order,  # global color ordering - returned from colorRenderOrder()
                               color_by,            # color_by field
                               count_by,            # count_by field
                               count_by_set,        # for the field set, count by set operation
                               x,                   # x coordinate of the bar base
                               y,                   # y coordinate of the bar base
                               bar_len,             # total bar length -- for vertical, this is the height
                               bar_sz,              # size of bar -- for vertical, this is the width
                               horz):               # true for horizontal bars (histogram), false for vertical bars
        svg = []
        if bar_len > 0:
            _co = self.co_mgr.getTVColor('data','default')
            # Default bar w/out color
            if horz:
                svg.append(f'<rect x="{x}" y="{y}" width="{bar_len}" height="{bar_sz}" fill="{_co}" />')
            else:
                svg.append(f'<rect x="{x}" y="{y-bar_len}" width="{bar_sz}" height="{bar_len}" fill="{_co}" />')
            # Colorize it
            if color_by is not None:
                quantities   = self.colorRenderOrder(df, color_by, count_by, count_by_set)
                value        = quantities.sum()['count'][0]
                quantities   = quantities.filter(pl.col('count') > value/bar_len)
                intersection = self.__myIntersection__(global_color_order['index'], quantities['index'])
                d = x if horz else y
                for cb_bin in intersection:
                    v = quantities.filter(pl.col('index') == cb_bin)['count'][0]
                    l = bar_len * v / value
                    if l >= 1.0:
                        _co = self.co_mgr.getColor(cb_bin)
                        if horz:
                            svg.append(f'<rect x="{d}" y="{y}" width="{l}" height="{bar_sz}" fill="{_co}" />')
                            d += l
                        else:
                            svg.append(f'<rect x="{x}" y="{d-l}" width="{bar_sz}" height="{l}" fill="{_co}" />')
                            d -= l
        return ''.join(svg)

    #
    # From https://www.geeksforgeeks.org/python-intersection-two-lists/
    #
    def __myIntersection__(self, lst1, lst2):
        temp = set(lst2)
        lst3 = [value for value in lst1 if value in temp]
        return lst3

    # Doesn't understand duplicates...
    #my_list_1 = [1, 2, 3, 10, 11, 12, 15, 18, 20, 20, 20]  # This ordering is kept
    #my_list_2 = [20, 0, 0, 0,  1, 11,  2,  7,  9, 100, 20]
    #intersection(my_list_1, my_list_2)

    #
    # svgText() - Render SVG Text In A Consistent Manner
    #
    def svgText(self,
                txt,
                x,
                y,
                txt_h    = 12,
                just_xy  = False,   # for the text block widget -- that will use an SVG group to consolidate the rendering
                color    = None,
                anchor   = 'start',
                font     = None,
                rotation = None):
        if txt == '\n' or txt == '' or txt == '\r' or txt == '\t': return ''
        if font  is None: font  = self.default_font
        if color is None: color = self.co_mgr.getTVColor('label','defaultfg')
        txt = str(txt)

        _html_txt = html.escape(txt)
        # The following breaks JupyterLab in some configs ... 2024-01-04
        #if ' ' in _html_txt:
        #    _html_txt = _html_txt.replace(' ','&nbsp;')

        if   just_xy:
            return f'<text x="{x:0.2f}" y="{y:0.2f}">{_html_txt}</text>'
        elif rotation is not None:
            return f'<text x="{x}" text-anchor="{anchor}" y="{y}" font-family="{font}" fill="{color}" font-size="{txt_h}px"' + \
                   f' transform="rotate({rotation},{x},{y})">{_html_txt}</text>'
        else:
            return f'<text x="{x}" text-anchor="{anchor}" y="{y}" font-family="{font}" fill="{color}" font-size="{txt_h}px">{_html_txt}</text>'

    #
    # Empirically-derived font metrics -- see the next javascript code block on the initial generation of these numbers
    # ... this is really just a starting point
    # ... looks like correct for visual studio code with a txt_h of 19.5 // 2023-02-05
    # ... updated on 2023-06-28 to load them from a file...
    #
    _font_metrics_ = None

    #
    # Javascript used to generate the above... with a little bit of editing for the results to fit into a dictionary (copied from JS Console)...
    # ... used at https://jsfiddle.net
    #
    _font_metrics_js_ = """
let str = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
svg = "<svg width=\"256\" height=\"256\">"
for (i=0;i<str.length;i++) {
	 svg += "<text id=\"_test_" + str[i] + "\" x=\"50\" y=\"50\">" + str[i] + "</text>"
}
svg += "</svg>"
document.write(svg)

for (i=0;i<str.length;i++) {
	let elem = document.getElementById("_test_" + str[i]);
	let rect = elem.getBoundingClientRect();
	console.log("\'" + str[i] + "\':" + rect.width)
}
"""

    #
    # 2023-06-28 // Updated javascript that includes the first 4K unicode characters as well...
    # ... however, it doesn't fill in the space character... which is integer value 32
    #
    _font_metrics_unicode_js_ = """
function toHexFour(i) {
	my_s = i.toString(16)
  while (my_s.length < 4) { my_s = "0" + my_s; }
  return my_s
}
let sz = 16;
let s  = '';
for (i=32;i<4096;i++) {
	if (i==32) c = '&nbsp;';
  else       c = '&#x' + toHexFour(i) + ';';
	s = s + '<text id="txt' + i + '" x="5" y="32" font-family="ariel" font-size="'+sz+'px">' + c + '</text>';
}
document.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>' +
               '<svg width="64" height="64">' + s + '</svg>');
for (i=32;i<4096;i++) {
	let elem = document.getElementById("txt"+i);
	let rect = elem.getBoundingClientRect();
	console.log("" + i + "," + sz + "," + rect.width);
}
"""

    #
    # This version checks 4096 unicode characters (as above)... and then 
    # does all pairs from 32...128 // but it needs to be more... and even
    # with these limits, takes forever... lots of memory.
    #
    _font_metrics_unicode_kern_js_ = """
function toHexFour(i) {
	my_s = i.toString(16)
  while (my_s.length < 4) { my_s = "0" + my_s; }
  return my_s
}
let sz = 16;
let s  = '';
for (i=32;i<4096;i++) {
	if (i==32) c = '&nbsp;';
  else       c = '&#x' + toHexFour(i) + ';';
	s += '<text id="txt' + i + '" x="5" y="32" font-family="ariel" font-size="'+sz+'px">' + c + '</text>';
}

for (i=32;i<128;i++) {
	if (i==32) c = '&nbsp;';
  else       c = '&#x' + toHexFour(i) + ';';
	for (j=32;j<128;j++) {
		if (j==32) d = '&nbsp;';
  	else       d = '&#x' + toHexFour(j) + ';';
		s += '<text id="txt' + i + '_' + j + '" x="5" y="32" font-family="ariel" font-size="'+sz+'px">' + c + d + '</text>';
  }
}
document.write('<?xml version="1.0" encoding="UTF-8" standalone="no"?>' +
               '<svg width="64" height="64">' + s + '</svg>');

const char_to_w = new Map();
for (i=32;i<4096;i++) {
	let elem = document.getElementById("txt"+i);
	let rect = elem.getBoundingClientRect();
  char_to_w[i] = rect.width;
  console.log("" + i + "," + rect.width)
}

for (i=32;i<128;i++) {
	for (j=32;j<128;j++) {
  	let elem = document.getElementById("txt"+i+"_"+j);
	  let rect = elem.getBoundingClientRect();
    if (rect.width != (char_to_w[i] + char_to_w[j])) {
    	console.log("" + i + "," + j + "," + rect.width)
	  }	    
  }
}
"""

    #
    # cropText() - Based on the height of the font, shorten the string to fit into a specific width...
    # ... empirically derived values for letters / so unlikely to work exactly right if the font changes
    #
    def cropText(self, txt, txt_h, w):
        # If it fits, it ships
        if self.textLength(txt,txt_h) <= w:
            return txt

        # Otherwise... iterate until it doesn't fit
        i = 1
        while self.textLength(txt[:i],txt_h) < w:
            i += 1
        
        # Assumption is the the '...' doesn't add too much...
        if i == 0:
            i += 1
        return txt[:i-1] + '...'

    #
    # To adjust, see the "rt_test_fontmetrics_2.ipynb" file
    # ... not really correct either... apparently monospace isn't monospaced (or at least not in firefox)
    #
    font_w_slope     = 0.54961443           # 0.5496575 
    font_w_intercept = 0.027930354650640865 # 0.1928203837408624
    
    #
    # textLength() - calculate the expected text length
    #
    def __Monospace__textLength__(self, txt, txt_h):
        char_w = self.font_w_slope * txt_h + self.font_w_intercept
        return len(txt)*char_w

    #
    # textLength() - calculate the expected text length
    # ... was eventuall modified for "Ariel" ... and then back to Times Serif since Ariel wasn't working with SVGLib
    # ... but the kerning made the calculation difficult... any lower case letters following an "f" were a problem...
    #
    def textLength(self, txt, txt_h):
        ratio14 = 15.8
        # Load the pre-calculated font-metrics
        if self._font_metrics_ is None:
            _rt_dir   = os.path.dirname(os.path.abspath(__file__))
            # _filename = os.path.join(_rt_dir, "../config", "20230629_ariel_14.txt")
            _filename = os.path.join(_rt_dir, "config", "20230628_times_serif_14.txt")
            with open(_filename) as file:
                lines = [line.rstrip() for line in file]
            self._font_metrics_ = {}
            for _line in lines:
                _parts = _line.split(',')
                _ord   = int(_parts[0])
                _sz    = int(_parts[1])
                _w     = float(_parts[2])
                if _sz not in self._font_metrics_.keys():
                    self._font_metrics_[_sz] = {}    
                self._font_metrics_[_sz][_ord] = _w
            self._font_metrics_[_sz][32] = 4.0
        # Calculate the width of the specified string
        w = 0
        for c in txt:
            if ord(c) in self._font_metrics_[14].keys():
                w += self._font_metrics_[14][ord(c)] * txt_h/ratio14
            else:
                w += 10 * txt_h/ratio14
        return w

    #
    # renderBoxPlotColumn() - render a single boxplot column (originally from the TemporalBarchart Implementation)
    #
    def renderBoxPlotColumn(self, style, k_df, cx, yT, group_by_max, group_by_min, bar_w, count_by, color_by, cap_swarm_at):
        if self.isPandas(k_df):
            return self.__renderBoxPlotColumn_pandas__(style, k_df, cx, yT, group_by_max, group_by_min, bar_w, count_by, color_by, cap_swarm_at)
        elif self.isPolars(k_df):
            return self.__renderBoxPlotColumn_polars__(style, k_df, cx, yT, group_by_max, group_by_min, bar_w, count_by, color_by, cap_swarm_at)
        else:
            raise Exception('renderBoxPlotColumn() - not a pandas or polars dataframe')

    def __renderBoxPlotColumn_pandas__(self, style, k_df, cx, yT, group_by_max, group_by_min, bar_w, count_by, color_by, cap_swarm_at):
        svg = ''
        if len(k_df) > 0:
            color = self.co_mgr.getTVColor('data','default') 

            # Just plot points if less than 5...
            if len(k_df) < 5:
                x_sz = 3
                for _value in k_df[count_by]:
                    sy = yT(_value)
                    svg += f'<line x1="{cx-x_sz}" y1="{sy-x_sz}" x2="{cx+x_sz}" y2="{sy+x_sz}" stroke="{color}" stroke-width="2" />'
                    svg += f'<line x1="{cx-x_sz}" y1="{sy+x_sz}" x2="{cx+x_sz}" y2="{sy-x_sz}" stroke="{color}" stroke-width="2" />'
            else:
                # Derived partially from: https://byjus.com/maths/box-plot/
                _med           = k_df[count_by].median()
                q3             = k_df[count_by].quantile(0.75)
                q1             = k_df[count_by].quantile(0.25)
                iqr            = q3-q1                           # difference between 1st and 3rd quartile
                q3_plus_15iqr  = q3 + 1.5*iqr
                q1_minus_15iqr = q1 - 1.5*iqr

                # for uniform distributions... non-normal distributions, the tops and bottoms can exceed the max and mins...
                upper_color,upper_is_max = color,False
                if q3_plus_15iqr > group_by_max:
                    q3_plus_15iqr = group_by_max
                    upper_color,upper_is_max   = self.co_mgr.getTVColor('label','error'),True
                lower_color,lower_is_min = color,False
                if q1_minus_15iqr < group_by_min:
                    q1_minus_15iqr = group_by_min
                    lower_color,lower_is_min   = self.co_mgr.getTVColor('label','error'),True

                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(q3_plus_15iqr)}"     x2="{cx+bar_w/2}"     y2="{yT(q3_plus_15iqr)}"    stroke="{upper_color}" stroke-width="1.5" />'
                svg += f'<rect  x="{cx-bar_w/2}"  y="{yT(q3)}"             width="{bar_w}"      height="{yT(q1)-yT(q3)}"        stroke="{color}"       stroke-width="1"   fill-opacity="0.0" />'
                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(q3)}"                x2="{cx+bar_w/2}"     y2="{yT(q3)}"               stroke="{color}"       stroke-width="1.5" />'
                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(_med)}"              x2="{cx+bar_w/2}"     y2="{yT(_med)}"             stroke="{color}"       stroke-width="1.5" />'
                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(q1)}"                x2="{cx+bar_w/2}"     y2="{yT(q1)}"               stroke="{color}"       stroke-width="1.5" />'
                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(q1_minus_15iqr)}"    x2="{cx+bar_w/2}"     y2="{yT(q1_minus_15iqr)}"   stroke="{lower_color}" stroke-width="1.5" />'

                svg += f'<line x1="{cx}"          y1="{yT(q3)}"                x2="{cx}"            y2="{yT(q3_plus_15iqr)}"    stroke="{upper_color}" stroke-width="0.5" />'
                if upper_is_max:
                    svg += f'<line x1="{cx}"      y1="{yT(q3_plus_15iqr)}"     x2="{cx+5}"          y2="{yT(q3_plus_15iqr)+5}"  stroke="{upper_color}" stroke-width="0.5" />'
                    svg += f'<line x1="{cx}"      y1="{yT(q3_plus_15iqr)}"     x2="{cx-5}"          y2="{yT(q3_plus_15iqr)+5}"  stroke="{upper_color}" stroke-width="0.5" />'
                svg += f'<line x1="{cx}"          y1="{yT(q1)}"                x2="{cx}"            y2="{yT(q1_minus_15iqr)}"   stroke="{lower_color}" stroke-width="0.5" />'
                if lower_is_min:
                    svg += f'<line x1="{cx}"      y1="{yT(q1_minus_15iqr)}"    x2="{cx+5}"          y2="{yT(q1_minus_15iqr)-5}" stroke="{upper_color}" stroke-width="0.5" />'
                    svg += f'<line x1="{cx}"      y1="{yT(q1_minus_15iqr)}"    x2="{cx-5}"          y2="{yT(q1_minus_15iqr)-5}" stroke="{upper_color}" stroke-width="0.5" />'

                # Add marks for any items that are outliers
                _df = k_df[(k_df[count_by] > q3_plus_15iqr) | (k_df[count_by] < q1_minus_15iqr)]
                for v in _df[count_by]:
                    if v > q3_plus_15iqr:
                        svg += f'<circle cx="{cx}" cy="{yT(v)}" r="1.5" fill="{color}" />'
                    if v < q3_plus_15iqr:
                        svg += f'<circle cx="{cx}" cy="{yT(v)}" r="1.5" fill="{color}" />'

                # Add the swarm elements
                if style == 'boxplot_w_swarm':
                    # Provide cap... because this could take forever for large dataframes
                    if cap_swarm_at is not None and len(k_df) > cap_swarm_at:
                        _df = k_df.sample(cap_swarm_at)
                    else:
                        _df = k_df

                    if color_by is None:
                        for v in _df[count_by]:
                            sy    = yT(v)
                            mycx = cx + random.random() * bar_w/2 - bar_w/4
                            svg += f'<line x1="{mycx-1}" y1="{sy-1}" x2="{mycx+1}" y2="{sy+1}" stroke="{color}" stroke-width="0.5" />'
                            svg += f'<line x1="{mycx-1}" y1="{sy+1}" x2="{mycx+1}" y2="{sy-1}" stroke="{color}" stroke-width="0.5" />'
                    else:
                        for ksw,ksw_df in _df.groupby(color_by):
                            my_color = self.co_mgr.getColor(ksw)
                            for v in ksw_df[count_by]:
                                sy    = yT(v)
                                mycx = cx + random.random() * bar_w/2 - bar_w/4
                                svg += f'<line x1="{mycx-1}" y1="{sy-1}" x2="{mycx+1}" y2="{sy+1}" stroke="{my_color}" stroke-width="0.5" />'
                                svg += f'<line x1="{mycx-1}" y1="{sy+1}" x2="{mycx+1}" y2="{sy-1}" stroke="{my_color}" stroke-width="0.5" />'
        return svg

    def __renderBoxPlotColumn_polars__(self, style, k_df, cx, yT, group_by_max, group_by_min, bar_w, count_by, color_by, cap_swarm_at):
        svg = ''
        if len(k_df) > 0:
            color = self.co_mgr.getTVColor('data','default') 

            # Just plot points if less than 5...
            if len(k_df) < 5:
                x_sz = 3
                for _value in k_df[count_by]:
                    sy = yT(_value)
                    svg += f'<line x1="{cx-x_sz}" y1="{sy-x_sz}" x2="{cx+x_sz}" y2="{sy+x_sz}" stroke="{color}" stroke-width="2" />'
                    svg += f'<line x1="{cx-x_sz}" y1="{sy+x_sz}" x2="{cx+x_sz}" y2="{sy-x_sz}" stroke="{color}" stroke-width="2" />'
            else:
                # Derived partially from: https://byjus.com/maths/box-plot/
                _med           = k_df[count_by].median()
                q3             = k_df[count_by].quantile(0.75)
                q1             = k_df[count_by].quantile(0.25)
                iqr            = q3-q1                           # difference between 1st and 3rd quartile
                q3_plus_15iqr  = q3 + 1.5*iqr
                q1_minus_15iqr = q1 - 1.5*iqr

                # for uniform distributions... non-normal distributions, the tops and bottoms can exceed the max and mins...
                upper_color,upper_is_max = color,False
                if q3_plus_15iqr > group_by_max:
                    q3_plus_15iqr = group_by_max
                    upper_color,upper_is_max   = self.co_mgr.getTVColor('label','error'),True
                lower_color,lower_is_min = color,False
                if q1_minus_15iqr < group_by_min:
                    q1_minus_15iqr = group_by_min
                    lower_color,lower_is_min   = self.co_mgr.getTVColor('label','error'),True

                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(q3_plus_15iqr)}"     x2="{cx+bar_w/2}"     y2="{yT(q3_plus_15iqr)}"    stroke="{upper_color}" stroke-width="1.5" />'
                svg += f'<rect  x="{cx-bar_w/2}"  y="{yT(q3)}"             width="{bar_w}"      height="{yT(q1)-yT(q3)}"        stroke="{color}"       stroke-width="1"   fill-opacity="0.0" />'
                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(q3)}"                x2="{cx+bar_w/2}"     y2="{yT(q3)}"               stroke="{color}"       stroke-width="1.5" />'
                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(_med)}"              x2="{cx+bar_w/2}"     y2="{yT(_med)}"             stroke="{color}"       stroke-width="1.5" />'
                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(q1)}"                x2="{cx+bar_w/2}"     y2="{yT(q1)}"               stroke="{color}"       stroke-width="1.5" />'
                svg += f'<line x1="{cx-bar_w/2}" y1="{yT(q1_minus_15iqr)}"    x2="{cx+bar_w/2}"     y2="{yT(q1_minus_15iqr)}"   stroke="{lower_color}" stroke-width="1.5" />'

                svg += f'<line x1="{cx}"          y1="{yT(q3)}"                x2="{cx}"            y2="{yT(q3_plus_15iqr)}"    stroke="{upper_color}" stroke-width="0.5" />'
                if upper_is_max:
                    svg += f'<line x1="{cx}"      y1="{yT(q3_plus_15iqr)}"     x2="{cx+5}"          y2="{yT(q3_plus_15iqr)+5}"  stroke="{upper_color}" stroke-width="0.5" />'
                    svg += f'<line x1="{cx}"      y1="{yT(q3_plus_15iqr)}"     x2="{cx-5}"          y2="{yT(q3_plus_15iqr)+5}"  stroke="{upper_color}" stroke-width="0.5" />'
                svg += f'<line x1="{cx}"          y1="{yT(q1)}"                x2="{cx}"            y2="{yT(q1_minus_15iqr)}"   stroke="{lower_color}" stroke-width="0.5" />'
                if lower_is_min:
                    svg += f'<line x1="{cx}"      y1="{yT(q1_minus_15iqr)}"    x2="{cx+5}"          y2="{yT(q1_minus_15iqr)-5}" stroke="{upper_color}" stroke-width="0.5" />'
                    svg += f'<line x1="{cx}"      y1="{yT(q1_minus_15iqr)}"    x2="{cx-5}"          y2="{yT(q1_minus_15iqr)-5}" stroke="{upper_color}" stroke-width="0.5" />'

                # Add marks for any items that are outliers
                _df = k_df.filter(pl.col(count_by) > q3_plus_15iqr)
                for v in _df[count_by]:
                    svg += f'<circle cx="{cx}" cy="{yT(v)}" r="1.5" fill="{color}" />'
                _df = k_df.filter(pl.col(count_by) < q1_minus_15iqr)
                for v in _df[count_by]:
                    svg += f'<circle cx="{cx}" cy="{yT(v)}" r="1.5" fill="{color}" />'

                # Add the swarm elements
                if style == 'boxplot_w_swarm':
                    # Provide cap... because this could take forever for large dataframes
                    if cap_swarm_at is not None and len(k_df) > cap_swarm_at:
                        _df = k_df.sample(cap_swarm_at)
                    else:
                        _df = k_df

                    if color_by is None:
                        for v in _df[count_by]:
                            sy    = yT(v)
                            mycx = cx + random.random() * bar_w/2 - bar_w/4
                            svg += f'<line x1="{mycx-1}" y1="{sy-1}" x2="{mycx+1}" y2="{sy+1}" stroke="{color}" stroke-width="0.5" />'
                            svg += f'<line x1="{mycx-1}" y1="{sy+1}" x2="{mycx+1}" y2="{sy-1}" stroke="{color}" stroke-width="0.5" />'
                    else:
                        for ksw,ksw_df in _df.groupby(color_by):
                            my_color = self.co_mgr.getColor(ksw)
                            for v in ksw_df[count_by]:
                                sy    = yT(v)
                                mycx = cx + random.random() * bar_w/2 - bar_w/4
                                svg += f'<line x1="{mycx-1}" y1="{sy-1}" x2="{mycx+1}" y2="{sy+1}" stroke="{my_color}" stroke-width="0.5" />'
                                svg += f'<line x1="{mycx-1}" y1="{sy+1}" x2="{mycx+1}" y2="{sy-1}" stroke="{my_color}" stroke-width="0.5" />'
        return svg

    # ===========================================================================================================================================================

    #
    # Calculate the angled position string top and bottom position
    #
    def calculateAngledLabelTopAndBottomPosition(self, x, y, bar_w, txt_h, angle):
        frac_vert,frac_horz,bar_y = angle/90, (90-angle)/90, 0
        as_rad = pi*(angle+90)/180.0 # more than just radian conversion...
        horz_tpos  = (x+4,               y+4)       # top of string begin if the string were rendered horizontally
        horz_bpos  = (x+4,               y+4+txt_h) # bottom of string begin if the string were rendered horizontally
        vert_tpos  = (x+bar_w/2+txt_h/2, y+4)       # top of string begin if the string were rendered vertically
        vert_bpos  = (x+bar_w/2-txt_h/2, y+4)       # bottom of string begin if the string were rendered vertically
        angle_tpos = (vert_tpos[0]*frac_vert + horz_tpos[0]*frac_horz, vert_tpos[1]*frac_vert + horz_tpos[1]*frac_horz)
        angle_bpos = (angle_tpos[0] + cos(as_rad)*txt_h,               angle_tpos[1] + sin(as_rad)*txt_h)
        return angle_tpos,angle_bpos

    #
    # Does the specified angle cause the label to not overlap with the next label?
    # ... there's a close formed solution here... but it's beyond me :(
    # ... so many wasted cpu cycles... so many...
    #
    # ... see the rt_test_rotated_label prototype for testing/derivation
    #
    def doesAngleWorkForLabel(self, bar_w, txt_h, angle):
        if angle < 0 or angle >= 90:
            raise Exception(f'RACETrack.doesAngleWorkForLabel() - angle must be between [0,90) ... supplied angle = {angle}')

        # Position of label 0 and then label 1
        angle0_tpos,angle0_bpos = self.calculateAngledLabelTopAndBottomPosition(0,    0, bar_w, txt_h, angle)
        angle1_tpos,angle1_bpos = self.calculateAngledLabelTopAndBottomPosition(bar_w,0, bar_w, txt_h, angle)

        # Line from angle0_tpos in the direction of the angle...  is it underneath the angle1_bpos?
        m = sin(pi*angle/180)
        b = angle0_tpos[1] - m*angle0_tpos[0]
        return (m*angle1_bpos[0] + b) > angle1_bpos[1]

    #
    # Best angle for rotated label?
    #
    def bestAngleForRotatedLabels(self, bar_w, txt_h):
        for angle in range(0,90):
            if self.doesAngleWorkForLabel(bar_w, txt_h, angle):
                return angle
        return 90
    
    #
    # Determine if a string is an integer
    # ... shouldn't be used at scale
    # ... there's got to be a better way :( ... or some kind of builtin
    #
    def strIsInt(self, x):
        try:
            int(x)
            return True
        except:
            return False

    #
    # Determine if a string is a float
    # ... shouldn't be used at scale
    # ... there's got to be a better way :( ... or some kind of builtin
    #
    def strIsFloat(self, x):
        try:
            float(x)
            return True
        except:
            return False
    
    #
    # spacer() - simple spacer object -- mostly for tiling
    #
    def spacer(self, w, h, _color_='#000000'):
        ''' Render a spacer -- usually for a tile '''
        return f'<svg x="0" y="0" width="{w}" height="{h}"><rect x="0" y="0" width="{w}" height="{h}" fill="{_color_}" /></svg>'


    def labeler(self, text_tuples, w, h=None, txt_h=16, x_ins=2, y_ins=4, h_gap=2):
        '''
        labeler - create an svg label from a list of text tuples

        parameters
        ----------
        text_tuples - list of text tuples, examples are as follows
            ['this', 'is', 'a', 'test'] # simple strings for labels -- defaults for all other values
            - or -
            a tuple consisting of the following items...
            ('my string')
            ('my string', my_txt_h)
            ('my string', my_txt_h, my_hex_color)
            ('my_string', my_txt_h, my_hex_color, my_h_gap)
            ('my_string', my_txt_h, my_hex_color, my_h_gap, my_top_gap)
            ('my_string', my_txt_h, my_hex_color, my_h_gap, my_top_gap, my_svg_icon)
        
        w     - width (required)
        h     - height (optional, if not provided will be calculated)  
        txt_h - default text height
        x_ins - x insert (left and right)
        y_ins - y insert (top and bottom)
        h_gap - horizontal gap between txt strings
        '''
        svg = []
        y_so_far = y_ins
        for _tuple_ in text_tuples:
            if   type(_tuple_) is str:
                svg.append(self.svgText(_tuple_, x_ins, y_so_far + txt_h, txt_h))
                y_so_far += txt_h + h_gap
            elif type(_tuple_) is tuple:
                my_str      = _tuple_[0]
                my_txt_h    = _tuple_[1] if len(_tuple_) >= 2 else txt_h
                my_color    = _tuple_[2] if len(_tuple_) >= 3 else None
                my_h_gap    = _tuple_[3] if len(_tuple_) >= 4 else h_gap
                my_top_gap  = _tuple_[4] if len(_tuple_) >= 5 else 0
                my_svg_icon = _tuple_[5] if len(_tuple_) >= 6 else None
                icon_w      = 0
                if my_svg_icon is not None:
                    icon_w, icon_h = self.__extractSVGWidthAndHeight__(my_svg_icon)
                    svg.append(self.__overwriteSVGXAndY__(my_svg_icon, (x_ins, my_top_gap + y_so_far + my_txt_h/2.0 - icon_h/2.0)))
                    icon_w += x_ins
                svg.append(self.svgText(my_str, x_ins + icon_w, my_top_gap + y_so_far + my_txt_h, my_txt_h, my_color))
                y_so_far += my_txt_h + my_h_gap
            else: raise Exception("Only Strings or Tuples Supported")

        my_h = h if h is not None else y_so_far + y_ins

        return f'<svg x="0" y="0" width="{w}" height="{my_h}">' + ''.join(svg) + '</svg>'

