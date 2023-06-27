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
import hashlib
import random
import urllib
import html

from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image
import io

from math import cos,sin,pi

from IPython.core import display as ipc_display

from IPython.display import Javascript, HTML, display

from rt_annotations_mixin       import RTAnnotationsMixin
from rt_boxplot_mixin           import RTBoxplotMixin
from rt_calendarheatmap_mixin   import RTCalendarHeatmapMixin
from rt_color_manager           import RTColorManager
from rt_datamanip_mixin         import RTDataManipMixin
from rt_geometry_mixin          import RTGeometryMixin
from rt_graph_layouts_mixin     import RTGraphLayoutsMixin
from rt_histogram_mixin         import RTHistogramMixin
from rt_layouts_mixin           import RTLayoutsMixin
from rt_linknode_mixin          import RTLinkNodeMixin
from rt_panel_mixin             import RTPanelMixin
from rt_periodic_barchart_mixin import RTPeriodicBarChartMixin
from rt_piechart_mixin          import RTPieChartMixin
from rt_shapes_mixin            import RTShapesMixin
from rt_small_multiples_mixin   import RTSmallMultiplesMixin
from rt_temporal_barchart_mixin import RTTemporalBarChartMixin
from rt_text_mixin              import RTTextMixin
from rt_timeline_mixin          import RTTimelineMixin
from rt_xy_mixin                import RTXYMixin

__name__ = 'racetrack'

class RACETrack(RTAnnotationsMixin,
                RTBoxplotMixin,
                RTCalendarHeatmapMixin,
                RTDataManipMixin,
                RTGeometryMixin,
                RTGraphLayoutsMixin,
                RTHistogramMixin,
                RTLayoutsMixin,                
                RTLinkNodeMixin,
                RTPanelMixin,
                RTPeriodicBarChartMixin,
                RTPieChartMixin,
                RTShapesMixin,
                RTSmallMultiplesMixin,
                RTTemporalBarChartMixin,
                RTTextMixin,
                RTTimelineMixin,
                RTXYMixin):
    #
    # Constructor (or whatever this is called in Python)
    #
    def __init__(self):
        # Visualization globals
        self.co_mgr            = RTColorManager(self)
        self.default_font      = "Times, serif"
        self.fformat           = '0.2f'
        
        # Field transformations
                                  #
                                  # Time-based transformations
                                  #
        self.transforms        = ['day_of_week',       # day of the week
                                  'day_of_week_hour',  # day of the week plus the hour of the day
                                  'year',              # year
                                  'quarter',           # quarter
                                  'year_quarter',      # year and quarter
                                  'month',             # month
                                  'year_month',        # year and month
                                  'year_month_day',    # year, month, and day
                                  'day',               # day (of the month)
                                  'day_of_year',       # day of the year
                                  'day_of_year_hour',  # day of the year w/ hour
                                  'hour',              # hour (of the day)
                                  'minute',            # minute (of the hour)
                                  'second',            # second (of the minute)
                                  #
                                  # Numeric transformations
                                  #
                                  'log_bins'           # log-based binning
                                  ]

        # Used for reflections
        self.widgets           = ['boxplot',
                                  'calendarHeatmap',
                                  'histogram',
                                  'linkNode',
                                  'periodicBarChart',
                                  'pieChart',
                                  'temporalBarChart',
                                  'xy']
        
        # Cache for converting strings to integers
        RACETrack.hashcode_lu  = {}
        
        # Inits for mixins...  probably a better way to do this...
        self.__annotations_mixin_init__()
        self.__panel_mixin_init__()
        self.__periodic_barchart_mixin_init__()
        self.__temporal_barchart_mixin_init__()

    #
    # Render the SVG as HTML and display it within a notebook
    #
    def displaySVG(self,_svg):
        if type(_svg) != str:
            _svg = _svg._repr_svg_()
        return display(HTML(_svg))

    #
    # Render the SVG as an Image and display it within a notebook
    # - Uses an in memory image buffer
    # - Image form should save processing power for complicated SVGs
    #
    def displaySVGAsImage(self, _svg):
        if type(_svg) != str:
            _svg = _svg._repr_svg_()
        b = io.BytesIO()
        renderPM.drawToFile(svg2rlg(io.StringIO(_svg)), b, 'PNG')
        return ipc_display.Image(data=b.getvalue(),format='png',embed=True)

    #
    # Return a consistent hashcode for a string
    #
    def hashcode(self,s):
        if type(s) != str: # Force non-strings to be strings
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
        return _enc

    #
    # Decode a string that was created by the encSVGID() method.
    #
    def decSVGID(self, s):
        if s.startswith('encsvgid_'):
            s_prime = s[len('encsvgid_'):]
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
            return s

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
            if type(x) == list:
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
        return tfield is not None and type(tfield) == str and tfield.startswith('|tr|')      
    
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
                for _date in pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='M'):
                    _order.append(f'{_date.year}-{_date.month:02}')
            elif transform == 'year_month_day':
                for _date in pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D'):
                    _order.append(f'{_date.year}-{_date.month:02}-{_date.day:02}')
            elif transform == 'day':
                for _day in range(1,32):
                    _order.append(f'{_day:02}')
            elif transform == 'day_of_year':
                if df[field].min().year == df[field].max().year:
                    for _day in range(df[field].min().day_of_year, df[field].max().day_of_year+1):
                        _order.append(f'{_day:03}')
                else:
                    for _day in range(1,366):
                        _order.append(f'{_day:03}')
            elif transform == 'day_of_year_hour':
                if df[field].min().year == df[field].max().year:
                    for _day in range(df[field].min().day_of_year, df[field].max().day_of_year+1):
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
        return _order

    #
    # Create a tranformation field (tfield)
    #
    def createTField(self,field,trans):
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
        if   type(something) == str:
            columns_set.add(something)
        elif type(something) == bool: # unclear why None may be converted to False // is that what's happening?
            pass # do nothing
        elif type(something) == list or type(something) == tuple:
            for x in something:
                self.__recursiveDecompose__(x, columns_set)
        else:
            raise Exception(f'Unknown type ("{type(something)}") for ("{something}") encountered in identifyColumnsFromParameters()')

    #
    # Determine If A Column Has To Be Counted By Set Operation
    #
    def countBySet(self, 
                   df,         # dataframe
                   count_by):  # field to check
        if count_by is None:
            return False
        if type(df) == list:
            for _df in df:
                if count_by in _df.columns:
                    if _df[count_by].dtypes != np.int64   and \
                       _df[count_by].dtypes != np.int32   and \
                       _df[count_by].dtypes != np.float64 and \
                       _df[count_by].dtypes != np.float32:
                       return True
            return False
        else:
            return df[count_by].dtypes != np.int64   and \
                   df[count_by].dtypes != np.int32   and \
                   df[count_by].dtypes != np.float64 and \
                   df[count_by].dtypes != np.float32
    
    #
    # fieldIsArithmetic()
    # ... determine if a field can be operated on by arithmetic
    # ... maybe the oposite of the above?
    #
    def fieldIsArithmetic(self, df, field):
        return df[field].dtypes == np.int64   or \
               df[field].dtypes == np.int32   or \
               df[field].dtypes == np.float64 or \
               df[field].dtypes == np.float32

    #
    # Determine color ordering based on quantity
    #
    def colorRenderOrder(self, 
                         df,                   # dataframe
                         color_by,             # color_by field
                         count_by,             # count_by field
                         count_by_set=False):  # for the field set, count by set operation
        if color_by is None:
            return None
        return self.colorQuantities(df, color_by, count_by, count_by_set).sort_values(ascending=False)

    #
    # Determine color quantities (unsorted)
    #
    def colorQuantities(self, 
                        df,                  # dataframe 
                        color_by,            # color_by field
                        count_by,            # count_by field
                        count_by_set=False): # for the field set, count by set operation
        if color_by is None:
            return None

        # Make sure we can count by numeric summation
        if count_by_set == False:
            count_by_set = self.countBySet(df, count_by)

        # For count by set... when count_by == color by... then we'll count by rows
        if count_by is not None and count_by_set and count_by == color_by:
            count_by = None

        if count_by is None:
            return df.groupby(color_by).size()
        elif count_by_set:
            _df = pd.DataFrame(df.groupby([color_by,count_by]).size()).reset_index()
            return _df.groupby(color_by).size()
        elif count_by == color_by:
            _df = df.groupby(color_by).size().reset_index()
            _df['__mult__'] = _df.apply(lambda x: x[color_by]*x[0],axis=1)
            return _df.groupby(color_by)['__mult__'].sum()
        else:
            return df.groupby(color_by)[count_by].sum()

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
        svg = ''
        if bar_len > 0:
            _co = self.co_mgr.getTVColor('data','default')

            # Default bar w/out color
            if horz:
                svg += f'<rect x="{x}" y="{y}" width="{bar_len}" height="{bar_sz}" fill="{_co}" />'
            else:
                svg += f'<rect x="{x}" y="{y-bar_len}" width="{bar_sz}" height="{bar_len}" fill="{_co}" />'
            
            # Colorize it
            if color_by is not None:
                quantities   = self.colorQuantities(df, color_by, count_by, count_by_set)
                value        = quantities.sum()
                quantities   = quantities[quantities > value/bar_len]
                intersection = self.__myIntersection__(global_color_order.index, quantities.index)
                if horz:
                    d = x
                else:
                    d = y
                for cb_bin in intersection:
                    v = quantities[cb_bin]
                    l = bar_len * v / value
                    if l >= 1.0:
                        _co = self.co_mgr.getColor(cb_bin)
                        if horz:
                            svg += f'<rect x="{d}" y="{y}" width="{l}" height="{bar_sz}" fill="{_co}" />'
                            d += l
                        else:
                            svg += f'<rect x="{x}" y="{d-l}" width="{bar_sz}" height="{l}" fill="{_co}" />'
                            d -= l
                            
        return svg

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
                txt_h,
                color    = None,
                anchor   = 'start',
                font     = None,
                rotation = None):
        if font is None:
            font = self.default_font
        if color is None:
            color = self.co_mgr.getTVColor('label','defaultfg')
        if rotation is not None:
            return f'<text x="{x}" text-anchor="{anchor}" y="{y}" font-family="{font}" fill="{color}" font-size="{txt_h}px"' + \
                   f' transform="rotate({rotation},{x},{y})">{html.escape(txt)}</text>'
        else:
            return f'<text x="{x}" text-anchor="{anchor}" y="{y}" font-family="{font}" fill="{color}" font-size="{txt_h}px">{html.escape(txt)}</text>'

    #
    # Empirically-derived font metrics -- see the next javascript code block on the initial generation of these numbers
    # ... this is really just a starting point
    # ... looks like correct for visual studio code with a txt_h of 19.5 // 2023-02-05
    #
    _font_metrics_ = {
        'a':9.100006103515625,
        'b':10,
        'c':9.100006103515625,
        'd':10,
        'e':9.100006103515625,
        'f':7.350006103515625,
        'g':10,
        'h':10,
        'i':6.4666595458984375,
        'j':6.4666595458984375,
        'k':10,
        'l':6.4666595458984375,
        'm':14.466659545898438,
        'n':10,
        'o':10,
        'p':10,
        'q':10,
        'r':7.350006103515625,
        's':8.25,
        't':6.4666595458984375,
        'u':10,
        'v':10,
        'w':13.566665649414062,
        'x':10,
        'y':10,
        'z':9.100006103515625,
        'A':13.566665649414062,
        'B':12.699996948242188,
        'C':12.699996948242188,
        'D':13.566665649414062,
        'E':11.76666259765625,
        'F':10.916671752929688,
        'G':13.566665649414062,
        'H':13.566665649414062,
        'I':7.350006103515625,
        'J':8.25,
        'K':13.566665649414062,
        'L':11.76666259765625,
        'M':16.25,
        'N':13.566665649414062,
        'O':13.566665649414062,
        'P':10.916671752929688,
        'Q':13.566665649414062,
        'R':12.699996948242188,
        'S':10.916671752929688,
        'T':11.76666259765625,
        'U':13.566665649414062,
        'V':13.566665649414062,
        'W':17.133331298828125,
        'X':13.566665649414062,
        'Y':13.566665649414062,
        'Z':11.76666259765625,
        '0':10,
        '1':10,
        '2':10,
        '3':10,
        '4':10,
        '5':10,
        '6':10,
        '7':10,
        '8':10,
        '9':10
    }

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
    # textLength() - calculate the expected text length
    #
    def textLength(self, txt, txt_h):
        w = 0
        for c in txt:
            if c in self._font_metrics_:
                w += self._font_metrics_[c] * txt_h/19.5
            else:
                w += 10 * txt_h/19.5
        return w

    #
    # renderBoxPlotColumn() - render a single boxplot column (originally from the TemporalBarchart Implementation)
    #
    def renderBoxPlotColumn(self, style, k_df, cx, yT, group_by_max, group_by_min, bar_w, count_by, color_by, cap_swarm_at):
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

