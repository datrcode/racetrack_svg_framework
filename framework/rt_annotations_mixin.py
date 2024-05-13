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

import re
import random

from datetime import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta

from math import sqrt,pi,atan,floor

__name__ = 'rt_annotations_mixin'

#
# Annotations Mixin
#
class RTAnnotationsMixin(object):
    #
    # Constructor
    #
    def __annotations_mixin_init__(self):
        self.annotations_ls = []

    #
    # tag() - tag a subset dataframe.
    # - simple tags ... just a word ... no bars (|) or equal (=) signs
    # - type-value tags ... word=value ... no bars (|) or equal (=) signs
    # - op = 'add', 'set'
    # - df_sub should be a subset of the df ... no idea what happens if it isn't
    #
    # NOTE:  THE POLARS AND PANDAS VERSION DIFFER IN WHAT DATAFRAME IS ACTUALLY
    # MODIFIED ... PANDAS VERSION MODIFIES IN_PLACE... POLARS DOESN"T AFFECT THE
    # IN_PLACE VERSION ... AS LONG AS THE CONVENTION IS "df = rt.tag()" THEN ITS
    # OKAY ... I'M SURE THIS IS GOING TO MESS SOMETHING UP DOWNSTREAM...
    #
    def tag(self, df, df_sub, tag, op='add', field='__tag__'):
        if op != 'add' and op != 'set':
            raise Exception('tag() - only operations supported are "set" and "add" (default)')
        if self.isPandas(df):
            return self.__tag_pandas__(df, df_sub, tag, op, field)
        elif self.isPolars(df):
            return self.__tag_polars__(df, df_sub, tag, op, field)
        else:
            raise Exception('tag() - only pandas and polars implemented')

    # pandas version
    def __tag_pandas__(self, df, df_sub, tag, op, field):
        if op == 'set' or field not in df.columns:
            if field not in df.columns:
                df[field] = ''
            df.loc[df.index.isin(df_sub.index), field] = tag
        else: # add
            df.update(df.loc[df.index.isin(df_sub.index), field].apply(lambda x: self.__addToTag__(x, tag)))
        return df

    # polar version
    def __tag_polars__(self, df, df_sub, tag, op, field):
        ee_but_tag = list(set(df.columns) - set([field]))
        if op == 'set' or field not in df.columns:
            if field not in df.columns:
                df = df.with_columns(pl.lit('').alias(field))
            df_sub = df_sub.with_columns(pl.lit(tag).alias(field))
            df = df.update(df_sub, on=ee_but_tag)
        else:
            _fn_ = lambda x: self.__addToTag__(x, tag)
            df_sub = df_sub.with_columns(pl.col(field).map_elements(_fn_))
            df = df.update(df_sub, on=ee_but_tag)
        return df

    # mechanics to add to a tag
    def __addToTag__(self, orig, to_add):
        if orig is None:
            return to_add
        else:
            orig, to_add = str(orig), str(to_add)
            _set_ = set(orig.split('|'))
            _set_.add(to_add)
            if '=' in to_add: # may need to dedupe type-value
                _tv_      = to_add.split('=')                 # split type value
                _type_    = _tv_[0]                           # pull out type
                _value_   = _tv_[1] if len(_tv_) == 2 else '' # pull out value
                _new_set_ = set()
                for x in _set_:
                    if '=' in x and x.split('=')[0] == _type_:
                        _new_set_.add(to_add)
                    else:
                        _new_set_.add(x)
                _set_ = _new_set_
            _joined_ = '|'.join(sorted(list(_set_)))
            if len(_joined_) > 0 and _joined_[0] == '|':
                _joined_ = _joined_[1:]
            return _joined_

    #
    # legendForSpectrum()
    #
    def legendForSpectrum(self, _min_=0.0, _max_=1.0, w=256, h=40, txt_h=12, draw_labels=True):
        svg =  f'<svg x="0" y="0" width="{w}" height="{h}">'
        co  =  self.co_mgr.getTVColor('background','default')
        svg += f'<rect x="0" y="0" width="{w}" height="{h}" fill="{co}" />'
        x0, x1 = 3, w-3
        bar_h  = h - (3+3+3+txt_h) if draw_labels else h - (3+3)
        for x in range(x0,x1):
            co = self.co_mgr.spectrum(x, x0, x1) # self.co_mgr.spectrumAbridged(x, x0, x1)
            svg += f'<line x1="{x}" y1="{3}" x2="{x}" y2="{3+bar_h}" stroke="{co}" stroke-width="1.5" />'
        if draw_labels:
            _min_str_ = self.readable(_min_)
            svg += self.svgText(_min_str_, x0, h-4, txt_h)
            _max_str_ = self.readable(_max_)
            svg += self.svgText(_max_str_, x1, h-4, txt_h, anchor='end')
        svg += '</svg>'
        return svg
    
    #
    # Add an annotation into the application state
    #
    def addAnnotation(self,
                      _annotation):  # Instance of subclasses within this file
        self.annotations_ls.append(_annotation)
        return _annotation

    #
    # List all annotations
    #
    def listAnnotations(self):
        return self.annotations_ls.copy()

    #
    # Remove an annotation
    #
    def removeAnnotation(self, 
                         _annotation): # From the listAnnotations results...
        self.annotations_ls.remove(_annotation)
        return _annotation
    
    #
    # Remove all annotations
    #
    def removeAllAnnotations(self):
        self.annotations_ls = []

    #
    # Convert the annotations into a dataframe
    # - note that modifying this dataframe won't modify the annotations within the class
    #
    def annotationsAsDataFrame(self):
        pass

    #
    # Import annotations from a dataframe
    # - should have the columns from the annotationsAsDataFrame()
    #
    def importAnnotationsFromDataFrame(self,
                                       replace_existing=True):
        pass

    #
    # Filter A DataFrame By Annotations
    #
    def filterDataFrameToAnnotations(self,
                                     df,
                                     annotations      = None, # (1) None == Use Stateful List; (2) List of Annotations; (3) Single Annotation
                                     ts_field         = None,
                                     ts_end_field     = None,
                                     lat_lon_fields   = None,
                                     entity_fields    = None): # (1) Single Field or (2) List of Fields
        # If none, use applications annotations
        if annotations is None:
            annotations = self.annotations_ls

        # If just single, make it into a single element list
        elif type(annotations) == RTAnnotation:
            annotations = [annotations]
        
        # Wrap entity_fields into a list
        if entity_fields is not None and type(entity_fields) == str:
            entity_fields = [entity_fields]

        matcher_booleans = None
        for annotation in annotations:
            _booleans = df.apply(lambda x: annotation.matches(x, entity_fields, ts_field, ts_end_field, lat_lon_fields), axis=1)

            if matcher_booleans is None:
                matcher_booleans =  _booleans
            else:
                matcher_booleans |= _booleans
        
        return df[matcher_booleans]


    #
    # Wrapper for creating EventAnnotation
    #
    def eventAnnotation(self,
                        event_str,                   # human readable event name
                        timestamp_str,               # beginning of timeframe for the event // precision matters
                        timestamp_end_str   = None,  # if set, the end of the timeframe... if not set, then the end precision of the timestamp_str will be used
                        description_str     = None,  # human readable description // not necessary
                        geospatial_bounds   = None,  # if set, the event will be geospatially constrained to the bounds
                        tags                = None): # dictionary of type-value tags for the event // for future use
        return RTAnnotation('event', self,
                            common_str           = event_str,
                            description_str      = description_str,
                            timestamp_str        = timestamp_str,
                            timestamp_end_str    = timestamp_end_str,
                            geospatial_bounds    = geospatial_bounds,
                            tags                 = tags)

    #
    # Wrapper for creating EntityAnnotation
    #
    def entityAnnotation(self,
                         entity_str,                # either exact match for field... or just the human readable entity name
                         description_str   = None,  # human readable description // not necessary
                         entity_regex      = None,  # if set, will be used to match entities
                         ignore_case       = True,  # for regex compares, ignore case flag
                         tags              = None,  # dictionary of type-value tags for the entity // for future use
                         timestamp_str     = None,  # if set, timeframe entity is good for // precision matters
                         timestamp_end_str = None): # if set w/ timestamp_str, timeframe end // precision matters
        return RTAnnotation('entity', self, 
                            common_str           = entity_str,
                            description_str      = description_str,
                            timestamp_str        = timestamp_str,
                            timestamp_end_str    = timestamp_end_str,
                            tags                 = tags,
                            entity_regex         = entity_regex,
                            ignore_case          = ignore_case)


    #
    # Wrapper for creating a GeoSpatialAnnotation
    #
    def geospatialAnnotation(self,
                             name_str,                   # human readable geospatial name
                             geospatial_bounds,          # either a list of point tuples or an SVG path description
                             description_str    = None,  # human readable description // not necessary
                             tags               = None,  # dictionary of type-value tags for the geospatial name // for future use 
                             timestamp_str      = None,  # if set, timeframe geospatial area is good for // precision matters
                             timestamp_end_str  = None): # if set w/ timestamp_str, timeframe end // precision matters
        return RTAnnotation('geospatial', self, 
                            common_str           = name_str,
                            description_str      = description_str,
                            timestamp_str        = timestamp_str,
                            timestamp_end_str    = timestamp_end_str,
                            geospatial_bounds    = geospatial_bounds,
                            tags                 = tags)

    #
    # Convert a Point List to a String // for keying into a dictionary
    #
    def __pointListToString__(self, ls):
        as_str = ''
        for pt in ls:
            as_str += str(pt[0]) + ' '
            as_str += str(pt[1]) + ' '
        return as_str

    #
    # Point within either a svg path description of a point list for polygon
    # ... does not work for arcs...
    #
    def pointWithinGeospatialBounds(self,
                                    point,               # (x,y)
                                    geospatial_bounds):  # [(x,y),(x,y),(x,y)...] or SVG Path description
        # Create Caches
        if hasattr(self,'__poly_cache__') == False:
            self.__poly_cache__        = {} # either [svg_str]    = [poly_points, poly_points, ...]
                                            # ... or [points_str] =  poly_points
            self.__poly_bounds_cache__ = {} # same string as __poly_cache__ but to (x0,y0,x1,y1)

        #
        # SVG Path Version
        #
        if   type(geospatial_bounds) == str:
            as_str = geospatial_bounds

            if as_str not in self.__poly_cache__:
                polygon_list,points = [],[]
                x0,y0,x1,y1 = None,None,None,None

                _s = as_str.lower()
                _s = " ".join(_s.split()) # make sure there's no extra whitespaces
                tokens = _s.lower().split(' ')
                i = 0
                while i < len(tokens):
                    if   tokens[i] == 'm':
                        if len(points) >= 3:
                            polygon_list.append(points)
                        _x,_y = float(tokens[i+1]),float(tokens[i+2])
                        points = []
                        points.append(MyPoint(_x,_y))
                        x0,y0,x1,y1 = self.__minsAndMaxes__(_x,_y,x0,y0,x1,y1)
                        i += 3
                    elif tokens[i] == 'l':
                        _x,_y = float(tokens[i+1]),float(tokens[i+2])
                        points.append(MyPoint(_x,_y))
                        x0,y0,x1,y1 = self.__minsAndMaxes__(_x,_y,x0,y0,x1,y1)
                        i += 3
                    elif tokens[i] == 'c':
                        for j in range(0,3):
                            _x,_y = float(tokens[i+1+j*2]),float(tokens[i+1+j*2+1])
                            points.append(MyPoint(_x,_y))
                            x0,y0,x1,y1 = self.__minsAndMaxes__(_x,_y,x0,y0,x1,y1)
                        i += 7
                    elif tokens[i] == 'z':
                        if len(points) >= 3:
                            polygon_list.append(points)
                        points = []
                        i += 1
                    else:
                        raise Exception(f'pointWithinGeospatialBounds() - do not understand svg path element "{tokens[i]}"')
                if len(points) >= 3:
                    polygon_list.append(points)
                self.__poly_cache__       [as_str] = polygon_list
                self.__poly_bounds_cache__[as_str] = (x0,y0,x1,y1)

            polygon_list = self.__poly_cache__       [as_str]
            x0,y0,x1,y1  = self.__poly_bounds_cache__[as_str]

            if point[0] >= x0 and point[0] <= x1 and point[1] >= y0 and point[1] <= y1:
                for points in polygon_list:
                    if myCheckInside(points, len(points), MyPoint(point[0],point[1])):
                        return True
                return False

        #
        # Point List Version
        #
        elif type(geospatial_bounds) == list:
            as_str = self.__pointListToString__(geospatial_bounds)

            if as_str not in self.__poly_cache__.keys():
                x0,y0,x1,y1 = None,None,None,None
                points = []
                for _pt in geospatial_bounds:
                    _x,_y = _pt[0],_pt[1]
                    points.append(MyPoint(_x,_y))
                    x0,y0,x1,y1 = self.__minsAndMaxes__(_x,_y,x0,y0,x1,y1)

                self.__poly_cache__       [as_str] = points
                self.__poly_bounds_cache__[as_str] = (x0,y0,x1,y1)

            points      = self.__poly_cache__       [as_str]
            x0,y0,x1,y1 = self.__poly_bounds_cache__[as_str]

            if point[0] >= x0 and point[0] <= x1 and point[1] >= y0 and point[1] <= y1:
                return myCheckInside(points, len(points), MyPoint(point[0],point[1]))
            else:
                return False
        
        #
        # Else Raise Exception
        #
        else:
            raise Exception(f'pointWithinGeospatialBounds() -- type "{type(geospatial_bounds)}" not recognized')

    #
    # svgLabelOnLine()
    # - render a label on a line
    #
    def svgLabelOnLine(self,
                       line=(10,10,20,20),  # x0,y0 -> x1,y1
                       text='not specified',
                       color='#ffffff',
                       offset=4,
                       txt_h=14):
        cx,cy = (line[0]+line[2])/2,(line[1]+line[3])/2

        # Unit Vector calculation
        dx,dy = line[2]-line[0],line[3]-line[1]
        _len  = sqrt(dx*dx+dy*dy)
        if _len < 0.001:
            _len = 0.001
        dx,dy = dx/_len,dy/_len
        pdx,pdy = dy,-dx

        if abs(dx) < 0.001:
            degrees = -90
        else:
            degrees = 180*atan(dy/dx)/pi
        
        if   line[0] == line[2]:
            cx += pdx*offset
            cy += pdy*offset
            degrees = 90
        elif line[0] < line[2]:
            cx += pdx*offset
            cy += pdy*offset
        else:
            cx -= pdx*offset
            cy -= pdy*offset

        svg =  f'<text x="{cx}" text-anchor="middle" y="{cy}" font-family="{self.default_font}" fill="{color}" font-size="{txt_h}px"'
        svg += f' transform="rotate({degrees},{cx},{cy})">{text}</text>'
        return svg

    #
    # svgComicDialog()
    # - create a comic style dialog
    # - return as svg markup
    #
    def svgComicDialogue(self,
                         _point=(5,5),               # what dialog is pointing at
                         _rect =(40, 40, 400, 400),  # dialog rectange - x,y,w,h
                         _curve_pixels=15,           # amount of curvature at the edges
                         _fill='#ffffff',            # interior fill color of dialogue
                         _stroke_width=2.0,          # surrounding stroke width
                         _stroke='#000000'):         # surrounding stroke color
        svg = f'<path d="M {_point[0]} {_point[1]}'

        # rectangle edges
        _x_l  = _rect[0]
        _x_r  = _rect[0] + _rect[2]
        _y_u  = _rect[1]
        _y_d  = _rect[1] + _rect[3]

        # shorts
        _x_ls = _rect[0] + _curve_pixels
        _x_rs = _rect[0] + _rect[2] - _curve_pixels
        _y_us = _rect[1] + _curve_pixels
        _y_ds = _rect[1] + _rect[3] - _curve_pixels

        #
        # To the Left
        #
        if   _point[0] <= _rect[0]:
            # Upper Left
            if   _point[1] <   _rect[1]:
                svg += f' L {_x_ls} {_y_u}'

                # Top and Upper Right Corner
                svg += f' L {_x_rs} {_y_u}'
                svg += f' C {_x_r-_curve_pixels/2} {_y_u} {_x_r} {_y_u+_curve_pixels/2} {_x_r}  {_y_us}'

                # Right and Bottom Right Corner
                svg += f' L {_x_r}  {_y_ds}'
                svg += f' C {_x_r} {_y_d-_curve_pixels/2} {_x_r-_curve_pixels/2} {_y_d} {_x_rs} {_y_d}'

                # Bottom and Bottom Left Corner
                svg += f' L {_x_ls} {_y_d}'
                svg += f' C {_x_l+_curve_pixels/2} {_y_d} {_x_l} {_y_d-_curve_pixels/2} {_x_l}  {_y_ds}'

                # Left and Top Left Corner
                svg += f' L {_x_l}  {_y_us}'
                # svg += f' C {_x_l} {_y_us-_curve_pixels/2} {_x_l+_curve_pixels/2} {_y_u} {_x_ls}  {_y_u}'

            # Exactly Left
            elif _point[1] <=  _rect[1]+_rect[3]:

                _y_start = _point[1] - _curve_pixels/2
                _y_end   = _point[1] + _curve_pixels/2
                if _y_start < _y_us:
                    _y_start = _y_us
                    _y_end   = _y_start + _curve_pixels
                if _y_end > _y_ds:
                    _y_end   = _y_ds
                    _y_start = _y_end - _curve_pixels

                svg += f' L {_x_l} {_y_start}'

                svg += f' L {_x_l} {_y_us}'
                svg += f' C {_x_l} {_y_us-_curve_pixels/2} {_x_l+_curve_pixels/2} {_y_u} {_x_ls}  {_y_u}'

                # Top and Upper Right Corner
                svg += f' L {_x_rs} {_y_u}'
                svg += f' C {_x_r-_curve_pixels/2} {_y_u} {_x_r} {_y_u+_curve_pixels/2} {_x_r}  {_y_us}'

                # Right and Bottom Right Corner
                svg += f' L {_x_r}  {_y_ds}'
                svg += f' C {_x_r} {_y_d-_curve_pixels/2} {_x_r-_curve_pixels/2} {_y_d} {_x_rs} {_y_d}'

                # Bottom and Bottom Left Corner
                svg += f' L {_x_ls} {_y_d}'
                svg += f' C {_x_l+_curve_pixels/2} {_y_d} {_x_l} {_y_d-_curve_pixels/2} {_x_l}  {_y_ds}'

                # Left and Top Left Corner
                svg += f' L {_x_l} {_y_end}'

            # Lower Left
            else:
                svg += f' L {_x_l} {_y_ds}'
                svg += f' L {_x_l} {_y_us}'
                svg += f' C {_x_l} {_y_us-_curve_pixels/2} {_x_l+_curve_pixels/2} {_y_u} {_x_ls}  {_y_u}'

                # Top and Upper Right Corner
                svg += f' L {_x_rs} {_y_u}'
                svg += f' C {_x_r-_curve_pixels/2} {_y_u} {_x_r} {_y_u+_curve_pixels/2} {_x_r}  {_y_us}'

                # Right and Bottom Right Corner
                svg += f' L {_x_r}  {_y_ds}'
                svg += f' C {_x_r} {_y_d-_curve_pixels/2} {_x_r-_curve_pixels/2} {_y_d} {_x_rs} {_y_d}'

                # Bottom and Bottom Left Corner
                svg += f' L {_x_ls} {_y_d}'
                
        #
        # To the Right
        #
        elif _point[0] >= _rect[0]+_rect[2]:
            # Upper Right
            if   _point[1] <   _rect[1]:

                svg += f' L {_x_r} {_y_us}'

                # Right and Bottom Right Corner
                svg += f' L {_x_r}  {_y_ds}'
                svg += f' C {_x_r} {_y_d-_curve_pixels/2} {_x_r-_curve_pixels/2} {_y_d} {_x_rs} {_y_d}'

                # Bottom and Bottom Left Corner
                svg += f' L {_x_ls} {_y_d}'
                svg += f' C {_x_l+_curve_pixels/2} {_y_d} {_x_l} {_y_d-_curve_pixels/2} {_x_l}  {_y_ds}'

                # Left and Top Left Corner
                svg += f' L {_x_l}  {_y_us}'
                svg += f' C {_x_l} {_y_us-_curve_pixels/2} {_x_l+_curve_pixels/2} {_y_u} {_x_ls}  {_y_u}'

                # Top and Upper Right Corner
                svg += f' L {_x_rs} {_y_u}'
                # svg += f' C {_x_r-_curve_pixels/2} {_y_u} {_x_r} {_y_u+_curve_pixels/2} {_x_r}  {_y_us}'

            # Exactly Right
            elif _point[1] <=  _rect[1]+_rect[3]:

                _y_start = _point[1] + _curve_pixels/2
                _y_end   = _point[1] - _curve_pixels/2
                if _y_end < _y_us:
                    _y_end   = _y_us
                    _y_start = _y_end + _curve_pixels
                if _y_start > _y_ds:
                    _y_start = _y_ds
                    _y_end   = _y_start - _curve_pixels

                svg += f' L {_x_r} {_y_start}'

                # Right and Bottom Right Corner
                svg += f' L {_x_r}  {_y_ds}'
                svg += f' C {_x_r} {_y_d-_curve_pixels/2} {_x_r-_curve_pixels/2} {_y_d} {_x_rs} {_y_d}'

                # Bottom and Bottom Left Corner
                svg += f' L {_x_ls} {_y_d}'
                svg += f' C {_x_l+_curve_pixels/2} {_y_d} {_x_l} {_y_d-_curve_pixels/2} {_x_l}  {_y_ds}'

                # Left and Top Left Corner
                svg += f' L {_x_l}  {_y_us}'
                svg += f' C {_x_l} {_y_us-_curve_pixels/2} {_x_l+_curve_pixels/2} {_y_u} {_x_ls}  {_y_u}'

                # Top and Upper Right Corner
                svg += f' L {_x_rs} {_y_u}'
                svg += f' C {_x_r-_curve_pixels/2} {_y_u} {_x_r} {_y_u+_curve_pixels/2} {_x_r}  {_y_us}'

                svg += f' L {_x_r} {_y_end}'

            # Lower Right
            else:
                svg += f' L {_x_rs} {_y_d}'

                # Bottom and Bottom Left Corner
                svg += f' L {_x_ls} {_y_d}'
                svg += f' C {_x_l+_curve_pixels/2} {_y_d} {_x_l} {_y_d-_curve_pixels/2} {_x_l}  {_y_ds}'

                # Left and Top Left Corner
                svg += f' L {_x_l}  {_y_us}'
                svg += f' C {_x_l} {_y_us-_curve_pixels/2} {_x_l+_curve_pixels/2} {_y_u} {_x_ls}  {_y_u}'

                # Top and Upper Right Corner
                svg += f' L {_x_rs} {_y_u}'
                svg += f' C {_x_r-_curve_pixels/2} {_y_u} {_x_r} {_y_u+_curve_pixels/2} {_x_r}  {_y_us}'

                # Right and Bottom Right Corner
                svg += f' L {_x_r}  {_y_ds}'
                # svg += f' C {_x_r} {_y_d-_curve_pixels/2} {_x_r-_curve_pixels/2} {_y_d} {_x_rs} {_y_d}'

        #
        # Centered horizontally on th rectangle
        #   
        else:
            # Above
            if   _point[1] <   _rect[1]:
                # Determine starting and stopping points
                _x_start = _point[0] + _curve_pixels/2
                _x_end   = _point[0] - _curve_pixels/2
                if _x_start > _x_rs:
                    _x_start = _x_rs
                    _x_end   = _x_start - _curve_pixels
                if _x_end < _x_ls:
                    _x_end   = _x_ls
                    _x_start = _x_end + _curve_pixels

                svg += f' L {_x_start} {_y_u}'

                # Top and Upper Right Corner
                svg += f' L {_x_rs} {_y_u}'
                svg += f' C {_x_r-_curve_pixels/2} {_y_u} {_x_r} {_y_u+_curve_pixels/2} {_x_r}  {_y_us}'

                # Right and Bottom Right Corner
                svg += f' L {_x_r}  {_y_ds}'
                svg += f' C {_x_r} {_y_d-_curve_pixels/2} {_x_r-_curve_pixels/2} {_y_d} {_x_rs} {_y_d}'

                # Bottom and Bottom Left Corner
                svg += f' L {_x_ls} {_y_d}'
                svg += f' C {_x_l+_curve_pixels/2} {_y_d} {_x_l} {_y_d-_curve_pixels/2} {_x_l}  {_y_ds}'

                # Left and Top Left Corner
                svg += f' L {_x_l}  {_y_us}'
                svg += f' C {_x_l} {_y_us-_curve_pixels/2} {_x_l+_curve_pixels/2} {_y_u} {_x_ls}  {_y_u}'

                svg += f' L {_x_end} {_y_u}'
            # Error
            elif _point[1] <=  _rect[1]+_rect[3]:
                raise Exception("svgComicDialog() - pointing at point is within the dialogue box")
            # Below
            else:
                # Determine starting and stopping points
                _x_start = _point[0] - _curve_pixels/2
                _x_end   = _point[0] + _curve_pixels/2
                if _x_end > _x_rs:
                    _x_end   = _x_rs
                    _x_start = _x_end - _curve_pixels
                if _x_start < _x_ls:
                    _x_start = _x_ls
                    _x_end   = _x_start + _curve_pixels

                svg += f' L {_x_start} {_y_d}'

                # Bottom and Bottom Left Corner
                svg += f' L {_x_ls} {_y_d}'
                svg += f' C {_x_l+_curve_pixels/2} {_y_d} {_x_l} {_y_d-_curve_pixels/2} {_x_l}  {_y_ds}'

                # Left and Top Left Corner
                svg += f' L {_x_l}  {_y_us}'
                svg += f' C {_x_l} {_y_us-_curve_pixels/2} {_x_l+_curve_pixels/2} {_y_u} {_x_ls}  {_y_u}'

                # Top and Upper Right Corner
                svg += f' L {_x_rs} {_y_u}'
                svg += f' C {_x_r-_curve_pixels/2} {_y_u} {_x_r} {_y_u+_curve_pixels/2} {_x_r}  {_y_us}'

                # Right and Bottom Right Corner
                svg += f' L {_x_r}  {_y_ds}'
                svg += f' C {_x_r} {_y_d-_curve_pixels/2} {_x_r-_curve_pixels/2} {_y_d} {_x_rs} {_y_d}'

                svg += f' L {_x_end} {_y_d}'

        svg += f' Z" fill="{_fill}" stroke="{_stroke}" stroke-width="{_stroke_width}" />'

        return svg

    #
    # readable() - make a readable string out of a large number...
    # - aim here is for four characters (or less)
    #
    def readable(self, x):
        if abs(x) < 1.0:
            return f'{x:0.2f}'
        elif abs(x) < 1000:
            return str(x)
        elif abs(x) < 1000000:
            return f'{x/1000:.1f}k'
        elif abs(x) < 1000000000:
            return f'{x/1000000:.1f}m'
        elif abs(x) < 1000000000000:
            return f'{x/1000000000:.1f}b'
        else:
            return str(x)

    #
    # Split the string into the most to least significant parts; return as a list/array.
    #
    def __timeStringParts__(self, _str):
        _str = _str.lower()
        if   't' in _str or ' ' in _str:
            if 't' in _str:
                _parts = _str.split('t')
            else:
                _parts = _str.split(' ')
            _yyyymmdd = _parts[0].split('-')
            _hhmmss   = _parts[1].split(':')
        else:
            _yyyymmdd = _str.split('-')
            _hhmmss  = []
        
        _return = []
        _return.extend(_yyyymmdd)
        _return.extend(_hhmmss)

        return _return

    #
    # Minimum timestamp string based on the precision provided
    #
    def minTimeForStringPrecision(self, _str):
        _parts = self.__timeStringParts__(_str)
        if   len(_parts) == 1:
            return _parts[0] + '-01-01 00:00:00.000000'
        elif len(_parts) == 2:
            return _parts[0] + '-' + _parts[1] + '-01 00:00:00.000000'
        elif len(_parts) == 3:
            return _parts[0] + '-' + _parts[1] + '-' + _parts[2] + ' 00:00:00.000000'
        elif len(_parts) == 4:
            return _parts[0] + '-' + _parts[1] + '-' + _parts[2] + ' ' + _parts[3] + ':00:00.000000'
        elif len(_parts) == 5:
            return _parts[0] + '-' + _parts[1] + '-' + _parts[2] + ' ' + _parts[3] + ':' + _parts[4] + ':00.000000'
        elif len(_parts) == 6:
            _ret = _parts[0] + '-' + _parts[1] + '-' + _parts[2] + ' ' + _parts[3] + ':' + _parts[4] + ':' + _parts[5]
            if '.' not in _parts[5]:
                _ret += '.'
            _ret += ('0'*(7 - (len(_ret)-_ret.index('.'))))
            return _ret
        else:
            raise Exception(f'minTimeForStringPrecision(_str="{_str}") - too many parts')


    #
    # Maximum timestamp string based on the precision provided... we'll make this
    # into the excluded version... i.e, the maximum precision for '2020-03-02' will 
    # be '2020-03-03 00:00:00' // pick your poison
    # - minTimeForStringPrecision(x) <= ts < maxTimeForStringPrecision(y)
    #
    def maxTimeForStringPrecision(self, _str):
        _parts = self.__timeStringParts__(_str)
        if   len(_parts) == 1:
            _rd = relativedelta(years=1)
        elif len(_parts) == 2:
            _rd = relativedelta(months=1)
        elif len(_parts) == 3:
            _rd = relativedelta(days=1)
        elif len(_parts) == 4:
            _rd = relativedelta(hours=1)
        elif len(_parts) == 5:
            _rd = relativedelta(minutes=1)
        elif len(_parts) == 6:
            _rd = relativedelta(seconds=1)
        else:
            raise Exception(f'maxTimeForStringPrecision(_str="{_str}") - too many parts')
        
        _min_str = self.minTimeForStringPrecision(_str)
        _max_str = str(pd.to_datetime(_min_str) + _rd)
        if '.' not in _max_str:
            _max_str += '.000000'       
        return _max_str

    #
    # annotateEntities()
    # - produce a svg description of the supplied visualization instance with the specified annotation(s)
    #
    def annotateEntities(self,
                         vis_instance,                       # must implement getEntityPositions()
                         annotations         = None,         # (1) None == Use Stateful List; (2) List of Annotations; (3) Single Annotation
                         txt_h               = 14,           # Annotation text height
                         txt_block_h_gap     = 16,           # Annotation text block gap
                         txt_block_v_gap     = 5,            # Annotation text block gap
                         max_line_w          = 96,           # Length of annotation line in pixels
                         max_lines           = 3,            # Max # of lines of annotation to render
                         annotation_color    = 'default',    # 'default', hex-color-string, 'common_name', 'tag:<tag_name>'
                         shape_outline_size  = None,         # None -- no outlines, Int/Float -- stroke width
                         instance_fade       = 0.5,          # 0.0 == no fade, 1.0 == full fade
                         x_ins               = 10,           # x inset
                         y_ins               = 5,            # y inset
                         draw_text_border    = False,        # Draw a border around the annotation text
                         draw_background     = True,         # Draw a background behind the entire composition
                         include_common_name = True,         # The annotation description is the common name (possible concatenate)
                         include_description = False):       # The annotation description is the description (possible concatenate)
        # Force a render
        _instance_svg_ = vis_instance.renderSVG()
        _instance_svg_w_, _instance_svg_h_ = self.__extractSVGWidthAndHeight__(vis_instance)

        # Create list of possible annotations that will be used
        _possibles = []
        if   annotations is None:
             for _annotation in self.annotations_ls:
                  _possibles.append(_annotation)
        elif type(annotations) == list or type(annotations) == set:
             for _annotation in annotations:
                if   type(_annotation) == str:
                    _possibles.append(self.entityAnnotation(_annotation))
                elif type(_annotation) == int:
                    _possibles.append(self.entityAnnotation(str(_annotation)))
                elif type(_annotation) == RTAnnotation:                      
                    _possibles.append(_annotation)
                else:
                    raise Exception(f'annotateTimelineInstances() - annotations type not understood -- "{type(_annotation)}"')
        elif type(annotations) == dict:
             for _k,_v in annotations.items():
                 _possibles.append(self.entityAnnotation(str(_k),str(_v)))
        elif type(annotations) == RTAnnotation:
             _possibles.append(annotations)
        elif type(annotations) == str or type(annotations) == int:
             _possibles.append(self.entityAnnotation(str(annotations)))
        else:
             raise Exception(f'annotateTimelineInstances() - annotations parameter must be None, a list of RTAnnotation, found type = "{type(annotations)}"')

        # Refine possibles into applicables... & figure out how much space we need
        _applicables_, to_positions, cols_needed, cols_filled, common_name_to_block_h = [], {}, 0, 0, {}
        common_name_to_actual_text = {}
        for _annotation_ in _possibles:
            if _annotation_.annotationType() == 'entity':
                _positions_ = vis_instance.entityPositions(_annotation_.commonName())
                if _positions_ is not None and len(_positions_) > 0:
                    _applicables_.append(_annotation_)
                    to_positions[_annotation_.commonName()] = _positions_
                    _str_ = _annotation_.commonName() if include_common_name else ''
                    if include_description and _annotation_.description() is not None:
                        _str_ += ' - ' + _annotation_.description() if (len(_str_) > 0) else _annotation_.description()
                    common_name_to_actual_text[_annotation_.commonName()] = _str_
                    txt_w    = self.textLength(_str_, txt_h)
                    my_lines = 1 + floor(txt_w / max_line_w)
                    my_lines = min(my_lines, max_lines)
                    block_h  = my_lines*txt_h + txt_h # a little extra for the border
                    common_name_to_block_h[_annotation_.commonName()] = block_h
                    if  cols_filled == 0 and block_h > _instance_svg_h_: # case where the block_h exceeds the svg height
                        cols_needed += 1
                    elif (cols_filled + block_h) > _instance_svg_h_:     # case where the block_h + the current fill exceeds the svg height
                        cols_needed += 1
                        cols_filled  = block_h + txt_block_v_gap
                    else:                                                # add to the current fill
                        cols_filled += block_h + txt_block_v_gap
        if cols_filled > 0: # any extra?  will need another column
            cols_needed += 1

        # Allocate the svg based on the columns needed
        col_x      = {}
        col_fill_h = {}
        x          = x_ins
        to_add = (cols_needed%2)
        half = floor(cols_needed/2)
        for i in range(half):
            col_x[i]      = x
            col_fill_h[i] = 0
            x += max_line_w + txt_block_h_gap
        _instance_x_ = x
        x += _instance_svg_w_ + txt_block_h_gap
        for i in range(half+to_add):
            col_x[half+i] = x
            col_fill_h[half+i] = 0
            x += max_line_w + txt_block_h_gap
        x += x_ins

        # Create the svg
        w_annotated, h_annotated = x, _instance_svg_h_ + 2*y_ins
        svg  = [f'<svg id="entity-annotation-{random.randint(0,2**32)}" x="0" y="0" width="{w_annotated}" height="{h_annotated}" xmlns="http://www.w3.org/2000/svg">']
        if draw_background: svg.append(f'<rect x="0" y="0" width="{w_annotated}" height="{h_annotated}" fill="{self.co_mgr.getTVColor("background","default")}" />')
        svg.append(f'<svg x="{_instance_x_}" y="{y_ins}" width="{_instance_svg_w_}" height="{_instance_svg_h_}" xmlns="http://www.w3.org/2000/svg">')
        svg.append(_instance_svg_)
        svg.append('</svg>')
        if instance_fade > 0.0:
            svg.append(f'<rect x="{_instance_x_}" y="{y_ins}" width="{_instance_svg_w_}" height="{_instance_svg_h_}" fill="{self.co_mgr.getTVColor("background","default")}" opacity="{instance_fade}" />')

        # Sort annotations into columns
        col_to_common_names = {}
        for i in range(cols_needed):
            col_to_common_names[i] = []
            col_fill_h[i]          = 0   # reuse filler
        if cols_needed > 1:
            # Sort the common names horizontally by position
            h_sorter = []
            for common_name in to_positions:
                xs = []
                for position in to_positions[common_name]:
                    xs.append(position.xy()[0])
                sorted(xs)
                x_median = xs[int(len(xs)/2)]
                h_sorter.append((x_median, common_name))
            h_sorter = sorted(h_sorter)

            # Fill in the columns w/ a greedy strategy
            unallocated_common_names = []
            m                        = int(len(h_sorter)/2)-1 # middle of sorter
            if m < 0:
                m = 0
            i, j                     = m, m+1                 # middle down and middle up pointers
            col_m                    = int(cols_needed/2)-1   # middle of columns
            if col_m < 0:
                col_m = 0
            col_i, col_j             = col_m, col_m+1         # middle down and middle up columns
            while i >= 0 or j < len(h_sorter):
                # fill to start
                if i >= 0:
                    common_name = h_sorter[i][1]
                    if col_i >= 0:
                        if   col_fill_h[col_i] == 0: # case where there's no fill... and by default put the next text here
                            pass
                        elif col_fill_h[col_i] + common_name_to_block_h[common_name] > _instance_svg_h_:
                            col_i -= 1
                    if col_i >= 0:
                        col_to_common_names[col_i].append(common_name)
                        col_fill_h[col_i] += common_name_to_block_h[common_name] + txt_block_v_gap
                    else:
                        unallocated_common_names.append(common_name)
                # fill to end
                if j < len(h_sorter):
                    common_name = h_sorter[j][1]
                    if col_j < cols_needed:
                        if   col_fill_h[col_j] == 0: # case where there's no fill... and by default put the next text here
                            pass
                        elif col_fill_h[col_j] + common_name_to_block_h[common_name] > _instance_svg_h_:
                            col_j += 1
                    if col_j < cols_needed:
                        col_to_common_names[col_j].append(common_name)
                        col_fill_h[col_j] += common_name_to_block_h[common_name] + txt_block_v_gap
                    else:
                        unallocated_common_names.append(common_name)
                # work our way from middle to the ends
                i -= 1
                j += 1
        else:
            for common_name in to_positions:
                col_to_common_names[0].append(common_name)

        # Render the annotations
        for i in range(cols_needed):
            x, y = col_x[i], y_ins + txt_h
            col_common_names = col_to_common_names[i]

            # Sort by y position
            v_sorter = []
            for common_name in col_common_names:
                ys = []
                for position in to_positions[common_name]:
                    ys.append(position.xy()[1])
                sorted(ys)
                y_median = ys[int(len(ys)/2)]
                v_sorter.append((y_median, common_name))
            v_sorter = sorted(v_sorter)

            # Render them in sorted order
            for _tuple_ in v_sorter:
                common_name = _tuple_[1]
                txt = common_name_to_actual_text[common_name]
                _color_ = self.co_mgr.getColor(txt)
                _split_lines_ = self.__splitAnnotationTextIntoLines__(txt, max_line_w, max_lines, txt_h)
                y_sub = y
                for line in _split_lines_:
                    svg.append(self.svgText(self.cropText(line, txt_h, max_line_w), x, y_sub, txt_h))
                    y_sub += txt_h
                if draw_text_border:
                    svg.append(f'<rect x="{x-8}" y="{y-txt_h}" width="{max_line_w+16}" height="{6+y_sub-y}" fill="none" stroke="{_color_}" stroke-width="0.4" rx="15" />')

                # Determine the attachment point
                if x < _instance_x_:
                    x_attach = x + max_line_w + 8
                else:
                    x_attach = x - 8
                y_attach = y - txt_h + (y_sub - y)/2

                # For all positions draw a line
                for _position_ in to_positions[common_name]:
                    _color_  = self.co_mgr.getTVColor("label","defaultfg")
                    _xy_     = _position_.xy()
                    _xy_off_ = _position_.xyOffset()
                    svg.append(f'<line x1="{x_attach}" y1="{y_attach}" x2="{_xy_[0]+_xy_off_[0]+_instance_x_}" y2="{_xy_[1]+_xy_off_[1]+y_ins}" stroke="{_color_}" stroke-width="1.5" />')

                    if shape_outline_size is not None:
                        _unadorned_ = _position_.svg()
                        svg.append(f'<g transform="translate({_xy_off_[0]+_instance_x_},{_xy_off_[1]+y_ins})" stroke="{_color_}" stroke-width="{shape_outline_size}" fill="none">{_unadorned_}</g>')

                # Increment for next text block
                y += common_name_to_block_h[common_name] + txt_block_v_gap

        # Return as an svg object
        svg.append('</svg>')
        return self.svgObject(''.join(svg))

    #
    # annotateTimelineInstances()
    # - produce a svg description of the supplied visualization instance with the specified annotation(s)
    #
    def annotateTimelineInstances(self,
                                  vis_instance,                    # must implement a timestampXCoord() method (at this point, only xy and temporalbarchart)
                                  annotations         = None,      # (1) None == Use Stateful List; (2) List of Annotations; (3) Single Annotation (will only use type == 'event')
                                  txt_h               = 14,        # Annotation text height
                                  max_line_w          = 64,        # Length of annotation line in pixels
                                  max_lines           = 3,         # Max # of lines of annotation to render
                                  annotation_color    = 'default', # 'default', hex-color-string, 'common_name', 'tag:<tag_name>' 
                                  draw_text_border    = False,     # Draw a border around the annotation text
                                  include_common_name = True,      # The annotation description is the common name (possible concatenate)
                                  include_description = True):     # The annotation description is the description (possible concatenate)
        # Create the individual renderings and determine the min and max timestamps...
        _instance_svg = vis_instance.renderSVG()
        _ts0,_ts1     = vis_instance.timestampExtents()
        
        # Create list of possible annotations that will be used
        _possibles = []
        if   annotations is None:
             for _annotation in self.annotations_ls:
                  _possibles.append(_annotation)
        elif type(annotations) == list:
             for _annotation in annotations:
                  _possibles.append(_annotation)                       
        elif type(annotations) == RTAnnotation:
             _possibles.append(annotations)
        else:
             raise Exception(f'annotateTimelineInstances() - annotations parameter must be None, a list of RTAnnotation, found type = "{type(annotations)}"')
             
        # Refine the possibles down to the applicables ... has to be type event... timeframes have to be within the frame ... fill in the split lines lookup and the dimensions lookup
        _applicables, _txt_lu, _txt_dims_lu = [], {}, {}
        for _annotation in _possibles:
            if _annotation.annotationType() == 'event':
                _ts0_inst,_ts1_inst = _annotation.timestampExtents()
                if _ts1_inst < _ts0 or _ts0_inst > _ts1:
                    pass
                else:
                    _applicables.append(_annotation)
                    _txt = ''
                    if include_common_name:
                        _txt = _annotation.commonName()
                    if include_description:
                        if len(_txt) > 0:
                            _txt += ': '
                        _txt += _annotation.description()
                    _split_lines = self.__splitAnnotationTextIntoLines__(_txt, max_line_w, max_lines, txt_h)
                    _txt_lu[_annotation] = _split_lines
                    _max_w = self.textLength(_split_lines[0], txt_h)
                    for i in range(1,len(_split_lines)):
                        _w = self.textLength(_split_lines[i], txt_h)
                        if _w > _max_w:
                            _max_w = _w                         
                    _txt_dims_lu[_annotation] = (_max_w, (txt_h+1)*len(_split_lines))

        # Pursue a greedy / suboptimal strategy to place the annotations
        _placements, _bar_placements, _txt_x0, _txt_x1,_stems = [], [], {}, {}, {}
        _placements.append([])     # append first row
        _bar_placements.append([]) # append one for the top
        _bar_placements.append([]) # ... and the bottom
        _applicables = sorted(_applicables,key=lambda x: x.timestampExtents())

        # Add the first annotation to get things moving
        _add = _applicables[0]
        _placements[0].append(_add)
        _bar_placements[0].append(_add)
        _x0,_x1,_txt_w = abs(vis_instance.timestampXCoord(_add.timestampExtents()[0])),abs(vis_instance.timestampXCoord(_add.timestampExtents()[1])),_txt_dims_lu[_add][0]
        _stem = _x0 + 5
        if _stem > _x1:
            _stem = (_x0+_x1)/2
        _txt_x0[_add], _txt_x1[_add], _stems[_add] = _x0, _x0 + _txt_w, _stem

        # Go through the rest of the annotations, adding them to the first available spot found
        # ... if no spot is found, then make a new row for annotations in the _placement data structure
        for i in range(1,len(_applicables)):
            # Get next annotation and add in the x components
            _add = _applicables[i]
            _x0,_x1,_txt_w = abs(vis_instance.timestampXCoord(_add.timestampExtents()[0])),abs(vis_instance.timestampXCoord(_add.timestampExtents()[1])),_txt_dims_lu[_add][0]

            # Go through all the current placements and attempt to add it if it fits
            _it_fits, _it_fits_j = False, -1
            for j in range(0,len(_placements)):
                if _it_fits == False:
                    _last_in_row    = _placements[j][-1]
                    _last_in_row_x1 = _txt_x1[_last_in_row]
                    _possible_x0    = _last_in_row_x1 + 3*txt_h
                    if _possible_x0 > _x0 and _possible_x0 < _x1:
                        _it_fits,_it_fits_j = True,j
                        _placements[j].append(_add)

                        if _x0 < _last_in_row_x1 + 3*txt_h:
                            _x0 = _last_in_row_x1 + 3*txt_h

                        _stem = _x0 + 5
                        if _stem > _x1:
                            _stem = (_x0+_x1)/2
                        _txt_x0[_add], _txt_x1[_add], _stems[_add] = _x0, _x0 + _txt_w, _stem

            # Else it didn't fit anywhere else, add it to a new row of annotations
            if _it_fits == False:
                _placements.append([])
                _placements[-1].append(_add)
                _it_fits_j = len(_placements)-1
                _stem = _x0 + 5
                if _stem > _x1:
                    _stem = (_x0+_x1)/2
                _txt_x0[_add], _txt_x1[_add], _stems[_add] = _x0, _x0 + _txt_w, _stem

            # Repeat for the bar placements // only check the correct side
            _it_fits = False
            if (_it_fits_j%2) == 0:  # Only search the top bars
                for j in range(0,len(_bar_placements),2):
                    if _it_fits == False:
                        if len(_bar_placements[j]) > 0:
                            if _add.ts0() >= _bar_placements[j][-1].ts1():
                                _it_fits = True
                                _bar_placements[j].append(_add)
                        else:
                            _it_fits = True
                            _bar_placements[j].append(_add)
            else:                    # Only search the bottom bars
                for j in range(1,len(_bar_placements),2):
                    if _it_fits == False:
                        if len(_bar_placements[j]) > 0:
                            if _add.ts0() >= _bar_placements[j][-1].ts1():
                                _it_fits = True
                                _bar_placements[j].append(_add)
                        else:
                            _it_fits = True
                            _bar_placements[j].append(_add)
            if _it_fits == False:
                _bar_placements.append([]) # Always add in pairs -- a top and a bottom bar
                _bar_placements.append([])
                if (_it_fits_j%2) == 0:
                    _bar_placements[-2].append(_add)
                else:
                    _bar_placements[-1].append(_add)

        # Determine the maximum height of each row
        _row_h = []
        for i in range(0,len(_placements)):
            _row   = _placements[i]
            _max_h = txt_h+1
            for _applicable in _row:
                if _txt_dims_lu[_applicable][1] > _max_h:
                    _max_h = _txt_dims_lu[_applicable][1]
            _row_h.append(_max_h)
        
        # Lookup for the annotation to it's bar index
        _annotation_to_bar_i, i = {}, 0
        for _bar in _bar_placements:
            for _annotation in _bar:
                _annotation_to_bar_i[_annotation] = i
            i += 1

        # Determine the width and height of the svg instance visualization
        _instance_svg_w = self.__extractSVGWidthAndHeight__(_instance_svg)[0]
        _instance_svg_h = self.__extractSVGWidthAndHeight__(_instance_svg)[1]

        # Calculate the final geometry -- height first
        _total_h = _instance_svg_h + (len(_placements)-2)*txt_h/2  # instance svg h with a little extra for spacing
        _bar_h   = 4                                               # individual extent bar heights
        for _h in _row_h:
            _total_h += _h + txt_h/2
        _total_h += len(_bar_placements)*_bar_h
        _total_h += txt_h

        # Calculate the final geometry -- width next
        _x0 = 0
        _x1 = _instance_svg_w
        for x in _txt_x0.keys():
            if _txt_x0[x] < _x0:
                _x0 = _txt_x0[x]
            if _txt_x1[x] > _x1:
                _x1 = _txt_x1[x]
        _x0 -= txt_h/2
        _x1 += txt_h/2

        # Place the rows in y space
        annotation_to_y = {}            # annotation to it's y coordinate
        y_top           = 0             # y increment from the top as each annotation row is added
        y_bottom        = _total_h      # y decrement from the bottom as each annotation row is added
        for i in range(0,len(_placements)):
            if (i%2) == 0: # above (evens)
                for _annotation in _placements[i]:
                    annotation_to_y[_annotation] = y_top
                y_top       += (_row_h[i] + txt_h/2)
            else:          # below (odds)
                for _annotation in _placements[i]:
                    annotation_to_y[_annotation] = y_bottom - _row_h[i] - txt_h/2
                y_bottom    -= (_row_h[i] + txt_h/2)
        
        # Render the SVG
        svg  = f'<svg width="{_x1-_x0}" height="{_total_h}">'
        background_color = self.co_mgr.getTVColor('background','default')
        svg += f'<rect width="{_x1-_x0}" height="{_total_h}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'

        y_vis_top = y_top+txt_h+_bar_h*len(_bar_placements)/2 # Half above... half below
        svg += self.__overwriteSVGOriginPosition__(_instance_svg, (-_x0 + _instance_svg_w/2, y_vis_top+_instance_svg_h/2), _instance_svg_w, _instance_svg_h)

        _co = self.co_mgr.getTVColor('data','default')

        # Iterate over all of the applicable annotations
        for i in range(0,len(_applicables)):
            # Get the annotation, x/y position, and text lines to render
            _annotation = _applicables[i]
            _stem_x = _stems[_annotation]
            y  = annotation_to_y[_annotation]
            t  = _txt_lu[_annotation]

            # Determine the color of the tag
            _co = self.co_mgr.getTVColor('data','default')
            if   annotation_color == 'default':
                _co = self.co_mgr.getTVColor('data','default')
            elif annotation_color == 'common_name':
                _co = self.co_mgr.getColor(_annotation.commonName())
            elif annotation_color.startswith('#') and len(annotation_color)==7:
                _co = annotation_color
            elif annotation_color.startswith('tag:'):
                _tag = annotation_color[4:]
                _tag_value = _annotation.tagValue(_tag)
                if _tag_value is not None:
                    _co = self.co_mgr.getColor(_tag_value)

            # Draw the text border if requested
            if draw_text_border:
                svg += f'<rect x="{_txt_x0[_annotation]-_x0-txt_h}" y="{y+1}" width="{_txt_x1[_annotation]-_txt_x0[_annotation]}" height="{len(t)*(txt_h+1)}" fill-opacity="0.0" stroke="{_co}" />'

            # Annotation text
            for j in range(0,len(t)):
                svg += self.svgText(t[j], _txt_x0[_annotation], y + (j+1)*txt_h, txt_h=txt_h)

            # Connecting line from the text to the bounds of the visualization
            if y < y_top: # above (evens)
                y_bracket = y_vis_top - 4 - _annotation_to_bar_i[_annotation]*_bar_h
                svg += f'<line x1="{_stem_x-_x0}" y1="{y+len(t)*(txt_h+1)}" x2="{_stem_x-_x0}" y2="{y_bracket}" stroke="{_co}" stroke-width="2" />'
            else:          # below (evens)
                y_bracket = y_vis_top + _instance_svg_h + 2 + _annotation_to_bar_i[_annotation]*_bar_h
                svg += f'<line x1="{_stem_x-_x0}" y1="{y}" x2="{_stem_x-_x0}" y2="{y_bracket}" stroke="{_co}" stroke-width="2" />'

            # Extent of the bracket
            x0_bracket = vis_instance.timestampXCoord(_annotation.timestampExtents()[0])
            x1_bracket = vis_instance.timestampXCoord(_annotation.timestampExtents()[1])

            svg += f'<line x1="{abs(x0_bracket)-_x0}" y1="{y_bracket}" x2="{abs(x1_bracket)-_x0}" y2="{y_bracket}" stroke="{_co}" stroke-width="2" />'
            if x0_bracket >= 0:
                svg += f'<line x1="{x0_bracket-_x0}" y1="{y_bracket-4}" x2="{x0_bracket-_x0}" y2="{y_bracket+4}" stroke="{_co}" stroke-width="1" />'
            else:
                x0_bracket = abs(x0_bracket)
                svg += f'<line x1="{x0_bracket-_x0}" y1="{y_bracket}" x2="{x0_bracket-_x0+4}" y2="{y_bracket+4}" stroke="{_co}" stroke-width="1" />'
                svg += f'<line x1="{x0_bracket-_x0}" y1="{y_bracket}" x2="{x0_bracket-_x0+4}" y2="{y_bracket-4}" stroke="{_co}" stroke-width="1" />'

            if x1_bracket >= 0:
                svg += f'<line x1="{x1_bracket-_x0}" y1="{y_bracket-4}" x2="{x1_bracket-_x0}" y2="{y_bracket+4}" stroke="{_co}" stroke-width="1" />'
            else:
                x1_bracket = abs(x1_bracket)
                svg += f'<line x1="{x1_bracket-_x0}" y1="{y_bracket}" x2="{x1_bracket-_x0-4}" y2="{y_bracket+4}" stroke="{_co}" stroke-width="1" />'
                svg += f'<line x1="{x1_bracket-_x0}" y1="{y_bracket}" x2="{x1_bracket-_x0-4}" y2="{y_bracket-4}" stroke="{_co}" stroke-width="1" />'
                
        svg += '</svg>'
        return self.svgObject(svg)

    #
    # annotateTimelineInstancesSubOptimal()
    # - produce a svg description of the supplied visualization instance with the specified annotation(s)
    #
    def annotateTimelineInstancesSubOptimal(self,
                                            vis_instance,                    # must implement a timestampXCoord() method (at this point, only xy and temporalbarchart)
                                            annotations         = None,      # (1) None == Use Stateful List; (2) List of Annotations; (3) Single Annotation (will only use type == 'event')
                                            txt_h               = 14,        # Annotation text height
                                            max_line_w          = 64,        # Length of annotation line in pixels
                                            max_lines           = 3,         # Max # of lines of annotation to render
                                            annotation_color    = 'default', # 'default', hex-color-string, 'common_name', 'tag:<tag_name>' 
                                            draw_text_border    = False,     # Draw a border around the annotation text
                                            include_common_name = True,      # The annotation description is the common name (possible concatenate)
                                            include_description = True):     # The annotation description is the description (possible concatenate)
        # Create the individual renderings and determine the min and max timestamps...
        _instance_svg = vis_instance.renderSVG()
        _ts0,_ts1     = vis_instance.timestampExtents()
        
        # Create list of possible annotations that will be used
        _possibles = []
        if   annotations is None:
             for _annotation in self.annotations_ls:
                  _possibles.append(_annotation)
        elif type(annotations) == list:
             for _annotation in annotations:
                  _possibles.append(_annotation)                       
        elif type(annotations) == RTAnnotation:
             _possibles.append(annotations)
        else:
             raise Exception(f'annotateTimelineInstances() - annotations parameter must be None, a list of RTAnnotation, found type = "{type(annotations)}"')
             
        # Refine the possibles down to the applicables ... has to be type event... timeframes have to be within the frame ... fill in the split lines lookup and the dimensions lookup
        _applicables, _txt_lu, _txt_dims_lu = [], {}, {}
        for _annotation in _possibles:
            if _annotation.annotationType() == 'event':
                _ts0_inst,_ts1_inst = _annotation.timestampExtents()
                if _ts1_inst < _ts0 or _ts0_inst > _ts1:
                    pass
                else:
                    _applicables.append(_annotation)
                    _txt = ''
                    if include_common_name:
                        _txt = _annotation.commonName()
                    if include_description:
                        if len(_txt) > 0:
                            _txt += ': '
                        _txt += _annotation.description()
                    _split_lines = self.__splitAnnotationTextIntoLines__(_txt, max_line_w, max_lines, txt_h)
                    _txt_lu[_annotation] = _split_lines
                    _max_w = self.textLength(_split_lines[0], txt_h)
                    for i in range(1,len(_split_lines)):
                        _w = self.textLength(_split_lines[i], txt_h)
                        if _w > _max_w:
                            _max_w = _w                         
                    _txt_dims_lu[_annotation] = (_max_w, (txt_h+1)*len(_split_lines))

        # Pursue a greedy / suboptimal strategy to place the annotations
        _placements, _bar_placements, _txt_x0, _txt_x1 = [], [], {}, {}
        _placements.append([])     # append first row
        _bar_placements.append([]) # append one for the top
        _bar_placements.append([]) # ... and the bottom
        _applicables = sorted(_applicables,key=lambda x: x.timestampExtents())

        # Add the first annotation to get things moving
        _add = _applicables[0]
        _placements[0].append(_add)
        _bar_placements[0].append(_add)
        _txt_cx       = (abs(vis_instance.timestampXCoord(_add.timestampExtents()[0])) + abs(vis_instance.timestampXCoord(_add.timestampExtents()[1])))/2
        _txt_x0[_add] = _txt_cx - _txt_dims_lu[_add][0]/2
        _txt_x1[_add] = _txt_cx + _txt_dims_lu[_add][0]/2

        # Go through the rest of the annotations, adding them to the first available spot found
        # ... if no spot is found, then make a new row for annotations in the _placement data structure
        for i in range(1,len(_applicables)):
            # Get next annotation and add in the x components
            _add = _applicables[i]
            _txt_cx       = (abs(vis_instance.timestampXCoord(_add.timestampExtents()[0])) + abs(vis_instance.timestampXCoord(_add.timestampExtents()[1])))/2
            _txt_x0[_add] = _txt_cx - _txt_dims_lu[_add][0]/2
            _txt_x1[_add] = _txt_cx + _txt_dims_lu[_add][0]/2

            # Go through all the current placements and attempt to add it if it fits
            _it_fits, _it_fits_j = False, -1
            for j in range(0,len(_placements)):
                if _it_fits == False:
                    _last_in_row    = _placements[j][-1]
                    _last_in_row_x1 = _txt_x1[_last_in_row]
                    if (_txt_x0[_add] - 2*txt_h) >= _last_in_row_x1:
                        _it_fits,_it_fits_j = True,j
                        _placements[j].append(_add)

            # Else it didn't fit anywhere else, add it to a new row of annotations
            if _it_fits == False:
                _placements.append([])
                _placements[-1].append(_add)
                _it_fits_j = len(_placements)-1
            
            # Repeat for the bar placements // only check the correct side
            _it_fits = False
            if (_it_fits_j%2) == 0:  # Only search the top bars
                for j in range(0,len(_bar_placements),2):
                    if _it_fits == False:
                        if len(_bar_placements[j]) > 0:
                            if _add.ts0() >= _bar_placements[j][-1].ts1():
                                _it_fits = True
                                _bar_placements[j].append(_add)
                        else:
                            _it_fits = True
                            _bar_placements[j].append(_add)
            else:                    # Only search the bottom bars
                for j in range(1,len(_bar_placements),2):
                    if _it_fits == False:
                        if len(_bar_placements[j]) > 0:
                            if _add.ts0() >= _bar_placements[j][-1].ts1():
                                _it_fits = True
                                _bar_placements[j].append(_add)
                        else:
                            _it_fits = True
                            _bar_placements[j].append(_add)
            if _it_fits == False:
                _bar_placements.append([]) # Always add in pairs -- a top and a bottom bar
                _bar_placements.append([])
                if (_it_fits_j%2) == 0:
                    _bar_placements[-2].append(_add)
                else:
                    _bar_placements[-1].append(_add)

        # Determine the maximum height of each row
        _row_h = []
        for i in range(0,len(_placements)):
            _row   = _placements[i]
            _max_h = txt_h+1
            for _applicable in _row:
                if _txt_dims_lu[_applicable][1] > _max_h:
                    _max_h = _txt_dims_lu[_applicable][1]
            _row_h.append(_max_h)
        
        # Lookup for the annotation to it's bar index
        _annotation_to_bar_i, i = {}, 0
        for _bar in _bar_placements:
            for _annotation in _bar:
                _annotation_to_bar_i[_annotation] = i
            i += 1

        # Determine the width and height of the svg instance visualization
        _instance_svg_w = self.__extractSVGWidthAndHeight__(_instance_svg)[0]
        _instance_svg_h = self.__extractSVGWidthAndHeight__(_instance_svg)[1]

        # Calculate the final geometry -- height first
        _total_h = _instance_svg_h + (len(_placements)-2)*txt_h/2  # instance svg h with a little extra for spacing
        _bar_h   = 4                                               # individual extent bar heights
        for _h in _row_h:
            _total_h += _h + txt_h/2
        _total_h += len(_bar_placements)*_bar_h
        _total_h += txt_h

        # Calculate the final geometry -- width next
        _x0 = 0
        _x1 = _instance_svg_w
        for x in _txt_x0.keys():
            if _txt_x0[x] < _x0:
                _x0 = _txt_x0[x]
            if _txt_x1[x] > _x1:
                _x1 = _txt_x1[x]
        _x0 -= txt_h/2
        _x1 += txt_h/2

        # Place the rows in y space
        annotation_to_y = {}            # annotation to it's y coordinate
        y_top           = 0             # y increment from the top as each annotation row is added
        y_bottom        = _total_h      # y decrement from the bottom as each annotation row is added
        for i in range(0,len(_placements)):
            if (i%2) == 0: # above (evens)
                for _annotation in _placements[i]:
                    annotation_to_y[_annotation] = y_top
                y_top       += (_row_h[i] + txt_h/2)
            else:          # below (odds)
                for _annotation in _placements[i]:
                    annotation_to_y[_annotation] = y_bottom - _row_h[i]
                y_bottom    -= (_row_h[i] + txt_h/2)
        
        # Render the SVG
        svg  = f'<svg width="{_x1-_x0}" height="{_total_h}">'
        background_color = self.co_mgr.getTVColor('background','default')
        svg += f'<rect width="{_x1-_x0}" height="{_total_h}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'

        y_vis_top = y_top+txt_h+_bar_h*len(_bar_placements)/2 # Half above... half below
        svg += self.__overwriteSVGOriginPosition__(_instance_svg, (-_x0 + _instance_svg_w/2, y_vis_top+_instance_svg_h/2), _instance_svg_w, _instance_svg_h)

        _co = self.co_mgr.getTVColor('data','default')

        # Iterate over all of the applicable annotations
        for i in range(0,len(_applicables)):
            # Get the annotation, x/y position, and text lines to render
            _annotation = _applicables[i]
            cx = (_txt_x0[_annotation] + _txt_x1[_annotation])/2
            y  = annotation_to_y[_annotation]
            t  = _txt_lu[_annotation]

            # Determine the color of the tag
            _co = self.co_mgr.getTVColor('data','default')
            if   annotation_color == 'default':
                _co = self.co_mgr.getTVColor('data','default')
            elif annotation_color == 'common_name':
                _co = self.co_mgr.getColor(_annotation.commonName())
            elif annotation_color.startswith('#') and len(annotation_color)==7:
                _co = annotation_color
            elif annotation_color.startswith('tag:'):
                _tag = annotation_color[4:]
                _tag_value = _annotation.tagValue(_tag)
                if _tag_value is not None:
                    _co = self.co_mgr.getColor(_tag_value)

            # Draw the text border if requested
            if draw_text_border:
                svg += f'<rect x="{_txt_x0[_annotation]-_x0}" y="{y+1}" width="{_txt_x1[_annotation]-_txt_x0[_annotation]}" height="{len(t)*(txt_h+1)}" fill-opacity="0.0" stroke="{_co}" />'

            # Annotation text
            for j in range(0,len(t)):
                svg += self.svgText(t[j], cx-_x0, y + (j+1)*txt_h, anchor='middle', txt_h=txt_h)

            # Connecting line from the text to the bounds of the visualization
            if y < y_top: # above (evens)
                y_bracket = y_vis_top - 4 - _annotation_to_bar_i[_annotation]*_bar_h
                svg += f'<line x1="{cx-_x0}" y1="{y+len(t)*(txt_h+1)}" x2="{cx-_x0}" y2="{y_bracket}" stroke="{_co}" stroke-width="2" />'
            else:          # below (evens)
                y_bracket = y_vis_top + _instance_svg_h + 2 + _annotation_to_bar_i[_annotation]*_bar_h
                svg += f'<line x1="{cx-_x0}" y1="{y}" x2="{cx-_x0}" y2="{y_bracket}" stroke="{_co}" stroke-width="2" />'

            # Extent of the bracket
            x0_bracket = vis_instance.timestampXCoord(_annotation.timestampExtents()[0])
            x1_bracket = vis_instance.timestampXCoord(_annotation.timestampExtents()[1])

            svg += f'<line x1="{abs(x0_bracket)-_x0}" y1="{y_bracket}" x2="{abs(x1_bracket)-_x0}" y2="{y_bracket}" stroke="{_co}" stroke-width="2" />'
            if x0_bracket >= 0:
                svg += f'<line x1="{x0_bracket-_x0}" y1="{y_bracket-4}" x2="{x0_bracket-_x0}" y2="{y_bracket+4}" stroke="{_co}" stroke-width="1" />'
            else:
                x0_bracket = abs(x0_bracket)
                svg += f'<line x1="{x0_bracket-_x0}" y1="{y_bracket}" x2="{x0_bracket-_x0+4}" y2="{y_bracket+4}" stroke="{_co}" stroke-width="1" />'
                svg += f'<line x1="{x0_bracket-_x0}" y1="{y_bracket}" x2="{x0_bracket-_x0+4}" y2="{y_bracket-4}" stroke="{_co}" stroke-width="1" />'


            if x1_bracket >= 0:
                svg += f'<line x1="{x1_bracket-_x0}" y1="{y_bracket-4}" x2="{x1_bracket-_x0}" y2="{y_bracket+4}" stroke="{_co}" stroke-width="1" />'
            else:
                x1_bracket = abs(x1_bracket)
                svg += f'<line x1="{x1_bracket-_x0}" y1="{y_bracket}" x2="{x1_bracket-_x0-4}" y2="{y_bracket+4}" stroke="{_co}" stroke-width="1" />'
                svg += f'<line x1="{x1_bracket-_x0}" y1="{y_bracket}" x2="{x1_bracket-_x0-4}" y2="{y_bracket-4}" stroke="{_co}" stroke-width="1" />'
                
        svg += '</svg>'
        return svg

    #
    # __splitAnnotationTextIntoLines__()
    # - split an annotation into multiple lines based on the parameters
    # - probably not performant...
    #
    def __splitAnnotationTextIntoLines__(self, txt, max_line_w, max_lines, txt_h):
        as_array = []
        parts = txt.split()
        line = ''
        for _part in parts:
            if len(line) == 0:
                line += _part
            elif self.textLength(line + ' ' + _part, txt_h) > max_line_w:
                as_array.append(line)
                line = _part
            else:
                line += ' ' + _part
        if len(line) > 0:
            as_array.append(line)

        if len(as_array) > max_lines:
            last   = as_array[max_lines-1]
            firsts = as_array[:max_lines-1]
            firsts.append(last + '...')
            as_array = firsts

        return as_array

#-------------------------------------------------------------------------------------------------------------------------------
#
# Source Code From Following Article:
#
# https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
#
# From this point...
#

class MyPoint:
	def __init__(self, x, y):
		self.x = x
		self.y = y

class MyLine:
	def __init__(self, p1, p2):
		self.p1 = p1
		self.p2 = p2

def myOnLine(l1, p):
	# Check whether p is on the line or not
	if (
		p.x <= max(l1.p1.x, l1.p2.x)
		and p.x <= min(l1.p1.x, l1.p2.x)
		and (p.y <= max(l1.p1.y, l1.p2.y) and p.y <= min(l1.p1.y, l1.p2.y))
	):
		return True
	return False

def myDirection(a, b, c):
	val = (b.y - a.y) * (c.x - b.x) - (b.x - a.x) * (c.y - b.y)
	if val == 0:
		# Colinear
		return 0
	elif val < 0:
		# Anti-clockwise direction
		return 2
	# Clockwise direction
	return 1

def myIsIntersect(l1, l2):
	# Four direction for two lines and points of other line
	dir1 = myDirection(l1.p1, l1.p2, l2.p1)
	dir2 = myDirection(l1.p1, l1.p2, l2.p2)
	dir3 = myDirection(l2.p1, l2.p2, l1.p1)
	dir4 = myDirection(l2.p1, l2.p2, l1.p2)

	# When intersecting
	if dir1 != dir2 and dir3 != dir4:
		return True

	# When p2 of line2 are on the line1
	if dir1 == 0 and myOnLine(l1, l2.p1):
		return True

	# When p1 of line2 are on the line1
	if dir2 == 0 and myOnLine(l1, l2.p2):
		return True

	# When p2 of line1 are on the line2
	if dir3 == 0 and myOnLine(l2, l1.p1):
		return True

	# When p1 of line1 are on the line2
	if dir4 == 0 and myOnLine(l2, l1.p2):
		return True

	return False

#
# True if within polygon
#
def myCheckInside(poly, n, p):
	# When polygon has less than 3 edge, it is not polygon
	if n < 3:
		return False

	# Create a point at infinity, y is same as point p
	exline = MyLine(p, MyPoint(9999, p.y))
	count = 0
	i = 0
	while True:
		# Forming a line from two consecutive points of poly
		side = MyLine(poly[i], poly[(i + 1) % n])
		if myIsIntersect(side, exline):
			# If side is intersects ex
			if (myDirection(side.p1, p, side.p2) == 0):
				return myOnLine(side, p);
			count += 1
		
		i = (i + 1) % n;
		if i == 0:
			break

	# When count is odd
	return count & 1;

# From
# https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/
#
# ... down to this point
#
#-------------------------------------------------------------------------------------------------------------------------------

#
# Annotation Class
#
class RTAnnotation(object):
    #
    # Constructor... just copies the parameters
    #
    def __init__(self,
                 annotation_type,             # 'event','entity','geospatial'
                 rt_self,                     # reference to racetrack application instance
                 common_str,                  # common name for the annotation
                 description_str    = None,   # human readable string for the annotation
                 timestamp_str      = None,   # begin timestamp for annotation // precision matters
                 timestamp_end_str  = None,   # if timestamp_str is set, this is the end timestamp // precision matters
                 geospatial_bounds  = None,   # either (1) list of point tuples or (2) SVG path description
                 tags               = None,   # dictionary of type-value pairs
                 entity_regex       = None,   # for entities, regex for matching to the entity // within the dataframe to visualize...
                 ignore_case        = True):  # for regex, ignore the case

        # Make a copy of the parameters
        self.annotation_type    = annotation_type
        self.rt_self            = rt_self
        self.common_str         = common_str
        self.description_str    = description_str
        self.timestamp_str      = str(timestamp_str)
        self.timestamp_end_str  = str(timestamp_end_str)
        self.geospatial_bounds  = geospatial_bounds   # should be a deep copy?
        self.tags               = tags                # should be a deep copy?
        self.entity_regex       = entity_regex
        self.ignore_case        = ignore_case

        # Determine the exact timing for event or geospatial bounds...
        if timestamp_str is not None:
            if timestamp_end_str is None:
                if type(timestamp_str) == str:
                    self.__ts0__ = pd.to_datetime(rt_self.minTimeForStringPrecision(timestamp_str))
                    self.__ts1__ = pd.to_datetime(rt_self.maxTimeForStringPrecision(timestamp_str))
                else:
                    self.__ts0__ = self.ts1 = pd.to_datetime(timestamp_str)
            else:
                if type(timestamp_str) == str:
                    self.__ts0__ = pd.to_datetime(rt_self.minTimeForStringPrecision(timestamp_str))
                else:
                    self.__ts0__ = timestamp_str
                if type(timestamp_end_str) == str:
                    self.__ts1__ = pd.to_datetime(rt_self.maxTimeForStringPrecision(timestamp_end_str))
                else:
                    self.__ts1__ = timestamp_end_str
        else:
                self.__ts0__ = None
                self.__ts1__ = None
    
    #
    # annotationType()
    # - return the annotation type
    #
    def annotationType(self):
         return self.annotation_type
    
    #
    # commonName()
    #
    def commonName(self):
         return self.common_str
    
    #
    # description()
    #
    def description(self):
         return self.description_str

    #
    # tagValue()
    #
    def tagValue(self, type):
        if self.tags is not None and type in self.tags.keys():
            return self.tags[type]
        return None

    #
    # ts0()
    #
    def ts0(self):
        return self.__ts0__
    
    #
    # ts1()
    #
    def ts1(self):
        return self.__ts1__

    #
    # timetampExtents()
    # - return the minimum and maximum timestamps as a pandas tuple
    #
    def timestampExtents(self):
         return self.__ts0__, self.__ts1__

    #
    # For lambda expression against a dataframe...
    #
    def matches(self,
                row,                      # Row from DataFrame
                entity_fields   = None,   # Single field... or list of fields to match against
                ts_field        = None,   # Timestamp field
                ts_end_field    = None,   # Optional -- for rows with a duration, this is the end timestamp field
                lat_lon_fields  = None):  # [latitude_field, longitude_field] or (latitude_field, longitude_field)
        #
        # Event Matcher
        #
        if   self.annotation_type == 'event':
            if ts_field is None:
                raise Exception('RTAnnotation.matches() - missing ts_field for filtering events')

            if ts_end_field is not None:
                if row[ts_end_field] < self.__ts0__ or row[ts_field] >= self.__ts1__:
                    return False
            else:
                if row[ts_field]     < self.__ts0__ or row[ts_field] >= self.__ts1__:
                    return False
            
            if lat_lon_fields is not None and self.geospatial_bounds is not None:
                latitude_field  = lat_lon_fields[0]
                longitude_field = lat_lon_fields[1]
                return self.rt_self.pointWithinGeospatialBounds((row[longitude_field],row[latitude_field]), self.geospatial_bounds)
            else:
                return True
        #
        # Entity Matcher
        #
        elif self.annotation_type == 'entity':
            if entity_fields is None:
                raise Exception('RTAnnotation.matches() - missing entity_fields for filtering entities')

            # Time compare first -- cheaper/easier
            if ts_field is not None and self.__ts0__ is not None:
                if ts_end_field is not None:
                    if row[ts_end_field] < self.__ts0__ or row[ts_field] >= self.__ts1__:
                        return False
                else:
                    if row[ts_field]     < self.__ts0__ or row[ts_field] >= self.__ts1__:
                        return False

            # Check for a match on the entity
            match_found = False
            if self.entity_regex is not None:
                for field in entity_fields:
                    if (self.ignore_case and re.match(self.entity_regex, row[field], re.IGNORECASE)) or re.match(self.entity_regex, row[field]):
                        match_found = True
                        break
            else:
                for field in entity_fields:
                    if self.common_str == row[field]:
                        match_found = True
                        break
            if match_found == False:
                return False
            else:
                return True
            
        #
        # Geospatial Matcher
        #
        elif self.annotation_type == 'geospatial':
            if lat_lon_fields is None or (type(lat_lon_fields) != list and type(lat_lon_fields) != tuple) or len(lat_lon_fields) != 2:
                raise Exception('RTAnnotation.matches() - missing lat_lon_fields for filtering geospatials... must be list or tuple... must be length of two')

            # Time compare first -- cheaper/easier
            if ts_field is not None and self.__ts0__ is not None:
                if ts_end_field is not None:
                    if row[ts_end_field] < self.__ts0__ or row[ts_field] >= self.__ts1__:
                        return False
                else:
                    if row[ts_field]     < self.__ts0__ or row[ts_field] >= self.__ts1__:
                        return False

            # Check the polygon bounds
            latitude_field  = lat_lon_fields[0]
            longitude_field = lat_lon_fields[1]
            return self.rt_self.pointWithinGeospatialBounds((row[longitude_field],row[latitude_field]), self.geospatial_bounds)

        else:
            raise Exception(f'RTAnnotation.matches() -- unknown annotation type "{self.annotation_type}"')
