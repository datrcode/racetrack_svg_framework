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
import networkx as nx
import random
import math
import re

from math import sqrt

from shapely.geometry import Polygon,LineString

from .rt_component import RTComponent

__name__ = 'rt_choroplethmap_mixin'

#
# Abstraction for Choropleth Map
#
class RTChoroplethMapMixin(object):
    #
    # choroplethMapPreferredDimensions()
    # - Return the preferred size
    #
    def choroplethMapPreferredDimensions(self, **kwargs):
        return (256,256)

    #
    # choroplethMapMinimumDimensions()
    # - Return the minimum size
    #
    def choroplethMapMinimumDimensions(self, **kwargs):
        return (32,32)

    #
    # choroplethMapSmallMultipleDimensions()
    # - Return the minimum size
    #
    def choroplethMapSmallMultipleDimensions(self, **kwargs):
        return (32,32)

    #
    # Identify the required fields in the dataframe from linknode parameters
    #
    def choroplethMapRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('shape_field', kwargs, columns_set)
        self.identifyColumnsFromParameters('shape_lu',    kwargs, columns_set)
        self.identifyColumnsFromParameters('color_by',    kwargs, columns_set)
        self.identifyColumnsFromParameters('count_by',    kwargs, columns_set)
        return columns_set

    #
    # choroplethMap
    #
    # Make the SVG for a choropleth map
    #    
    def choroplethMap(self,
                      df,                                     # dataframe to render
                      shape_field,                            # dataframe field to use for shape_lu
                      shape_lu,                               # shape lookup ... should be the value in the shape_field to a shapely Polygon
                      view_window               = None,       # (wx0, wy0, wx1, wy1) // if none, will be derived from the shape lu's parameter
                      bounds_from_all_shapes    = True,       # if false, will only use what's rendered as the calculation (assumes the view_window is None)
                      bounds_percent            = 0.05,       # inset the view into the view by this percent... so that the shapes aren't right at the edges                       
                      outline_all_shapes        = True,       # if true, renders outlines for all shapes in the lookup
                      draw_outlines             = True,       # draw outlines (at all)
                      color_by                  = None,       # NOT USED... Here For Compatibility...
                      count_by                  = None,       # none means just count rows, otherwise, use a field to sum by
                      count_by_set              = False,      # count by summation (by default)... count_by column is checked
                      widget_id                 = None,       # naming the svg elements                 
                      label_only                = None,       # label only set ... "None" means label all
                      global_max                = None,       # max amounts (important for small multiples)
                      global_min                = None,       # min amounts (important for small multiples)                         
                      # ------------------------------------- # background polygons // copied mostly from the xy implementation
                      bg_shape_lu               = None,       # lookup for background shapes -- key will be used for varying colors (if bg_shape_label_color == 'vary')
                                                              # ['key'] = [(x0,y0),(x1,y1),...] OR
                                                              # ['key'] = svg path description
                      bg_shape_label_color      = None,       # None = no label, 'vary', lookup to hash color, or a hash color
                      bg_shape_opacity          = 1.0,        # None (== 0.0), number, lookup to opacity
                      bg_shape_fill             = None,       # None, 'vary', lookup to hash color, or a hash color
                      bg_shape_stroke_w         = 1.0,        # None, number, lookup to width
                      bg_shape_stroke           = 'default',  # None, 'default', lookup to hash color, or a hash color
                      # ------------------------------------- # small multiple options
                      sm_type                  = None,        # should be the method name // similar to the smallMultiples method
                      sm_w                     = None,        # override the width of the small multiple
                      sm_h                     = None,        # override the height of the small multiple
                      sm_params                = {},          # dictionary of parameters for the small multiples
                      sm_x_axis_independent    = True,        # Use independent axis for x (xy, temporal, and linkNode)
                      sm_y_axis_independent    = True,        # Use independent axis for y (xy, temporal, periodic, pie)
                      # ------------------------------------- # visualization geometry / etc.
                      track_state              = False,       # track state for interactive filtering
                      x_view                   = 0,           # x offset for the view
                      y_view                   = 0,           # y offset for the view
                      w                        = 256,         # width of the view
                      h                        = 256,         # height of the view
                      x_ins                    = 3,
                      y_ins                    = 3,
                      txt_h                    = 12,          # text height for labeling
                      draw_labels              = True,        # draw labels flag # not implemented yet
                      draw_border              = True):       # draw a border around the graph
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTChoroplethMap(self, **_params_)

    #
    # RTChoroplethMap Class
    #
    class RTChoroplethMap(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     **kwargs):
            self.parms                      = locals().copy()
            self.rt_self                    = rt_self
            self.df                         = rt_self.copyDataFrame(kwargs['df'])
            self.shape_field                = kwargs['shape_field']
            self.shape_lu                   = kwargs['shape_lu']
            self.view_window                = kwargs['view_window']
            self.view_window_orig           = kwargs['view_window'] # Orig will be used for user requests to reset the view

            self.bounds_from_all_shapes     = kwargs['bounds_from_all_shapes']
            self.bounds_percent             = kwargs['bounds_percent']

            self.outline_all_shapes         = kwargs['outline_all_shapes']
            self.draw_outlines              = kwargs['draw_outlines']

            self.color_by                   = kwargs['color_by']      # Not used?  More for just conformity for other components
            self.count_by                   = kwargs['count_by']
            self.count_by_set               = kwargs['count_by_set']
            self.widget_id                  = kwargs['widget_id']

            self.label_only                 = kwargs['label_only']

            self.global_max                 = kwargs['global_max']
            self.global_min                 = kwargs['global_min']

            self.bg_shape_lu                = kwargs['bg_shape_lu']           # Copied from xy implementation --vvv
            self.bg_shape_label_color       = kwargs['bg_shape_label_color']
            self.bg_shape_opacity           = kwargs['bg_shape_opacity']
            self.bg_shape_fill              = kwargs['bg_shape_fill']
            self.bg_shape_stroke_w          = kwargs['bg_shape_stroke_w']
            self.bg_shape_stroke            = kwargs['bg_shape_stroke']       # Copied from xy implementation --^^^

            self.sm_type                    = kwargs['sm_type']
            self.sm_w                       = kwargs['sm_w']
            self.sm_h                       = kwargs['sm_h']
            self.sm_params                  = kwargs['sm_params']
            self.sm_x_axis_independent      = kwargs['sm_x_axis_independent']
            self.sm_y_axis_independent      = kwargs['sm_y_axis_independent']

            self.track_state                = kwargs['track_state']

            self.x_view                     = kwargs['x_view']
            self.y_view                     = kwargs['y_view']
            self.w                          = kwargs['w']
            self.h                          = kwargs['h']
            self.x_ins                      = kwargs['x_ins']
            self.y_ins                      = kwargs['y_ins']

            self.txt_h                      = kwargs['txt_h']
            self.draw_labels                = kwargs['draw_labels']
            self.draw_border                = kwargs['draw_border']

            if self.widget_id is None:
                self.widget_id = "choroplethmap_" + str(random.randint(0,65535))

            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)

            # Check the count_by column across all the df's...  if any of them
            # don't work.. then it's count_by_set
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)
            
            # Tracking state
            self.geom_to_df  = {}
            self.last_render = None

        #
        # __calculateGeometry__() - determine the geometry for the view
        #
        def __calculateGeometry__(self):
            # Calculate world coordinates
            self.wx0 =  math.inf
            self.wy0 =  math.inf
            self.wx1 = -math.inf
            self.wy1 = -math.inf

            consider_set = set(self.shape_lu.keys()) if self.bounds_from_all_shapes else set(self.df[self.shape_field])
            for k in consider_set:
                _bounds_ = self.shape_lu[k].bounds
                self.wx0 = min(self.wx0, _bounds_[0])
                self.wy0 = min(self.wy0, _bounds_[1])
                self.wx1 = max(self.wx1, _bounds_[2])
                self.wy1 = max(self.wy1, _bounds_[3])

            # Make it sane some more
            if self.wx0 == self.wx1:
                self.wx0 -= 0.5
                self.wx1 += 0.5
            if self.wy0 == self.wy1:
                self.wy0 -= 0.5
                self.wy1 += 0.5

            # Give some air around the boundaries
            if self.bounds_percent != 0:
                in_x = (self.wx1-self.wx0)*self.bounds_percent
                self.wx0 -= in_x
                self.wx1 += in_x
                in_y = (self.wy1-self.wy0)*self.bounds_percent
                self.wy0 -= in_y
                self.wy1 += in_y

        #
        # __renderChoroplethMap__() - render the choropleth map
        #
        def __renderChoroplethMap__(self):
            _svg_, _labels_ = [],[]

            # Render all shape outlines
            _label_co_ , _co_ = self.rt_self.co_mgr.getTVColor('label','defaultfg'), self.rt_self.co_mgr.getTVColor('axis','default')
            if self.draw_outlines and self.outline_all_shapes:
                to_render = set(self.shape_lu.keys()) - set(self.df[self.shape_field])
                for k in to_render:
                    _shape_svg_, _label_svg_ = self.rt_self.__transformBackgroundShapes__(k, self.shape_lu[k], self.xT, self.yT, 
                                                                                          _label_co_, 1.0, 'none', 1.0, _co_, self.txt_h)
                    _svg_.append(_shape_svg_)
                    if self.label_only is None or k in self.label_only: _labels_.append(_label_svg_)

            # Calculate the intensity
            if   self.rt_self.isPandas(self.df):
                if   self.count_by is None: _ = self.df.groupby(self.shape_field).size().reset_index().rename({0:'_count_'},axis=1)
                elif self.count_by_set:     _ = self.df.groupby(self.shape_field)[self.count_by].nunique().reset_index().rename({self.count_by:'_count_'},axis=1)
                else:                       _ = self.df.groupby(self.shape_field)[self.count_by].sum().reset_index().rename({self.count_by:'_count_'},axis=1)
                _min_,_max_  = _['_count_'].min(),_['_count_'].max()
                if self.global_max is not None and self.global_min is not None:
                    _max_ = self.global_max
                    _min_ = self.global_min
                if _min_ == _max_:
                    _min_ -= 0.5
                    _max_ += 0.5
                _['_count_'] = (_['_count_'] - _min_) / (_max_ - _min_)
            elif self.rt_self.isPolars(self.df):
                if self.count_by is None: _ = self.df.group_by(self.shape_field).agg(pl.count().alias('_count_'))
                elif self.count_by_set:   _ = self.df.group_by(self.shape_field).agg(pl.col(self.count_by).n_unique().alias('_count_'))
                else:                     _ = self.df.group_by(self.shape_field).agg(pl.col(self.count_by).sum().alias('_count_'))
                _min_,_max_ = _['_count_'].min(), _['_count_'].max()
                if self.global_max is not None and self.global_min is not None:
                    _max_ = self.global_max
                    _min_ = self.global_min
                if _min_ == _max_:
                    _min_ -= 0.5
                    _max_ += 0.5
                _ = _.with_columns((pl.col('_count_') - _min_) / (_max_ - _min_))
            else:
                raise Exception('RTChoroplethMap.__renderChoroplethMap__() - only pandas and polars supported')

            # Render the shapes
            if self.track_state:
                if self.rt_self.isPandas(self.df):   gb = self.df.groupby(self.shape_field)
                elif self.rt_self.isPolars(self.df): gb = self.df.partition_by(self.shape_field, as_dict=True)

            for i in range(len(_)):
                if _[self.shape_field][i] in self.shape_lu.keys():
                    _intensity_co_ = self.rt_self.co_mgr.spectrum(_['_count_'][i], 0.0, 1.0)
                    _co_ = 'none' if not self.draw_outlines else _co_
                    _shape_svg_, _label_svg_ = self.rt_self.__transformBackgroundShapes__(_[self.shape_field][i], self.shape_lu[_[self.shape_field][i]], self.xT, self.yT, 
                                                                                          _label_co_, 1.0, _intensity_co_, 1.0, _co_, self.txt_h)
                    _svg_.append(_shape_svg_)
                    if self.label_only is None or _[self.shape_field][i] in self.label_only:                        
                        _labels_.append(_label_svg_)
                    if self.track_state:
                        self.geom_to_df[self.shape_lu[_[self.shape_field][i]]] = gb.get_group(_[self.shape_field][i]) if self.rt_self.isPandas(self.df) else gb[_[self.shape_field][i]]

            return ''.join(_svg_), ''.join(_labels_), _min_, _max_

        #
        # __renderBackgroundShapes__() - render background shapes
        # - mostly a copy of the xy implementation
        #
        def __renderBackgroundShapes__(self):
            _svg_ = []
            # Draw the background shapes
            if self.bg_shape_lu is not None:
                _bg_shape_labels = []
                for k in self.bg_shape_lu.keys():
                    shape_desc = self.bg_shape_lu[k]
                    _shape_svg, _label_svg = self.__transformBackgroundShapes__(k,                         shape_desc,
                                                                                self.xT,                   self.yT,
                                                                                self.bg_shape_label_color, self.bg_shape_opacity,
                                                                                self.bg_shape_fill,        self.bg_shape_stroke_w,
                                                                                self.bg_shape_stroke,      self.txt_h)
                    _svg_.append(_shape_svg)
                    _bg_shape_labels.append(_label_svg) # Defer render

                # Render the labels
                for _label_svg in _bg_shape_labels: _svg_.append(_label_svg)
            
            return ''.join(_svg_)

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None: self.renderSVG()
            return self.last_render
        
        #
        # applyScrollEvent()
        # - zoom in or out based on the specified coordinate.
        #
        def applyScrollEvent(self, scroll_amount, coordinate=None):
            scroll_amount = scroll_amount / 1000.0
            if coordinate is not None:
                coord_wx = self.xT_inv(coordinate[0])
                coord_wy = self.yT_inv(coordinate[1])
                coordinate = (coord_wx, coord_wy)
            self.setViewWindow(self.rt_self.viewWindowZoom(self.view_window, scroll_amount, coordinate))
            return True

        #
        # applyMiddleClick()
        # - reset the view
        #
        def applyMiddleClick(self, coordinate):
            if self.view_window != self.view_window_orig:
                self.setViewWindow(self.view_window_orig)
                return True
            return False

        #
        # applyMiddleDrag()
        # - draw the view
        #        
        def applyMiddleDrag(self, coordinate, delta):
            if self.view_window is not None:
                wx0,wy0,wx1,wy1 = self.xT_inv(coordinate[0]), self.yT_inv(coordinate[1]),self.xT_inv(coordinate[0]+delta[0]), self.yT_inv(coordinate[1]+delta[1])
                dwx,dwy         = wx1-wx0, wy1-wy0
                self.setViewWindow((self.view_window[0]+dwx, self.view_window[1]+dwy, self.view_window[2]+dwx, self.view_window[3].dwy))
                return True
            return False

        #
        # applyViewConfiguration()
        # - adjust the view window based on the other view window
        # - return True if the view actually changed (and needs a re-render)
        #
        def applyViewConfiguration(self, other):
             other_view_window = other.getViewWindow()
             if other_view_window != self.getViewWindow():
                 self.setViewWindow(other_view_window)
                 return True
             return False

        #
        # setViewWindow() - Set the view window and set flag to re-render on next call to _repr_svg_()
        # - will force a re-render on next call to _repr_svg_()
        #         
        def setViewWindow(self, view_window):
            self.view_window = view_window
            self.last_render = None
        
        #
        # getViewWindow() - return the current view window
        #
        def getViewWindow(self):
            return self.view_window

        #
        # renderSVG() - render as SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state:
                self.geom_to_df = {}

            # Determine geometry
            if self.view_window is None:
                self.__calculateGeometry__()
                self.view_window = self.view_window_orig = (self.wx0, self.wy0, self.wx1, self.wy1)
            else:
                self.wx0, self.wy0, self.wx1, self.wy1 = self.view_window
                
            # Coordinate transform lambdas (and inverse lambdas)
            self.xT = lambda __x__: self.x_ins + (self.w - 2*self.x_ins) * (__x__ - self.wx0)/(self.wx1-self.wx0)
            self.yT = lambda __y__: (self.h - self.y_ins) - (self.h - 2*self.y_ins) * (__y__ - self.wy0)/(self.wy1-self.wy0)

            self.xT_inv = lambda __sx__: self.wx0 + (self.wx1 - self.wx0) * (__sx__ - self.x_ins)/(self.w - 2*self.x_ins)
            self.yT_inv = lambda __sy__: -1.0 * (__sy__ - (self.h - self.y_ins)) * (self.wy1-self.wy0) / (self.h - 2*self.y_ins) + self.wy0

            # Start the SVG Frame
            svg = [f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">']
            background_color = self.rt_self.co_mgr.getTVColor('background','default')
            svg.append(f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />')

            # Render background shapes, convex hulls, links, and then nodes
            _svg_, _labels_, _min_, _max_ = self.__renderChoroplethMap__()
            if just_calc_max: return _min_, _max_
            svg.append(_svg_)
            svg.append(self.__renderBackgroundShapes__())

            # Defer label render
            if self.draw_labels: svg.append(_labels_)

            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg.append(f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill-opacity="0.0" stroke="{border_color}" />')

            svg.append('</svg>')
            self.last_render = ''.join(svg)
            return self.last_render

        #
        # Determine which dataframe geometries overlap with a specific
        #
        def overlappingDataFrames(self, to_intersect):
            pts_w = []
            for pt_s in to_intersect.exterior.coords:
                pt_w = (self.xT_inv(pt_s[0]), self.yT_inv(pt_s[1]))
                pts_w.append(pt_w)
            to_intersect = Polygon(pts_w)

            _dfs = []
            for _poly in self.geom_to_df.keys():
                if _poly.intersects(to_intersect):
                    _dfs.append(self.geom_to_df[_poly]) # <== SLIGHTLY DIFFERENT THAN ALL OF THE OTHERS...
            if len(_dfs) > 0:
                return self.rt_self.concatDataFrames(_dfs)
            else:
                return None
