import pandas as pd
import numpy as np
import rtsvg
rt = rtsvg.RACETrack()
from shapely.geometry import Polygon, Point
from math import cos, sin, pi, atan2

__name__ = 'scu_pyramid_method_diagram'

#
# Visualization of the Pyramid Method as described in the following paper:
#
# Evaluating Content Selection in Summarization: The Pyramid Method
# Ani Nenkova and Rebecca Passonneau
# Columbia University
#
# https://aclanthology.org/N04-1019.pdf
#
class SCUPyramidMethodDiagram(object):
    #
    # __init__()
    # - copy the input variables to member variables
    # - compute the tabularized version of the input table
    # - compute the pyramid levels (per q_id)
    #
    def __init__(self, rt_self, df, q_id_field, scu_field, summary_source_field, 
                 txt_h=16, level_h_min = 16, r_scu = 4.0, 
                 draw_q_id_label=True, q_id_multiple=3.0,
                 tri_inset=12, x_ins=16, y_ins=16, w=256, h=256):
        _columns_to_drop_ = set(df.columns) - set([q_id_field, scu_field, summary_source_field])
        self.df_tab = df.groupby([q_id_field, scu_field])                          \
                        .nunique()                                                 \
                        .rename(columns={summary_source_field:'occurences'})       \
                        .sort_values(by=['occurences',scu_field], ascending=False) \
                        .reset_index()
        self.rt_self              = rt_self
        self.df                   = df
        self.q_id_field           = q_id_field
        self.scu_field            = scu_field
        self.summary_source_field = summary_source_field
        self.pyramid_levels       = {}                     # pyramid_levels[q_id] = # of pyramid levels
        for k, k_df in df.groupby(q_id_field):
            q_id                      = k_df[q_id_field].unique()[0]
            self.pyramid_levels[q_id] = len(k_df[summary_source_field].unique())

        # Render Options
        self.draw_q_id_label = draw_q_id_label
        self.level_h_min     = level_h_min
        self.tri_inset       = tri_inset
        self.q_id_multiple   = q_id_multiple
        self.r_scu           = r_scu

        # Geometry Information
        self.w, self.h, self.txt_h, self.x_ins, self.y_ins = w, h, txt_h, x_ins, y_ins
        self.top_xy       = (self.w/2.0, self.y_ins)
        self.b_left_xy    = (self.x_ins,          self.h - self.y_ins - self.txt_h)
        self.b_right_xy   = (self.w - self.x_ins, self.h - self.y_ins - self.txt_h)
        self.tri_path     = f'M {self.top_xy[0]} {self.top_xy[1]} L {self.b_left_xy[0]} {self.b_left_xy[1]} L {self.b_right_xy[0]} {self.b_right_xy[1]} Z'
        self.tri_h        = self.b_left_xy[1] - self.top_xy[1]

        self.top_xy_inner     = (self.top_xy[0],                      self.top_xy[1]     + self.tri_inset)
        self.b_left_xy_inner  = (self.b_left_xy[0]  + self.tri_inset, self.b_left_xy[1]  - self.tri_inset/2.0)
        self.b_right_xy_inner = (self.b_right_xy[0] - self.tri_inset, self.b_right_xy[1] - self.tri_inset/2.0)
        self.tri_path_inner   = f'M {self.top_xy_inner[0]} {self.top_xy_inner[1]} L {self.b_left_xy_inner[0]} {self.b_left_xy_inner[1]} L {self.b_right_xy_inner[0]} {self.b_right_xy_inner[1]} Z'

        # Layout Information (filled in by __computeLayout__)
        self.scu_to_xy = {} # scu_to_xy[q_id][scy] = (x,y)

    #
    # __calculateXBoundsForTriangleLevel__() - calculate the left and right boundaries for a specific y value
    #
    def __calculateXBoundsForTriangleLevel__(self, y):
        _segment_ = ((0,y),(self.w,y))
        _tuple0_  = self.rt_self.segmentsIntersect(_segment_, (self.top_xy, self.b_left_xy))
        _tuple1_  = self.rt_self.segmentsIntersect(_segment_, (self.top_xy, self.b_right_xy))
        return _tuple0_[1], _tuple1_[1]

    #
    # __pointsWithinPoly__() - fill in point within a polygon
    #
    def __pointsWithinPoly__(self, xy, poly, r_max, n):
        r_max_formula = np.sqrt(n)
        _golden_ratio_ = (1 + np.sqrt(5)) / 2
        xys = []
        for i in range(100):
            _angle_  = i * 2 * np.pi / _golden_ratio_
            _radius_ = r_max * np.sqrt(i) / r_max_formula
            _xy_ = (xy[0] + _radius_ * np.cos(_angle_), xy[1] + _radius_ * np.sin(_angle_))
            if Point(_xy_).within(poly): xys.append(_xy_)
        return xys

    def __randomLayout__(self, poly, n, inter_min):
        xys         = []
        x0,y0,x1,y1 = poly.bounds
        for i in range(n):
            for _try_ in range(10+2*len(xys)):
                _xy_ = (x0 + np.random.rand() * (x1-x0), y0 + np.random.rand() * (y1-y0))
                if Point(_xy_).within(poly):
                    too_close = False
                    for xy in xys:
                        if np.sqrt((xy[0] - _xy_[0])**2 + (xy[1] - _xy_[1])**2) < inter_min:
                            too_close = True
                            break
                    if too_close == False:
                        xys.append(_xy_)
                        break
        return xys

    #
    # __computeLayout__() - compute the layout of the pyramid for a specific question_id
    #
    def __computeLayout__(self, q_id):
        self.scu_to_xy [q_id] = {}
        self.level_bars       = {}
        self.level_bars[q_id] = {}
        self.mid_bars         = {}
        self.mid_bars  [q_id] = {}
        levels               = self.pyramid_levels[q_id] # level 0 is the base level, level 1 is the next up... level n-1 is the top level
        scu_count_at_level   = {}
        scus_at_level        = {}
        levels_w_zero        = 0
        for level in range(0,levels):
            level_plus_1 = level + 1
            _df_ = self.df_tab.query(f'`{self.q_id_field}` == @q_id and occurences == @level_plus_1')
            scu_count_at_level[level] = len(_df_)
            scus_at_level[level]      = list(_df_[self.scu_field])
            if scu_count_at_level[level] == 0: levels_w_zero += 1
        # Calculate the height of each level (except for the empty levels which will be level_h_min)
        level_h = (self.tri_h - levels_w_zero * self.level_h_min) / (levels - levels_w_zero)
        y_base  = self.b_left_xy[1]
        for level in range(0, levels):
            _h_     = level_h if scu_count_at_level[level] > 0 else self.level_h_min
            bot_x0, bot_x1      = self.__calculateXBoundsForTriangleLevel__(y_base-self.tri_inset)
            y_base_last         = y_base
            y_base             -= _h_
            top_x0,    top_x1   = self.__calculateXBoundsForTriangleLevel__(y_base+self.tri_inset)
            level_x0, level_x1  = self.__calculateXBoundsForTriangleLevel__(y_base)
            self.level_bars[q_id][level] = ((level_x0, y_base),(level_x1, y_base))
            y_mid               = (y_base_last + y_base) / 2
            mid_x0,   mid_x1    = self.__calculateXBoundsForTriangleLevel__(y_mid)
            self.mid_bars  [q_id][level] = ((mid_x0, y_mid), (mid_x1, y_mid))
            _poly_          = Polygon([(bot_x0+self.tri_inset, y_base_last-self.tri_inset/2.0),
                                       (bot_x1-self.tri_inset, y_base_last-self.tri_inset/2.0),
                                       (top_x1-self.tri_inset, y_base     +self.tri_inset/2.0),
                                       (top_x0+self.tri_inset, y_base     +self.tri_inset/2.0)])
            if level == levels-1:
                _poly_          = Polygon([(bot_x0+self.tri_inset, y_base_last-self.tri_inset/2.0),
                                           (bot_x1-self.tri_inset, y_base_last-self.tri_inset/2.0),
                                           ((top_x0+top_x1)/2.0,   y_base     +self.tri_inset),    
                                           ((top_x0+top_x1)/2.0,   y_base     +self.tri_inset)])

            if len(scus_at_level[level]) > 3:
                attempts, inter_min = 1, 6.0
                xys = self.__randomLayout__(_poly_, len(scus_at_level[level]), inter_min)
                while attempts < 100 and len(xys) < len(scus_at_level[level]):
                    inter_min *= 0.9
                    xys = self.__randomLayout__(_poly_, len(scus_at_level[level]), inter_min)
                if len(xys) < len(scus_at_level[level]): xys = self.__randomLayout__(_poly_, scus_at_level[level], 0.0)

                ''' # this doesn't really produce the right density (across higher values)
                my_r_max, my_n = self.w*2, 1000
                while my_r_max > (bot_x1 - bot_x0)/2.0:
                    xys = self.__pointsWithinPoly__((self.w/2.0, (y_base_last+y_base)/2.0), _poly_, my_r_max, my_n)
                    if len(xys) >= len(scus_at_level[level]): break
                    my_r_max  = int(my_r_max - 1)
                    my_n      = int(my_n * 2)
                '''
            else:
                if   len(scus_at_level[level]) == 1 or len(scus_at_level[level]) == 0:
                    xys = [(self.w/2.0, (y_base_last+y_base)/2.0)]
                elif len(scus_at_level[level]) == 2:
                    _x0_, _x1_ = self.__calculateXBoundsForTriangleLevel__((y_base_last+y_base)/2.0)
                    d          = (_x1_ - _x0_)/3.0
                    xys = [(_x0_+d, (y_base_last+y_base)/2.0), 
                           (_x1_-d, (y_base_last+y_base)/2.0)]
                elif len(scus_at_level[level]) == 3:
                    _x0_, _x1_ = self.__calculateXBoundsForTriangleLevel__(2*(y_base_last+y_base)/3.0)
                    d          = (_x1_ - _x0_)/3.0
                    xys = [(self.w/2.0, 2*(y_base_last+y_base)/4.0), 
                           (_x0_+d, 2*(y_base_last+y_base)/3.0), 
                           (_x1_-d, 2*(y_base_last+y_base)/3.0)]
            if len(scus_at_level[level]) > len(xys): 
                raise Exception(f'length of scus ({len(scus_at_level[level])}) is greater than length of xys ({len(xys)})')
            for i in range(len(scus_at_level[level])):
                _scu_, _xy_ = scus_at_level[level][i], xys[i]
                self.scu_to_xy[q_id][_scu_] = _xy_

    #
    # svgPyramid() - return an SVG representation of the pyramid for a specific question_id
    # - if summary_source is None, then all summary sources are included
    # - if summary source is set, then only that summary source is included
    #
    def svgPyramid(self, q_id, summary_source=None):
        if q_id not in self.scu_to_xy: self.__computeLayout__(q_id)
        levels = self.pyramid_levels[q_id]

        # SVG Setup
        _svg_ = [f'<svg x="0" y="0" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">']
        _svg_.append(f'<rect x="0" y="0" width="{self.w}" height="{self.h}" fill="{self.rt_self.co_mgr.getTVColor("background","default")}" />')

        # Draw the Pyramid Levels
        for level in range(0, levels):
            _xy0_, _xy1_ = self.level_bars[q_id][level]
            _svg_.append(f'<line x1="{_xy0_[0]}" y1="{_xy0_[1]}" x2="{_xy1_[0]}" y2="{_xy1_[1]}" stroke="{self.rt_self.co_mgr.getTVColor("axis","default")}" stroke-width="2" />')
        
        # Draw # of SCUs at each level
        for level in range(0, levels):
            level_plus_1 = level + 1
            _df_         = self.df_tab.query(f'`{self.q_id_field}` == @q_id and occurences == @level_plus_1')
            _count_      = len(_df_)
            _xy0_, _xy1_ = self.mid_bars[q_id][level]
            _rotation_   = atan2(self.top_xy[1] - _xy0_[1], self.top_xy[0] - _xy0_[0]) / (pi/180.0)
            _uv_         = self.rt_self.unitVector((self.top_xy, _xy0_))
            _perp_       = (-_uv_[1], _uv_[0])
            _svg_.append(self.rt_self.svgText(f'{_count_}', _xy0_[0] + _perp_[0]*3, _xy0_[1] + _perp_[1]*3, txt_h=self.txt_h, 
                                              color=self.rt_self.co_mgr.getTVColor('axis','default'), 
                                              anchor='middle', rotation=_rotation_))

        # Draw ALL the SCUs -- if the summary source is not set, then draw them as a little grayed out
        _color_ = self.rt_self.co_mgr.getTVColor('data','default') if summary_source is None else self.rt_self.co_mgr.getTVColor('context','default')
        for _scu_ in set(self.df_tab.query(f'`{self.q_id_field}` == @q_id')[self.scu_field]):
            _xy_ = self.scu_to_xy[q_id][_scu_]
            _svg_.append(f'<circle cx="{_xy_[0]}" cy="{_xy_[1]}" r="{self.r_scu}" fill="{_color_}" stroke="none" />')

        # For the summary source, re-draw the points
        if summary_source is not None:
            _df_    = self.df.query(f'`{self.q_id_field}` == @q_id and `{self.summary_source_field}` == @summary_source')
            _color_ = self.rt_self.co_mgr.getColor(summary_source)
            for _scu_ in set(_df_[self.scu_field]):
                _xy_ = self.scu_to_xy[q_id][_scu_]
                _svg_.append(f'<circle cx="{_xy_[0]}" cy="{_xy_[1]}" r="{self.r_scu}" fill="{_color_}" stroke="{self.rt_self.co_mgr.getTVColor("axis","default")}" />')
            # Add text about how many scus for this source
            _on_ = [self.q_id_field,self.scu_field]
            _df_ = self.df.set_index(_on_).join(self.df_tab.set_index(_on_), how='left', lsuffix='_left', rsuffix='_right').reset_index()
            for level in range(0, levels):
                level_plus_1 = level + 1
                _count_ = _df_.query(f'`{self.summary_source_field}` == @summary_source and occurences == @level_plus_1')[self.scu_field].nunique()
                _xy0_, _xy1_ = self.mid_bars[q_id][level]
                _rotation_   = atan2(self.top_xy[1] - _xy1_[1], self.top_xy[0] - _xy1_[0]) / (pi/180.0)
                _uv_         = self.rt_self.unitVector((self.top_xy, _xy1_))
                _perp_       = (-_uv_[1], _uv_[0])
                _svg_.append(self.rt_self.svgText(f'{_count_}', _xy1_[0] - _perp_[0]*3, _xy1_[1] - _perp_[1]*3, txt_h=self.txt_h, 
                                                color=_color_, anchor='middle', rotation=_rotation_+180.0))
                
        # Triangle Shape
        _svg_.append(f'<path d="{self.tri_path}"       fill="none" stroke="{self.rt_self.co_mgr.getTVColor("axis","default")}" stroke-width="3" />')
        _svg_.append(f'<path d="{self.tri_path_inner}" fill="none" stroke="{self.rt_self.co_mgr.getTVColor("axis","minor")}"   stroke-width="0.5" />')

        # Labeling
        if self.draw_q_id_label: _svg_.append(self.rt_self.svgText(f"{q_id}", self.x_ins, self.y_ins, txt_h=self.txt_h*self.q_id_multiple, color="#c0c0c0", anchor='left', rotation=90))
        _str_ = "All" if summary_source is None else summary_source
        _svg_.append(self.rt_self.svgText(_str_, self.w/2.0, self.h - self.y_ins/2.0 - 2, txt_h=self.txt_h, 
                                          color=self.rt_self.co_mgr.getTVColor('label','defaultfg'), anchor='middle'))
        _svg_.append('</svg>')
        return '\n'.join(_svg_)
