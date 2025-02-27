import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage
import numpy as np
import random
import string
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
        # Only accepts pandas
        if rt_self.isPandas(df) == False: raise Exception('df must be a pandas DataFrame')
        
        # Tabularize
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
        self.polys      = [] # debug
        self.scu_to_xy  = {} # scu_to_xy[q_id][scy] = (x,y)
        self.level_bars = {}
        self.mid_bars   = {}

    #
    # scusMissingForQuestionIDAndSource()
    #
    def scusMissingForQuestionIDAndSource(self, question_id, source):
        _all_scus_for_this_source_   = set(self.df.query(f'`{self.summary_source_field}` == @source and `{self.q_id_field}` == @question_id')[self.scu_field])
        _all_scus_for_this_question_ = set(self.df.query(f'                                             `{self.q_id_field}` == @question_id')[self.scu_field])
        print(f'SCUs Missing From "{source}" For Question "{question_id}" ({len(_all_scus_for_this_source_)} SCUs Captured For This Source, {len(_all_scus_for_this_question_)} SCUs Captured For This Question)\n')
        _num_of_models_ = self.df[self.summary_source_field].nunique()
        for level in range(_num_of_models_, 0, -1):
            if   level == _num_of_models_: level_label = 'top'
            elif level == 1:               level_label = 'bottom'
            else:                          level_label = ''
            _scus_at_this_level_ = set(self.df_tab.query(f'occurences == @level and `{self.q_id_field}` == @question_id')[self.scu_field])
            print(f'** Level {level} {level_label:6} ** ({len(_scus_at_this_level_)} SCUs At This Level (in total))')
            _missing_scus_       = list(_scus_at_this_level_ - _all_scus_for_this_source_)
            _missing_scus_.sort()
            for _scu_ in _missing_scus_: print(f'   "{_scu_}"')
            print()

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
        self.level_bars[q_id] = {}
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
                _poly_          = Polygon([(bot_x0, y_base_last-self.tri_inset/2.0),
                                           (bot_x1, y_base_last-self.tri_inset/2.0),
                                           self.top_xy_inner])
                
            self.polys.append(_poly_) # debug

            if len(scus_at_level[level]) > 5:
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
                _y_mid_    = (y_base_last+y_base)/2.0
                _x0_, _x1_ = self.__calculateXBoundsForTriangleLevel__(_y_mid_)
                _w_adj_    = (_x1_ - _x0_)/3.0
                _w_adj2_   = (_x1_ - _x0_)/5.0
                _h_adj_    = (y_base_last-y_base)/5.0
                _h_adj2_   = (y_base_last-y_base)/9.0
                if   len(scus_at_level[level]) == 1 or len(scus_at_level[level]) == 0:
                    xys = [(self.w/2.0, _y_mid_)]
                elif len(scus_at_level[level]) == 2:
                    xys = [(_x0_+_w_adj_, _y_mid_), (_x1_-_w_adj_, _y_mid_)]
                elif len(scus_at_level[level]) == 3:
                    xys = [(self.w/2.0, _y_mid_-_h_adj_), (_x0_+_w_adj_, _y_mid_+_h_adj_), (_x1_-_w_adj_, _y_mid_+_h_adj_)]
                elif len(scus_at_level[level]) == 4:
                    xys = [(_x0_+_w_adj_,  _y_mid_-_h_adj2_), (_x1_-_w_adj_,  _y_mid_-_h_adj2_), 
                           (_x0_+_w_adj2_, _y_mid_+_h_adj2_), (_x1_-_w_adj2_, _y_mid_+_h_adj2_)]
                elif len(scus_at_level[level]) == 5:
                    xys = [(_x0_+_w_adj_,  _y_mid_-_h_adj2_), (_x1_-_w_adj_,  _y_mid_-_h_adj2_), 
                           (_x0_+_w_adj2_, _y_mid_+_h_adj2_), (_x1_-_w_adj2_, _y_mid_+_h_adj2_),
                           (self.w/2.0, _y_mid_)]

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
                _count_ = _df_.query(f'`{self.q_id_field}` == @q_id and `{self.summary_source_field}` == @summary_source and occurences == @level_plus_1')[self.scu_field].nunique()
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

    #
    # svgSnowman()
    #
    def svgSnowman(self, q_id, 
                   q_id_multiple=1.5, r_scu=12,
                   txt_h=12, w=384, h=384, x_ins=16, y_ins=40):
        # SVG Setup
        w_usable, h_usable = w - 2*x_ins, h - 2*y_ins
        _svg_ = [f'<svg x="0" y="0" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">']
        _svg_.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{self.rt_self.co_mgr.getTVColor("background","default")}" />')
        #_svg_.append(f'<line x1="{w/2.0}" y1="0" x2="{w/2.0}" y2="{h}" stroke="{self.rt_self.co_mgr.getTVColor("axis","minor")}" stroke-width="0.25" />')
        if self.draw_q_id_label: _svg_.append(self.rt_self.svgText(f"{q_id}", 4, y_ins, txt_h=txt_h*q_id_multiple, color="#c0c0c0", anchor='left', rotation=90))
        # Filter down to just this question
        df_q     = self.df.query(f'`{self.q_id_field}` == @q_id')
        df_q_tab = self.df_tab.query(f'`{self.q_id_field}` == @q_id')
        # Get the number of levels & calculate the scu's per level
        levels          = df_q[self.summary_source_field].nunique()
        level_scu_count = {}
        level_scu_list  = {}
        max_scus        = 0
        for _level_ in range(0, levels):
            l_plus_1    = _level_ + 1
            num_of_scus = df_q_tab.query(f'occurences == @l_plus_1')[self.scu_field].nunique()
            level_scu_count[l_plus_1] = num_of_scus
            _scu_set_                 = set(df_q_tab.query(f'occurences == @l_plus_1')[self.scu_field])
            if _level_ == 0:
                level_scu_list [l_plus_1] = list(df_q.query(f'`{self.scu_field}` in @_scu_set_').sort_values(by=self.summary_source_field)[self.scu_field])
            else:
                level_scu_list [l_plus_1] = list(_scu_set_)
            max_scus                  = max(num_of_scus, max_scus)
        x_spacing = w_usable/max_scus

        # Create the glyph representation
        glyph_geometry = self.rt_self.setGlyphGeometry(set(df_q[self.summary_source_field]), r_scu, r_scu//2)

        # Calculate the level geometries
        xy_to_scu              = {}
        level_to_scu_placement = {}
        for _level_ in range(0, levels):
            l_plus_1 = _level_ + 1
            level_to_scu_placement[l_plus_1] = []
            y        = y_ins + h_usable - _level_ * h_usable/(levels-1)
            _count_  = level_scu_count[l_plus_1]            
            if _count_ > 0:
                if _count_%2 == 0: x_base = w/2.0 + x_spacing/2.0 - _count_//2 * x_spacing
                else:              x_base = w/2.0                 - _count_//2 * x_spacing
                for i in range(_count_):
                    x = x_base + i * x_spacing
                    if x_spacing < 2*(r_scu+2): y_toggle = -1 if (i%2) == 0 else 1
                    else:                       y_toggle = 0
                    _xy_ = (x, y+y_toggle*r_scu*1.2)
                    level_to_scu_placement[l_plus_1].append(_xy_)
                    xy_to_scu[_xy_] = level_scu_list[l_plus_1][i]

        # Render Outlines For The Levels
        for _level_ in range(0, levels):
            l_plus_1 = _level_ + 1
            _count_  = level_scu_count[l_plus_1]
            if _count_ > 0:
                _xy_                = level_to_scu_placement[l_plus_1][0]
                xmin,ymin,xmax,ymax = _xy_[0], _xy_[1], _xy_[0], _xy_[1]
                for _xy_ in level_to_scu_placement[l_plus_1]:
                    xmin,ymin,xmax,ymax = min(xmin,_xy_[0]), min(ymin,_xy_[1]), max(xmax,_xy_[0]), max(ymax,_xy_[1])
                    rb     = 5
                    rx, ry = xmin-r_scu-rb, ymin-r_scu-rb
                    rw, rh = (xmax - xmin) + 2*(r_scu+rb), (ymax-ymin) + 2*(r_scu+rb)
                _color_ = self.rt_self.co_mgr.getColor(l_plus_1)
                _svg_.append(f'<rect x="{rx}" y="{ry}" width="{rw}" height="{rh}" fill="none" stroke="{_color_}" fill-opacity="0.1" rx="{r_scu}" />')

        # Render the Levels
        _color_ = self.rt_self.co_mgr.getTVColor("data","default")
        for _level_ in range(0, levels):
            l_plus_1 = _level_ + 1
            y        = y_ins + h_usable - _level_ * h_usable/(levels-1)
            _count_  = level_scu_count[l_plus_1]
            if _count_ == 0:
                _line_color_ = self.rt_self.co_mgr.getTVColor("data","default") if _count_ > 0 else self.rt_self.co_mgr.getTVColor('context','highlight')
                _svg_.append(f'<line x1="{x_ins}" y1="{y}" x2="{w - x_ins}" y2="{y}" stroke="{_line_color_}" stroke-width="0.5" />')
                _svg_.append(rt.svgText(f"Level Empty", w/2.0, y-2, txt_h=txt_h, color=_line_color_, anchor='middle'))   
            #_svg_.append(rt.svgText(f"{l_plus_1}", w/2.0, y-2, txt_h=txt_h, color="#c0c0c0", anchor='middle'))
            for _xy_ in level_to_scu_placement[l_plus_1]:
                _scu_     = xy_to_scu[_xy_]
                _sources_ = set(df_q.query(f'`{self.scu_field}` == @_scu_')[self.summary_source_field])
                _svg_.append(self.rt_self.renderSetGlyph(_sources_, _xy_, glyph_geometry))

        _svg_.append('</svg>')
        return '\n'.join(_svg_)

    #
    # orderSCUsBySources()
    # - used to order each layer of the cairn viz
    # - most similar by source...
    #
    def orderSCUsBySources(self, scus, df_q):
        # Handle some base cases
        if len(scus) == 0: return []
        if len(scus) == 1: return list(scus)
        if len(scus) == 2: return list(scus)

        # Make sure it's a list
        if type(scus) != list: scus = list(scus)
        # Make a corresponding list of the sources -- this lines up with the scus list
        sources_sets = []
        for _scu_ in scus:
            _sources_ = set(df_q.query(f'`{self.scu_field}` == @_scu_')[self.summary_source_field])
            sources_sets.append(_sources_)
        # Create the distance matrix
        _dmat_ = []
        for i in range(len(sources_sets)):
            _row_ = []
            for j in range(len(sources_sets)):
                _similarity_ = len(sources_sets[i] & sources_sets[j]) / len(sources_sets[i] | sources_sets[j])
                _row_.append(_similarity_)
            _dmat_.append(_row_)
        # Hierarchical clustering
        linkage_matrix = linkage(_dmat_, method='ward')
        # Place into a tree
        parent_to_children = {}
        next_node_id       = len(sources_sets)
        for row in linkage_matrix:
            to_merge_0, to_merge_1 = int(row[0]), int(row[1])
            parent_to_children[next_node_id] = [to_merge_0, to_merge_1]
            next_node_id += 1
        root_node = next_node_id - 1
        # Walk the leaves of the dendrogram
        def leafWalk(node_id):
            if node_id < len(sources_sets):
                return [node_id]
            left_child, right_child = parent_to_children[node_id]
            return leafWalk(left_child) + leafWalk(right_child)
        order = leafWalk(root_node)
        scu_order = []
        for i in order: scu_order.append(scus[i])
        return scu_order

    #
    # svgCairn()
    #
    def svgCairn(self,
                 q_id, 
                 q_id_multiple              = 2, 
                 cell_w                     = 48,
                 cell_x_spacing             = 4,
                 cell_h                     = 40,
                 cell_y_spacing             = 8,
                 histogram                  = True,
                 histogram_w                = 48,
                 attach_histogram_to_levels = False,
                 rx                         = 8,
                 txt_h                      = 12, 
                 w                          = 384, 
                 h                          = 384, 
                 x_ins                      = 32, 
                 y_ins                      = 32):
        # SVG Setup
        w_usable, h_usable = w - 2*x_ins, h - 2*y_ins
        if histogram: w_usable -= histogram_w
        widget_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))
        _svg_ = [f'<svg x="0" y="0" width="{w}" height="{h}" id="{widget_id}" xmlns="http://www.w3.org/2000/svg">']
        _svg_.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="{self.rt_self.co_mgr.getTVColor("background","default")}" />')
        #_svg_.append(f'<rect x="0" y="0" width="{w}" height="{h}" fill="#d0d0d0" />')
        # _svg_.append(f'<line x1="{w/2.0}" y1="0" x2="{w/2.0}" y2="{h}" stroke="{self.rt_self.co_mgr.getTVColor("axis","minor")}" stroke-width="0.25" />')
        if self.draw_q_id_label: _svg_.append(self.rt_self.svgText(f"{q_id}", w - x_ins/2.0 - 2.0, y_ins, txt_h=txt_h*q_id_multiple, color="#c0c0c0", anchor='left', rotation=90))
        # Filter down to just this question
        df_q     = self.df.query(f'`{self.q_id_field}` == @q_id')
        df_q_tab = self.df_tab.query(f'`{self.q_id_field}` == @q_id')

        # Get the number of levels & calculate the scu's per level
        levels          = df_q[self.summary_source_field].nunique()
        level_scu_count = {}
        level_scu_list  = {}
        max_scus        = 0
        for _level_ in range(0, levels):
            l_plus_1    = _level_ + 1
            num_of_scus = df_q_tab.query(f'occurences == @l_plus_1')[self.scu_field].nunique()
            level_scu_count[l_plus_1] = num_of_scus
            _scu_set_                 = set(df_q_tab.query(f'occurences == @l_plus_1')[self.scu_field])
            level_scu_list [l_plus_1] = self.orderSCUsBySources(list(_scu_set_), df_q)                
            max_scus                  = max(num_of_scus, max_scus)
        
        # Adjust the cell sizes (if necessary)
        needed_h = 2*y_ins + cell_h*levels + cell_y_spacing*(levels-1)
        if needed_h > h_usable: cell_h = (h_usable - cell_y_spacing*(levels-1)) / levels
        needed_w = 2*x_ins + cell_w*max_scus + cell_x_spacing*(max_scus-1)
        if needed_w > w_usable: cell_w = (w_usable - cell_x_spacing*(max_scus-1)) / max_scus

        # Calculate the level geometries
        before_space           = histogram_w if histogram and attach_histogram_to_levels == False else 0
        xywh_to_scu            = {}
        level_to_scu_placement = {}
        level_to_outline       = {} # (x,y,w,h)
        for _level_ in range(0, levels):
            l_plus_1 = _level_ + 1
            level_to_scu_placement[l_plus_1] = []
            y        = y_ins + h_usable - _level_ * h_usable/(levels-1)
            _count_  = level_scu_count[l_plus_1]
            if _count_ > 0:
                level_w = cell_w * _count_ + cell_x_spacing * (_count_-1)
                level_to_outline[l_plus_1] = (before_space + x_ins + (w_usable)/2 - level_w/2, y - cell_h/2, level_w, cell_h)
                for i in range(_count_):
                    x = before_space + x_ins + (w_usable)/2 - level_w/2 + i * (cell_w + cell_x_spacing)
                    _xywh_ = (x, y - cell_h/2, cell_w, cell_h)
                    xywh_to_scu[_xywh_] = level_scu_list[l_plus_1][i]
                    level_to_scu_placement[l_plus_1].append(_xywh_)
            else:
                level_to_outline[l_plus_1] = (before_space + x_ins + (w_usable)/2 - cell_w/2,  y - cell_h/2, cell_w,  cell_h)

        # Assign offsets for the sources
        _sources_ = sorted(list(set(df_q[self.summary_source_field])))
        source_y_offset, source_h = {}, cell_h/len(_sources_)
        for i in range(len(_sources_)): source_y_offset[_sources_[i]] = i * source_h

        # Render the outlines for the levels
        for _level_ in range(0, levels):
            l_plus_1 = _level_ + 1
            _count_  = level_scu_count[l_plus_1]
            if _count_ == 0: _color_, _dash_array_, _stroke_width_ = self.rt_self.co_mgr.getTVColor("context", "highlight"),  'stroke-dasharray="10 5 3 2"', 1.0
            else:            _color_, _dash_array_, _stroke_width_ = self.rt_self.co_mgr.getTVColor("axis",    "major"),      '',                            2.0
            _bounds_ = level_to_outline[l_plus_1]
            _svg_.append(f'<rect x="{_bounds_[0]-2}" y="{_bounds_[1]-2}" width="{_bounds_[2]+4}" height="{_bounds_[3]+4}" fill="none" stroke="{_color_}" stroke-width="{_stroke_width_}" rx="{rx}" {_dash_array_} />')

        # Render the SCU's
        clip_num, clip_paths, overall_max_count = 0, [], 1
        for _level_ in range(0, levels):
            l_plus_1 = _level_ + 1
            # Do the counts per source for the histogram
            counts_per_source = {}
            for _source_ in set(df_q[self.summary_source_field]): counts_per_source[_source_] = 0
            # Render each SCU
            for _xywh_ in level_to_scu_placement[l_plus_1]:
                _scu_     = xywh_to_scu[_xywh_]
                _sources_ = set(df_q.query(f'`{self.scu_field}` == @_scu_')[self.summary_source_field])
                _svg_.append(f'<rect x="{_xywh_[0]}" y="{_xywh_[1]}" width="{_xywh_[2]}" height="{_xywh_[3]}" fill="none" stroke="{self.rt_self.co_mgr.getTVColor("axis","major")}" stroke-width="0.5" rx="{rx}" />')
                # Create a unique clip id for each SCU - defer the addition until the end
                clip_id = f'{widget_id}_{clip_num}'
                clip_num += 1
                clip_paths.append(f'<clipPath id="{clip_id}"><rect x="{_xywh_[0]}" y="{_xywh_[1]}" width="{_xywh_[2]}" height="{_xywh_[3]}" rx="{rx}"/></clipPath>')
                # Go through the sources & render them individually w/ the clip path
                for _source_ in _sources_:
                    counts_per_source[_source_] += 1
                    _color_ = self.rt_self.co_mgr.getColor(_source_)
                    _svg_.append(f'<rect x="{_xywh_[0]}" y="{_xywh_[1]+source_y_offset[_source_]}" width="{_xywh_[2]}" height="{source_h}" fill="{_color_}" stroke="none" clip-path="url(#{clip_id})" />')
            # Track max for possible histogram...
            _max_count_ = max(counts_per_source.values())
            if _max_count_ > overall_max_count: overall_max_count = _max_count_

        # Render the histogram if requested
        if histogram:
            for _level_ in range(0, levels):
                l_plus_1    = _level_ + 1
                # Do the counts per source for the histogram
                counts_per_source = {}
                for _source_ in set(df_q[self.summary_source_field]): counts_per_source[_source_] = 0
                # Same as previous loop... but to count...
                for _xywh_ in level_to_scu_placement[l_plus_1]:
                    _scu_     = xywh_to_scu[_xywh_]
                    _sources_ = set(df_q.query(f'`{self.scu_field}` == @_scu_')[self.summary_source_field])
                    for _source_ in _sources_:
                        counts_per_source[_source_] += 1
                # Render the histogram  
                _bounds_    = level_to_outline[l_plus_1]
                x, y        = _bounds_[0] + _bounds_[2] + 2*cell_x_spacing, _bounds_[1]
                if attach_histogram_to_levels is False: x = w - x_ins/2.0
                for _source_ in counts_per_source:
                    _count_ = counts_per_source[_source_]
                    if _count_ == 0: continue
                    _color_ = self.rt_self.co_mgr.getColor(_source_)
                    _bar_w_ = histogram_w * (_count_ / overall_max_count)
                    if attach_histogram_to_levels:
                        _svg_.append(f'<rect x="{x}" y="{y+source_y_offset[_source_]}" width="{_bar_w_}" height="{source_h}" fill="{_color_}" stroke="none" rx="{source_h*0.2}"/>')
                    else:
                        _svg_.append(f'<rect x="{x_ins/2.0}" y="{y+source_y_offset[_source_]}" width="{_bar_w_}" height="{source_h}" fill="{_color_}" stroke="none" rx="{source_h*0.2}"/>')

        _svg_.append('<defs>'+''.join(clip_paths)+'</defs>')
        _svg_.append('</svg>')
        return '\n'.join(_svg_)

    #
    # fillScore() - prototype scoring for the shape of the pyramid
    #
    def fillScore(self, q_id, top_perc=0.6):
        _num_sources_ = self.df.query(f'`{self.q_id_field}` == @q_id')[self.summary_source_field].nunique()
        _df_ = self.df.query(f'`{self.q_id_field}` == @q_id')                 \
                .groupby([self.scu_field])                                 \
                .count()                                                   \
                .reset_index()                                             \
                .groupby([self.summary_source_field])                      \
                .count()                                                   \
                .reset_index()                                             \
                .sort_values([self.summary_source_field], ascending=False) \
                .reset_index()                                             \
                .drop(['index'], axis=1)
        _level_lu_ = {}
        for i in range(_num_sources_): _level_lu_[i+1] = 0
        max_level_width = 1
        for i in range(len(_df_)):
            _sources_, _num_of_scus_ = _df_.iloc[i][self.summary_source_field], _df_.iloc[i][self.scu_field]
            _level_lu_[_sources_] = _num_of_scus_
            if _sources_ > max_level_width: max_level_width = _sources_
        _score_, _whats_left_ = 0.0, 1.0
        for i in range(_num_sources_,0,-1):
            _top_        =  _whats_left_ * top_perc
            _score_      += _level_lu_[i] * _top_ / max_level_width
            _whats_left_ =  _whats_left_ * (1.0 - top_perc)
        return float(_score_)
