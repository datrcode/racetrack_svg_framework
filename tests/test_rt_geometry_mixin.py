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
import unittest
import pandas as pd
import polars as pl
import numpy as np
import random

from math import sin, cos, sqrt, pi

from rtsvg import *

class Testrt_geometry_mixin(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rt_self = RACETrack()

    def test_averagDegrees(self):
        self.assertAlmostEqual(self.rt_self.averageDegrees([180]), 180)
        self.assertAlmostEqual(self.rt_self.averageDegrees([180,190]), 185)
        self.assertAlmostEqual(self.rt_self.averageDegrees([358,0]), 359)
        self.assertAlmostEqual(self.rt_self.averageDegrees([90,270]), 180)

    def test_concetricCirclesGlyph(self):
        _lu_ = {'__count__':[ 5,    10,   15,   12,   1,    10],
                '__dir__'  :['fm', 'to', 'fm', 'to', 'fm', 'to'],
                '__nbor__' :['a',  'a',  'b',  'b',  'c',  'c']}
        _order_ = ['b','a','c']

        self.rt_self.co_mgr.str_to_color_lu['a'] = '#ff0000'
        self.rt_self.co_mgr.str_to_color_lu['b'] = '#03ac13'
        self.rt_self.co_mgr.str_to_color_lu['c'] = '#0000ff'

        df      = pl.DataFrame(_lu_)

        _order_ = self.rt_self.colorRenderOrder(df, '__nbor__', '__count__', False)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 50,  50, 0.0, 0.5,  order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 85,  50, 0.5, 0.75, order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 125, 50, 1.0, 1.0,  order=_order_)

        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 200, 50, 0.0, 0.0,  df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 240, 50, 0.5, 0.33, df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 290, 50, 1.0, 0.66, df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)

        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 310+50,  50, 0.0, 0.5,  order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 310+85,  50, 0.5, 0.75, order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 310+125, 50, 1.0, 1.0,  order=_order_)

        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 310+200, 50, 0.0, 0.0,  df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 310+240, 50, 0.5, 0.33, df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 310+290, 50, 1.0, 0.66, df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)

        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 620+50,  50, 0.0, 0.5,  order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 620+85,  50, 0.5, 0.75, order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 620+125, 50, 1.0, 1.0,  pie_color="#ff0000", order=_order_)

        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 620+200, 50, 0.0, 0.0,  df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 620+240, 50, 0.5, 0.33, df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)
        self.rt_self.concentricGlyph(df.filter(pl.col('__dir__') == 'to'), 620+290, 50, 1.0, 0.66, df_outer=df.filter(pl.col('__dir__') == 'fm'), order=_order_)

    def test_crunchCircles(self):
        circles      = []
        n_circles    = 50
        w,     h     = 400, 400
        r_min, r_max = 10,  20
        min_inter_circle_d = 5
        for i in range(n_circles):
            circles.append((w*random.random(), h*random.random(), random.randint(r_min, r_max)))
        _placed_ = self.rt_self.crunchCircles(circles, min_d=min_inter_circle_d)


    def test_circularPathRouter(self):
        _n_paths_         = 30
        _n_circles_       = 20
        _radius_min_      = 20
        _radius_max_      = 30
        _min_circle_sep_  = 30
        _half_sep_        = _min_circle_sep_/2.0   # Needs to be more than the _radius_inc_test_
        _radius_inc_test_ = 4
        _radius_start_    = _radius_inc_test_ + 1  # Needs to be more than the _radius_inc_test_ ... less than the _min_circle_sep_
        _escape_px_       = 10                     # less than the _min_circle_sep_

        def createCircleDataset(n_circles=_n_circles_, n_paths=_n_paths_, radius_min=_radius_min_, radius_max=_radius_max_, min_circle_sep=_min_circle_sep_, radius_inc_test=_radius_inc_test_):
            circle_geoms = []
            def circleOverlaps(cx, cy, r):
                for _geom_ in circle_geoms:
                    dx, dy = _geom_[0] - cx, _geom_[1] - cy
                    d      = sqrt(dx*dx+dy*dy)
                    if d < (r + _geom_[2] + _min_circle_sep_): # at least 10 pixels apart...
                        return True
                return False
            def findOpening():
                _max_attempts_ = 100
                attempts  = 0
                cx, cy, r = random.randint(radius_max+min_circle_sep, 600-radius_max-min_circle_sep), \
                            random.randint(radius_max+min_circle_sep, 400-radius_max-min_circle_sep), random.randint(radius_min,radius_max)
                while circleOverlaps(cx,cy,r) and attempts < _max_attempts_:
                    cx, cy, r = random.randint(radius_max+min_circle_sep, 600-radius_max-min_circle_sep), \
                                random.randint(radius_max+min_circle_sep, 400-radius_max-min_circle_sep), random.randint(radius_min,radius_max)
                    attempts += 1
                if attempts == _max_attempts_:
                    return None
                return cx, cy, r

            # Randomize the circles
            for i in range(n_circles):
                to_unpack = findOpening()
                if to_unpack is not None:
                    cx, cy, r = to_unpack
                    circle_geoms.append((cx,cy,r))

            # Randomize the entry point
            c0         = random.randint(0, len(circle_geoms)-1)
            cx, cy, r  = circle_geoms[c0]
            a0         = random.random() * 2 * pi
            entry_pt   = (cx+(r+_radius_inc_test_+0.5)*cos(a0),cy+(r+_radius_inc_test_+0.5)*sin(a0),c0)
                        
            # Randomize the exit points
            exit_pts = []
            for i in range(n_paths):
                c1 = random.randint(0,len(circle_geoms)-1)
                while c1 == c0:
                    c1 = random.randint(0,len(circle_geoms)-1)
                cx, cy, r  = circle_geoms[c1]
                a1         = random.random() * 2 * pi
                exit_pts.append((cx+(r+radius_inc_test+0.5)*cos(a1),cy+(r+radius_inc_test+0.5)*sin(a1),c1))

            return entry_pt, exit_pts, circle_geoms

        _entry_pt_,_exit_pts_,_circle_geoms_ = createCircleDataset()
        self.rt_self.circularPathRouter(_entry_pt_,_exit_pts_,_circle_geoms_)

    def test_levelSets(self):
        _w,_h = 128,128
        _base = [[None for x in range(_w)] for y in range(_h)]
        for x in range(60,70):
            for y in range(60,70):
                _base[y][x] = -1

        for x in range(30,32):
            for y in range(0,50):
                _base[y][x] = -1
            for y in range(55,128):
                _base[y][x] = -1

        _base[10][10] = 1
        _base[90][90] = 2
        _base[2][120] = 3
        _base[90][5]  = 4

        _state, _found_time, _origin = self.rt_self.levelSet(_base)
        _state, _found_time, _origin = self.rt_self.levelSetFast(_base)
        self.rt_self.levelSetStateAndFoundTimeSVG(_state,_found_time)

    def test_levelSetsBalanced(self):
        my_raster = state = [[None for x in range(128)] for y in range(64)]  # node that found the pixel
        my_raster[10][10]  = set([1,2,3])
        my_raster[0][0]    = set([4])
        my_raster[50][10]  = set()
        my_raster[3][120]  = set([5,6])
        my_raster[62][50]  = set([7])
        my_raster[63][127] = set([8])
        my_raster[32][100] = set([9,10,11])
        my_raster[1][5]    = set([12])
        my_raster[55][93]  = set([13,14])
        my_raster[50][91]  = set([15,16,17])
        my_raster[63][0]   = set([18])
        my_origins         = [5, 10, 4, 18, 15]

        my_state, my_found_time, my_finds, my_progress_lu = self.rt_self.levelSetBalanced(my_raster, my_origins, 0)

    # Copied from voronoi.ipynb
    def test_circleOrdering(self):
        n,w,h = 10, 600, 500
        c     = (300,250,200)
        _svg_ = [f'<svg x="0" y="0" width="{w}" height="{h}">']
        _svg_.append('<rect x="0" y="0" width="{w}" height="{h}" fill="#ffffff" />')
        _svg_.append(f'<circle cx="{c[0]}" cy="{c[1]}" r="{c[2]}" stroke="#000000" fill="none" />')
        pts   = []
        for i in range(n):
            _angle_ = 2.0 * pi * random.random()
            x, y = c[0] + c[2] * cos(_angle_), c[1] + c[2] * sin(_angle_)
            pts.append((x,y))
            #_svg_.append(f'<text x="{x}" y="{y}" font-size="20" fill="#000000" text-anchor="middle">{i}</text>')
        for i in range(20):
            x, y = random.randint(0,w), random.randint(0,h)
            if self.rt_self.pointWithinThreePointCircle((x,y), pts[0], pts[1], pts[2]): _co_ = '#ff0000'
            else:                                                                       _co_ = '#a0a0a0'
            _svg_.append(f'<circle cx="{x}" cy="{y}" r="2" fill="{_co_}" stroke="none" />')

        ordered_pts = self.rt_self.counterClockwiseOrder(pts, c)
        for i in range(len(ordered_pts)):
            x,y = ordered_pts[i]
            _svg_.append(f'<text x="{x}" y="{y}" font-size="14" fill="#ff0000" text-anchor="middle">{i}</text>')

    # Copied from voronoi.ipynb
    def test_bowyerWatson(self):
        _n_, w, h = 5, 400, 300
        pts = []
        for i in range(_n_):
            x, y = random.randint(20,w-20), random.randint(20,h-20)
            pts.append((x,y))

        _triangulation_ = self.rt_self.bowyerWatson(pts)
        _triangles_ = []
        for t in _triangulation_:
            _triangles_.append(f'<polygon points="{t[0][0]},{t[0][1]} {t[1][0]},{t[1][1]} {t[2][0]},{t[2][1]}" fill="none" stroke="#ff0000" />')

    # Copied from voronoi.ipynb
    def test_isEdgarVoronoi(self):
        _n_, w, h = 5, 400, 300
        pts = []
        for i in range(_n_):
            x, y = random.randint(20,w-20), random.randint(20,h-20)
            pts.append((x,y))

        polys = self.rt_self.isedgarVoronoi(pts)
        _voronoi_svg_ = []
        for i in range(len(polys)):
            poly, _co_ = polys[i], self.rt_self.co_mgr.getColor(i)
            l = [f'M {poly[0][0]} {poly[0][1]}']
            for j in range(1, len(poly)): l.append(f'L {poly[j][0]} {poly[j][1]}')
            l.append('Z')
            _voronoi_svg_.append(f'<path d="{"".join(l)}" fill="{_co_}" stroke="{_co_}" fill-opacity="0.25" stroke-width="2"/>')
            _voronoi_svg_.append(f'<circle cx="{pts[i][0]}" cy="{pts[i][1]}" r="5" fill="none" stroke="{_co_}" stroke-width="2"/>')

if __name__ == '__main__':
    unittest.main()
