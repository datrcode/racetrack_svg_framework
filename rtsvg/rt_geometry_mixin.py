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

from shapely.geometry              import Polygon, LineString, GeometryCollection, MultiLineString
from shapely.geometry.multipolygon import MultiPolygon
from math import sqrt, acos, pi, cos, sin, atan2
import random

import heapq

__name__ = 'rt_geometry_mixin'


#
# Geometry Methods
#
class RTGeometryMixin(object):
    #
    # averageDegrees() - return the average angle for a list of degrees
    # - not efficient due to use of cos, sin, atan2
    #
    def averageDegrees(self, angles): # list of angles in degrees
        to_rad     = lambda angle: pi*angle/180.0
        xsum, ysum = 0.0, 0.0
        for angle in angles:
            xsum += cos(to_rad(angle))
            ysum += sin(to_rad(angle))
        if xsum == 0.0 and ysum == 0.0:
            avg_angle = sum(angles)/len(angles)
        else:    
            avg_angle = atan2(ysum, xsum) * 180.0 / pi
            if avg_angle < 0.0: avg_angle += 360.0
        return avg_angle

    #
    # Converted from the following description
    # https://stackoverflow.com/questions/5736398/how-to-calculate-the-svg-path-for-an-arc-of-a-circle
    #
    # - angle in degrees (0 degrees is 12 o'clock)
    #
    def polarToCartesian(self, cx, cy, r, deg):
        rads = (deg-90) * pi / 180.0
        return cx + (r*cos(rads)), cy + (r*sin(rads))
    def pieSlice(self, cx, cy, r, deg0, deg1, color='#000000'):
        x0,y0 = self.polarToCartesian(cx,cy,r,deg1)
        x1,y1 = self.polarToCartesian(cx,cy,r,deg0)
        flag = "0" if (deg1 - deg0) <= 180.0 else "1"
        return f'<path d="M {cx} {cy} L {x0} {y0} A {r} {r} 0 {flag} 0 {x1} {y1}" fill="{color}" stroke="{color}" stroke-width="2" />'
    
    #
    # genericArc()
    # - angles in degrees (0 degrees is 3 o'clock)
    #
    def genericArc(self, cx, cy, a0, a1, r_inner, r_outer):
        _fn_ = lambda a,r: (cx + r * cos(a * pi / 180.0), cy + r * sin(a * pi / 180.0))
        x0_out,  y0_out  = _fn_(a0, r_outer)
        x0_in,   y0_in   = _fn_(a0, r_inner)
        x1_out,  y1_out  = _fn_(a1, r_outer)
        x1_in,   y1_in   = _fn_(a1, r_inner)
        large_arc = 0 if (a1-a0) <= 180.0 else 1
        _path_ = f'M {x0_out} {y0_out} A {r_outer} {r_outer} 0 {large_arc} 1 {x1_out} {y1_out} L {x1_in} {y1_in} ' + \
                                    f' A {r_inner} {r_inner} 0 {large_arc} 0 {x0_in}  {y0_in}  Z'
        return _path_

    #
    # concentricGlyphCircumference() - forms the circular circumference around a glyph
    # - assumes that df has already been grouped -- i.e., no duplicate nbors
    #
    def concentricGlyphCircumference(self, df, cx, cy, r_inner, r_outer, order=None, nbor='__nbor__', count_by='__count__', count_by_set=False, angle_min=10.0):
        if order is None: order = self.colorRenderOrder(df, nbor, count_by, count_by_set)
        order = order.with_row_index('__sorter__')
        df    = df.join(order.drop(['count']), left_on=nbor, right_on='index').sort('__sorter__')
        total = df[count_by].sum()
        df    = df.with_columns((360.0 * pl.col(count_by)/total).alias('__angle__')).filter(pl.col('__angle__') > angle_min)
        # df.with_columns(pl.col('__angle__').cum_sum().alias('__cumsum__'))
        angle = 270.0 # start at the top... like a clock...
        svgs  = []
        for i in range(len(df)):
            angle_to = angle + df['__angle__'][i]
            _color_  = self.co_mgr.getColor(df[nbor][i])
            _path_   = self.genericArc(cx, cy, angle, angle_to, r_inner, r_outer)
            svgs.append(f'<path d="{_path_}" stroke="white" stroke-width="1" fill="{_color_}" />')
            angle    = angle_to
        
        # The above doesn't work for a single arc
        if len(df) == 1:
            _color_  = self.co_mgr.getColor(df[nbor][0])
            svgs.append(f'<circle cx="{cx}" cy="{cy}" r="{(r_inner + r_outer)/2.0}" fill="none" stroke-width="{r_outer - r_inner}" stroke="{_color_}" />')

        return ''.join(svgs)

    #
    # concentricGlyph() - complete concentric glyph code
    # ... not all that efficient
    # ... df and df_outer should already be grouped -- i.e., no duplicate nbors
    # ... only accepts polars
    #
    def concentricGlyph(self, df, cx, cy, r_norm, pie_perc, pie_color=None, r_min=7.4, r_max=14.6, bar_w=4.5, df_outer=None, order=None, nbor='__nbor__', count_by='__count__', count_by_set=False, angle_min=10.0):
        r      = r_min + r_norm * (r_max - r_min)
        rp     = r + 2*bar_w if df_outer is not None else r + bar_w
        bg     = self.co_mgr.getTVColor('background','default')
        pie_co = self.co_mgr.getTVColor('axis','default') if pie_color is None else pie_color
        svgs   = [f'<circle cx="{cx}" cy="{cy}" r="{rp}" fill="{bg}" stroke="{bg}" stroke-width="1" />']

        svgs.append(self.concentricGlyphCircumference(df, cx, cy, r, r+bar_w, order=order, nbor=nbor, count_by=count_by, count_by_set=count_by_set, angle_min=angle_min))
        if df_outer is not None:
            svgs.append(self.concentricGlyphCircumference(df_outer, cx, cy, r+bar_w, r+2*bar_w, order=order, nbor=nbor, count_by=count_by, count_by_set=count_by_set, angle_min=angle_min))

        if   pie_perc == 1.0: svgs.append(f'<circle cx="{cx}" cy="{cy}" r="{r-2}" fill="{pie_co}" stroke="{pie_co}" stroke-width="2" />')
        elif pie_perc == 0.0: pass 
        else:                 svgs.append(self.pieSlice(cx, cy, r-2, 0, 360*pie_perc, color=pie_co))

        return ''.join(svgs)

    #
    # crunchCircles() - compress circles with a packing algorithm
    #
    def crunchCircles(self,
                      circles: list[tuple[float, float, float]],
                      min_d:   float = 20):
        """
        Compress circles with a packing algorithm.
        :param circles: list of circle x, y, and r as individual tuples
        :param min_d:   minimum distance for packing
        :return:        same list as input but with circles packed
        """
        n_circles = len(circles)

        # Find the "middle" circle
        s, placed = [], []
        for i in range(n_circles):
            cx, cy, r = circles[i][:3]
            placed.append(circles[i])  # this will be the updated placement ... just initializing here
            s.append((cx, i))
        s.sort()
        s2 = []
        m = int(len(s) / 2)
        for _tuple_ in s[m - 1:m + 2]:
            cx, cy, r = circles[_tuple_[1]][:3]
            s2.append((cy, _tuple_[1]))
        s2.sort()
        middle_i = s2[1][1]

        # Place the middle circle
        mx, my, mr       = circles[middle_i][:3]
        placed[middle_i] = circles[middle_i]  # not really necessary
        placed_set       = set([middle_i])

        # Sort all circles relative to the middle
        to_place = []
        cx_m, cy_m, r_m = circles[middle_i][:3]
        for i in range(n_circles):
            if i == middle_i:
                continue
            cx, cy, r = circles[i][:3]
            d         = sqrt((cx-cx_m)**2 + (cy-cy_m)**2)
            to_place.append((d, i))
        to_place.sort()

        def overlapsWithPlaced(cx, cy, r):
            for j in placed_set:
                cx2, cy2, r2 = placed[j][:3]
                d = sqrt((cx-cx2)**2 + (cy-cy2)**2)
                if (d-min_d) < (r+r2):
                    return True
            return False

        # Place the circles
        for k in range(len(to_place)):
            i = to_place[k][1]
            cx, cy, r = circles[i][:3]
            uv = self.unitVector(((mx, my), (cx, cy)))
            if uv[0] == 0 and uv[1] == 0:
                uv = (1, 0)
            fail_after = 0
            while overlapsWithPlaced(cx, cy, r) and fail_after < 1000:
                cx, cy = cx + uv[0]*min_d, cy + uv[1]*min_d
                fail_after += 1
            fail_after = 0
            last_cx, last_cy = cx, cy
            while overlapsWithPlaced(cx, cy, r) is False and fail_after < 1000:
                last_cx, last_cy = cx, cy
                cx, cy = cx - uv[0]*min_d/4, cy - uv[1]*min_d/4
                fail_after += 1
            placed_set.add(i)
            placed[i] = (last_cx, last_cy) + placed[i][2:]

        # Return the placed circles
        return placed

    #
    # circularPathRouter() - route exits to a single entry around circles
    # - all points needs to be at least circle_radius + radius_inc_test + 1.0 from the circle centers...
    #
    def circularPathRouter(self,
                           entry_pt,                  # (x,y,circle_i) -- where circle_i is the circle index from circle_geoms
                           exit_pts,                  # [(x,y,circle_i),(x,y,circle_i),(x,y,circle_i), ...] -- where circle_i is the circle index from circle_geoms
                           circle_geoms,              # [(cx,cy,r),(cx,cy,r), ...]
                           escape_px          = 5,    # length to push the exit points (and entry point) away from circle
                           min_circle_sep     = 30,   # minimum distance between circles
                           half_sep           = 15,   # needs to be more than the radius_inc_test ... half separation (but doesn't have to be)
                           radius_inc_test    = 4,    # for routing around circles, how much to test with
                           radius_start       = 5,    # needs to be more than the radius_inc_test ... less than the min_circle_sep
                           max_pts_per_node   = 50,   # maximum points per node for the xy quad tree
                           merge_distance_min = 5):   # minimum distance necessary to merge a path into an already exiting path
        # Calculate a path around the circle geometries
        def calculatePathAroundCircles(pts):
            def breakSegment(_segment_):
                if self.segmentLength(_segment_) < 2.0:
                    return _segment_
                for _geom_ in circle_geoms:
                    _circle_plus_ = (_geom_[0], _geom_[1], _geom_[2]+radius_inc_test)
                    _dist_, _inter_  = self.segmentIntersectsCircle(_segment_, _circle_plus_)
                    if _dist_ <= _circle_plus_[2]:
                        if _inter_[0] == _geom_[0] and _inter_[1] == _geom_[1]:
                            dx, dy   = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
                            l        = sqrt(dx*dx+dy*dy)
                            dx,  dy  = dx/l, dy/l
                            pdx, pdy = -dy, dx
                            return [(_segment_[0][0], _segment_[0][1]), (_geom_[0] + pdx*(_geom_[2]+half_sep), _geom_[1] + pdy*(_geom_[2]+half_sep)), (_segment_[1][0], _segment_[1][1])]
                        else:
                            dx, dy = _inter_[0] - _geom_[0], _inter_[1] - _geom_[1]
                            l      = sqrt(dx*dx+dy*dy)
                            dx, dy = dx/l, dy/l
                            return [(_segment_[0][0], _segment_[0][1]), (_geom_[0]  + dx*(_geom_[2]+half_sep), _geom_[1]  + dy*(_geom_[2]+half_sep)), (_segment_[1][0], _segment_[1][1])]
                return _segment_
            last_length = 0
            _segments_  = []
            for _pt_ in pts:
                _segments_.append(_pt_)
            while last_length != len(_segments_):
                last_length    = len(_segments_)
                _new_segments_ = []
                for i in range(len(_segments_)-1):
                    _new_ = breakSegment([_segments_[i], _segments_[i+1]])
                    if len(_new_) == 3:
                        _new_segments_.append(_new_[0])
                        _new_segments_.append(_new_[1])
                    else:
                        _new_segments_.append(_new_[0])
                _new_segments_.append(_new_[-1])
                _segments_ = _new_segments_
            return _segments_

        # Fix up the the entry and exit points...
        x_min, y_min, x_max, y_max = entry_pt[0], entry_pt[1], entry_pt[0], entry_pt[1]
        entries = []
        x0, y0, ci  = entry_pt
        uv          = self.unitVector(((circle_geoms[ci][0], circle_geoms[ci][1]), (x0, y0)))
        x0s, y0s    = x0+uv[0]*escape_px, y0+uv[1]*escape_px
        for pt in exit_pts:
            x1, y1, ci  = pt
            uv          = self.unitVector(((circle_geoms[ci][0], circle_geoms[ci][1]), (x1, y1)))
            x1s, y1s    = x1+uv[0]*escape_px, y1+uv[1]*escape_px
            entries.append([(x0, y0), (x0s, y0s), (x1s, y1s), (x1, y1)])
            x_min, y_min, x_max, y_max = min(x_min, x1),  min(y_min, y1),  max(x_max, x1),  max(y_max, y1)
            x_min, y_min, x_max, y_max = min(x_min, x1s), min(y_min, y1s), max(x_max, x1s), max(y_max, y1s)

        # XY Quad Tree
        xy_tree = self.xyQuadTree((x_min-half_sep, y_min-half_sep, x_max+half_sep, y_max+half_sep), max_pts_per_node=max_pts_per_node)

        # Sort paths by length (longest first)
        exit_sorter = []
        for i in range(len(entries)):
            _entry_ = entries[i]
            l = self.segmentLength((_entry_[0], _entry_[3]))
            exit_sorter.append((l, i))
        exit_sorter = sorted(exit_sorter)
        exit_sorter.reverse()

        # keep track of all of the final paths
        paths, merge_info = [], []
        for i in range(len(entries)):
            paths.append(entries[i])
            merge_info.append((-1, -1))

        # plot out the longest path
        i_longest        = exit_sorter[0][1]
        pts              = entries[i_longest]
        _path_           = calculatePathAroundCircles(pts)
        _path_smooth_    = self.smoothSegments(self.expandSegmentsIntoPiecewiseCurvedParts(_path_, amp=5.0, ampends=8.0, max_travel=1))
        _path_smooth_.reverse()
        paths[i_longest] = _path_smooth_
        for i in range(len(_path_smooth_)):
            pt = (_path_smooth_[i][0], _path_smooth_[i][1], i_longest, i)
            xy_tree.add([pt])

        # analyze the other paths
        for i in range(1, len(exit_sorter)):
            i_path        = exit_sorter[i][1]
            pts           = entries[i_path]
            _path_        = calculatePathAroundCircles(pts)
            _path_smooth_ = self.smoothSegments(self.expandSegmentsIntoPiecewiseCurvedParts(_path_, amp=5.0, ampends=8.0, max_travel=1))
            # merge with existing path
            merged_flag   = False
            _path_merged_ = [_path_smooth_[-1]]
            for j in range(len(_path_smooth_)-2, 2, -1):  # only down to 2... because the stem will exist from the longest path created
                closest = xy_tree.closest((_path_smooth_[j][0], _path_smooth_[j][1]), n=1)
                _path_merged_.append(_path_smooth_[j])
                if closest[0][0] < merge_distance_min:
                    _path_merged_.append((closest[0][1][0], closest[0][1][1]))
                    merged_flag = True
                    break
            # save the path off
            paths[i_path] = _path_merged_
            if merged_flag:
                merge_info[i_path] = (closest[0][1][2], closest[0][1][3])  # path index ... path point
            # update xy tree
            for j in range(len(_path_merged_)-3):  # don't include the exit points (don't want merges with them...)
                pt = (_path_merged_[j][0], _path_merged_[j][1], i_path, j)
                xy_tree.add([pt])

        # return the merged paths
        return paths, merge_info

    #
    # segmentLength()
    # - _segment_ = [(x0,y0),(x1,y1)]
    #
    def segmentLength(self, _segment_: tuple[tuple[float, float], tuple[float, float]]):
        """
        Returns the length of the segment.
        :param _segment_: a tuple of ((x0,y0), (x1,y1))
        :return: length of the segment as a float
        """
        dx, dy = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
        return sqrt(dx*dx+dy*dy)

    #
    # unitVector()
    # - _segment_ = [(x0,y0),(x1,y1)]
    #
    def unitVector(self, _segment_: tuple[tuple[float, float], tuple[float, float]]):
        """
        Returns the unit vector of the segment.
        :param _segment_: a tuple of ((x0,y0), (x1,y1))
        :return: unit vector of the segment as a tuple
        """
        dx, dy = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
        _len_  = sqrt(dx*dx+dy*dy)
        if _len_ < 0.0001:
            _len_ = 1.0
        return (dx/_len_, dy/_len_)

    #
    # bezierCurve() - parametric bezier curve object
    # - from Bezier Curve on wikipedia.org
    #
    def bezierCurve(self,
                    pt1:  tuple[float, float],
                    pt1p: tuple[float, float],
                    pt2p: tuple[float, float],
                    pt2:  tuple[float, float]):
        """
        Returns the parametric bezier curve object.
        :param pt1: point 1
        :param pt1p: point 1 parametric
        :param pt2p: point 2 parametric
        :param pt2: point 2
        :return: parametric bezier curve object -- call object with _obj_(t) where t is 0.0 to 1.0
        """
        class BezierCurve(object):

            def __init__(self, pt1, pt1p, pt2p, pt2):
                self.pt1, self.pt1p, self.pt2p, self.pt2 = pt1, pt1p, pt2p, pt2

            def __call__(self, t):
                return (1-t)**3*self.pt1[0]+3*(1-t)**2*t*self.pt1p[0]+3*(1-t)*t**2*self.pt2p[0]+t**3*self.pt2[0], \
                       (1-t)**3*self.pt1[1]+3*(1-t)**2*t*self.pt1p[1]+3*(1-t)*t**2*self.pt2p[1]+t**3*self.pt2[1]
        return BezierCurve(pt1, pt1p, pt2p, pt2)

    #
    # closestPointOnSegment() - find the closest point on the specified segment.
    # returns distance, point
    # ... for example:  10, (1,2)
    def closestPointOnSegment(self,
                              _segment_: tuple[tuple[float, float], tuple[float, float]],
                              _pt_:      tuple[float, float]):
        """
        :param _segment_: a tuple of ((x0,y0), (x1,y1))
        :param _pt_: a tuple of (x,y)
        :return: distance to the segment, closet point on the segment point in (x,y) tuple
        """
        if _segment_[0][0] == _segment_[1][0] and _segment_[0][1] == _segment_[1][1]:  # not a segment...
            dx, dy = _pt_[0] - _segment_[0][0], _pt_[1] - _segment_[0][1]
            return sqrt(dx*dx+dy*dy), _segment_[0]
        else:
            dx, dy = _pt_[0] - _segment_[0][0], _pt_[1] - _segment_[0][1]
            d0 = dx*dx+dy*dy
            dx, dy = _pt_[0] - _segment_[1][0], _pt_[1] - _segment_[1][1]
            d1 = dx*dx+dy*dy

            dx,  dy  = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
            pdx, pdy = -dy, dx
            _pt_line_ = (_pt_, (_pt_[0] + pdx, _pt_[1] + pdy))
            _ret_ = self.lineSegmentIntersectionPoint(_pt_line_, _segment_)
            if _ret_ is not None:
                dx, dy = _pt_[0] - _ret_[0], _pt_[1] - _ret_[1]
                d2 = dx*dx+dy*dy
                if d2 < d0 and d2 < d1:
                    return sqrt(d2), _ret_
                elif d0 < d1:
                    return sqrt(d0), _segment_[0]
                else:
                    return sqrt(d1), _segment_[1]
            else:
                if d0 < d1:
                    return sqrt(d0), _segment_[0]
                else:
                    return sqrt(d1), _segment_[1]

    #
    # intersectionPoint() - determine where two lines intersect
    # - returns None if the lines do not intersect
    #
    # From https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    #
    def intersectionPoint(self, 
                          line1: tuple[tuple[float, float], tuple[float, float]], 
                          line2: tuple[tuple[float, float], tuple[float, float]]):
        """
        Determine where two lines intersect.
        :param line1: a tuple of ((x0,y0), (x1,y1))
        :param line2: a tuple of ((x0,y0), (x1,y1))
        :return: intersection point in (x,y) tuple or None if the lines do not intersect
        """
        def floatify(line): return (float(line[0][0]), float(line[0][1])), (float(line[1][0]), float(line[1][1]))
        line1, line2 = floatify(line1), floatify(line2)
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b): return a[0] * b[1] - a[1] * b[0]
        div = det(xdiff, ydiff)
        if abs(div) < 0.0001 or div == 0: return None
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    #
    # lineSegmentIntersectionPoint() - determine where a line intersects a segment
    # - returns None if the line does not intersect the segment
    #
    def lineSegmentIntersectionPoint(self, 
                                     line:    tuple[tuple[float, float], tuple[float, float]], 
                                     segment: tuple[tuple[float, float], tuple[float, float]]):
        """
        Determine where a line intersects a segment.
        :param line: a tuple of ((x0,y0), (x1,y1))
        :param segment: a tuple of ((x0,y0), (x1,y1))
        :return: intersection point in (x,y) tuple or None if the line does not intersect the segment
        """
        def floatify(line): return (float(line[0][0]), float(line[0][1])), (float(line[1][0]), float(line[1][1]))
        line, segment = floatify(line), floatify(segment)
        # Would they intersect if they were both lines?
        results = self.intersectionPoint(line, segment)
        if results is None: return None
        # They intersect as lines... are the results on the segment?
        x, y = results
        if x >= min(segment[0][0], segment[1][0]) and x <= max(segment[0][0], segment[1][0]) and \
           y >= min(segment[0][1], segment[1][1]) and y <= max(segment[0][1], segment[1][1]): return x,y
        # That test should usually work... but rounding errors can cause it to fail...
        dx, dy = segment[1][0] - segment[0][0], segment[1][1] - segment[0][1] # x_t = x0 + t * dx and y_t = y0 + t * dy for t in [0,1]
        if abs(dx) >= 0.000001:
            t = (x - segment[0][0]) / dx
            if t >= 0.0 and t <= 1.0: return x, segment[0][1] + t * dy
        if abs(dy) >= 0.000001:
            t = (y - segment[0][1]) / dy
            if t >= 0.0 and t <= 1.0: return segment[0][0] + t * dx, y
        return None

    #
    # pointWithinSegment()
    #
    def pointWithinSegment(self, x: float, y: float, x0: float, y0: float, x1: float, y1: float):
        """
        Determine if a point is within a segment.
        :param x: x coordinate
        :param y: y coordinate
        :param x0: x coordinate of the segment start
        :param y0: y coordinate of the segment start
        :param x1: x coordinate of the segment end
        :param y1: y coordinate of the segment end
        :return: True if the point is within the segment, False otherwise and the fraction along the segment
        """
        dx, dy = x1 - x0, y1 - y0
        _xmin, _xmax = min(x0, x1), max(x0, x1)
        _ymin, _ymax = min(y0, y1), max(y0, y1)
        if x < _xmin or x > _xmax or y < _ymin or y > _ymax:
            return False, 0.0
        # xp = x0 + t * dx
        # yp = y0 + t * dy
        if dx == 0 and dy == 0:  # segment is a point...
            if x == x0 and y == y0:
                return True, 0.0
        elif dx == 0:  # it's vertical...
            t      = (y - y0)/dy
            xp, yp = x0 + t*dx, y0 + t*dy
            if xp == x and yp == y and t >= 0.0 and t <= 1.0:
                return True, t
        else:  # it's horizontal... or at least conforms to f(x)
            t      = (x - x0)/dx
            xp, yp = x0 + t*dx, y0 + t*dy
            if xp == x and yp == y and t >= 0.0 and t <= 1.0:
                return True, t
        return False, 0.0

    #
    # pointOnLine()
    #
    def pointOnLine(self, point, line):
        """ Determine if a point is on a line.  Returns True if the point is on the line, False otherwise and the fraction along the line.
        :param point: a tuple of (x,y)
        :param line: a tuple of ((x0,y0), (x1,y1))
        :return: True if the point is on the line, False otherwise and the fraction along the line
        """
        x0, y0, x1, y1 = line[0][0], line[0][1], line[1][0], line[1][1]
        x,  y          = point[0], point[1]
        dx, dy = x1 - x0, y1 - y0
        # xp = x0 + t * dx
        # yp = y0 + t * dy
        if dx == 0 and dy == 0:  # segment is a point...
            if x == x0 and y == y0:
                return True, 0.0
        elif dx == 0:  # it's vertical...
            t      = (y - y0)/dy
            xp, yp = x0 + t*dx, y0 + t*dy
            if xp == x and yp == y:
                return True, t
        else:  # it's horizontal... or at least conforms to f(x)
            t      = (x - x0)/dx
            xp, yp = x0 + t*dx, y0 + t*dy
            if xp == x and yp == y:
                return True, t
        return False, 0.0

    #
    # segmentsIntersect()
    # - do two segments intersect?
    # - return BOOLEAN, x_intersect, y_intersect, t_for_s0, t_for_s1 # I think :(
    #
    def segmentsIntersect(self, s0, s1):
        """
        Determine if two segments intersect.
        :param s0: a tuple of ((x0,y0), (x1,y1))
        :param s1: a tuple of ((x2,y2), (x3,y3))
        :return: True if the segments intersect, False otherwise, the x and the y coordinates of the intersection, and the fractions along each of the segments
        """
        x0, y0, x1, y1 = s0[0][0], s0[0][1], s0[1][0], s0[1][1]
        x2, y2, x3, y3 = s1[0][0], s1[0][1], s1[1][0], s1[1][1]
        _xmin, _ymin, _amin, _bmin = min(x0, x1), min(y0, y1), min(x2, x3), min(y2, y3)
        _xmax, _ymax, _amax, _bmax = max(x0, x1), max(y0, y1), max(x2, x3), max(y2, y3)

        # Determine if they share a point
        _small_number_ = 0.00001
        if abs(x0-x2) < _small_number_ and abs(y0-y2) < _small_number_: return True, x0, y0, 0.0, 0.0
        if abs(x0-x3) < _small_number_ and abs(y0-y3) < _small_number_: return True, x0, y0, 0.0, 1.0
        if abs(x1-x2) < _small_number_ and abs(y1-y2) < _small_number_: return True, x1, y1, 1.0, 0.0
        if abs(x1-x3) < _small_number_ and abs(y1-y3) < _small_number_: return True, x1, y1, 1.0, 1.0

        # Simple overlapping bounds test... as inexpensive as it gets...
        if _xmin > _amax or _amin > _xmax or _ymin > _bmax or _bmin > _ymax: return False, 0.0, 0.0, 0.0, 0.0
        # Both segments are points... Are they the same point?
        if _xmin == _xmax and _ymin == _ymax and _amin == _amax and _bmin == _bmax:
            if x0 == x2 and y0 == y2: return True, x0, y0, 0.0, 0.0
            return False, 0.0, 0.0, 0.0, 0.0

        A, B, C, D = y3 - y2, x3 - x2, x1 - x0, y1 - y0

        # x = x0 + t * C
        # t = (x - x0) / C
        # y = y0 + t * D
        # t = (y - y0) / D

        # Deal with parallel lines
        denom = B * D - A * C                # Cross Product
        if denom == 0.0:                     # Parallel...  and if co-linear, overlap because of the previous bounds test...
            online0, t0 = self.pointOnLine((x2, y2), ((x0, y0), (x1, y1)))
            online1, t1 = self.pointOnLine((x0, y0), ((x2, y2), (x3, y3)))
            if online0 or online1:
                onseg, t = self.pointWithinSegment(x0, y0, x2, y2, x3, y3)
                if onseg: return True, x0, y0, 0.0, t
                onseg, t = self.pointWithinSegment(x1, y1, x2, y2, x3, y3)
                if onseg: return True, x1, y1, 1.0, t
                onseg, t = self.pointWithinSegment(x2, y2, x0, y0, x1, y1)
                if onseg: return True, x2, y2, t, 0.0
                onseg, t = self.pointWithinSegment(x3, y3, x0, y0, x1, y1)
                if onseg: return True, x3, y3, t, 1.0

        # Normal calculation...
        t0 = (A*(x0 - x2) - B*(y0 - y2))/denom
        if t0 >= 0.0 and t0 <= 1.0:
            x    = x0 + t0 * (x1 - x0)
            y    = y0 + t0 * (y1 - y0)
            if (x3 - x2) != 0:
                t1   = (x - x2)/(x3 - x2)
                if t1 >= 0.0 and t1 <= 1.0: return True, x, y, t0, t1
            if (y3 - y2) != 0:
                t1   = (y - y2)/(y3 - y2)
                if t1 >= 0.0 and t1 <= 1.0: return True, x, y, t0, t1
        return False, 0.0, 0.0, 0.0, 0.0

    #
    # segmentIntersectsCircle() - does a line segment intersect a circle
    # - segment is ((x0,y0),(x1,y1))
    # - circle  is (cx, cy, r)
    # - modification from the following:
    # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
    #
    def segmentIntersectsCircle(self, segment, circle):
        """ Determines if a line segment intersects with a circle.
        :param segment: a tuple of ((x0,y0),(x1,y1))
        :param circle:  a tuple of (cx, cy, r)
        :return:        distance (from what?) ... and an xy tuple (of something)
        """
        A, B, C = segment[0], segment[1], (circle[0], circle[1])
        sub  = lambda a, b: (a[0] - b[0], a[1] - b[1])
        AC, AB = sub(C, A), sub(B, A)
        add  = lambda a, b: (a[0] + b[0], a[1] + b[1])
        dot  = lambda a, b: a[0]*b[0] + a[1]*b[1]
        def proj(a, b):
            k = dot(a, b)/dot(b, b)
            return (k*b[0], k*b[1])
        D = add(proj(AC, AB), A)
        AD = sub(D, A)
        if abs(AB[0]) > abs(AB[1]): k = AD[0] / AB[0]
        else:                       k = AD[1] / AB[1]
        hypot2 = lambda a, b: dot(sub(a, b), sub(a, b))
        if   k <= 0.0: return sqrt(hypot2(C, A)), C
        elif k >= 1.0: return sqrt(hypot2(C, B)), B
        else:          return sqrt(hypot2(C, D)), D

    #
    # Create a background lookup table (and the fill table) from a shape file...
    # ... fill table is complicated because line strings don't require fill...
    # ... utility method... tested on five shape files (land, coastlines, states, counties, zipcodes)...
    #
    # keep_as_shapely = keep as the shapely polygon
    # clip_rect       = 4-tuple of x0, y0, x1, y1
    # fill            = hex color string
    # naming          = naming function for series from geopandas dataframe
    #
    def createBackgroundLookupsFromShapeFile(self,
                                             shape_file,
                                             keep_as_shapely = True,
                                             clip_rect       = None,
                                             fill            = '#000000',
                                             naming          = None):
        """ Create a background lookup table (and the fill table) from a shape file...
        :param shape_file: path to shape file
        :param keep_as_shapely: keep as the shapely polygon
        :param clip_rect: 4-tuple of x0, y0, x1, y1
        :param fill: hex color string
        :param naming: naming function for series from geopandas dataframe
        :return: tuple of (bg_shape_lu, bg_fill_lu)
        """
        import geopandas as gpd
        gdf = gdf_orig = gpd.read_file(shape_file)
        if clip_rect is not None:
            gdf = gdf.clip_by_rect(clip_rect[0], clip_rect[1], clip_rect[2], clip_rect[3])
        bg_shape_lu, bg_fill_lu = {}, {}
        for i in range(len(gdf)):
            _series_ = gdf_orig.iloc[i]
            if clip_rect is None:
                _poly_ = gdf.iloc[i].geometry
            else:
                _poly_ = gdf.iloc[i]

            # Probably want to keep it as shapely if transforms are needed
            if keep_as_shapely:
                d = _poly_
            else:
                d = self.shapelyPolygonToSVGPathDescription(_poly_)

            # Store it
            if d is not None:
                _name_ = i
                if naming is not None:  # if naming function, call it with gpd series
                    _name_ = naming(_series_, i)
                bg_shape_lu[_name_] = d
                if type(_poly_) is LineString or type(_poly_) is MultiLineString: bg_fill_lu[_name_] = None
                else:                                                             bg_fill_lu[_name_] = fill
        return bg_shape_lu, bg_fill_lu

    #
    # Converts a shapely polygon to an SVG path...
    # ... assumes that the ordering (CW, CCW) of both the exterior and interior points is correct...
    # - if there's no shape in _poly, will return None
    #
    def shapelyPolygonToSVGPathDescription(self, _poly):
        """ Converts a shapely polygon to an SVG path...
        :param _poly: shapely polygon
        :return: SVG path string
        """
        #
        # MultiPolygon -- just break into individual polygons...
        #
        if type(_poly) is MultiPolygon:
            path_str = ''
            for _subpoly in _poly.geoms:
                if len(path_str) > 0: path_str += ' '
                path_str += self.shapelyPolygonToSVGPathDescription(_subpoly)
            return path_str
        #
        # LineString -- segments
        #
        elif type(_poly) is LineString:
            coords = _poly.coords
            path_str = f'M {coords[0][0]} {coords[0][1]} '
            for i in range(1, len(coords)): path_str += f'L {coords[i][0]} {coords[i][1]} '
            return path_str
        #
        # Multiple LineStrings -- break into individual line strings
        #
        elif type(_poly) is MultiLineString:
            path_str = ''
            for _subline in _poly.geoms:
                if len(path_str) > 0: path_str += ' '
                path_str += self.shapelyPolygonToSVGPathDescription(_subline)
            return path_str
        #
        # Polygon -- base polygon processing
        #
        elif type(_poly) is Polygon:
            # Draw the exterior shape first
            xx, yy = _poly.exterior.coords.xy
            path_str = f'M {xx[0]} {yy[0]} '
            for i in range(1, len(xx)): path_str += f' L {xx[i]} {yy[i]}'
            # Determine if the interior exists... and then render that...
            interior_len = len(list(_poly.interiors))
            if interior_len > 0:
                for interior_i in range(0, interior_len):
                    xx, yy = _poly.interiors[interior_i].coords.xy
                    path_str += f' M {xx[0]} {yy[0]}'
                    for i in range(1, len(xx)): path_str += f' L {xx[i]} {yy[i]}'
            return path_str + ' Z'
        #
        # GeometryCollection -- unsure of what this actual is...
        #
        elif type(_poly) is GeometryCollection:
            if len(_poly.geoms) > 0:  # Haven't seen this... so unsure of how to process
                raise Exception('shapelyPolygonToSVGPathDescription() - geometrycollection not empty')
            return None
        else:
            raise Exception('shapelyPolygonToSVGPathDescription() - cannot process type', type(_poly))

    #
    # Determine counterclockwise angle
    #
    def grahamScan_ccw(self, pt1, pt2, pt3):
        x1, y1 = pt1[0], pt1[1]
        x2, y2 = pt2[0], pt2[1]
        x3, y3 = pt3[0], pt3[1]
        return (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)

    #
    # grahamScan()
    # - compute the convex hull of x,y points in a lookup table
    # - lookup table is what networkx uses for layouts
    #
    # https://en.wikipedia.org/wiki/Graham_scan
    #
    def grahamScan(self, pos):
        """ Compute the convex hull of x,y points in a lookup table using the Graham Scan algorithm
        https://en.wikipedia.org/wiki/Graham_scan
        :param pos: lookup table of keys to xy tuples
        :return: convex hull as a list of xy tuples
        """
        # Find the lowest point... if same y coordinate, find the leftmost point
        pt_low = None
        for k in pos.keys():
            if pt_low is None:
                pt_low = k
            elif pos[k][1] < pos[pt_low][1]:
                pt_low = k
            elif pos[k][1] == pos[pt_low][1] and pos[k][0] < pos[pt_low][0]:
                pt_low = k

        # Sort all the other points by the polar angle from this point
        polar_lu = {}
        polar_d  = {}
        for k in pos.keys():
            if k != pt_low:
                dx    = pos[k][0] - pos[pt_low][0]
                dy    = pos[k][1] - pos[pt_low][1]
                l     = sqrt(dx*dx+dy*dy)
                if l < 0.001:
                    l = 0.001
                dx    = dx/l
                dy    = dy/l
                theta = acos(dx)
                if theta not in polar_lu.keys() or polar_d[theta] < l:
                    polar_lu[theta] = k
                    polar_d [theta] = l

        to_sort = []
        for x in polar_lu.keys(): to_sort.append((x, polar_lu[x]))
        points = sorted(to_sort)

        stack  = []
        for point in points:
            while len(stack) > 1 and self.grahamScan_ccw(pos[stack[-2][1]], pos[stack[-1][1]], pos[point[1]]) <= 0: stack = stack[:-1]
            stack.append(point)

        ret = []
        ret.append(pt_low)
        for x in stack: ret.append(x[1])
        return ret

    #
    # extrudePolyLine()
    # - Extrude the polyline returned by the grahamScan() method
    # - Returns a string designed for the path svg element
    #
    def extrudePolyLine(self,
                        pts,   # return value from grahamScan()
                        pos,   # original lookup passed into the grahamScan() algorithm
                        r=8):  # radius of the extrusion
        """ Extrude the polyline returned by the grahamScan() method
        :param pts: return value from grahamScan()
        :param pos: original lookup passed into the grahamScan() algorithm
        :param r: radius of the extrusion
        :return: string designed for the path svg element
        """
        d_str = ''

        for i in range(0, len(pts)):
            pt0 = pts[i]
            pt1 = pts[(i+1)%len(pts)]
            pt2 = pts[(i+2)%len(pts)]

            x0, y0 = pos[pt0][0], pos[pt0][1]
            x1, y1 = pos[pt1][0], pos[pt1][1]
            x2, y2 = pos[pt2][0], pos[pt2][1]

            dx, dy = x1 - x0, y1 - y0
            l  = sqrt(dx*dx+dy*dy)
            if l < 0.001: l = 0.001
            dx /= l
            dy /= l
            pdx =  dy
            pdy = -dx

            dx2, dy2 = x2 - x1, y2 - y1
            l2  = sqrt(dx2*dx2+dy2*dy2)
            if l2 < 0.001: l2 = 0.001
            dx2 /= l2
            dy2 /= l2
            pdx2 =  dy2
            pdy2 = -dx2

            # First point is a move to...
            if len(d_str) == 0: d_str += f'M {x0+pdx*r} {y0+pdy*r} '

            # Line along the the polygon edge
            d_str += f'L {x1+pdx*r} {y1+pdy*r} '

            # Curved cap
            cx0 = x1+pdx  * r + dx  * r/4
            cy0 = y1+pdy  * r + dy  * r/4
            cx1 = x1+pdx2 * r - dx2 * r/4
            cy1 = y1+pdy2 * r - dy2 * r/4

            d_str += f'C {cx0} {cy0} {cx1} {cy1} {x1+pdx2*r} {y1+pdy2*r}'

            # d_str += f'L {cx0} {cy0} '
            # d_str += f'L {cx1} {cy1} '
            d_str += f'L {x1+pdx2*r} {y1+pdy2*r} '

        return d_str

    #
    # Algorithm source & description from:
    # "A simple algorithm for 2D Voronoi diagrams"
    # https://www.youtube.com/watch?v=I6Fen2Ac-1U
    # https://gist.github.com/isedgar/d445248c9ff6c61cef44fc275cb2398f
    #
    def isedgarVoronoi(self, S, Box=None, pad = 10):
        # return bisector of points p0 and p1 -- bisector is ((x,y),(u,v)) where u,v is the vector of the bisector
        def bisector(p0, p1):
            x, y     = (p0[0] + p1[0])/2.0, (p0[1] + p1[1])/2.0
            uv       = self.unitVector((p0, p1))
            pdx, pdy = uv[1], -uv[0]
            return ((x,y), (pdx,pdy))
        # returns vertices that intersect the polygon
        def intersects(bisects, poly):
            inters = []
            for i in range(0, len(poly)):
                p0, p1 = poly[i], poly[(i+1)%len(poly)]
                xy = self.lineSegmentIntersectionPoint((bisects[0], (bisects[0][0] + bisects[1][0], bisects[0][1] + bisects[1][1])),
                                                       (p0, p1))
                if xy is not None and xy != p1: inters.append((xy, i, (i+1)%len(poly)))
            return inters
        # conctenate a point, a list, and a point into a new list
        def createCell(my_cell, p0, p1, i0, i1, sign):
            l = [p0]
            i = i0
            while i != i1:
                l.append(my_cell[i])
                i += sign
                if i < 0:             i += len(my_cell)
                if i >= len(my_cell): i -= len(my_cell)
            l.append(my_cell[i1])
            l.append(p1)
            return l
        # contain true if pt is in poly
        def contains(poly, pt):
            inter_count = 0
            for i in range(len(poly)):
                p0, p1 = poly[i], poly[(i+1)%len(poly)]
                _tuple_ = self.segmentsIntersect((pt,(pt[0]+1e9,pt[1])),(p0,p1)) # use a ray from the pt to test
                if _tuple_[0] and (_tuple_[1], _tuple_[2]) != p1: inter_count += 1
            if inter_count%2 == 1: return True
            return False
        # actual algorithm
        if Box is None:
            x_l, x_r, y_t, y_b = S[0][0], S[0][0], S[0][1], S[0][1]
            for pt in S: x_l, x_r, y_t, y_b = min(x_l, pt[0]), max(x_r, pt[0]), max(y_t, pt[1]), min(y_b, pt[1])
            Box = [(x_l-pad, y_t+pad), (x_r+pad, y_t+pad), (x_r+pad, y_b-pad), (x_l-pad, y_b-pad)]
        cells = []
        for i in range(len(S)):
            p    = S[i]
            cell = Box
            for q in S:
                if p == q: continue
                B            = bisector(p,q)
                B_intersects = intersects(B, cell)
                if len(B_intersects) == 2:
                    t1, t2 = B_intersects[0][0], B_intersects[1][0]
                    xi, xj = B_intersects[0][2], B_intersects[1][1]
                    newCell      = createCell(cell, t1, t2, xi, xj, 1)
                    xi, xj = B_intersects[0][1], B_intersects[1][2]
                    otherNewCell = createCell(cell, t1, t2, xi, xj, -1)
                    if contains(newCell, p) == False: newCell = otherNewCell
                    cell = newCell
            cells.append(cell)
        return cells

    #
    # threePointCircle() - given three points, determine the circle that passes through those points
    # ... does not check for collinearity ... just fails ungracefully :(
    #
    # Derived From:
    # https://math.stackexchange.com/questions/213658/get-the-equation-of-a-circle-when-given-3-points
    #
    # Original Source (from that page):
    # https://web.archive.org/web/20161011113446/http://www.abecedarical.com/zenosamples/zs_circle3pts.html
    #
    def threePointCircle(self, p1, p2, p3):
        x1,y1 = p1
        x2,y2 = p2
        x3,y3 = p3
        _denom_ = np.linalg.det([[x1, y1, 1.0],
                                [x2, y2, 1.0],
                                [x3, y3, 1.0]])
        _x_num_ = np.linalg.det([[x1**2 + y1**2, y1, 1.0],
                                [x2**2 + y2**2, y2, 1.0], 
                                [x3**2 + y3**2, y3, 1.0]])
        _y_num_ = np.linalg.det([[x1**2 + y1**2, x1, 1.0],
                                [x2**2 + y2**2, x2, 1.0], 
                                [x3**2 + y3**2, x3, 1.0]])    
        x0 = ( 1.0/2.0) * _x_num_/_denom_
        y0 = (-1.0/2.0) * _y_num_/_denom_
        r0 = np.sqrt((x0 - x1)**2 + (y0 - y1)**2)
        return (x0, y0, r0)

    #
    # counterClockwiseOrder() - arrange the pts in counter-clockwise order
    # pts = [(x0,y0), (x1,y1) ...]
    # c   = (x,y,r) # circle
    #
    # Derived From:
    # https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order
    #
    def counterClockwiseOrder(self, pts, c):
        det = lambda xy0, xy1: (xy0[0] - c[0]) * (xy1[1] - c[1]) - (xy1[0] - c[0]) * (xy0[1] - c[1])
        def less(xy0, xy1):
            if xy0[0] - c[0] >= 0 and xy1[0] - c[0] < 0:  return False
            if xy0[0] - c[0] <  0 and xy1[0] - c[0] >= 0: return True
            if xy0[0] - c[0] == 0 and xy1[0] - c[0] == 0:
                if xy0[1] - c[1] >= 0 or xy1[1] - c[1] >= 0: return xy0[1] <= xy1[1]
                return xy1[1] < xy0[1]
            if det(xy0,xy1) >= 0: return True
            else:                 return False
        # insertion sort
        ordered = [pts[0]]
        for i in range(1, len(pts)):
            j = 0
            while j < len(ordered) and less(ordered[j], pts[i]): j += 1
            ordered.insert(j, pts[i])
        # reverse
        return ordered[::-1]

    #
    # should be called "pointWithinCircumcircle" # but i'll never remember what that means
    # ... is this really faster than doing a square root?
    #
    # Derived From:
    # https://en.wikipedia.org/wiki/Delaunay_triangulation
    #
    def pointWithinThreePointCircle(self, pt, p1, p2, p3):
        d          = pt
        x0, y0, r0 = self.threePointCircle(p1, p2, p3)
        ordered    = self.counterClockwiseOrder([p1, p2, p3], (x0,y0,r0))
        a, b, c    = ordered[0], ordered[1], ordered[2]
        return np.linalg.det([[a[0] - d[0], a[1] - d[1], (a[0] - d[0])**2 + (a[1] - d[1])**2],
                              [b[0] - d[0], b[1] - d[1], (b[0] - d[0])**2 + (b[1] - d[1])**2],
                              [c[0] - d[0], c[1] - d[1], (c[0] - d[0])**2 + (c[1] - d[1])**2]]) <= 0

    #
    # bowyerWatson() - O(n^2) delaunay triangulation implementation
    # -- not sure this is correct
    #
    # Derived from:
    # https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm
    #
    def bowyerWatson(self, pts):
        # Bounds
        x_min, y_min, x_max, y_max = pts[0][0], pts[0][1], pts[0][0], pts[0][1]
        for i in range(1, len(pts)):
            x_min, y_min = min(x_min, pts[i][0]), min(y_min, pts[i][1])
            x_max, y_max = max(x_max, pts[i][0]), max(y_max, pts[i][1])

        # Super triangle
        _tip_                 = ((x_min+x_max)/2.0, (y_min - (y_max-y_min)))
        _y_base_              = y_max + (y_max-y_min)
        _x0_base_, _x1_base_  = x_min - (x_max-x_min), x_max + (x_max-x_min)
        super_triangle        = (_tip_, (_x0_base_, _y_base_), (_x1_base_, _y_base_))

        # Triangulation
        triangulation = set()
        triangulation.add(super_triangle)

        # Main loop
        for point in pts:
            # Find triangles that contain this point in their circumcircles
            bad_triangles = set()
            for triangle in triangulation:
                xy0, xy1, xy2 = triangle
                if self.pointWithinThreePointCircle(point, xy0, xy1, xy2):
                    bad_triangles.add(triangle)
            # Normalize an edge
            def normEdge(p0, p1):
                if   p0[0] < p1[0]: return (p0, p1)
                elif p0[0] > p1[0]: return (p1, p0)
                elif p0[1] < p1[1]: return (p0, p1)
                else:               return (p1, p0)
            # Determine which edges are unique by filling in the edge lookup w/ points to their associated triangles
            edge_lu = {}
            for triangle in bad_triangles:
                xy0, xy1, xy2 = triangle
                e0, e1, e2 = normEdge(xy0, xy1), normEdge(xy1, xy2), normEdge(xy2, xy0)
                if e0 not in edge_lu: edge_lu[e0] = set()
                if e1 not in edge_lu: edge_lu[e1] = set()
                if e2 not in edge_lu: edge_lu[e2] = set()
                edge_lu[e0].add(triangle), edge_lu[e1].add(triangle), edge_lu[e2].add(triangle)

            # Construct a polygon of the unique edges
            polygon = set()
            for edge in edge_lu:
                if len(edge_lu[edge]) == 1: polygon.add(edge)
            
            # Remove all bad triangles from the triangulation
            triangulation = triangulation - bad_triangles

            # For each edge in the polygon, add a new triangle
            for edge in polygon:
                xy0, xy1 = edge
                triangulation.add((xy0, xy1, point))
        
        # Remove any triangles that had a vertex from the super triangle
        to_remove = set()
        for triangle in triangulation:
            if triangle[0] in super_triangle or triangle[1] in super_triangle or triangle[2] in super_triangle:
                to_remove.add(triangle)
        triangulation = triangulation - to_remove
        
        return triangulation

    #
    # levelSetFast()
    # - raster is a two dimensional structure ... _raster[y][x]
    # - "0" or None means to calculate
    # - "-1" means a wall / immovable object
    # - "> 0" means the class to expand
    # - Faster version doesn't correctly model obstacles... slower version is more precise
    #
    def levelSetFast(self,
                     _raster):
        """ Perform a level set operation on an integer raster (2d array of integers).

        Within the raster parameter, the following schema is used:
        - "0" or None means to calculate
        - "-1" means a wall / immovable object
        - "> 0" means the class to expand

        Faster version doesn't correctly model obstacles... slower version is more precise

        :param _raster: 2d array of integers
        :return: 2d array of integers (class), 2d array of floats (time found or distance), and 2d array of xy tuples (origin of class)
        """
        h, w = len(_raster), len(_raster[0])

        # Allocate the level set
        state      = [[None for x in range(w)] for y in range(h)]  # node that found the pixel
        found_time = [[None for x in range(w)] for y in range(h)]  # when node was found
        origin     = [[None for x in range(w)] for y in range(h)]  # when node was found

        # Distance lambda function
        dist = lambda _x0, _y0, _x1, _y1: sqrt((_x0-_x1)*(_x0-_x1)+(_y0-_y1)*(_y0-_y1))

        # Copy the _raster
        for x in range(0, len(_raster[0])):
            for y in range(0, len(_raster)):
                if _raster[y][x] is not None and _raster[y][x] != 0:
                    state[y][x]      = _raster[y][x]  # class of the find
                    found_time[y][x] = 0              # what time it was found
                    origin[y][x]     = (y, x)         # origin of the finder

        # Initialize the heap
        _heap = []
        for x in range(0, len(_raster[0])):
            for y in range(0, len(_raster)):
                if state[y][x] is not None and state[y][x] > 0:  # Only expand non-walls and set indices...
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0: continue
                            xn, yn = x+dx, y+dy
                            if xn >= 0 and yn >= 0 and xn < w and yn < h:
                                if state[yn][xn] is None or state[yn][x] == 0:
                                    t = dist(x, y, xn, yn)
                                    heapq.heappush(_heap, (t, xn, yn, state[y][x], origin[y][x][0], origin[y][x][1]))

        # Go through the heap
        while len(_heap) > 0:
            t, xi, yi, _class, y_origin, x_origin = heapq.heappop(_heap)
            if state[yi][xi] is not None and state[yi][xi] < 0:           # Check for a wall
                continue
            if found_time[yi][xi] is None or found_time[yi][xi] > t:      # Deterimine if we should overwrite the state
                state [yi][xi]     = _class
                found_time[yi][xi] = t
                origin[yi][xi]      = (y_origin, x_origin)
                for dx in range(-1, 2):                                    # Add the neighbors to the priority queue
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        xn = xi + dx
                        yn = yi + dy
                        if xn >= 0 and yn >= 0 and xn < w and yn < h:
                            # This calculation isn't exactly correct... because it doesn't consider that the
                            # position may have been reached direct via line-of-sight...  however, because we
                            # have the possibility of walls, unsure how to smooth this one out...
                            t = found_time[yi][xi] + dist(xi, yi, xn, yn)
                            if found_time[yn][xn] is None or found_time[yn][xn] > t:
                                heapq.heappush(_heap, (t, xn, yn, state[y_origin][x_origin], y_origin, x_origin))

        return state, found_time, origin


    #
    # levelSetBalanced() - grow a balanced level set from a set of origins
    # ... results are not guaranteed to have the origin in the nodes found by that origin..
    # ... this occurs because upto 3 originals may be located on the same raster coordinate
    #
    def levelSetBalanced(self,
                        raster,      # raster[y][x]  - either None, empty set, or a set of integers
                        origins,     # origins       - list of origins (as integer indices)
                        epsilon=3):  # epsilon       - max delta in set sizes
        h, w = len(raster), len(raster[0])
        origins_as_set = set(origins)

        # Allocate the level set
        state      = [[None for x in range(w)] for y in range(h)]  # node that found the pixel
        found_time = [[None for x in range(w)] for y in range(h)]  # time when pixel was found

        # Distance lambda function
        dist = lambda _xy0_, _xy1_: sqrt((_xy0_[0]-_xy1_[0])**2+(_xy0_[1]-_xy1_[1])**2)

        # Copy the raster
        node_to_xy = {} 
        for x in range(0, len(raster[0])):
            for y in range(0, len(raster)):
                if raster[y][x] is not None:
                    for node in raster[y][x]: node_to_xy[node] = (x, y)

        # Every origin gets it's own heap
        heaps       = {} # for each origin, a priority queue of the current frontier
        finds       = {} # for each origin, a set of nodes that have been found
        progress_lu = {'origin':[], 'iteration':[], 'heapsize':[]}
        for o in origins_as_set: 
            heaps[o] = []
            finds[o] = set()
            progress_lu['origin'].append(o), progress_lu['iteration'].append(0), progress_lu['heapsize'].append(0)

        # Initialize the heaps
        for x in range(0, len(raster[0])):
            for y in range(0, len(raster)):
                if raster[y][x] is not None:
                    origins_here = list(raster[y][x] & origins_as_set)
                    if len(origins_here) > 3: raise Exception(f'levelSetBalanced() - too many origins {origins_here} at ({x}, {y}) (no more than 3)')
                    if len(origins_here) > 0:
                        if len(origins_here) == 1: heapq.heappush(heaps[origins_here[0]], (0.0, (x, y), origins_here[0], (x, y)))
                        else:
                            if x == 0 or y == 0 or x == w-1 or y == h-1:
                                raise Exception('levelSetBalanced() - 2 --> 8 not implemented if the origin node is on the edge of the raster')
                            if   len(origins_here) == 2:
                                heapq.heappush(heaps[origins_here[0]], (0.0, (x,   y), origins_here[0], (x,   y)))
                                heapq.heappush(heaps[origins_here[1]], (0.0, (x+1, y), origins_here[1], (x+1, y)))
                            elif len(origins_here) == 3:
                                heapq.heappush(heaps[origins_here[0]], (0.0, (x,   y),   origins_here[0], (x,   y)))
                                heapq.heappush(heaps[origins_here[1]], (0.0, (x+1, y),   origins_here[1], (x+1, y)))
                                heapq.heappush(heaps[origins_here[2]], (0.0, (x,   y+1), origins_here[2], (x ,  y+1)))
                                
        def maxInFinds(): return max(len(finds[o]) for o in origins_as_set) # what is the max number of finds from any origin?

        # Main loop
        def anyHeapsHaveElements(): return any(len(heap) > 0 for heap in heaps.values()) # do any of the heaps have elements to process?
        iterations = 0
        while anyHeapsHaveElements(): # and iterations < len(origins)*w*h:
            # What is the maximum size of any of the finds?
            _max_ = maxInFinds()

            # Determine which origins will progress -- anything smaller than the _max_ found (plus the epsilon)
            # ... in cases, where nothing meets that criteria, only progress the smallest one that still has a heap
            _origins_to_progress_ = []
            for o in origins_as_set:
                if len(heaps[o]) > 0 and len(finds[o]) < _max_ + epsilon: 
                    _origins_to_progress_.append(o)

            if len(_origins_to_progress_) == 0: # pick the smallest one to progress
                _sz_, _best_ = 1e9, None
                for o in origins:
                    if len(heaps[o]) > 0:
                        if   _best_ is None:       _best_, _sz_ = o, len(finds[o])
                        elif _sz_ > len(finds[o]): _best_, _sz_ = o, len(finds[o])
                _origins_to_progress_ = [_best_]
            
            # Progress the selected origins
            for o in _origins_to_progress_:
                if len(heaps[o]) == 0: continue
                t, xyi, _, xyo = heapq.heappop(heaps[o])
                if state[xyi[1]][xyi[0]] is None or found_time[xyi[1]][xyi[0]] > t: # either not found (None) or found later than this origin would have found it in...
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0: continue
                            xn  = xyi[0] + dx
                            yn  = xyi[1] + dy
                            xyn = (xn,yn)
                            if xn < 0 or xn >= w or yn < 0 or yn >= h: continue # off edge... skip
                            t_found = dist(xyo,xyn)
                            if state[yn][xn] is None or t_found < found_time[yn][xn]: # if not found or found earlier (by this origin)
                                heapq.heappush(heaps[o], (t_found, xyn, o, xyo))
                    if raster[xyi[1]][xyi[0]] is not None and (len(raster[xyi[1]][xyi[0]])) > 0: # finds to add
                        if state[xyi[1]][xyi[0]] is not None: # need to remove them from the other finder...
                            _other_finder_ = state[xyi[1]][xyi[0]]
                            for _finds_ in raster[xyi[1]][xyi[0]]: finds[_other_finder_].remove(_finds_) # removal
                        for _finds_ in raster[xyi[1]][xyi[0]]: finds[o].add(_finds_) # add to the new finder

                    state[xyi[1]][xyi[0]], found_time[xyi[1]][xyi[0]] = o, t

            for o in origins_as_set: 
                progress_lu['iteration'].append(iterations+1) # because zero was initialized earlier...
                progress_lu['origin'].append(o)
                progress_lu['heapsize'].append(len(heaps[o]))

            iterations += 1

        if anyHeapsHaveElements():
            for x in heaps:
                print(f'heap for {x} has {len(heaps[x])} elements')

        all_finds = set()
        for x in finds: all_finds = all_finds.union(finds[x])
        if len(all_finds) != len(node_to_xy.keys()):
            print(finds)
            raise Exception(f'levelSetBalanced() - not all nodes found {all_finds} != {node_to_xy.keys()}')

        # progress_lu is just for debug...
        # ... recommended usage:  rt.xy(pl.DataFrame(my_progress_lu), x_field='iteration', y_field='heapsize', color_by='origin', dot_size='small', w=1024, h=128)
        return state, found_time, finds, progress_lu

    #
    # Implemented from pseudocode on https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    #
    def __bresenhamsLow__(self, x0, y0, x1, y1):
        dx, dy, yi, pts = x1 - x0, y1 - y0, 1, []
        if dy < 0:
            yi, dy = -1, -dy
        D, y = (2*dy)-dx, y0
        for x in range(x0, x1+1):
            pts.append((x, y))
            if D > 0:
                y += yi
                D += 2*(dy-dx)
            else:
                D += 2*dy
        return pts

    #
    # Implemented from pseudocode on https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    #
    def __bresenhamsHigh__(self, x0, y0, x1, y1):
        dx, dy, xi, pts = x1 - x0, y1 - y0, 1, []
        if dx < 0:
            xi, dx = -1, -dx
        D, x = (2*dx)-dy, x0
        for y in range(y0, y1+1):
            pts.append((x, y))
            if D > 0:
                x += xi
                D += 2*(dx-dy)
            else:
                D += 2*dx
        return pts

    #
    # bresenhams() - returns list of points on the pixelized (discrete) line from (x0,y0) to (x1,y1)
    # - Implemented from pseudocode on https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    #
    def bresenhams(self, x0, y0, x1, y1):
        if abs(y1-y0) < abs(x1-x0): return self.__bresenhamsLow__ (x1, y1, x0, y0) if (x0 > x1) else self.__bresenhamsLow__ (x0, y0, x1, y1)
        else:                       return self.__bresenhamsHigh__(x1, y1, x0, y0) if (y0 > y1) else self.__bresenhamsHigh__(x0, y0, x1, y1)

    #
    # levelSet() - slower version but more precise (takes objects into consideration)
    # - takes approximately 10x times as long as the fast method... (with small rasters... < 256x256)
    # - raster is a two dimensional structure ... _raster[y][x]
    # - "0" or None means to calculate
    # - "-1" means a wall / immovable object
    # - "> 0" means the class to expand
    #
    def levelSet(self, _raster):
        """ Perform a level set operation on an integer raster (2d array of integers).

        Within the raster parameter, the following schema is used:
        - "0" or None means to calculate
        - "-1" means a wall / immovable object
        - "> 0" means the class to expand

        This is the slower version and (more) precisely models obstacles.  Use levelSetFast() if possible.

        :param _raster: 2d array of integers
        :return: 2d array of integers (class), 2d array of floats (time found or distance), and 2d array of xy tuples (origin of class)
        """
        h, w = len(_raster), len(_raster[0])

        # Allocate the level set
        state      = [[None for x in range(w)] for y in range(h)]  # node that found the pixel
        found_time = [[None for x in range(w)] for y in range(h)]  # when node was found
        origin     = [[None for x in range(w)] for y in range(h)]  # when node was found

        # Distance lambda function
        dist = lambda _x0, _y0, _x1, _y1: sqrt((_x0-_x1)*(_x0-_x1)+(_y0-_y1)*(_y0-_y1))

        # Copy the _raster
        for x in range(0, len(_raster[0])):
            for y in range(0, len(_raster)):
                if _raster[y][x] is not None and _raster[y][x] != 0:
                    state[y][x]      = _raster[y][x]  # class of the find
                    found_time[y][x] = 0              # what time it was found
                    origin[y][x]     = (y, x)         # origin of the finder

        # Initialize the heap
        _heap = []
        for x in range(0, len(_raster[0])):
            for y in range(0, len(_raster)):
                if state[y][x] is not None and state[y][x] > 0:  # Only expand non-walls and set indices...
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue
                            xn, yn = x+dx, y+dy
                            if xn >= 0 and yn >= 0 and xn < w and yn < h:
                                if state[yn][xn] is None or state[yn][x] == 0:
                                    t = dist(x, y, xn, yn)
                                    heapq.heappush(_heap, (t, xn, yn, state[y][x], origin[y][x][0], origin[y][x][1]))

        # Go through the heap
        while len(_heap) > 0:
            t, xi, yi, _class, y_origin, x_origin = heapq.heappop(_heap)
            t = dist(xi, yi, x_origin, y_origin) + found_time[y_origin][x_origin]
            if state[yi][xi] is not None and state[yi][xi] < 0:           # Check for a wall
                continue
            if found_time[yi][xi] is None or found_time[yi][xi] > t:      # Deterimine if we should overwrite the state
                state [yi][xi]     = _class
                found_time[yi][xi] = t
                origin[yi][xi]      = (y_origin, x_origin)
                for dx in range(-1, 2):                                   # Add the neighbors to the priority queue
                    for dy in range(-1, 2):
                        if dx == 0 and dy == 0:
                            continue
                        xn, yn = xi + dx, yi + dy
                        # Within bounds?
                        if xn >= 0 and yn >= 0 and xn < w and yn < h:
                            t = found_time[yi][xi] + dist(xi, yi, xn, yn)
                            # Adjust the origin if we can't see the origin from the new point...
                            x_origin_adj, y_origin_adj = x_origin, y_origin
                            path = self.bresenhams(xn, yn, x_origin, y_origin)
                            for pt in path:
                                if state[pt[1]][pt[0]] is not None and state[pt[1]][pt[0]] < 0: x_origin_adj, y_origin_adj = xi, yi
                            if found_time[yn][xn] is None or found_time[yn][xn] > t:
                                heapq.heappush(_heap, (t, xn, yn, state[y_origin][x_origin], y_origin_adj, x_origin_adj))
        return state, found_time, origin

    #
    #
    #
    def levelSetStateAndFoundTimeSVG(self, _state, _found_time):
        _w, _h = len(_state[0]), len(_state)
        svg = f'<svg x="0" y="0" width="{_w*2}" height="{_h}">'
        _tmax = 1
        for y in range(0, _h):
            for x in range(0, _w):
                if _found_time[y][x] is not None and _found_time[y][x] > _tmax: _tmax = _found_time[y][x]
        for y in range(0, _h):
            for x in range(0, _w):
                if _state[y][x] == -1: _co = '#000000'
                else:                  _co = self.co_mgr.getColor(_state[y][x])
                svg += f'<rect x="{x}" y="{y}" width="{1}" height="{1}" fill="{_co}" stroke-opacity="0.0" />'
                if _found_time[y][x] is not None:
                    if _state[y][x] == -1: _co = '#000000'
                    else:                  _co = self.co_mgr.spectrum(_found_time[y][x], 0, _tmax)
                else: _co = '#ffffff'  # shouldn't really ever be here...
                svg += f'<rect x="{x+_w}" y="{y}" width="{1}" height="{1}" fill="{_co}" stroke-opacity="0.0" />'
        return svg + '</svg>'

    #
    # smoothSegments() - smooth out segments with a 3 window kernel.
    #
    def smoothSegments(self, segments):
        """ Smooth out a list of segments with a 3 window kernel.
        :param segments: list of (x, y) tuples (not segments as the name would imply)
        :return:         smoothed list of (x, y) tuples 
        """
        smoothed = [segments[0]]
        for i in range(1, len(segments)-1):
            x, y = (segments[i-1][0] + segments[i][0] + segments[i+1][0])/3.0, (segments[i-1][1] + segments[i][1] + segments[i+1][1])/3.0
            smoothed.append((x, y))
        smoothed.append(segments[-1])
        return smoothed

    #
    # expandSegmentsIntoPiecewiseCurvePartsFIXEDINC()
    # ... old version of this method...
    #
    def expandSegmentsIntoPiecewiseCurvedPartsFIXEDINC(self, segments, amp=5.0, ampends=20.0, t_inc=0.1):
        _piecewise_ = [segments[0], segments[1]]
        for i in range(1, len(segments)-2):
            _amp_ = ampends if ((i == 1) or (i == len(segments)-3)) else amp
            v0 = self.unitVector([segments[i],   segments[i-1]])
            v1 = self.unitVector([segments[i+1], segments[i+2]])
            bc = self.bezierCurve(segments[i],  (segments[i][0]-_amp_*v0[0], segments[i][1]-_amp_*v0[1]), (segments[i+1][0]-_amp_*v1[0], segments[i+1][1]-_amp_*v1[1]), segments[i+1])
            t = 0.0
            while t < 1.0:
                _piecewise_.append(bc(t))
                t += t_inc
        _piecewise_.append(segments[-1])
        return _piecewise_

    #
    # expandSegmentsIntoPiecewiseCurvedParts()
    # - expand segments into piecewise segments
    #
    def expandSegmentsIntoPiecewiseCurvedParts(self, segments, amp=5.0, ampends=20.0, max_travel=2.0):
        """ Expand linear segments into curved bezier segments by using the before and after segments to control the curve.
        :param segments: list of (x, y) tuples (not segments as the name would imply)
        :param amp:      how much to expand the curve in the before and after segments
        :param ampends:  how much to expand the curve in the ends
        :return:         list of (x, y) tuples
        """
        _piecewise_ = [segments[0], segments[1]]
        for i in range(1, len(segments)-2):
            _amp_ = ampends if ((i == 1) or (i == len(segments)-3)) else amp
            v0 = self.unitVector([segments[i],   segments[i-1]])
            v1 = self.unitVector([segments[i+1], segments[i+2]])
            bc = self.bezierCurve(segments[i], (segments[i][0]-_amp_*v0[0], segments[i][1]-_amp_*v0[1]), (segments[i+1][0]-_amp_*v1[0], segments[i+1][1]-_amp_*v1[1]), segments[i+1])
            t_lu = {}
            ts   = []
            t_lu[0.0] = bc(0.0)
            ts.append(0.0)
            t_lu[1.0] = bc(1.0)
            ts.append(1.0)
            j = 0
            while j < len(ts)-1:
                l = self.segmentLength((t_lu[ts[j]], t_lu[ts[j+1]]))
                if l > max_travel:
                    t_new = (ts[j] + ts[j+1])/2.0
                    ts.insert(j+1, t_new)
                    t_lu[t_new] = bc(t_new)
                else:
                    j += 1
            for j in range(0, len(ts)-1):
                _piecewise_.append(t_lu[ts[j]])
        _piecewise_.append(segments[-1])
        return _piecewise_

    #
    # piecewiseCubicBSpline() - interpolate piecewise cubic b-spline
    # - Replicates formulas/implementation from "Hierarchical Edge Bundles: Visualization of Adjacency Relations in Hierarchical Data" by Danny Holten (2006)
    # - [fixed] there's an error in the continuity between the 4th degree and 3rd degree connections (on both ends)
    # -- see the hierarchical_edge_bundling.ipynb test file for an example
    # -- fixed by hacking the first three points and last three points (not consistent w/ 2nd and 3rd order cubic b-splines)
    # - this version returns points which is less efficient for constructing svg structures
    # - t_inc should be something that (when added to a single precision number (e.g., 0.1)
    #   will eventually add to another single precision number evenly)
    #
    def piecewiseCubicBSpline(self, pts, beta=0.8, t_inc=0.1):
        _min_d_threshold_ = 0.5
        points = []
        # Formula 1 form the Holten Paper - generates a control point
        def cP(i, n, p_0, p_i, p_n_minus_1):
            _fn_ = lambda k: beta * p_i[k] + (1.0 - beta) * (p_0[k] + ((i/(n-1)) * (p_n_minus_1[k] - p_0[k])))
            return (_fn_(0), _fn_(1))

        # Generate all the control points
        i, cps = 0, []
        for i in range(len(pts)):
            xy = cP(i, len(pts), pts[0], pts[i], pts[-1])
            cps.append(xy)

        # Hack the first three points into a bezier curve
        pt_beg = cps[0]
        pt_end   = ((1/6) *   (cps[0][0] + 4*cps[1][0] + cps[2][0]),  (1/6) *   (cps[0][1] + 4*cps[1][1] + cps[2][1]))
        pt_end_d = ((1/3) * (2*cps[1][0] +   cps[2][0]),              (1/3) * (2*cps[1][1] +   cps[2][1]))
        pt_mid_e = (pt_end[0] + (pt_end[0] - pt_end_d[0]), pt_end[1] + (pt_end[1] - pt_end_d[1]))
        pt_mid_b = (pt_beg[0] + pt_mid_e[0])/2, (pt_beg[1] + pt_mid_e[1])/2
        bezier   = self.bezierCurve(pt_beg, pt_mid_b, pt_mid_e, pt_end)
        t = 0.0
        while t < 1.0:
            _xy_ = bezier(t)
            if len(points) > 0: d2 = sqrt((points[-1][0] - _xy_[0])**2 + (points[-1][1] - _xy_[1])**2)
            else:               d2 = _min_d_threshold_
            if d2 >= _min_d_threshold_: points.append(_xy_)
            t += t_inc

        # For every four points, use the wikipedia interpolation...
        # - it'd be faster to use the bezier implementation from SVG (see the test file) ... but if you want to colorize it,
        #   there's no implementation within SVG to shade across the curve...
        for i in range(len(cps)-3):
            # Copied from wikipedia page on B-splines -- https://en.wikipedia.org/wiki/B-spline
            b0, b1, b2, b3 = cps[i], cps[i+1], cps[i+2], cps[i+3]
            t = 0.0
            while t < 1.0:
                cT = lambda _t_, k: (1/6) * ((-b0[k] + 3*b1[k] - 3*b2[k] + b3[k])*_t_**3 + (3*b0[k] - 6*b1[k] + 3*b2[k])*_t_**2 + (-3*b0[k] + 3*b2[k])*_t_ + (b0[k] + 4*b1[k] + b2[k]))
                x1, y1 = cT(t, 0), cT(t, 1)
                d2     = sqrt((x1-points[-1][0])**2 + (y1-points[-1][1])**2)
                if d2 >= _min_d_threshold_: points.append((x1, y1))
                t += t_inc

        # Hack the last three points
        pt_beg   = ((1/6) * (cps[-3][0] + 4*cps[-2][0] + cps[-1][0]), (1/6) * (cps[-3][1] + 4*cps[-2][1] + cps[-1][1]))
        pt_beg_d = ((1/3) * (cps[-3][0] + 2*cps[-2][0]),              (1/3) * (cps[-3][1] + 2*cps[-2][1]))
        pt_mid_b = pt_beg[0] + (pt_beg[0] - pt_beg_d[0]), pt_beg[1] + (pt_beg[1] - pt_beg_d[1])
        pt_end   = cps[-1]
        pt_mid_e = (pt_end[0] + pt_mid_b[0])/2, (pt_end[1] + pt_mid_b[1])/2
        bezier   = self.bezierCurve(pt_beg, pt_mid_b, pt_mid_e, pt_end)
        t = 0.0
        while t < 1.0:
            _xy_ = bezier(t)
            d2   = sqrt((_xy_[0]-points[-1][0])**2 + (_xy_[1]-points[-1][1])**2)
            if d2 >= _min_d_threshold_: points.append(_xy_)
            t += t_inc

        return points

    #
    # svgPathCubicBSpline() - cubic b-spline as an SVG path
    # - should produce the same as above (but more efficient since it uses SVG's built in bezier curves)
    #
    def svgPathCubicBSpline(self, pts, beta=0.8):
        svg_path = []
        # Formula 1 form the Holten Paper - generates a control point
        def cP(i, n, p_0, p_i, p_n_minus_1):
            _fn_ = lambda k: beta * p_i[k] + (1.0 - beta) * (p_0[k] + ((i/(n-1)) * (p_n_minus_1[k] - p_0[k])))
            return (_fn_(0), _fn_(1))

        # Generate all the control points
        i, cps = 0, []
        for i in range(len(pts)):
            xy = cP(i, len(pts), pts[0], pts[i], pts[-1])
            cps.append(xy)

        # Hack the first three points into a bezier curve
        pt_beg = cps[0]
        pt_end   = ((1/6) *   (cps[0][0] + 4*cps[1][0] + cps[2][0]),  (1/6) *   (cps[0][1] + 4*cps[1][1] + cps[2][1]))
        pt_end_d = ((1/3) * (2*cps[1][0] +   cps[2][0]),              (1/3) * (2*cps[1][1] +   cps[2][1]))
        pt_mid_e = (pt_end[0] + (pt_end[0] - pt_end_d[0]), pt_end[1] + (pt_end[1] - pt_end_d[1]))
        pt_mid_b = (pt_beg[0] + pt_mid_e[0])/2, (pt_beg[1] + pt_mid_e[1])/2

        svg_path.append(f'M {pt_beg[0]} {pt_beg[1]}')
        svg_path.append(f'C {pt_mid_b[0]} {pt_mid_b[1]} {pt_mid_e[0]} {pt_mid_e[1]} {pt_end[0]} {pt_end[1]} ')

        # For every four points, use the wikipedia interpolation...
        # - it'd be faster to use the bezier implementation from SVG (see the test file) ... but if you want to colorize it,
        #   there's no implementation within SVG to shade across the curve...
        for i in range(len(cps)-3):
            # Copied from wikipedia page on B-splines -- https://en.wikipedia.org/wiki/B-spline
            # p0 = ( (1/6) * (cps[i][0] + 4*cps[i+1][0] + cps[i+2][0]) ,  (1/6) * (cps[i][1] + 4*cps[i+1][1] + cps[i+2][1]) )
            p1 = ((1/3) * (2*cps[i+1][0] + cps[i+2][0]),               (1/3) * (2*cps[i+1][1] + cps[i+2][1]))
            p2 = ((1/3) * (cps[i+1][0] + 2*cps[i+2][0]),               (1/3) * (cps[i+1][1] + 2*cps[i+2][1]))
            p3 = ((1/6) * (cps[i+1][0] + 4*cps[i+2][0] + cps[i+3][0]), (1/6) * (cps[i+1][1] + 4*cps[i+2][1] + cps[i+3][1]))
            svg_path.append(f'C {p1[0]} {p1[1]} {p2[0]} {p2[1]} {p3[0]} {p3[1]}')

        # Hack the last three points
        pt_beg   = ((1/6) * (cps[-3][0] + 4*cps[-2][0] + cps[-1][0]), (1/6) * (cps[-3][1] + 4*cps[-2][1] + cps[-1][1]))
        pt_beg_d = ((1/3) * (cps[-3][0] + 2*cps[-2][0]),              (1/3) * (cps[-3][1] + 2*cps[-2][1]))
        pt_mid_b = pt_beg[0] + (pt_beg[0] - pt_beg_d[0]), pt_beg[1] + (pt_beg[1] - pt_beg_d[1])
        pt_end   = cps[-1]
        pt_mid_e = (pt_end[0] + pt_mid_b[0])/2, (pt_end[1] + pt_mid_b[1])/2

        svg_path.append(f'C {pt_mid_b[0]} {pt_mid_b[1]} {pt_mid_e[0]} {pt_mid_e[1]} {pt_end[0]} {pt_end[1]}')

        return ' '.join(svg_path)

    #
    # segmentOctTree() - return a segment octree
    # - bounds == (x0,y0,x1,y1)
    # - DONT USE!!!
    #
    def segmentOctTree(self, bounds, max_segments_per_cell=20):
        return SegmentOctTree(self, bounds, max_segments_per_cell=max_segments_per_cell)

    #
    # xyQuadTree() - returns an xy quadtree
    #
    def xyQuadTree(self, bounds, max_pts_per_node=50):
        return XYQuadTree(self, bounds, max_pts_per_node=max_pts_per_node)


#
# XYQuadTree() - implementation of an xy quad tree...
#
class XYQuadTree(object):
    #
    # __init__()
    # - bounds = (xmin,ymin,xmax,ymax)
    #
    def __init__(self, rt_self, bounds, max_pts_per_node=50):
        self.rt_self              = rt_self
        self.bounds               = bounds
        self.max_pts_per_node     = max_pts_per_node
        self.node_pts             = {}
        self.node_pts['']         = set()
        self.node_bounds          = {}
        self.node_bounds['']      = bounds
        self.y_to_node            = {}         # track which nodes have which y bounds
        self.y_to_node[bounds[1]] = set([''])
        self.y_to_node[bounds[3]] = set([''])
        self.x_to_node            = {}         # track which nodes have which x bounds
        self.x_to_node[bounds[0]] = set([''])
        self.x_to_node[bounds[2]] = set([''])

    #
    # __split__() - split a quad... maybe...
    #
    def __split__(self, q):
        b = self.node_bounds[q]
        xs, ys = [], []
        for pt in self.node_pts[q]: xs.append(pt[0]), ys.append(pt[1])
        xs, ys = sorted(xs), sorted(ys)
        mid = int(len(xs)/2)
        if xs[0] == xs[-1] and ys[0] == ys[-1]: return  # not splitable...
        if xs[0] == xs[-1]: xsplit = (b[0] + b[2])/2.0
        else:               xsplit = xs[mid]
        if ys[0] == ys[-1]: ysplit = (b[1] + b[3])/2.0
        else:               ysplit = ys[mid]

        self.node_pts   [q+'0'] = set()
        self.node_bounds[q+'0'] = (b[0],    b[1],     xsplit,  ysplit)
        self.node_pts   [q+'1'] = set()
        self.node_bounds[q+'1'] = (xsplit,  b[1],     b[2],    ysplit)
        self.node_pts   [q+'2'] = set()
        self.node_bounds[q+'2'] = (b[0],    ysplit,   xsplit,  b[3])
        self.node_pts   [q+'3'] = set()
        self.node_bounds[q+'3'] = (xsplit,  ysplit,   b[2],    b[3])
        for node in [q+'0', q+'1', q+'2', q+'3']:
            b = self.node_bounds[node]
            if b[0] not in self.x_to_node.keys():
                self.x_to_node[b[0]] = set()
            if b[2] not in self.x_to_node.keys():
                self.x_to_node[b[2]] = set()
            if b[1] not in self.y_to_node.keys():
                self.y_to_node[b[1]] = set()
            if b[3] not in self.y_to_node.keys():
                self.y_to_node[b[3]] = set()
            self.x_to_node[b[0]].add(node), self.x_to_node[b[2]].add(node)
            self.y_to_node[b[1]].add(node), self.y_to_node[b[3]].add(node)

        for pt in self.node_pts[q]:
            if   pt[0] <= xsplit and pt[1] <= ysplit:
                s = '0'
            elif                     pt[1] <= ysplit:
                s = '1'
            elif pt[0] <= xsplit:
                s = '2'
            else:
                s = '3'
            self.node_pts[q+s].add(pt)

        self.node_pts[q] = set()

    #
    # quad() - find the quad string describing where the point should reside
    #
    def quad(self, pt):
        s = ''
        while (s+'0') in self.node_pts.keys():
            b = self.node_bounds[(s+'0')]
            if   pt[0] <= b[2] and pt[1] <= b[3]:  s += '0'
            elif                   pt[1] <= b[3]:  s += '1'
            elif pt[0] <= b[2]:                    s += '2'
            else:                                  s += '3'
        return s

    #
    # add() - add a list of points to the quadtree
    #
    def add(self, pts):
        for pt in pts:
            q = self.quad(pt)
            self.node_pts[q].add(pt)
            if len(self.node_pts[q]) > self.max_pts_per_node:
                self.__split__(q)

    #
    # __nbors__() - return the neighbors to the specified quad
    # - return set will also include the node passed as a parameter
    # - if q is a set, the set will be iterated over and added as nbors
    #
    def __nbors__STRICT__(self, q):
        if type(q) is set:
            _set_ = set()
            for _q_ in q:
                _set_ = _set_ | set([_q_]) | self.__nbors__(_q_)
            return _set_
        else:
            _set_ = set()
            _set_.add(q)
            b = self.node_bounds[q]
            # to_check = self.x_to_node[b[0]] | self.x_to_node[b[2]] | self.y_to_node[b[1]] | self.y_to_node[b[3]]
            for n in self.node_bounds.keys():
                _b_ = self.node_bounds[n]
                if _b_[0] == b[2] or \
                   _b_[2] == b[0]:
                    if (_b_[1] >=  b [1] and _b_[1] <=  b [3]) or \
                       (_b_[3] >=  b [1] and _b_[3] <=  b [3]) or \
                       (_b_[1] <=  b [1] and _b_[3] >= _b_[3]) or \
                       (b  [1] >= _b_[1] and  b [1] <= _b_[3]) or \
                       (b  [3] >= _b_[1] and  b [3] <= _b_[3]) or \
                       (b  [1] <= _b_[1] and  b [3] >= _b_[3]):
                        _set_.add(n)
                if _b_[1] == b[3] or \
                   _b_[3] == b[1]:
                    if (_b_[0] >=  b [0] and _b_[0] <=  b [2]) or \
                       (_b_[2] >=  b [0] and _b_[2] <=  b [2]) or \
                       (_b_[0] <=  b [0] and _b_[2] >= _b_[2]) or \
                       (b  [0] >= _b_[0] and  b [0] <= _b_[2]) or \
                       (b  [2] >= _b_[0] and  b [2] <= _b_[2]) or \
                       (b  [0] <= _b_[0] and  b [2] >= _b_[2]):
                        _set_.add(n)
            return _set_

    #
    # __nbors__() - return the neighbors to the specified quad
    # - return set will also include the node passed as a parameter
    # - if q is a set, the set will be iterated over and added as nbors
    #
    def __nbors__(self, q):
        if type(q) is set:
            _set_ = set()
            for _q_ in q:
                _set_ = _set_ | set([_q_]) | self.__nbors__(_q_)
            return _set_
        else:
            _set_ = set()
            _set_.add(q)
            b = self.node_bounds[q]
            ex, ey = (b[2]-b[0])*0.1, (b[3]-b[1])*0.1  # within 10 percent...
            for n in self.node_bounds.keys():
                _b_ = self.node_bounds[n]
                if _b_[0] == b[2] or \
                   _b_[2] == b[0]:
                    if (_b_[1] >=  (b [1]-ey) and _b_[1] <=  (b [3]+ey)) or \
                       (_b_[3] >=  (b [1]-ey) and _b_[3] <=  (b [3]+ey)) or \
                       (_b_[1] <=  (b [1])    and _b_[3] >= (_b_[3]))    or \
                       (b  [1] >= (_b_[1]-ey) and  b [1] <= (_b_[3]+ey)) or \
                       (b  [3] >= (_b_[1]-ey) and  b [3] <= (_b_[3]+ey)) or \
                       (b  [1] <= (_b_[1])    and  b [3] >=  _b_[3]):
                        _set_.add(n)
                if _b_[1] == b[3] or \
                   _b_[3] == b[1]:
                    if (_b_[0] >=  (b [0]-ex) and _b_[0] <=  (b [2]+ex)) or \
                       (_b_[2] >=  (b [0]-ex) and _b_[2] <=  (b [2]+ex)) or \
                       (_b_[0] <=   b [0]     and _b_[2] >=  _b_[2])     or \
                       (b  [0] >= (_b_[0]-ex) and  b [0] <= (_b_[2]+ex)) or \
                       (b  [2] >= (_b_[0]-ex) and  b [2] <= (_b_[2]+ex)) or \
                       (b  [0] <=  _b_[0]     and  b [2] >=  _b_[2]):
                        _set_.add(n)
            return _set_

    #
    # closest() - return the closest points
    #
    def closest(self, pt, n=10):
        if n >= self.max_pts_per_node/4:
            raise Exception(f'QuadTree.closest() - {n} shouldn\'t be larger than (max_pts_per_node/4) == {self.max_pts_per_node/4}')
        q       = self.quad(pt)
        nodes   = self.__nbors__(set([q]))
        if q != '':
            for x in ['0', '1', '2', '3']:
                nodes.add(q[:-1]+x)
        sorter  = []
        for node in nodes:
            for _pt_ in self.node_pts[node]:
                sorter.append((self.rt_self.segmentLength((pt, _pt_)), _pt_))
        return sorted(sorter)[:n]

    #
    # _repr_svg_()
    #
    def _repr_svg_(self):
        w     = h     = 600
        x_ins = y_ins = 5
        svg = []
        svg.append(f'<svg x="0" y="0" width="{w+2*x_ins}" height="{h+2*y_ins}">')
        svg.append(f'<rect x="0" y="0" width="{w+2*x_ins}" height="{h+2*y_ins}" fill="#ffffff" />')
        xT = lambda x: x_ins + w * (x - self.bounds[0])/(self.bounds[2] - self.bounds[0])
        yT = lambda y: y_ins + h * (y - self.bounds[1])/(self.bounds[3] - self.bounds[1])
        for q in self.node_bounds.keys():
            if q+'0' not in self.node_bounds.keys():
                b = self.node_bounds[q]
                _color_ = self.rt_self.co_mgr.getColor(q)
                svg.append(f'<rect x="{xT(b[0])}" y="{yT(b[1])}" width="{xT(b[2])-xT(b[0])}" height="{yT(b[3])-yT(b[1])}" fill="{_color_}" opacity="0.4" />')
        for q in self.node_pts.keys():
            for pt in self.node_pts[q]:
                svg.append(f'<circle cx="{xT(pt[0])}" cy="{yT(pt[1])}" r="0.8" fill="#000000" />')
        svg.append('</svg>')
        return ''.join(svg)


#
# SegmentOctTree -- oct tree implementation for faster segment discovery.
# - splits based on the median values in both x and y
# ... this is probably actually a "QuadTree" :(
#
class SegmentOctTree(object):
    #
    # bounds == (x0,y0,x1,y1)
    #
    def __init__(self, rt_self, bounds, max_segments_per_cell=20):
        self.rt_self                = rt_self
        self.bounds                 = bounds
        self.max_segments_per_cell  = max_segments_per_cell
        self.tree                   = {}
        self.tree_bounds            = {}
        self.tree['']               = set()
        self.tree_bounds['']        = self.bounds
        self.node_already_split     = {}
        self.node_already_split[''] = False
        # self.bad_svgs               = []

        if bounds[0] >= bounds[2]: raise Exception(f'SegmentOctTree.__init__() - [1] bounds messed up {bounds}')
        if bounds[1] >= bounds[3]: raise Exception(f'SegmentOctTree.__init__() - [2] bounds messed up {bounds}')

        # For Debugging...
        self.pieces = set()  # for debugging...

    #
    # findOctet() - find the placement of a point within the octtree
    # - check_parents -- parents should all be empty...
    #
    def findOctet(self, pt, check_parents=False):
        s = ''
        while (s+'0') in self.tree.keys():
            if check_parents and len(self.tree[s]) > 0:
                raise Exception(f'SegmentOctTree.findOctet("{pt}") has children... but node "{s}" not empty (len={len(self.tree[s])})')
            b = self.tree_bounds[s+'0']
            if   pt[0] <= b[2] and pt[1] <= b[3]: s += '0'
            elif pt[0] <= b[2]:                   s += '2'
            elif                   pt[1] <= b[3]: s += '1'
            else:                                 s += '3'
        return s

    #
    # __split__() - split a tree node into four parts ... not thread safe
    # def lineSegmentIntersectionPoint(self, line, segment):
    #
    def __split__(self, node):
        if self.node_already_split[node]:  # shouldn't have anything in it...
            if len(self.tree[node]) != 0:
                raise Exception(f'SegmentOcttree.__split__(node="{node}") shouldn\'t be non-zero')
            return

        # determine the split points -- median of the coordinates in x and y
        b = self.tree_bounds[node]
        xs, ys = [], []
        for piece in self.tree[node]:
            valid = True
            x0, y0, x1, y1 = piece[0][0], piece[0][1], piece[1][0], piece[1][1]
            # clip segment first...
            if (x0 <= b[0] and x1 <= b[0]) or (x0 >= b[2] and x1 >= b[2]):
                valid = False
            else:
                if x0 < b[0]:
                    pt = self.rt_self.lineSegmentIntersectionPoint(((b[0], 0), (b[0], 1)), piece)
                    if pt is None: valid  = False
                    else:          x0, y0 = pt
                if x0 > b[2]:
                    pt = self.rt_self.lineSegmentIntersectionPoint(((b[2], 0), (b[2], 1)), piece)
                    if pt is None: valid  = False
                    else:          x0, y0 = pt
                if x1 < b[0]:
                    pt = self.rt_self.lineSegmentIntersectionPoint(((b[0], 0), (b[0], 1)), piece)
                    if pt is None: valid  = False
                    else:          x1, y1 = pt
                if x1 > b[2]:
                    pt = self.rt_self.lineSegmentIntersectionPoint(((b[2], 0), (b[2], 1)), piece)
                    if pt is None: valid  = False
                    else:          x1, y1 = pt
            if (y0 <= b[1] and y1 <= b[1]) or (y0 >= b[3] and y1 >= b[3]): valid = False
            else:
                if y0 < b[1]:
                    pt = self.rt_self.lineSegmentIntersectionPoint(((0, b[1]), (1, b[1])), piece)
                    if pt is None: valid  = False
                    else:          x0, y0 = pt
                if y0 > b[3]:
                    pt = self.rt_self.lineSegmentIntersectionPoint(((0, b[3]), (1, b[3])), piece)
                    if pt is None: valid  = False
                    else:          x0, y0 = pt
                if y1 < b[1]:
                    pt = self.rt_self.lineSegmentIntersectionPoint(((0, b[1]), (1, b[1])), piece)
                    if pt is None: valid  = False
                    else:          x1, y1 = pt
                if y1 > b[3]:
                    pt = self.rt_self.lineSegmentIntersectionPoint(((0, b[3]), (1, b[3])), piece)
                    if pt is None: valid  = False
                    else:          x1, y1 = pt
            if valid:
                xs.append(x0), xs.append(x1), ys.append(y0), ys.append(y1)
        xs, ys = sorted(xs), sorted(ys)

        if len(xs) == 0 or len(ys) == 0:  # nothing if we don't have values to sort
            return  # prevent bad cases
        if xs[0] == xs[-1] or ys[0] == ys[-1]:  # nothing if the values are all the same
            return  # prevent bad cases

        x_split = xs[int(len(xs)/2)]
        y_split = ys[int(len(ys)/2)]

        if x_split == b[0] or x_split == b[2] or y_split == b[1] or y_split == b[3]:  # nothing if the split equals a current boundary
            return  # prevent bad cases

        if abs(x_split - b[0]) < 1.0 or abs(x_split - b[2]) < 1.0 or abs(y_split - b[1]) < 1.0 or abs(y_split - b[3]) < 1.0:  # nothing less than one...
            return  # prevent bad cases

        self.node_already_split[node] = True  # marks that the node has already been split and shouldn't be split again...

        self.tree       [node+'0'] = set()
        self.tree_bounds[node+'0'] = (b[0],     b[1],    x_split, y_split)
        self.node_already_split[node+'0'] = False

        self.tree       [node+'1'] = set()
        self.tree_bounds[node+'1'] = (x_split,  b[1],    b[2],    y_split)
        self.node_already_split[node+'1'] = False

        self.tree       [node+'2'] = set()
        self.tree_bounds[node+'2'] = (b[0],     y_split, x_split, b[3])
        self.node_already_split[node+'2'] = False

        self.tree       [node+'3'] = set()
        self.tree_bounds[node+'3'] = (x_split,  y_split, b[2],    b[3])
        self.node_already_split[node+'3'] = False

        to_check =      [node+'0', node+'1', node+'2', node+'3']
        for piece in self.tree[node]:
            # x_min, y_min, x_max, y_max = min(piece[0][0], piece[1][0]), min(piece[0][1], piece[1][1]), max(piece[0][0], piece[1][0]), max(piece[0][1], piece[1][1])
            # oct0, oct1 = self.findOctet(piece[0], check_parents=False), self.findOctet(piece[1], check_parents=False)
            piece_addition_count = 0
            for k in to_check:
                if self.__segmentTouchesNode__(piece, k):
                    self.tree[k].add(piece)
                    piece_addition_count += 1
            if piece_addition_count == 0:
                pass  # ... not necessarily a problem -- happens when the segment touches the exact boundary only
                #       ... should probably keep track to ensure all segments are represented in at least one place instead...
                # print(f"Error -- No additions for piece {piece} ... node = '{node}' | node_sz = {len(self.tree[k])}")
                # svg = '<svg width="500" height="500">\n'
                # svg += f'<rect x="0" y="0" width="500" height="500" fill="#ffffff"/>\n'
                # for k in to_check:
                #     b = self.tree_bounds[k]
                #     svg += f'<rect x="{b[0]}" y="{b[1]}" width="{b[2]-b[0]}" height="{b[3]-b[1]}" fill="none" stroke="#000000"/>\n'
                # svg += f'<line x1="{piece[0][0]}" y1="{piece[0][1]}" x2="{piece[1][0]}" y2="{piece[1][1]}" stroke="#ff0000"/>\n'
                # svg += "</svg>"
                # self.bad_svgs.append(svg)
        self.tree[node] = set()
        for k in to_check:
            if len(self.tree[k]) > self.max_segments_per_cell:
                self.__split__(k)

    #
    # addSegments() -- add segments to the tree
    # - segments = [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
    def addSegments(self, segments):
        for i in range(len(segments)-1):
            piece = ((segments[i][0], segments[i][1]), (segments[i+1][0], segments[i+1][1]))  # make sure it's a tuple
            self.pieces.add(piece)
            oct0   = self.findOctet(segments[i])
            x0, y0 = segments[i]
            oct1   = self.findOctet(segments[i+1])
            x1, y1 = segments[i+1]
            # x_min, y_min, x_max, y_max = min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
            if oct0 == oct1:
                self.tree[oct0].add(piece)
                if len(self.tree[oct0]) > self.max_segments_per_cell:
                    self.__split__(oct0)
            else:
                to_split = set()  # to avoid messing with the keys in this iteration
                for k in self.tree.keys():
                    if self.__segmentTouchesNode__(piece, k):
                        self.tree[k].add(piece)
                        if len(self.tree) > self.max_segments_per_cell:
                            to_split.add(k)
                for k in to_split:
                    self.__split__(k)

    #
    # closestSegmentToPoint() - find the closest segment to the specified point.
    # - pt = (x,y)
    # - returns distance, segment,              segment_pt
    #           10.0,     ((x0,y0),(x1,y1))     (x3,y3)
    def closestSegmentToPoint(self, pt):
        oct       = self.findOctet(pt)
        oct_nbors = self.__neighbors__(oct) | set([oct])

        closest_d, closest_segment, closest_xy = None, None, None
        for node in oct_nbors:
            for segment in self.tree[node]:
                d, xy = self.rt_self.closestPointOnSegment(segment, pt)
                if closest_d is None:
                    closest_d, closest_segment, closest_xy = d, segment, xy
                elif d < closest_d:
                    closest_d, closest_segment, closest_xy = d, segment, xy

        return closest_d, closest_segment, closest_xy

    #
    # closestSegmentsToPoint() - find the closest segments to the specifid point.
    # ... may not be able to guarantee these are the closest...
    # ... return list of the following tuples:  (distance, segment, xy) where xy is the closest part of the segment to the point.
    #
    def closestSegmentsToPoint(self, pt, n=10):
        n = min(n, len(self.pieces))  # can't find more pieces than are in the data structure...
        nbors     = set([self.findOctet(pt)])
        sorter    = []
        checked   = set()  # nodes that have been checked already
        while len(sorter) < n:
            # expand to all neighbors from the current region
            new_nbors = set()
            for node in nbors:
                new_nbors |= self.__neighbors__(node)
            nbors = new_nbors | nbors
            # check those nodes for segments
            for node in nbors:
                if node in checked:
                    continue
                checked.add(node)
                for segment in self.tree[node]:
                    if segment in checked:
                        continue
                    checked.add(segment)
                    d, xy = self.rt_self.closestPointOnSegment(segment, pt)
                    sorter.append((d, segment, xy))
        return sorted(sorter)[:n]

    #
    # __segmentTouchesNode__() - does a segment intersect (or belong in) a node?
    # ... changes here should be propagated to the _reason_ version...
    #
    def __segmentTouchesNode__(self, segment, node):
        if self.node_already_split[node]:
            return False
        if node == self.findOctet(segment[0]) or node == self.findOctet(segment[1]):
            return True
        x_min, y_min = min(segment[0][0], segment[1][0]), min(segment[0][1], segment[1][1])
        x_max, y_max = max(segment[0][0], segment[1][0]), max(segment[0][1], segment[1][1])
        b            = self.tree_bounds[node]
        if   x_max < b[0] or x_min > b[2] or y_max < b[1] or y_min > b[3]:
            return False
        else:
            flag0, x, y, ts0, ts1 = self.rt_self.segmentsIntersect(segment, ((b[0], b[1]), (b[0], b[3])))
            if flag0: return True
            else:
                flag1, x, y, ts0, ts1 = self.rt_self.segmentsIntersect(segment, ((b[0], b[1]), (b[2], b[1])))
                if flag1: return True
                else:
                    flag2, x, y, ts0, ts1 = self.rt_self.segmentsIntersect(segment, ((b[2], b[3]), (b[0], b[3])))
                    if flag2: return True
                    else:
                        flag3, x, y, ts0, ts1 = self.rt_self.segmentsIntersect(segment, ((b[2], b[3]), (b[2], b[1])))
                        if flag3: return True
        return False

    #
    # __segmentTouchesNode_reason_() - does a segment intersect (or belong in) a node?  gives the reason -- copy of the above...
    #
    def __segmentTouchesNode_reason__(self, segment, node):
        if self.node_already_split[node]:
            return 'false - node already split'
        if node == self.findOctet(segment[0]) or node == self.findOctet(segment[1]):
            return 'true - one endpoint originates in the specified node'
        x_min, y_min = min(segment[0][0], segment[1][0]), min(segment[0][1], segment[1][1])
        x_max, y_max = max(segment[0][0], segment[1][0]), max(segment[0][1], segment[1][1])
        b            = self.tree_bounds[node]
        if   x_max < b[0] or x_min > b[2] or y_max < b[1] or y_min > b[3]:
            return 'false - bounds test failed'
        else:
            flag0, x, y, ts0, ts1 = self.rt_self.segmentsIntersect(segment, ((b[0], b[1]), (b[0], b[3])))
            if flag0: return f'flag0 ... (({b[0]},{b[1]}),({b[0]},{b[3]}))'
            else:
                flag1, x, y, ts0, ts1 = self.rt_self.segmentsIntersect(segment, ((b[0], b[1]), (b[2], b[1])))
                if flag1: return 'flag1 ... (({b[0]},{b[1]}),({b[2]},{b[1]}))'
                else:
                    flag2, x, y, ts0, ts1 = self.rt_self.segmentsIntersect(segment, ((b[2], b[3]), (b[0], b[3])))
                    if flag2: return 'flag2 ... (({b[2]},{b[3]}),({b[0]},{b[3]}))'
                    else:
                        flag3, x, y, ts0, ts1 = self.rt_self.segmentsIntersect(segment, ((b[2], b[3]), (b[2], b[1])))
                        if flag3: return 'flag3 ... (({b[2]},{b[3]}),({b[2]},{b[1]}))'
        return False

    #
    # closestSegment() - return the closest segment to the specified segment.
    # - _segment_ = ((x0,y0),(x1,y1))
    # - returns distance, other_segment
    #
    # ... i don't really think this will return the absolute closest segment :(
    #
    def closestSegment(self, segment):
        # Figure out which tree leaves to check
        oct0       = self.findOctet(segment[0])
        oct0_nbors = self.__neighbors__(oct0)
        oct1       = self.findOctet(segment[1])
        to_check   = set([oct0, oct1])
        if    oct0 == oct1:
            to_check |= oct0_nbors
        elif  oct1 in oct0_nbors:
            to_check |= oct0_nbors | self.__neighbors__(oct1)
        else:  # :( ... have to search for all possibles...
            for k in self.tree_bounds.keys():
                if self.__segmentTouchesNode__(segment, k):
                    to_check.add(k)
            all_nbors = set()
            for node in to_check:
                all_nbors |= self.__neighbors__(node)
            to_check |= all_nbors

        # Find the closest...
        nodes_checked = set()
        closest_d = closest_segment = None
        for node in to_check:
            nodes_checked.add(node)
            for other_segment in self.tree[node]:
                d = self.__segmentDistance__(segment, other_segment)
                if closest_d is None:
                    closest_d, closest_segment = d, other_segment
                elif d < closest_d:
                    closest_d, closest_segment = d, other_segment

        # Return the results
        return closest_d, closest_segment

    # __segmentDistance__() ... probably biased towards human scale numbers... 0 to 1000
    # ... this is a really bad way to do this... for example, take two segments that intersect like a plus...
    # ... those would have intersected but have a far distance...
    def __segmentDistance__(self, _s0_, _s1_):
        d0 = self.rt_self.segmentLength((_s0_[0], _s1_[0]))
        v0 = self.rt_self.unitVector(_s0_)
        d1 = self.rt_self.segmentLength((_s0_[1], _s1_[1]))
        v1 = self.rt_self.unitVector(_s1_)
        return d0 + d1 + abs(v0[0]*v1[0]+v0[1]*v1[1])

    # __neighbors__() ... return the neighbors of a node...
    def __neighbors__(self, node):
        _set_ = set()
        if node == '':
            return _set_
        node_b = self.tree_bounds[node]
        for k in self.tree_bounds:
            if self.node_already_split[k]:  # don't bother with split nodes
                continue
            b = self.tree_bounds[k]
            right, left  = (b[0] == node_b[2]), (b[2] == node_b[0])
            above, below = (b[3] == node_b[1]), (b[1] == node_b[3])
            # diagonals:
            if (right and above) or (right and below) or (left and above) or (left and below):
                _set_.add(k)
            elif right or left:
                if (b[1] >= node_b[1] and b[1] <= node_b[3]) or \
                   (b[3] >= node_b[1] and b[3] <= node_b[3]) or \
                   (node_b[1] >= b[1] and node_b[1] <= b[3]) or \
                   (node_b[3] >= b[1] and node_b[3] <= b[3]):
                    _set_.add(k)
            elif above or below:
                if (b[0] >= node_b[0] and b[0] <= node_b[2]) or \
                   (b[2] >= node_b[0] and b[2] <= node_b[2]) or \
                   (node_b[0] >= b[0] and node_b[0] <= b[2]) or \
                   (node_b[2] >= b[0] and node_b[2] <= b[2]):
                    _set_.add(k)
        return _set_

    #
    # _repr_svg_() - return an SVG version of the oct tree
    #
    def _repr_svg_(self):
        w,  h, x_ins, y_ins = 800, 800, 50, 50
        xa, ya, xb, yb      = self.tree_bounds['']
        xT = lambda x: x_ins + w*(x - xa)/(xb-xa)
        yT = lambda y: y_ins + h*(y - ya)/(yb-ya)
        svg =  f'<svg x="0" y="0" width="{w+2*x_ins}" height="{h+2*y_ins}" xmlns="http://www.w3.org/2000/svg">'
        all_segments = set()
        for k in self.tree:
            all_segments = all_segments | self.tree[k]
            b = self.tree_bounds[k]
            _color_ = self.rt_self.co_mgr.getColor(k)
            svg += f'<rect x="{xT(b[0])}" y="{yT(b[1])}" width="{xT(b[2])-xT(b[0])}" height="{yT(b[3])-yT(b[1])}" fill="{_color_}" opacity="0.4" stroke="{_color_}" stroke-width="0.5" stroke-opacity="1.0" />'
            svg += f'<text x="{xT(b[0])+2}" y="{yT(b[3])-2}" font-size="10px">{k}</text>'
        for segment in self.pieces:
            svg += f'<line x1="{xT(segment[0][0])}" y1="{yT(segment[0][1])}" x2="{xT(segment[1][0])}" y2="{yT(segment[1][1])}" stroke="#ffffff" stroke-width="4.0" />'
            nx,  ny  = self.rt_self.unitVector(segment)
            pnx, pny = -ny, nx
            svg += f'<line x1="{xT(segment[0][0]) + pnx*3}" y1="{yT(segment[0][1]) + pny*3}" x2="{xT(segment[0][0]) - pnx*3}" y2="{yT(segment[0][1]) - pny*3}" stroke="#000000" stroke-width="0.5" />'
            svg += f'<line x1="{xT(segment[1][0]) + pnx*3}" y1="{yT(segment[1][1]) + pny*3}" x2="{xT(segment[1][0]) - pnx*3}" y2="{yT(segment[1][1]) - pny*3}" stroke="#000000" stroke-width="0.5" />'
        for segment in all_segments:
            svg += f'<line x1="{xT(segment[0][0])}" y1="{yT(segment[0][1])}" x2="{xT(segment[1][0])}" y2="{yT(segment[1][1])}" stroke="#ff0000" stroke-width="2.0" />'

        # Draw example neighbors
        _as_list_ = list(self.tree.keys())
        _node_    = _as_list_[random.randint(0, len(_as_list_)-1)]
        while self.node_already_split[_node_]:  # find a non-split node...
            _node_    = _as_list_[random.randint(0, len(_as_list_)-1)]
        _node_b_  = self.tree_bounds[_node_]
        xc, yc    = (_node_b_[0]+_node_b_[2])/2.0, (_node_b_[1]+_node_b_[3])/2.0
        _nbors_   = self.__neighbors__(_node_)
        for _nbor_ in _nbors_:
            _nbor_b_ = self.tree_bounds[_nbor_]
            xcn, ycn = (_nbor_b_[0]+_nbor_b_[2])/2.0, (_nbor_b_[1]+_nbor_b_[3])/2.0
            svg += f'<line x1="{xT(xc)}" y1="{yT(yc)}" x2="{xT(xcn)}" y2="{yT(ycn)}" stroke="#000000" stroke-width="0.5" />'
        svg +=  '</svg>'
        return svg

    #
    # renderNode() - render single node and its contents.
    #
    def renderNode(self, node, w=800, h=800, x_ins=5, y_ins=5):
        xa, ya, xb, yb      = self.tree_bounds['']
        xT = lambda x: x_ins + w*(x - xa)/(xb-xa)
        yT = lambda y: y_ins + h*(y - ya)/(yb-ya)
        svg =  f'<svg x="0" y="0" width="{w+2*x_ins}" height="{h+2*y_ins}" xmlns="http://www.w3.org/2000/svg">'
        svg += f'<rect x="0" y="0" width="{w+2*x_ins}" height="{h+2*y_ins}" fill="#ffffff" />'
        for k in self.tree:
            b = self.tree_bounds[k]
            _color_ = '#c0c0c0'
            svg += f'<rect x="{xT(b[0])}" y="{yT(b[1])}" width="{xT(b[2])-xT(b[0])}" height="{yT(b[3])-yT(b[1])}" fill="none" stroke="{_color_}" stroke-width="0.5" stroke-opacity="1.0" />'

        b = self.tree_bounds[node]
        _color_ = self.rt_self.co_mgr.getColor(node)
        svg += f'<rect x="{xT(b[0])}" y="{yT(b[1])}" width="{xT(b[2])-xT(b[0])}" height="{yT(b[3])-yT(b[1])}" fill="none" stroke="{_color_}" stroke-width="3" stroke-opacity="1.0" />'
        svg += f'<text x="{xT(b[0])+2}" y="{yT(b[3])-2}" font-size="10px">{node}</text>'

        for segment in self.tree[node]:
            if self.__segmentTouchesNode__(segment, node):
                _color_ = '#000000'
                _width_ = 1.0
            else:
                _color_ = '#ff0000'
                _width_ = 3.0
            svg += f'<line x1="{xT(segment[0][0])}" y1="{yT(segment[0][1])}" x2="{xT(segment[1][0])}" y2="{yT(segment[1][1])}" stroke="{_color_}" stroke-width="{_width_}" />'

        svg +=  '</svg>'
        return svg
