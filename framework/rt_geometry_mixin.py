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

from shapely.geometry              import Point, Polygon, LineString, GeometryCollection, MultiLineString
from shapely.geometry.multipolygon import MultiPolygon
from math import sqrt,acos
import random

import heapq

__name__ = 'rt_geometry_mixin'

#
# Geometry Methods
#
class RTGeometryMixin(object):
    #
    # segmentLength()
    # - _segment_ = [(x0,y0),(x1,y1)]
    #
    def segmentLength(self, _segment_):
        dx, dy = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
        return sqrt(dx*dx+dy*dy)

    #
    # unitVector()
    # - _segment_ = [(x0,y0),(x1,y1)]
    #
    def unitVector(self, _segment_):
        dx, dy = _segment_[1][0] - _segment_[0][0], _segment_[1][1] - _segment_[0][1]
        _len_  = sqrt(dx*dx+dy*dy)
        if _len_ < 0.0001:
            _len_ = 1.0
        return (dx/_len_, dy/_len_)

    #
    # bezierCurve() - parametric bezier curve object
    # - from Bezier Curve on wikipedia.org
    #
    def bezierCurve(self, pt1, pt1p, pt2p, pt2):
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
    def closestPointOnSegment(self, _segment_, _pt_):
        if _segment_[0][0] == _segment_[1][0] and _segment_[0][1] == _segment_[1][1]: # not a segment...
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
    def intersectionPoint(self, line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])
        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]
        div = det(xdiff, ydiff)
        if abs(div) < 0.0001 or div == 0:
            return None
        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    #
    # lineSegmentIntersectionPoint() - determine where a line intersects a segment
    # - returns None if the line does not intersect the segment
    #
    def lineSegmentIntersectionPoint(self, line, segment):
        # Would they intersect if they were both lines?
        results = self.intersectionPoint(line, segment)
        if results is None:
            return None
        # They intersect as lines... are the results on the segment?
        x,y = results
        if x < min(segment[0][0], segment[1][0]) or x > max(segment[0][0], segment[1][0]):
            return None
        if y < min(segment[0][1], segment[1][1]) or y > max(segment[0][1], segment[1][1]):
            return None
        return x,y

    #
    # pointWithinSegment()
    #
    def pointWithinSegment(self, x, y, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        _xmin,_xmax = min(x0,x1),max(x0,x1)
        _ymin,_ymax = min(y0,y1),max(y0,y1)
        if x < _xmin or x > _xmax or y < _ymin or y > _ymax:
            return False, 0.0
        # xp = x0 + t * dx
        # yp = y0 + t * dy
        if dx == 0 and dy == 0: # segment is a point...
            if x == x0 and y == y0:
                return True, 0.0
        elif dx == 0: # it's vertical...
            t  = (y - y0)/dy
            xp,yp = x0 + t*dx, y0 + t*dy
            if xp == x and yp == y and t >= 0.0 and t <= 1.0:
                return True, t
        else: # it's horizontal... or at least conforms to f(x)
            t  = (x - x0)/dx
            xp,yp = x0 + t*dx, y0 + t*dy
            if xp == x and yp == y and t >= 0.0 and t <= 1.0:
                return True, t
        return False, 0.0

    #
    # pointOnLine()
    #
    def pointOnLine(self, x, y, x0, y0, x1, y1):
        dx, dy = x1 - x0, y1 - y0
        # xp = x0 + t * dx
        # yp = y0 + t * dy
        if dx == 0 and dy == 0: # segment is a point...
            if x == x0 and y == y0:
                return True, 0.0
        elif dx == 0: # it's vertical...
            t  = (y - y0)/dy
            xp,yp = x0 + t*dx, y0 + t*dy
            if xp == x and yp == y:
                return True, t
        else: # it's horizontal... or at least conforms to f(x)
            t  = (x - x0)/dx
            xp,yp = x0 + t*dx, y0 + t*dy
            if xp == x and yp == y:
                return True, t
        return False, 0.0

    #
    # segmentsIntersect()
    # - do two segments intersect?
    #
    def segmentsIntersect(self, s0, s1):
        x0,y0,x1,y1 = s0[0][0],s0[0][1],s0[1][0],s0[1][1]
        x2,y2,x3,y3 = s1[0][0],s1[0][1],s1[1][0],s1[1][1]
        _xmin,_ymin,_amin,_bmin = min(x0,x1),min(y0,y1),min(x2,x3),min(y2,y3)
        _xmax,_ymax,_amax,_bmax = max(x0,x1),max(y0,y1),max(x2,x3),max(y2,y3)

        # Simple overlapping bounds test... as inexpensive as it gets...
        if _xmin > _amax or _amin > _xmax or _ymin > _bmax or _bmin > _ymax:
            return False, 0.0, 0.0, 0.0, 0.0
        # Both segments are points... Are they the same point?
        if _xmin == _xmax and _ymin == _ymax and _amin == _amax and _bmin == _bmax:
            if x0 == x2 and y0 == y2:
                return True, x0, y0, 0.0, 0.0
            return False,0.0,0.0,0.0,0.0

        A,B,C,D = y3 - y2, x3 - x2, x1 - x0, y1 - y0

        #x = x0 + t * C
        #t = (x - x0) / C
        #y = y0 + t * D
        #t = (y - y0) / D

        # Deal with parallel lines
        denom = B * D - A * C                # Cross Product
        if denom == 0.0:                     # Parallel...  and if co-linear, overlap because of the previous bounds test...
            online0, t0 = self.pointOnLine(x2, y2, x0, y0, x1, y1)
            online1, t1 = self.pointOnLine(x0, y0, x2, y2, x3, y3)
            if online0 or online1:
                onseg, t = self.pointWithinSegment(x0, y0, x2, y2, x3, y3)
                if onseg:
                    return True, x0, y0, 0.0, t
                onseg, t = self.pointWithinSegment(x1, y1, x2, y2, x3, y3)
                if onseg:
                    return True, x1, y1, 1.0, t
                onseg, t = self.pointWithinSegment(x2, y2, x0, y0, x1, y1)
                if onseg:
                    return True, x2, y2, t, 0.0
                onseg, t = self.pointWithinSegment(x3, y3, x0, y0, x1, y1)
                if onseg:
                    return True, x3, y3, t, 1.0

        # Normal calculation...
        t0 = (A*(x0 - x2) - B*(y0 - y2))/denom
        if t0 >= 0.0 and t0 <= 1.0:
            x    = x0 + t0 * (x1 - x0)
            y    = y0 + t0 * (y1 - y0)
            if (x3 -x2) != 0:
                t1   = (x - x2)/(x3 - x2)
                if t1 >= 0.0 and t1 <= 1.0:
                    return True, x, y, t0, t1
            if (y3 - y2) != 0:
                t1   = (y - y2)/(y3 - y2)
                if t1 >= 0.0 and  t1 <= 1.0:
                    return True, x, y, t0, t1
        return False, 0.0, 0.0, 0.0, 0.0

    #
    # segmentIntersectsCircle() - does a line segment intersect a circle
    # - segment is ((x0,y0),(x1,y1))
    # - circle  is (cx, cy, r)
    # - modification from the following:
    # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm
    #
    def segmentIntersectsCircle(self, segment, circle):
        A, B, C = segment[0], segment[1], (circle[0], circle[1])
        sub  = lambda a, b: (a[0] - b[0], a[1] - b[1])
        AC, AB = sub(C,A), sub(B,A)
        add  = lambda a, b: (a[0] + b[0], a[1] + b[1])
        dot  = lambda a, b: a[0]*b[0] + a[1]*b[1]
        def proj(a,b):
            k = dot(a,b)/dot(b,b)
            return (k*b[0],k*b[1])
        D = add(proj(AC, AB), A)
        AD = sub(D,A)
        if abs(AB[0]) > abs(AB[1]):
            k = AD[0] / AB[0]
        else:
            k = AD[1] / AB[1]
        hypot2 = lambda a, b: dot(sub(a,b),sub(a,b))
        if k <= 0.0:
            return sqrt(hypot2(C,A)), C
        elif k >= 1.0:
            return sqrt(hypot2(C,B)), B
        else:
            return sqrt(hypot2(C,D)), D

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
                if naming is not None: # if naming function, call it with gpd series
                    _name_ = naming(_series_, i)
                bg_shape_lu[_name_] = d
                if type(_poly_) == LineString or type(_poly_) == MultiLineString:
                    bg_fill_lu[_name_] = None
                else:
                    bg_fill_lu[_name_] = fill
        return bg_shape_lu, bg_fill_lu

    #
    # Converts a shapely polygon to an SVG path...
    # ... assumes that the ordering (CW, CCW) of both the exterior and interior points is correct...
    # - if there's no shape in _poly, will return None
    #
    def shapelyPolygonToSVGPathDescription(self, _poly):
        #
        # MultiPolygon -- just break into individual polygons...
        #
        if type(_poly) == MultiPolygon:
            path_str = ''
            for _subpoly in _poly.geoms:
                if len(path_str) > 0:
                    path_str += ' '
                path_str += self.shapelyPolygonToSVGPathDescription(_subpoly)
            return path_str
        #
        # LineString -- segments
        #
        elif type(_poly) == LineString:
            coords = _poly.coords
            path_str = f'M {coords[0][0]} {coords[0][1]} '
            for i in range(1,len(coords)):
                path_str += f'L {coords[i][0]} {coords[i][1]} '
            return path_str
        #
        # Multiple LineStrings -- break into individual line strings
        #
        elif type(_poly) == MultiLineString:
            path_str = ''
            for _subline in _poly.geoms:
                if len(path_str) > 0:
                    path_str += ' '
                path_str += self.shapelyPolygonToSVGPathDescription(_subline)
            return path_str
        #
        # Polygon -- base polygon processing
        #
        elif type(_poly) == Polygon:
            # Draw the exterior shape first
            xx, yy = _poly.exterior.coords.xy
            path_str = f'M {xx[0]} {yy[0]} '
            for i in range(1,len(xx)):
                path_str += f' L {xx[i]} {yy[i]}'
            # Determine if the interior exists... and then render that...
            interior_len = len(list(_poly.interiors))
            if interior_len > 0:
                for interior_i in range(0, interior_len):
                    xx,yy = _poly.interiors[interior_i].coords.xy
                    path_str += f' M {xx[0]} {yy[0]}'
                    for i in range(1,len(xx)):
                        path_str += f' L {xx[i]} {yy[i]}'
            return path_str + ' Z'
        #
        # GeometryCollection -- unsure of what this actual is...
        #
        elif type(_poly) == GeometryCollection:
            if len(_poly.geoms) > 0: # Haven't seen this... so unsure of how to process
                raise Exception('shapelyPolygonToSVGPathDescription() - geometrycollection not empty')    
            return None
        else:
            raise Exception('shapelyPolygonToSVGPathDescription() - cannot process type', type(_poly))

    #
    # Determine counterclockwise angle
    #
    def grahamScan_ccw(self,pt1,pt2,pt3):
        x1,y1 = pt1[0],pt1[1]
        x2,y2 = pt2[0],pt2[1]
        x3,y3 = pt3[0],pt3[1]
        return (x2-x1)*(y3-y1)-(y2-y1)*(x3-x1)

    #
    # grahamScan()
    # - compute the convex hull of x,y points in a lookup table
    # - lookup table is what networkx uses for layouts
    #
    # https://en.wikipedia.org/wiki/Graham_scan
    #
    def grahamScan(self,pos):
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
        for x in polar_lu.keys():
            to_sort.append((x, polar_lu[x]))
        points = sorted(to_sort)

        stack  = []
        for point in points:
            while len(stack) > 1 and self.grahamScan_ccw(pos[stack[-2][1]],pos[stack[-1][1]],pos[point[1]]) <= 0:
                stack = stack[:-1]
            stack.append(point)

        ret = []
        ret.append(pt_low)
        for x in stack:
            ret.append(x[1])
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

        d_str = ''

        for i in range(0,len(pts)):
            pt0 = pts[i]
            pt1 = pts[(i+1)%len(pts)]
            pt2 = pts[(i+2)%len(pts)]

            x0,y0 = pos[pt0][0],pos[pt0][1]
            x1,y1 = pos[pt1][0],pos[pt1][1]
            x2,y2 = pos[pt2][0],pos[pt2][1]

            dx,dy = x1 - x0,y1 - y0
            l  = sqrt(dx*dx+dy*dy)
            if l < 0.001:
                l = 0.001
            dx /= l
            dy /= l
            pdx =  dy
            pdy = -dx

            dx2,dy2 = x2 - x1,y2 - y1
            l2  = sqrt(dx2*dx2+dy2*dy2)
            if l2 < 0.001:
                l2 = 0.001
            dx2 /= l2
            dy2 /= l2
            pdx2 =  dy2
            pdy2 = -dx2

            # First point is a move to...
            if len(d_str) == 0:
                d_str += f'M {x0+pdx*r} {y0+pdy*r} '
            
            # Line along the the polygon edge
            d_str += f'L {x1+pdx*r} {y1+pdy*r} '

            # Curved cap
            cx0 = x1+pdx  *r + dx  * r/4
            cy0 = y1+pdy  *r + dy  * r/4
            cx1 = x1+pdx2 *r - dx2 * r/4
            cy1 = y1+pdy2 *r - dy2 * r/4

            d_str += f'C {cx0} {cy0} {cx1} {cy1} {x1+pdx2*r} {y1+pdy2*r}'

            #d_str += f'L {cx0} {cy0} '
            #d_str += f'L {cx1} {cy1} '
            d_str += f'L {x1+pdx2*r} {y1+pdy2*r} '

        return d_str

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
        h,w = len(_raster),len(_raster[0])

        # Allocate the level set
        state      = [[None for x in range(w)] for y in range(h)] # node that found the pixel
        found_time = [[None for x in range(w)] for y in range(h)] # when node was found
        origin     = [[None for x in range(w)] for y in range(h)] # when node was found

        # Distance lambda function
        dist = lambda _x0,_y0,_x1,_y1: sqrt((_x0-_x1)*(_x0-_x1)+(_y0-_y1)*(_y0-_y1))

        # Copy the _raster 
        for x in range(0,len(_raster[0])):
            for y in range(0,len(_raster)):
                if _raster[y][x] is not None and _raster[y][x] != 0:
                    state[y][x]      = _raster[y][x]  # class of the find
                    found_time[y][x] = 0              # what time it was found
                    origin[y][x]     = (y,x)          # origin of the finder

        # Initialize the heap
        _heap = []
        for x in range(0,len(_raster[0])):
            for y in range(0,len(_raster)):
                if state[y][x] is not None and state[y][x] > 0: # Only expand non-walls and set indices...
                    for dx in range(-1,2):
                        for dy in range(-1,2):
                            if dx == 0 and dy == 0:
                                continue
                            xn,yn = x+dx,y+dy
                            if xn >= 0 and yn >= 0 and xn < w and yn < h:
                                if state[yn][xn] is None or state[yn][x] == 0:
                                    t = dist(x, y, xn, yn)
                                    heapq.heappush(_heap,(t, xn, yn, state[y][x], origin[y][x][0], origin[y][x][1]))

        # Go through the heap
        while len(_heap) > 0:
            t,xi,yi,_class,y_origin,x_origin = heapq.heappop(_heap)
            if state[yi][xi] is not None and state[yi][xi] < 0:           # Check for a wall
                continue
            if found_time[yi][xi] is None or found_time[yi][xi] > t:      # Deterimine if we should overwrite the state
                state [yi][xi]     = _class
                found_time[yi][xi] = t
                origin[yi][xi]      = (y_origin,x_origin)
                for dx in range(-1,2):                                    # Add the neighbors to the priority queue
                    for dy in range(-1,2):
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
                                heapq.heappush(_heap,(t, xn, yn, state[y_origin][x_origin], y_origin, x_origin))

        return state, found_time, origin


    #
    # Implemented from pseudocode on https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    #
    def __bresenhamsLow__(self, x0,y0,x1,y1):
        dx, dy, yi, pts = x1 - x0, y1 - y0, 1, []
        if dy < 0:
            yi, dy = -1, -dy
        D, y = (2*dy)-dx, y0
        for x in range(x0,x1+1):
            pts.append((x,y))
            if D > 0:
                y += yi
                D += 2*(dy-dx)
            else:
                D += 2*dy
        return pts
    #
    # Implemented from pseudocode on https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm
    #
    def __bresenhamsHigh__(self, x0,y0,x1,y1):
        dx, dy, xi, pts = x1 - x0, y1 - y0, 1, []
        if dx < 0:
            xi, dx = -1, -dx
        D, x = (2*dx)-dy, x0
        for y in range(y0,y1+1):
            pts.append((x,y))
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
    def bresenhams(self, x0,y0,x1,y1):
        if abs(y1-y0) < abs(x1-x0):
            return self.__bresenhamsLow__(x1,y1,x0,y0)  if (x0 > x1) else self.__bresenhamsLow__(x0,y0,x1,y1)
        else:
            return self.__bresenhamsHigh__(x1,y1,x0,y0) if (y0 > y1) else self.__bresenhamsHigh__(x0,y0,x1,y1)

    #
    # levelSet() - slower version but more precise (takes objects into consideration)
    # - takes approximately 10x times as long as the fast method... (with small rasters... < 256x256)
    # - raster is a two dimensional structure ... _raster[y][x]
    # - "0" or None means to calculate
    # - "-1" means a wall / immovable object
    # - "> 0" means the class to expand 
    #
    def levelSet(self, _raster):
        h,w = len(_raster),len(_raster[0])

        # Allocate the level set
        state      = [[None for x in range(w)] for y in range(h)] # node that found the pixel
        found_time = [[None for x in range(w)] for y in range(h)] # when node was found
        origin     = [[None for x in range(w)] for y in range(h)] # when node was found

        # Distance lambda function
        dist = lambda _x0,_y0,_x1,_y1: sqrt((_x0-_x1)*(_x0-_x1)+(_y0-_y1)*(_y0-_y1))

        # Copy the _raster 
        for x in range(0,len(_raster[0])):
            for y in range(0,len(_raster)):
                if _raster[y][x] is not None and _raster[y][x] != 0:
                    state[y][x]      = _raster[y][x]  # class of the find
                    found_time[y][x] = 0              # what time it was found
                    origin[y][x]     = (y,x)          # origin of the finder

        # Initialize the heap
        _heap = []
        for x in range(0,len(_raster[0])):
            for y in range(0,len(_raster)):
                if state[y][x] is not None and state[y][x] > 0: # Only expand non-walls and set indices...
                    for dx in range(-1,2):
                        for dy in range(-1,2):
                            if dx == 0 and dy == 0:
                                continue
                            xn,yn = x+dx,y+dy
                            if xn >= 0 and yn >= 0 and xn < w and yn < h:
                                if state[yn][xn] is None or state[yn][x] == 0:
                                    t = dist(x, y, xn, yn)
                                    heapq.heappush(_heap,(t, xn, yn, state[y][x], origin[y][x][0], origin[y][x][1]))

        # Go through the heap
        while len(_heap) > 0:
            t,xi,yi,_class,y_origin,x_origin = heapq.heappop(_heap)
            t = dist(xi,yi,x_origin,y_origin) + found_time[y_origin][x_origin]
            if state[yi][xi] is not None and state[yi][xi] < 0:           # Check for a wall
                continue
            if found_time[yi][xi] is None or found_time[yi][xi] > t:      # Deterimine if we should overwrite the state
                state [yi][xi]     = _class
                found_time[yi][xi] = t
                origin[yi][xi]      = (y_origin,x_origin)
                for dx in range(-1,2):                                    # Add the neighbors to the priority queue
                    for dy in range(-1,2):
                        if dx == 0 and dy == 0:
                            continue
                        xn, yn = xi + dx, yi + dy
                        # Within bounds?
                        if xn >= 0 and yn >= 0 and xn < w and yn < h:
                            t = found_time[yi][xi] + dist(xi, yi, xn, yn)
                            # Adjust the origin if we can't see the origin from the new point...
                            x_origin_adj, y_origin_adj = x_origin, y_origin
                            path = self.bresenhams(xn,yn,x_origin,y_origin)
                            for pt in path:
                                if state[pt[1]][pt[0]] is not None and state[pt[1]][pt[0]] < 0:
                                    x_origin_adj, y_origin_adj = xi, yi
                            if found_time[yn][xn] is None or found_time[yn][xn] > t:
                                heapq.heappush(_heap,(t, xn, yn, state[y_origin][x_origin], y_origin_adj, x_origin_adj))
        return state, found_time, origin

    #
    #
    #
    def levelSetStateAndFoundTimeSVG(self, _state, _found_time):
        _w,_h = len(_state[0]),len(_state)
        svg = f'<svg x="0" y="0" width="{_w*2}" height="{_h}">'

        _tmax = 1
        for y in range(0,_h):
            for x in range(0,_w):
                if _found_time[y][x] is not None and _found_time[y][x] > _tmax:
                    _tmax = _found_time[y][x]

        for y in range(0,_h):
            for x in range(0,_w):
                if _state[y][x] == -1:
                    _co = '#000000'
                else:
                    _co = self.co_mgr.getColor(_state[y][x])
                svg += f'<rect x="{x}" y="{y}" width="{1}" height="{1}" fill="{_co}" stroke-opacity="0.0" />'

                if _found_time[y][x] is not None:
                    if _state[y][x] == -1:
                        _co = '#000000'
                    else:
                        _co = self.co_mgr.spectrum(_found_time[y][x], 0, _tmax)
                else:
                    _co = '#ffffff' # shouldn't really ever be here...
                svg += f'<rect x="{x+_w}" y="{y}" width="{1}" height="{1}" fill="{_co}" stroke-opacity="0.0" />'

        return svg + '</svg>'

    #
    # smoothSegments() - smooth out segments with a 3 window kernel.
    #
    def smoothSegments(self, segments):
        smoothed = [segments[0]]
        for i in range(1, len(segments)-1):
            x, y = (segments[i-1][0] + segments[i][0] + segments[i+1][0])/3.0 , (segments[i-1][1] + segments[i][1] + segments[i+1][1])/3.0
            smoothed.append((x,y))
        smoothed.append(segments[-1])
        return smoothed
    
    #
    # expandSegmentsIntoPiecewiseCurvePartsFIXEDINC()
    # ... old version of this method...
    #
    def expandSegmentsIntoPiecewiseCurvedPartsFIXEDINC(self, segments, amp=5.0, ampends=20.0, t_inc=0.1):
        _piecewise_ = [segments[0], segments[1]]
        for i in range(1,len(segments)-2):
            _amp_ = ampends if ((i == 1) or (i == len(segments)-3)) else amp
            v0 = self.unitVector([segments[i],   segments[i-1]])
            v1 = self.unitVector([segments[i+1], segments[i+2]])
            bc = self.bezierCurve(segments[i], ( segments[i][0]-_amp_*v0[0] , segments[i][1]-_amp_*v0[1] ), ( segments[i+1][0]-_amp_*v1[0] , segments[i+1][1]-_amp_*v1[1] ), segments[i+1])
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
        _piecewise_ = [segments[0], segments[1]]
        for i in range(1,len(segments)-2):
            _amp_ = ampends if ((i == 1) or (i == len(segments)-3)) else amp
            v0 = self.unitVector([segments[i],   segments[i-1]])
            v1 = self.unitVector([segments[i+1], segments[i+2]])
            bc = self.bezierCurve(segments[i], ( segments[i][0]-_amp_*v0[0] , segments[i][1]-_amp_*v0[1] ), ( segments[i+1][0]-_amp_*v1[0] , segments[i+1][1]-_amp_*v1[1] ), segments[i+1])
            t_lu = {}
            ts   = []
            t_lu[0.0] = bc(0.0)
            ts.append(0.0)
            t_lu[1.0] = bc(1.0)
            ts.append(1.0)
            j = 0
            while j < len(ts)-1:
                l = self.segmentLength((  t_lu[ts[j]]  ,  t_lu[ts[j+1]]  ))
                if l > max_travel:
                    t_new = (ts[j] + ts[j+1])/2.0
                    ts.insert(j+1, t_new)
                    t_lu[t_new] = bc(t_new)
                else:
                    j += 1
            for j in range(0,len(ts)-1):
                _piecewise_.append(t_lu[ts[j]])
        _piecewise_.append(segments[-1])
        return _piecewise_

    #
    # segmentOctTree() - return a segment octree
    # - bounds == (x0,y0,x1,y1)    
    #
    def segmentOctTree(self, bounds, max_segments_per_cell=20):
        return SegmentOctTree(self, bounds, max_segments_per_cell=max_segments_per_cell)

#
# SegmentOctTree -- oct tree implementation for faster segment discovery.
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
        self.tree_already_split     = {}
        self.tree_already_split[''] = False

        # For Debugging...
        self.pieces                 = set()            # for debugging...
        debug = False
        if debug:
            self.__split__('')
            iters = 0
            while (iters < 4):
                ks = set(self.tree.keys())
                for k in ks:
                    self.__split__(k)
                iters += 1

    #
    # findOctet() - find octet for point.
    #
    def findOctet(self, pt):
        last_s = s = ''
        b = self.bounds
        while s in self.tree.keys():
            b = self.tree_bounds[s]
            if    pt[0] <= (b[0]+b[2])/2.0 and pt[1] <= (b[1]+b[3])/2.0:
                n = '0'
            elif  pt[0] >  (b[0]+b[2])/2.0 and pt[1] <= (b[1]+b[3])/2.0:
                n = '1'
            elif  pt[0] <= (b[0]+b[2])/2.0 and pt[1] >  (b[1]+b[3])/2.0:
                n = '2'
            elif  pt[0] >  (b[0]+b[2])/2.0 and pt[1] >  (b[1]+b[3])/2.0:
                n = '3'
            last_s = s
            s += n
        return last_s

    #
    # __split__() - split a tree node into four parts ... not thread safe
    #
    def __split__(self, node):
        if self.tree_already_split[node]:
            return
        else:
            self.tree_already_split[node] = True
        
        b = self.tree_bounds[node]
        self.tree       [node+'0'] = set()
        self.tree_bounds[node+'0'] = (b[0],            b[1],            (b[0]+b[2])/2.0, (b[1]+b[3])/2.0)
        self.tree_already_split[node+'0'] = False

        self.tree       [node+'1'] = set()
        self.tree_bounds[node+'1'] = ((b[0]+b[2])/2.0, b[1],            b[2],            (b[1]+b[3])/2.0)        
        self.tree_already_split[node+'1'] = False

        self.tree       [node+'2'] = set()
        self.tree_bounds[node+'2'] = (b[0],            (b[1]+b[3])/2.0, (b[0]+b[2])/2.0, b[3])
        self.tree_already_split[node+'2'] = False

        self.tree       [node+'3'] = set()
        self.tree_bounds[node+'3'] = ((b[0]+b[2])/2.0, (b[1]+b[3])/2.0, b[2],            b[3])
        self.tree_already_split[node+'3'] = False


        to_check =      [node+'0', node+'1', node+'2', node+'3']
        for piece in self.tree[node]:
            x_min, y_min, x_max, y_max = min(piece[0][0], piece[1][0]), min(piece[0][1], piece[1][1]), max(piece[0][0], piece[1][0]), max(piece[0][1], piece[1][1])
            oct0, oct1, piece_addition_count = self.findOctet(piece[0]), self.findOctet(piece[1]), 0
            for k in to_check:
                b = self.tree_bounds[k]                                    
                if   x_max < b[0] or x_min > b[2] or y_max < b[1] or y_min > b[3]:
                        pass
                elif oct0 == oct1 and oct0 == k:
                    self.tree[k].add(piece)
                    piece_addition_count +=1
                elif self.rt_self.segmentsIntersect(piece, ((b[0],b[1]),(b[0],b[3]))) or \
                     self.rt_self.segmentsIntersect(piece, ((b[0],b[1]),(b[2],b[1]))) or \
                     self.rt_self.segmentsIntersect(piece, ((b[2],b[3]),(b[0],b[3]))) or \
                     self.rt_self.segmentsIntersect(piece, ((b[2],b[3]),(b[2],b[1]))):
                        self.tree[k].add(piece)
                        piece_addition_count += 1
            if piece_addition_count == 0:
                print(f"Error -- No additions for piece {piece} ... node = {node}")
        self.tree[node] = set()
        for k in to_check:
            if len(self.tree[k]) > self.max_segments_per_cell:
                self.__split__(k)

    #
    # addSegments() -- add segments to the tree
    # - segments = [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
    def addSegments(self, segments):
        for i in range(len(segments)-1):
            piece = ((segments[i][0], segments[i][1]), (segments[i+1][0], segments[i+1][1])) # make sure it's a tuple
            self.pieces.add(piece)
            oct0  = self.findOctet(segments[i])
            x0,y0 = segments[i]
            oct1  = self.findOctet(segments[i+1])
            x1,y1 = segments[i+1]
            x_min,y_min,x_max,y_max = min(x0,x1), min(y0,y1), max(x0,x1), max(y0,y1)
            if oct0 == oct1:
                self.tree[oct0].add(piece)
                if len(self.tree[oct0]) > self.max_segments_per_cell:
                    self.__split__(oct0)
            else:
                to_split = set() # to avoid messing with the keys in this iteration
                self.tree[oct0].add(piece)
                if len(self.tree[oct0]) > self.max_segments_per_cell:
                    to_split.add(oct0)
                self.tree[oct1].add(piece)
                if len(self.tree[oct1]) > self.max_segments_per_cell:
                    to_split.add(oct1)
                for k in self.tree_bounds.keys():
                    b      = self.tree_bounds[k]
                    if k != oct0 and k != oct1:
                        if   x_max < b[0] or x_min > b[2] or y_max < b[1] or y_min > b[3]:
                             pass
                        elif self.rt_self.segmentsIntersect(piece, ((b[0],b[1]),(b[0],b[3]))) or \
                             self.rt_self.segmentsIntersect(piece, ((b[0],b[1]),(b[2],b[1]))) or \
                             self.rt_self.segmentsIntersect(piece, ((b[2],b[3]),(b[0],b[3]))) or \
                             self.rt_self.segmentsIntersect(piece, ((b[2],b[3]),(b[2],b[1]))):
                             self.tree[k].add(piece)
                             if len(self.tree[k]) > self.max_segments_per_cell:
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
        to_check   = set([oct0,oct1])
        if    oct0 == oct1:
            to_check |= oct0_nbors
        elif  oct1 in oct0_nbors:
            to_check |= oct0_nbors | self.__neighbors__(oct1)
        else: # :( ... have to search for all possibles...
            x_min, y_min = min(segment[0][0], segment[1][0]), min(segment[0][1], segment[1][1])
            x_max, y_max = max(segment[0][0], segment[1][0]), max(segment[0][1], segment[1][1])
            for k in self.tree_bounds.keys():
                b      = self.tree_bounds[k]
                if k != oct0 and k != oct1:
                    if   x_max < b[0] or x_min > b[2] or y_max < b[1] or y_min > b[3]:
                        pass
                    elif self.rt_self.segmentsIntersect(segment, ((b[0],b[1]),(b[0],b[3]))) or \
                         self.rt_self.segmentsIntersect(segment, ((b[0],b[1]),(b[2],b[1]))) or \
                         self.rt_self.segmentsIntersect(segment, ((b[2],b[3]),(b[0],b[3]))) or \
                         self.rt_self.segmentsIntersect(segment, ((b[2],b[3]),(b[2],b[1]))):
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
            if self.tree_already_split[k]: # don't bother with split nodes
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
        _node_    = _as_list_[random.randint(0,len(_as_list_)-1)]
        while self.tree_already_split[_node_]: # find a non-split node...
            _node_    = _as_list_[random.randint(0,len(_as_list_)-1)]
        _node_b_  = self.tree_bounds[_node_]
        xc, yc    = (_node_b_[0]+_node_b_[2])/2.0, (_node_b_[1]+_node_b_[3])/2.0
        _nbors_   = self.__neighbors__(_node_)
        for _nbor_ in _nbors_:
            _nbor_b_ = self.tree_bounds[_nbor_]
            xcn, ycn = (_nbor_b_[0]+_nbor_b_[2])/2.0, (_nbor_b_[1]+_nbor_b_[3])/2.0
            svg += f'<line x1="{xT(xc)}" y1="{yT(yc)}" x2="{xT(xcn)}" y2="{yT(ycn)}" stroke="#000000" stroke-width="0.5" />'
            
        svg +=  '</svg>'
        return svg

