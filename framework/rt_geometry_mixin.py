# Copyright 2022 David Trimm
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

