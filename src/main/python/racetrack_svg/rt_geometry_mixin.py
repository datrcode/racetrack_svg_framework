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

from shapely.geometry              import Point, Polygon
from shapely.geometry.multipolygon import MultiPolygon
from math import sqrt,acos

import heapq

__name__ = 'rt_geometry_mixin'

#
# Geometry Methods
#
class RTGeometryMixin(object):

    #
    # Converts a shapely polygon to an SVG path...
    # ... assumes that the ordering (CW, CCW) of both the exterior and interior points is correct...
    #
    def shapelyPolygonToSVGPathDescription(self, _poly):
        if type(_poly) == MultiPolygon:
            path_str = ''
            for _subpoly in _poly.geoms:
                if len(path_str) > 0:
                    path_str += ' '
                path_str += self.shapelyPolygonToSVGPathDescription(_subpoly)
            return path_str
        else:
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
    # levelSet()
    # - raster is a two dimensional structure ... _raster[y][x]
    # - "0" or None means to calculate
    # - "-1" means a wall / immovable object
    # - "> 0" means the class to expand 
    #
    def levelSet(self,
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

