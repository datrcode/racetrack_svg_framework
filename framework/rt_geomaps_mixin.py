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

from math import sin,cos,sqrt,pi
from shapely import Polygon

__name__ = 'rt_geomaps_mixin'

#
# Abstraction for <Simplified> Geographic Maps
#
class RTGeoMapsMixin(object):
    #
    # geoMapsUSStates() - returns a 2-digit state lookup for states to their shapely polygon.
    #
    def geoMapsUSStates(self, version='hex'):
        if version == 'hex':
            return self.__geoMapsUSStates_hex__()
        else:
            raise Exception(f'RTGeoMaps.geoMapsUSStates() - unknown version "{version}"')

    #
    # __hexagon__() - creates a hexagon, counterclockwise points, flat tops...
    #
    def __hexagon__(self, cx, cy, l):
        pts = []
        l2  = l/2.0
        a   = pi * 60.0 / 180.0
        h   = l*sin(a)
        pts.append([cx-l,  cy])
        pts.append([cx-l2, cy-h])
        pts.append([cx+l2, cy-h])
        pts.append([cx+l,  cy])
        pts.append([cx+l2, cy+h])
        pts.append([cx-l2, cy+h])
        return pts

    def __geoMapsUSStates_hex__(self, l=10.0):
        #
        # Started with "standard map" from article:
        # ... https://www.flerlagetwins.com/2018/11/what-hex-brief-history-of-hex_68.html
        # ... modified to work with the representations in this library
        # ... hexagons from that site were pointy on top/bottom... this implementation is the
        #     opposite (flat on top/bottom)... unclear which is should be...
        #     maybe start with the states' shared border graph?
        #
        locs = [
            ['',  '',  '',  '',  '',  '',  '',  '',  '',  'nh'],
            ['',  '',  '',  '',  '',  '',  '',  '',  'vt','me'],
            ['wa','mt','nd','mn','wi','',  'mi','ny','ma','ri'],
            ['id','wy','sd','ia','il','in','oh','pa','nj','ct'],
            ['or','nv','co','ne','mo','ky','wv','md','de'],
            ['ca','az','ut','ks','ar','tn','va','nc'],
            ['',  '',  'nm','ok','la','ms','al','sc','',  'dc'],
            ['',  '',  'tx','',  '',  '',  'ga'],
            ['hi','ak','',  '',  '',  '',  '',  'fl']
        ]

        l2  = l/2.0
        a   = pi * 60.0 / 180.0
        h   = l*sin(a)

        lu = {}
        x  = 0.0
        y  = 0.0
        for row_i in range(len(locs)):
            xoff = 0.0 if (row_i%2) == 0 else (1.5*l)
            for col_i in range(len(locs[row_i])):
                cx, cy = x + xoff + col_i*3*l, y-row_i*h
                if locs[row_i][col_i] != '':
                    lu[locs[row_i][col_i]] = Polygon(self.__hexagon__(cx, cy, l))
        return lu

