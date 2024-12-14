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
    # toGeoJSONPoints() - converts a Polars DataFrame to GeoJSON points
    #
    def toGeoJSONPoints(self, df, lat_field='latitude', lon_field='longitude', id_field='id'):
        _geojson_ = {"type": "FeatureCollection","features":[]}
        for k, k_df in df.group_by([lat_field, lon_field]):
            _id_set_ = set(k_df[id_field].unique())
            _id_     = _id_set_.pop() if len(_id_set_) == 1 else 'unknown'
            _dict_   = {'type':'Feature', 'geometry':{'type':'Point', 'coordinates':[k[1], k[0]]}, 'properties':{'id':_id_}}
            _geojson_['features'].append(_dict_)
        return _geojson_

    #
    # toGeoJSONPaths() - converts a Polars DataFrame to GeoJSON paths
    #
    def toGeoJSONPaths(self, df, lat_field='latitude', lon_field='longitude', id_field='id', seq_field='timestamp'):
        _geojson_ = {"type": "FeatureCollection","features":[]}
        df = df.sort(seq_field) 
        for k, k_df in df.group_by(id_field, maintain_order=True):
            _dict_ = {'type':'Feature', 'geometry':{'type':'LineString', 'coordinates':list(zip(k_df[lon_field], k_df[lat_field]))}, 'properties':{'id':k}}
            _geojson_['features'].append(_dict_)
        return _geojson_

    #
    # geoMapsUSStates() - returns a 2-digit state lookup for states to their shapely polygon.
    #
    def geoMapsUSStates(self, version='hex'):
        if   version == 'hex':
            return self.__geoMapsUSStates_hex_v3__()
        elif version == 'circles':
            return self.__geoMapsUSStates_circles__()
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

    #
    # __geoMapUSStatesBorderGraph__() - return a lookup for each state and the states that it borders
    #
    def __geoMapsUSStatesBorderGraph__(self):
        lu =   {'ak':set(),
                'al':set(['ms','tn','ga','fl']),
                'ar':set(['la','tx','ok','mo','tn','ms']),
                'az':set(['ca','nv','ut','nm','co']),
                'ca':set(['or','nv','az']),
                'ct':set(['ri','ma','ny']),
                'co':set(['ut','nm','wy','ne','ks','ok','az']),
                'de':set(['md','pa','nj']),
                'ga':set(['sc','nc','tn','al','fl']),
                'fl':set(['al','ga']),
                'ks':set(['ok','co','ne','mo']),
                'ky':set(['tn','mo','il','in','oh','wv','va']),
                'hi':set(),
                'ia':set(['ne','sd','mn','wi','il','mo']),
                'id':set(['wa','or','mt','nv','ut','wy']),
                'il':set(['mo','ia','wi','in','ky','mi']),
                'in':set(['il','mi','ky','oh']),
                'la':set(['tx','ar','ms']),
                'ma':set(['ny','vt','nh','ct','ri']),
                'md':set(['va','wv','pa','de']),
                'me':set(['nh']),
                'mi':set(['in','wi','oh','il','mn']),
                'mn':set(['nd','sd','ia','wi','mi']),
                'mo':set(['ok','ks','ne','ia','il','ky','tn','ar']),
                'ms':set(['la','ar','tn','al']),
                'mt':set(['id','wy','sd','nd']),
                'nc':set(['sc','ga','tn','va']),
                'nd':set(['mt','sd','mn']),
                'ne':set(['wy','sd','ia','mo','ks','co']),
                'nh':set(['me','vt','ma']),
                'nj':set(['pa','de','ny']),
                'nm':set(['az','co','ok','tx','ut']),
                'nv':set(['ca','az','ut','id','or']),
                'ny':set(['pa','nj','ct','ma','vt','ri']),
                'oh':set(['in','mi','ky','wv','pa']),
                'ok':set(['nm','tx','ar','mo','ks','co']),
                'or':set(['wa','id','nv','ca']),
                'pa':set(['oh','wv','md','de','nj','ny']),
                'ri':set(['ma','ct','ny']),
                'sc':set(['ga','nc']),
                'sd':set(['wy','mt','nd','mn','ia','ne']),
                'tn':set(['mo','ky','va','ga','al','ms','ar','nc']),
                'tx':set(['nm','ok','ar','la']),
                'ut':set(['nv','az','co','wy','id','nm']),
                'va':set(['wv','md','ky','tn','nc']),
                'vt':set(['nh','ma','ny']),
                'wa':set(['or','id']),
                'wi':set(['mn','ia','il','mi']),
                'wv':set(['md','pa','oh','ky','va']),
                'wy':set(['id','mt','ut','co','ne','sd'])}
        # Internally consistent?
        for x in lu.keys():
            for k in lu[x]:
                if x not in lu[k]:            
                    print(f'"{x}" not in "{k}"')        
        # Print out in descending order how many...
        for x in range(10,-1,-1):
            matches = set()
            for k in lu.keys():
                if len(lu[k]) == x:
                    matches.add(k)
            # print(x,sorted(list(matches)))
        return lu

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

    #
    #
    #
    def __geoMapsUSStates_hex_v2__(self, l=10.0):
        slices = [
            ['ak','',  'wa','or','nv','ca','',  'hi'],
            ['',  'mt','id','wy','ut','az'],
            ['',  '',  'nd','sd','ne','co','nm'],
            ['',  'mn','ia','mo','ks','ok','tx'],
            ['',  '',  'wi','il','ar','la','ms'],
            ['',  'mi','in','ky','tn','al'],
            ['',  '',  'oh','wv','md','nc','ga'],
            ['',  'ny','pa','nj','va','sc','fl'],
            ['',  'vt','ma','ri','de'],
            ['me','nh','ct','',  '',  'dc']
        ]
        return self.__hexEncoder__(slices, l)

    def __geoMapsUSStates_hex_v3__(self, l=10.0):
        slices = [
            ['ak','',  'wa','or','nv','ca','',  'hi'],
            ['',  'mt','id','wy','ut','az'],
            ['',  '',  'nd','sd','ne','co','nm'],
            ['',  'mn','ia','mo','ks','ok','tx'],
            ['',  '',  'wi','il','ar','la','ms'],
            ['',  'mi','in','ky','tn','al'],
            ['',  '',  'oh','wv','md','nc','ga'],
            ['',  'ny','pa','nj','va','sc','fl'],
            ['',  'vt','ma','ri','de'],
            ['',  'nh','ct','',  '',  'dc'],
            ['',  'me']
        ]
        return self.__hexEncoder__(slices, l)

    def __hexEncoder__(self, slices, l):
        l2 = l/2.0
        a  = pi*60.0/180.0
        h  = l*sin(a)
        lu,x,y,states = {}, 0.0, 0.0, set()
        for col_i in range(len(slices)):
            cx = x + col_i*1.5*l
            for row_i in range(len(slices[col_i])):
                y_off = 0 if (col_i%2) == 0 else h
                cy = y - y_off - row_i*2*h
                if slices[col_i][row_i] != '':
                    lu[slices[col_i][row_i]] = Polygon(self.__hexagon__(cx, cy, l))
                    states.add(slices[col_i][row_i])
        return lu

    #
    # __geoMapsUSStates_circles__() - circle version
    #
    def __geoMapsUSStates_circles__(self):
        slices = [
            ['',   'wa', 'or', 'ca', '',   '',   'ak'],
            ['',   'id', 'nv', 'ut', 'az', '',   'hi'],
            ['',   'mt', 'wy', 'co', 'nm'],
            ['',   'nd', 'sd', 'ne', 'ks', 'ok', 'tx'],
            ['',   'mn', 'ia', 'mo', 'ar', 'la'],
            ['',   'wi', 'il', 'in', 'ky', 'ms'],
            ['',   'mi', 'oh', 'wv', 'tn', 'al'],
            ['',   '',   'pa', 'va', 'sc', 'ga'],
            ['',   'ny', 'nj', 'md', 'nc', '',   'fl'],
            ['vt', 'ma', 'ct', 'de'],
            ['me', 'nh', 'ri', '',   'dc']
        ]
        lu = {}
        for col_i in range(len(slices)):
            cx = col_i
            for row_i in range(len(slices[col_i])):
                cy = -row_i
                if slices[col_i][row_i] != '':
                    lu[slices[col_i][row_i]] = self.__circle__(cx,cy,0.5)
        return lu

    #
    # __circle__() - approximates a circle with a polygon
    #
    def __circle__(self, cx, cy, r):
        pts = []
        for a in range(360, 0, -2):
            rads = pi * a/180.0
            pts.append([cx+r*cos(rads),cy+r*sin(rads)])
        return Polygon(pts)

