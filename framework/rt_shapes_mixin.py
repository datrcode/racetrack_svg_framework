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

__name__ = 'rt_shapes_mixin'

class RTShapesMixin(object):
    #
    # Render Shape
    #
    def renderShape(self,
                    _shape, # "ellipse", "triangle", "utriangle", "diamond", "plus", "x"
                    _x,
                    _y,
                    _sz,
                    _co,
                    _co_border=None,
                    _opacity=1.0):

        if _co_border is None:
            _co_border=_co
            
        if   _shape is None or _shape == 'ellipse':
            return f'<circle cx="{_x}" cy="{_y}" r="{_sz}" fill="{_co}" stroke="{_co_border}" fill-opacity="{_opacity}" stroke-opacity="{_opacity}" />'
        elif _shape == 'square':
            return f'<rect x="{_x-_sz}" y="{_y-_sz}" width="{2*_sz}" height="{2*_sz}" fill="{_co}" stroke="{_co_border}" fill-opacity="{_opacity}" stroke-opacity="{_opacity}" />'
        elif _shape == 'triangle':
            return f'<path d="M {_x} {_y-_sz} l {_sz} {2*_sz} l {-2*_sz} 0 z" fill="{_co}" stroke="{_co_border}" fill-opacity="{_opacity}" stroke-opacity="{_opacity}" />'
        elif _shape == 'utriangle':
            return f'<path d="M {_x} {_y+_sz} l {-_sz} {-2*_sz} l {2*_sz} 0 z" fill="{_co}" stroke="{_co_border}" fill-opacity="{_opacity}" stroke-opacity="{_opacity}" />'
        elif _shape == 'diamond':
            svg  = f'<path d="M {_x} {_y-_sz} l {_sz} {_sz} l {-_sz} {_sz} '
            svg += f'l {-_sz} {-_sz} z" fill="{_co}" stroke="{_co_border}" fill-opacity="{_opacity}" stroke-opacity="{_opacity}" />'
            return svg
        elif _shape == 'plus':
            return f'<path d="M {_x} {_y-_sz} v {2*_sz} M {_x-_sz} {_y} h {2*_sz}" stroke="{_co}" stroke="{_co_border}" fill-opacity="{_opacity}" stroke-opacity="{_opacity}" />'
        elif _shape == 'x':
            svg  = f'<path d="M {_x-_sz} {_y-_sz} l {2*_sz} {2*_sz} '
            svg += f'M {_x-_sz} {_y+_sz} l {2*_sz} {-2*_sz}" stroke="{_co}" stroke="{_co_border}" fill-opacity="{_opacity}" stroke-opacity="{_opacity}" />'
            return svg
        else:
            return f'<ellipse cx="{_x}" cy="{_y}" rx="{_sz}" fill="{_co}" stroke="{_co_border}" fill-opacity="{_opacity}" stroke-opacity="{_opacity}" />'

    #
    # shapeByDataFrameLength()
    # ... example of how to write a shape function
    # ... beta... subject to change until determine how this usually works
    #
    def shapeByDataFrameLength(self,
                               _df,
                               _key_tuple,
                               _x,
                               _y,
                               _w,
                               _color,
                               _opacity):
        _len = len(_df)
        if   _len == 0:
            return 'x'
        elif _len == 1:
            return 'plus'
        elif _len <  100:
            return 'triangle'
        elif _len <  1000:
            return 'ellipse'
        else:
            return 'square'
