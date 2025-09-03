# Copyright 2025 David Trimm
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

import polars as pl
from math import sqrt

__name__ = 'rt_shapes_mixin'

class RTShapesMixin(object):
    #
    # Render Shape
    #
    def renderShape(self,
                    shape, # "ellipse", "square", "triangle", "utriangle", "diamond", "plus", "x"
                    x,
                    y,
                    sz        = 5,
                    co        = None,
                    co_border = None,
                    opacity   = None,
                    svg_id    = None):
        id_str    = '' if svg_id is None else f'id="{svg_id}"'

        append_ls = []
        append_ls.append('' if co        is None else f'fill="{co}"')
        append_ls.append('' if co_border is None else f'stroke="{co_border}"')
        append_ls.append('' if opacity   is None else f'fill-opacity="{opacity}" stroke-opacity="{opacity}"')
        _append_ = ' '.join(append_ls)

        if   shape is None      or \
             shape == 'ellipse' or \
             shape == 'circle':    return f'<circle {id_str} cx="{x}" cy="{y}" r="{sz}" {_append_} />'
        elif shape == 'square':    return f'<rect {id_str} x="{x-sz}" y="{y-sz}" width="{2*sz}" height="{2*sz}" {_append_} />'
        elif shape == 'triangle':  return f'<path {id_str} d="M {x} {y-sz} l {sz} {2*sz} l {-2*sz} 0 z" {_append_} />'
        elif shape == 'utriangle': return f'<path {id_str} d="M {x} {y+sz} l {-sz} {-2*sz} l {2*sz} 0 z" {_append_} />'
        elif shape == 'diamond':   return f'<path {id_str} d="M {x} {y-sz} l {sz} {sz} l {-sz} {sz} l {-sz} {-sz} z" {_append_} />'
        elif shape == 'plus':      return f'<path {id_str} d="M {x} {y-sz} v {2*sz} M {x-sz} {y} h {2*sz}" {_append_} />'
        elif shape == 'x':         return f'<path {id_str} d="M {x-sz} {y-sz} l {2*sz} {2*sz} M {x-sz} {y+sz} l {2*sz} {-2*sz}" {_append_} />'
        else:                      return f'<circle {id_str} cx="{x}" cy="{y}" r="{sz}" {_append_} />'

    #
    # renderShapePolars() - same as renderShape() but using polars expressions
    #
    def renderShapePolars(self,
                          shape,                # "ellipse", "square", "triangle", "utriangle", "diamond", "plus", "x"
                          x,                    # pl.col()
                          y,                    # pl.col()
                          sz,                   # float or pl.col()
                          fill         = None,  # none, string, or pl.col()
                          stroke       = None,  # none, string, or pl.col()
                          stroke_width = None,  # none, float, or pl.col()
                          opacity      = None): # none, float, or pl.col()

        if type(sz) is float:
            if   shape == 'square':    _op_ = [pl.lit('<rect x="'),    x-sz, pl.lit('" y="'),  y-sz, pl.lit(f'" width="{2*sz}" height="{2*sz}"')]
            elif shape == 'triangle':  _op_ = [pl.lit('<path d="M '),  x,    pl.lit(' '),      y-sz, pl.lit(f' l {sz} {2*sz} l {-2*sz} 0 z"')]
            elif shape == 'utriangle': _op_ = [pl.lit('<path d="M '),  x,    pl.lit(' '),      y+sz, pl.lit(f' l {-sz} {-2*sz} l {2*sz} 0 z"')]
            elif shape == 'diamond':   _op_ = [pl.lit('<path d="M '),  x,    pl.lit(' '),      y-sz, pl.lit(f' l {sz} {sz} l {-sz} {sz} l {-sz} {-sz} z"')]
            elif shape == 'plus':      _op_ = [pl.lit('<path d="M '),  x,    pl.lit(' '),      y-sz, pl.lit(f' v {2*sz} M {x-sz} {y} h {2*sz}"')]
            elif shape == 'x':         _op_ = [pl.lit('<path d="M '),  x-sz, pl.lit(' '),      y-sz, pl.lit(f' l {2*sz} {2*sz} M {x-sz} {y+sz} l {2*sz} {-2*sz}"')]
            else:                      _op_ = [pl.lit('<circle cx="'), x,    pl.lit('" cy="'),    y, pl.lit(f'" r="{sz}"')]
        else:
            if   shape == 'square':    _op_ = [pl.lit('<rect x="'),    x-sz, pl.lit('" y="'),  y-sz, pl.lit('" width="'), 2*sz, pl.lit('" height="'), 2*sz, pl.lit('"')]
            elif shape == 'triangle':  _op_ = [pl.lit('<path d="M '),  x,    pl.lit(' '),      y-sz, pl.lit(' l '),         sz, pl.lit(' '),          2*sz, pl.lit(' l '), -2*sz, pl.lit(' 0 z"')]
            elif shape == 'utriangle': _op_ = [pl.lit('<path d="M '),  x,    pl.lit(' '),      y+sz, pl.lit(' l '),        -sz, pl.lit(' '),         -2*sz, pl.lit(' l '),  2*sz, pl.lit(' 0 z"')]
            elif shape == 'diamond':   _op_ = [pl.lit('<path d="M '),  x,    pl.lit(' '),      y-sz, pl.lit(' l '),         sz, pl.lit(' '),            sz, pl.lit(' l '),   -sz, pl.lit(' '),     sz, pl.lit(' l '),  -sz, pl.lit(' '),   -sz, pl.lit(' z"')]
            elif shape == 'plus':      _op_ = [pl.lit('<path d="M '),  x,    pl.lit(' '),      y-sz, pl.lit(' v '),       2*sz, pl.lit(' M '),        x-sz, pl.lit(' '),       y, pl.lit(' h '), 2*sz, pl.lit('"')]
            elif shape == 'x':         _op_ = [pl.lit('<path d="M '),  x-sz, pl.lit(' '),      y-sz, pl.lit(' l '),       2*sz, pl.lit(' '),          2*sz, pl.lit(' M '),  x-sz, pl.lit(' '),   y+sz, pl.lit(' l '), 2*sz, pl.lit(' '), -2*sz, pl.lit('"')]
            else:                      _op_ = [pl.lit('<circle cx="'), x,    pl.lit('" cy="'),    y, pl.lit('" r="'),       sz, pl.lit('"')]

        if fill is not None:
             if type(fill) is str:        _op_.append(pl.lit(f' fill="{fill}"'))
             else:                        _op_.extend([pl.lit(f' fill="'), fill, pl.lit('"')])
        if stroke is not None:
             if type(stroke) is str: _op_.append(pl.lit(f' stroke="{stroke}"'))
             else:                      _op_.extend([pl.lit(f' stroke="'), stroke, pl.lit('"')])
        if stroke_width is not None:
             if type(stroke_width) is float: _op_.append(pl.lit(f' stroke-width="{stroke_width}"'))
             else:                           _op_.extend([pl.lit(f' stroke-width="'), stroke_width, pl.lit('"')])
        if opacity is not None:
             if type(opacity) is float: _op_.append(pl.lit(f' opacity="{opacity}"'))
             else:                      _op_.extend([pl.lit(f' opacity="'), opacity, pl.lit('"')])
        
        _op_.append(pl.lit('/>'))

        return _op_

    #
    # shapeAttachmentPoint() - determine the attachment point for a shape
    # - shape, x, y, and sz should be the same as "renderShape()"
    # - xp, yp is the external point to match to th edge of the shape
    # - return (xa,ya) -- attachment coordinates
    #
    def shapeAttachmentPoint(self, shape, x, y, sz, xp, yp):
        # for plus or x just return the shape center
        if   shape == 'plus' or shape == 'x' or sz < 2:
            return x,y
        # for circle, just a vector the size of the circle radius
        if shape is None or shape == 'ellipse' or shape == 'circle':
            vx, vy = xp-x, yp-y
            l = sqrt(vx**2 + vy**2)
            if l > 0.1:
                vx, vy = vx/l, vy/l
                return x+vx*sz, y+vy*sz
        # these shapes are from above -- using the same exact geometry
        elif shape == 'square' or shape == 'triangle' or shape == 'utriangle' or shape == 'diamond':
            to_match = []
            if   shape == 'square':
                to_match = [((x-sz,y-sz),(x+sz,y-sz)),((x-sz,y+sz),(x+sz,y+sz)),((x-sz,y-sz),(x-sz,y+sz)),((x+sz,y-sz),(x+sz,y+sz))]
            elif shape == 'triangle':
                to_match = [((x,y-sz),(x+sz,y+sz)),((x,y-sz),(x-sz,y+sz)),((x-sz,y+sz),(x+sz,y+sz))]
            elif shape == 'utriangle':
                to_match = [((x,y+sz),(x+sz,y-sz)),((x,y+sz),(x-sz,y-sz)),((x-sz,y-sz),(x+sz,y-sz))]
            elif shape == 'diamond':
                to_match = [((x-sz,y),(x,y-sz)),((x+sz,y),(x,y-sz)),((x-sz,y),(x,y+sz)),((x+sz,y),(x,y+sz))]
            # test against all segments -- if one matches, return the intersection point
            _segment_ = ((x,y),(xp,yp))
            for shape_segment in to_match:
                inter_flag, x_inter, y_inter, t_s0, t_s1 = self.segmentsIntersect(_segment_, shape_segment)
                if inter_flag:
                    return x_inter, y_inter
        # give up and return the coordinates   
        return x,y

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
