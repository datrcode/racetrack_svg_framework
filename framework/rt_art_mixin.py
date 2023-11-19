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

__name__ = 'rt_art_mixin'

#
# Art Mixin
# - Icons & Other Types Of Art
# - Standardized on 100 x 100 in most cases
#
class RTArtMixin(object):
    #
    # iconLinkNode()
    #
    def iconLinkNode(self, x=0, y=0, w=200, h=200, node_fill='#ffffff', fg='#000000', bg='none', stroke_width=3, x_ins=10, y_ins=10):
        border_stroke_width = max(stroke_width - 1, 1)
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="{-x_ins} {-y_ins} {100+x_ins} {100+y_ins}" xmlns="http://www.w3.org/2000/svg">' + \
               f'<path d="M 10 0 L  90   0  C   95   0   100   5   100   10' + \
               f'                L 100  90  C  100  95    95 100    90  100' + \
               f'                L  10 100  C    5 100     0  95     0   90' + \
               f'                L  0   10  C    0   5     5   0    10    0" fill="{bg}" stroke="{fg}" stroke-width="{border_stroke_width}" />' + \
               f'<line x1="40"  y1="0"    x2="50" y2="30" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="70"  y1="0"    x2="50" y2="30" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="0"   y1="90"   x2="30" y2="70" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="0"   y1="70"   x2="30" y2="70" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="0"   y1="50"   x2="30" y2="70" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="100" y1="60"   x2="70" y2="70" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="80"  y1="100"  x2="70" y2="70" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="30"  y1="70"   x2="50" y2="30" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="70"  y1="70"   x2="50" y2="30" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<line x1="70"  y1="70"   x2="30" y2="70" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<circle cx="50" cy="30" r="10" stroke="{fg}" stroke-width="{stroke_width}" fill="{node_fill}" />' + \
               f'<circle cx="30" cy="70" r="10" stroke="{fg}" stroke-width="{stroke_width}" fill="{node_fill}" />' + \
               f'<circle cx="70" cy="70" r="10" stroke="{fg}" stroke-width="{stroke_width}" fill="{node_fill}" />' + \
               f'</svg>'

    #
    # iconSetCurrentAsRoot()
    #
    def iconSetCurrentAsRoot(self, x=0, y=0, w=200, h=200, node_fill='#808080', fg='#000000', bg='none', arrow_stroke_width=5, stroke_width=3, x_ins=10, y_ins=10):
        border_stroke_width = max(stroke_width - 1, 1)
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="{-x_ins} {-y_ins} {100+x_ins} {100+y_ins}" xmlns="http://www.w3.org/2000/svg">' + \
               f'<path d="M 10 0 L  90   0   C   95   0   100   5   100   10' + \
               f'                L 100  90   C  100  95    95 100    90  100' + \
               f'                L  10 100   C    5 100     0  95     0   90' + \
               f'                L   0  10   C    0   5     5   0    10    0" fill="{bg}" stroke="{fg}" stroke-width="{border_stroke_width}" />' + \
               f'<rect x="10" y="40" width="20" height="20" stroke="{fg}" stroke-width="{stroke_width}" fill="{node_fill}" />' + \
               f'<rect x="40" y="40" width="20" height="20" stroke="{fg}" stroke-width="{stroke_width}" fill="{node_fill}" />' + \
               f'<rect x="70" y="40" width="20" height="20" stroke="{fg}" stroke-width="{stroke_width}" fill="{bg}" />' + \
               f'<path d="M 60 35 C 50 10 20 10 20 35 L 12 20 L 20 35 L 35 25" stroke="{fg}" stroke-width="{arrow_stroke_width}" fill="{bg}" />' + \
               f'</svg>'

    #
    # iconCurrentMinusRoot()
    #
    def iconCurrentMinusRoot(self, x=0, y=0, w=200, h=200, node_fill='#808080', deleted_node_fill='#ffa0a0', arrow_fg='#ff0000', 
                             fg='#000000', bg='none', arrow_stroke_width=5, stroke_width=3, x_ins=10, y_ins=10):
        border_stroke_width = max(stroke_width - 1, 1)
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="{-x_ins} {-y_ins} {100+x_ins} {100+y_ins}" xmlns="http://www.w3.org/2000/svg">' + \
               f'<path d="M 10 0 L  90   0  C   95   0   100   5   100   10' + \
               f'                L 100  90  C  100  95    95 100    90  100' + \
               f'                L  10 100  C    5 100     0  95     0   90' + \
               f'                L  0  10   C    0   5     5   0    10    0" fill="{bg}" stroke="{fg}" stroke-width="{border_stroke_width}" />' + \
               f'<path d="M 10 40 L 30 40 L 30 50 L 20 50 L 20 60 L 10 60 Z" stroke="{fg}" stroke-width="{stroke_width}" fill="{node_fill}" />' + \
               f'<rect x="40" y="40" width="20" height="20" stroke="{arrow_fg}" stroke-width="{stroke_width}" fill="{deleted_node_fill}" />' + \
               f'<rect x="20" y="50" width="11" height="11" stroke="{arrow_fg}" stroke-width="2" fill="{deleted_node_fill}">' + \
               f'     <animate attributeName="x" values="20;24;20" dur="4s" repeatCount="indefinite" />' + \
               f'     <animate attributeName="y" values="50;54;50" dur="4s" repeatCount="indefinite" />' + \
               f' </rect>' + \
               f' <rect x="70" y="40" width="20" height="20" stroke="{fg}" stroke-width="{stroke_width}" fill="{bg}" />' + \
               f' <path d="M 60 35 C 50 10 20 10 20 35 L 12 20 L 20 35 L 35 25" stroke="{arrow_fg}" stroke-width="{arrow_stroke_width}" fill="{bg}" />' + \
               f' </svg>'
    