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

import random

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
    # iconHistogram()
    #
    def iconHistogram(self, x=0, y=0, w=200, h=200, fill='#a0a0a0', fg='#000000', bg='none', stroke_width=3, x_ins=10, y_ins=10):
         return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="{-x_ins} {-y_ins} {100+x_ins} {100+y_ins}" xmlns="http://www.w3.org/2000/svg">' + \
                f'<path d="M 10 0 L  90   0  C   95   0   100   5   100   10' + \
                f'                L 100  90  C  100  95    95 100    90  100' + \
                f'                L  10 100  C    5 100     0  95     0   90' + \
                f'                L  0   10  C    0   5     5   0    10    0" fill="{bg}" stroke="{fg}" stroke-width="{stroke_width}" />' + \
                f'<rect x="15" y="5"  width="70" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'<rect x="15" y="15" width="65" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'<rect x="15" y="25" width="55" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'<rect x="15" y="35" width="30" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'<rect x="15" y="45" width="30" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'<rect x="15" y="55" width="20" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'<rect x="15" y="65" width="15" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'<rect x="15" y="75" width="10" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'<rect x="15" y="85" width="10" height="10" stroke="{fg}" fill="{fill}" />' + \
                f'</svg>'

    #
    # iconTemporalBarChart()
    #
    def iconTemporalBarChart(self, x=0, y=0, w=200, h=200, fill='#a0a0a0', fg='#000000', bg='none', stroke_width=3, x_ins=10, y_ins=10):
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="{-x_ins} {-y_ins} {100+x_ins} {100+y_ins}" xmlns="http://www.w3.org/2000/svg">' + \
               f'<path d="M 10 0 L  90   0  C   95   0   100   5   100   10' + \
               f'                L 100  90  C  100  95    95 100    90  100' + \
               f'                L  10 100  C    5 100     0  95     0   90' + \
               f'                L  0   10  C    0   5     5   0    10    0" fill="{bg}" stroke="{fg}" stroke-width="{stroke_width}" />' + \
               f'<rect x="12" y="50"  width="10" height="30" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="22" y="45"  width="10" height="35" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="32" y="40"  width="10" height="40" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="42" y="45"  width="10" height="35" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="52" y="50"  width="10" height="30" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="62" y="55"  width="10" height="25" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="72" y="60"  width="10" height="20" stroke="{fg}" fill="{fill}" />' + \
               f'<path d="M 10 80 L 90 80 L 80 85" fill="none" stroke="{fg}"/>' + \
               f'<path d="M 10 80 L 10 10 L 5 20"  fill="none" stroke="{fg}"/>' + \
               f'<text x="50" y="95" font-family="times" font-size="12" text-anchor="middle">Time</text>' + \
               f'</svg>'

    #
    # iconPeriodicBarChart()
    #
    def iconPeriodicBarChart(self, x=0, y=0, w=200, h=200, fill='#a0a0a0', fg='#000000', bg='none', stroke_width=3, x_ins=10, y_ins=10):
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="{-x_ins} {-y_ins} {100+x_ins} {100+y_ins}" xmlns="http://www.w3.org/2000/svg">' + \
               f'<path d="M 10 0 L  90   0  C   95   0   100   5   100   10' + \
               f'         L 100  90  C  100  95    95 100    90  100' + \
               f'         L  10 100  C    5 100     0  95     0   90' + \
               f'         L  0   10  C    0   5     5   0    10    0" fill="{bg}" stroke="{fg}" stroke-width="{stroke_width}" />' + \
               f'<rect x="12" y="50"  width="10" height="30" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="22" y="45"  width="10" height="35" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="32" y="40"  width="10" height="40" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="42" y="45"  width="10" height="35" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="52" y="50"  width="10" height="30" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="62" y="55"  width="10" height="25" stroke="{fg}" fill="{fill}" />' + \
               f'<rect x="72" y="60"  width="10" height="20" stroke="{fg}" fill="{fill}" />' + \
               f'<path d="M 10 80 L 85 80" fill="{bg}" stroke="{fg}"/>' + \
               f'<path d="M 10 80 L 10 30 L 5 40" fill="{bg}" stroke="{fg}"/>' + \
               f'<path d="M 70 50 A 20 20 0 0 0 30 50 M 70 50 A 20 20 0 0 1 30 50" fill="{bg}" stroke="{fg}" stroke-width="{stroke_width+5}"/>' + \
               f'<path d="M 58 30 l 20 5 l -20 15 Z" fill="{fg}"/>' + \
               f'<text x="50" y="95" font-family="times" font-size="12" text-anchor="middle">Time</text>' + \
               f'</svg>'


    #
    # iconPieChart()
    #
    def iconPieChart(self, x=0, y=0, w=200, h=200, fill_a='#a0a0a0', fill_b='#404040', fg='#000000', bg='none', stroke_width=3, x_ins=10, y_ins=10):
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="{-x_ins} {-y_ins} {100+x_ins} {100+y_ins}" xmlns="http://www.w3.org/2000/svg">' + \
               f'<path d="M 10 0 L  90   0  C   95   0   100   5   100   10' + \
               f'         L 100  90  C  100  95    95 100    90  100' + \
               f'         L  10 100  C    5 100     0  95     0   90' + \
               f'         L  0   10  C    0   5     5   0    10    0" fill="{bg}" stroke="{fg}" stroke-width="{stroke_width}" />' + \
               f'<path d="M 90 50 A 20 20 0 0 0 10 50 M 90 50 A 20 20 0 0 1 10 50" fill="{bg}" stroke="{fg}" stroke-width="{stroke_width+2}"/>' + \
               f'<path d="M 50 50 l 0  -40 a 40 40 0 0 10 80 Z" fill="{fill_a}" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
               f'<path d="M 50 50 l 35 -20 a 40 40 0 0 10 40 Z" fill="{fill_b}" stroke="{fg}" stroke-width="{stroke_width}"/>' + \
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
    
    #
    # iconResetView()
    #
    def iconResetView(self, x=0, y=0, w=200, h=200, fg='#000000', bg='none', stroke_width=3, x_ins=10, y_ins=10):
        border_stroke_width = max(stroke_width - 1, 1)
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="{-x_ins} {-y_ins} {100+x_ins} {100+y_ins}" xmlns="http://www.w3.org/2000/svg">' + \
	        f'<path d="M  10   0 L  90   0   C   95    0   100   5   100   10 ' + \
               f'         L 100  90 C 100  95  95  100   90   100 ' + \
               f'         L  10 100 C   5 100   0   95    0    90  ' + \
               f'         L   0  10 C   0   5   5    0   10     0" fill="{bg}" stroke="{fg}" stroke-width="{stroke_width}" /> ' + \
               f'<path d="M 35 10 L 10 10 L 10 35 M 10 10 L 40 40" stroke="{fg}" stroke-width="{stroke_width+1}" fill="none"/> ' + \
               f'<path d="M 65 90 L 90 90 L 90 65 M 90 90 L 60 60" stroke="{fg}" stroke-width="{stroke_width+1}" fill="none"/> ' + \
               f'<path d="M 35 90 L 10 90 L 10 65 M 10 90 L 40 60" stroke="{fg}" stroke-width="{stroke_width+1}" fill="none"/> ' + \
               f'<path d="M 65 10 L 90 10 L 90 35 M 90 10 L 60 40" stroke="{fg}" stroke-width="{stroke_width+1}" fill="none"/> ' + \
               f'</svg>'
    
    #
    # iconColorBy()
    #
    def iconColorBy(self, x=0, y=0, w=100, h=50, bg='none'):
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="30 -5 10 50" xmlns="http://www.w3.org/2000/svg">' + \
	        f'<path d="M 10 0  L 65 0 C 70 0 75 5 75 10  L 75 25 C 75 30 70 35 65 35  L 10 35 C 5 35 0 30 0 25  L 0 10 C 0 5 5 0 10 0"' + \
               f' fill="{bg}" stroke="#000000" stroke-width="3" />' + \
               f'<text x="10" y="28" font-size="28px" stroke="#ff0000" fill="#ff0000">R</text>' + \
               f'<text x="28" y="28" font-size="28px" stroke="#00a000" fill="#00a000">G</text>' + \
               f'<text x="48" y="28" font-size="28px" stroke="#0000ff" fill="#0000ff">B</text>' + \
               f'</svg>'
    
    #
    # iconCountBy()
    #
    def iconCountBy(self, x=0, y=0, w=100, h=50, bg='none'):
        return f'<svg x="{x}" y="{y}" width="{w}" height="{h}" viewbox="30 -5 10 50" xmlns="http://www.w3.org/2000/svg">' + \
	        f'<path d="M 10 0  L 65 0 C 70 0 75 5 75 10  L 75 25 C 75 30 70 35 65 35  L 10 35 C 5 35 0 30 0 25  L 0 10 C 0 5 5 0 10 0"' + \
               f' fill="{bg}" stroke="#000000" stroke-width="3" />' + \
               f'<text x="10" y="28" font-size="28px" stroke="#000000" fill="#202020">1</text>' + \
               f'<text x="28" y="28" font-size="28px" stroke="#000000" fill="#202020">2</text>' + \
               f'<text x="48" y="28" font-size="28px" stroke="#000000" fill="#202020">3</text>' + \
               f'</svg>'

    #
    # iconCloud()
    #
    # Source: https://www.svgrepo.com/svg/520637/cloud
    # License:  CC Attribution License
    # COLLECTION: Xnix Circular Interface Icons
    # AUTHOR: Ankush Syal
    # Modified To Remove Bottom Path (Second Path)
    # -- will be further modified for scaling purposes
    # -- i don't think the width parameter changes anything... only the height seems to be of use...
    #
    def iconCloud(self, x=0, y=0, w=100, h=50, fg='#000000', bg='#ffffff', id=None):
        if id is None: id = f'cloudicon-{random.randint(2**10,2**30)}'
        #return f'<g id="{id}" transform="translate({-w/2},{-h/2})">' + \
        #       f'<svg x="{x}" y="{y}" width="{w}px" height="{h}px" viewBox="0 -0.5 25 25" fill="none" xmlns="http://www.w3.org/2000/svg">' + \
        #       f'<path fill-rule="evenodd" clip-rule="evenodd" d="M14.091 7.00151C14.9928 6.9746 15.8684 7.30725 16.5249 7.9262C17.1813 8.54515 17.5649 9.39965 17.591 10.3015C17.5914 10.6221 17.5425 10.9408 17.446 11.2465C18.6091 11.4334 19.4729 12.4239 19.5 13.6015C19.4586 14.9664 18.32 16.0402 16.955 16.0015H8.045C6.67999 16.0402 5.54137 14.9664 5.5 13.6015C5.52293 12.4783 6.31258 11.5171 7.41 11.2765C7.41 11.2512 7.41 11.2262 7.41 11.2015C7.45137 9.83659 8.58999 8.76283 9.955 8.80151C10.2738 8.80108 10.5901 8.85764 10.889 8.96851C11.4867 7.74927 12.7333 6.98347 14.091 7.00151Z" fill="{bg}" stroke="{fg}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' + \
        #        '</svg></g>'
        return f'<g id="{id}" transform="translate({-w/2},{-h/2})">' + \
               f'<svg x="{x}" y="{y}" width="{w}px" height="{h}px" viewBox="-5 -5.5 35 35" fill="none" xmlns="http://www.w3.org/2000/svg">' + \
               f'<path fill-rule="evenodd" clip-rule="evenodd" d="M14.091 7.00151C14.9928 6.9746 15.8684 7.30725 16.5249 7.9262C17.1813 8.54515 17.5649 9.39965 17.591 10.3015C17.5914 10.6221 17.5425 10.9408 17.446 11.2465C18.6091 11.4334 19.4729 12.4239 19.5 13.6015C19.4586 14.9664 18.32 16.0402 16.955 16.0015H8.045C6.67999 16.0402 5.54137 14.9664 5.5 13.6015C5.52293 12.4783 6.31258 11.5171 7.41 11.2765C7.41 11.2512 7.41 11.2262 7.41 11.2015C7.45137 9.83659 8.58999 8.76283 9.955 8.80151C10.2738 8.80108 10.5901 8.85764 10.889 8.96851C11.4867 7.74927 12.7333 6.98347 14.091 7.00151Z" fill="{bg}" stroke="{fg}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>' + \
                '</svg></g>'


align_view_to_selection = '''

<svg x="0" y="0" width="600" height="600" viewbox="-50 -50 200 200" xmlns="http://www.w3.org/2000/svg">
<line x1="40"  y1="0"    x2="50" y2="30" stroke="#a0a0a0" stroke-width="4"/>
<line x1="70"  y1="0"    x2="50" y2="30" stroke="#a0a0a0" stroke-width="4"/>
<line x1="0"   y1="90"   x2="30" y2="70" stroke="#a0a0a0" stroke-width="4"/>
<line x1="0"   y1="70"   x2="30" y2="70" stroke="#a0a0a0" stroke-width="4"/>
<line x1="0"   y1="50"   x2="30" y2="70" stroke="#a0a0a0" stroke-width="4"/>
<line x1="100" y1="60"   x2="70" y2="70" stroke="#a0a0a0" stroke-width="4"/>
<line x1="80"  y1="100"  x2="70" y2="70" stroke="#a0a0a0" stroke-width="4"/>
<line x1="30"  y1="70"   x2="50" y2="30" stroke="#a0a0a0" stroke-width="4"/>
<line x1="70"  y1="70"   x2="50" y2="30" stroke="#a0a0a0" stroke-width="4"/>
<line x1="70"  y1="70"   x2="30" y2="70" stroke="#a0a0a0" stroke-width="4"/>
<circle cx="50" cy="30" r="10" stroke="#000000" stroke-width="4" fill="#ff0000" />
<circle cx="30" cy="70" r="10" stroke="#000000" stroke-width="4" fill="#ff0000" />
<circle cx="70" cy="70" r="10" stroke="#000000" stroke-width="4" fill="#ffffff" />
<path d="M 15 12 L 65 12 C 70 12 70 12 70 20  
                 L 70 77 C 70 87 70 87 65 87       
                 L 15 87 C 10 87 10 87 10 77
                 L 10 20 C 10 12 10 12 15 12
         " fill="none" stroke="#000000" stroke-width="2.0" />
<path d="M 10 0 L  90   0  C   95   0   100   5   100   10
                L 100  90  C  100  95    95 100    90  100
                L  10 100  C    5 100     0  95     0   90
                L  0   10  C    0   5     5   0    10    0" fill="none" stroke="#000000" stroke-width="4.0" />
</svg>

'''

