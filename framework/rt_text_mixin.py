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

import pandas as pd
import numpy as np

from rt_component import RTComponent

from shapely.geometry import Polygon

__name__ = 'rt_text_mixin'

#
# Abstraction for Text
#
class RTTextMixin(object):
    #
    # textBlock 
    # - render into an svg text block with looks up for text locations
    # - does not include the SVG wrapper
    #
    def textBlock(self, 
                  txt, 
                  txt_h=14,
                  line_space_px=3,
                  word_wrap=False,
                  w=512,
                  x_ins=5,
                  y_ins=3):
        # Break the text into lines
        lines = txt.split('\n')
        # Wrap the words if the option is set...
        if word_wrap:
            i = 0
            while i < len(lines):
                if self.textLength(lines[i], txt_h) > (w-2*x_ins):
                    words,j = lines[i].split(' '),0
                    first,finished = words[j],False
                    j += 1
                    while finished == False and \
                          j < len(words)    and \
                          self.textLength(first, txt_h) < (w-2*x_ins):
                        if self.textLength(first + ' ' + words[j],txt_h) > (w-2*x_ins):
                            finished = True
                        else:
                            first += ' ' + words[j]
                            j += 1
                    if j < len(words):
                        second = words[j]
                        j += 1
                        while j < len(words):
                            second += ' ' + words[j]
                            j += 1
                        lines.insert(i+1, second)
                    lines.insert(i+1, first)
                    if i == 0:
                        lines = lines[1:]
                    else:
                        lines = lines[:i] + lines[i+1:]
                i += 1

        # Remove Trailing and Leading blank lines...
        while len(lines) > 0 and lines[0] == '':
            lines = lines[1:]
        while len(lines) > 0 and lines[-1] == '':
            lines = lines[:-1]
        
        svg,y,iword,ipoly,line_no = '',txt_h+y_ins,{},{},0
        for _line in lines:
            words,x,word_no = _line.split(' '),x_ins,1
            for _word in words:
                svg += self.svgText(_word, x, y, txt_h)
                word_len = self.textLength(_word, txt_h)
                _poly = Polygon([[x,y],[x+word_len,y],[x+word_len,y-txt_h],[x,y-txt_h]])
                _pos  = (line_no,word_no)
                iword[_pos] = _word
                ipoly[_pos] = _poly
                x += word_len + self.textLength(' ', txt_h)
                word_no += 1
            y += txt_h + line_space_px
            line_no += 1
        bounds = (0,0,w,y-txt_h+y_ins)

        return RTTextBlock(self, txt, lines, svg, bounds, iword, ipoly, txt_h, line_space_px)

#
# RTTextBlock - instance of rendered text block
#
class RTTextBlock(object):
    def __init__(self,
                 rt_self,
                 txt,
                 lines,
                 svg,
                 bounds,
                 iword,
                 ipoly,
                 txt_h,
                 line_space_px):
        self.rt_self       = rt_self
        self.txt           = txt
        self.lines         = lines
        self.svg           = svg
        self.bounds        = bounds
        self.pos_word      = iword
        self.pos_poly      = ipoly
        self.txt_h         = txt_h
        self.line_space_px = line_space_px
    
    #
    #
    #
    def _svg_repr_(self):
        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               self.svg + \
               '</svg>'
