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
        
        svg,y,iword,ipoly,line_no,orig_to_xy,orig_i = '',txt_h+y_ins,{},{},0,{},-1
        orig_to_xy[0] = (x_ins,y)
        for _line in lines:
            words,x,word_no = _line.split(' '),x_ins,0
            for _word in words:
                svg += self.svgText(_word, x, y, txt_h)
                word_len_px = self.textLength(_word, txt_h)

                orig_i = txt.index(_word, orig_i+1)
                orig_to_xy[orig_i] = (x,y)
                orig_to_xy[orig_i + len(_word)] = (x+word_len_px,y)

                if len(_word) > 0: # Happens if multiple spaces occur together...
                    so_far = self.textLength(_word[0], txt_h)
                    for j in range(1,len(_word)):
                        orig_to_xy[orig_i+j] = (x+so_far,y)
                        so_far += self.textLength(_word[j], txt_h)

                _pos  = (x, y, word_len_px, txt_h, orig_i, line_no, word_no)
                iword[_pos] = _word

                _poly = Polygon([[x,y],[x+word_len_px,y],[x+word_len_px,y-txt_h],[x,y-txt_h]])
                ipoly[_pos] = _poly

                x += word_len_px + self.textLength(' ', txt_h)
                word_no += 1
            y += txt_h + line_space_px
            line_no += 1

        # Fill in missing originals...

        # Calculate the bounds
        bounds = (0,0,w,y-txt_h+y_ins)
        return RTTextBlock(self, txt, lines, svg, bounds, iword, ipoly, orig_to_xy, txt_h, line_space_px)

    #
    # joinLines() - join lines together and remove extra spaces.
    #
    def joinNewLines(self, txt):
        joined = ' '.join(txt.split('\n'))
        while len(joined) > 0 and joined[0] == ' ':
            joined = joined[1:]
        while len(joined) > 0 and joined[-1] == ' ':
            joined = joined[:-1]
        words = joined.split(' ')
        wout_blanks = []
        for word in words:
            if len(word) > 0:
                wout_blanks.append(word)
        return ' '.join(wout_blanks)
    
    #
    # maxLinePixels() - determine the max line length (in pixels)
    # - method first splits by newline character...
    #
    def maxLinePixels(self, txt, txt_h=14):
        _max = 0
        lines = txt.split('\n')
        for _line in lines:
            _len = self.textLength(_line, txt_h)
            _max = max(_len,_max)
        return _max + 6

#
# RTTextBlock - instance of rendered text block
#
class RTTextBlock(object):
    #
    # Constructor
    #
    def __init__(self,
                 rt_self,
                 txt,
                 lines,
                 svg,
                 bounds,
                 iword,
                 ipoly,
                 orig_to_xy,
                 txt_h,
                 line_space_px):
        self.rt_self       = rt_self
        self.txt           = txt
        self.lines         = lines
        self.svg           = svg
        self.bounds        = bounds
        self.pos_word      = iword
        self.pos_poly      = ipoly
        self.orig_to_xy    = orig_to_xy
        self.txt_h         = txt_h
        self.line_space_px = line_space_px
    
    #
    # SVG Representation -- adds the svg begin/end markup...
    #
    def _repr_svg_(self):
        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               self.svg + \
               '</svg>'
    
    #
    # Debugging Original Indices
    #
    def __debug_svgOfOverlayOriginalIndices__(self):
        svg_overlay = ''
        _co = self.rt_self.co_mgr.getTVColor('data','default')
        for i in self.orig_to_xy:
            x,y = self.orig_to_xy[i]
            svg_overlay += f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y-self.txt_h}" stroke="{_co}" />'

        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               self.svg + \
               svg_overlay + \
               '</svg>'
