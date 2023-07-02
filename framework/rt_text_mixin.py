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
import re

from rt_component import RTComponent

from shapely.geometry import Polygon

__name__ = 'rt_text_mixin'

#
# Abstraction for Text
#
class RTTextMixin(object):
    #
    # textBlock() - render a textblock and track positional information of characters and words.
    #
    def textBlock(self,
                  txt,
                  txt_h=14,
                  line_space_px=3,
                  word_wrap=False,
                  w=512,
                  x_ins=5,
                  y_ins=3):
        
        svg,x,y,line = '',x_ins,y_ins+txt_h,''
        orig_to_xy = {}
        last_was_space = True

        for i in range(0,len(txt)):
            c = txt[i]
            if c == '\n':
                # CODE BLOCK B
                svg           += self.svgText(line, x_ins, y, txt_h=txt_h)
                orig_to_xy[i]  = (x,y)
                x              = x_ins
                y             += txt_h + line_space_px
                line           = ''
                last_was_space = True
            elif word_wrap == False:
                # CODE BLOCK A
                line          += c
                orig_to_xy[i]  = (x,y)
                x             += self.textLength(c,txt_h)
            else:
                if last_was_space and self.__whitespace__(c) == False:
                    j,x_j = i+1,x+self.textLength(c,txt_h)
                    while j < len(txt) and self.__whitespace__(txt[j]) == False:
                        x_j += self.textLength(txt[j],txt_h)
                        j   += 1
                    if   x_j > (w-x_ins) and x != x_ins: # new word exceeds the maximum width / start new line
                        # CODE BLOCK B
                        svg           += self.svgText(line, x_ins, y, txt_h=txt_h)
                        orig_to_xy[i]  = (x,y)
                        x              = x_ins
                        y             += txt_h + line_space_px
                        line           = ''
                        last_was_space = True
                        # CODE BLOCK A
                        line          += c
                        orig_to_xy[i]  = (x,y)
                        x             += self.textLength(c,txt_h)
                    elif (x_j > (w-x_ins) and x == x_ins) or ((x_j - x) >= (w-x_ins)): # a chunk of text is too long to fit on a line
                        # CODE BLOCK B-mod
                        svg           += self.svgText(line, x_ins, y, txt_h=txt_h)
                        orig_to_xy[i]  = (x,y)
                        x              = x_ins
                        # y             += txt_h + line_space_px # MOD HERE
                        line           = ''
                        last_was_space = False # MOD HERE
                        # CODE BLOCK A
                        line          += c
                        orig_to_xy[i]  = (x,y)
                        x             += self.textLength(c,txt_h)                        
                    else:                        # fine to add the word
                        # CODE BLOCK A
                        line          += c
                        orig_to_xy[i]  = (x,y)
                        x             += self.textLength(c,txt_h)
                else:
                        # CODE BLOCK A
                        line          += c
                        orig_to_xy[i]  = (x,y)
                        x             += self.textLength(c,txt_h)
                if self.__whitespace__(c):
                    last_was_space = True
        # If there's a left over line, add it here...
        if len(line) > 0:
            svg  += self.svgText(line, x_ins, y, txt_h=txt_h)
            y    += txt_h + line_space_px

        # Calculate geom_to_word
        geom_to_word = {}
        i,last_was_space = 0,True
        while i < len(txt):
            if self.__whitespace__(txt[i]) or self.__punctuation__(txt[i]):
                last_was_space = True
                i += 1
            else:
                if last_was_space and self.__whitespace__(txt[i]) == False and self.__punctuation__(txt[i]) == False:
                    i0 = i
                    while i < len(txt)                           and \
                          self.__whitespace__ (txt[i])  == False and \
                          self.__punctuation__(txt[i])  == False:
                        i += 1
                        i1 = i
                    x0,y0 =  orig_to_xy[i0]
                    x1,y1 =  orig_to_xy[i1-1]
                    x1    += self.textLength(txt[i-1],txt_h)
                    _polygon = Polygon([[x0,y0+line_space_px], [x1,y1+line_space_px], [x1,y1-txt_h], [x0,y1-txt_h]])
                    geom_to_word[_polygon] = txt[i0:i1]
                last_was_space = False
                i = i1

        bounds = (0,0,w,y-txt_h+y_ins)
        return RTTextBlock(self, txt, svg, bounds, geom_to_word, orig_to_xy, txt_h, line_space_px)

    # Is character whitespace?
    def __whitespace__ (self, c):
        return c == ' ' or c == '\t' or c == '\n'
    
    # Is character punctuation?
    def __punctuation__(self, c):
        _str = '''!.?,[]{}:;`~%^&*()-_+='"<>/\\'''
        return c in _str
    
    #
    # joinLines() - join lines together and remove extra spaces.
    # - expect that this is a utility to call before textBlock()
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
    # joinNewLinesBetter() - keep newlines (if single) intact...
    # - more closely mirrors the (de facto) standard of using double line returns 
    #   to separate paragraphs
    # - expect that this is a utility to call before textBlock()
    #
    def joinNewLinesBetter(self, txt):
        re_match = re.findall(r'([\n]{2,})',txt)
        if re_match is None:
            return self.joinNewLines(txt)
        else:
            i,_ret = 0,''
            for _match in re_match:
                j = txt.find(_match,i)
                _ret += self.joinNewLines(txt[i:j])
                for k in range(len(_match)-1):
                    _ret += '\n'
                i = j+len(_match)
            _ret += self.joinNewLines(txt[i:])
            return _ret
    
    #
    # maxLinePixels() - split a string by new line characters, then determine
    # the maximum line length (in pixels).
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
                 svg,
                 bounds,
                 geom_to_word,
                 orig_to_xy,
                 txt_h,
                 line_space_px):
        self.rt_self       = rt_self
        self.txt           = txt
        self.svg           = svg
        self.bounds        = bounds
        self.geom_to_word  = geom_to_word
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
    
    #
    # Debugging Word Geometries
    #
    def __debug_svgOfWordColors__(self):
        svg_underlay = ''

        for _poly in self.geom_to_word:
            _word = self.geom_to_word[_poly]
            _co   = self.rt_self.co_mgr.getColor(_word)
            svg_underlay += f'<path d="{self.rt_self.shapelyPolygonToSVGPathDescription(_poly)}" fill="{_co}" />'

        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               svg_underlay + \
               self.svg     + \
               '</svg>'

