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

import spacy
import nltk

import re

from rt_component import RTComponent

from shapely.geometry import Polygon, MultiPolygon

__name__ = 'rt_text_mixin'

#
# Abstraction for Text
#
class RTTextMixin(object):
    #
    # Constructor
    # 
    def __text_mixin_init__(self):
        self.spacy_loaded_flag = False

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
        _dn = 4 # downward shift...
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
                    _polygon = Polygon([[x0,y0+line_space_px+_dn], 
                                        [x1,y1+line_space_px+_dn], 
                                        [x1,y1-txt_h+_dn], 
                                        [x0,y1-txt_h+_dn]])
                    geom_to_word[_polygon] = txt[i0:i1]
                last_was_space = False
                i = i1

        bounds = (0,0,w,y-txt_h+y_ins)
        return RTTextBlock(self, txt, txt_h, line_space_px, word_wrap, w, x_ins, y_ins, svg, bounds, geom_to_word, orig_to_xy)

    # Is character whitespace?
    def __whitespace__ (self, c):
        return c == ' ' or c == '\t' or c == '\n'
    
    # Is character punctuation?
    def __punctuation__(self, c):
        _str = '''!.?,[]{}:;`~%^&*()-_+='"<>/\\'''
        return c in _str
    
    #
    # textJoinLines() - join lines together and remove extra spaces.
    # - expect that this is a utility to call before textBlock()
    #
    def textJoinNewLines(self, txt):
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
    # textJoinNewLinesBetter() - keep newlines (if single) intact...
    # - more closely mirrors the (de facto) standard of using double line returns 
    #   to separate paragraphs
    # - expect that this is a utility to call before textBlock()
    #
    def textJoinNewLinesBetter(self, txt):
        re_match = re.findall(r'([\n]{2,})',txt)
        if re_match is None:
            return self.joinNewLines(txt)
        else:
            i,_ret = 0,''
            for _match in re_match:
                j = txt.find(_match,i)
                _ret += self.textJoinNewLines(txt[i:j])
                for k in range(len(_match)-1):
                    _ret += '\n'
                i = j+len(_match)
            _ret += self.textJoinNewLines(txt[i:])
            return _ret
    
    #
    # maxLinePixels() - split a string by new line characters, then determine
    # the maximum line length (in pixels).
    #
    def textMaxLinePixels(self, txt, txt_h=14):
        _max = 0
        lines = txt.split('\n')
        for _line in lines:
            _len = self.textLength(_line, txt_h)
            _max = max(_len,_max)
        return _max + 6
    
    #
    # textExtractSentences() - extract sentences
    #
    def textExtractSentences(self,
                             txt):
        tokens,sentences = nltk.sent_tokenize(txt),[]
        if len(tokens) > 0:
            i = txt.index(tokens[0])-1
        for _token in tokens:
            i = txt.index(_token,i+1)
            sentences.append((_token, i, i + len(_token)))
            i += len(_token)
        return sentences

    #
    # textExtractEntities() - extract entities.
    #
    def textExtractEntities(self, 
                            txt, 
                            algo='spacy'):
        if algo == 'spacy':
            return self.__textExtractEntitiesSpacy__(txt)
        else:
            raise Exception(f'RACETrack.textExtractEntities() - unknown algorithm "{algo}"')

    #
    # __extractEntitiesSpacy__() - extract entities using SpaCy.
    #
    def __textExtractEntitiesSpacy__(self,txt):
        if self.spacy_loaded_flag == False:
            self.nlp_spacy = spacy.load('en_core_web_sm')
            self.spacy_loaded_flag = True
        doc = self.nlp_spacy(txt)
        ret = []
        for entity in doc.ents:
            ret.append((entity.text, entity.label_, entity.end_char - len(entity.text), entity.end_char))
        return ret

#
# RTTextBlock - instance of rendered text block
#
class RTTextBlock(object):
    #
    # Constructor
    #
    def __init__(self,
                 rt_self,           # Reference to parent class instance
                 txt,               # Original text string
                 txt_h,             # Text height in pixels
                 line_space_px,     # Pixel space between paragraphs
                 word_wrap,         # Word wrap flag
                 w,                 # Width of SVG results
                 x_ins,             # x insert left & right
                 y_ins,             # y insert top & bottom
                 svg,               # rendered svg (w/out svg begin/end wrapper)
                 bounds,            # Four tuple of x,y,w,h
                 geom_to_word,      # Shapely polygon to word
                 orig_to_xy):       # Original text index to xy-tuple
        self.rt_self        = rt_self
        self.txt            = txt
        self.txt_h          = txt_h
        self.line_space_px  = line_space_px
        self.word_wrap      = word_wrap
        self.w              = w
        self.x_ins          = x_ins
        self.y_ins          = y_ins
        self.svg            = svg
        self.bounds         = bounds
        self.geom_to_word   = geom_to_word
        self.orig_to_xy     = orig_to_xy
        
    #
    # spanGeometry() - return a polygon that covers a specified text span.
    #
    def spanGeometry(self, i, j):
        last_c = ' '
        if len(self.txt) > 0:
            last_c = self.txt[-1]

        if i >= len(self.txt):
            xy0    = self.orig_to_xy[len(self.txt)-1]
            xy0    = (xy0[0] + self.rt_self.textLength(last_c,self.txt_h),xy0[1])
        else:
            xy0    = self.orig_to_xy[i]
        if j >= len(self.txt):
            xy1    = self.orig_to_xy[len(self.txt)-1]
            xy1    = (xy1[0] + self.rt_self.textLength(last_c,self.txt_h),xy1[1])
        else:
            xy1    = self.orig_to_xy[j]

        _dn = 4 # downward shift...
        if     xy0[1]                                    == xy1[1]: # On same line...
            return Polygon([[xy0[0],xy0[1]+_dn],
                            [xy1[0],xy1[1]+_dn],
                            [xy1[0],xy1[1]-self.txt_h+_dn],
                            [xy0[0],xy0[1]-self.txt_h+_dn]
                            ])
        elif  (xy0[1] + self.txt_h + self.line_space_px) == xy1[1] and (xy1[0] < xy0[0]):
            _poly0 = Polygon([[xy0[0],              xy0[1]+_dn],
                              [self.w - self.x_ins, xy0[1]+_dn],
                              [self.w - self.x_ins, xy0[1]-self.txt_h+_dn],
                              [xy0[0],              xy0[1]-self.txt_h+_dn]])
            _poly1 = Polygon([[xy1[0],              xy1[1]+_dn],
                              [xy1[0],              xy1[1]-self.txt_h+_dn],
                              [self.x_ins,          xy1[1]-self.txt_h+_dn],
                              [self.x_ins,          xy1[1]+_dn]])
            return MultiPolygon([_poly0,_poly1])
        else: # Multiple lines...
            return Polygon([[xy0[0],              xy0[1]+_dn],
                            [self.x_ins,          xy0[1]+_dn],
                            [self.x_ins,          xy1[1]+_dn],
                            [xy1[0],              xy1[1]+_dn],
                            [xy1[0],              xy1[1]-self.txt_h+_dn],
                            [self.w - self.x_ins, xy1[1]-self.txt_h+_dn],
                            [self.w - self.x_ins, xy0[1]-self.txt_h+_dn],
                            [xy0[0],              xy0[1]-self.txt_h+_dn]
                            ])

    #
    # highlights() - highlight user-specified text.
    # - lu: either a [1] (i,j) tuple or [2] a regex string to either a [A] seven-character hex color string or a [B] string color
    #   - lu[(0,10)]            = '#ff0000'
    #   - lu['regex substring'] = '#000000'
    #   - lu['many']            = 'whatever' # any 'many' substrings will get colored with 'whatever' color lookup
    #
    def highlights(self, lu, opacity=1.0):
        svg_underlay = ''
        for k in lu:
            _co = lu[k]
            if _co.startswith('#') == False or len(_co) != 7: # If it's not a hex hash color string... then look it up...
                _co = self.rt_self.co_mgr.getColor(_co)
            if   type(k) == tuple:
                _poly = self.spanGeometry(k[0],k[1])
                svg_underlay += f'<path d="{self.rt_self.shapelyPolygonToSVGPathDescription(_poly)}" fill="{_co}" fill-opacity="{opacity}" />'
            elif type(k) == str:
                i = 0
                re_match = re.findall(k,self.txt)
                if re_match is not None:
                    i = 0
                    for _match in re_match:
                        i = self.txt.index(k,i)
                        j = i + len(_match)
                        _poly = self.spanGeometry(i,j)
                        svg_underlay += f'<path d="{self.rt_self.shapelyPolygonToSVGPathDescription(_poly)}" fill="{_co}" fill-opacity="{opacity}" />'
                        i += len(_match)
            else:
                raise Exception(f'RTTextBlock.highlights() - do not understand key value type {type(k)}')

        x,y,w,h = self.bounds
        _co     = self.rt_self.co_mgr.getTVColor('background','default')
        return f'<svg width="{w}" height="{h}">' + \
               f'<rect x="0" y="0" width="{w}" height="{h}" fill="{_co}" />' + \
               svg_underlay + \
               self.svg + \
               '</svg>'


    #
    # unwrappedSVG() - return the unwrapped version of the SVG.
    #
    def unwrappedSVG(self):
        return self.svg

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

