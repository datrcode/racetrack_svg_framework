# Copyright 2022 David Trimm
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

##############################################################################################################
#
# NCL / NCAR Command Language Color Schema From:
#
# https://www.ncl.ucar.edu/Document/Graphics/ColorTables/cividis.shtml
#
# From this page, placed the dividis.rgb file in the config directory
#

##############################################################################################################
#
# Niccoli & Lynch Color Scale From:
# "A More Perceptual Color Palette for Structure Maps" (2014), Niccoli and Lynch
#
# https://www.searchanddiscovery.com/pdfz/documents/2014/41297niccoli/ndx_niccoli.pdf.html
#
# Need to find a Python implementation of this schema...
#

import colorsys

from math import log10

__name__ = 'rt_color_manager'

#
# Abstraction for colorscale // by default this is the light theme
#
class RTColorManager:
    #
    # Initialize the type value colors
    #
    def __init__(self, racetrack):
        self.str_to_color_lu = {}
        self.type_color_lu   = {}
        self.racetrack       = racetrack

        self.highlights_flag = False
        self.highlights_lu   = {}

        self.type_color_lu['label'] = {}
        self.type_color_lu['label']['defaultfg']    = '#000000'
        self.type_color_lu['label']['defaultbg']    = '#ffffff'
        self.type_color_lu['label']['error']        = '#ff0000'

        self.type_color_lu['axis'] = {}
        self.type_color_lu['axis']['default']       = '#101010'
        self.type_color_lu['axis']['major']         = '#909090'
        self.type_color_lu['axis']['minor']         = '#c0c0c0'

        self.type_color_lu['border'] = {}
        self.type_color_lu['border']['default']     = '#000000'        

        self.type_color_lu['background'] = {}
        self.type_color_lu['background']['default'] = '#ffffff'        

        self.type_color_lu['data'] = {}
        self.type_color_lu['data']['default']        = '#4988b6'
        self.type_color_lu['data']['default_border'] = '#2f54d0'
        self.type_color_lu['data']['alternate']      = '#1fd655' # only for the y-axis distribution in the xy plot...
        
        self.type_color_lu['context'] = {}
        self.type_color_lu['context']['default']    = '#eeeeee'
        self.type_color_lu['context']['text']       = '#808080'
        
        # Fix Some specific colors // mostly just for testing...
        self.str_to_color_lu['blue']   = '#0000ff'
        self.str_to_color_lu['red']    = '#ff0000'
        self.str_to_color_lu['green']  = '#00ff00'
        self.str_to_color_lu['yellow'] = '#f7dc6f'
        self.str_to_color_lu['brown']  = '#795548'
        self.str_to_color_lu['pink']   = '#f5b7b1'
        self.str_to_color_lu['orange'] = '#e59866'
        self.str_to_color_lu['white']  = '#808080'  # Really, just gray... but has to separate from the background
        self.str_to_color_lu['black']  = '#000000' 

        # ColorBrewer2.org
        # - https://colorbrewer2.org/#
        # - © Cynthia Brewer, Mark Harrower and The Pennsylvania State University
        
        # Days of the Week
        # https://colorbrewer2.org/?type=sequential&scheme=YlOrBr&n=7
        my_strs   = ['Sun',    'Mon',    'Tue',    'Wed',    'Thu',    'Fri',    'Sat']
        my_colors = ['#ffffd4','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#8c2d04']
        for i in range(0,len(my_strs)):
            self.str_to_color_lu[my_strs[i]] = my_colors[i]

        # Months of the Year
        # https://colorbrewer2.org/?type=qualitative&scheme=Set3&n=12
        my_strs   = ['Jan',    'Feb',    'Mar',    'Apr',    'May',    'Jun',    'Jul',    'Aug',    'Sep',    'Oct',    'Nov',    'Dec']
        my_colors = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
        for i in range(0,len(my_strs)):
            self.str_to_color_lu[my_strs[i]] = my_colors[i]
        
        # Hour of the Day ... it's 9 (but we'll remove the first since that's almost white)
        # https://colorbrewer2.org/?type=sequential&scheme=Blues&n=9
        my_strs   = ['ignore', '00',     '01',     '02',     '03',     '04',     '05',     '06',     '07']
        my_colors = ['#f7fbff','#deebf7','#c6dbef','#9ecae1','#6baed6','#4292c6','#2171b5','#08519c','#08306b']
        for i in range(0,len(my_strs)):
            self.str_to_color_lu[my_strs[i]] = my_colors[i]
                     
        # https://colorbrewer2.org/?type=sequential&scheme=Greens&n=9
        my_strs   = ['ignore', '08',     '09',     '10',     '11',     '12',     '13',     '14',     '15']
        my_colors = ['#f7fcf5','#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d','#238b45','#006d2c','#00441b']
        for i in range(0,len(my_strs)):
            self.str_to_color_lu[my_strs[i]] = my_colors[i]
                     
        # https://colorbrewer2.org/?type=sequential&scheme=Oranges&n=9
        my_strs   = ['ignore', '16',     '17',     '18',     '19',     '20',     '21',     '22',     '23']
        my_colors = ['#fff5eb','#fee6ce','#fdd0a2','#fdae6b','#fd8d3c','#f16913','#d94801','#a63603','#7f2704']
        for i in range(0,len(my_strs)):
            self.str_to_color_lu[my_strs[i]] = my_colors[i]

        # log bins // strings are from the logBins transform strings
        # colorbrewer2.org // sequential of 7 (yellows -> reds), then two sequential blues for zero and less than zero
        my_strs   = ['< 0',     '= 0',     '<= 1',   '<= 10',   '<= 100',  '<= 1K',   '<= 100K',  '<= 1M',   '> 1M']
        my_colors = ['#2c7fb8', '#7fcdbb', '#ffffd4','#fee391', '#fec44f', '#fe9929', '#ec7014',  '#cc4c02', '#8c2d04']
        for i in range(0,len(my_strs)):
            self.str_to_color_lu[my_strs[i]] = my_colors[i]

        # True / False Colors
        self.str_to_color_lu[False] = '#ff0000'
        self.str_to_color_lu[True]  = '#0000ff'

    #
    # __allhex__() - are all the character hex characters?
    #
    def __allhex__(self,s):
        for c in s:
            if (c >= 'a' and c <= 'f') or \
               (c >= 'A' and c <= 'F') or \
               (c >= '0' and c <= '9'):
                pass
            else:
                return False
        return True

    #
    # Return a color for a string in "#ff00ff" format // default for SVG
    #
    def getColor(self,s):
        # Enable highlights
        if self.highlights_flag:
            if s in self.highlights_lu.keys():
                return self.highlights_lu[s]
            else:
                return self.getTVColor('data','default')

        # Default method
        else:
            if s not in self.str_to_color_lu.keys():
                if len(s) == 7 and s[0] == '#' and self.__allhex__(s[1:]):
                    self.str_to_color_lu[s] = s
                else:    
                    hc = self.racetrack.hashcode(s)

                    # Updated Mixtures // 2022-11-27
                    h  =             ((hc>>16)&0x00ffff)/65535.0
                    s  = 0.2 + 0.8 * ((hc>> 8)&0x0000ff)/255.0
                    v  = 0.6 + 0.4 * ((hc>> 0)&0x0000ff)/255.0
                                
                    (r,g,b) = colorsys.hsv_to_rgb(h,s,v)
                    rgb = ((int(r*255)&0x00ff)<<16) | ((int(g*255)&0x00ff)<<8) | ((int(b*255)&0x00ff)<<0)
                    as_hex = format(rgb,'x')
                    self.str_to_color_lu[s] = '#' + ('0' * (6 - len(as_hex)) + as_hex)
            return self.str_to_color_lu[s]
    
    #
    # Return type, value color string in the "#ff00ff" format
    #
    def getTVColor(self,t,v):
        return self.type_color_lu[t][v]


    #
    # Return spectrum color
    #
    def spectrumAbridged(self, _v, _min, _max, _magnitude='linear'):
        _base  = 0.3
        _range = 0.7
        if _magnitude == 'linear':
            h = _base + _range * (_v - _min)/(_max - _min)
        else:
            _norm   = _v   - _min + 1
            _delta  = _max - _min + 1
            if _norm == 0:
                _norm  += 1
                _delta += 1
            h = _base + _range * log10(_norm)/log10(_delta)
        s = v = 1.0
        (r,g,b) = colorsys.hsv_to_rgb(h,s,v)
        rgb = ((int(r*255)&0x00ff)<<16) | ((int(g*255)&0x00ff)<<8) | ((int(b*255)&0x00ff)<<0)
        as_hex = format(rgb,'x')
        _str = '#' + ('0' * (6 - len(as_hex)) + as_hex)
        return _str


    #
    # Based on BrewerColors // more reasonable for interpretation
    #
    def spectrum(self, _v, _min, _max, _linear=True):
        # _co = ['#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
        # _co = ['#ccebc5','#a8ddb5','#7bccc4','#4eb3d3','#2b8cbe','#0868ac','#084081']
        _co = ['#a8ddb5','#7bccc4','#0868ac','#084081', '#ffa500', '#ff0000']

        if _linear:
            h = (_v - _min)/(_max - _min)
        else:
            _norm   = _v   - _min + 1
            _delta  = _max - _min + 1
            if _norm == 0:
                _norm  += 1
                _delta += 1
            h = log10(_norm)/log10(_delta)

        i0 = int(h*(len(_co)-1))
        i1 = i0+1

        if i1 >= len(_co):
            i1 = i0

        (r0,g0,b0) = self.hashColorToRGB(_co[i0])
        (r1,g1,b1) = self.hashColorToRGB(_co[i1])

        perc = h*(len(_co)-1) - i0

        r = r0 * (1.0 - perc) + r1 * (perc)
        g = g0 * (1.0 - perc) + g1 * (perc)
        b = b0 * (1.0 - perc) + b1 * (perc)

        rgb = ((int(r*255)&0x00ff)<<16) | ((int(g*255)&0x00ff)<<8) | ((int(b*255)&0x00ff)<<0)
        as_hex = format(rgb,'x')
        _str = '#' + ('0' * (6 - len(as_hex)) + as_hex)
        return _str

    #
    # Convert a hash RGB string into the three rgb components
    #
    def hashColorToRGB(self, hc):
        return int('0' + hc[1:3],16)/255,int('0' + hc[3:5],16)/255,int('0' + hc[5:7],16)/255

    #
    # Enable Highlight Coloring
    #
    def enableHighlights(self,
                        entities,
                        scale_type='qualitative'):
        self.highlights_lu   = {}
        colors = self.brewerColors(scale_type,len(entities))
        for i in range(0,len(entities)):
            self.highlights_lu[entities[i]] = colors[i]
        self.highlights_flag = True

    #
    # Disable Highlight Coloring
    #
    def disableHighlights(self):
        self.highlights_flag = False

    #
    # Optimize Categorical Colors (up to 12)
    #
    def optimizeCategoricalColors(self, cats):
        cat_ordered = sorted(list(set(cats))) # want it to be the same
        hc = self.racetrack.hashcode('|'.join(cat_ordered)) & 0x00ffff
        cs = self.brewerColors('qualitative',len(cats),hc)
        for i in range(0,len(cat_ordered)):
            self.str_to_color_lu[cat_ordered[i]] = cs[i]

    #
    # brewerColors
    # - https://colorbrewer2.org/#
    # - © Cynthia Brewer, Mark Harrower and The Pennsylvania State University
    #
    def brewerColors(self,
                     scale_type, # 'sequential, 'diverging', or 'qualitative'
                     n,          # number of colors needed
                     alt=0):     # alternative versions
        if   scale_type == 'sequential':
            sequential_5      = {}
            sequential_5[0]   = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
            sequential_5[1]   = ['#feebe2','#fbb4b9','#f768a1','#c51b8a','#7a0177']
            sequential_9      = {}
            sequential_9[0]   = ['#ffffcc','#ffeda0','#fed976','#feb24c','#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026']
            sequential_9[1]   = ['#ffffd9','#edf8b1','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#253494','#081d58']
            if   n == 5:
                return sequential_5[alt%2]
            elif n == 9:
                return sequential_9[alt%2]
            else:
                raise Exception(f'brewerColors() - n={n} not supported for sequential - try 5 or 9')
        elif scale_type == 'diverging':
            diverging_11      = {}
            diverging_11[0]   = ['#67001f','#b2182b','#d6604d','#f4a582','#fddbc7','#f7f7f7','#d1e5f0','#92c5de','#4393c3','#2166ac','#053061']
            diverging_11[1]   = ['#543005','#8c510a','#bf812d','#dfc27d','#f6e8c3','#f5f5f5','#c7eae5','#80cdc1','#35978f','#01665e','#003c30']
            if n == 11:
                return diverging_11[alt%2]
            else:
                raise Exception(f'brewerColors() - n={n} not supported for diverging - try 11.')
        elif scale_type == 'qualitative':
            # at 8, brewercolors has the full range of qualitative scales
            qualitative_8     = {}
            qualitative_8[0]  = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5']
            qualitative_8[1]  = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f','#e5c494','#b3b3b3']
            qualitative_8[2]  = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00','#ffff33','#a65628','#f781bf']
            qualitative_8[3]  = ['#b3e2cd','#fdcdac','#cbd5e8','#f4cae4','#e6f5c9','#fff2ae','#f1e2cc','#cccccc'] # pastels
            qualitative_8[4]  = ['#fbb4ae','#b3cde3','#ccebc5','#decbe4','#fed9a6','#ffffcc','#e5d8bd','#fddaec'] # pastels
            qualitative_8[5]  = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00']
            qualitative_8[6]  = ['#1b9e77','#d95f02','#7570b3','#e7298a','#66a61e','#e6ab02','#a6761d','#666666']
            qualitative_8[7]  = ['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0','#f0027f','#bf5b17','#666666']
            # only two scales are left at 18
            qualitative_12    = {}
            qualitative_12[0] = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
            qualitative_12[1] = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
            if   n <= 8:
                return qualitative_8[alt%8][:n]
            elif n <= 12:
                return qualitative_12[alt%2][:n]
            else:
                raise Exception(f'brewerColors() - n={n} not supported for qualitative - try 12 or less.')
        else:
            raise Exception(f'brewerColors() - unknown scale {scale_type}')

#
# Dark Theme Color Manager
#
class RTColorManagerDarkTheme(RTColorManager):
    #
    # Initialize the type value colors
    #
    def __init__(self, racetrack):
        super().__init__(racetrack)

        self.type_color_lu['label']['defaultfg']    = '#afafaf'
        self.type_color_lu['label']['defaultbg']    = '#000000'
        self.type_color_lu['label']['error']        = '#ff0000'

        self.type_color_lu['border']['default']     = '#e0e0e0'        

        self.type_color_lu['background']['default'] = '#000000'

        self.type_color_lu['data']['default']        = '#b8b8b8'
        self.type_color_lu['data']['default_border'] = '#e8e8e8'
        
        #self.type_color_lu['context']['default']    = '#eeeeee'
        #self.type_color_lu['context']['text']       = '#808080'
