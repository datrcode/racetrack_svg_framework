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

import pandas as pd
import polars as pl
import numpy as np
import random
import heapq

from math import pi, sin, cos

from shapely.geometry import Polygon

from rt_component import RTComponent

__name__ = 'rt_chord_diagram_mixin'

#
# Chord Diagram Mixin
#
class RTChordDiagramMixin(object):
    #
    # dendrogramOrdering() - create an order of the fm/to nodes based on hierarchical clustering
    #
    def dendrogramOrdering(self, df, fm, to, count_by, count_by_set, _connector_ = ' <-|-> ', _sep_ = '|||'):
        # concats two strings in alphabetical order
        def fromToString(x):
            _fm_, _to_ = str(x[fm]), str(x[to])
            return (_fm_+_connector_+_to_) if (_fm_<_to_) else (_to_+_connector_+_fm_)
        # separates the concatenated string back into it's two parts
        def fromToStringParts(x):
            i = x.index(_connector_)
            return x[:i],x[i+len(_connector_):]
        # merges names (which themselves can be merged names)
        def mergedName(a, b):
            return _sep_.join(sorted(list(set(a.split(_sep_))|set(b.split(_sep_)))))
        # separates merged names back into parts
        def breakdownMerge(a):
            return a.split(_sep_)
        
        # perform the dataframe summation
        df = self.copyDataFrame(df)
        df['__fmto__'] = df.apply(lambda x: fromToString(x), axis=1)
        if count_by is None:
            df_den   = df.groupby('__fmto__').size().reset_index().rename({0:'__countby__'},axis=1)
            count_by = '__countby__'
        elif count_by_set:
            df_den = df.groupby('__fmto__')[count_by].nunique().reset_index()
        else:
            df_den = df.groupby('__fmto__')[count_by].sum().reset_index()

        # create the initial graph and heap
        _heap_ , _graph_ = [] , {}
        for r_i,r in df_den.iterrows():
            heapq.heappush(_heap_,(-r[count_by], r['__fmto__']))
            x, y = fromToStringParts(r['__fmto__'])
            if x != y:
                if x not in _graph_.keys():
                    _graph_[x] = {}
                _graph_[x][y] = -r[count_by]
                if y not in _graph_.keys():
                    _graph_[y] = {}
                _graph_[y][x] = -r[count_by]

        # iteratively merge the closest nodes together
        _tree_ = {}
        _merged_already_ = set()
        while len(_heap_) > 0:
            _strength_, _fmto_ = heapq.heappop(_heap_)
            _fm_, _to_ = fromToStringParts(_fmto_)
            if _fm_ != _to_ and _fm_ not in _merged_already_ and _to_ not in _merged_already_:
                _merged_already_.add(_fm_), _merged_already_.add(_to_)
                _new_ = mergedName(_fm_, _to_)
                _tree_[_new_] = (_fm_, _to_)
                _graph_[_new_] = {}
                # Rewire for _fm_
                for x in _graph_[_fm_].keys():
                    if x not in _graph_[_new_].keys():
                        _graph_[_new_][x] = 0    
                    _graph_[_new_][x] += _graph_[_fm_][x]
                # Rewire for _to_
                for x in _graph_[_to_].keys():
                    if x not in _graph_[_new_].keys():
                        _graph_[_new_][x] = 0
                    _graph_[_new_][x] += _graph_[_to_][x]
                # Rewire the neighbors & add the new values to the heap
                for x in _graph_[_new_].keys():
                    _graph_[x][_new_] = _graph_[_new_][x]
                    heapq.heappush(_heap_,(_graph_[_new_][x], _new_ + ' <-|-> ' + x))
                # Remove the old nodes and their nbor connections
                for x in _graph_[_fm_]:
                    _graph_[x].pop(_fm_)
                _graph_.pop(_fm_)
                for x in _graph_[_to_]:
                    _graph_[x].pop(_to_)
                _graph_.pop(_to_)

        # walk a tree in leaf order
        def leafWalk(t, n=None):
            if n is None:
                for x in t.keys():
                    n = x if (n is None) or (len(x) > len(n)) else n # root will be longest string
            if _sep_ not in n or n not in t.keys():
                return [n]
            else:
                l = leafWalk(t, t[n][0])
                r = leafWalk(t, t[n][1])
                if l is None and r is None:
                    return []
                elif l is None:
                    return r
                elif r is None:
                    return l
                else:
                    _extended_ = l
                    _extended_.extend(r)
                    return _extended_
        
        return leafWalk(_tree_)


    #
    # linkNode
    #
    # Make the SVG for a link node from a set of dataframes
    #    
    def chordDiagram(self,
                     df,                             # dataframe to render
                     relationships,                  # same convention as linknode [('fm','to')]
                     # ----------------------------- # everything else is a default...
                     color_by            = None,     # just the default color or a string for a field
                     count_by            = None,     # none means just count rows, otherwise, use a field to sum by
                     count_by_set        = False,    # count by summation (by default)... count_by column is checked
                     widget_id           = None,     # naming the svg elements                 
                     # ----------------------------- # chord diagram visualization
                     node_color          = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
                                                     # ... or a dictionary of the node string to either a string to color hash or a "#xxxxxx"
                     node_labels         = None,     # Dictionary of node string to array of strings for additional labeling options
                     node_labels_only    = False,    # Only label based on the node_labels dictionary
                     node_h              = 10,       # height of node from circle edge
                     node_gap            = 5,        # node gap in pixels (gap between the arcs)
                     link_color          = None,     # none means default color, 'vary' by color_by, or specific color "#xxxxxx"
                     link_opacity        = '1.0',    # link opacity
                     label_only          = set(),    # label only set
                     bidirectional       = False,    # initial version will not be bidirectionaly
                     # ----------------------------- # visualization geometry / etc.
                     track_state         = False,    # track state for interactive filtering
                     x_view              = 0,        # x offset for the view
                     y_view              = 0,        # y offset for the view
                     w                   = 256,      # width of the view
                     h                   = 256,      # height of the view
                     x_ins               = 3,
                     y_ins               = 3,
                     txt_h               = 12,       # text height for labeling
                     draw_labels         = True,     # draw labels flag # not implemented yet
                     draw_border         = True):    # draw a border around the graph
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTChordDiagram(self, **_params_)

    #
    # RTChordDiagram Class
    #
    class RTChordDiagram(RTComponent):
        #
        # Constructor
        #
        def __init__(self,
                     rt_self,
                     **kwargs):
            self.parms            = locals().copy()
            self.rt_self          = rt_self
            self.df               = rt_self.copyDataFrame(kwargs['df'])
            self.relationships    = kwargs['relationships']
            self.color_by         = kwargs['color_by']
            self.count_by         = kwargs['count_by']
            self.count_by_set     = kwargs['count_by_set']
            self.widget_id        = kwargs['widget_id']
            if self.widget_id is None:
                self.widget_id = 'chorddiagram_' + str(random.randint(0,65535))          
            self.node_color       = kwargs['node_color']
            self.node_labels      = kwargs['node_labels']
            self.node_labels_only = kwargs['node_labels_only']
            self.node_h           = kwargs['node_h']
            self.node_gap         = kwargs['node_gap']
            self.link_color       = kwargs['link_color']
            self.link_opacity     = kwargs['link_opacity']
            self.label_only       = kwargs['label_only']
            self.bidirectional    = kwargs['bidirectional']
            self.track_state      = kwargs['track_state']
            self.x_view           = kwargs['x_view']
            self.y_view           = kwargs['y_view']
            self.w                = kwargs['w']
            self.h                = kwargs['h']
            self.x_ins            = kwargs['x_ins']
            self.y_ins            = kwargs['y_ins']
            self.txt_h            = kwargs['txt_h']
            self.draw_labels      = kwargs['draw_labels']
            self.draw_border      = kwargs['draw_border']

            # Apply count-by transforms
            if self.count_by is not None and rt_self.isTField(self.count_by):
                self.df,self.count_by = rt_self.applyTransform(self.df, self.count_by)

            # Apply color-by transforms
            if self.color_by is not None and rt_self.isTField(self.color_by):
                self.df,self.color_by = rt_self.applyTransform(self.df, self.color_by)

            # Apply transforms to nodes
            for _edge in self.relationships:
                for _node in _edge:
                    if type(_node) == str:
                        if rt_self.isTField(_node) and rt_self.tFieldApplicableField(_node) in self.df.columns:
                            self.df,_throwaway = rt_self.applyTransform(self.df, _node)
                    else:
                        for _tup_part in _node:
                            if rt_self.isTField(_tup_part) and rt_self.tFieldApplicableField(_tup_part) in self.df.columns:
                                self.df,_throwaway = rt_self.applyTransform(self.df, _tup_part)

            # Check the count_by column
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)
            
            # Tracking state
            self.geom_to_df  = {}
            self.last_render = None

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
            return self.last_render

        #
        # renderSVG() - render as SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state:
                self.geom_to_df = {}

            # Determine the node order
            self.fm    = self.relationships[0][0]
            self.to    = self.relationships[0][1]
            self.order = self.rt_self.dendrogramOrdering(self.df, self.fm, self.to, self.count_by, self.count_by_set)
            print(self.order)

            # Determine the node volumes
            if   self.count_by is None:
                df_fm      = self.df.groupby(self.fm).size().reset_index().rename({self.fm:'__node__',0:'__fm_count__'},axis=1)
                df_to      = self.df.groupby(self.to).size().reset_index().rename({self.to:'__node__',0:'__to_count__'},axis=1)
            elif self.count_by_set:
                df_fm      = self.df.groupby(self.fm)[self.count_by].nunique().reset_index().rename({self.fm:'__node__',self.count_by:'__fm_count__'},axis=1)
                df_to      = self.df.groupby(self.to)[self.count_by].nunique().reset_index().rename({self.to:'__node__',self.count_by:'__to_count__'},axis=1)
            else:
                df_fm      = self.df.groupby(self.fm)[self.count_by].sum().reset_index().rename({self.fm:'__node__',self.count_by:'__fm_count__'},axis=1)
                df_to      = self.df.groupby(self.to)[self.count_by].sum().reset_index().rename({self.to:'__node__',self.count_by:'__to_count__'},axis=1)
            df_counter = df_fm.set_index('__node__').join(df_to.set_index('__node__')).reset_index()
            df_counter['__count__'] = df_counter['__fm_count__'] + df_counter['__to_count__']
            counter_lu = {}
            for row_i, row in df_counter.iterrows():
                counter_lu[row['__node__']] = row['__count__']
            counter_sum = df_counter['__count__'].sum()

            # Determine the geometry
            self.cx, self.cy = self.w/2, self.h/2
            self.rx, self.ry = (self.w - 2 * self.x_ins)/2, (self.h - 2 * self.y_ins)/2
            self.r           = self.rx if (self.rx < self.ry) else self.ry
            self.circ        = 2.0 * pi * self.r
            gap_pixels       = len(self.order) * self.node_gap
            if gap_pixels > 0.2 * self.circ:
                self.node_gap_adj = (0.2*self.circ)/len(self.order)
            else:
                self.node_gap_adj = self.node_gap
            self.node_gap_degs = 360.0 * (self.node_gap_adj / self.circ)
            left_over_degs  = 360.0 - self.node_gap_degs * len(self.order)
            self.node_to_arc   = {}
            a = 0.0
            for node in self.order:
                counter_perc  = counter_lu[node] / counter_sum
                node_degrees  = counter_perc * left_over_degs
                self.node_to_arc[node] = (a, a+node_degrees)
                a += node_degrees + self.node_gap_degs

            # Start the SVG Frame
            svg = []
            svg.append(f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">')
            background_color, axis_color = self.rt_self.co_mgr.getTVColor('background','default'), self.rt_self.co_mgr.getTVColor('axis','default')
            svg.append(f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />')
            svg.append(f'<circle cx="{self.cx}" cy="{self.cy}" r="{self.r}" fill="{background_color}" stroke-width="0.5" stroke="{axis_color}" />')

            # Draw the nodes
            _color_ = self.rt_self.co_mgr.getTVColor('data','default')
            for node in self.node_to_arc.keys():
                a0, a1 = self.node_to_arc[node]
                x0_out,  y0_out  = self.cx + self.r                 * cos(pi*a0/180.0), self.cy + self.r                 * sin(pi*a0/180.0)                
                x0_in,   y0_in   = self.cx + (self.r - self.node_h) * cos(pi*a0/180.0), self.cy + (self.r - self.node_h) * sin(pi*a0/180.0)
                x1_out,  y1_out  = self.cx + self.r                 * cos(pi*a1/180.0), self.cy + self.r                 * sin(pi*a1/180.0)                
                x1_in,   y1_in   = self.cx + (self.r - self.node_h) * cos(pi*a1/180.0), self.cy + (self.r - self.node_h) * sin(pi*a1/180.0)
                large_arc = 0 if (a1-a0) <= 180.0 else 1
                _path_ = f'M {x0_out} {y0_out} A {self.r} {self.r} 0 {large_arc} 1 {x1_out} {y1_out} L {x1_in} {y1_in} ' + \
                                            f' A {self.r-self.node_h} {self.r-self.node_h} 0 {large_arc} 0 {x0_in}  {y0_in}  Z'
                svg.append(f'<path d="{_path_}" stroke-width="0.8" stroke="{_color_}" fill="#ff0000" />')
                
            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg.append(f'<rect width="{self.w-1}" height="{self.h}" x="0" y="0" fill-opacity="0.0" fill="none" stroke="{border_color}" />')

            svg.append('</svg>')
            self.last_render = ''.join(svg)
            return self.last_render