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
import networkx as nx
import time

from rt_component import RTComponent

__name__ = 'rt_linknodeshortest_mixin'

#
# Linknode Shortest Mixin
#
class RTLinkNodeShortestMixin(object):
    def linkNodeShortest(self,
                        df, 
                        relationships,                     # [('src','dst'), ('sbj', 'obj', 'vrb'), ... ]
                        pairs,
                        g_orig                = None,
                        color_by              = None,
                        count_by              = None,
                        count_by_set          = False,
                        use_digraph           = False,
                        max_degree_to_show    = 30,        # annotate any nodes with this degree or higher ... None means no annotations
                        node_color            = None,      # None, "#xxxxxx", dict[node] = "#xxxxxx", "label"
                        node_size             = 'medium',  # N one, 'small', 'medium', 'large', 'vary' (vary by count_by), or a number
                        node_labels           = None,      # dict[node] = "label"
                        link_color            = None,      # None, "#xxxxxx", "relationship", "vary" (vary by color_by)
                        link_size             = 2,         # Number of "vary" (vary by count_by)
                        y_path_gap            = 15,        # Needs to be larger than txt_h
                        x_ins                 = 10,
                        y_ins                 = 10,
                        txt_h                 = 10,
                        draw_labels           = False,
                        label_links           = True,      # if the third tuple is set in the relationship
                        max_label_w           = 64,        # actuall max label height... but we'll stick with convention ... None means no limit
                        w                     = 1024):
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTLinkNodeShortest(self, **_params_)

    #
    # RTLinkNodeShortest()
    #
    class RTLinkNodeShortest(RTComponent):
        def __init__(self, rt_self, **kwargs):
            self.rt_self             = rt_self
            self.df                  = kwargs['df']
            self.relationships       = kwargs['relationships']    # [('fm','to'), (('fm1','fm2'),('to1','to2'))]
            self.pairs               = kwargs['pairs']            # [(src,dst), (sbj,obj), (sbj,obj,1,2,3)]
            self.g_orig              = kwargs['g_orig']
            self.color_by            = kwargs['color_by']
            self.count_by            = kwargs['count_by']
            self.count_by_set        = kwargs['count_by_set']
            self.use_digraph         = kwargs['use_digraph']      # use a directed graph
            self.max_degree_to_show  = kwargs['max_degree_to_show']
            self.node_color          = kwargs['node_color']
            self.node_size           = kwargs['node_size']
            self.node_labels         = kwargs['node_labels']
            self.link_color          = kwargs['link_color']
            self.link_size           = kwargs['link_size']
            self.y_path_gap          = kwargs['y_path_gap']
            self.x_ins               = kwargs['x_ins']
            self.y_ins               = kwargs['y_ins']
            self.txt_h               = kwargs['txt_h']
            self.draw_labels         = kwargs['draw_labels']
            self.label_links         = kwargs['label_links']
            self.max_label_w         = kwargs['max_label_w']
            self.w                   = kwargs['w']

            self.last_render         = None
            self.nodes_rendered      = set()
            self.time_lu             = {'link_labels':0.0}

            # Fix up the pairs / make sure it's a list
            if type(self.pairs) != list: self.pairs = [self.pairs]

            # If either from or to are tuples, concat them together... // could improve a little by ensuring any same tuples are not created more than once
            _ts_ = time.time()
            new_relationships = []
            for i in range(len(self.relationships)):
                _fm_ = self.relationships[i][0]
                if type(_fm_) == list or type(_fm_) == tuple:
                    new_fm = f'__fmcat{i}__'
                    self.df = self.rt_self.createConcatColumn(self.df, _fm_, new_fm)
                    _fm_ = new_fm
                _to_ = self.relationships[i][1]
                if type(_to_) == list or type(_to_) == tuple:
                    new_to = '__tocat{i}__'
                    self.df = self.rt_self.createConcatColumn(self.df, _to_, new_to)
                    _to_ = new_to
                if len(self.relationships[i]) == 2: new_relationships.append((_fm_,_to_))
                else:                               new_relationships.append((_fm_,_to_,self.relationships[i][2]))
            self.relationships = new_relationships
            self.time_lu['concat_columns'] = time.time() - _ts_

            self.node_size_px = 3
            if self.node_size is not None:
                if type(self.node_size) == int or type(self.node_size) == float: self.node_size_px = self.node_size
                elif self.node_size == 'small':  self.node_size_px = 2
                elif self.node_size == 'medium': self.node_size_px = 4
                elif self.node_size == 'large':  self.node_size_px = 6
            
            self.link_size_px = 2
            if self.link_size is not None:
                if type(self.link_size) == int or type(self.link_size) == float: self.link_size_px = self.link_size
                elif self.link_size == 'small':  self.link_size_px = 1
                elif self.link_size == 'medium': self.link_size_px = 2
                elif self.link_size == 'large':  self.link_size_px = 3

        # __linkRelationships__(self, n0, n1) - if the third tuple is set in the relationship
        # ... not as efficient as a group by ... but don't expect to be running this on more than a handful of edges...
        def __linkRelationships__(self, n0, n1):
            _ts_ = time.time()
            results = set()
            for _relationship_ in self.relationships:
                if len(_relationship_) == 3:
                    if self.rt_self.isPandas(self.df):
                        _df_ = self.df.query(f'{_relationship_[0]} == @n0 and {_relationship_[1]} == @n1')
                        if len(_df_) > 0: results |= set(_df_[_relationship_[2]])
                        if self.use_digraph == False:
                            _df_ = self.df.query(f'{_relationship_[0]} == @n1 and {_relationship_[1]} == @n0')
                            if len(_df_) > 0: results |= set(_df_[_relationship_[2]])
                    elif self.rt_self.isPolars(self.df):
                        _df_ = self.df.filter((pl.col(_relationship_[0]) == n0) & (pl.col(_relationship_[1]) == n1))
                        if len(_df_) > 0: results |= set(_df_[_relationship_[2]])
                        if self.use_digraph == False:
                            _df_ = self.df.filter((pl.col(_relationship_[0]) == n1) & (pl.col(_relationship_[1]) == n0))
                            if len(_df_) > 0: results |= set(_df_[_relationship_[2]])
                    else:
                        raise Exception('__linkRelationships__() - only supports pandas and polars')
            self.time_lu['link_labels'] += time.time() - _ts_
            return results

        # def _repr_svg_(self):
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
            return self.last_render

        # def renderSVG(self):
        def renderSVG(self):
            self.nodes_rendered = set()
            svg = []

            # Create the original graph, find the shortest path and make a version that removes that path
            _ts_ = time.time()
            if self.g_orig is None: self.g_orig = self.rt_self.createNetworkXGraph(self.df, self.relationships, use_digraph=self.use_digraph)
            self.time_lu['create_graph'] = time.time() - _ts_

            # Go through the pairs
            for _pair_ in self.pairs:

                y_base = self.y_ins
                p      = nx.shortest_path(self.g_orig, _pair_[0], _pair_[1])
                g      = self.g_orig.copy()
                for i in range(len(p)-1): g.remove_edge(p[i],p[i+1])

                # Determine the label geometry
                _label_w_ = self.y_ins + self.y_path_gap
                if self.draw_labels:
                    for i in range(len(p)):
                        if self.node_labels is not None and p[i] in self.node_labels: _node_ = str(self.node_labels[p[i]])
                        else:                                                         _node_ = str(p[i])
                        _w_       = self.rt_self.textLength(_node_, self.txt_h) + self.y_ins + self.y_path_gap
                        _label_w_ = max(_label_w_, _w_)
                    if self.max_label_w is not None: _label_w_ = min(_label_w_, self.max_label_w)

                # Figure out the range
                _range_ = range(1, len(p)-1)
                if len(_pair_) > 2: _range_ = _pair_[2]

                # For all view path indices (VPI's) render a small graph showing what that looks like
                for _vpi_ in _range_:
                    # Geometry
                    y_top       = y_base
                    y_floors    = max(max(abs(len(p) - _vpi_), abs(_vpi_ - len(p))), _vpi_)
                    y_base     += self.y_path_gap*y_floors
                    x_path_gap  = (self.w - 2*self.x_ins)/(len(p)-1)
                    node_to_xy  = {}

                    # Render the base path
                    for i in range(len(p)-1):
                        n0, n1 = p[i], p[i+1]
                        x0 = self.x_ins + x_path_gap*i
                        node_to_xy[n0] = (x0, y_base)
                        x1 = self.x_ins + x_path_gap*(i+1)
                        node_to_xy[n1] = (x1, y_base)
                        _color_ = self.rt_self.co_mgr.getTVColor('data','default')
                        if (self.draw_labels and self.label_links) or self.link_color == 'relationship':
                            link_relationships = self.__linkRelationships__(n0, n1)
                            if   len(link_relationships) >  1: _str_ = '*'
                            elif len(link_relationships) == 1:
                                _pop_ = link_relationships.pop() 
                                _str_ = str(_pop_)
                                if self.link_color == 'relationship': _color_ = self.rt_self.co_mgr.getColor(_pop_)
                            else:                              _str_ = None
                            if _str_ is not None and self.draw_labels and self.label_links: 
                                svg.append(self.rt_self.svgText(self.rt_self.cropText(_str_, self.txt_h, x1-x0), (x0+x1)/2, y_base-2, self.txt_h, anchor='middle'))
                        svg.append(f'<line x1="{x0}" y1="{y_base}" x2="{x1}" y2="{y_base}" stroke="{_color_}" stroke-width="{self.link_size_px}" />')

                    # Render the base path nodes
                    for _node_ in node_to_xy:
                        _color_ = self.rt_self.co_mgr.getTVColor('data','default')
                        if self.node_color is not None:
                            if   self.node_color.startswith('#') and len(self.node_color) == 7: _color_ = self.node_color
                            elif self.node_color == 'label':
                                if self.node_labels is not None and _node_ in self.node_labels: _color_ = self.rt_self.co_mgr.getColor(_node_)                         
                                else:                                                           _color_ = self.rt_self.co_mgr.getColor(_node_)
                        self.nodes_rendered.add(_node_)
                        svg.append(f'<circle cx="{node_to_xy[_node_][0]}" cy="{node_to_xy[_node_][1]}" r="{self.node_size_px}" stroke="{_color_}" fill="{_color_}" stroke-width="1" />')

                        # Max Degree Annotation
                        if self.max_degree_to_show is not None and self.g_orig.degree[_node_] >= self.max_degree_to_show:
                            _color_ = self.rt_self.co_mgr.getTVColor('axis','major')
                            svg.append(f'<circle cx="{node_to_xy[_node_][0]}" cy="{node_to_xy[_node_][1]}" r="{self.node_size_px+2}" stroke="{_color_}" fill="none" stroke-width="2" />')
                        
                        # Render the node labels
                        if self.draw_labels:
                            if self.node_labels is not None and _node_ in self.node_labels: _node_label_ = str(self.node_labels[_node_])
                            else:                                                           _node_label_ = str(_node_)
                            _cropped_ = self.rt_self.cropText(str(_node_label_), self.txt_h, _label_w_-self.y_ins)
                            svg.append(self.rt_self.svgText(_cropped_, node_to_xy[_node_][0]-self.txt_h/2, node_to_xy[_node_][1]+self.txt_h, self.txt_h, anchor='start', rotation=90))

                    # Bottom of this specific graph        
                    y_bot = y_base + _label_w_

                    # Render the alternative paths if the baseline path wasn't in existence
                    if _vpi_ > 0 and _vpi_ < len(p)-1:
                        node_center = p[_vpi_]
                        svg.append(f'<line x1="{node_to_xy[node_center][0]}" y1="{y_top}" x2="{node_to_xy[node_center][0]}" y2="{y_base}" stroke="gray" stroke-width="0.5" stroke-dasharray="1 5 1" />')
                    offset = 1
                    while offset < len(p):
                        for _side_ in ['backward', 'forward']:
                            _my_txt_x_offset_ = -4    if _side_ == 'backward' else 4
                            _my_anchor_       = 'end' if _side_ == 'backward' else 'start'
                            j = _vpi_ - offset if _side_ == 'backward' else _vpi_ + offset                    
                            y = y_base - self.y_path_gap*offset
                            if j >= 0 and j <= len(p)-1:
                                try:    pp = nx.shortest_path(g, p[j],p[_vpi_]) if _side_ == 'backward' else nx.shortest_path(g, p[_vpi_],p[j])
                                except: pp = None
                                if pp is not None:
                                    x0, x1 = node_to_xy[pp[0]][0]+x_path_gap/4, node_to_xy[pp[-1]][0]-x_path_gap/4
                                    svg.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="gray" stroke-width="0.5" />')
                                    svg.append(f'<line x1="{x0}" y1="{y}" x2="{node_to_xy[pp[0]][0]}" y2="{node_to_xy[pp[0]][1]}" stroke="gray" stroke-width="0.5" />')
                                    svg.append(f'<line x1="{x1}" y1="{y}" x2="{node_to_xy[pp[-1]][0]}" y2="{node_to_xy[pp[-1]][1]}" stroke="gray" stroke-width="0.5" />')
                                    svg.append(self.rt_self.svgText(f'{len(pp)}', node_to_xy[p[_vpi_]][0] + _my_txt_x_offset_, y + self.txt_h/2, self.txt_h, anchor=_my_anchor_))
                                    if len(pp) > 3:
                                        my_x_path_gap = (x1-x0)/(len(pp)-3)
                                        for k in range(1, len(pp)-1):
                                            _color_ = self.rt_self.co_mgr.getTVColor('data','default')
                                            if self.node_color is not None:
                                                if   self.node_color.startswith('#') and len(self.node_color) == 7: _color_ = self.node_color
                                                elif self.node_color == 'label':
                                                    if self.node_labels is not None and pp[k] in self.node_labels: _color_ = self.rt_self.co_mgr.getColor(pp[k])
                                                    else:                                                           _color_ = self.rt_self.co_mgr.getColor(pp[k])
                                            self.nodes_rendered.add(pp[k])
                                            svg.append(f'<circle cx="{x0+(k-1)*my_x_path_gap}" cy="{y}" r="{self.node_size_px}" stroke="{_color_}" fill="{_color_}" stroke-width="0.5" />')

                                            # Max Degree Annotation
                                            if self.max_degree_to_show is not None and self.g_orig.degree[pp[k]] >= self.max_degree_to_show:
                                                _color_ = self.rt_self.co_mgr.getTVColor('axis','major')
                                                svg.append(f'<circle cx="{x0+(k-1)*my_x_path_gap}" cy="{y}" r="{self.node_size_px+2}" stroke="{_color_}" fill="none" stroke-width="2" />')
                                    else:
                                        _color_ = self.rt_self.co_mgr.getTVColor('data','default')
                                        if self.node_color is not None:
                                            if   self.node_color.startswith('#') and len(self.node_color) == 7: _color_ = self.node_color
                                            elif self.node_color == 'label':
                                                if self.node_labels is not None and pp[0] in self.node_labels: _color_ = self.rt_self.co_mgr.getColor(pp[0])
                                                else:                                                          _color_ = self.rt_self.co_mgr.getColor(pp[0])
                                        self.nodes_rendered.add(pp[0])
                                        svg.append(f'<circle cx="{x0+x_path_gap/2}" cy="{y}" r="{self.node_size_px}" stroke="{_color_}" fill="{_color_}" stroke-width="0.5" />')

                                        # Max Degree Annotation
                                        if self.max_degree_to_show is not None and self.g_orig.degree[pp[0]] >= self.max_degree_to_show:
                                            _color_ = self.rt_self.co_mgr.getTVColor('axis','major')
                                            svg.append(f'<circle cx="{x0+x_path_gap/2}" cy="{y}" r="{self.node_size_px+2}" stroke="{_color_}" fill="none" stroke-width="2" />')
                                else:
                                    svg.append(self.rt_self.svgText('∅', node_to_xy[p[_vpi_]][0] + _my_txt_x_offset_, y + self.txt_h/2, self.txt_h, anchor=_my_anchor_))
                        offset += 1
                    y_base = y_bot+self.y_path_gap
                y_base += self.y_ins
            svg.insert(0, f'<svg width="{self.w}" height="{y_base}">')
            svg.append('</svg>')
            self.last_render = ''.join(svg)
