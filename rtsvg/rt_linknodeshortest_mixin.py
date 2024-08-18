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
import random

from math import ceil, floor

from .rt_component import RTComponent
from .rt_entity_position import RTEntityPosition

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
                        widget_id             = None,      # naming the svg elements
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
                        link_labels           = True,      # if the third tuple is set in the relationship
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
            self.widget_id           = kwargs['widget_id']
            if self.widget_id is None: self.widget_id = 'rtlns_' + str(random.randint(0,2**24))
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
            self.link_labels         = kwargs['link_labels']
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

        #  __updateEntityPositions__()
        def __updateEntityPositions__(self, _node_, _node_label_, _x_, _y_, _r_):
            _node_label_ = str(_node_)
            if _node_ not in self.entity_positions: self.entity_positions[_node_] = []
            _instance_no_ = len(self.entity_positions[_node_])
            _svg_id_ = self.rt_self.encSVGID(self.widget_id + '-' + str(_node_) + '-' + str(_instance_no_))
            _tuple_ = (_svg_id_, _node_, _node_label_, _x_, _y_, _r_)
            self.entity_positions[_node_].append(_tuple_)
            return _svg_id_

        #
        # entityPositions() - return information about the entity geometry for rendering
        # - Empty list means either not implemented... or entity not in view...
        # - return the positions of the entity ... rendering had to have happened first
        def __entityPositions__(self, entity):
            results = []
            if entity in self.entity_positions:
                for _tuple_ in self.entity_positions[entity]:
                    rtep = RTEntityPosition(entity,
                                            self.rt_self,
                                            self,
                                            (_tuple_[3], _tuple_[4]),
                                            (_tuple_[3], _tuple_[4], 0.0, _tuple_[5]),
                                            _tuple_[0],
                                            f'<circle cx="{_tuple_[3]}" cy="{_tuple_[4]}" r="{_tuple_[5]}"/>',
                                            self.widget_id)
                    results.append(rtep)
            return results

        def entityPositions(self, entity_or_label):
            if entity_or_label in self.entity_positions:
                return self.__entityPositions__(entity_or_label)
            elif str(entity_or_label) in self.entity_positions:
                return self.__entityPositions__(str(entity_or_label))
            elif len(self.node_labels) > 0:
                rteps = []
                for entity in self.node_labels:
                    if self.node_labels[entity] == entity_or_label:
                        _results_ = self.__entityPositions__(entity)
                        for rtep in _results_:
                            rtep.entity = entity_or_label
                            rteps.append(rtep)
                return rteps
            return []

        # _repr_svg_(self):
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
            return self.last_render

        # renderSVG(self):
        def renderSVG(self):
            self.nodes_rendered   = set()
            self.entity_positions = {}
            svg, svg_edges, svg_labels = [], [], []

            def __renderNode__(_node_, _x_, _y_, _z_):
                _node_label_ = self.node_labels[_node_] if self.node_labels is not None and _node_ in self.node_labels else _node_
                _color_ = self.rt_self.co_mgr.getTVColor('data','default')
                if self.node_color is not None:
                    if   self.node_color.startswith('#') and len(self.node_color) == 7: _color_ = self.node_color
                    elif self.node_color == 'label':                                    _color_ = self.rt_self.co_mgr.getColor(_node_label_)
                self.nodes_rendered.add(_node_label_)
                _svg_id_ = self.__updateEntityPositions__(_node_, _node_label_, _x_, _y_, _r_)
                svg.append(f'<circle id="{_svg_id_}" cx="{_x_}" cy="{_y_}" r="{_r_}" stroke="{_color_}" fill="{_color_}" stroke-width="1" />')
                if self.max_degree_to_show is not None and self.g_orig.degree[_node_] >= self.max_degree_to_show:
                    _color_ = self.rt_self.co_mgr.getTVColor('axis','major')
                    svg.append(f'<circle cx="{_x_}" cy="{_y_}" r="{_r_+2}" stroke="{_color_}" fill="none" stroke-width="2" />')
                return _node_label_

            # Create the original graph, find the shortest path and make a version that removes that path
            _ts_ = time.time()
            if self.g_orig is None: self.g_orig = self.rt_self.createNetworkXGraph(self.df, self.relationships, use_digraph=self.use_digraph)
            self.time_lu['create_graph'] = time.time() - _ts_

            # Go through the pairs
            y_base = self.y_ins
            for _pair_ in self.pairs:
                # Find the base path (and all paths of the same length)
                p_gen         = nx.all_shortest_paths(self.g_orig, _pair_[0], _pair_[1])
                g             = self.g_orig.copy()
                p_gen_list    = []
                edges_removed = set()
                for p in p_gen:
                    p_gen_list.append(p)
                    for i in range(len(p)-1):
                        to_remove = (p[i],p[i+1])
                        if to_remove not in edges_removed:
                            g.remove_edge(p[i],p[i+1])
                            edges_removed.add(to_remove)
                p = p_gen_list[0]

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
                if len(_pair_) > 2:
                    _range_ = []
                    for i in _pair_[2:]:
                        if i == 0: _range_.append(floor(len(p)/2))
                        else:      _range_.append(i)

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
                        if (self.draw_labels and self.link_labels) or self.link_color == 'relationship':
                            link_relationships = self.__linkRelationships__(n0, n1)
                            if   len(link_relationships) >  1: _str_ = '*'
                            elif len(link_relationships) == 1:
                                _pop_ = link_relationships.pop() 
                                _str_ = str(_pop_)
                                if self.link_color == 'relationship': _color_ = self.rt_self.co_mgr.getColor(_pop_)
                            else:                              _str_ = None
                            if _str_ is not None and self.draw_labels and self.link_labels: 
                                svg.append(self.rt_self.svgText(self.rt_self.cropText(_str_, self.txt_h, x1-x0), (x0+x1)/2, y_base-2, self.txt_h, anchor='middle'))
                        svg_edges.append(f'<line x1="{x0}" y1="{y_base}" x2="{x1}" y2="{y_base}" stroke="{_color_}" stroke-width="{self.link_size_px}" />')

                    # Render the base path nodes
                    for i in range(len(p)):
                        _node_ = p[i]
                        _x_, _y_, _r_ = node_to_xy[_node_][0], node_to_xy[_node_][1], self.node_size_px
                        _node_already_drawn_ = set([_node_])
                        for j in range(len(p_gen_list)-1, 0, -1): # draw in reverse order
                            _alt_p_ = p_gen_list[j]
                            _alt_n_ = _alt_p_[i]
                            if j == 0 or _alt_n_ not in _node_already_drawn_:
                                __renderNode__(_alt_n_, _x_+j*_r_, _y_+j*_r_, _r_)
                                _node_already_drawn_.add(_alt_n_)
                        _node_label_ = __renderNode__(_node_, _x_, _y_, _r_)
                        # Render the node labels
                        if self.draw_labels:
                            _cropped_ = self.rt_self.cropText(str(_node_label_), self.txt_h, _label_w_-self.y_ins)
                            svg.append(self.rt_self.svgText(_cropped_, node_to_xy[_node_][0]-self.txt_h/2, node_to_xy[_node_][1]+self.txt_h, self.txt_h, anchor='start', rotation=90))

                    # Bottom of this specific graph        
                    y_bot = y_base + _label_w_

                    # Render the alternative paths if the baseline path wasn't in existence
                    if _vpi_ > 0 and _vpi_ < len(p)-1:
                        node_center = p[_vpi_]
                        svg_edges.append(f'<line x1="{node_to_xy[node_center][0]}" y1="{y_top}" x2="{node_to_xy[node_center][0]}" y2="{y_base}" stroke="gray" stroke-width="0.5" stroke-dasharray="1 5 1" />')
                    offset = 1
                    while offset < len(p):
                        for _side_ in ['backward', 'forward']:
                            _my_txt_x_offset_ = -4    if _side_ == 'backward' else 4
                            _my_anchor_       = 'end' if _side_ == 'backward' else 'start'
                            j = _vpi_ - offset if _side_ == 'backward' else _vpi_ + offset                    
                            y = y_base - self.y_path_gap*offset
                            if j >= 0 and j <= len(p)-1:
                                _pp_gen_ = []
                                try:
                                    _pp_gen_ = nx.all_shortest_paths(g, p[j],p[_vpi_]) if _side_ == 'backward' else nx.all_shortest_paths(g, p[_vpi_],p[j])
                                    _pps_    = list(_pp_gen_)
                                    if len(_pps_) > 0: pp = _pps_[0]
                                    else:              pp = None                                    
                                except: pp = None
                                if pp is not None:
                                    x0, x1 = node_to_xy[pp[0]][0]+x_path_gap/4, node_to_xy[pp[-1]][0]-x_path_gap/4
                                    svg_edges.append(f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" stroke="gray" stroke-width="0.5" />')
                                    svg_edges.append(f'<line x1="{x0}" y1="{y}" x2="{node_to_xy[pp[0]][0]}" y2="{node_to_xy[pp[0]][1]}" stroke="gray" stroke-width="0.5" />')
                                    svg_edges.append(f'<line x1="{x1}" y1="{y}" x2="{node_to_xy[pp[-1]][0]}" y2="{node_to_xy[pp[-1]][1]}" stroke="gray" stroke-width="0.5" />')
                                    if len(_pps_) > 1: _path_len_label_ = f'{len(pp)}/{len(_pps_)}'
                                    else:              _path_len_label_ = f'{len(pp)}'

                                    svg.append(self.rt_self.svgText(_path_len_label_, node_to_xy[p[_vpi_]][0] + _my_txt_x_offset_, y + self.txt_h/2, self.txt_h, anchor=_my_anchor_))
                                    if len(pp) > 3:
                                        my_x_path_gap = (x1-x0)/(len(pp)-3)
                                        for k in range(1, len(pp)-1):
                                            _node_ = pp[k]
                                            _x_, _y_, _r_ = x0+(k-1)*my_x_path_gap, y, self.node_size_px
                                            _node_label_ = __renderNode__(_node_,_x_,_y_,_r_)
                                            if self.draw_labels:
                                                _cropped_ = self.rt_self.cropText(str(_node_label_), self.txt_h-2, _label_w_-self.y_ins)
                                                svg_labels.append(self.rt_self.svgText(_cropped_, _x_-(self.txt_h-2)/2, _y_+self.txt_h, 
                                                                                       self.txt_h-2, self.rt_self.co_mgr.getTVColor('context','text'), anchor='start', rotation=90))

                                    else:
                                        _node_ = pp[0]
                                        _x_, _y_, _r_ = x0+x_path_gap/2, y, self.node_size_px
                                        _node_label_ = __renderNode__(_node_,_x_,_y_,_r_)
                                else:
                                    svg.append(self.rt_self.svgText('∅', node_to_xy[p[_vpi_]][0] + _my_txt_x_offset_, y + self.txt_h/2, self.txt_h, anchor=_my_anchor_))
                        offset += 1
                    y_base = y_bot+self.y_path_gap
                y_base += self.y_ins

            _bg_color_ = self.rt_self.co_mgr.getTVColor('background','default')
            self.h = y_base
            self.last_render = f'<svg x="0" y="0" width="{self.w}" height="{y_base}">' + \
                               f'<rect x="0" y="0" width="{self.w}" height="{y_base}" fill="{_bg_color_}" />' + \
                               ''.join(svg_edges) + ''.join(svg_labels) + ''.join(svg) + \
                               '</svg>'
            return self.last_render
