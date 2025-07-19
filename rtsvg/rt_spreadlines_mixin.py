# Copyright 2025 David Trimm
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

__name__ = 'rt_spreadlines_mixin'

import polars as pl
import random
import time

from shapely.geometry import Point,Polygon,LineString

from .rt_component import RTComponent
from .rt_entity_position import RTEntityPosition

class RTSpreadLinesMixin(object):
    def spreadLinesPreferredDimensions(self, **kwargs):
        return (160, 96)
    def spreadLinesMinimumDimensions(self, **kwargs):
        return (160, 96)
    def spreadLinesSmallMultiplesDimensions(self, **kwargs):
        return (160, 96)
    def spreadLinesRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('color_by', kwargs, columns_set)
        return columns_set
    
    def spreadLines(self,
                    df,
                    relationships,
                    node_focus,                   # node or set of nodes
                    only_render_nodes    = None,  # set of nodes to render... if None, just render normally
                    ts_field             = None,  # Will attempt to guess based on datatypes
                    every                = '1d',  # "the every field for the group_by_dynamic" ... 1d, 1h, 1m
                    color_by             = None,
                    count_by             = None,  # does nothing
                    count_by_set         = False, # does nothing
                    node_color           = None,  # none means default color, 'vary' by color_by, or 'node' to convert the node string into a color
                                                  # ... or a dictionary of the node string to either a string to color hash or a "#xxxxxx"
                    alter_inter_d        = 192,   # distance between the alters
                    max_bin_w            = 64,    # max width of the bin
                    max_bin_h            = 450*2, # max height of the bin
                    min_channel_w        = 8,     # min width of the channel
                    max_channel_w        = 16,    # max width of the channel
                    channel_inter_d      = 4,     # distance between the channels
                    r_min                = 4.0, 
                    r_pref               = 7.0, 
                    circle_inter_d       = 2.0, 
                    circle_spacer        = 3,
                    alter_separation_h   = 48, 
                    h_collapsed_sections = 16,

                    include_svg_viewbox  = True,
                    widget_id            = None,
                    draw_labels          = True,
                    w                    = 1024,
                    h                    = 960,
                    x_view               = 0,
                    y_view               = 0,
                    x_ins                = 32,
                    y_ins                = 8,
                    txt_h                = 12):
        if self.isPolars(df) == False: raise Exception('spreadLines() - only supports polars dataframe')
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTSpreadLines(self, **_params_)


    class RTSpreadLines(RTComponent):
        #
        # transform all fields (if they are t-field)
        # - replace those fields w/ the new versions (i actually don't think the names change...)
        #
        def __transformFields__(self):
            # Gather up all of the fields that are going to be used
            _all_columns_ = [self.ts_field]
            if self.color_by is not None: _all_columns_.append(self.color_by)
            if self.count_by is not None: _all_columns_.append(self.count_by)
            for _relationship_ in self.relationships:
                _fm_, _to_ = _relationship_[0], _relationship_[1]
                if   type(_fm_) is str: _all_columns_.append(_fm_)
                elif type(_fm_) is tuple:
                    for i in range(len(_fm_)): _all_columns_.append(_fm_[i])
                if   type(_to_) is str: _all_columns_.append(_to_)
                elif type(_to_) is tuple:
                    for i in range(len(_to_)): _all_columns_.append(_to_[i])
            # Transform the fields
            self.df, _new_columns_ = self.rt_self.transformFieldListAndDataFrame(self.df, _all_columns_)
            # Remap them
            col_i = 0
            self.ts_field        = _new_columns_[col_i]
            col_i += 1
            if self.color_by is not None: 
                self.color_by = _new_columns_[col_i]
                col_i += 1
            if self.count_by is not None:
                self.count_by = _new_columns_[col_i]
                col_i += 1
            _new_relationships_ = []
            for _relationship_ in self.relationships:
                _fm_, _to_ = _relationship_[0], _relationship_[1]
                if   type(_fm_) is str: 
                    _fm_ = _new_columns_[col_i]
                    col_i += 1
                elif type(_fm_) is tuple:
                    as_list = []
                    for i in range(len(_fm_)):
                        as_list.append(_new_columns_[col_i])                    
                        col_i += 1
                    _fm_ = tuple(as_list)
                if   type(_to_) is str: 
                    _to_ = _new_columns_[col_i]
                    col_i += 1
                elif type(_to_) is tuple:
                    as_list = []
                    for i in range(len(_to_)): 
                        as_list.append(_new_columns_[col_i])
                        col_i += 1
                    _to_ = tuple(as_list)
                _new_relationships_.append((_fm_, _to_))
            self.relationships = _new_relationships_


        #
        # __consolidateRelationships__() - simplify the relationship fields into a single field
        # ... and use standard naming
        # ... replaces the "relationships" field w/ the consolidated field names
        # ... use (__fm0__, __to0__),( __fm1__, __to1__), etc.
        #
        def __consolidateRelationships__(self):
            new_relationships = []
            for i in range(len(self.relationships)):
                _fm_, _to_ = self.relationships[i]
                new_fm = f'__fm{i}__'
                new_to = f'__to{i}__'
                if type(_fm_) is str: self.df = self.df.with_columns(pl.col(_fm_).alias(new_fm))
                else:                 self.df = self.rt_self.createConcatColumn(self.df, _fm_, new_fm)
                if type(_to_) is str: self.df = self.df.with_columns(pl.col(_to_).alias(new_to))
                else:                 self.df = self.rt_self.createConcatColumn(self.df, _to_, new_to)
                new_relationships.append((new_fm, new_to))
            self.relationships = new_relationships

        #
        #
        #
        def __init__(self, rt_self, **kwargs):
            self.rt_self              = rt_self
            self.df                   = rt_self.copyDataFrame(kwargs['df'])
            self.relationships        = kwargs['relationships']

            self.node_focus           = kwargs['node_focus']
            if   type(self.node_focus) is     list: self.node_focus = set(self.node_focus)
            elif type(self.node_focus) is not set:  self.node_focus = set([self.node_focus])

            self.only_render_nodes    = kwargs['only_render_nodes']
            self.ts_field             = self.rt_self.guessTimestampField(self.df) if kwargs['ts_field'] is None else kwargs['ts_field']
            self.every                = kwargs['every']
            self.color_by             = kwargs['color_by']
            self.count_by             = kwargs['count_by']
            self.count_by_set         = kwargs['count_by_set']
            self.node_color           = kwargs['node_color']

            self.alter_inter_d        = kwargs['alter_inter_d']
            self.max_bin_w            = kwargs['max_bin_w']
            self.max_bin_h            = kwargs['max_bin_h']
            self.min_channel_w        = kwargs['min_channel_w']
            self.max_channel_w        = kwargs['max_channel_w']
            self.channel_inter_d      = kwargs['channel_inter_d']
            self.r_min                = kwargs['r_min']
            self.r_pref               = kwargs['r_pref']
            self.circle_inter_d       = kwargs['circle_inter_d']
            self.circle_spacer        = kwargs['circle_spacer']
            self.alter_separation_h   = kwargs['alter_separation_h']
            self.h_collapsed_sections = kwargs['h_collapsed_sections']

            self.include_svg_viewbox = kwargs['include_svg_viewbox']
            self.widget_id           = f'spreadlines_{random.randint(0,65535)}' if kwargs['widget_id'] is None else kwargs['widget_id']
            self.draw_labels         = kwargs['draw_labels']
            self.w                   = kwargs['w']
            self.h                   = kwargs['h']
            self.x_view              = kwargs['x_view']
            self.y_view              = kwargs['y_view']
            self.x_ins               = kwargs['x_ins']
            self.y_ins               = kwargs['y_ins']
            self.txt_h               = kwargs['txt_h']

            # Performance information
            self.time_lu       = {}
            # Unwrap any fields w/ the appropriate transforms
            t0 = time.time()
            self.__transformFields__()
            self.time_lu['transforms'] = time.time() - t0
            # Consolidate the fm's and to's into a simple field (__fm0__, __to0__),( __fm1__, __to1__), etc.
            t0 = time.time()
            self.__consolidateRelationships__()
            self.time_lu['consolidate_relationships'] = time.time() - t0

            # Binning Stage
            self.df = self.df.sort(self.ts_field)
        
            # Create other variables (to be used later ... but make sure they exist now)
            self.vx0, self.vy0, self.vx1, self.vy1 = None, None, None, None # view bounds
            self.selected_entities        = set()
            self.bin_to_bounds            = {}
            self.bin_to_node_to_xyrepstat = {}
            self.last_render              = None

        #
        # __binNeighboringNodes__() - place nodes into timeslices & alters based on the focal node(s)
        #
        def __binNeighboringNodes__(self):
            self.bin_to_timestamps             = {}
            self.bin_to_alter1s                = {} # [_bin_]['fm'] and [_bin_]['to']
            self.bin_to_alter2s                = {} # [_bin_]['fm'] and [_bin_]['to']
            self.discontinuity_count_after_bin = {} # counts the missing bins (because the focal node wasn't present)
            self.bin_to_focal_nodes_present    = {}
            t0 = time.time()
            for i in range(len(self.relationships)):
                _bin_            = 0
                _fm_, _to_       = self.relationships[i]
                _df_             = self.df.group_by_dynamic(self.ts_field, every=self.every, group_by=[_fm_,_to_]).agg()
                _one_degree_     = _df_.filter((pl.col(_fm_).is_in(self.node_focus)) | (pl.col(_to_).is_in(self.node_focus)))
                _one_degree_set_ = set(_one_degree_[_fm_]) | set(_one_degree_[_to_])
                _df_             = _df_.filter((pl.col(_fm_).is_in(_one_degree_set_)) | (pl.col(_to_).is_in(_one_degree_set_)))
                _df_             = _df_.sort(self.ts_field)
                for k, k_df in _df_.group_by_dynamic(self.ts_field, every=self.every):
                    _timestamp_   = k[0]
                    _fm_is_focus_    = k_df.filter((pl.col(_fm_).is_in(self.node_focus)) & (~pl.col(_to_).is_in(self.node_focus)))
                    _to_is_focus_    = k_df.filter((pl.col(_to_).is_in(self.node_focus)) & (~pl.col(_fm_).is_in(self.node_focus)))
                    _fm_to_conn_     = k_df.filter((pl.col(_to_).is_in(self.node_focus)) & ( pl.col(_fm_).is_in(self.node_focus)))
                    # if no focal nodes present, then this is a discontinuity
                    if len(_fm_is_focus_) == 0 and len(_to_is_focus_) == 0:
                        if _bin_ not in self.discontinuity_count_after_bin: self.discontinuity_count_after_bin[_bin_] = 0
                        self.discontinuity_count_after_bin[_bin_] += 1
                        continue
                    # Record the timestamp information and set up the lookup sets
                    if _bin_ not in self.bin_to_timestamps:
                        self.bin_to_alter1s   [_bin_]          = {'fm': set(), 'to': set()}
                        self.bin_to_alter2s   [_bin_]          = {'fm': set(), 'to': set()}
                        self.bin_to_timestamps[_bin_]          = _timestamp_
                        self.bin_to_focal_nodes_present[_bin_] = set()
                    # Record the 'to' bins
                    if len(_fm_is_focus_) > 0: 
                        _set_ = set(_fm_is_focus_[_to_])
                        self.bin_to_alter1s[_bin_]['to'] |= _set_
                        _alter2s_ = k_df.filter(pl.col(_to_).is_in(_set_) | (pl.col(_fm_).is_in(_set_)))
                        self.bin_to_alter2s[_bin_]['to'] |= set(_alter2s_[_fm_]) | set(_alter2s_[_to_])
                    # Record the 'fm' bins
                    if len(_to_is_focus_) > 0: 
                        _set_ = set(_to_is_focus_[_fm_])
                        self.bin_to_alter1s[_bin_]['fm'] |= _set_
                        _alter2s_ = k_df.filter(pl.col(_fm_).is_in(_set_) | (pl.col(_to_).is_in(_set_)))
                        self.bin_to_alter2s[_bin_]['fm'] |= set(_alter2s_[_fm_]) | set(_alter2s_[_to_])
                    # Record the focal nodes present
                    self.bin_to_focal_nodes_present[_bin_] |= set(_fm_is_focus_[_fm_]) | set(_to_is_focus_[_to_]) | set(_fm_to_conn_[_fm_]) | set(_fm_to_conn_[_to_])
                    # Increment to the next bin
                    _bin_ += 1
            self.time_lu['alter_binning_step'] = time.time() - t0

            # Make sure the sets are distinct & don't have overlaps
            t0 = time.time()
            for _bin_ in self.bin_to_alter1s:
                self.bin_to_alter1s[_bin_]['to'] -= self.bin_to_alter1s[_bin_]['fm']                                                                         # 'fm' side has the bidirectional nodes
                self.bin_to_alter2s[_bin_]['fm'] -= (self.bin_to_alter1s[_bin_]['fm'] | self.bin_to_alter1s[_bin_]['to'])                                    # 'fm' side has the bidirectional nodes
                self.bin_to_alter2s[_bin_]['to'] -= (self.bin_to_alter1s[_bin_]['fm'] | self.bin_to_alter1s[_bin_]['to'] | self.bin_to_alter2s[_bin_]['fm']) # 'to' side has the bidirectional nodes
                self.bin_to_alter1s[_bin_]['fm'] -= self.node_focus
                self.bin_to_alter1s[_bin_]['to'] -= self.node_focus
                self.bin_to_alter2s[_bin_]['fm'] -= self.node_focus
                self.bin_to_alter2s[_bin_]['to'] -= self.node_focus
            self.time_lu['deduplicate_alters'] = time.time() - t0


        # nodesInBin() - return the set of nodes that exist in this bin
        def nodesInBin(self, bin):
            nodes_in_this_bin = set()
            if bin in self.bin_to_alter1s and 'fm' in self.bin_to_alter1s[bin]: nodes_in_this_bin |= self.bin_to_alter1s[bin]['fm']
            if bin in self.bin_to_alter1s and 'to' in self.bin_to_alter1s[bin]: nodes_in_this_bin |= self.bin_to_alter1s[bin]['to']
            if bin in self.bin_to_alter2s and 'fm' in self.bin_to_alter2s[bin]: nodes_in_this_bin |= self.bin_to_alter2s[bin]['fm']
            if bin in self.bin_to_alter2s and 'to' in self.bin_to_alter2s[bin]: nodes_in_this_bin |= self.bin_to_alter2s[bin]['to']
            return nodes_in_this_bin

        # nodesExistInOtherBins() - return the set of nodes that exist in this bin AND'ed with all the other bins
        def nodesExistsInOtherBins(self, bin):
            nodes_in_this_bin = self.nodesInBin(bin)
            all_other_bins    = set()
            for _bin_ in (self.bin_to_alter1s.keys()|self.bin_to_alter2s.keys()):
                if _bin_ == bin: continue
                all_other_bins |= self.nodesInBin( _bin_)
            return nodes_in_this_bin & all_other_bins

        # __dateFormat__() - various date formats based on the value of self.every
        def __dateFormat__(self):
            if   'd' in self.every: return '%Y-%m-%d'
            elif 'h' in self.every: return '%Y-%m-%d %H'
            else:                   return '%Y-%m-%d %H:%M'

        # packagle() - pack the nodes into the available space
        def packable(self, nodes, x, y, y_max, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer):
            node_to_xy = {}
            h = abs(y - y_max)
            n = len(nodes)
            if n > 0:
                # single strand
                r = ((h - (n-1)*circle_inter_d)/n)/2.0
                if r >= r_min:
                    r          = min(r, r_pref)
                    left_overs = 0
                    out_of     = n
                    for _node_i_ in range(len(nodes)):
                        _node_ = nodes[-(_node_i_+1)]
                        #if mul == -1: _node_ = nodes[_node_i_]
                        #else:         _node_ = nodes[-(_node_i_+1)]
                        node_to_xy[_node_] = (x, y+mul*r, r)
                        y += mul*(2*r+circle_inter_d)
                else:
                    # m-strands
                    m_max = w_max / (2*r_min+circle_spacer)
                    for m in range(2,int(m_max)+1):
                        r = (h - (n//m)*circle_inter_d)/(n//m)/2.0
                        if r >= r_min:
                            r = min(r, r_pref)
                            total_width_required = m*(2*r) + (m-1)*circle_spacer
                            if total_width_required > w_max: continue
                            _col_, nodes_in_this_column = 0, 0
                            nodes_per_column = n//m
                            left_overs       = n - nodes_per_column*m
                            out_of           = nodes_per_column
                            if left_overs > 0: m += 1
                            total_width_required = m*(2*r) + (m-1)*circle_spacer
                            _columns_ = []
                            _column_  = []
                            for _node_ in nodes:
                                _x_col_ = x - total_width_required/2.0 + _col_*(2*r+circle_spacer) + r
                                _y_row_ = y+mul*r+mul*nodes_in_this_column*(2*r+circle_inter_d)                        
                                _column_.append((_x_col_, _y_row_))
                                nodes_in_this_column += 1
                                if nodes_in_this_column >= nodes_per_column: 
                                    _columns_.append(_column_)
                                    _column_  = []
                                    _col_, nodes_in_this_column = _col_+1, 0
                            if len(_column_) > 0: _columns_.append(_column_)
                            # Allocate the across first... and then down...
                            _xi_, _yi_ = 0, 0
                            for _node_i_ in range(len(nodes)):
                                if mul == -1: _node_ = nodes[len(nodes) - 1 - _node_i_]
                                else:         _node_ = nodes[_node_i_]
                                if _yi_ >= len(_columns_[_xi_]): _yi_, _xi_ = _yi_ + 1, 0
                                _xy_ = _columns_[_xi_][_yi_]
                                node_to_xy[_node_] = (_xy_[0], _xy_[1], r) 
                                _xi_ += 1
                                if _xi_ >= len(_columns_): _yi_, _xi_ = _yi_ + 1, 0
                            break

            if len(node_to_xy) == 0: return None, None, None
            return node_to_xy, left_overs, out_of

        #
        # renderAlter()  - render an alter / this is just to render the nodes (or clouds) within the alter
        # ... the actual shape of the bin & (alters too) is rendered elsewhere
        #
        # def renderAlter(self, nodes, befores, afters, x, y, y_max, w_max, mul=1, r_min=4.0, r_pref=7.0, circle_inter_d=2.0, circle_spacer=3, h_collapsed_sections=16):
        def renderAlter(self, nodes, befores, afters, x, y, y_max, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections, _bin_, _alter_, _alter_side_):
            # Bounds state & node positioning
            xmin, ymin, xmax, ymax = x, y, x, y
            node_to_xyrepstat = {} # node to (x, y, representation, stat) where representation is ['single','cloud'] and state is ['start,'stop','isolated','continuous']
            h   = abs(y_max - y)
            svg = []
            # Determine the state of the node
            def nodeState(seen_before, seen_after):
                if   seen_before and seen_after: return 'continuous' # node is seen both before and after this bin
                elif seen_before:                return 'stopped'    # node was seen before this bin (but not after)
                elif seen_after:                 return 'started'    # node seen after this bin (but not before)
                else:                            return 'isolated'   # node is only seen in this bin (and no other bin)
            # Create the started/stopped triangles for a single node
            def svgTriangle(x,y,r,s,d):
                nonlocal xmin, ymin, xmax, ymax
                p0      = (x+d*(r/2.0), y)
                p1      = (x+d*(r+s),   y+r)
                p2      = (x+d*(r+s),   y-r)
                for _pt_ in [p0,p1,p2]: xmin, ymin, xmax, ymax = min(xmin, _pt_[0]), min(ymin, _pt_[1]), max(xmax, _pt_[0]), max(ymax, _pt_[1])
                _path_  = f'M {p0[0]} {p0[1]} L {p1[0]} {p1[1]} L {p2[0]} {p2[1]} Z'
                _color_ = '#ff0000' if d == 1 else '#0000ff'
                return f'<path d="{_path_}" stroke="none" fill="{_color_}" />'
            # Create the started/stopped triangles for the clouds
            def svgCloudTriangle(x,y,offset,s,d):
                nonlocal xmin, ymin, xmax, ymax
                p0      = (x+d*(offset), y)
                p1      = (x+d*(offset+s),   y+s)
                p2      = (x+d*(offset+s),   y-s)
                for _pt_ in [p0,p1,p2]: xmin, ymin, xmax, ymax = min(xmin, _pt_[0]), min(ymin, _pt_[1]), max(xmax, _pt_[0]), max(ymax, _pt_[1])
                _path_  = f'M {p0[0]} {p0[1]} L {p1[0]} {p1[1]} L {p2[0]} {p2[1]} Z'
                _color_ = '#d3494e' if d == 1 else '#658cbb'
                return f'<path d="{_path_}" stroke="none" fill="{_color_}" />'
            # Place the nodes onto the canvas
            def placeNodeToXYs(n2xy):
                nonlocal xmin, ymin, xmax, ymax, svg
                for _node_, _xyr_ in n2xy.items():
                    _color_ = self.__nodeColor__(_node_)
                    svg.append(f'<circle cx="{_xyr_[0]}" cy="{_xyr_[1]}" r="{_xyr_[2]}" stroke="{_color_}" stroke-width="1.25" fill="none"/>')
                    xmin, ymin, xmax, ymax = min(xmin, _xyr_[0]-_xyr_[2]), min(ymin, _xyr_[1]-_xyr_[2]), max(xmax, _xyr_[0]+_xyr_[2]), max(ymax, _xyr_[1]+_xyr_[2])
                    if _node_ not in befores: svg.append(svgTriangle(_xyr_[0], _xyr_[1], _xyr_[2], circle_spacer/2, -1))
                    if _node_ not in afters:  svg.append(svgTriangle(_xyr_[0], _xyr_[1], _xyr_[2], circle_spacer/2,  1))
                    _xyrepstat_ = (_xyr_[0], _xyr_[1], 'single', nodeState(_node_ in befores, _node_ in afters), _bin_, _alter_, _alter_side_, (_xyr_[2]))
                    node_to_xyrepstat[_node_] = _xyrepstat_
                    self.bin_to_node_to_xyrepstat[_bin_][_node_] = _xyrepstat_
            # Render the summarization cloud
            def summarizationCloud(n, y_cloud, ltriangle, rtriangle, nodes_in_cloud):
                nonlocal xmin, ymin, xmax, ymax, svg
                svg.append(self.rt_self.iconCloud(x,y_cloud, fg='#e0e0e0', bg='#e0e0e0'))
                if ltriangle: svg.append(svgCloudTriangle(x, y_cloud, 16, 6, -1))
                if rtriangle: svg.append(svgCloudTriangle(x, y_cloud, 16, 6,  1))
                svg.append(self.rt_self.svgText(str(n), x, y_cloud + 4, 'black', anchor='middle'))
                xmin, ymin, xmax, ymax = min(xmin, x-16), min(ymin, y_cloud-6), max(xmax, x+16), max(ymax, y_cloud+6)
                for _node_ in nodes_in_cloud:
                    _xyrepstat_                                  = (x, y_cloud, 'cloud', nodeState(not ltriangle, not rtriangle), _bin_, _alter_, _alter_side_, (None))
                    node_to_xyrepstat[_node_]                    = _xyrepstat_
                    self.bin_to_node_to_xyrepstat[_bin_][_node_] = _xyrepstat_
            # Make sure there are nodes...
            if len(nodes) > 0:
                # Sort the nodes into the 4 categories
                nodes_sorter = []
                nodes_isolated, nodes_started, nodes_stopped, nodes_continuous = [], [], [], []
                for _node_ in nodes:
                    if   _node_ in befores and _node_ in afters: nodes_sorter.append((3, _node_)), nodes_continuous.append(_node_)
                    elif _node_ in befores:                      nodes_sorter.append((2, _node_)), nodes_stopped   .append(_node_)
                    elif _node_ in afters:                       nodes_sorter.append((1, _node_)), nodes_started   .append(_node_)
                    else:                                        nodes_sorter.append((0, _node_)), nodes_isolated  .append(_node_)
                nodes_sorter = sorted(nodes_sorter) # , reverse=(mul > 0))
                
                if self.only_render_nodes is not None:
                    continuous_set, isolated_set, started_set, stopped_set = set(), set(), set(), set()
                    nodes_ordered = []
                    for i in range(len(nodes_sorter)):
                        _node_ = nodes_sorter[i][1]
                        if   _node_ in self.only_render_nodes: nodes_ordered.  append(_node_)
                        elif _node_ in nodes_continuous:       continuous_set. add   (_node_)
                        elif _node_ in nodes_isolated:         isolated_set.   add   (_node_)
                        elif _node_ in nodes_started:          started_set.    add   (_node_)
                        elif _node_ in nodes_stopped:          stopped_set.    add   (_node_)
                    ybase = ymin if mul < 0 else ymax
                    if len(nodes_ordered) > 0:
                        node_to_xy, leftovers, out_of = self.packable(nodes_ordered, x, y, y_max, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer)
                        if node_to_xy is not None:    placeNodeToXYs(node_to_xy) # no summarization necessary
                        else:                         summarizationCloud(len(nodes_ordered),  ybase+mul*0.5*h_collapsed_sections, False, False, nodes_ordered)
                    ybase = ymin if mul < 0 else ymax
                    if len(continuous_set) > 0:     
                        summarizationCloud(len(continuous_set), ybase+mul*0.5*h_collapsed_sections, False, False, list(continuous_set))
                        ybase = ymin if mul < 0 else ymax
                    if len(started_set)    > 0:     
                        summarizationCloud(len(started_set),    ybase+mul*0.5*h_collapsed_sections, True,  False, list(started_set))
                        ybase = ymin if mul < 0 else ymax
                    if len(stopped_set)    > 0:     
                        summarizationCloud(len(stopped_set),    ybase+mul*0.5*h_collapsed_sections, False, True,  list(stopped_set))
                        ybase = ymin if mul < 0 else ymax
                    if len(isolated_set)   > 0:     
                        summarizationCloud(len(isolated_set),   ybase+mul*0.5*h_collapsed_sections, True,  True,  list(isolated_set))
                else:
                    # Try putting them all down first... which won't work for any non-trivial number of nodes
                    nodes_ordered = [x[1] for x in nodes_sorter]
                    node_to_xy, leftovers, out_of = self.packable(nodes_ordered, x, y, y_max, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer)
                    if node_to_xy is not None:
                        placeNodeToXYs(node_to_xy) # no summarization necessary
                    else:
                        top_adjust = h_collapsed_sections if mul == 1 else -h_collapsed_sections
                        node_to_xy, leftovers, out_of = self.packable(nodes_started+nodes_stopped+nodes_continuous, x, y, y_max-top_adjust, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer)
                        if node_to_xy is not None:
                            placeNodeToXYs(node_to_xy) # summarize isolated nodes only
                            y_off = ymin if mul == 1 else ymax
                            if len(nodes_isolated) > 0: summarizationCloud(len(nodes_isolated), y_off+mul*0.5*h_collapsed_sections, True, True, nodes_isolated)
                        else:
                            top_adjust = 2*h_collapsed_sections if mul == 1 else -2*h_collapsed_sections
                            node_to_xy, leftovers, out_of = self.packable(nodes_started              +nodes_continuous, x, y, y_max-top_adjust, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer)
                            if node_to_xy is not None:
                                placeNodeToXYs(node_to_xy) # summarize isolated nodes and nodes_stopped
                                y_off = ymax if mul == 1 else ymin
                                if len(nodes_stopped)  > 0: summarizationCloud(len(nodes_stopped),  y_off+mul*0.5*h_collapsed_sections, False,  True, nodes_stopped)
                                y_off = ymax if mul == 1 else ymin
                                if len(nodes_isolated) > 0: summarizationCloud(len(nodes_isolated), y_off+mul*0.5*h_collapsed_sections, True,   True, nodes_isolated)
                            else:
                                node_to_xy, leftovers, out_of = self.packable(nodes_stopped+nodes_continuous, x, y, y_max-top_adjust, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer)
                                if node_to_xy is not None:
                                    placeNodeToXYs(node_to_xy) # summarize isolated nodes and nodes_started
                                    y_off = ymax if mul == 1 else ymin
                                    if len(nodes_started)  > 0: summarizationCloud(len(nodes_started),   y_off+mul*0.5*h_collapsed_sections, True,  False, nodes_started)
                                    y_off = ymax if mul == 1 else ymin
                                    if len(nodes_isolated) > 0: summarizationCloud(len(nodes_isolated),  y_off+mul*0.5*h_collapsed_sections, True,  True,  nodes_isolated)
                                else:
                                    top_adjust = 3*h_collapsed_sections if mul == 1 else -3*h_collapsed_sections
                                    node_to_xy, leftovers, out_of = self.packable(nodes_continuous, x, y, y_max-top_adjust, w_max, mul, r_min, r_pref, circle_inter_d, circle_spacer)
                                    if node_to_xy is not None:
                                        placeNodeToXYs(node_to_xy) # summarize everyting but the continuous nodes (nodes seen in both directions)
                                        y_off = ymax if mul == 1 else ymin
                                        if len(nodes_started)  > 0: summarizationCloud(len(nodes_started),   y_off+mul*0.5*h_collapsed_sections, True,  False, nodes_started)
                                        y_off = ymax if mul == 1 else ymin
                                        if len(nodes_stopped)  > 0: summarizationCloud(len(nodes_stopped),   y_off+mul*0.5*h_collapsed_sections, False, True,  nodes_stopped)
                                        y_off = ymax if mul == 1 else ymin
                                        if len(nodes_isolated) > 0: summarizationCloud(len(nodes_isolated),  y_off+mul*0.5*h_collapsed_sections, True,  True,  nodes_isolated)
                                    else:
                                        # everything is summarized :(
                                        y_off = ymax if mul == 1 else ymin
                                        if len(nodes_continuous) > 0: summarizationCloud(len(nodes_continuous), y_off+mul*0.5*h_collapsed_sections, False,  False, nodes_continuous)
                                        y_off = ymax if mul == 1 else ymin
                                        if len(nodes_started)    > 0: summarizationCloud(len(nodes_started),    y_off+mul*0.5*h_collapsed_sections, True,   False, nodes_started)
                                        y_off = ymax if mul == 1 else ymin
                                        if len(nodes_stopped)    > 0: summarizationCloud(len(nodes_stopped),    y_off+mul*0.5*h_collapsed_sections, False,  True,  nodes_stopped)
                                        y_off = ymax if mul == 1 else ymin
                                        if len(nodes_isolated)   > 0: summarizationCloud(len(nodes_isolated),   y_off+mul*0.5*h_collapsed_sections, True,   True,  nodes_isolated)
                
            xmin, ymin, xmax, ymax = xmin - r_pref, ymin - r_pref, xmax + r_pref, ymax + r_pref
            # svg.append(f'<rect x="{xmin}" y="{ymin}" width="{xmax-xmin}" height="{ymax-ymin}" stroke="{self.rt_self.co_mgr.getTVColor("axis","major")}" stroke-width="0.8" fill="none" rx="{r_pref}" />')
            return ''.join(svg), (xmin, ymin, xmax, ymax), node_to_xyrepstat

        # bubbleNumberOnLine() - draw a bubble with a number on a line
        def bubbleNumberOnLine(self, x0, x1, y, txt, txt_h=12, color="#e0e0e0", width=3.0):
            xm = (x0+x1)/2.0
            txt_w = self.rt_self.textLength(txt, txt_h)
            x0_1  = xm - 3*txt_w/4.0
            x0_2  = x0_1 - txt_w/2
            x1_1  = xm + 3*txt_w/4.0
            x1_2  = x1_1 + txt_w/2
            y_top = y-txt_h/2 - 2
            y_bot = y+txt_h/2 + 2
            h     = txt_h+4
            svg   = []
            #svg.append(f'<line x1="{x0}" y1="{y}" x2="{x0_1}" y2="{y}" stroke="{color}" stroke-width="{width}" />')
            #svg.append(f'<line x1="{x1}" y1="{y}" x2="{x1_1}" y2="{y}" stroke="{color}" stroke-width="{width}" />')
            #svg.append(f'<rect x="{x0_1}" y="{y_top}" width="{1.5*txt_w}" height="{h}" fill="{color}" />')
            p = [f'M {x0} {y} L {x0_2} {y}']
            p.append(f'C {x0_1} {y} {x0_2} {y_top} {x0_1} {y_top}')
            p.append(f'L {x1_1} {y_top}')
            p.append(f'C {x1_2} {y_top} {x1_1} {y} {x1_2} {y}')
            p.append(f'L {x1} {y} L {x1_2} {y}')
            p.append(f'C {x1_1} {y} {x1_2} {y_bot} {x1_1} {y_bot}')
            p.append(f'L {x0_1} {y_bot}')
            p.append(f'C {x0_2} {y_bot} {x0_1} {y} {x0_2} {y}')
            p.append(f'L {x0} {y}')
            svg.append(f'<path d="{" ".join(p)}" stroke="{color}" stroke-width="{width}" fill="{color}"/>')
            svg.append(self.rt_self.svgText(txt, xm, y+txt_h/3, txt_h=txt_h, anchor='middle'))
            return ''.join(svg)


        # svgCrossConnect() - draws a cross connect
        def svgCrossConnect(self, x0, y0, x1, y1, launch=None, shift0=None, shift1=None, color="#000000", width=1.0):
            if launch is None: launch = (x1-x0)*0.1
            if shift0 is None: shift0 = 0
            if shift1 is None: shift1 = 0
            xmid = (x0+x1)/2
            return f'<path d="M {x0} {y0} L {x0+launch} {y0} C {xmid+shift0} {y0} {xmid-shift1} {y1} {x1-launch} {y1} L {x1} {y1}" stroke="{color}" stroke-width="{width}" fill="none" />'

        #
        # renderBin()
        #
        def renderBin(self, 
                      bin,                        # bin index
                      x,                          # center of the bin 
                      y,                          # center of the bin
                      max_w,                      # max width of the bin (i.e., the max width of any of the alters)
                      max_h):                     # max height of the bin (halfed in each direction from y)      
            r_min                = self.r_min 
            r_pref               = self.r_pref
            circle_inter_d       = self.circle_inter_d
            circle_spacer        = self.circle_spacer
            alter_separation_h   = self.alter_separation_h
            h_collapsed_sections = self.h_collapsed_sections

            self.bin_to_node_to_xyrepstat[bin] = {}

            _all_nodes_in_this_bin = self.nodesInBin(bin)
            _nodes_in_other_bins_  = self.nodesExistsInOtherBins(bin)
            _befores_, _afters_    = set(), set()
            for i in range(bin):                                       _befores_ |= self.nodesInBin(i)
            for i in range(bin+1, len(self.bin_to_timestamps.keys())): _afters_  |= self.nodesInBin(i)
            
            # Render the focal node(s) first
            if len(self.bin_to_focal_nodes_present[bin]) == 1: 
                _svg_rep_ = 'single'
                svg = [f'<circle cx="{x}" cy="{y}" r="{r_pref}" stroke="{self.rt_self.co_mgr.getTVColor("axis","minor")}" stroke-width="0.4" fill="{self.rt_self.co_mgr.getTVColor("data","default")}" />']
            else:                                              
                _svg_rep_ = 'cloud'
                svg =[self.rt_self.iconCloud(x,y, fg='#e0e0e0', bg='#e0e0e0')]
                svg.append(self.rt_self.svgText(str(len(self.bin_to_focal_nodes_present[bin])), x, y + 4, 'black', anchor='middle'))
            _xyrepstat_ = (x, y, _svg_rep_, 'continuous', bin, None, None, (r_pref)) # middlenotation
            for _node_ in self.bin_to_focal_nodes_present[bin]: self.bin_to_node_to_xyrepstat[bin][_node_] = _xyrepstat_

            max_alter_h = max_h/5.0

            node_2_xyrs = dict()

            # Actual alters
            if bin in self.bin_to_alter1s and 'fm' in self.bin_to_alter1s[bin]:
                _svg_, _bounds_, _n2xyrs_ = self.renderAlter(self.bin_to_alter1s[bin]['fm'], _befores_, _afters_, x, y-r_pref-2*circle_inter_d, y-r_pref-max_alter_h,                  max_w, -1, r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections, bin, 1, 'fm')
                svg.append(_svg_), node_2_xyrs.update(_n2xyrs_)
                alter1s_fm_bounds = _bounds_
            else:
                alter1s_fm_bounds = None
                _bounds_          = (x-r_pref, y-r_pref-2*circle_inter_d-5, x+r_pref, y-r_pref-2*circle_inter_d)

            if bin in self.bin_to_alter2s and 'fm' in self.bin_to_alter2s[bin] and len(self.bin_to_alter2s[bin]['fm']) > 0:
                _svg_, _bounds_, _n2xyrs_ = self.renderAlter(self.bin_to_alter2s[bin]['fm'], _befores_, _afters_, x, _bounds_[1]-alter_separation_h, _bounds_[1]-alter_separation_h-max_alter_h, max_w, -1, r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections, bin, 2, 'fm')
                svg.append(_svg_), node_2_xyrs.update(_n2xyrs_)
                alter2s_fm_bounds = _bounds_
            else: alter2s_fm_bounds = None

            if bin in self.bin_to_alter1s and 'to' in self.bin_to_alter1s[bin]:
                _svg_, _bounds_, _n2xyrs_ = self.renderAlter(self.bin_to_alter1s[bin]['to'], _befores_, _afters_, x, y+r_pref+2*circle_inter_d, y+r_pref+2*circle_inter_d+max_alter_h, max_w,  1, r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections, bin, 1, 'to')
                svg.append(_svg_), node_2_xyrs.update(_n2xyrs_)
                alter1s_to_bounds = _bounds_
            else: 
                _bounds_ = (x-r_pref, y+r_pref+2*circle_inter_d, x+r_pref, y+r_pref+2*circle_inter_d+5)
                alter1s_to_bounds = None

            if bin in self.bin_to_alter2s and 'to' in self.bin_to_alter2s[bin] and len(self.bin_to_alter2s[bin]['to']) > 0:
                _svg_, _bounds_, _n2xyrs_ = self.renderAlter(self.bin_to_alter2s[bin]['to'], _befores_, _afters_, x, _bounds_[3]+alter_separation_h, _bounds_[3]+alter_separation_h+max_alter_h, max_w, 1, r_min, r_pref, circle_inter_d, circle_spacer, h_collapsed_sections, bin, 2, 'to')
                svg.append(_svg_), node_2_xyrs.update(_n2xyrs_)
                alter2s_to_bounds = _bounds_
            else: alter2s_to_bounds = None

            # Calculate the outline of the bin
            _w_  = 3*r_pref
            if alter1s_fm_bounds is not None: _w_ = max(_w_, alter1s_fm_bounds[2]-alter1s_fm_bounds[0])
            if alter1s_to_bounds is not None: _w_ = max(_w_, alter1s_to_bounds[2]-alter1s_to_bounds[0])
            if alter2s_fm_bounds is not None: _w_ = max(_w_, alter2s_fm_bounds[2]-alter2s_fm_bounds[0])
            if alter2s_to_bounds is not None: _w_ = max(_w_, alter2s_to_bounds[2]-alter2s_to_bounds[0])
            _w2_ = _w_/2.0

            # path_description = f'M {x-_w2_} {y+r_pref} L {x+_w2_} {y+r_pref} L {x+_w2_} {y-r_pref} L {x-_w2_} {y-r_pref} Z'
            _ind_ = max(r_pref,_w_/4.0) # indentation

            # Bottom alters
            pd = [f'M {x-_w2_} {y}']
            if alter1s_to_bounds is not None:
                pd.append(f'L {x-_w2_} {alter1s_to_bounds[3]}')
                if alter2s_to_bounds is not None:
                    pd.append(f'L {x-_w2_+_ind_} {alter1s_to_bounds[3]}')
                    pd.append(f'L {x-_w2_+_ind_} {alter2s_to_bounds[1]}')
                    pd.append(f'L {x-_w2_}       {alter2s_to_bounds[1]}')
                    pd.append(f'L {x-_w2_}       {alter2s_to_bounds[3]}')
                    pd.append(f'L {x+_w2_}       {alter2s_to_bounds[3]}')
                    pd.append(f'L {x+_w2_}       {alter2s_to_bounds[1]}')
                    pd.append(f'L {x+_w2_-_ind_} {alter2s_to_bounds[1]}')
                    pd.append(f'L {x+_w2_-_ind_} {alter1s_to_bounds[3]}')
                    pd.append(f'L {x+_w2_}       {alter1s_to_bounds[3]}')
                else:
                    pd.append(f'L {x+_w2_} {alter1s_to_bounds[3]}')
            else:
                pd.append(f'L {x-_w2_} {y+r_pref} L {x+_w2_} {y+r_pref}')
            pd.append(f'L {x+_w2_} {y}')

            # Top alters
            if alter1s_fm_bounds is not None:
                pd.append(f'L {x+_w2_} {alter1s_fm_bounds[1]}')
                if alter2s_fm_bounds is not None:
                    pd.append(f'L {x+_w2_-_ind_} {alter1s_fm_bounds[1]}')
                    pd.append(f'L {x+_w2_-_ind_} {alter2s_fm_bounds[3]}')
                    pd.append(f'L {x+_w2_}       {alter2s_fm_bounds[3]}')
                    pd.append(f'L {x+_w2_}       {alter2s_fm_bounds[1]}')
                    pd.append(f'L {x-_w2_}       {alter2s_fm_bounds[1]}')
                    pd.append(f'L {x-_w2_}       {alter2s_fm_bounds[3]}')
                    pd.append(f'L {x-_w2_+_ind_} {alter2s_fm_bounds[3]}')
                    pd.append(f'L {x-_w2_+_ind_} {alter1s_fm_bounds[1]}')
                    pd.append(f'L {x-_w2_}       {alter1s_fm_bounds[1]}')
                else:
                    pd.append(f'L {x-_w2_} {alter1s_fm_bounds[1]}')
            else:
                pd.append(f'L {x+_w2_} {y-r_pref} L {x-_w2_} {y-r_pref}')
            pd.append(f'Z')

            path_description = ' '.join(pd)

            def pathBounds(s):
                x0, y0, x1, y1 = 1e10, 1e10, -1e10, -1e10
                ps = s.split()
                i  = 0
                while i < len(ps):
                    if ps[i] == "M":
                        xm, ym = float(ps[i+1]), float(ps[i+2])
                        x0, y0, x1, y1 = min(x0,xm), min(y0,ym), max(x1,xm), max(y1,ym)
                        i += 3
                    elif ps[i] == "L":
                        xl, yl = float(ps[i+1]), float(ps[i+2])
                        x0, y0, x1, y1 = min(x0,xl), min(y0,yl), max(x1,xl), max(y1,yl)
                        i += 3
                    elif ps[i] == "C":
                        cx0, cy0 = float(ps[i+1]), float(ps[i+2])
                        cx1, cy1 = float(ps[i+3]), float(ps[i+4])
                        xc,  yc  = float(ps[i+5]), float(ps[i+6])
                        x0, y0, x1, y1 = min(x0,xc), min(y0,yc), max(x1,xc), max(y1,yc)
                        i += 7
                    elif ps[i] == 'Z': 
                        i += 1
                    else: raise Exception(f"Unknown command {ps[i]}")
                return x0, y0, x1, y1

            svg.append(f'<path d="{self.rt_self.svgSmoothPath(path_description)}" stroke="{self.rt_self.co_mgr.getTVColor("axis","major")}" stroke-width="2.0" fill="none" />')

            return ''.join(svg), pathBounds(path_description), node_2_xyrs

        # __nodeColor__() - determine the color of a node... still need to do "vary"
        def __nodeColor__(self, _node_):
            if   self.node_color is None:                                      _color_ = self.rt_self.co_mgr.getTVColor('axis','major')
            elif self.node_color == 'node':                                    _color_ = self.rt_self.co_mgr.getColor(_node_)
            elif type(self.node_color) is dict and _node_ in self.node_color:  _color_ = self.rt_self.getColor(self.node_color[_node_])
            else:                                                              _color_ = self.rt_self.co_mgr.getTVColor('axis','major')
            return _color_

        #
        # renderSVG()
        #
        def renderSVG(self):            
            svg = []

            # Reset the following before the rendering starts (these were init'd in the __init__() method too)
            self.vx0, self.vy0, self.vx1, self.vy1 = None, None, None, None # view bounds
            self.bin_to_bounds            = {}
            self.bin_to_node_to_xyrepstat = {}

            # Assign nodes to bins
            self.__binNeighboringNodes__()

            alter_inter_d   = self.alter_inter_d        # distance between the bins
            max_bin_w       = self.max_bin_w            # maximum width of a bin
            max_bin_h       = self.max_bin_h            # maximum height of a bin (this is an approximation)
            min_channel_w   = self.min_channel_w        # size of the channel (width of the channel bar ... but i don't think it's used that way)
            max_channel_w   = self.max_channel_w        # ditto
            channel_inter_d = self.channel_inter_d      # distance to use to separate channels

            # Bin Creation
            _bins_ordered_ = list(self.bin_to_timestamps.keys())
            _bins_ordered_.sort()
            bin_to_n2xyrs  = {}
            x, y = alter_inter_d, (self.h-max_bin_h)/2 + max_bin_h/2
            for _bin_ in _bins_ordered_:
                _svg_, _bounds_, _n2xyrs_ = self.renderBin(_bin_, x, y, max_bin_w, max_bin_h)
                bin_to_n2xyrs     [_bin_] = _n2xyrs_
                self.bin_to_bounds[_bin_] = _bounds_
                svg.append(_svg_)
                xmin, ymin, xmax, ymax = _bounds_
                x = xmax + alter_inter_d
                if self.vx0 is None: self.vx0, self.vy0, self.vx1, self.vy1 = _bounds_[0], _bounds_[1], _bounds_[2], _bounds_[3]
                self.vx0, self.vy0, self.vx1, self.vy1 = min(self.vx0, _bounds_[0]-alter_inter_d/3.0), min(self.vy0, _bounds_[1] - 3*channel_inter_d), max(self.vx1, _bounds_[2]+alter_inter_d/3.0), max(self.vy1, _bounds_[3]+3*channel_inter_d)

            # Determine if two bounds overlap - used to separate channels (prevent overlap between channels)
            def boundsOverlap(a,b): return a[0] < b[0]+b[2] and a[0]+a[2] > b[0] and a[1] < b[1]+b[3] and a[1]+a[3] > b[1]

            # Channel Allocation
            bin_to_nodes_to_channel                    = {}
            max_nodes_to_channel, min_nodes_to_channel = 0, 1e10
            tuple_to_channel_geometry                  = {}
            channel_tuples                             = []

            for _fm_to_ in ['to','fm']:
                for i in range(len(_bins_ordered_)-1, 1, -1):
                    _bin_   = _bins_ordered_[i]
                    _nodes_ = set()                                                                    # Get all the nodes in the "fm" side of this bin
                    if _bin_ in self.bin_to_alter1s and _fm_to_ in self.bin_to_alter1s[_bin_]: _nodes_ |= self.bin_to_alter1s[_bin_][_fm_to_]
                    if _bin_ in self.bin_to_alter2s and _fm_to_ in self.bin_to_alter2s[_bin_]: _nodes_ |= self.bin_to_alter2s[_bin_][_fm_to_]

                    _nodes_  = _nodes_ - self.nodesInBin(_bins_ordered_[i-1])                             # These will be direct connects / so don't need to channel them

                    if _fm_to_ == 'fm': y_clearance = self.bin_to_bounds[_bins_ordered_[i-1]][1] - max_channel_w - channel_inter_d # The channel has to clear this height (this is at the "top")
                    else:               y_clearance = self.bin_to_bounds[_bins_ordered_[i-1]][3] + max_channel_w + channel_inter_d # The channel has to clear this height (this is at the "bottom")

                    _befores_ = set()                                                                         # All the nodes before this bin
                    for j in range(i): _befores_ |= self.nodesInBin(_bins_ordered_[j])
                    _nodes_                         = _nodes_ & _befores_                                     # These are now all the nodes that we need to channel...
                    number_of_nodes_in_this_channel = len(_nodes_)
                    max_nodes_to_channel, min_nodes_to_channel = max(len(_nodes_), max_nodes_to_channel), min(len(_nodes_), min_nodes_to_channel)

                    if len(_nodes_) > 0:                                                                               # If there are any nodes to channel...
                        _saving_for_later_ = []
                        for j in range(i-2, -1, -1):
                            _here_       = _bins_ordered_[j]
                            _here_nodes_ = self.nodesInBin(_here_)
                            if len(_nodes_ & _here_nodes_) > 0: 
                                for _node_ in _nodes_ & _here_nodes_: _saving_for_later_.append((_bin_, _here_, _node_))
                            _nodes_ = _nodes_ - _here_nodes_                                      
                            if len(_nodes_) == 0: break                                                                                    # If there are no more nodes to channel, we're done
                            if _fm_to_ == 'fm': y_clearance = min(y_clearance, self.bin_to_bounds[_here_][1] - max_channel_w - channel_inter_d) # Otherwise, we have to clear this height
                            else:               y_clearance = max(y_clearance, self.bin_to_bounds[_here_][3] + max_channel_w + channel_inter_d) # Otherwise, we have to clear this height
                        _channel_tuple_ = (_here_, _bin_, y_clearance, number_of_nodes_in_this_channel, _fm_to_)                           # start bin -> end bin, y_clearance, number of nodes, fm-to side
                        channel_tuples.append(_channel_tuple_)                                                                             # will determine the actual geometry later
                        for _saved_ in _saving_for_later_:
                            _to_bin_, _fm_bin_, _node_ = _saved_
                            if _fm_bin_ not in bin_to_nodes_to_channel:           bin_to_nodes_to_channel[_fm_bin_]           = {}
                            if _to_bin_ not in bin_to_nodes_to_channel[_fm_bin_]: bin_to_nodes_to_channel[_fm_bin_][_to_bin_] = {}
                            if _node_ in bin_to_nodes_to_channel[_fm_bin_][_to_bin_]: raise Exception(f'Duplicate node {_node_} in bin {_bin_} -> {_to_bin_}')
                            bin_to_nodes_to_channel[_fm_bin_][_to_bin_][_node_]                                               = _channel_tuple_
                
            # Sort the channels & render the channels
            _channel_max_y_ = 0
            channel_tuples.sort(key=lambda x: x[2]) # slightly non-optimal... because the two sides (fm, to) should be sorted in opposite directions
            for i in range(len(channel_tuples)):
                _start_, _end_, _y_, _n_, _fm_to_ = channel_tuples[i]
                _div_                             = (max_nodes_to_channel - min_nodes_to_channel)
                if _div_ == 0:  _h_               = min_channel_w
                else:           _h_               = (_n_ - min_nodes_to_channel)/_div_ * (max_channel_w - min_channel_w) + min_channel_w
                _w_                               = self.bin_to_bounds[_end_][0] - self.bin_to_bounds[_start_][2] - 1.5*alter_inter_d
                _x_                               = self.bin_to_bounds[_start_][2] + alter_inter_d

                placement_okay = False
                while placement_okay == False:
                    placement_okay = True
                    for _other_ in tuple_to_channel_geometry:
                        _geom_ = tuple_to_channel_geometry[_other_]
                        if boundsOverlap((_geom_[0], _geom_[1]-channel_inter_d, _geom_[2], _geom_[3]+2*channel_inter_d), (_x_, _y_-channel_inter_d, _w_, _h_+2*channel_inter_d)):
                            placement_okay = False
                            break
                    if not placement_okay: 
                        if _fm_to_ == 'fm':  _y_ -= channel_inter_d
                        else:                _y_ += channel_inter_d
                
                self.vy0 = min(self.vy0, _y_       - 3*channel_inter_d)
                self.vy1 = max(self.vy1, _y_ + _h_ + 3*channel_inter_d)

                tuple_to_channel_geometry[channel_tuples[i]] = (_x_, _y_, _w_, _h_)
                svg.append(self.bubbleNumberOnLine(_x_, _x_ + _w_, _y_ + _h_/2.0, str(_n_), txt_h=12, color=self.rt_self.co_mgr.getTVColor('axis','major'), width=2.0))
                _channel_max_y_ = max(_y_ + _h_ + self.txt_h, _channel_max_y_)

            # Draw the direct connects & the channel connections
            for i in range(len(_bins_ordered_)-1):
                _bin0_     = _bins_ordered_[i]
                _bin1_     = _bins_ordered_[i+1]
                _bounds0_  = self.bin_to_bounds[_bin0_]
                _bounds1_  = self.bin_to_bounds[_bin1_]

                _already_drawn_ = set()

                # direct connects
                _nodes_dc_ = bin_to_n2xyrs[_bin0_].keys() & bin_to_n2xyrs[_bin1_].keys()
                for _node_ in _nodes_dc_:
                    _x0_, _y0_, _r0_, _s0_, _b_, _alt_, _altsd_, _render_info_ = bin_to_n2xyrs[_bin0_][_node_]
                    _x1_, _y1_, _r1_, _s1_, _b_, _alt_, _altsd_, _render_info_ = bin_to_n2xyrs[_bin1_][_node_]
                    _coords_ = (_bounds0_[2], _y0_, _bounds1_[0], _y1_)
                    if _coords_ not in _already_drawn_:
                        _color_ = self.__nodeColor__(_node_)
                        # Render the direct connection & records that it has been rendered -- may prevent node_color == 'vary' from rendering correctly
                        svg.insert(0, self.svgCrossConnect(_bounds0_[2], _y0_, _bounds1_[0], _y1_, color=_color_, width=1.5))
                        _already_drawn_.add(_coords_)
                
                # channel connections
                for _fm_to_ in ['fm','to']: # different alter sides
                    if _bin0_ in bin_to_nodes_to_channel: # across the bins
                        for _bin_n_ in bin_to_nodes_to_channel[_bin0_]: # which bins does _bin0_ connect to?
                            for _node_ in bin_to_nodes_to_channel[_bin0_][_bin_n_]: # pick up those nodes
                                _xyrs_             = bin_to_n2xyrs[_bin0_][_node_]
                                _channel_tuple_    = bin_to_nodes_to_channel[_bin0_][_bin_n_][_node_]
                                _channel_geometry_ = tuple_to_channel_geometry[_channel_tuple_]
                                _halfway_          = _bounds1_[0]
                                if _halfway_ < _channel_geometry_[0]: _halfway_ = _channel_geometry_[0]
                                _channel_vmiddle_  = _channel_geometry_[1] + _channel_geometry_[3]/2.0
                                _coords_           = (_bounds0_[2], _xyrs_[1], _channel_geometry_[0], _channel_vmiddle_)
                                if _coords_ not in _already_drawn_:
                                    svg.insert(0, self.svgCrossConnect(_bounds0_[2], _xyrs_[1], _halfway_, _channel_vmiddle_, color=self.rt_self.co_mgr.getTVColor('axis','major'), width=2.0))
                                    _already_drawn_.add(_coords_)
                                _xyrs_endpt_       = bin_to_n2xyrs[_bin_n_][_node_]
                                _boundsn_          = self.bin_to_bounds[_bin_n_]
                                _boundsn_minus_1_  = self.bin_to_bounds[_bin_n_-1]
                                _halfway_          = (_boundsn_minus_1_[2] + _boundsn_[0])/2.0
                                _coords_           = (_boundsn_[0], _xyrs_endpt_[1], _channel_geometry_[0] + _channel_geometry_[2], _channel_vmiddle_)
                                if _coords_ not in _already_drawn_:
                                    svg.insert(0, self.svgCrossConnect(_boundsn_[0], _xyrs_endpt_[1], _channel_geometry_[0] + _channel_geometry_[2], _channel_vmiddle_, color=self.rt_self.co_mgr.getTVColor('axis','major'), width=2.0))
                                    _already_drawn_.add(_coords_)

            # Add information about missing bins (because no data existed at that point in time)
            for _bin_ in self.bin_to_bounds:
                _bounds_        = self.bin_to_bounds[_bin_]
                _channel_max_y_ = max(_bounds_[3], _channel_max_y_)
            _hrun_, _vrun_ = self.r_pref*1.25, self.r_pref*2
            for _bin_ in self.discontinuity_count_after_bin:
                if _bin_ == 0: continue # don't bother drawing before the first bin
                _missing_number_ = self.discontinuity_count_after_bin[_bin_]
                if _bin_ not in self.bin_to_bounds: continue
                _bounds_         = self.bin_to_bounds[_bin_]
                _x_              = _bounds_[0] - self.alter_inter_d/2.0
                _y_              = self.vy0
                _d_              = [f'M {_x_-_hrun_} {_y_}']
                _y_             += _vrun_
                _turns_          = int(1+(self.vy1-self.vy0)/_vrun_)
                for i in range(_turns_):
                    if i%2 == 0: _d_.append(f'L {_x_+_hrun_} {_y_}')
                    else:        _d_.append(f'L {_x_-_hrun_} {_y_}')
                    _y_ += _vrun_
                svg.insert(0, f'<path d="{" ".join(_d_)}" stroke="{self.rt_self.co_mgr.getTVColor("context","highlight")}" stroke-width="0.5" fill="none" stroke-dasharray="3 3"/>')

            # Draw labels (if selected)
            if self.draw_labels:
                for _bin_ in self.bin_to_bounds:
                    _bounds_        = self.bin_to_bounds[_bin_]
                    _channel_max_y_ = max(_bounds_[3] + self.txt_h, _channel_max_y_)
                for _bin_ in self.bin_to_timestamps:
                    if _bin_ in self.bin_to_bounds:
                        _timestamp_ = self.bin_to_timestamps[_bin_]
                        _bounds_    = self.bin_to_bounds[_bin_]
                        _x_         = (_bounds_[0] + _bounds_[2])/2.0
                        svg.insert(0, self.rt_self.svgText(self.formatTimestamp(_timestamp_), _x_, _channel_max_y_, txt_h=self.txt_h, anchor='middle'))
                self.vy1 = _channel_max_y_ + self.txt_h

            # Add the header and the footer
            if self.vx0 is None: self.vx0, self.vy0, self.vx1, self.vy1, xmin, xmax = 0.0, 0.0, 1.0, 1.0, 0.0, 1.0
            if self.include_svg_viewbox: svg.insert(0, f'<svg x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" viewBox="{self.viewBox()}">')
            else:                        svg.insert(0, f'<svg x="0" y="0" width="{self.vx1+self.vx0}" height="{self.vy1+self.vy0}" >')
            svg.insert(1, f'<rect x="{self.vx0}" y="{self.vy0}" width="{self.vx1-self.vx0}" height="{self.vy1-self.vy0}" fill="{self.rt_self.co_mgr.getTVColor("background","default")}" />')
            svg.insert(2, f'<line x1="{alter_inter_d}" y1="{y}" x2="{x-alter_inter_d - (xmax-xmin)/2}" y2="{y}" stroke="{self.rt_self.co_mgr.getTVColor("axis","major")}" stroke-width="3.0" />')

            # Conclude the SVG
            svg.append('</svg>')
            self.last_render = ''.join(svg)
            return self.last_render

        #
        # viewBox()
        #
        def viewBox      (self): 
            if self.vx0 is None: self._repr_svg_() # force a render
            return f'{self.vx0} {self.vy0} {self.vx1-self.vx0} {self.vy1-self.vy0}'
        def viewBoxRect  (self): 
            if self.vx0 is None: self._repr_svg_() # force a render
            return self.vx0, self.vy0, self.vx1-self.vx0, self.vy1-self.vy0
        def viewBoxBounds(self): 
            if self.vx0 is None: self._repr_svg_() # force a render
            return self.vx0, self.vy0, self.vx1, self.vy1


        #
        # worldXYToScreenXY() - convert from widget coordinates to SVG coordinates
        # 
        # wx = (vx + vw/2.0) + (sx - self.w/2.0)*_ratio_
        # wx - (vx + vw/2.0) = (sx - self.w/2.0)*_ratio_
        # self.w/2.0 + (wx - (vx + vw/2.0))/_ratio_ = sx
        #
        def worldXYToScreenXY(self, wxy):
            vx, vy, vw, vh = self.viewBoxRect()
            if ((vw/vh) > (self.w/self.h)): _ratio_ = vw/self.w
            else:                           _ratio_ = vh/self.h
            return self.w/2.0 + (wxy[0] - (vx + vw/2.0))/_ratio_, self.h/2.0 + (wxy[1] - (vy + vh/2.0))/_ratio_

        #
        # from the javascript code 
        #
        #sw        = svgparent.getAttribute("width"); sh        = svgparent.getAttribute("height"); _ratio_   = data.vw/sw;
        #if ((data.vw/data.vh) > (sw/sh)) { _ratio_ = data.vw/sw; } else { _ratio_ = data.vh/sh;}
        #state.x_trans = (data.vx + data.vw/2) + (state.x_raw - sw/2)*_ratio_; state.y_trans = (data.vy + data.vh/2) + (state.y_raw - sh/2)*_ratio_;

        #
        # screenXYToWorldXY() - convert from SVG coordinates to widget coordinates
        # - adapted from the rt_spreadlines_interactive_panel transform
        # - inverse of the above function
        #
        def screenXYToWorldXY(self, sxy):
            vx, vy, vw, vh = self.viewBoxRect()
            if ((vw/vh) > (self.w/self.h)): _ratio_ = vw/self.w
            else:                           _ratio_ = vh/self.h
            return (vx + vw/2.0) + (sxy[0] - self.w/2.0)*_ratio_, (vy + vh/2.0) + (sxy[1] - self.h/2.0)*_ratio_ 

        #
        # formatTimestamp()
        #
        def formatTimestamp(self, timestamp):
            if   'h' in self.every: _format_ = "%Y-%m-%d %H"
            elif 'd' in self.every: _format_ = "%Y-%m-%d"
            else                  : _format_ = "%Y-%m-%d %H:%M:%S"
            return timestamp.strftime(_format_)

        #
        # overlappingEntities() - Determine which entity geometrics overlap with a specific region
        # - to_intersect should be a shapely shape
        # - return value is a list of entities (possibly an empty list) or None
        # - copied from rt_link_mixin
        #
        def overlappingEntities(self, to_intersect):
            _set_ = set()
            # Determine which bins overlap with the to_intersect region
            for _bin_ in self.bin_to_bounds:
                _bounds_ = self.bin_to_bounds[_bin_]
                # If the bin overlaps, go through the nodes in the bin
                if to_intersect.intersects(Polygon([(_bounds_[0],_bounds_[1]),(_bounds_[2],_bounds_[1]),(_bounds_[2],_bounds_[3]),(_bounds_[0],_bounds_[3])])):
                    for _node_ in self.bin_to_node_to_xyrepstat[_bin_]:
                        _xyrepstat_ = self.bin_to_node_to_xyrepstat[_bin_][_node_]
                        _xy_        = (_xyrepstat_[0], _xyrepstat_[1])
                        if to_intersect.contains(Point(_xy_)): _set_.add(_node_)
            return list(_set_)

        #
        # entitiesAtPoint() - Determine all the entities under a specific point
        #
        def entitiesAtPoint(self, xy):
            _poly_ = Polygon([(xy[0]-5,xy[1]-5),(xy[0]-5,xy[1]+5),(xy[0]+5,xy[1]+5),(xy[0]+5,xy[1]-5)])            
            return self.overlappingEntities(_poly_)

        #
        # __createPathDescriptionOfSelectedEntities__()
        #
        def __createPathDescriptionOfSelectedEntities__(self, my_selection=None):
            if my_selection is None: my_selection = self.selected_entities
            _xys_ = set()
            for _bin_ in self.bin_to_node_to_xyrepstat:
                for _entity_ in my_selection:
                    if _entity_ in self.bin_to_node_to_xyrepstat[_bin_]:
                        _xyrepstat_ = self.bin_to_node_to_xyrepstat[_bin_][_entity_]
                        _xy_        = (_xyrepstat_[0], _xyrepstat_[1])
                        _xys_.add(_xy_)
            _strs_ = []
            for _xy_ in _xys_: _strs_.append(f'M {_xy_[0]-5} {_xy_[1]-5} l 10 0 l 0 10 l -10 0 z')
            return ' '.join(_strs_)

        #
        # __createPathDescriptionForAllEntities__()
        #
        def __createPathDescriptionForAllEntities__(self):
            _xys_ = set()
            for _bin_ in self.bin_to_node_to_xyrepstat:
                for _entity_ in self.bin_to_node_to_xyrepstat[_bin_]:
                    _xyrepstat_ = self.bin_to_node_to_xyrepstat[_bin_][_entity_]
                    _xy_        = (_xyrepstat_[0], _xyrepstat_[1])
                    _xys_.add(_xy_)
            _strs_ = []
            for _xy_ in _xys_: _strs_.append(f'M {_xy_[0]-5} {_xy_[1]-5} l 10 0 l 0 10 l -10 0 z')
            return ' '.join(_strs_)

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None: self.renderSVG()
            return self.last_render

        #
        # entityPositions() - return the position(s) of an entity
        # - an entity may exist in more than one bin necessitating a list
        #
        def entityPositions(self, entity):
            _results_ = []
            for _bin_ in self.bin_to_node_to_xyrepstat:
                if entity in self.bin_to_node_to_xyrepstat[_bin_]:
                    _xyrepstat_ = self.bin_to_node_to_xyrepstat[_bin_][entity]
                    _xy_        = (_xyrepstat_[0], _xyrepstat_[1])
                    if _xyrepstat_[2] == 'single': _svg_markup_ = f'<circle cx="{_xy_[0]}" cy="{_xy_[1]}" r="{_xyrepstat_[7]}" stroke="#000000" stroke-width="1.25" fill="none"/>'
                    else:                          _svg_markup_ = self.rt_self.iconCloud(_xy_[0], _xy_[1], fg='#e0e0e0', bg='#e0e0e0')
                    rtep = RTEntityPosition(entity, self.rt_self, self, _xy_, (_xy_[0], _xy_[1], 0.0, 1.0), 
                                            None, _svg_markup_, self.widget_id)
                    _results_.append(rtep)
            return _results_

        def setNodeFocus(self, node_focus):
            self.node_focus = node_focus
            if   type(self.node_focus) is list: self.node_focus = set(self.node_focus)
            elif type(self.node_focus) is not set: self.node_focus = set([self.node_focus])
            self.invalidateRender()
