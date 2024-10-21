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
import copy
import random
import heapq
import time
import networkx as nx

from math import pi, sin, cos, ceil, floor, sqrt

from shapely.geometry import Polygon

from .rt_component       import RTComponent
from .rt_entity_position import RTEntityPosition

__name__ = 'rt_chord_diagram_mixin'

#
# Chord Diagram Mixin
#
class RTChordDiagramMixin(object):
    # concats two strings in alphabetical order
    def __den_fromToString__(self, x, fm, to, _connector_ = ' <-|-> ', _type_sep_ = '|>>>|'):
        _fm_, _to_ = str(x[fm]), str(x[to])
        if type(x[fm]) != str:
            _fm_ += _type_sep_+str(type(x[fm]))
            if x[fm] != self.__den_retyped__(_fm_): raise Exception(f'RTChordDiagramMixin.__den_fromToString__() - type mismatch between "{x[fm]}" and "{_fm_}" [fm]')
        if type(x[to]) != str: 
            _to_ += _type_sep_+str(type(x[to]))
            if x[to] != self.__den_retyped__(_to_): raise Exception(f'RTChordDiagramMixin.__den_fromToString__() - type mismatch between "{x[to]}" and "{_to_}" [to]')
        return (_fm_+_connector_+_to_) if (_fm_<_to_) else (_to_+_connector_+_fm_)
    # separates the concatenated string back into it's two parts
    def __den_fromToStringParts__(self, x, _connector_ = ' <-|-> '):
        i = x.index(_connector_)
        fm = x[:i]
        to = x[i+len(_connector_):]
        return fm, to
    # merges names (which themselves can be merged names)
    def __den_mergedName__(self, a, b, _sep_ = '|||'):
        ls = a.split(_sep_)
        ls.extend(b.split(_sep_))
        return _sep_.join(ls)
    # separates merged names back into parts
    def __den_breakdownMerge__(self, a, _sep_ = '|||'):
        return a.split(_sep_)
    def __den_retyped__(self, x, _type_sep_ = '|>>>|'):
        if _type_sep_ in x:
            x_type = x[x.index(_type_sep_)+len(_type_sep_):]
            x      = x[:x.index(_type_sep_)]
            if   x_type == 'int'    or x_type == '<class \'int\'>'   or \
                 x_type == 'long'   or x_type == '<class \'long\'>'  or \
                 x_type == '<class \'numpy.int64\'>':                   x = int(x)
            elif x_type == 'float'  or x_type == '<class \'float\'>'  or \
                 x_type == 'double' or x_type == '<class \'double\'>' or \
                 x_type == '<class \'numpy.float64\'>':                 x = float(x)
            else: raise Exception('RTChordDiagramMixin.__den_retyped__() - unknown x type: ' + x_type + f' for "{x}"')
        return x

    # fixes the type inclusion for the list
    def __den_fixType__(self, still_typed, _type_sep_ = '|>>>|'):
        fixed_typed = []
        for x in still_typed: 
            if _type_sep_ in x: fixed_typed.append(self.__den_retyped__(x, _type_sep_))
            else:               fixed_typed.append(x)
        return fixed_typed

    # __dendrogramHelper_pandas__()
    def __dendrogramHelper_pandas__(self, df, fm, to, count_by, count_by_set):
        # concats two strings in alphabetical order        
        df = self.copyDataFrame(df)
        df['__fmto__'] = df.apply(lambda x: self.__den_fromToString__(x, fm, to), axis=1)
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
            x, y = self.__den_fromToStringParts__(r['__fmto__'])
            if x != y:
                if x not in _graph_.keys():
                    _graph_[x] = {}
                _graph_[x][y] = -r[count_by]
                if y not in _graph_.keys():
                    _graph_[y] = {}
                _graph_[y][x] = -r[count_by]

        return _heap_, _graph_

    # __dendrogramHelper_polars__()
    def __dendrogramHelper_polars__(self, df, fm, to, count_by, count_by_set):
        # concats two strings together in alphabetical order
        df = self.copyDataFrame(df)
        __lambda__ = lambda x: self.__den_fromToString__(x, fm, to)
        df = df.with_columns(pl.struct([fm,to]).map_elements(__lambda__, return_dtype=pl.String).alias('__fmto__'))
        df_den = self.polarsCounter(df, '__fmto__', count_by, count_by_set)

        # create the initial graph and heap
        count_by_col , fmto_col = df_den['__count__'], df_den['__fmto__']
        _heap_ , _graph_ = [] , {}
        for i in range(len(df_den)):
            heapq.heappush(_heap_,(-count_by_col[i], fmto_col[i]))
            x, y = self.__den_fromToStringParts__(fmto_col[i])
            if x != y:
                if x not in _graph_.keys():
                    _graph_[x] = {}
                _graph_[x][y] = -count_by_col[i]
                if y not in _graph_.keys():
                    _graph_[y] = {}
                _graph_[y][x] = -count_by_col[i]

        return _heap_, _graph_

    #
    # dendrogramOrdering() - create an order of the fm/to nodes based on hierarchical clustering
    #
    def dendrogramOrdering(self, df, fm, to, count_by, count_by_set, _sep_ = '|||', _connector_ = ' <-|-> '):
        # perform the dataframe summation
        if   self.isPandas(df):
            _heap_, _graph_ = self.__dendrogramHelper_pandas__(df, fm, to, count_by, count_by_set)
        elif self.isPolars(df):
            _heap_, _graph_ = self.__dendrogramHelper_polars__(df, fm, to, count_by, count_by_set)
        else:
            raise Exception('dendrogramOrdering() - only handles pandas or polars')

        # iteratively merge the closest nodes together
        _tree_ = {}
        _merged_already_ = set()
        while len(_heap_) > 0:
            _strength_, _fmto_ = heapq.heappop(_heap_)
            _fm_, _to_ = self.__den_fromToStringParts__(_fmto_)
            if _fm_ != _to_ and _fm_ not in _merged_already_ and _to_ not in _merged_already_:
                _merged_already_.add(_fm_), _merged_already_.add(_to_)
                _new_ = self.__den_mergedName__(_fm_, _to_)
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
                    heapq.heappush(_heap_,(_graph_[_new_][x], _new_ + _connector_ + x))
                # Remove the old nodes and their nbor connections
                for x in _graph_[_fm_]:
                    _graph_[x].pop(_fm_)
                _graph_.pop(_fm_)
                for x in _graph_[_to_]:
                    _graph_[x].pop(_to_)
                _graph_.pop(_to_)

        # ensure that there's a root node...
        if len(_graph_.keys()) > 1:
            _root_parts_ = []
            for x in _graph_.keys():
                _root_parts_.extend(self.__den_breakdownMerge__(x))
            _root_ = _sep_.join(sorted(_root_parts_))
            _tree_[_root_] = []
            for x in _graph_.keys():
                _tree_[_root_].append(x)

        # walk a tree in leaf order
        def leafWalk(t, n=None):
            if n is None:
                for x in t.keys():
                    n = x if (n is None) or (len(x) > len(n)) else n # root will be longest string
            if _sep_ not in n or n not in t.keys():
                return [n]
            elif len(t[n]) > 2:
                _extended_ = []
                for i in range(len(t[n])):
                    lw = leafWalk(t, t[n][i])
                    if lw is not None:
                        _extended_.extend(lw)
                return _extended_
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
                    _extended_ = r
                    _extended_.extend(l)
                    return _extended_
        
        return self.__den_fixType__(leafWalk(_tree_))

    #
    # dendrogramOrdering() - create an order of the fm/to nodes based on hierarchical clustering
    #
    def dendrogramOrdering_HDBSCAN(self, df, fm, to, count_by, count_by_set, _sep_ = '|||'):
        if self.isPandas(df):
            df = self.copyDataFrame(df)
            df['__fmto__'] = df.apply(lambda x: self.__den_fromToString__(x, fm, to), axis=1)
            if count_by is None:
                df_den   = df.groupby('__fmto__').size().reset_index().rename({0:'__countby__'},axis=1)
                count_by = '__countby__'
            elif count_by_set:
                df_den = df.groupby('__fmto__')[count_by].nunique().reset_index()
            else:
                df_den = df.groupby('__fmto__')[count_by].sum().reset_index()
            dist_lu, dist_max = {}, 1.0
            for row_i,row in df_den.iterrows():                
                _count_ = row[count_by]
                dist_max = max(_count_,dist_max)                
                _fm_, _to_ = self.__den_fromToStringParts__(row['__fmto__'])
                if _fm_ not in dist_lu:
                    dist_lu[_fm_] = {}
                dist_lu[_fm_][_to_] = _count_
        elif self.isPolars(df):
            # concats two strings together in alphabetical order
            df = self.copyDataFrame(df)
            __lambda__ = lambda x: self.__den_fromToString__(x, fm, to)
            df = df.with_columns(pl.struct([fm,to]).map_elements(__lambda__).alias('__fmto__'))
            df_den = self.polarsCounter(df, '__fmto__', count_by, count_by_set)
            count_by = '__count__'
            dist_lu, dist_max = {}, 1.0
            for i in range(len(df_den)):
                _count_ = df_den[count_by][i]
                dist_max = max(_count_,dist_max)
                _fm_, _to_ = self.__den_fromToStringParts__(df_den['__fmto__'][i])
                if _fm_ not in dist_lu:
                    dist_lu[_fm_] = {}
                dist_lu[_fm_][_to_] = _count_            
        else:
            raise Exception('RTChordDiagram.dendrogramOrdering_HDBSCAN() - only accepts pandas or polars')

        # Arrange the items in the appropriate structure
        items_actual = list(set(df[fm]) | set(df[to]))
        items        = [(int(x),) for x in range(len(items_actual))]

        # Create a custom distance function
        def __dist__(ai,bi):
            a = items_actual[int(ai[0])]
            b = items_actual[int(bi[0])]
            if (a,b) in dist_lu:
                return 0.01 + 1.0 - dist_lu[(a,b)]/dist_max
            elif (b,a) in dist_lu:
                return 0.01 + 1.0 - dist_lu[(b,a)]/dist_max
            else:
                return 2.0
        
        # Cluster the fms/tos...
        import hdbscan
        clusterer = hdbscan.HDBSCAN(metric=__dist__)
        clusterer.fit(items)

        _not_this_ = '''
        # Construct and disect the single linkage tree from the clustering operation
        fms, tos, children, all, parent_to_children = [],[], set(), set(), {}
        for edge in clusterer.single_linkage_tree_.to_networkx().edges():
            _fm_, _to_ = int(edge[0]), int(edge[1])
            fms.append(_fm_), tos.append(_to_)
            children.add(_to_), all.add(_fm_), all.add(_to_)
            if _fm_ not in parent_to_children.keys():
                parent_to_children[_fm_] = []
            parent_to_children[_fm_].append(_to_)
        root = (all - children).__iter__().__next__()

        # Perform a leaf walk
        def __leafWalk__(node):
            if node in parent_to_children.keys():
                ls = []
                for child in parent_to_children[node]:
                    ls_children = __leafWalk__(child)
                    if ls_children is not None:
                        ls.extend(ls_children)
                return ls
            else:
                return [node]
        leaves_in_order = __leafWalk__(root)
        leaves_in_order_actual = []
        for i in leaves_in_order:
            leaves_in_order_actual.append(items_actual[i])
        return leaves_in_order_actual
        '''

        _df_ = clusterer.condensed_tree_.to_pandas()
        leaves_in_order = list(_df_[_df_['child_size'] == 1]['child'])
        leaves_in_order_actual = []
        for i in leaves_in_order:
            leaves_in_order_actual.append(items_actual[i])
        return leaves_in_order_actual

    # __dendrogramHelperTuples_pandas__()
    def __dendrogramHelperTuples_pandas__(self, df, fm, to, count_by, count_by_set):
        # concats two strings in alphabetical order   
        df = self.copyDataFrame(df)
        df['__fmto__'] = df.apply(lambda x: self.__den_fromToString__(x, fm, to), axis=1)
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
            x, y = self.__den_fromToStringParts__(r['__fmto__'])
            heapq.heappush(_heap_,(-r[count_by], ((x,),(y,))))
            if x != y:
                if (x,) not in _graph_.keys():
                    _graph_[(x,)] = {}
                _graph_[(x,)][(y,)] = -r[count_by]
                if (y,) not in _graph_.keys():
                    _graph_[(y,)] = {}
                _graph_[(y,)][(x,)] = -r[count_by]
        return _heap_, _graph_
    
    # __dendrogramHelperTuples_polars__()
    def __dendrogramHelperTuples_polars__(self, df, fm, to, count_by, count_by_set):
        # concats two strings together in alphabetical order
        df = self.copyDataFrame(df)
        __lambda__ = lambda x: self.__den_fromToString__(x, fm, to)
        df = df.with_columns(pl.struct([fm,to]).map_elements(__lambda__, return_dtype=pl.String).alias('__fmto__'))
        df_den = self.polarsCounter(df, '__fmto__', count_by, count_by_set)

        # create the initial graph and heap
        count_by_col , fmto_col = df_den['__count__'], df_den['__fmto__']
        _heap_ , _graph_ = [] , {}
        for i in range(len(df_den)):
            x, y = self.__den_fromToStringParts__(fmto_col[i])
            heapq.heappush(_heap_,(-count_by_col[i], ((x,),(y,))))
            if x != y:
                if (x,) not in _graph_.keys():
                    _graph_[(x,)] = {}
                _graph_[(x,)][(y,)] = -count_by_col[i]
                if (y,) not in _graph_.keys():
                    _graph_[(y,)] = {}
                _graph_[(y,)][(x,)] = -count_by_col[i]
        return _heap_, _graph_

    #
    # dendorgramOrderingTuples() - yet another version attempting to fix the suboptimal nature of the original version...
    #
    def dendrogramOrderingTuples(self, df, fm, to, count_by, count_by_set, _sep_ = '|||'):
        if   self.isPandas(df):
            _heap_,_graph_ = self.__dendrogramHelperTuples_pandas__(df, fm, to, count_by, count_by_set)
        elif self.isPolars(df):
            _heap_,_graph_ = self.__dendrogramHelperTuples_polars__(df, fm, to, count_by, count_by_set)
        else:
            raise Exception('RTChordDiagram.dendrogramOrderingTuples() - only pandas and polars implemented')

        _graph_orig_ = copy.deepcopy(_graph_)

        def optimalArrangement(t0,t1):
            if len(t0) == 1 and len(t1) == 1:
                return t0 + t1
            elif len(t0) == 1:
                f,b = 0,0
                for i in range(len(t1)):
                    if (t1[i],) in _graph_orig_[t0].keys():
                        s = _graph_orig_[t0][(t1[i],)]
                        f += s * 1/(1+i)
                        b += s * 1/(len(t1)-i)
                if f > b:
                    return t1 + t0
                else:
                    return t0 + t1
            elif len(t1) == 1:
                f,b = 0,0
                for i in range(len(t0)):
                    if (t0[i],) in _graph_orig_[t1].keys():
                        s = _graph_orig_[t1][(t0[i],)]
                        f += s * 1/(1+i)
                        b += s * 1/(len(t0)-i)
                if f > b:
                    return t0 + t1
                else:
                    return t1 + t0
            else:
                # print('happens!') # does this actually happen? ... sigh... yes it does )
                pass
            return t0 + t1

        _merged_already_ = set()
        while len(_heap_) > 0:
            _strength_, _fmto_ = heapq.heappop(_heap_)
            _fm_, _to_ = _fmto_
            if type(_fm_) != tuple:
                _fm_ = (_fm_,)
            if type(_to_) != tuple:
                _to_ = (_to_,)
            if _fm_ != _to_ and _fm_ not in _merged_already_ and _to_ not in _merged_already_:
                _merged_already_.add(_fm_), _merged_already_.add(_to_)
                _new_ = optimalArrangement(_fm_, _to_)
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
                    heapq.heappush(_heap_,(_graph_[_new_][x], (_new_, x)))
                # Remove the old nodes and their nbor connections
                for x in _graph_[_fm_]:
                    _graph_[x].pop(_fm_)
                _graph_.pop(_fm_)
                for x in _graph_[_to_]:
                    _graph_[x].pop(_to_)
                _graph_.pop(_to_)
        _tuple_ = ()
        for k in _graph_.keys():
            _tuple_ += k
        return self.__den_fixType__(list(_tuple_))

    #
    # hierarchicalCIDRParentLookups() - helper method to create parent lookups for CIDR hierarchies
    #
    def hierarchicalCIDRParentLookups(self, df, cols, cidr24=True, cidr16=True, cidr08=True):
        parent_lu, base = {}, set()
        for col in cols:
            base = base | set(df[col])
        for i in range(3,0,-1):
            if i == 3 and cidr24 == False:
                continue
            if i == 2 and cidr16 == False:
                continue
            if i == 1 and cidr08 == False:
                continue
            base_next = set()
            for ip in base:
                parent_lu[ip] = '.'.join(ip.split('.')[:i])
                base_next.add(parent_lu[ip])
            base = base_next
        return parent_lu

    #
    # chordDiagramPreferredDimensions()
    # - Return the preferred size
    #
    def chordDiagramPreferredDimensions(self, **kwargs):
        return (256,256)

    #
    # chordDiagramMinimumDimensions()
    # - Return the minimum size
    #
    def chordDiagramMinimumDimensions(self, **kwargs):
        return (128,128)

    #
    # chordDiagramSmallMultipleDimensions()
    # - Return the minimum size
    #
    def chordDiagramSmallMultipleDimensions(self, **kwargs):
        return (128,128)

    #
    # Identify the required fields in the dataframe from chord diagram parameters
    #
    def chordDiagramRequiredFields(self, **kwargs):
        columns_set = set()
        self.identifyColumnsFromParameters('relationships', kwargs, columns_set)
        self.identifyColumnsFromParameters('color_by',      kwargs, columns_set)
        self.identifyColumnsFromParameters('count_by',      kwargs, columns_set)
        return columns_set

    #
    # chordDiagram()
    #
    # Make the SVG for a chord diagram.
    #    
    def chordDiagram(self,
                     df,                                         # dataframe to render
                     relationships,                              # same convention as linknode [('fm','to')]
                     # ----------------------------------------- # everything else is a default...
                     color_by                   = None,          # none (default) or field name (note that node_color or link_color needs to be 'vary')
                     count_by                   = None,          # none means just count rows, otherwise, use a field to sum by
                     count_by_set               = False,         # count by summation (by default)... count_by column is checked
                     widget_id                  = None,          # naming the svg elements                 
                     # ----------------------------------------- # node options
                     node_color                 = None,          # none means color by node name, 'vary' by color_by, or specific color "#xxxxxx"
                     node_h                     = 10,            # height of node from circle edge
                     node_gap                   = 5,             # node gap in pixels (gap between the arcs)
                     node_labels                = None,          # node labels (substitutes for dataframe node values)
                     order                      = None,          # override calculated ordering... "None" in the list means user-specified gaps
                     parent_lu                  = None,          # parent lookup (if filled in & order is None, it will be calculated)
                     label_only                 = set(),         # label only set
                     equal_size_nodes           = False,         # equal size nodes
                     # ----------------------------------------- # link options
                     link_color                 = None,          # None, 'src', 'dst', 'vary' by color_by, 'shade_fm_to' to match the hierarchical bundle paper (expensive), or specific color "#xxxxxx"
                     link_opacity               = 0.5,           # link opacity
                     link_arrow                 = 'subtle',      # None, 'subtle', or 'sharp' - only applies to the "wide" linkstyle
                     arrow_px                   = 16,            # arrow size in pixels
                     arrow_ratio                = 0.05,          # arrow size as a ratio of the radius
                     link_style                 = 'narrow',      # 'narrow', 'wide', 'bundled'
                     link_size_min              = 0.8,           # for 'narrow', min link size
                     link_size_max              = 4.0,           # for 'narrow', max link size
                     global_min                 = None,          # for small multiples, this is the largest across all of the variations
                     global_max                 = None,          # for small multiples, this is the smallest across all of the variations
                     # ----------------------------------------- # small multiples config
                     structure_template         = None,          # existing RTChordDiagram() ... e.g., for small multiples
                     dendrogram_algorithm       = None,          # 'original', 'hdbscan', or None
                     skeleton_algorithm         = 'hexagonal',   # 'hexagonal', 'hdbscan', 'simple', 'kmeans'
                     skeleton_rings             = 4,             # number of rings in the skeleton
                     beta                       = 0.8,           # cubic b-spline smoothing
                     t_inc                      = 0.1,           # cubic b-spline increment (for piecewise only)
                     # ----------------------------------------- # visualization geometry / etc.
                     track_state                = False,         # track state for interactive filtering
                     track_routes               = False,         # track routes for skeleton analysis
                     x_view                     = 0,             # x offset for the view
                     y_view                     = 0,             # y offset for the view
                     w                          = 256,           # width of the view
                     h                          = 256,           # height of the view
                     x_ins                      = 3,
                     y_ins                      = 3,
                     txt_h                      = 10,            # text height for labeling
                     txt_offset                 = 0,             # text offset from the outer radius of the circle
                     draw_labels                = False,         # draw labels flag # not implemented yet
                     label_style                = 'radial',      # 'radial' or 'circular'
                     draw_border                = True,          # draw a border around the graph
                     draw_circular_background   = True,          # draw the background for just the circular part of the graph
                     draw_background            = False):        # useful to turn off in small multiples settings
        '''Implementation of a chord diagram in SVG.

        Required Parameters
        -------------------

        df : pandas.DataFrame | polars.DataFrame
            Dataframe to render.

        relationships : list[tuple]
            The list of relationships to be drawn.  Examples include:
            [('fm','to')]                # single relationship
            [('fm','to'), ('src','dst')] # multiple relationships
            [(('fm','fm_sub'),'to')]     # multi-field from relationship with single field to relationship

        Useful Parameters
        -----------------

        color_by : str
            Field name to color the edges by.

        count_by : str
            Field name to count the edges by.  None indicates count by rows.

        count_by_set : bool
            Count by set operation when set to True.  Otherwise, count by summation.

        skeleton_algorithm : str
            'hexagonal', 'hdbscan', 'simple', 'kmeans'

        skeleton_rings : int
            number of rings (for hdbscan only)
        
        label_style
            'radial' (default) or 'circular'

        Node Parameters
        ---------------

        node_color : str          
            None means color by node name, 'vary' by color_by, or specific color "#xxxxxx"

        node_h : float | int        
            Height of node from circle edge

        node_gap : float | int
            Node gap in pixels (gap between the arcs)

        node_labels : dict
            Replacement labels for the elements found within the dataframe

        order : list
            Override calculated ordering... "None" in the list means user-specified gaps

        parent_lu : dict
            Parent lookup (if filled in & order is None, it will be calculated)

        label_only : set
            Label only set

        equal_size_nodes : bool
            Use equal size nodes
            

        Link Parameters
        ---------------

        link_color : str
            None, 'src', 'dst', 'vary' by color_by, 'shade_fm_to' to match the hierarchical bundle paper (expensive), or specific color "#xxxxxx"

        link_opacity : float
            Link opacity

        link_arrow : str
            None, 'subtle' (default), or 'sharp' - only applies to the "wide" linkstyle

        arrow_px : float | int
            Arrow size in pixels

        arrow_ratio : float
            Arrow size as a ratio of the radius

        link_style : str
            'narrow' (default), 'wide', 'bundled'

        link_size_min : float
            for 'narrow', min link size

        link_size_max : float
            for 'narrow', max link size

        '''
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTChordDiagram(self, **_params_)


    #
    # createConcatColumn() - concatenate multiple columns together into a single column
    # - should be refactored with method of same name in base rtsvg.py file // 2024-05-18
    #
    def createConcatColumn(self, df, columns, new_column):
        def catFields(x, flds):
            s = str(x[flds[0]])
            for i in range(1,len(flds)):
                s += '|' + str(x[flds[i]])
            return s
        if self.isPandas(df):
            df[new_column] = df.apply(lambda x: catFields(x, columns), axis=1)
        elif self.isPolars(df):
            to_concat_new, str_casts = [], []
            for x in columns:
                if df[x].dtype != pl.String:
                    str_casts.append(pl.col(x).cast(str).alias('__' + x + '_as_str__'))
                    to_concat_new.append(pl.col('__' + x + '_as_str__'))
                else:
                    to_concat_new.append(pl.col(x))
            df = df.with_columns(*str_casts).with_columns(pl.concat_str(to_concat_new, separator='|').alias(new_column))
        else:
            raise Exception('createConcatColumn() - only pandas and polars supported')
        return df
    
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
            self.parms                  = locals().copy()
            self.rt_self                = rt_self
            self.df                     = rt_self.copyDataFrame(kwargs['df']) # still needs polars!
            self.relationships          = kwargs['relationships']             # done!
            self.color_by               = kwargs['color_by']                  # done! (nothing to handle)
            self.count_by               = kwargs['count_by']                  # done!
            self.count_by_set           = kwargs['count_by_set']              # done!
            self.widget_id              = kwargs['widget_id']                 # done!
            if self.widget_id is None:
                self.widget_id = 'chorddiagram_' + str(random.randint(0,8*65535))          
            self.node_color             = kwargs['node_color']                # done! (maybe)
            self.node_h                 = kwargs['node_h']                    # done!
            self.node_gap               = kwargs['node_gap']                  # done!
            self.node_labels            = kwargs['node_labels']
            if self.node_labels is None: self.node_labels = {}
            self.order                  = kwargs['order']                     # done!
            self.parent_lu              = kwargs['parent_lu']                 # done!
            self.label_only             = kwargs['label_only']                # done!
            self.equal_size_nodes       = kwargs['equal_size_nodes']          # done! (needs testing)
            self.link_color             = kwargs['link_color']                # done!
            self.link_opacity           = kwargs['link_opacity']              # done!
            self.link_arrow             = kwargs['link_arrow']                # done!
            self.arrow_px               = kwargs['arrow_px']                  # done!
            self.arrow_ratio            = kwargs['arrow_ratio']               # done!
            self.link_style             = kwargs['link_style']                # done!
            self.link_size_min          = kwargs['link_size_min']             # done!
            self.link_size_max          = kwargs['link_size_max']             # done!
            self.global_max             = kwargs['global_max']                # done!
            self.global_min             = kwargs['global_min']                # done!
            self.track_state            = kwargs['track_state']               # <--- still needs to be done
            self.x_view                 = kwargs['x_view']                    # n/a
            self.y_view                 = kwargs['y_view']                    # n/a
            self.w                      = kwargs['w']                         # n/a
            self.h                      = kwargs['h']                         # n/a
            self.x_ins                  = kwargs['x_ins']                     # n/a
            self.y_ins                  = kwargs['y_ins']                     # n/a
            self.txt_h                  = kwargs['txt_h']                     # done!
            self.txt_offset             = kwargs['txt_offset']                # done!
            self.draw_labels            = kwargs['draw_labels']               # done!
            self.label_style            = kwargs['label_style']               # done!
            self.draw_border            = kwargs['draw_border']               # done!
            self.draw_circular_background = kwargs['draw_circular_background'] # done!
            self.draw_background        = kwargs['draw_background']           # done!
            self.dendrogram_algorithm   = kwargs['dendrogram_algorithm']      # done!
            self.skeleton_algorithm     = kwargs['skeleton_algorithm']        # done!
            self.skeleton_rings         = kwargs['skeleton_rings']            # done!
            self.beta                   = kwargs['beta']                      # done!
            self.t_inc                  = kwargs['t_inc']                     # done!
            self.time_lu                = {}

            # Apply count-by transforms
            _ts_ = time.time()
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
            self.time_lu['transforms'] = time.time() - _ts_

            # If either from or to are lists, concat them together...
            _ts_ = time.time()
            _fm_ = self.relationships[0][0]
            if type(_fm_) == list or type(_fm_) == tuple:
                new_fm = '__fmcat__'
                self.df = self.rt_self.createConcatColumn(self.df, _fm_, new_fm)
                _fm_ = new_fm
            _to_ = self.relationships[0][1]
            if type(_to_) == list or type(_to_) == tuple:
                new_to = '__tocat__'
                self.df = self.rt_self.createConcatColumn(self.df, _to_, new_to)
                _to_ = new_to
            self.relationships = [(_fm_,_to_)]
            self.fm, self.to = _fm_, _to_
            self.time_lu['concat_columns'] = time.time() - _ts_

            # Get rid of self references
            if   self.rt_self.isPandas(self.df): self.df = self.df[self.df[_fm_] != self.df[_to_]]
            elif self.rt_self.isPolars(self.df): self.df = self.df.filter(pl.col(self.fm).cast(pl.Utf8) != pl.col(self.to).cast(pl.Utf8))
            else: raise Exception('RTChordDiagram() - only pandas and polars supported [3]')

            # Check the count_by column
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)
            
            # Create the order in parent_lu mode
            self.hierarchical_labels = set()
            self.hierarchical_to_arc = {} # (a0, a1, r0, r1)
            if self.parent_lu is not None and self.order is None:
                self.order = self.orderedChildren(self.parent_lu)

            # Tracking state
            self.geom_to_df  = {}
            self.last_render = None

            # Track routes (for bundled edges)
            self.track_routes              = kwargs['track_routes'] 
            self.track_routes_segments     = {}
            self.track_routes_segments_fms = {}
            self.track_routes_segments_tos = {}

            # Geometric construction... these members map the nodes into the circle...
            # ... manipulating these prior to render is how small multiples needs to work
            self.node_to_arc         = None
            self.node_dir_arc        = None
            self.node_to_arc_ct      = None
            self.node_dir_arc_ct     = None
            self.node_dir_arc_ct_min = None
            self.node_dir_arc_ct_max = None
            self.skeleton            = None
            self.skeleton_svg        = None
            if kwargs['structure_template'] is not None:
                other = kwargs['structure_template']
                # Force render if necessary... ### COPY OF APPLYVIEWCONFIGUATION() BELOW
                if other.node_to_arc is None:
                    other._repr_svg_()
                self.order               = other.order
                self.node_to_arc         = other.node_to_arc
                self.node_dir_arc        = other.node_dir_arc
                self.node_to_arc_ct      = other.node_to_arc_ct
                self.node_dir_arc_ct     = other.node_dir_arc_ct
                self.node_dir_arc_ct_min = other.node_dir_arc_ct_min
                self.node_dir_arc_ct_max = other.node_dir_arc_ct_max
                self.skeleton            = other.skeleton
                self.skeleton_svg        = other.skeleton_svg

        #
        # INCORRECT_orderedChildren() - original... failed to consider complete hierarchy
        #
        def INCORRECT_orderedChildren(self, lu):
            children = {}
            for child in lu:
                if lu[child] not in children:
                    children[lu[child]] = []
                children[lu[child]].append(child)
            ordered = []
            for parent in children:
                one_added = False
                for x in children[parent]:
                    if x not in children:
                        ordered.append(x)
                        one_added = True
                if one_added:
                    ordered.append(None)
            return ordered
        #
        # orderedChildren() - return a list of children ordered by parent
        # - lu is a lookup dictionary from child to parent
        # - lu[child] = parent
        # - returns a list of children ordered firstly by parent (only leaf children are included)
        # -- None is used as a separator
        #
        def orderedChildren(self, lu):
            child_lu, roots = {}, set()
            for child in lu:
                parent = lu[child]
                if parent not in child_lu:
                    child_lu[parent] = []
                child_lu[parent].append(child)
                if parent not in lu:
                    roots.add(parent)

            def leafWalk(t, root):
                if root in t:
                    _order_ = []
                    for child in t[root]:
                        _order_.extend(leafWalk(t,child))
                    _order_.append(None)
                    return _order_
                else:
                    return [root]

            order = []
            for root in roots:
                suborder = leafWalk(child_lu, root)
                order.extend(suborder)
                order.append(None)

            new_order, last_was_none = [], False
            for x in order:
                if x is None:
                    if not last_was_none:
                        new_order.append(None)
                    last_was_none = True
                else:
                    new_order.append(x)
                    last_was_none = False
                    
            return new_order

        # __entityPositions__() - return information about the entity geometry for rendering
        # - return the positions of the entity ... rendering had to have happened first
        # - list with the following tuples:
        #   (entity, xy_point_to, xy_attachment, svg_entity_id, svg_markup)
        #    ... xy_point_to   = (x,y)       // if you were to point to the entity, this would be that location
        #    ... xy_attachment = (x,y,xv,yv) // attachment location -- (x,y) is the attachment location on the entity geometry
        #    ... where xv,yv is the unit vector for exiting from this entity (zeros indicate any direction works)
        #    ... svg_entity_id = the svg entity id w/in the current markup
        #    ... svg_markup    = the (unadorned) svg markup for the entity (which may differ from the svg_entity_id)
        def __entityPositions__(self, entity):
            if entity in self.node_to_arc or entity in self.hierarchical_to_arc:
                to_rad = lambda a: a * pi/180
                if entity in self.node_to_arc:
                    a0, a1     = self.node_to_arc[entity]
                    r0, r1     = self.r - self.node_h, self.r
                    r_avg      = (r0+r1)/2
                    a_avg      = (a0+a1)/2
                    a_avg_rad  = to_rad(a_avg)
                    svg_markup = f'<path d="{self.__entityArc__(entity)}" />'
                else:
                    a0, a1, r0, r1 = self.hierarchical_to_arc[entity]
                    a_avg, r_avg   = (a0+a1)/2, (r0+r1)/2
                    a_avg_rad      = to_rad(a_avg)
                    svg_markup = f'<path d="{self.__genericArc__(a0, a1, r0, r1)}" />'
                
                def AP(radius, radians):
                    return (self.cx + (radius)*cos(radians), self.cy + (radius)*sin(radians), cos(radians), sin(radians))

                rtep = RTEntityPosition(entity,  self.rt_self,
                                                 self,
                                                (self.cx + (r_avg)*cos(a_avg_rad), 
                                                 self.cy + (r_avg)*sin(a_avg_rad)),
                                                AP(r1, a_avg_rad), 
                                                self.__entityID__(entity), 
                                                svg_markup,
                                                self.widget_id)
                if (a1 - a0) > 30.0:
                    rtep.addAttachmentPointVec(AP(r1, to_rad(a0+5)))
                    rtep.addAttachmentPointVec(AP(r1, to_rad(a1-5)))
                return [rtep]
            else:
                return []

        # entityPositions() - return information about the entity geometry for rendering
        # - return the positions of the entity ... rendering had to have happened first
        def entityPositions(self, entity_or_label):
            if entity_or_label in self.node_to_arc or entity_or_label in self.hierarchical_to_arc:
                return self.__entityPositions__(entity_or_label)
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

        #
        # applyViewConfiguration()
        # - apply the view configuration from another RTComponent (of the same type)
        # - return True if the view actually changed (and needs a re-render)
        # - COPIED INTO THE CONSTRUCTOR -- MAKE SURE TO MIRROR CHANGES
        #
        def applyViewConfiguration(self, other):
            # Force render if necessary...
            if other.node_to_arc is None:
                other._repr_svg_()
            self.order               = other.order
            self.node_to_arc         = other.node_to_arc
            self.node_dir_arc        = other.node_dir_arc
            self.node_to_arc_ct      = other.node_to_arc_ct
            self.node_dir_arc_ct     = other.node_dir_arc_ct
            self.node_dir_arc_ct_min = other.node_dir_arc_ct_min
            self.node_dir_arc_ct_max = other.node_dir_arc_ct_max
            self.skeleton            = other.skeleton
            self.skeleton_svg        = other.skeleton_svg
            return True
        
        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
            return self.last_render

        #
        # __countingCalc__() - dataframe independent counting method
        #
        def __countingCalc__(self):
            if self.rt_self.isPandas(self.df):
                return self.__countingCalc_pandas__()
            elif self.rt_self.isPolars(self.df):
                return self.__countingCalc_polars__()
            else:
                raise Exception('RTChordDiagram.__countingCalc__() - only pandas and polars supported')

        # __countingCalc_polars__() - polars verison of counting method
        def __countingCalc_polars__(self):
            # Counting methodologies
            df_fm = self.rt_self.polarsCounter(self.df, self.fm, count_by=self.count_by, count_by_set=self.count_by_set).rename({'__count__':'__fmcount__', self.fm:'__node__'})
            df_to = self.rt_self.polarsCounter(self.df, self.to, count_by=self.count_by, count_by_set=self.count_by_set).rename({'__count__':'__tocount__', self.to:'__node__'})
            df_counter = df_fm.join(df_to, on='__node__', how='full', coalesce=True).fill_null(0).with_columns((pl.col('__fmcount__') + pl.col('__tocount__')).alias('__count__'))

            # Transposition into a dictionary
            counter_lu = {}
            for i in range(len(df_counter)):
                counter_lu[df_counter['__node__'][i]] = df_counter['__count__'][i] 
            counter_sum = df_counter['__count__'].sum()

            # From-To and To-From Lookup
            fmto_lu, tofm_lu = {}, {}
            df_fm_to = self.rt_self.polarsCounter(self.df, [self.fm, self.to], count_by=self.count_by, count_by_set=self.count_by_set)
            for i in range(len(df_fm_to)):
                _fm_, _to_, __count__ = df_fm_to[self.fm][i], df_fm_to[self.to][i], df_fm_to['__count__'][i]
                if _fm_ not in fmto_lu.keys():
                    fmto_lu[_fm_] = {}
                fmto_lu[_fm_][_to_] = __count__
                if _to_ not in tofm_lu.keys():
                    tofm_lu[_to_] = {}
                tofm_lu[_to_][_fm_] = __count__

            # From-To Color Lookup
            fmto_color_lu = {}
            if self.link_color == 'vary' and self.color_by is not None and self.color_by in self.df.columns:
                if   self.color_by == self.fm or self.color_by == self.to:
                    for k, k_df in self.df.group_by([self.fm, self.to]):
                        _fm_, _to_ = k
                        if _fm_ not in fmto_color_lu.keys():
                            fmto_color_lu[_fm_] = {}
                        fmto_color_lu[_fm_][_to_] = self.rt_self.co_mgr.getColor(_fm_) if self.color_by == self.fm else self.rt_self.co_mgr.getColor(_to_)
                else:
                    df_color       = self.df.drop(set(self.df.columns)-set([self.fm,self.to,self.color_by])).group_by([self.fm,self.to]).n_unique().rename({self.color_by:'__nuniqs__'}).sort([self.fm,self.to])
                    df_color_first = self.df.drop(set(self.df.columns)-set([self.fm,self.to,self.color_by])).group_by([self.fm,self.to]).first().sort([self.fm,self.to])
                    for i in range(len(df_color)):
                        _fm_, _to_, _uniqs_ = df_color[self.fm][i], df_color[self.to][i], df_color['__nuniqs__'][i]
                        _color_ = self.rt_self.co_mgr.getColor(df_color_first[self.color_by][i]) if (_uniqs_ == 1) else \
                                  self.rt_self.co_mgr.getTVColor('data','default')
                        if _fm_ not in fmto_color_lu.keys():
                            fmto_color_lu[_fm_] = {}
                        fmto_color_lu[_fm_][_to_] = _color_

            # Node Color Lookup
            node_color_lu = {}
            if self.node_color == 'vary' and self.color_by is not None and self.color_by in self.df.columns:
                if self.color_by == self.fm or self.color_by == self.to:
                    for k, k_df in self.df.group_by([self.fm, self.to]):
                        _fm_, _to_ = k
                        node_color_lu[_fm_] = self.rt_self.co_mgr.getColor(_fm_)
                        node_color_lu[_to_] = self.rt_self.co_mgr.getColor(_to_)
                else:
                    df_fm       = self.df.drop(set(self.df.columns)-set([self.fm,self.color_by])).group_by(self.fm).n_unique().rename({self.color_by:'__nuniqs__'}).sort(self.fm)
                    df_fm_first = self.df.drop(set(self.df.columns)-set([self.fm,self.color_by])).group_by(self.fm).first().sort(self.fm)
                    df_to       = self.df.drop(set(self.df.columns)-set([self.to,self.color_by])).group_by(self.to).n_unique().rename({self.color_by:'__nuniqs__'}).sort(self.to)
                    df_to_first = self.df.drop(set(self.df.columns)-set([self.to,self.color_by])).group_by(self.to).first().sort(self.to)
                    for i in range(len(df_fm)):
                        node, _uniqs_ = df_fm[self.fm][i], df_fm['__nuniqs__'][i]
                        _color_ = self.rt_self.co_mgr.getColor(df_fm_first[self.color_by][i]) if (_uniqs_ == 1) else \
                                  self.rt_self.co_mgr.getTVColor('data','default')
                        node_color_lu[node] = _color_
                    for i in range(len(df_to)):
                        node, _uniqs_ = df_to[self.to][i], df_to['__nuniqs__'][i]
                        _color_ = self.rt_self.co_mgr.getColor(df_to_first[self.color_by][i]) if (_uniqs_ == 1) else \
                                  self.rt_self.co_mgr.getTVColor('data','default')
                        if node in node_color_lu.keys():
                            if node_color_lu[node] != _color_:
                                node_color_lu[node] = self.rt_self.co_mgr.getTVColor('data','default')
                        else:
                            node_color_lu[node] = _color_

            # If stateful tracking, partition the dataframe by fm/to
            _partition_by_ = self.df.partition_by([self.fm,self.to], as_dict=True) if self.track_state else None

            return counter_lu, counter_sum, fmto_lu, tofm_lu, fmto_color_lu, node_color_lu, _partition_by_

        # __countingCalc_pandas__() - pandas verison of counting method
        def __countingCalc_pandas__(self):            
            # Counting methodologies
            df_fm_to_gb = self.df.groupby([self.fm,self.to])
            if   self.count_by is None:
                df_fm      = self.df.groupby(self.fm).size().reset_index().rename({self.fm:'__node__',0:'__fm_count__'},axis=1)
                df_to      = self.df.groupby(self.to).size().reset_index().rename({self.to:'__node__',0:'__to_count__'},axis=1)
                df_fm_to   = df_fm_to_gb.size().reset_index().rename({0:'__count__', self.fm:'__fm__', self.to:'__to__'}, axis=1)
            elif self.count_by_set:
                df_fm      = self.df.groupby(self.fm)[self.count_by].nunique().reset_index().rename({self.fm:'__node__',self.count_by:'__fm_count__'},axis=1)
                df_to      = self.df.groupby(self.to)[self.count_by].nunique().reset_index().rename({self.to:'__node__',self.count_by:'__to_count__'},axis=1)
                df_fm_to   = df_fm_to_gb[self.count_by].nunique().reset_index().rename({self.count_by:'__count__', self.fm:'__fm__', self.to:'__to__'}, axis=1)
            else:
                df_fm      = self.df.groupby(self.fm)[self.count_by].sum().reset_index().rename({self.fm:'__node__',self.count_by:'__fm_count__'},axis=1)
                df_to      = self.df.groupby(self.to)[self.count_by].sum().reset_index().rename({self.to:'__node__',self.count_by:'__to_count__'},axis=1)
                df_fm_to   = df_fm_to_gb[self.count_by].sum().reset_index().rename({self.count_by:'__count__', self.fm:'__fm__', self.to:'__to__'}, axis=1)
            df_counter = df_fm.set_index('__node__').join(df_to.set_index('__node__'), how='outer').reset_index().fillna(0.0)
            df_counter['__count__'] = df_counter['__fm_count__'] + df_counter['__to_count__']

            # Transposition into a dictionary
            counter_lu      = {}
            for row_i, row in df_counter.iterrows():
                counter_lu[row['__node__']] = row['__count__']
            counter_sum = df_counter['__count__'].sum()

            # From-To and To-From lookup
            fmto_lu, tofm_lu = {}, {}
            for row_i, row in df_fm_to.iterrows():
                _fm_ = row['__fm__']
                _to_ = row['__to__']
                if _fm_ not in fmto_lu.keys():
                    fmto_lu[_fm_] = {}
                fmto_lu[_fm_][_to_] = row['__count__']
                if _to_ not in tofm_lu.keys():
                    tofm_lu[_to_] = {}
                tofm_lu[_to_][_fm_] = row['__count__']

            # From-To Color Lookup
            fmto_color_lu = {}
            if self.link_color == 'vary' and self.color_by is not None and self.color_by in self.df.columns:
                if   self.color_by == self.fm or self.color_by == self.to:
                    for k,k_df in df_fm_to_gb:
                        _fm_, _to_ = k
                        if _fm_ not in fmto_color_lu.keys():
                            fmto_color_lu[_fm_] = {}
                        fmto_color_lu[_fm_][_to_] = self.rt_self.co_mgr.getColor(_fm_) if self.color_by == self.fm else self.rt_self.co_mgr.getColor(_to_)
                else:
                    df_color       = self.df.groupby([self.fm,self.to])[self.color_by].nunique().reset_index().rename({self.color_by:'__nuniqs__'},axis=1)
                    df_color_first = self.df.groupby([self.fm,self.to])[self.color_by].first()
                    for row_i, row in df_color.iterrows():
                        _fm_, _to_, _uniqs_ = row[self.fm], row[self.to], row['__nuniqs__']
                        _color_ = self.rt_self.co_mgr.getColor(df_color_first.loc[_fm_,_to_]) if (_uniqs_ == 1) else \
                                  self.rt_self.co_mgr.getTVColor('data','default')
                        if _fm_ not in fmto_color_lu.keys():
                            fmto_color_lu[_fm_] = {}
                        fmto_color_lu[_fm_][_to_] = _color_

            # Node Color Lookup
            node_color_lu = {}
            if self.node_color == 'vary' and self.color_by is not None and self.color_by in self.df.columns:
                if self.color_by == self.fm or self.color_by == self.to:
                    for k,k_df in df_fm_to_gb:
                        _fm_, _to_ = k
                        node_color_lu[_fm_] = self.rt_self.co_mgr.getColor(_fm_)
                        node_color_lu[_to_] = self.rt_self.co_mgr.getColor(_to_)
                else:
                    df_fm       = self.df.groupby(self.fm)[self.color_by].nunique().reset_index().rename({self.color_by:'__nuniqs__'},axis=1)
                    df_fm_first = self.df.groupby(self.fm)[self.color_by].first()
                    df_to       = self.df.groupby(self.to)[self.color_by].nunique().reset_index().rename({self.color_by:'__nuniqs__'},axis=1)
                    df_to_first = self.df.groupby(self.to)[self.color_by].first()
                    for row_i, row in df_fm.iterrows():
                        node, _uniqs_ = row[self.fm], row['__nuniqs__']
                        _color_ = self.rt_self.co_mgr.getColor(df_fm_first.loc[node]) if (_uniqs_ == 1) else \
                                  self.rt_self.co_mgr.getTVColor('data','default')
                        node_color_lu[node] = _color_
                    for row_i, row in df_to.iterrows():
                        node, _uniqs_ = row[self.to], row['__nuniqs__']
                        _color_ = self.rt_self.co_mgr.getColor(df_to_first.loc[node]) if (_uniqs_ == 1) else \
                                  self.rt_self.co_mgr.getTVColor('data','default')
                        if node in node_color_lu.keys():
                            if node_color_lu[node] != _color_:
                                node_color_lu[node] = self.rt_self.co_mgr.getTVColor('data','default')
                        else:
                            node_color_lu[node] = _color_

            return counter_lu, counter_sum, fmto_lu, tofm_lu, fmto_color_lu, node_color_lu, df_fm_to_gb

        #
        # __genericArc__()
        #
        def __genericArc__(self, a0, a1, r_inner, r_outer):
            _fn_ = lambda a,r: (self.cx + r * cos(a * pi / 180.0), self.cy + r * sin(a * pi / 180.0))
            x0_out,  y0_out  = _fn_(a0, r_outer)
            x0_in,   y0_in   = _fn_(a0, r_inner)
            x1_out,  y1_out  = _fn_(a1, r_outer)
            x1_in,   y1_in   = _fn_(a1, r_inner)
            large_arc = 0 if (a1-a0) <= 180.0 else 1
            _path_ = f'M {x0_out} {y0_out} A {r_outer} {r_outer} 0 {large_arc} 1 {x1_out} {y1_out} L {x1_in} {y1_in} ' + \
                                        f' A {r_inner} {r_inner} 0 {large_arc} 0 {x0_in}  {y0_in}  Z'
            return _path_

        #
        # __entityArc__() - return the svg arc path
        #
        def __entityArc__(self, node):
            a0, a1 = self.node_to_arc[node]
            x0_out,  y0_out  = self.xTo(a0), self.yTo(a0)
            x0_in,   y0_in   = self.xTi(a0), self.yTi(a0)
            x1_out,  y1_out  = self.xTo(a1), self.yTo(a1)
            x1_in,   y1_in   = self.xTi(a1), self.yTi(a1)
            large_arc = 0 if (a1-a0) <= 180.0 else 1
            _path_ = f'M {x0_out} {y0_out} A {self.r} {self.r} 0 {large_arc} 1 {x1_out} {y1_out} L {x1_in} {y1_in} ' + \
                                        f' A {self.r-self.node_h} {self.r-self.node_h} 0 {large_arc} 0 {x0_in}  {y0_in}  Z'
            return _path_

        #
        # __entityID__() - return the svg entity id
        #
        def __entityID__(self, node):
            _id_ = self.rt_self.encSVGID(node)
            id_str = f'id="{self.widget_id}-{_id_}"'
            return id_str

        #
        # __renderNodes__() - render the nodes (outer edges of the circle)
        #
        def __renderNodes__(self, node_color_lu):
            svg = []

            # draw the base ring
            _color_ = self.rt_self.co_mgr.getTVColor('data','default')
            for node in self.node_to_arc.keys():
                _path_  = self.__entityArc__(node)
                if   type(self.node_color) == str and len(self.node_color) == 7 and self.node_color.startswith('#'):
                    _node_color_ = self.node_color
                elif self.color_by is not None and self.node_color == 'vary':
                    _node_color_ = node_color_lu[node]
                else:
                    _node_color_ = self.rt_self.co_mgr.getColor(str(node))
                svg.append(f'<path {self.__entityID__(node)} d="{_path_}" stroke-width="0.8" stroke="{_node_color_}" fill="{_node_color_}" />')

            # draw the hierarchical layers (further out rings)
            last_arc_lu = self.node_to_arc
            ring_r = self.r
            if self.draw_labels and self.label_style == 'circular':
                ring_r += self.txt_h + 4
            if self.parent_lu is not None:
                while len(last_arc_lu.keys()) > 0:
                    arc_lu = {}
                    for node in last_arc_lu:
                        if node in self.parent_lu:
                            parent = self.parent_lu[node]
                            a0, a1 = last_arc_lu[node]
                            if parent not in arc_lu:
                                arc_lu[parent] = (a0,a1)
                            else:
                                b0, b1 = arc_lu[parent]
                                b0, b1 = min(b0, a0), max(b1, a1)
                                arc_lu[parent] = (b0,b1)
                    for node in arc_lu:
                        a0, a1 = arc_lu[node]
                        _path_  = self.__genericArc__(a0, a1, ring_r, ring_r + self.node_h)
                        if   type(self.node_color) == str and len(self.node_color) == 7 and self.node_color.startswith('#'):
                            _node_color_ = self.node_color
                        elif self.color_by is not None and self.node_color == 'vary':
                            _node_color_ = node_color_lu[node]
                        else:
                            _node_color_ = self.rt_self.co_mgr.getColor(str(node))
                        svg.append(f'<path {self.__entityID__(node)} d="{_path_}" stroke-width="0.8" stroke="{_node_color_}" fill="{_node_color_}" />')
                        # labeling later...
                        self.hierarchical_labels.add(node)
                        self.hierarchical_to_arc[node] = (a0, a1, ring_r, ring_r + self.node_h)
                    last_arc_lu  = arc_lu
                    ring_r      += self.node_h
                    if self.draw_labels and self.label_style == 'circular':
                        ring_r += self.txt_h + 4
            return ''.join(svg)

        #
        # __renderEdges_wide__(self) - render the edges (as large filled areas)
        # ... for context, the local parts are for use in small multiples ... otherwise, the self.node_* methods are used        
        #
        def __renderEdges_wide__(self, struct_matches_render, fmto_lu, 
                                 local_dir_arc_ct, local_dir_arc_ct_min, local_dir_arc_ct_max, 
                                 fmto_color_lu):
            svg = []
            for node in self.node_dir_arc.keys():
                for _fm_ in self.node_dir_arc[node].keys():
                    if node != _fm_:
                        continue
                    for _to_ in self.node_dir_arc[node][_fm_].keys():
                        nbor = _fm_ if node != _fm_ else _to_
                        a0, a1 = self.node_dir_arc[node][_fm_][_to_]                            
                        b0, b1 = self.node_dir_arc[nbor][_fm_][_to_]
                        if struct_matches_render == False:
                            if _fm_ not in fmto_lu.keys() or _to_ not in fmto_lu[_fm_].keys():
                                continue
                            if self.node_dir_arc_ct[node][_fm_][_to_] != local_dir_arc_ct[node][_fm_][_to_]:
                                #perc = local_dir_arc_ct[node][_fm_][_to_] / self.node_dir_arc_ct[node][_fm_][_to_]
                                #perc = min(perc, 1.0)
                                perc = (local_dir_arc_ct[node][_fm_][_to_] - local_dir_arc_ct_min)/(local_dir_arc_ct_max - local_dir_arc_ct_min)
                                a1   = a0 + perc * (a1 - a0)
                                b1   = b0 + perc * (b1 - b0)

                        b_avg  = (b0+b1)/2 # for arrow points

                        xa0, ya0, xa1, ya1  = self.xTi(a0), self.yTi(a0), self.xTi(b1), self.yTi(b1)
                        xb0, yb0, xb1, yb1  = self.xTi(a1), self.yTi(a1), self.xTi(b0), self.yTi(b0)
                        xarrow0, yarrow0    = self.xTarrow(b0), self.yTarrow(b0)
                        xarrow_pt,yarrow_pt = self.xTi(b_avg),  self.yTi(b_avg)
                        xarrow1, yarrow1    = self.xTarrow(b1), self.yTarrow(b1)
                        
                        if self.link_arrow is None:
                            _path_ = f'M {xa0} {ya0} C {self.cx} {self.cy} {self.cx} {self.cy} {xa1} {ya1} ' + \
                                     f'A {self.r-self.node_h} {self.r-self.node_h} 0 0 0 {xb1} {yb1} ' + \
                                     f'C {self.cx} {self.cy} {self.cx} {self.cy} {xb0} {yb0} ' + \
                                     f'A {self.r-self.node_h} {self.r-self.node_h} 0 0 0 {xa0} {ya0}'
                        elif self.link_arrow == 'sharp':
                            _path_ = f'M {xa0} {ya0} C {self.cx} {self.cy} {self.cx} {self.cy} {xarrow1} {yarrow1} ' + \
                                     f'L {xarrow_pt} {yarrow_pt} L {xarrow0} {yarrow0} ' + \
                                     f'C {self.cx} {self.cy} {self.cx} {self.cy} {xb0} {yb0} ' + \
                                     f'A {self.r-self.node_h} {self.r-self.node_h} 0 0 0 {xa0} {ya0}'
                        else: # 'subtle'
                            _path_ = f'M {xa0} {ya0} C {self.cx} {self.cy} {self.cx} {self.cy} {xarrow1} {yarrow1} ' + \
                                     f'A {self.r-2*self.node_h} {self.r-2*self.node_h} 0 0 0 {xarrow_pt} {yarrow_pt} ' + \
                                     f'A {self.r-2*self.node_h} {self.r-2*self.node_h} 0 0 0 {xarrow0} {yarrow0} ' + \
                                     f'C {self.cx} {self.cy} {self.cx} {self.cy} {xb0} {yb0} ' + \
                                     f'A {self.r-self.node_h} {self.r-self.node_h} 0 0 0 {xa0} {ya0}'

                        # should be refactored
                        if   self.link_color is not None and type(self.link_color) == str and len(self.link_color) == 7 and self.link_color[0] == '#':
                            _link_color_ = self.link_color
                        elif self.link_color is None or self.color_by is None:
                            _link_color_ = self.rt_self.co_mgr.getTVColor('data', 'default')
                        elif self.link_color == 'src':
                            _link_color_ = self.rt_self.co_mgr.getColor(str(_fm_))
                        elif self.link_color == 'dst':
                            _link_color_ = self.rt_self.co_mgr.getColor(str(_to_))
                        else: # 'vary'
                            _link_color_ = fmto_color_lu[_fm_][_to_]

                        svg.append(f'<path d="{_path_}" stroke="{_link_color_}" stroke-opacity="1.0" fill="{_link_color_}" opacity="{self.link_opacity}" />')

            return ''.join(svg)

        #
        # __renderEdges_narrow__(self) - render the edges (links)
        # ... for context, the local parts are for use in small multiples ... otherwise, the self.node_* methods are used
        #
        def __renderEdges_narrow__(self, struct_matches_render, fmto_lu, 
                                   local_dir_arc_ct, local_dir_arc_ct_min, local_dir_arc_ct_max, 
                                   fmto_color_lu):
            svg = []
            for node in self.node_dir_arc.keys():
                for _fm_ in self.node_dir_arc[node].keys():
                    if node != _fm_:
                        continue
                    for _to_ in self.node_dir_arc[node][_fm_].keys():
                        nbor = _fm_ if node != _fm_ else _to_
                        a0, a1 = self.node_dir_arc[node][_fm_][_to_]                            
                        b0, b1 = self.node_dir_arc[nbor][_fm_][_to_]
                        link_w_perc = (self.node_dir_arc_ct[nbor][_fm_][_to_] - self.node_dir_arc_ct_min) / (self.node_dir_arc_ct_max - self.node_dir_arc_ct_min)
                        if struct_matches_render == False:
                            if _fm_ not in fmto_lu.keys() or _to_ not in fmto_lu[_fm_].keys():
                                continue
                            if self.node_dir_arc_ct[node][_fm_][_to_] != local_dir_arc_ct[node][_fm_][_to_]:
                                #perc = local_dir_arc_ct[node][_fm_][_to_] / self.node_dir_arc_ct[node][_fm_][_to_]
                                #perc = min(perc, 1.0)
                                perc = (local_dir_arc_ct[node][_fm_][_to_] - local_dir_arc_ct_min)/(local_dir_arc_ct_max - local_dir_arc_ct_min)
                                link_w_perc *= perc
                                a1   = a0 + perc * (a1 - a0)
                                b1   = b0 + perc * (b1 - b0)
                                
                        a_avg, b_avg  = (a0+a1)/2, (b0+b1)/2 # for arrow points

                        xa0, ya0, xa1, ya1  = self.xTi(a0), self.yTi(a0), self.xTi(b1), self.yTi(b1)
                        xb0, yb0, xb1, yb1  = self.xTi(a1), self.yTi(a1), self.xTi(b0), self.yTi(b0)
                        xarrow0_pt,yarrow0_pt = self.xTarrow(a_avg), self.yTarrow(a_avg)
                        xarrow1_pt,yarrow1_pt = self.xTarrow(b_avg), self.yTarrow(b_avg)

                        # should be refactored (2nd copy)
                        if   self.link_color is not None and type(self.link_color) == str and len(self.link_color) == 7 and self.link_color[0] == '#':
                            _link_color_ = self.link_color
                        elif self.link_color is None or self.color_by is None:
                            _link_color_ = self.rt_self.co_mgr.getTVColor('data', 'default')
                        elif self.link_color == 'src':
                            _link_color_ = self.rt_self.co_mgr.getColor(str(_fm_))
                        elif self.link_color == 'dst':
                            _link_color_ = self.rt_self.co_mgr.getColor(str(_to_))
                        else: # 'vary'
                            _link_color_ = fmto_color_lu[_fm_][_to_]

                        if   (a1-a0) < 4:
                            pass
                        elif (a1-a0) < 30:
                            _path_ = f'M {xa1} {ya1} ' + \
                                     f'A {self.r-self.node_h} {self.r-self.node_h} 0 0 0 {xb1} {yb1} ' + \
                                     f'L {xarrow1_pt} {yarrow1_pt} L {xa1} {ya1} Z'
                            svg.append(f'<path d="{_path_}" stroke="{_link_color_}" stroke-opacity="1.0" fill="{_link_color_}" opacity="{self.link_opacity}" />')
                        else:
                            _path_ = f'M {xa1} {ya1} ' + \
                                     f'A {self.r-self.node_h} {self.r-self.node_h} 0 0 0 {xb1} {yb1} ' + \
                                     f'C {self.xTarrow(b0)} {self.yTarrow(b0)} {self.xTo(b_avg)} {self.yTo(b_avg)} {xarrow1_pt} {yarrow1_pt}' + \
                                     f'C {self.xTo(b_avg)} {self.yTo(b_avg)} {self.xTarrow(b1)} {self.yTarrow(b1)} {xa1} {ya1} '
                            svg.append(f'<path d="{_path_}" stroke="{_link_color_}" stroke-opacity="1.0" fill="{_link_color_}" opacity="{self.link_opacity}" />')

                        if   (b1-b0) < 4:
                            pass
                        elif (b1-b0) < 30:
                            _path_ = f'M {xb0} {yb0} ' + \
                                     f'A {self.r-self.node_h} {self.r-self.node_h} 0 0 0 {xa0} {ya0}' + \
                                     f'L {xarrow0_pt} {yarrow0_pt} L {xb0} {yb0} Z'
                            svg.append(f'<path d="{_path_}" stroke="{_link_color_}" stroke-opacity="1.0" fill="{_link_color_}" opacity="{self.link_opacity}" />')
                        else:
                            _path_ = f'M {xb0} {yb0} ' + \
                                     f'A {self.r-self.node_h} {self.r-self.node_h} 0 0 0 {xa0} {ya0} ' + \
                                     f'C {self.xTarrow(a0)} {self.yTarrow(a0)} {self.xTo(a_avg)} {self.yTo(a_avg)} {xarrow0_pt} {yarrow0_pt}' + \
                                     f'C {self.xTo(a_avg)} {self.yTo(a_avg)} {self.xTarrow(a1)} {self.yTarrow(a1)} {xb0} {yb0} '
                            svg.append(f'<path d="{_path_}" stroke="{_link_color_}" stroke-opacity="1.0" fill="{_link_color_}" opacity="{self.link_opacity}" />')

                        angle_d   = 180 - abs(abs(a_avg - b_avg) - 180)
                        _ratio_   = 0.8 - 0.8 * angle_d/180
                        x_pull0, y_pull0 = self.cx + self.r * _ratio_ * cos(pi*a_avg/180.0), self.cy + self.r * _ratio_ * sin(pi*a_avg/180.0)
                        x_pull1, y_pull1 = self.cx + self.r * _ratio_ * cos(pi*b_avg/180.0), self.cy + self.r * _ratio_ * sin(pi*b_avg/180.0)
                        _path_ = f'M {xarrow0_pt} {yarrow0_pt} C {x_pull0} {y_pull0} {x_pull1} {y_pull1} {xarrow1_pt} {yarrow1_pt}'
                        if self.link_arrow is not None:
                            _curve_ = self.rt_self.bezierCurve((xarrow0_pt, yarrow0_pt), (x_pull0, y_pull0), (x_pull1, y_pull1), (xarrow1_pt, yarrow1_pt))
                            uv        = self.rt_self.unitVector((_curve_(0.8),(xarrow1_pt, yarrow1_pt)))
                            arrow_len, arrow_scale = min(self.r * self.arrow_ratio, self.arrow_px), 0.5
                            _path_ += f' l {  arrow_len * (-uv[0] + arrow_scale*uv[1])}  {  arrow_len * (-uv[1] - arrow_scale*uv[0])}'
                            _path_ += f' m {-(arrow_len * (-uv[0] + arrow_scale*uv[1]))} {-(arrow_len * (-uv[1] - arrow_scale*uv[0]))}'
                            _path_ += f' l {  arrow_len * (-uv[0] - arrow_scale*uv[1])}  {  arrow_len * (-uv[1] + arrow_scale*uv[0])}'

                        link_w = self.link_size_min + link_w_perc * (self.link_size_max - self.link_size_min)
                        link_w = max(min(link_w, self.link_size_max), self.link_size_min)

                        svg.append(f'<path d="{_path_}" stroke="{_link_color_}" stroke-opacity="{self.link_opacity}" stroke-width="{link_w}" fill="none" />')
                        #svg.append(f'<circle cx="{x_pull0}" cy="{y_pull0}" r="4" fill="none" stroke="{_link_color_}"/>') # debug - control points
                        #svg.append(f'<circle cx="{x_pull1}" cy="{y_pull1}" r="4" fill="none" stroke="{_link_color_}"/>') # debug - control points

            return ''.join(svg)

        #
        # __renderEdges_createSkeletonHexagonal__() - create the skeleton graph using a honeycomb like structure
        #
        def __renderEdges_createSkeletonHexagonal__(self, fmtos, fmtos_angles, fmto_fm_angle, fmto_to_angle, fmto_fm_pos, fmto_to_pos):
            skeleton_svg, skeleton = [], nx.Graph()
            fmto_entry, fmto_exit  = {}, {}

            def pointsToPath(points):
                p = 'M '+str(points[0][0])+','+str(points[0][1])
                for i in range(1,len(points)):
                    p += ' L '+str(points[i][0])+','+str(points[i][1])
                return p

            def xDigitPrecision(_tuples_, precision=3):
                return [(round(_t_[0],precision),round(_t_[1],precision)) for _t_ in _tuples_]

            # Geometrical variables
            r_div  = self.skeleton_rings # Overload the value for this purpose
            hx_e   = self.r/r_div
            hx_h   = sqrt(hx_e**2 - (hx_e/2)**2)
            hx_p   = lambda x,y: f"M {x} {y} m {-hx_e} {0} l {hx_e/2} {hx_h} l {hx_e} {0} l {hx_e/2} {-hx_h} l {-hx_e/2} {-hx_h} l {-hx_e} {0} l {-hx_e/2} {hx_h}"
            hx_pos = lambda x,y: [(x-hx_e, y), (x-hx_e/2, y+hx_h), (x+hx_e/2, y+hx_h), (x+hx_e, y), (x+hx_e/2, y-hx_h), (x-hx_e/2, y-hx_h), (x-hx_e, y)]
            adj_r  = self.r - self.node_h

            # Create the skeleton graph
            points, exists = set(), set()
            skeleton_svg.append(f'<circle cx="{self.cx}" cy="{self.cy}" r="{self.r}" fill="none" stroke="#a0a0a0" />')
            skeleton_svg.append(f'<circle cx="{self.cx}" cy="{self.cy}" r="2" fill="#ff0000" stroke="#ff0000" />')
            for j in range(r_div):
                y_shift = hx_h if (j%2)==0 else 0.0
                for i in range(r_div):
                    for pnx in [-1, 1]:
                        for pny in [-1, 1]:
                            x,y = self.cx+pnx*hx_e*1.5*(j),self.cy+y_shift+pny*2*hx_h*(i)
                            pts = xDigitPrecision(hx_pos(x,y))
                            for _edge_ in zip(pts,pts[1:]):
                                if self.rt_self.segmentLength((_edge_[0], (self.cx, self.cy))) < adj_r * 0.9 and \
                                   self.rt_self.segmentLength((_edge_[1], (self.cx, self.cy))) < adj_r * 0.9 and \
                                   _edge_ not in exists:
                                    skeleton_svg.append(f'<path d="{pointsToPath(_edge_)}" fill="none" stroke="#ff0000" stroke-width="0.4" />')
                                    skeleton.add_edge(_edge_[0], _edge_[1], weight=self.rt_self.segmentLength((_edge_[0], _edge_[1])))
                                    exists.add(_edge_), exists.add((_edge_[1], _edge_[0]))
                                    points.add(_edge_[0]), points.add(_edge_[1])
                            for _pt_ in pts:
                                if self.rt_self.segmentLength((_pt_, (self.cx, self.cy))) < adj_r * 0.9:
                                    skeleton_svg.append(f'<line x1="{_pt_[0]}" y1="{_pt_[1]}" x2="{x}" y2="{y}" stroke="#ff0000" stroke-width="0.1" />')
                                    skeleton.add_edge(_pt_, (x, y), weight=self.rt_self.segmentLength((_pt_, (x, y))))

            # Create the quad tree and fill in the lookups from the edge nodes to the skeleton graph based on closest point found
            q_tree = self.rt_self.xyQuadTree((0.0, 0.0, self.w, self.h), max_pts_per_node=16)
            q_tree.add(points)
            to_deg = lambda angle: pi*angle/180.0
            for i in range(len(fmtos)):
                _fmto_                         = fmtos[i]
                _fm_avg_angle_, _to_avg_angle_ = fmtos_angles[i]

                fm_pos              = (self.cx + adj_r*cos(to_deg(_fm_avg_angle_)), self.cy + adj_r*sin(to_deg(_fm_avg_angle_)))
                fmto_fm_pos[_fmto_] = fm_pos
                _closest_           = q_tree.closest(fm_pos, n=3)
                fmto_entry[fm_pos]  = _closest_[0][1]

                to_pos              = (self.cx + adj_r*cos(to_deg(_to_avg_angle_)), self.cy + adj_r*sin(to_deg(_to_avg_angle_)))
                fmto_to_pos[_fmto_] = to_pos
                _closest_           = q_tree.closest(to_pos, n=3)
                fmto_exit[to_pos]   = _closest_[0][1]

            return skeleton, skeleton_svg, fmto_entry, fmto_exit
        
        #
        # __renderEdges_createSkeletonKMeans__() - create the skeleton graph using the kmeans clustering algorithm
        #
        def __renderEdges_createSkeletonKMeans__(self, 
                                                 fmtos,          # array of fmto tuples ('fm','to')
                                                 fmtos_angles,   # array of fmto angles tuples (fm_angle, to_angle)
                                                 fmto_fm_angle,  # dictionary [_fmto_] = (min_angle, max_angle) # for the from portion
                                                 fmto_to_angle,  # dictionary [_fmto_] = (min_angle, max_angle) # for the to portion
                                                 fmto_fm_pos,    # dictionary -- to be filled in by the method - fmto_fm_pos[_fmto_] = (x,y)
                                                 fmto_to_pos):   # dictionary -- to be filled in by the method - fmto_to_pos[_fmto_] = (x,y)
            to_rad = lambda angle: pi*angle/180.0            
            skeleton_svg, fmto_entry, fmto_exit, skeleton  = [], {}, {}, nx.Graph()

            adj_r  = self.r - self.node_h

            # Gather the initial points
            points, pos_fms, pos_tos, fmto_to_i = [], {}, {}, {}
            for i in range(len(fmtos)):
                _fmto_                         = fmtos[i]
                _fm_avg_angle_, _to_avg_angle_ = fmtos_angles[i]
                fmto_to_i[_fmto_]              = i
                fm_pos              = (self.cx + adj_r*cos(to_rad(_fm_avg_angle_)), self.cy + adj_r*sin(to_rad(_fm_avg_angle_)))
                fmto_fm_pos[_fmto_] = fm_pos
                if fm_pos not in pos_fms: pos_fms[fm_pos] = []
                pos_fms[fm_pos].append(_fmto_)
                points.append(fm_pos)
                to_pos              = (self.cx + adj_r*cos(to_rad(_to_avg_angle_)), self.cy + adj_r*sin(to_rad(_to_avg_angle_)))
                fmto_to_pos[_fmto_] = to_pos
                if to_pos not in pos_tos: pos_tos[to_pos] = []
                pos_tos[to_pos].append(_fmto_)
                points.append(to_pos)

            #        outermost circle ...                    innermost circle
            radii = [3.0 * self.r / 4.0, 2.0 * self.r / 4.0, 1.0 * self.r / 4.0]
            ks    = [30,                 10,                 5]          

            # First stage clustering of the initial points
            all_angles, angle_to_pos = set(), {}
            cluster_centers, center_assignments = self.rt_self.kMeans2D(points, k=ks[0], iterations=20)
            for _center_ in center_assignments:
                if len(center_assignments[_center_]) == 0: continue
                fm_angles, to_angles = [], []
                for _pos_ in center_assignments[_center_]:
                    if _pos_ in pos_fms:
                        for _fmto_ in pos_fms[_pos_]: 
                            fm_angles.append(fmtos_angles[fmto_to_i[_fmto_]][0])
                    if _pos_ in pos_tos: 
                        for _fmto_ in pos_tos[_pos_]:
                            to_angles.append(fmtos_angles[fmto_to_i[_fmto_]][1])

                if len(fm_angles) > 0:
                    fm_avg_angle     = self.rt_self.averageDegrees(fm_angles)
                    fm_avg_angle_pos = (self.cx + radii[0]*cos(to_rad(fm_avg_angle)), self.cy + radii[0]*sin(to_rad(fm_avg_angle)))
                    all_angles.add(fm_avg_angle)
                    angle_to_pos[fm_avg_angle] = fm_avg_angle_pos


                if len(to_angles) > 0:
                    to_avg_angle     = self.rt_self.averageDegrees(to_angles)
                    to_avg_angle_pos = (self.cx + radii[0]*cos(to_rad(to_avg_angle)), self.cy + radii[0]*sin(to_rad(to_avg_angle)))
                    all_angles.add(to_avg_angle)
                    angle_to_pos[to_avg_angle] = to_avg_angle_pos

                for _pos_ in center_assignments[_center_]:
                    if _pos_ in pos_fms:
                        for _fmto_ in pos_fms[_pos_]:
                            _fm_avg_angle_, _to_avg_angle_ = fmtos_angles[fmto_to_i[_fmto_]]
                            _xy_             = (self.cx + adj_r*cos(to_rad(_fm_avg_angle_)), self.cy + adj_r*sin(to_rad(_fm_avg_angle_)))
                            fmto_entry[_xy_] = fm_avg_angle_pos
                    if _pos_ in pos_tos: 
                        for _fmto_ in pos_tos[_pos_]:
                            _fm_avg_angle_, _to_avg_angle_ = fmtos_angles[fmto_to_i[_fmto_]]
                            _xy_             = (self.cx + adj_r*cos(to_rad(_to_avg_angle_)), self.cy + adj_r*sin(to_rad(_to_avg_angle_)))
                            fmto_exit [_xy_] = to_avg_angle_pos
            
            # Connect the first ring of points // avoiding connecting because this routing is too close...
            _sorted_ = sorted(list(all_angles))
            for i in range(len(_sorted_)):
                a0, a1   = _sorted_[i], _sorted_[(i+1)%len(_sorted_)]
                xy0, xy1 = angle_to_pos[a0], angle_to_pos[a1]
                #skeleton.add_edge(xy0, xy1, weight=self.rt_self.segmentLength((xy0, xy1)))
                #skeleton_svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{xy1[0]}" y2="{xy1[1]}" stroke="black" />')

            # Second stage clustering for the intermediate ring
            points, pos_to_angle = [], {}
            for _angle_ in angle_to_pos: 
                points.append(angle_to_pos[_angle_])
                pos_to_angle[angle_to_pos[_angle_]] = _angle_
            all_angles = set()
            cluster_centers, center_assignments = self.rt_self.kMeans2D(points, k=ks[1], iterations=20)
            for _center_ in center_assignments:
                if len(center_assignments[_center_]) == 0: continue
                angles = []
                for _pos_ in center_assignments[_center_]: angles.append(pos_to_angle[_pos_])
                avg_angle = self.rt_self.averageDegrees(angles)
                all_angles.add(avg_angle)
                _xy_      = (self.cx + radii[1]*cos(to_rad(avg_angle)), self.cy + radii[1]*sin(to_rad(avg_angle)))
                for _pos_ in center_assignments[_center_]:
                    skeleton.add_edge(_pos_, _xy_, weight=self.rt_self.segmentLength((_pos_, _xy_)))
                    skeleton_svg.append(f'<line x1="{_pos_[0]}" y1="{_pos_[1]}" x2="{_xy_[0]}" y2="{_xy_[1]}" stroke="black" />')
                    for _other_pos_ in center_assignments[_center_]:
                        if _other_pos_ == _pos_: continue
                        skeleton.add_edge(_pos_, _other_pos_, weight=self.rt_self.segmentLength((_pos_, _other_pos_)))
                        skeleton_svg.append(f'<line x1="{_pos_[0]}" y1="{_pos_[1]}" x2="{_other_pos_[0]}" y2="{_other_pos_[1]}" stroke="black" />')

            # Connect the second ring of points
            angle_to_pos = {}
            _sorted_ = sorted(list(all_angles))
            for i in range(len(_sorted_)):
                a0, a1   = _sorted_[i], _sorted_[(i+1)%len(_sorted_)]
                xy0 = (self.cx + radii[1]*cos(to_rad(a0)), self.cy + radii[1]*sin(to_rad(a0)))
                angle_to_pos[a0] = xy0
                xy1 = (self.cx + radii[1]*cos(to_rad(a1)), self.cy + radii[1]*sin(to_rad(a1)))
                angle_to_pos[a1] = xy1
                skeleton.add_edge(xy0, xy1, weight=self.rt_self.segmentLength((xy0, xy1)))
                skeleton_svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{xy1[0]}" y2="{xy1[1]}" stroke="black" />')
                a2  = _sorted_[(i+2)%len(_sorted_)]
                xy2 = (self.cx + radii[1]*cos(to_rad(a2)), self.cy + radii[1]*sin(to_rad(a2)))
                skeleton.add_edge(xy0, xy2, weight=self.rt_self.segmentLength((xy0, xy2)))
                skeleton_svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{xy2[0]}" y2="{xy2[1]}" stroke="black" />')
                a3  = _sorted_[(i+3)%len(_sorted_)]
                xy3 = (self.cx + radii[1]*cos(to_rad(a3)), self.cy + radii[1]*sin(to_rad(a3)))
                skeleton.add_edge(xy0, xy3, weight=self.rt_self.segmentLength((xy0, xy3)))
                skeleton_svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{xy3[0]}" y2="{xy3[1]}" stroke="black" />')

            # final stage clustering for the intermediate ring
            points, pos_to_angle = [], {}
            for _angle_ in angle_to_pos: 
                points.append(angle_to_pos[_angle_])
                pos_to_angle[angle_to_pos[_angle_]] = _angle_
            all_angles = set()
            cluster_centers, center_assignments = self.rt_self.kMeans2D(points, k=ks[1], iterations=20)
            for _center_ in center_assignments:
                if len(center_assignments[_center_]) == 0: continue
                angles = []
                for _pos_ in center_assignments[_center_]: angles.append(pos_to_angle[_pos_])
                avg_angle = self.rt_self.averageDegrees(angles)
                all_angles.add(avg_angle)
                _xy_      = (self.cx + radii[2]*cos(to_rad(avg_angle)), self.cy + radii[2]*sin(to_rad(avg_angle)))
                for _pos_ in center_assignments[_center_]:
                    skeleton.add_edge(_pos_, _xy_, weight=self.rt_self.segmentLength((_pos_, _xy_)))
                    skeleton_svg.append(f'<line x1="{_pos_[0]}" y1="{_pos_[1]}" x2="{_xy_[0]}" y2="{_xy_[1]}" stroke="black" />')

            # Connect the final ring of points -- and connect them to the center
            angle_to_pos = {}
            _sorted_ = sorted(list(all_angles))
            for i in range(len(_sorted_)):
                a0, a1   = _sorted_[i], _sorted_[(i+1)%len(_sorted_)]
                xy0 = (self.cx + radii[2]*cos(to_rad(a0)), self.cy + radii[2]*sin(to_rad(a0)))
                angle_to_pos[a0] = xy0
                xy1 = (self.cx + radii[2]*cos(to_rad(a1)), self.cy + radii[2]*sin(to_rad(a1)))
                angle_to_pos[a1] = xy1
                skeleton.add_edge(xy0, xy1, weight=self.rt_self.segmentLength((xy0, xy1)))
                skeleton_svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{xy1[0]}" y2="{xy1[1]}" stroke="black" />')
                _center_ = (self.cx, self.cy)
                skeleton.add_edge(xy0, _center_, weight=self.rt_self.segmentLength((xy0, _center_)))
                skeleton_svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{_center_[0]}" y2="{_center_[1]}" stroke="black" />')


            return skeleton, skeleton_svg, fmto_entry, fmto_exit

        #
        # __renderEdges_createSkeletonHDBSCAN__() - create the skeleton graph using the hdbscan clustering algorithm
        #
        def __renderEdges_createSkeletonHDBSCAN_v2__(self, 
                                                     fmtos,          # array of fmto tuples ('fm','to')
                                                     fmtos_angles,   # array of fmto angles tuples (fm_angle, to_angle)
                                                     fmto_fm_angle,  # dictionary [_fmto_] = (min_angle, max_angle) # for the from portion
                                                     fmto_to_angle,  # dictionary [_fmto_] = (min_angle, max_angle) # for the to portion
                                                     fmto_fm_pos,    # dictionary -- to be filled in by the method - fmto_fm_pos[_fmto_] = (x,y)
                                                     fmto_to_pos):   # dictionary -- to be filled in by the method - fmto_to_pos[_fmto_] = (x,y)
            to_deg = lambda angle: pi*angle/180.0            
            skeleton_svg, fmto_entry, fmto_exit, skeleton  = [], {}, {}, nx.Graph()

            # Draw the original fmtos
            fmtos_pos = []
            for i in range(len(fmtos)):
                _fmto_                 = fmtos[i]
                _fm_angle_, _to_angle_ = fmtos_angles[i]
                _r_                    = self.r - self.node_h
                fmto_fm_pos[_fmto_]    = (self.cx + _r_*cos(to_deg(_fm_angle_)), self.cy + _r_*sin(to_deg(_fm_angle_)))
                fmto_to_pos[_fmto_]    = (self.cx + _r_*cos(to_deg(_to_angle_)), self.cy + _r_*sin(to_deg(_to_angle_)))
                fmtos_pos.append((fmto_fm_pos[_fmto_], fmto_to_pos[_fmto_]))
                skeleton_svg.append(f'<circle cx="{fmtos_pos[i][0][0]}" cy="{fmtos_pos[i][0][1]}" r="2" fill="black" />')
                skeleton_svg.append(f'<circle cx="{fmtos_pos[i][1][0]}" cy="{fmtos_pos[i][1][1]}" r="3" fill="none" stroke="red" />')

            l_fmtos_angles, l_fmtos_poses = fmtos_angles, fmtos_pos
            for _ring_ in range(1, self.skeleton_rings): # ring 0 is the outer ring
                # Cluster the fm-to angle tuples
                import hdbscan
                clusterer = hdbscan.HDBSCAN()
                clusterer.fit(l_fmtos_angles)
                _labels_  = clusterer.labels_
                if len(set(_labels_)) == 1: # stop when there's no way to distance between the clusters
                    break

                # Calculate the radius of this ring
                _r_adj_ = (self.r - self.node_h)
                _r_     = _r_adj_ - _r_adj_ * _ring_ / self.skeleton_rings
                skeleton_svg.append(f'<circle cx="{self.cx}" cy="{self.cy}" r="{_r_}" fill="none" stroke="#a0a0a0" />')

                fmtos_angles, fmto_fm_angle, fmto_to_angle, fmto_fm_pos, fmto_to_pos = [], {}, {}, {}, {}

                # For each label, calculate the sum of the angles and the number of samples
                _last_negative_label_ = -1
                fm_angle_sum, to_angle_sum, samples, _labels_updated_ = {}, {}, {}, []
                fm_angle_list, to_angle_list = {}, {}
                for i in range(len(_labels_)):
                    _label_ = _labels_[i]
                    if _label_ == -1: # -1 means unclustered... just make it its own cluster
                        _label_ = _last_negative_label_ - 1
                        _last_negative_label_ -= 1
                    if _label_ not in fm_angle_sum:
                        fm_angle_sum[_label_],  to_angle_sum[_label_], samples[_label_] = 0, 0, 0
                        fm_angle_list[_label_], to_angle_list[_label_] = [], []
                    fm_angle_sum[_label_] += l_fmtos_angles[i][0]
                    to_angle_sum[_label_] += l_fmtos_angles[i][1] # or is it 0?
                    fm_angle_list[_label_].append(l_fmtos_angles[i][0])
                    to_angle_list[_label_].append(l_fmtos_angles[i][1]) # or is it 0?
                    samples[_label_]      += 1
                    _labels_updated_.append(_label_)

                # Calculate the new positions as the averages of the sums from the last block of code
                fmtos_angles, fmtos_pos, label_to_i, angle_to_pos = [], [], {}, {}
                for _label_ in fm_angle_sum.keys():
                    label_to_i[_label_] = len(fmtos_angles)                    
                    # fm_angle_avg, to_angle_avg = fm_angle_sum[_label_]/samples[_label_], to_angle_sum[_label_]/samples[_label_]
                    fm_angle_avg, to_angle_avg = self.rt_self.averageDegrees(fm_angle_list[_label_]), self.rt_self.averageDegrees(to_angle_list[_label_])
                    fmtos_angles.append((fm_angle_avg, to_angle_avg))
                    fmtos_pos.append(((self.cx + _r_*cos(to_deg(fm_angle_avg)), self.cy + _r_*sin(to_deg(fm_angle_avg))),
                                      (self.cx + _r_*cos(to_deg(to_angle_avg)), self.cy + _r_*sin(to_deg(to_angle_avg)))))
                    angle_to_pos[fm_angle_avg], angle_to_pos[to_angle_avg] = fmtos_pos[-1][0], fmtos_pos[-1][1]
                    skeleton_svg.append(f'<circle cx="{fmtos_pos[-1][0][0]}" cy="{fmtos_pos[-1][0][1]}" r="2" fill="black" />')
                    skeleton_svg.append(f'<circle cx="{fmtos_pos[-1][1][0]}" cy="{fmtos_pos[-1][1][1]}" r="3" fill="none" stroke="red" />')
                
                if _ring_ == 1: # setup the fmto_entry and fmto_exit
                    for i in range(len(l_fmtos_angles)):
                        _label_        = _labels_updated_[i]
                        lxy_fm, lxy_to = l_fmtos_poses[i][0], l_fmtos_poses[i][1]
                        xy_fm,  xy_to  = fmtos_pos[label_to_i[_label_]][0], fmtos_pos[label_to_i[_label_]][1]
                        fmto_entry[lxy_fm], fmto_exit[lxy_to] = xy_fm, xy_to
                else:           # connect the spokes
                    for i in range(len(l_fmtos_angles)):
                        _label_        = _labels_updated_[i]
                        lxy_fm, lxy_to = l_fmtos_poses[i][0], l_fmtos_poses[i][1]
                        xy_fm,  xy_to  = fmtos_pos[label_to_i[_label_]][0], fmtos_pos[label_to_i[_label_]][1]
                        skeleton_svg.append(f'<line x1="{lxy_fm[0]}" y1="{lxy_fm[1]}" x2="{xy_fm[0]}" y2="{xy_fm[1]}" stroke="black" />')
                        skeleton.add_edge(lxy_fm, xy_fm, weight=self.rt_self.segmentLength((lxy_fm, xy_fm)))
                        skeleton.add_edge(lxy_to, xy_to, weight=self.rt_self.segmentLength((lxy_to, xy_to)))

                # connect the ring
                _sorter_ = list(angle_to_pos.keys())
                _sorter_.sort()
                for i in range(len(_sorter_)):
                    a0,  a1  = _sorter_[i], _sorter_[(i+1)%len(_sorter_)]
                    xy0, xy1 = angle_to_pos[a0], angle_to_pos[a1]
                    skeleton_svg.append(f'<line x1="{xy0[0]}" y1="{xy0[1]}" x2="{xy1[0]}" y2="{xy1[1]}" stroke="black" />')
                    skeleton.add_edge(xy0, xy1, weight=self.rt_self.segmentLength((xy0, xy1)))

                l_fmtos_angles, l_fmtos_poses = fmtos_angles, fmtos_pos
            
            return skeleton, skeleton_svg, fmto_entry, fmto_exit
                
        #
        # __renderEdges_createSkeletonHDBSCAN_20240505_() - create the skeleton graph using the hdbscan clustering algorithm
        # - older version ... failed under some configurations
        #
        def __renderEdges_createSkeletonHDBSCAN__(self, fmtos, fmtos_angles, fmto_fm_angle, fmto_to_angle, fmto_fm_pos, fmto_to_pos):
            skeleton_svg = []
            fmto_entry, fmto_exit  = {}, {}

            import hdbscan
            clusterer = hdbscan.HDBSCAN()
            clusterer.fit(fmtos_angles)
            self.clusterer = clusterer

            # Create the skeleton graph
            last_fm_i_pos  = None
            last_to_i_pos  = None
            last_fm_i_avg  = None
            last_to_i_avg  = None
            fm_i_pos       = {}
            to_i_pos       = {}
            skeleton       = nx.Graph()

            def __connectRing__(fip, tip, fia, tia):
                seen, avg_to_pos, avgs = set(), {}, []
                for j in fip.keys():
                    pos, avg = fip[j], fia[j]
                    if pos not in seen:
                        avg_to_pos[avg] = pos
                        avgs.append(avg)
                        seen.add(pos)
                for j in tip.keys():
                    pos, avg = tip[j], tia[j]
                    if pos not in seen:
                        avg_to_pos[avg] = pos
                        avgs.append(avg)
                        seen.add(pos)
                avgs.sort()
                for j in range(len(avgs)):
                    k = (j+1)%len(avgs)
                    _segment_ = (avg_to_pos[avgs[j]], avg_to_pos[avgs[k]])
                    skeleton.add_edge(_segment_[0], _segment_[1], weight=self.rt_self.segmentLength(_segment_))
                    skeleton_svg.append(f'<line x1="{_segment_[0][0]}" y1="{_segment_[0][1]}" x2="{_segment_[1][0]}" y2="{_segment_[1][1]}" stroke="#000000" stroke-width="0.2" />') # debug

            slt_as_np      = clusterer.single_linkage_tree_.to_numpy()
            d, d_max       = 0.00, slt_as_np[-1][2]
            d_inc          = d_max / self.skeleton_rings
            r, r_dec, ring = 1.0,  (1.0-0.1)/(floor(d_max/d_inc)), 0
            while d < d_max:
                _labels_ = clusterer.single_linkage_tree_.get_clusters(d, min_cluster_size=1)

                # Angles sum
                fm_sum, to_sum, samples = {}, {}, {}
                for i in range(len(_labels_)):
                    _label_    = _labels_[i]
                    fmto_key   = fmtos[i]
                    _fm_, _to_ = fmto_key
                    fm_angle, to_angle = fmto_fm_angle[fmto_key], fmto_to_angle[fmto_key]
                    fm_sum[_label_], to_sum[_label_] = fm_sum.get(_label_, 0) + fm_angle[0] + fm_angle[1], to_sum.get(_label_, 0) + to_angle[0] + to_angle[1]
                    samples[_label_] = samples.get(_label_, 0) + 2
                fm_i_avg, to_i_avg, fm_i_pos, to_i_pos = {}, {}, {}, {}

                # Angles average
                to_deg = lambda angle: pi*angle/180.0
                r_actual = (self.r - self.node_h) * r
                skeleton_svg.append(f'<circle cx="{self.cx}" cy="{self.cy}" r="{r_actual}" stroke="#00ff00" stroke-width="0.2" fill="none" />')
                for _label_ in set(_labels_):
                    fm_avg, to_avg = fm_sum[_label_] / samples[_label_], to_sum[_label_] / samples[_label_]
                    fm_pos = (self.cx + r_actual * cos(to_deg(fm_avg)), self.cy + r_actual * sin(to_deg(fm_avg)))
                    to_pos = (self.cx + r_actual * cos(to_deg(to_avg)), self.cy + r_actual * sin(to_deg(to_avg)))
                    skeleton_svg.append(f'<circle cx="{fm_pos[0]}" cy="{fm_pos[1]}" r="1.5" stroke="#ff0000" fill="none" stroke-width="0.2" />')    # debug
                    skeleton_svg.append(f'<circle cx="{fm_pos[0]}" cy="{fm_pos[1]}" r="0.8" stroke="#ff0000" fill="#ff0000" stroke-width="0.2" />') # debug
                    skeleton_svg.append(f'<circle cx="{to_pos[0]}" cy="{to_pos[1]}" r="1.3" fill="#000000" stroke-width="0.2" />')                  # debug
                    for i in range(len(_labels_)):
                        if _labels_[i] == _label_:
                            fm_i_avg[i], to_i_avg[i], fm_i_pos[i], to_i_pos[i] = fm_avg, to_avg, fm_pos, to_pos
                            if d == 0.0:
                                fmto_key = fmtos[i]
                                fmto_fm_pos[fmto_key], fmto_to_pos[fmto_key] = fm_pos, to_pos

                # Add the edges to the skeleton
                segment_added = set()
                if ring > 1:
                    for i in range(len(_labels_)):
                        _segment_ = (last_fm_i_pos[i], fm_i_pos[i])
                        skeleton_svg.append(f'<line x1="{_segment_[0][0]}" y1="{_segment_[0][1]}" x2="{_segment_[1][0]}" y2="{_segment_[1][1]}" stroke="#000000" stroke-width="0.2" />') # debug
                        if _segment_ not in segment_added:
                            skeleton.add_edge(_segment_[0], _segment_[1], weight=self.rt_self.segmentLength(_segment_))
                            segment_added.add(_segment_)
                        _segment_ = (last_to_i_pos[i], to_i_pos[i])
                        skeleton_svg.append(f'<line x1="{_segment_[0][0]}" y1="{_segment_[0][1]}" x2="{_segment_[1][0]}" y2="{_segment_[1][1]}" stroke="#000000" stroke-width="0.2" />') # debug
                        if _segment_ not in segment_added:
                            skeleton.add_edge(_segment_[0], _segment_[1], weight=self.rt_self.segmentLength(_segment_))
                            segment_added.add(_segment_)
                elif ring == 1:
                    for i in range(len(_labels_)):
                        fmto_entry[last_fm_i_pos[i]] = fm_i_pos[i]
                        fmto_exit [last_to_i_pos[i]] = to_i_pos[i]

                # Connect certain rings rotationally...
                # if ring == 3 or ring == (self.skeleton_rings-2) or ring == (self.skeleton_rings-1): # original implementation... but doesn't match paper
                # if ring == (self.skeleton_rings-1): # fails
                if ring > 1: # this can't be too small or the last one won't be filled in ... OR the spline will fail because of not enough points
                    __connectRing__(last_fm_i_pos, last_to_i_pos, last_fm_i_avg, last_to_i_avg)
                
                # Connect the most inside ring to the center point
                if ring == (self.skeleton_rings-1):
                    for i in range(len(_labels_)):
                        _segment_ = ((self.cx,self.cy), fm_i_pos[i])
                        skeleton_svg.append(f'<line x1="{_segment_[0][0]}" y1="{_segment_[0][1]}" x2="{_segment_[1][0]}" y2="{_segment_[1][1]}" stroke="#000000" stroke-width="0.2" />') # debug
                        if _segment_ not in segment_added:
                            skeleton.add_edge(_segment_[0], _segment_[1], weight=self.rt_self.segmentLength(_segment_))
                            segment_added.add(_segment_)
                        _segment_ = ((self.cx,self.cy), to_i_pos[i])
                        skeleton_svg.append(f'<line x1="{_segment_[0][0]}" y1="{_segment_[0][1]}" x2="{_segment_[1][0]}" y2="{_segment_[1][1]}" stroke="#000000" stroke-width="0.2" />') # debug
                        if _segment_ not in segment_added:
                            skeleton.add_edge(_segment_[0], _segment_[1], weight=self.rt_self.segmentLength(_segment_))
                            segment_added.add(_segment_)

                last_fm_i_pos, last_to_i_pos, last_fm_i_avg, last_to_i_avg = fm_i_pos, to_i_pos, fm_i_avg, to_i_avg
                d, r, ring = d + d_inc, r - r_dec, ring + 1
    
            return skeleton, skeleton_svg, fmto_entry, fmto_exit
                
        #
        # __renderEdges_bundled__(self) - render the edges (using the edge bundling from Holten 2006)
        #
        def __renderEdges_bundled__(self, struct_matches_render, fmto_lu, 
                                    local_dir_arc_ct, local_dir_arc_ct_min, local_dir_arc_ct_max, 
                                    fmto_color_lu):
            svg, skeleton_svg = [], []

            # Cluster the fm/to connections
            fmto_fm_angle,     fmto_to_angle     = {}, {}
            fmto_fm_angle_avg, fmto_to_angle_avg = {}, {}
            fmto_fm_pos,       fmto_to_pos       = {}, {}
            fmtos, fmtos_angles = [], []

            _ts_ = time.time()
            for node in self.node_dir_arc.keys():
                for _fm_ in self.node_dir_arc[node].keys():
                    if node != _fm_: # just scan the fm -> to directions
                        continue
                    for _to_ in self.node_dir_arc[node][_fm_].keys():
                        nbor = _fm_ if node != _fm_ else _to_
                        a0, a1 = self.node_dir_arc[node][_fm_][_to_]                            
                        b0, b1 = self.node_dir_arc[nbor][_fm_][_to_]
                        link_w_perc = (self.node_dir_arc_ct[nbor][_fm_][_to_] - self.node_dir_arc_ct_min) / (self.node_dir_arc_ct_max - self.node_dir_arc_ct_min)
                        if struct_matches_render == False:
                            if _fm_ not in fmto_lu.keys() or _to_ not in fmto_lu[_fm_].keys():
                                continue
                            if self.node_dir_arc_ct[node][_fm_][_to_] != local_dir_arc_ct[node][_fm_][_to_]:
                                # perc = local_dir_arc_ct[node][_fm_][_to_] / self.node_dir_arc_ct[node][_fm_][_to_]
                                # perc = min(perc, 1.0)
                                perc = (local_dir_arc_ct[node][_fm_][_to_] - local_dir_arc_ct_min)/(local_dir_arc_ct_max - local_dir_arc_ct_min)
                                link_w_perc *= perc
                                a1   = a0 + perc * (a1 - a0)
                                b1   = b0 + perc * (b1 - b0)
                        a_avg, b_avg = (a0+a1)/2, (b0+b1)/2
                        fmto_key = (_fm_,_to_)
                        fmto_fm_angle[fmto_key], fmto_to_angle[fmto_key] = (a0,a1), (b0,b1)
                        fmto_fm_angle_avg[fmto_key], fmto_to_angle_avg[fmto_key] = a_avg,b_avg
                        fmtos.append(fmto_key),  fmtos_angles.append((a_avg,b_avg))
            self.time_lu['bundler_prep'] = time.time() - _ts_

            # Skeleton option
            _ts_ = time.time()
            if self.skeleton is None: # if wouldn't be none in the case of small multiples (x-axis dependency)
                if   self.skeleton_algorithm == 'hdbscan' or self.skeleton_algorithm == 'hdbscanv2' or self.skeleton_algorithm == 'kmeans':
                    if len(fmtos) > 8:
                        if   self.skeleton_algorithm == 'hdbscan':
                            skeleton, skeleton_svg, fmto_entry, fmto_exit = self.__renderEdges_createSkeletonHDBSCAN__   (fmtos, fmtos_angles, fmto_fm_angle, fmto_to_angle, fmto_fm_pos, fmto_to_pos)
                        elif self.skeleton_algorithm == 'hdbscanv2':
                            skeleton, skeleton_svg, fmto_entry, fmto_exit = self.__renderEdges_createSkeletonHDBSCAN_v2__(fmtos, fmtos_angles, fmto_fm_angle, fmto_to_angle, fmto_fm_pos, fmto_to_pos)
                        elif self.skeleton_algorithm == 'kmeans':
                            skeleton, skeleton_svg, fmto_entry, fmto_exit = self.__renderEdges_createSkeletonKMeans__    (fmtos, fmtos_angles, fmto_fm_angle, fmto_to_angle, fmto_fm_pos, fmto_to_pos)
                        else: raise Exception('__renderEdges_bundled__() - only skeleton_methodology supported are "hdbscan", "hdbscanv2", or "kmeans"')
                    else:
                        skeleton, skeleton_svg, fmto_entry, fmto_exit = self.__renderEdges_createSkeletonHexagonal__ (fmtos, fmtos_angles, fmto_fm_angle, fmto_to_angle, fmto_fm_pos, fmto_to_pos)
                elif self.skeleton_algorithm == 'hexagonal' or self.skeleton_algorithm == 'simple':
                    skeleton, skeleton_svg, fmto_entry, fmto_exit     = self.__renderEdges_createSkeletonHexagonal__ (fmtos, fmtos_angles, fmto_fm_angle, fmto_to_angle, fmto_fm_pos, fmto_to_pos)
                else:
                    raise Exception('RTChordDiagram.__renderEdges_bundled__() - only skeleton_methodology supported are "hdbscan", "simple", or "hexagonal"')
            self.time_lu['bundler_skeleton'] = time.time() - _ts_

            # Bundle the edges
            use_all_pairs   = False
            ts_path_calc    = 0.0 
            ts_edge_render  = 0.0
            ts_track_routes = 0.0

            if use_all_pairs:
                _ts_ = time.time()
                shortest_paths = dict(nx.all_pairs_dijkstra_path(skeleton, weight='weight'))
                self.time_lu['path_calc'] = time.time() - _ts_

            for node in self.node_dir_arc.keys():
                for _fm_ in self.node_dir_arc[node].keys():
                    if node != _fm_: # just scan the fm -> to directions
                        continue
                    for _to_ in self.node_dir_arc[node][_fm_].keys():
                        if _fm_ not in fmto_lu.keys() or _to_ not in fmto_lu[_fm_].keys():
                            continue
                        fmto_key    = (_fm_,_to_)
                        link_w_perc = (self.node_dir_arc_ct[node][_fm_][_to_] - self.node_dir_arc_ct_min) / (self.node_dir_arc_ct_max - self.node_dir_arc_ct_min)
                        if struct_matches_render == False:
                            if self.node_dir_arc_ct[node][_fm_][_to_] != local_dir_arc_ct[node][_fm_][_to_]:
                                perc = (local_dir_arc_ct[node][_fm_][_to_] - local_dir_arc_ct_min)/(local_dir_arc_ct_max - local_dir_arc_ct_min)
                                link_w_perc *= perc
                        link_w      = self.link_size_min + link_w_perc * (self.link_size_max - self.link_size_min)
                        link_w      = max(min(link_w, self.link_size_max), self.link_size_min)
                        fm_pos      = fmto_fm_pos[fmto_key]
                        to_pos      = fmto_to_pos[fmto_key]

                        if use_all_pairs:
                            _shortest_ = shortest_paths[fmto_entry[fm_pos]][fmto_exit[to_pos]]
                            _shortest_.insert(0, fm_pos)
                            _shortest_.append(to_pos)
                        else:
                            _ts_ = time.time()
                            _shortest_ = nx.shortest_path(skeleton, fmto_entry[fm_pos], fmto_exit[to_pos], weight='weight')
                            _shortest_.insert(0, fm_pos)
                            _shortest_.append(to_pos)
                            ts_path_calc += time.time() - _ts_

                        _ts_ = time.time()
                        if self.track_routes:
                            for i in range(1, len(_shortest_)-2):
                                _segment_ = (_shortest_[i], _shortest_[i+1])
                                if _segment_ not in self.track_routes_segments: 
                                    self.track_routes_segments[_segment_] = 0
                                    self.track_routes_segments_fms[_segment_] = set()
                                    self.track_routes_segments_tos[_segment_] = set()
                                self.track_routes_segments[_segment_] += 1
                                self.track_routes_segments_fms[_segment_].add(_fm_)
                                self.track_routes_segments_tos[_segment_].add(_to_)
                        ts_track_routes += time.time() - _ts_

                        _ts_ = time.time()
                        if self.link_color == 'shade_fm_to':
                            _pts_ = self.rt_self.piecewiseCubicBSpline(_shortest_, beta=self.beta, t_inc=self.t_inc)
                            for i in range(len(_pts_)-1):
                                _link_color_ = self.rt_self.co_mgr.spectrum(i, 0, len(_pts_))
                                svg.append(f'<line x1="{_pts_[i][0]}" y1="{_pts_[i][1]}" x2="{_pts_[i+1][0]}" y2="{_pts_[i+1][1]}" stroke="{_link_color_}" stroke-width="{link_w}" stroke-opacity="{self.link_opacity}" />')
                        else:
                            # should be refactored (3rd copy)
                            if   self.link_color is not None and type(self.link_color) == str and len(self.link_color) == 7 and self.link_color[0] == '#':
                                _link_color_ = self.link_color
                            elif self.link_color is None or self.color_by is None:
                                _link_color_ = self.rt_self.co_mgr.getTVColor('data', 'default')
                            elif self.link_color == 'src':
                                _link_color_ = self.rt_self.co_mgr.getColor(str(_fm_))
                            elif self.link_color == 'dst':
                                _link_color_ = self.rt_self.co_mgr.getColor(str(_to_))
                            else: # 'vary'
                                _link_color_ = fmto_color_lu[_fm_][_to_]
                                
                            svg.append(f'<path d="{self.rt_self.svgPathCubicBSpline(_shortest_, beta=self.beta)}" fill="none" stroke="{_link_color_}" stroke-width="{link_w}" stroke-opacity="{self.link_opacity}" />')
                        ts_edge_render += time.time() - _ts_

            if use_all_pairs == False: self.time_lu['path_calc'] = ts_path_calc
            self.time_lu['track_routes'] = ts_track_routes
            self.time_lu['render_links'] = ts_edge_render

            self.skeleton     = skeleton
            self.skeleton_svg = f'<svg x="0" y="0" width="512" height="512" viewBox="0 0 {self.w} {self.h}" xmlns="http://www.w3.org/2000/svg">'+''.join(skeleton_svg)+'</svg>'

            return ''.join(svg)

        #
        # __calculateNodeArcs__() - calculate the node positions.
        # - note that the next method (__calculateNodeArcs_equal__) was derived from this method
        # -- so, any changes here should be propagated to the next method
        #
        def __calculateNodeArcs__(self, counter_lu, counter_sum, left_over_degs, fmto_lu, tofm_lu):
            nones_in_count = (None in self.order)
            a = 0.0
            for i in range(len(self.order)):
                node = self.order[i]
                if node is None: # None in order are treated as user specified gaps
                    a += self.node_gap_degs
                    continue
                counter_perc  = counter_lu[node] / counter_sum
                node_degrees  = counter_perc * left_over_degs
                self.node_to_arc    [node] = (a, a+node_degrees)
                self.node_to_arc_ct [node] = counter_lu[node]
                self.node_dir_arc   [node] = {}
                self.node_dir_arc_ct[node] = {}

                b, j = a, i - 1
                for k in range(len(self.order)):
                    dest = self.order[j]
                    if node in fmto_lu.keys() and dest in fmto_lu[node].keys():
                        b_inc = node_degrees*fmto_lu[node][dest]/counter_lu[node]
                        if node not in self.node_dir_arc[node].keys():
                            self.node_dir_arc   [node][node] = {}
                            self.node_dir_arc_ct[node][node] = {}
                        self.node_dir_arc   [node][node][dest] = (b, b+b_inc)
                        _value_ = fmto_lu[node][dest]
                        self.node_dir_arc_ct[node][node][dest] = _value_
                        if self.node_dir_arc_ct_min is None:
                            self.node_dir_arc_ct_min = self.node_dir_arc_ct_max = _value_
                        self.node_dir_arc_ct_min, self.node_dir_arc_ct_max = min(_value_, self.node_dir_arc_ct_min), max(_value_, self.node_dir_arc_ct_max)
                        b += b_inc
                    if node in tofm_lu.keys() and dest in tofm_lu[node].keys():
                        b_inc = node_degrees*tofm_lu[node][dest]/counter_lu[node]
                        if dest not in self.node_dir_arc[node].keys():
                            self.node_dir_arc   [node][dest] = {}
                            self.node_dir_arc_ct[node][dest] = {}
                        self.node_dir_arc   [node][dest][node] = (b, b+b_inc)
                        _value_ = tofm_lu[node][dest]
                        self.node_dir_arc_ct[node][dest][node] = _value_
                        if self.node_dir_arc_ct_min is None:
                            self.node_dir_arc_ct_min = self.node_dir_arc_ct_max = _value_
                        self.node_dir_arc_ct_min, self.node_dir_arc_ct_max = min(_value_, self.node_dir_arc_ct_min), max(_value_, self.node_dir_arc_ct_max)
                        b += b_inc
                    j = j - 1
                a += node_degrees
                if nones_in_count == False:
                    a += self.node_gap_degs

        #
        # __calculateNodeArcs_equal__() - calculate the node arcs using equal spacing.
        # - almost an exact duplicate of the above method
        #
        def __calculateNodeArcs_equal__(self, counter_lu, counter_sum, left_over_degs, fmto_lu, tofm_lu):
            nones_in_count = (None in self.order)
            node_degrees   = left_over_degs / (len(self.order) - self.order.count(None))
            a = 0.0
            for i in range(len(self.order)):
                node = self.order[i]
                if node is None: # None in order are treated as user specified gaps
                    a += self.node_gap_degs
                    continue
                self.node_to_arc    [node] = (a, a+node_degrees)
                self.node_to_arc_ct [node] = counter_lu[node]
                self.node_dir_arc   [node] = {}
                self.node_dir_arc_ct[node] = {}

                b, j = a, i - 1
                for k in range(len(self.order)):
                    dest = self.order[j]
                    if node in fmto_lu.keys() and dest in fmto_lu[node].keys():
                        if node not in self.node_dir_arc[node].keys():
                            self.node_dir_arc   [node][node] = {}
                            self.node_dir_arc_ct[node][node] = {}
                        self.node_dir_arc   [node][node][dest] = (a, a + node_degrees/2.0)
                        _value_ = fmto_lu[node][dest]
                        self.node_dir_arc_ct[node][node][dest] = _value_
                        if self.node_dir_arc_ct_min is None:
                            self.node_dir_arc_ct_min = self.node_dir_arc_ct_max = _value_
                        self.node_dir_arc_ct_min, self.node_dir_arc_ct_max = min(_value_, self.node_dir_arc_ct_min), max(_value_, self.node_dir_arc_ct_max)
                    if node in tofm_lu.keys() and dest in tofm_lu[node].keys():
                        if dest not in self.node_dir_arc[node].keys():
                            self.node_dir_arc   [node][dest] = {}
                            self.node_dir_arc_ct[node][dest] = {}
                        self.node_dir_arc   [node][dest][node] = (a + node_degrees/2.0, a + node_degrees)
                        _value_ = tofm_lu[node][dest]
                        self.node_dir_arc_ct[node][dest][node] = _value_
                        if self.node_dir_arc_ct_min is None:
                            self.node_dir_arc_ct_min = self.node_dir_arc_ct_max = _value_
                        self.node_dir_arc_ct_min, self.node_dir_arc_ct_max = min(_value_, self.node_dir_arc_ct_min), max(_value_, self.node_dir_arc_ct_max)
                    j = j - 1
                a += node_degrees
                if nones_in_count == False:
                    a += self.node_gap_degs

        #
        # renderSVG() - render as SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state:
                self.geom_to_df = {}

            # Determine the node order
            _ts_ = time.time()
            if self.order is None:
                if   self.dendrogram_algorithm == 'hdbscan': 
                    self.order = self.rt_self.dendrogramOrdering_HDBSCAN(self.df, self.fm, self.to, self.count_by, self.count_by_set)
                elif self.dendrogram_algorithm == 'original':
                    self.order = self.rt_self.dendrogramOrdering(self.df, self.fm, self.to, self.count_by, self.count_by_set)
                else:
                    self.order = self.rt_self.dendrogramOrderingTuples(self.df, self.fm, self.to, self.count_by, self.count_by_set)
            self.time_lu['dendrogram'] = time.time() - _ts_

            # Counting calcs
            _ts_ = time.time()
            counter_lu, counter_sum, fmto_lu, tofm_lu, fmto_color_lu, node_color_lu, df_lu = self.__countingCalc__()
            self.time_lu['counting_calc'] = time.time() - _ts_

            # Determine the geometry
            self.rx, self.ry = (self.w - 2 * self.x_ins)/2, (self.h - 2 * self.y_ins)/2
            self.r           = self.rx if (self.rx < self.ry) else self.ry
            if self.draw_labels:
                self.r -= self.txt_h
            self.cx, self.cy = self.w/2, self.h/2
            self.circ        = 2.0 * pi * self.r

            # Gap pixels adjustment ... if supplied by caller, None's in the list will be treated as the gaps (w/ no other gaps displayed)
            if None not in self.order:
                gap_pixels         = len(self.order) * self.node_gap # total amount of pixels needed with user supplied gap size
                self.node_gap_adj  = (0.2*self.circ)/len(self.order) if gap_pixels > 0.2 * self.circ else self.node_gap # if more than 20% of the circle, use the default
                self.node_gap_degs = 360.0 * (self.node_gap_adj / self.circ) # total number of degrees used by gaps
                left_over_degs     = 360.0 - self.node_gap_degs * len(self.order) # left over degrees for the nodes
            else:
                none_count         = self.order.count(None)       # number of gaps
                gap_pixels         = none_count * self.node_gap   # total amount of pixels needed with user supplied gap size
                self.node_gap_adj  = (0.2*self.circ)/none_count if gap_pixels > 0.2 * self.circ else self.node_gap # if more than 20% of the circle, use the default
                self.node_gap_degs = 360.0 * (self.node_gap_adj / self.circ)
                left_over_degs     = 360.0 - self.node_gap_degs * none_count

            # Node to arc calculation
            _ts_ = time.time()
            local_dir_arc_ct                                   = None
            local_dir_arc_ct_min,     local_dir_arc_ct_max     = None, None
            self.node_dir_arc_ct_min, self.node_dir_arc_ct_max = None, None

            if self.node_to_arc is None or self.node_dir_arc is None or self.node_to_arc_ct is None or self.node_dir_arc_ct is None:
                self.node_to_arc,    self.node_dir_arc    = {}, {}
                self.node_to_arc_ct, self.node_dir_arc_ct = {}, {} # counts for the info... for small multiples
                if self.equal_size_nodes:
                    self.__calculateNodeArcs_equal__(counter_lu, counter_sum, left_over_degs, fmto_lu, tofm_lu)
                else:
                    self.__calculateNodeArcs__(counter_lu, counter_sum, left_over_degs, fmto_lu, tofm_lu)
                struct_matches_render = True   # to faciliate faster rendering (because structure should all line up with dataframe)
            else:
                local_dir_arc_ct = {}
                for i in range(len(self.order)):
                    node = self.order[i]
                    local_dir_arc_ct[node] = {}
                    j = i-1
                    for k in range(len(self.order)):
                        dest = self.order[j]
                        if node in fmto_lu.keys() and dest in fmto_lu[node].keys():
                            if node not in local_dir_arc_ct[node].keys():
                                local_dir_arc_ct[node][node] = {}
                            _value_ = fmto_lu[node][dest]
                            local_dir_arc_ct[node][node][dest] = _value_
                            if local_dir_arc_ct_min is None:
                                local_dir_arc_ct_min = local_dir_arc_ct_max = _value_
                            local_dir_arc_ct_min, local_dir_arc_ct_max = min(_value_, local_dir_arc_ct_min), max(_value_, local_dir_arc_ct_max)

                        if node in tofm_lu.keys() and dest in tofm_lu[node].keys():
                            if dest not in local_dir_arc_ct[node].keys():
                                local_dir_arc_ct[node][dest] = {}
                            _value_ = tofm_lu[node][dest]
                            local_dir_arc_ct[node][dest][node] = _value_
                            if local_dir_arc_ct_min is None:
                                local_dir_arc_ct_min = local_dir_arc_ct_max = _value_
                            local_dir_arc_ct_min, local_dir_arc_ct_max = min(_value_, local_dir_arc_ct_min), max(_value_, local_dir_arc_ct_max)

                        j = j - 1
                struct_matches_render = False  # adjusts rendering based on another diagrams structure

            self.time_lu['calc_node_arcs'] = time.time() - _ts_

            # return mins and maxes
            if just_calc_max:
                if local_dir_arc_ct_min is None:
                    return self.node_dir_arc_ct_min, self.node_dir_arc_ct_max
                else:
                    return local_dir_arc_ct_min,     local_dir_arc_ct_max
            elif self.global_max is not None:
                if local_dir_arc_ct is None:
                    self.node_dir_arc_ct_min, self.node_dir_arc_ct_max = self.global_min, self.global_max
                else:
                    local_dir_arc_ct_min, local_dir_arc_ct_max = self.global_min, self.global_max

            # Avoid div by zero later...
            if   self.node_dir_arc_ct_min is None:
                self.node_dir_arc_ct_min = 0.0
                self.node_dir_arc_ct_max = 1.0
            elif self.node_dir_arc_ct_min == self.node_dir_arc_ct_max:
                self.node_dir_arc_ct_min -= 1.0
                self.node_dir_arc_ct_max += 1.0
            if   local_dir_arc_ct_min is None:
                local_dir_arc_ct_min = 0.0
                local_dir_arc_ct_max = 1.0
            elif local_dir_arc_ct_min == local_dir_arc_ct_max:
                local_dir_arc_ct_min -= 1.0
                local_dir_arc_ct_max += 1.0

            # Start the SVG Frame
            svg = []
            svg.append(f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">')
            background_color, axis_color = self.rt_self.co_mgr.getTVColor('background','default'), self.rt_self.co_mgr.getTVColor('axis','default')
            if self.draw_background:
                svg.append(f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />')
            if self.draw_circular_background:
                svg.append(f'<circle cx="{self.cx}" cy="{self.cy}" r="{self.r}" fill="{background_color}" stroke="{background_color}" />')

            self.xTo     = lambda a: self.cx + self.r                   * cos(pi*a/180.0) # Outer Circle - x transform
            self.xTi     = lambda a: self.cx + (self.r - self.node_h)   * cos(pi*a/180.0) # Inner Circle - x transform
            self.yTo     = lambda a: self.cy + self.r                   * sin(pi*a/180.0) # Outer Circle - y transform
            self.yTi     = lambda a: self.cy + (self.r - self.node_h)   * sin(pi*a/180.0) # Inner Circle - y transform
            self.xTc     = lambda a: self.cx + 20                       * cos(pi*a/180.0) # 20 pixels from center
            self.yTc     = lambda a: self.cy + 20                       * sin(pi*a/180.0) # 20 pixels from center
            self.xTarrow = lambda a: self.cx + (self.r - 2*self.node_h) * cos(pi*a/180.0)
            self.yTarrow = lambda a: self.cy + (self.r - 2*self.node_h) * sin(pi*a/180.0)

            # Draw the nodes
            _ts_ = time.time()
            svg.append(self.__renderNodes__(node_color_lu))
            self.time_lu['render_nodes'] = time.time() - _ts_

            # Draw the edges from the node to the neighbors
            if   self.link_style == 'wide':
                _ts_ = time.time()
                svg.append(self.__renderEdges_wide__(struct_matches_render, fmto_lu, 
                                                     local_dir_arc_ct, local_dir_arc_ct_min, local_dir_arc_ct_max, 
                                                     fmto_color_lu))
                self.time_lu['render_links'] = time.time() - _ts_
            elif self.link_style == 'narrow':
                _ts_ = time.time()
                svg.append(self.__renderEdges_narrow__(struct_matches_render, fmto_lu, 
                                                       local_dir_arc_ct, local_dir_arc_ct_min, local_dir_arc_ct_max, 
                                                       fmto_color_lu))
                self.time_lu['render_links'] = time.time() - _ts_
            elif self.link_style == 'bundled':
                svg.append(self.__renderEdges_bundled__(struct_matches_render, fmto_lu, 
                                                        local_dir_arc_ct, local_dir_arc_ct_min, local_dir_arc_ct_max, 
                                                        fmto_color_lu))
            else:
                raise Exception(f'RTChordDiagram.renderSVG() -- unknown link_style "{self.link_style}"')

            # Draw the labels
            if self.draw_labels:
                for node in self.node_to_arc.keys():
                    _label_ = self.node_labels[node] if node in self.node_labels else node
                    if len(self.label_only) == 0 or node in self.label_only or _label_ in self.label_only:
                        if self.label_style == 'circular':
                            _id_ = self.rt_self.encSVGID(node)
                            txt_offset = -3 - self.txt_offset
                            svg.append(f'''<text width="500" font-family="{self.rt_self.default_font}" font-size="{self.txt_h}px" dy="{txt_offset}" >''')
                            svg.append(f'''<textPath alignment-baseline="top" xlink:href="#{self.widget_id}-{_id_}">{_label_}</textPath></text>''')
                        elif self.label_style == 'radial':                            
                            angle_avg  = (self.node_to_arc[node][0] + self.node_to_arc[node][1])/2.0
                            txt_offset = self.txt_offset + self.txt_h/3
                            x_text    = self.cx + (self.r+txt_offset) * cos(pi*angle_avg/180.0)
                            y_text    = self.cy + (self.r+txt_offset) * sin(pi*angle_avg/180.0)
                            if angle_avg >= 270.0 or angle_avg < 90.0:
                                svg.append(self.rt_self.svgText(_label_, x_text, y_text, self.txt_h, anchor = 'start', rotation=angle_avg))
                            else:
                                svg.append(self.rt_self.svgText(_label_, x_text, y_text, self.txt_h, anchor = 'end',   rotation=angle_avg-180.0))
                        else:
                            raise Exception(f'RTChordDiagram.renderSVG() -- unknown label_style "{self.label_style}"')
                if self.label_style == 'circular' and len(self.hierarchical_labels) > 0:
                    for node in self.hierarchical_labels:
                        _label_ = self.node_labels[node] if node in self.node_labels else node
                        _id_ = self.rt_self.encSVGID(node)
                        txt_offset = -3 - self.txt_offset
                        svg.append(f'''<text width="500" font-family="{self.rt_self.default_font}" font-size="{self.txt_h}px" dy="{txt_offset}" >''')
                        svg.append(f'''<textPath alignment-baseline="top" xlink:href="#{self.widget_id}-{_id_}">{_label_}</textPath></text>''')

            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg.append(f'<rect width="{self.w-1}" height="{self.h}" x="0" y="0" fill-opacity="0.0" fill="none" stroke="{border_color}" />')

            svg.append('</svg>')
            self.last_render = ''.join(svg)
            return self.last_render#
        
