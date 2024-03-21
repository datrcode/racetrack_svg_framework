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
import hdbscan
import networkx as nx

from math import pi, sin, cos, ceil, floor

from shapely.geometry import Polygon

from rt_component import RTComponent

__name__ = 'rt_chord_diagram_mixin'

#
# Chord Diagram Mixin
#
class RTChordDiagramMixin(object):
    # concats two strings in alphabetical order
    def __den_fromToString__(self, x, fm, to, _connector_ = ' <-|-> '):
        _fm_, _to_ = str(x[fm]), str(x[to])
        return (_fm_+_connector_+_to_) if (_fm_<_to_) else (_to_+_connector_+_fm_)
    # separates the concatenated string back into it's two parts
    def __den_fromToStringParts__(self, x, _connector_ = ' <-|-> '):
        i = x.index(_connector_)
        return x[:i],x[i+len(_connector_):]
    # merges names (which themselves can be merged names)
    def __den_mergedName__(self, a, b, _sep_ = '|||'):
        #return _sep_.join(sorted(list(set(a.split(_sep_))|set(b.split(_sep_)))))
        #return _sep_.join(list(set(a.split(_sep_))|set(b.split(_sep_))))
        ls = a.split(_sep_)
        ls.extend(b.split(_sep_))
        return _sep_.join(ls)
    # separates merged names back into parts
    def __den_breakdownMerge__(self, a, _sep_ = '|||'):
        return a.split(_sep_)

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
        df = df.with_columns(pl.struct([fm,to]).map_elements(__lambda__).alias('__fmto__'))
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
        
        return leafWalk(_tree_)

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
        df = df.with_columns(pl.struct([fm,to]).map_elements(__lambda__).alias('__fmto__'))
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
        return list(_tuple_)

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
                     df,                                    # dataframe to render
                     relationships,                         # same convention as linknode [('fm','to')]
                     # ------------------------------------ # everything else is a default...
                     color_by                   = None,     # none (default) or field name (note that node_color or link_color needs to be 'vary')
                     count_by                   = None,     # none means just count rows, otherwise, use a field to sum by
                     count_by_set               = False,    # count by summation (by default)... count_by column is checked
                     widget_id                  = None,     # naming the svg elements                 
                     # ----------------------------- # node options
                     node_color                 = None,     # none means color by node name, 'vary' by color_by, or specific color "#xxxxxx"
                     node_h                     = 10,       # height of node from circle edge
                     node_gap                   = 5,        # node gap in pixels (gap between the arcs)
                     order                      = None,     # override calculated ordering...
                     label_only                 = set(),    # label only set
                     equal_size_nodes           = False,    # equal size nodes
                     # ------------------------------------ # link options
                     link_color                 = None,     # none means color by source node name, 'vary' by color_by, or specific color "#xxxxxx"
                     link_opacity               = 0.5,      # link opacity
                     link_arrow                 = 'subtle', # None, 'subtle', or 'sharp'
                     arrow_px                   = 16,       # arrow size in pixels
                     arrow_ratio                = 0.05,     # arrow size as a ratio of the radius
                     link_style                 = 'narrow', # 'narrow', 'wide', 'bundled'
                     min_link_size              = 0.8,      # for 'narrow', min link size
                     max_link_size              = 4.0,      # for 'narrow', max link size
                     # ------------------------------------ # small multiples config
                     structure_template         = None,     # existing RTChordDiagram() ... e.g., for small multiples
                     dendrogram_algorithm       = None,     # 'original', 'hdbscan', or None
                     # ------------------------------------ # visualization geometry / etc.
                     track_state                = False,    # track state for interactive filtering
                     x_view                     = 0,        # x offset for the view
                     y_view                     = 0,        # y offset for the view
                     w                          = 256,      # width of the view
                     h                          = 256,      # height of the view
                     x_ins                      = 3,
                     y_ins                      = 3,
                     txt_h                      = 10,       # text height for labeling
                     draw_labels                = False,    # draw labels flag # not implemented yet
                     draw_border                = True,     # draw a border around the graph
                     draw_background            = False):   # useful to turn off in small multiples settings

        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTChordDiagram(self, **_params_)


    #
    # createConcatColumn() - concatenate multiple columns together into a single column
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
            self.parms            = locals().copy()
            self.rt_self          = rt_self
            self.df               = rt_self.copyDataFrame(kwargs['df']) # still needs polars!
            self.relationships    = kwargs['relationships']             # done!
            self.color_by         = kwargs['color_by']                  # done! (nothing to handle)
            self.count_by         = kwargs['count_by']                  # done!
            self.count_by_set     = kwargs['count_by_set']              # done!
            self.widget_id        = kwargs['widget_id']                 # done!
            if self.widget_id is None:
                self.widget_id = 'chorddiagram_' + str(random.randint(0,8*65535))          
            self.node_color       = kwargs['node_color']                # done! (maybe)
            self.node_h           = kwargs['node_h']                    # done!
            self.node_gap         = kwargs['node_gap']                  # done!
            self.order            = kwargs['order']                     # done!
            self.label_only       = kwargs['label_only']                # done!
            self.equal_size_nodes = kwargs['equal_size_nodes']          # done! (needs testing)
            self.link_color       = kwargs['link_color']                # done!
            self.link_opacity     = kwargs['link_opacity']              # done!
            self.link_arrow       = kwargs['link_arrow']                # done!
            self.arrow_px         = kwargs['arrow_px']                  # done!
            self.arrow_ratio      = kwargs['arrow_ratio']               # done!
            self.link_style       = kwargs['link_style']                # done!
            self.min_link_size    = kwargs['min_link_size']             # done!
            self.max_link_size    = kwargs['max_link_size']             # done!
            self.track_state      = kwargs['track_state']               # <--- still needs to be done
            self.x_view           = kwargs['x_view']                    # n/a
            self.y_view           = kwargs['y_view']                    # n/a
            self.w                = kwargs['w']                         # n/a
            self.h                = kwargs['h']                         # n/a
            self.x_ins            = kwargs['x_ins']                     # n/a
            self.y_ins            = kwargs['y_ins']                     # n/a
            self.txt_h            = kwargs['txt_h']                     # done!
            self.draw_labels      = kwargs['draw_labels']               # done!
            self.draw_border      = kwargs['draw_border']               # done!
            self.draw_background  = kwargs['draw_background']           # done!
            self.dendrogram_algorithm = kwargs['dendrogram_algorithm']
            self.cluster_rings    = 7
            self.time_lu          = {}

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
            if   self.rt_self.isPandas(self.df):
                self.df = self.df[self.df[_fm_] != self.df[_to_]]
            elif self.rt_self.isPolars(self.df):
                self.df = self.df.filter(pl.col(self.fm) != pl.col(self.to))
            else:
                raise Exception('RTChordDiagram() - only pandas and polars supported [3]')

            # Check the count_by column
            if self.count_by_set == False:
                self.count_by_set = rt_self.countBySet(self.df, self.count_by)
            
            # Tracking state
            self.geom_to_df  = {}
            self.last_render = None

            # Geometric construction... these members map the nodes into the circle...
            # ... manipulating these prior to render is how small multiples needs to work
            self.node_to_arc         = None
            self.node_dir_arc        = None
            self.node_to_arc_ct      = None
            self.node_dir_arc_ct     = None
            self.node_dir_arc_ct_min = None
            self.node_dir_arc_ct_max = None
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
            df_counter = df_fm.join(df_to, on='__node__', how='outer_coalesce').fill_null(0).with_columns((pl.col('__fmcount__') + pl.col('__tocount__')).alias('__count__'))

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
        # __renderNodes__() - render the nodes (outer edges of the circle)
        #
        def __renderNodes__(self, node_color_lu):
            svg     = []
            _color_ = self.rt_self.co_mgr.getTVColor('data','default')
            for node in self.node_to_arc.keys():
                a0, a1 = self.node_to_arc[node]
                x0_out,  y0_out  = self.xTo(a0), self.yTo(a0)
                x0_in,   y0_in   = self.xTi(a0), self.yTi(a0)
                x1_out,  y1_out  = self.xTo(a1), self.yTo(a1)
                x1_in,   y1_in   = self.xTi(a1), self.yTi(a1)
                large_arc = 0 if (a1-a0) <= 180.0 else 1
                _path_ = f'M {x0_out} {y0_out} A {self.r} {self.r} 0 {large_arc} 1 {x1_out} {y1_out} L {x1_in} {y1_in} ' + \
                                            f' A {self.r-self.node_h} {self.r-self.node_h} 0 {large_arc} 0 {x0_in}  {y0_in}  Z'
                if   type(self.node_color) == str and len(self.node_color) == 7 and self.node_color.startswith('#'):
                    _node_color_ = self.node_color
                elif self.color_by is not None and self.node_color == 'vary':
                    _node_color_ = node_color_lu[node]
                else:
                    _node_color_ = self.rt_self.co_mgr.getColor(str(node))
                _id_ = self.rt_self.encSVGID(node)
                svg.append(f'<path id="{self.widget_id}-{_id_}" d="{_path_}" stroke-width="0.8" stroke="{_node_color_}" fill="{_node_color_}" />')

            return ''.join(svg)

        #
        # __renderEdges_wide__(self) - render the edges (as large filled areas)
        #
        def __renderEdges_wide__(self, struct_matches_render, fmto_lu, local_dir_arc_ct, fmto_color_lu):
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
                                perc = local_dir_arc_ct[node][_fm_][_to_] / self.node_dir_arc_ct[node][_fm_][_to_]
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
                        
                        if self.link_color is None or self.color_by is None:
                            _link_color_ = self.rt_self.co_mgr.getColor(str(_fm_))
                        elif type(self.link_color) == str and len(self.link_color) == 7 and self.link_color[0] == '#':
                            _link_color_ = self.link_color
                        else: # 'vary'
                            _link_color_ = fmto_color_lu[_fm_][_to_]

                        svg.append(f'<path d="{_path_}" stroke="{_link_color_}" stroke-opacity="1.0" fill="{_link_color_}" opacity="{self.link_opacity}" />')

            return ''.join(svg)

        #
        # __renderEdges_narrow__(self) - render the edges (links)
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
                                perc = local_dir_arc_ct[node][_fm_][_to_] / self.node_dir_arc_ct[node][_fm_][_to_]
                                link_w_perc *= perc
                                a1   = a0 + perc * (a1 - a0)
                                b1   = b0 + perc * (b1 - b0)
                                
                        a_avg, b_avg  = (a0+a1)/2, (b0+b1)/2 # for arrow points

                        xa0, ya0, xa1, ya1  = self.xTi(a0), self.yTi(a0), self.xTi(b1), self.yTi(b1)
                        xb0, yb0, xb1, yb1  = self.xTi(a1), self.yTi(a1), self.xTi(b0), self.yTi(b0)
                        xarrow0_pt,yarrow0_pt = self.xTarrow(a_avg), self.yTarrow(a_avg)
                        xarrow1_pt,yarrow1_pt = self.xTarrow(b_avg), self.yTarrow(b_avg)

                        if self.link_color is None or self.color_by is None:
                            _link_color_ = self.rt_self.co_mgr.getColor(str(_fm_))
                        elif type(self.link_color) == str and len(self.link_color) == 7 and self.link_color[0] == '#':
                            _link_color_ = self.link_color
                        else: # 'vary'
                            _link_color_ = fmto_color_lu[_fm_][_to_]

                        if (a1-a0) < 30:
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

                        if (b1-b0) < 30:
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

                        link_w = self.min_link_size + link_w_perc * (self.max_link_size - self.min_link_size)

                        svg.append(f'<path d="{_path_}" stroke="{_link_color_}" stroke-opacity="{self.link_opacity}" stroke-width="{link_w}" fill="none" />')
                        #svg.append(f'<circle cx="{x_pull0}" cy="{y_pull0}" r="4" fill="none" stroke="{_link_color_}"/>') # debug - control points
                        #svg.append(f'<circle cx="{x_pull1}" cy="{y_pull1}" r="4" fill="none" stroke="{_link_color_}"/>') # debug - control points

            return ''.join(svg)


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
                                perc = local_dir_arc_ct[node][_fm_][_to_] / self.node_dir_arc_ct[node][_fm_][_to_]
                                link_w_perc *= perc
                                a1   = a0 + perc * (a1 - a0)
                                b1   = b0 + perc * (b1 - b0)
                        a_avg, b_avg = (a0+a1)/2, (b0+b1)/2
                        fmto_key = (_fm_,_to_)
                        fmto_fm_angle[fmto_key], fmto_to_angle[fmto_key] = (a0,a1), (b0,b1)
                        fmto_fm_angle_avg[fmto_key], fmto_to_angle_avg[fmto_key] = a_avg,b_avg
                        fmtos.append(fmto_key),  fmtos_angles.append((a_avg,b_avg))

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
            d_inc          = d_max / self.cluster_rings
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
                if last_fm_i_pos is not None:
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

                # Connect certain rings rotationally...
                if ring == 3 or ring == (self.cluster_rings-2) or ring == (self.cluster_rings-1):
                    __connectRing__(last_fm_i_pos, last_to_i_pos, last_fm_i_avg, last_to_i_avg)

                last_fm_i_pos, last_to_i_pos, last_fm_i_avg, last_to_i_avg = fm_i_pos, to_i_pos, fm_i_avg, to_i_avg
                d, r, ring = d + d_inc, r - r_dec, ring + 1
            
            # Bundle the edges
            for node in self.node_dir_arc.keys():
                for _fm_ in self.node_dir_arc[node].keys():
                    if node != _fm_: # just scan the fm -> to directions
                        continue
                    for _to_ in self.node_dir_arc[node][_fm_].keys():
                        fmto_key   = (_fm_,_to_)
                        fm_pos     = fmto_fm_pos[fmto_key]
                        to_pos     = fmto_to_pos[fmto_key]
                        _shortest_ = nx.shortest_path(skeleton, fm_pos, to_pos, weight='weight')

                        if self.link_color is None or self.color_by is None:
                            _link_color_ = self.rt_self.co_mgr.getColor(str(_fm_))
                        elif type(self.link_color) == str and len(self.link_color) == 7 and self.link_color[0] == '#':
                            _link_color_ = self.link_color
                        else: # 'vary'
                            _link_color_ = fmto_color_lu[_fm_][_to_]

                        link_w = self.min_link_size + link_w_perc * (self.max_link_size - self.min_link_size)

                        svg.append(f'<path d="{self.rt_self.svgPathCubicBSpline(_shortest_)}" fill="none" stroke="{_link_color_}" stroke-width="{link_w}" stroke-opacity="{self.link_opacity}" />')

            self.skeleton_svg = f'<svg x="0" y="0" width="1024" height="1024" viewBox="0 0 {self.w} {self.h}" xmlns="http://www.w3.org/2000/svg">'+''.join(skeleton_svg)+'</svg>'

            return ''.join(svg)

        #
        # __calculateNodeArcs__() - calculate the node positions.
        # - note that the next method (__calculateNodeArcs_equal__) was derived from this method
        # -- so, any changes here should be propagated to the next method
        #
        def __calculateNodeArcs__(self, counter_lu, counter_sum, left_over_degs, fmto_lu, tofm_lu):
            a = 0.0
            for i in range(len(self.order)):
                node = self.order[i]
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
                a += node_degrees + self.node_gap_degs

        #
        # __calculateNodeArcs_equal__() - calculate the node arcs using equal spacing.
        # - almost an exact duplicate of the above method
        #
        def __calculateNodeArcs_equal__(self, counter_lu, counter_sum, left_over_degs, fmto_lu, tofm_lu):
            node_degrees = (360.0 / len(self.order)) - self.node_gap_degs
            a = 0.0
            for i in range(len(self.order)):
                node = self.order[i]
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
                a += node_degrees + self.node_gap_degs

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

            # Gap pixels adjustment
            gap_pixels       = len(self.order) * self.node_gap
            if gap_pixels > 0.2 * self.circ:
                self.node_gap_adj = (0.2*self.circ)/len(self.order)
            else:
                self.node_gap_adj = self.node_gap
            self.node_gap_degs = 360.0 * (self.node_gap_adj / self.circ)
            left_over_degs  = 360.0 - self.node_gap_degs * len(self.order)

            # Node to arc calculation
            _ts_ = time.time()
            local_dir_arc_ct = None
            local_dir_arc_ct_min, local_dir_arc_ct_max = None, None
            self.node_dir_arc_ct_min, self.node_dir_arc_ct_max = None, None

            if self.node_to_arc is None or self.node_dir_arc is None or self.node_to_arc_ct is None or self.node_dir_arc_ct is None:
                self.node_to_arc,    self.node_dir_arc    = {}, {}
                self.node_to_arc_ct, self.node_dir_arc_ct = {}, {} # counts for the info... for small multiples
                if self.equal_size_nodes:
                    self.__calculateNodeArcs_equal__(counter_lu, counter_sum, left_over_degs, fmto_lu, tofm_lu)
                else:
                    self.__calculateNodeArcs__(counter_lu, counter_sum, left_over_degs, fmto_lu, tofm_lu)
                struct_matches_render = True   # to faciliate faster rendering
            else:
                local_dir_arc_ct = {}
                local_dir_arc_ct_min, local_dir_arc_ct_max = None, None
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
            _ts_ = time.time()
            if   self.link_style == 'wide':
                svg.append(self.__renderEdges_wide__(struct_matches_render, fmto_lu, local_dir_arc_ct, fmto_color_lu))
            elif self.link_style == 'narrow':
                svg.append(self.__renderEdges_narrow__(struct_matches_render, fmto_lu, 
                                                       local_dir_arc_ct, local_dir_arc_ct_min, local_dir_arc_ct_max, 
                                                       fmto_color_lu))
            elif self.link_style == 'bundled':
                svg.append(self.__renderEdges_bundled__(struct_matches_render, fmto_lu, 
                                                        local_dir_arc_ct, local_dir_arc_ct_min, local_dir_arc_ct_max, 
                                                        fmto_color_lu))
            else:
                raise Exception(f'RTChordDiagram.renderSVG() -- unknown link_style "{self.link_style}"')
            self.time_lu['render_links'] = time.time() - _ts_

            # Draw the labels
            if self.draw_labels:
                for node in self.node_to_arc.keys():
                    if len(self.label_only) == 0 or node in self.label_only:
                        _id_ = self.rt_self.encSVGID(node)
                        svg.append(f'''<text width="500" font-family="{self.rt_self.default_font}" font-size="{self.txt_h}px" y="-3" >''')
                        svg.append(f'''<textPath alignment-baseline="top" xlink:href="#{self.widget_id}-{_id_}">{node}</textPath></text>''')

            # Draw the border
            if self.draw_border:
                border_color = self.rt_self.co_mgr.getTVColor('border','default')
                svg.append(f'<rect width="{self.w-1}" height="{self.h}" x="0" y="0" fill-opacity="0.0" fill="none" stroke="{border_color}" />')

            svg.append('</svg>')
            self.last_render = ''.join(svg)
            return self.last_render#
        
