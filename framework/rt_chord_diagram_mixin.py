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
        return _sep_.join(sorted(list(set(a.split(_sep_))|set(b.split(_sep_)))))
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
    def dendrogramOrdering(self, df, fm, to, count_by, count_by_set, _sep_ = '|||'):        
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
                    heapq.heappush(_heap_,(_graph_[_new_][x], _new_ + ' <-|-> ' + x))
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
                    _extended_ = l
                    _extended_.extend(r)
                    return _extended_
        
        return leafWalk(_tree_)

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
                     df,                             # dataframe to render
                     relationships,                  # same convention as linknode [('fm','to')]
                     # ----------------------------- # everything else is a default...
                     color_by            = None,     # none (default) or field name (note that node_color or link_color needs to be 'vary')
                     count_by            = None,     # none means just count rows, otherwise, use a field to sum by
                     count_by_set        = False,    # count by summation (by default)... count_by column is checked
                     widget_id           = None,     # naming the svg elements                 
                     # ----------------------------- # node options
                     node_color          = None,     # none means color by node name, 'vary' by color_by, or specific color "#xxxxxx"
                     node_labels         = None,     # Dictionary of node string to array of strings for additional labeling options
                     node_labels_only    = False,    # Only label based on the node_labels dictionary
                     node_h              = 10,       # height of node from circle edge
                     node_gap            = 5,        # node gap in pixels (gap between the arcs)
                     order               = None,     # override calculated ordering...
                     # ----------------------------- # link options
                     link_color          = None,     # none means color by source node name, 'vary' by color_by, or specific color "#xxxxxx"
                     link_opacity        = 0.1,      # link opacity
                     link_arrow          = 'suble',  # None, 'subtle', or 'sharp'
                     label_only          = set(),    # label only set
                     # ----------------------------- # small multiples config
                     structure_template  = None,     # existing RTChordDiagram()
                     # ----------------------------- # visualization geometry / etc.
                     track_state         = False,    # track state for interactive filtering
                     x_view              = 0,        # x offset for the view
                     y_view              = 0,        # y offset for the view
                     w                   = 256,      # width of the view
                     h                   = 256,      # height of the view
                     x_ins               = 3,
                     y_ins               = 3,
                     txt_h               = 10,       # text height for labeling
                     draw_labels         = False,    # draw labels flag # not implemented yet
                     draw_border         = True):    # draw a border around the graph
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
                self.widget_id = 'chorddiagram_' + str(random.randint(0,65535))          
            self.node_color       = kwargs['node_color']                # done! (maybe)
            self.node_labels      = kwargs['node_labels']
            self.node_labels_only = kwargs['node_labels_only']
            self.node_h           = kwargs['node_h']                    # done!
            self.node_gap         = kwargs['node_gap']                  # done!
            self.order            = kwargs['order']                     # done!
            self.link_color       = kwargs['link_color']                # done!
            self.link_opacity     = kwargs['link_opacity']              # done!
            self.link_arrow       = kwargs['link_arrow']                # done!
            self.label_only       = kwargs['label_only']
            self.track_state      = kwargs['track_state']
            self.x_view           = kwargs['x_view']                    # n/a
            self.y_view           = kwargs['y_view']                    # n/a
            self.w                = kwargs['w']                         # n/a
            self.h                = kwargs['h']                         # n/a
            self.x_ins            = kwargs['x_ins']                     # n/a
            self.y_ins            = kwargs['y_ins']                     # n/a
            self.txt_h            = kwargs['txt_h']
            self.draw_labels      = kwargs['draw_labels']
            self.draw_border      = kwargs['draw_border']               # done!

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

            # If either from or to are lists, concat them together...
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
            self.node_to_arc     = None
            self.node_dir_arc    = None
            self.node_to_arc_ct  = None
            self.node_dir_arc_ct = None
            if kwargs['structure_template'] is not None:
                other = kwargs['structure_template']
                # Force render if necessary... ### COPY OF APPLYVIEWCONFIGUATION() BELOW
                if other.node_to_arc is None:
                    other._repr_svg_()
                self.order           = other.order
                self.node_to_arc     = other.node_to_arc
                self.node_dir_arc    = other.node_dir_arc
                self.node_to_arc_ct  = other.node_to_arc_ct
                self.node_dir_arc_ct = other.node_dir_arc_ct


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
            self.order           = other.order
            self.node_to_arc     = other.node_to_arc
            self.node_dir_arc    = other.node_dir_arc
            self.node_to_arc_ct  = other.node_to_arc_ct
            self.node_dir_arc_ct = other.node_dir_arc_ct
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
        # renderSVG() - render as SVG
        #
        def renderSVG(self, just_calc_max=False):
            if self.track_state:
                self.geom_to_df = {}

            # Determine the node order
            if self.order is None:
                self.order = self.rt_self.dendrogramOrdering(self.df, self.fm, self.to, self.count_by, self.count_by_set)

            # Counting calcs
            counter_lu, counter_sum, fmto_lu, tofm_lu, fmto_color_lu, node_color_lu, df_lu = self.__countingCalc__()

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
            local_dir_arc_ct = None
            if self.node_to_arc is None or self.node_dir_arc is None or self.node_to_arc_ct is None or self.node_dir_arc_ct is None:
                self.node_to_arc,    self.node_dir_arc    = {}, {}
                self.node_to_arc_ct, self.node_dir_arc_ct = {}, {} # counts for the info... for small multiples
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
                            self.node_dir_arc_ct[node][node][dest] = fmto_lu[node][dest]
                            b += b_inc
                        if node in tofm_lu.keys() and dest in tofm_lu[node].keys():
                            b_inc = node_degrees*tofm_lu[node][dest]/counter_lu[node]
                            if dest not in self.node_dir_arc[node].keys():
                                self.node_dir_arc   [node][dest] = {}
                                self.node_dir_arc_ct[node][dest] = {}
                            self.node_dir_arc   [node][dest][node] = (b, b+b_inc)
                            self.node_dir_arc_ct[node][dest][node] = tofm_lu[node][dest]
                            b += b_inc
                        j = j - 1
                    a += node_degrees + self.node_gap_degs
                struct_matches_render = True   # to faciliate faster rendering
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
                            local_dir_arc_ct[node][node][dest] = fmto_lu[node][dest]
                        if node in tofm_lu.keys() and dest in tofm_lu[node].keys():
                            if dest not in local_dir_arc_ct[node].keys():
                                local_dir_arc_ct[node][dest] = {}
                            local_dir_arc_ct[node][dest][node] = tofm_lu[node][dest]
                        j = j - 1
                struct_matches_render = False  # adjusts rendering based on another diagrams structure

            # Start the SVG Frame
            svg = []
            svg.append(f'<svg id="{self.widget_id}" x="{self.x_view}" y="{self.y_view}" width="{self.w}" height="{self.h}" xmlns="http://www.w3.org/2000/svg">')
            background_color, axis_color = self.rt_self.co_mgr.getTVColor('background','default'), self.rt_self.co_mgr.getTVColor('axis','default')
            svg.append(f'<rect width="{self.w-1}" height="{self.h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />')

            xTo = lambda a: self.cx + self.r                 * cos(pi*a/180.0) # Outer Circle - x transform
            xTi = lambda a: self.cx + (self.r - self.node_h) * cos(pi*a/180.0) # Inner Circle - x transform
            yTo = lambda a: self.cy + self.r                 * sin(pi*a/180.0) # Outer Circle - y transform
            yTi = lambda a: self.cy + (self.r - self.node_h) * sin(pi*a/180.0) # Inner Circle - y transform
            xTc = lambda a: self.cx + 20                     * cos(pi*a/180.0) # 20 pixels from center
            yTc = lambda a: self.cy + 20                     * sin(pi*a/180.0) # 20 pixels from center
            xTarrow = lambda a: self.cx + (self.r - 2*self.node_h) * cos(pi*a/180.0)
            yTarrow = lambda a: self.cy + (self.r - 2*self.node_h) * sin(pi*a/180.0)

            # Draw the nodes
            _color_ = self.rt_self.co_mgr.getTVColor('data','default')
            for node in self.node_to_arc.keys():
                a0, a1 = self.node_to_arc[node]
                x0_out,  y0_out  = xTo(a0), yTo(a0)
                x0_in,   y0_in   = xTi(a0), yTi(a0)
                x1_out,  y1_out  = xTo(a1), yTo(a1)
                x1_in,   y1_in   = xTi(a1), yTi(a1)
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

                # Draw the edges from this node to the nbors
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

                            xa0, ya0, xa1, ya1  = xTi(a0), yTi(a0), xTi(b1), yTi(b1)
                            xb0, yb0, xb1, yb1  = xTi(a1), yTi(a1), xTi(b0), yTi(b0)
                            xarrow0, yarrow0    = xTarrow(b0), yTarrow(b0)
                            xarrow_pt,yarrow_pt = xTi(b_avg),  yTi(b_avg)
                            xarrow1, yarrow1    = xTarrow(b1), yTarrow(b1)
                            
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

            # Draw the labels
            if self.draw_labels:
                for node in self.node_to_arc.keys():
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
        
