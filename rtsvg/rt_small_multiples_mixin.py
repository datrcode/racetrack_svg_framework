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
import polars as pl
import numpy as np
import random
import inspect

# import xml.etree.ElementTree as ET # Proper way to manipulate XML Tree... but removed because of the overhead requirements...

from shapely.geometry import Polygon

from .rt_component import RTComponent

__name__ = 'rt_small_multiples_mixin'

#
# Small Multiples Mixin
#
class RTSmallMultiplesMixin(object):
    #
    # categoryOrder() - determine the order of the categories
    # - category_by should be a list of fields
    #
    def categoryOrder(self, df, category_by, sort_by, sort_by_field):
        #if type(category_by) != list:
        #    category_by = [category_by]
        if   self.isPandas(df): return self.__categoryOrder_pandas__(df, category_by, sort_by, sort_by_field)
        elif self.isPolars(df): return self.__categoryOrder_polars__(df, category_by, sort_by, sort_by_field)
        else: raise Exception('RTSmallMultiples.categoryOrder() - only pandas and polars are supported')

    #
    def __categoryOrder_pandas__(self, df, category_by, sort_by, sort_by_field):
        cat_gb = df.groupby(category_by)
        if   sort_by is None or sort_by == 'alpha':
            cat_order = cat_gb.count()
        elif type(sort_by) == list:            
            cat_order = pd.Series(np.zeros(len(sort_by)), index=sort_by)
        elif sort_by == 'records' or sort_by_field is None or sort_by_field in category_by:    
            cat_order = cat_gb.size().sort_values(ascending=False)
        elif sort_by == 'field':
            # Count by numeric summation
            if self.fieldIsArithmetic(df, sort_by_field):
                cat_order = cat_gb[sort_by_field].sum().sort_values(ascending=False)            
            # Count by set operation
            else:
                _list = list(category_by)
                _list.append(sort_by_field)
                tmp_gb = df.groupby(_list)
                tmp_df = pd.DataFrame(tmp_gb.size()).reset_index()
                cat_order  = tmp_df.groupby(category_by).size().sort_values(ascending=False)
        else:
            raise Exception('smallMultiples() - sort by must be "records", "field", "alpha", "similarity", or a list')
        return cat_gb, cat_order

    #
    def __categoryOrder_polars__(self, df, category_by, sort_by, sort_by_field):
        if   sort_by is None or sort_by == 'alpha': # (Y_check)
            df_min    = df.drop(set(df.columns) - set(category_by))
            cat_order = df_min.unique(subset=category_by).sort(category_by)
        elif type(sort_by) == list: # (Y)
            cat_order = pl.DataFrame({'category_by':sort_by})
        elif sort_by == 'records' or sort_by_field is None or sort_by_field in category_by: # (Y_check)
            cat_order = df.group_by(category_by, maintain_order=True).agg(pl.len().alias('__count__')).sort('__count__', descending=True)
        elif sort_by == 'field': # (Y)
            # Count by numeric summation
            if self.fieldIsArithmetic(df, sort_by_field): # (Y_check)
                df_min    = df.drop(set(df.columns) - set(category_by) - set([sort_by_field]))
                cat_order = df_min.group_by(category_by, maintain_order=True).agg(pl.sum(sort_by_field).alias('__count__')).sort('__count__', descending=True)            
            # Count by set operation
            else: # (Y_check)
                df_min    = df.drop(set(df.columns) - set(category_by) - set([sort_by_field]))
                cat_order = df_min.group_by(category_by, maintain_order=True).n_unique()
                cat_order = cat_order.rename({sort_by_field:'__count__'}).sort('__count__', descending=True)
        else:
            raise Exception('smallMultiples() - sort by must be "records", "field", "alpha", "similarity", or a list')
        
        return df.partition_by(category_by, as_dict=True), cat_order

    #
    # For future reference, to make this work with a new widget:
    #
    # - Add to list of widgets in the racetrack.py
    # - Add to widget check in the beginning of this method
    # - If the widget uses timestamps, add it to the 'guess the timestamp column' area
    # - Add to widget dependent axis parts
    #
    # NOTE:  ANY MODIFICATIONS TO THESE PARAMETERS NEED TO BE REFLECTED IN THE INSTANCE CREATION
    #        ... AND PROBABLY THE STANDALONE RENDERER AS WELL // ALL WITHIN THIS FILE
    #
    def __smallMultiples_impl__(self,
                                df,                                 # Dataframe
                                category_by,                        # Field(s) to separate small multiples by
                                sm_type,                            # Visualization type (e.g., 'xy', 'linkNode', ...)

                                #-----------------------------------# Defaults after this line

                                sm_params             = {},         # Dictionary for customizing widget
                                customize_params_fn   = None,       # Customize the parameters function

                                ts_field              = None,       # For any temporal components
                                count_by              = None,       # Passed to the widgets
                                color_by              = None,       # Passed to the widgets
                                global_color_order    = None,       # color by ordering... if none (default), will be calculated
                                count_by_set          = False,      # count by using a set operation

                                temporal_granularity  = None,       # Minimum temporal granularity for the temporalBarChart component

                                #-----------------------------------# Small multiple params

                                show_df_multiple      = True,       # Show the "all data" version small multiple // note issues with xy scatterplots when data is aggregated
                                max_categories        = None,       # Limit the number of small multiples shown
                                grid_view             = False,      # For two category fields, make it into a grid
                                shrink_wrap_rows      = False,      # For a grid view, shrink wrap rows
                                sort_by               = 'records',  # 'records','alpha','field', 'similarity', or a list in the category_by schema
                                sort_by_field         = None,       # For sort_by == 'field', the field name... for 'similarity', the exemplar key
                                faded_sm_set          = None,       # small multiple labels to render as faded -- stored in a set as the string label (not the index tuple)
                                faded_opacity         = 0.7,        # ... opacity to use when fading

                                x_axis_independent    = True,       # Use independent axis for x (xy, temporal, and linkNode)
                                y_axis_independent    = True,       # Use independent axis for y (xy, temporal, periodic, pie)

                                category_to_sm        = None,       # If set to a dictionary, will be filled in with svg element per category
                                category_to_instance  = None,       # If set to a dictionary, will be filled in with the class instance
                                category_to_df        = None,       # If set to a dictionary, will be filled with the df subsetted to the category

                                #-----------------------------------# Render-specific params

                                widget_id             = None,       # Uniquely identify this widget -- embedded into svg element ids
                                x_view                = 0,          # View coordinates
                                y_view                = 0,
                                w                     = 768,        # Width of the sm container
                                h                     = 768,        # Height of the sm container
                                w_sm_override         = None,       # Override the small multiple width
                                h_sm_override         = None,       # Override the small multiple height
                                txt_h                 = 14,         # Text height for the small multiple captions
                                x_ins                 = 2,          # Left/right inserts
                                y_ins                 = 2,          # Top/bottom inserts
                                x_inter               = 2,          # Horizontal spacing between small multiples
                                y_inter               = 4,          # Vertical spacing between small multiples
                                background_override   = None,       # Override the background color
                                draw_labels           = True,       # Draw label under each small multiple
                                draw_border           = True):      # Draw border around the whole chart
        
        my_params = locals().copy()

        # Preserve original
        df = self.copyDataFrame(df)

        # Check widget ... since there's widget specific processing
        _implemented_types = ['boxplot', 'calendarHeatmap', 'chordDiagram', 'choroplethMap', 'histogram', 'linkNode', 'link',
                              'periodicBarChart', 'pieChart', 'temporalBarChart', 'wordCloud', 'xy']
        if (sm_type in _implemented_types) == False:
            raise Exception(f'smallMultipes: widget type "{sm_type}" not implemented (initial check)')
        
        # Generate a widget id if it's not already set
        if widget_id is None:
            widget_id = "smallmultiples_" + str(random.randint(0,65535))

        ### ***************************************************************************************************************************
        ### PARAMETERS
        ### ***************************************************************************************************************************

        if sm_params is not None and 'count_by' in sm_params: count_by = sm_params['count_by']
        if sm_params is not None and 'color_by' in sm_params: color_by = sm_params['color_by']

        # Make the categories into a list (if not already so)
        if type(category_by) != list: category_by = [category_by]

        # Organize by similarity...        
        if sort_by == 'similarity':
            params_copy = my_params.copy()
            params_copy.pop('self')
            sort_by = self.__orderSmallMultiplesBySimilarity__(**params_copy)

        # Transform the categories if necessary (and the count and color bys as well)
        df, category_by = self.transformFieldListAndDataFrame(df, category_by)
        df, color_bys   = self.transformFieldListAndDataFrame(df, [color_by])
        color_by        = color_bys[0]
        df, count_bys   = self.transformFieldListAndDataFrame(df, [count_by])
        count_by        = count_bys[0]

        # Transform any of the sm params...
        required_columns = getattr(self, f'{sm_type}RequiredFields')(**sm_params)
        for _field in required_columns:
            df, _throwaway = self.transformFieldListAndDataFrame(df, [_field])

        # Ensure the timestamp field (ts_field) is set
        if (sm_type == 'temporalBarChart' or \
            sm_type == 'periodicBarChart' or \
            sm_type == 'calendarHeatmap') and ts_field is None:
            ts_field = self.guessTimestampField(df)
                
        # Calculate temporal_granulaity if needed
        if sm_type == 'temporalBarChart' and temporal_granularity is None:
            temporal_granularity = self.temporalGranularity(df, ts_field)

        # Determine categories and ordering // cat_order and cat_gb need to be set
        cat_gb, cat_order = self.categoryOrder(df, category_by, sort_by, sort_by_field)

        # Determine the color ordering (not for xy though...)
        if count_by_set == False:
            count_by_set = self.countBySet(df, count_by)

        # Create a consistent color-by ordering
        if color_by is not None and global_color_order is None and \
           (sm_type == 'boxplot' or \
            sm_type == 'histogram' or \
            sm_type == 'periodicBarChart' or \
            sm_type == 'pieChart' or \
            sm_type == 'temporalBarChart'):
            global_color_order = self.colorRenderOrder(df, color_by, count_by, count_by_set)

        ### ***************************************************************************************************************************
        ### SMALL MULTIPLE PARAMETERS
        ### ***************************************************************************************************************************

        # Get most of the params ready... most the params == params that won't change between small multiples
        widget_func         = getattr(self, sm_type)

        most_params = sm_params.copy()
        most_params['count_by']    = count_by
        most_params['color_by']    = color_by
        most_params['x_view']      = x_ins
        most_params['y_view']      = y_ins

        accepted_args = set(inspect.getfullargspec(getattr(self, sm_type)).args)
        
        if 'global_color_order' in accepted_args:
            most_params['global_color_order'] = global_color_order
        if 'ts_field' in accepted_args:
            most_params['ts_field'] = ts_field
        if 'temporal_granularity' in accepted_args:
            most_params['temporal_granularity'] = temporal_granularity
        if sm_type == 'linkNode' or sm_type == 'link':
            most_params['use_pos_for_bounds'] = False

        # Handle dependent axes ... unfortunately, this is widget dependent
        polars_requires_repartition = False
        if x_axis_independent == False or y_axis_independent == False:
            #
            # xy and x-axis
            #
            if sm_type == 'xy' and x_axis_independent == False:
                sm_x_axis = widget_id + "_x"
                x_field_is_scalar = True # Default for the xy widget
                if 'x_field_is_scalar' in sm_params.keys():
                    x_field_is_scalar = sm_params['x_field_is_scalar']
                xOrder = None
                if 'x_order' in sm_params.keys():
                    xOrder = sm_params['x_order']
                xFillTransforms = True # Default Value...
                if 'x_fill_transforms' in sm_params.keys():
                    xFillTransforms = sm_params['x_fill_transforms']
                df, x_is_time,x_label_min,x_label_max,xTrans,xOrder,xMin,xMax = self.xyCreateAxisColumn(df, sm_params['x_field'], x_field_is_scalar, sm_x_axis, xOrder, xFillTransforms)
                most_params['x_axis_col']        = sm_x_axis
                most_params['x_is_time']         = x_is_time
                most_params['x_label_min']       = x_label_min
                most_params['x_label_max']       = x_label_max
                most_params['x_trans_func']      = xTrans
                most_params['x_order']           = xOrder
                most_params['x_fill_transforms'] = xFillTransforms
                polars_requires_repartition      = True

            #
            # xy and y-axis
            #
            if sm_type == 'xy' and y_axis_independent == False:
                sm_y_axis = widget_id + "_y"
                y_field_is_scalar = True # Default for the xy widget
                if 'y_field_is_scalar' in sm_params.keys():
                    y_field_is_scalar = sm_params['y_field_is_scalar']
                yOrder = None
                if 'y_order' in sm_params.keys():
                    yOrder = sm_params['y_order']
                yFillTransforms = True # Default Value...
                if 'y_fill_transforms' in sm_params.keys():
                    yFillTransforms = sm_params['y_fill_transforms']
                df, y_is_time,y_label_min,y_label_max,yTrans,yOrder,yMin,yMax = self.xyCreateAxisColumn(df, sm_params['y_field'], y_field_is_scalar, sm_y_axis, yOrder, yFillTransforms)
                most_params['y_axis_col']        = sm_y_axis
                most_params['y_is_time']         = y_is_time
                most_params['y_label_min']       = y_label_min
                most_params['y_label_max']       = y_label_max
                most_params['y_trans_func']      = yTrans
                most_params['y_order']           = yOrder
                most_params['y_fill_transforms'] = yFillTransforms
                polars_requires_repartition      = True
            #
            # temporalBarChart and x-axis
            #
            if x_axis_independent == False and sm_type == 'temporalBarChart':
                most_params['ts_min'] = df[ts_field].min()
                most_params['ts_max'] = df[ts_field].max()
            
            #
            # linkNode and position for bounds
            #
            if (sm_type == 'linkNode' or sm_type == 'link') and x_axis_independent == False:
                most_params['use_pos_for_bounds'] = True
            
            #
            # chordDiagram - x-axis is considered the ordering of the nodes (the y-axis -- in the next block, is the width of the lines/connections)
            #
            if sm_type == 'chordDiagram' and x_axis_independent == False:
                most_params['structure_template'] = self.chordDiagram(df, sm_params['relationships'])

            #
            # histogram/periodicBarChart/temporalBarChart/boxplot and y-axis
            #
            if y_axis_independent == False and (sm_type == 'histogram' or sm_type == 'periodicBarChart' or sm_type == 'temporalBarChart' or sm_type == 'boxplot' or sm_type == 'choroplethMap' or sm_type == 'chordDiagram'):
                global_min,global_max = None,None

                if max_categories is None:
                    max_categories = len(cat_order)

                for cat_i in range(0,max_categories):
                    if   self.isPandas(df):
                        key    = cat_order.index[cat_i]
                        if type(key) != tuple:
                            key = (key,)
                        key_df = cat_gb.get_group(key)
                    elif self.isPolars(df):
                        key    = cat_order[category_by][cat_i].rows()[0]
                        # Fix for the polars-version... for some reason, sometimes it's get tupled...
                        # - 2024-02-07 -- think a polars update changed this...
                        #if type(key) == tuple and len(key) == 1:
                        #    key = key[0]
                        key_df = cat_gb[key]
                    my_params = most_params.copy()
                    my_params['df'] = key_df
                    rt_comp_instance = widget_func(**my_params)
                    local_min,local_max = rt_comp_instance.renderSVG(just_calc_max=True)
                    if global_min is None:
                        global_min,global_max = local_min,local_max
                    global_min,global_max = min(global_min,local_min),max(global_max,local_max)
                most_params['global_max'] = global_max
                most_params['global_min'] = global_min
            
            #
            # calendarHeatmap
            #
            if sm_type == 'calendarHeatmap':
                global_max,global_min = None,None

                if max_categories is None:
                    max_categories = len(cat_order)

                for cat_i in range(0,max_categories):
                    if self.isPandas(df):
                        key    = cat_order.index[cat_i]
                        key_df = cat_gb.get_group(key)
                    else:
                        key    = cat_order[category_by][cat_i].rows()[0]
                        key_df = cat_gb[key]
                    my_params = most_params.copy()
                    my_params['df'] = key_df
                    rt_comp_instance = widget_func(**my_params)
                    local_min,local_max = rt_comp_instance.renderSVG(just_calc_max=True)
                    if global_max is None:
                        global_max,global_min = local_max,local_min
                    else:
                        if local_max > global_max:
                            global_max = local_max
                        if local_min < global_min:
                            global_min = local_min
                most_params['global_max'] = global_max
                most_params['global_min'] = global_min

        # When you change the base df, that change does not propagate to previous partition_by's..
        if polars_requires_repartition and self.isPolars(df):
            cat_gb = df.partition_by(category_by, as_dict=True)

        ### ***************************************************************************************************************************
        ### POSITIONING & SIZING
        ### ***************************************************************************************************************************

        # If grid view is enabled, determine the alternate mapping / placement
        grid_lu = None
        if grid_view and len(category_by) == 2:
            show_df_multiple = False
            max_categories   = len(cat_order)
            grid_lu          = {}

            # If the order is specified, then use it
            if type(sort_by) == list:
                row_order,col_order = [],[]
                for _tuple_pair in sort_by:
                    _row,_col = _tuple_pair[0],_tuple_pair[1]
                    if _col not in col_order:
                        col_order.append(_col)
                    if _row not in row_order:
                        row_order.append(_row)
                    grid_lu[_tuple_pair] = (col_order.index(_col), row_order.index(_row))
                
                if draw_labels:
                    my_txt_h = txt_h
                    draw_grid_column_header = True
                    draw_grid_row_header    = True
                else:
                    my_txt_h = 0
                    draw_grid_column_header = False
                    draw_grid_row_header    = False

            # Else... calculate the placement based on the data... without shrinkwrap
            elif shrink_wrap_rows == False:
                col_sort = fieldOrder(self, df, category_by[1], sort_by, sort_by_field)
                row_sort = fieldOrder(self, df, category_by[0], sort_by, sort_by_field)

                col_lu,col_order = {},[]
                for i in range(0,len(col_sort)):
                    if self.isPandas(df):
                        col_lu[col_sort.index[i]] = i
                        col_order.append(col_sort.index[i])
                    else:
                        col_lu[col_sort['field'][i]] = i
                        col_order.append(col_sort['field'][i])

                row_lu,row_order = {},[]
                for i in range(0,len(row_sort)):
                    if self.isPandas(df):
                        row_lu[row_sort.index[i]] = i
                        row_order.append(row_sort.index[i])
                    else:
                        row_lu[row_sort['field'][i]] = i
                        row_order.append(row_sort['field'][i])

                if self.isPandas(df):
                    for key, key_df in cat_gb: # this is a groupby generator
                        grid_lu[key] = (col_lu[key[1]], row_lu[key[0]])
                else:
                    for key in cat_gb: # because this is actually a dictionary
                        grid_lu[key] = (col_lu[key[1]], row_lu[key[0]])

                if draw_labels:
                    my_txt_h = txt_h
                    draw_grid_column_header = True
                    draw_grid_row_header    = True
                else:
                    my_txt_h = 0
                    draw_grid_column_header = False
                    draw_grid_row_header    = False

            # Else... calculate the placement based on the data... with shrinkwrapping on the rows
            else:
                row_sort = fieldOrder(self, df, category_by[0], sort_by, sort_by_field)
                if   self.isPandas(df):
                    row_gb   = df.groupby(category_by[0])
                elif self.isPolars(df):
                    row_gb   = df.partition_by(category_by[0], as_dict=True)

                longest_row = 1

                row_lu,row_order,grid_rows = {},[],[]
                for i in range(0,len(row_sort)):
                    if   self.isPandas(df):
                        k = row_sort.index[i]
                    elif self.isPolars(df):
                        k = row_sort['field'][i]
                    row_lu[k] = i
                    row_order.append(k)
                    grid_rows.append(k)
                    k_df = row_gb.get_group(k) if self.isPandas(df) else row_gb[k] if self.isPolars(df) else None
                    this_rows_order = fieldOrder(self, k_df, category_by[1], sort_by, sort_by_field)
                    for j in range(0,len(this_rows_order)):
                        l = this_rows_order.index[j]
                        grid_lu[(k,l)] = (j,i)
                    
                    if longest_row < len(this_rows_order):
                        longest_row = len(this_rows_order)
                    
                if w_sm_override is not None and h_sm_override is not None:
                    w_sm = w_sm_override
                    h_sm = h_sm_override

                    h = 2*y_ins + len(row_order) * h_sm + (len(row_order) - 1) * y_inter
                    if draw_labels:
                        h += len(row_order) * txt_h
                    
                    w = 2*x_ins + longest_row * w_sm + (longest_row - 1) * x_inter
                    if draw_labels:
                        w += txt_h
                else:
                    h_sm = (h - 2*y_ins - (len(row_order)-1) * y_inter) / len(row_order)
                    if draw_labels:
                        h_sm -= txt_h
                    
                    if draw_labels:
                        w_sm = (w - 2*x_ins - (longest_row-1) * x_inter - txt_h) / longest_row
                    else:
                        w_sm = (w - 2*x_ins - (longest_row-1) * x_inter) / longest_row

                draw_grid_column_header = False
                draw_grid_row_header    = draw_labels

            # Common to both of the above non-shrinkwrap otpions // should be refactored...
            if shrink_wrap_rows == False:
                grid_max_rows,grid_max_cols = len(row_order),len(col_order)
                grid_rows,grid_cols         = row_order,col_order
                max_categories              = len(row_order)*len(col_order)

                # Minimum dimensions for the component ... should be using this to make the dimensions congruent
                #dim_min  = getattr(self, f'{sm_type}MinimumDimensions')  (**sm_params)
                if draw_grid_row_header:
                    w_sm = ((w - 2*x_ins - txt_h)/grid_max_cols) - x_inter
                else:
                    w_sm = ((w - 2*x_ins        )/grid_max_cols) - x_inter

                if draw_grid_column_header:
                    h_sm = ((h - 2*y_ins - txt_h)/grid_max_rows) - y_inter # - my_txt_h
                else:
                    h_sm = ((h - 2*y_ins        )/grid_max_rows) - y_inter # - my_txt_h

        else:
            # Figure out the size of each small multiple ... start with the minimum dimension
            # ... to_fit == all the small multiples that will be rendered (including the "all" version)
            # ... max_categories == excludes the "all" version... i.e., what will be rended from the group_by categories
            if max_categories is None:
                max_categories = len(cat_order)
            if max_categories > len(cat_order):
                max_categories = len(cat_order)
            to_fit = (max_categories + 1) if show_df_multiple else max_categories

            # Determine the width and height of the small multiples themselves
            # Minimum dimensions for the widget
            dim_min  = getattr(self, f'{sm_type}MinimumDimensions')  (**most_params)
            
            # Binary search to find the best fit
            w_min_adj = dim_min[0] + x_inter
            h_min_adj = dim_min[1] + y_inter
            my_txt_h = txt_h if draw_labels else 0
                
            w_sm, h_sm = findOptimalFit(w_min_adj, h_min_adj, my_txt_h, w - x_ins, h - y_ins, to_fit)
            sm_columns = int((w-2*x_ins)/(w_sm+x_inter))
            if sm_columns == 0:
                sm_columns = 1

            # Force it to fill exactly
            w_sm = (w - 2*x_ins - (sm_columns-1)*x_inter)/sm_columns
            rows_needed = int(to_fit/sm_columns)
            if (to_fit%sm_columns) != 0:
                rows_needed += 1
            if rows_needed == 0:
                rows_needed = 1
            h_sm = (h - 2*y_ins - (rows_needed-1)*y_inter)/rows_needed
            h_sm -= my_txt_h

        ###
        ### OVERRIDES for SM SIZE
        ### - reworked on 2023-11-05 -- if the sm_w_override and sm_h_override's are set,
        ###   then this part calculates a new sm_columns based on the overall w specified...
        ###   and the length is whatever it takes to make it work...
        ###
        if w_sm_override is not None and h_sm_override is not None:
            # Override the size... this is only useful when the small multiples are being generated
            # for another purpose -- i.e, another visualization
            w_sm = w_sm_override
            h_sm = h_sm_override
            if grid_view:
                if shrink_wrap_rows == False:
                    w = 2*x_ins + len(col_sort) * w_sm + x_inter * (len(col_sort) - 1) 
                    h = 2*y_ins + len(row_sort) * h_sm + y_inter * (len(row_sort) - 1)
                    if draw_labels:
                        w += txt_h
                        h += txt_h
                else:
                    raise Exception("smallMultiples() - shrink_wrap_rows not implemented (w_sm_override)")
            else:
                sm_columns = int((w-2*x_ins)/(w_sm+x_inter))
                w          = 2*x_ins + sm_columns  * w_sm + x_inter*(sm_columns  - 1)
                rows_needed = int(to_fit/sm_columns)
                if (to_fit%sm_columns) != 0:
                    rows_needed += 1
                if rows_needed == 0:
                    rows_needed = 1
                h          = 2*y_ins + rows_needed * h_sm + y_inter*(rows_needed - 1)
                if draw_labels:
                    h += txt_h * rows_needed
                #h    = 2*y_ins + rows_needed * h_sm + y_inter*(rows_needed - 1)
                #w    = 2*x_ins + sm_columns  * w_sm + x_inter*(sm_columns  - 1)
                #if draw_labels:
                #    h += txt_h * rows_needed

        most_params['w']           = w_sm
        most_params['h']           = h_sm

        ### ***************************************************************************************************************************
        ### RENDERING
        ### ***************************************************************************************************************************

        # Start the SVG return result
        svg = f'<svg id="{widget_id}" x="{x_view}" y="{y_view}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        if background_override is None:
            background_color = self.co_mgr.getTVColor('background','default')
        else:
            background_color = background_override
        svg += f'<rect width="{w-1}" height="{h-1}" x="0" y="0" fill="{background_color}" stroke="{background_color}" />'
                
        text_fg     = self.co_mgr.getTVColor('label','defaultfg')
                
        # Iterate through the categories and render each one individually
        # ... draw the "all" version first is specified
        tile_i = 0
        if show_df_multiple:
            # svg += f'<rect width="{w_sm}" height="{h_sm}" x="{x_ins}" y="{y_ins}" />' # Placeholder to show layout
            my_params = most_params.copy()
            my_params['df']        = df
            my_params['x_view']    = x_ins
            my_params['y_view']    = y_ins
            my_params['widget_id'] = widget_id + "_all" 
            my_params.pop('global_max',None) # Global Max is just for the categories...

            if customize_params_fn is not None:
                customize_params_fn(None, my_params)

            sm_svg = widget_func(**my_params)
            svg += sm_svg._repr_svg_()

            if category_to_sm is not None:
                category_to_sm['__show_df_multiple__'] = sm_svg

            if category_to_instance is not None:
                instance_params = my_params.copy()
                category_to_instance['__show_df_multiple__'] = widget_func(**instance_params)
            
            if category_to_df is not None:
                category_to_df['__show_df_multiple__'] = df
                       
            if draw_labels:
                svg += f'<text x="{x_ins+w_sm/2}" y="{y_ins+h_sm+txt_h-2}" text-anchor="middle" '
                svg += f'font-family="{self.default_font}" fill="{text_fg}" font-size="{txt_h}px">Vis</text>'
            tile_i += 1
        
        # ... draw the non-grid version
        if grid_lu is None:
            for cat_i in range(0,max_categories):
                # Convert key to a category string
                key = cat_order.index[cat_i] if self.isPandas(df) else cat_order[category_by][cat_i].rows()[0] if self.isPolars(df) else None

                if type(key) == tuple:
                    key_str = ''
                    for _part in key:
                        if len(key_str) > 0:
                            key_str += '|'
                        key_str += str(_part)
                else:
                    key_str = str(key)

                # Fix for the polars-version... for some reason, sometimes it's get tupled...
                # ... 2024-02-07 -- commented this out because the polars update (assumption) fixes how this is supposed to work...
                #if type(key) == tuple and len(key) == 1:
                #    key = key[0]

                if   self.isPandas(df):
                    _tuple_ = key
                    if type(_tuple_) != tuple:
                        _tuple_ = (_tuple_, )
                    key_df = cat_gb.get_group(_tuple_)
                elif self.isPolars(df):
                    key_df = cat_gb[key]

                # Calculate placement
                xi_sm  = tile_i%sm_columns                            # index position
                yi_sm  = int(tile_i/sm_columns)
                x_sm   = x_ins + xi_sm * (w_sm + x_inter)             # screen position
                y_sm   = y_ins + yi_sm * (h_sm + my_txt_h + y_inter)

                # Render the individual small multiple
                my_params = most_params.copy()
                my_params['df']        = key_df
                my_params['x_view']    = x_sm
                my_params['y_view']    = y_sm
                my_params['widget_id'] = widget_id + "_" + str(tile_i)

                if customize_params_fn is not None:
                    customize_params_fn(key, my_params)

                sm_svg = widget_func(**my_params)
                svg += sm_svg._repr_svg_()

                # Save the small multiple svg if the parameter was passed to the method
                if category_to_sm is not None:
                    category_to_sm[key] = sm_svg
                if category_to_instance is not None:
                    instance_params = my_params.copy()
                    category_to_instance[key] = widget_func(**instance_params)
                if category_to_df is not None:
                    category_to_df[key] = key_df

                # Add the labels
                if draw_labels:
                    cropped_key_str = self.cropText(key_str, txt_h, w_sm - 0.1*w_sm)
                    svg += self.svgText(cropped_key_str, x_sm+w_sm/2, y_sm+h_sm+txt_h-2, txt_h, anchor='middle')

                # Fade any small multiples listed in the faded_sm_set
                if faded_sm_set is not None and key_str in faded_sm_set:
                    _add_txt_h = 0
                    if draw_labels:
                        _add_txt_h = y_inter + txt_h + 2
                    svg += f'<rect x="{x_sm}" y="{y_sm}" width="{w_sm}" height="{h_sm+_add_txt_h}" fill="{background_color}" fill-opacity="{faded_opacity}" stroke="None" />'

                tile_i += 1
        
        # ... draw the grid
        else:
            for key in grid_lu.keys():
                key_df = cat_gb.get_group(key) if self.isPandas(df) else cat_gb[key] if self.isPolars(df) else None
                xi_sm,yi_sm = grid_lu[key]

                if type(key) == tuple:
                    key_str = ''
                    for _part in key:
                        if len(key_str) > 0:
                            key_str += '|'
                        key_str += str(_part)
                else:
                    key_str = str(key)

                if draw_grid_row_header:
                    x_sm = x_ins + txt_h + xi_sm * (w_sm + x_inter)
                else:
                    x_sm = x_ins +         xi_sm * (w_sm + x_inter)

                if draw_grid_column_header:
                    y_sm = y_ins + txt_h + yi_sm * (h_sm + y_inter) # (h_sm + my_txt_h + y_inter) // 2023-04-21
                else:
                    if shrink_wrap_rows and draw_labels:
                        y_sm = y_ins +         yi_sm * (h_sm + txt_h + y_inter) # (h_sm + my_txt_h + y_inter) // 2023-04-26
                    else:
                        y_sm = y_ins +         yi_sm * (h_sm + y_inter) # (h_sm + my_txt_h + y_inter) // 2023-04-21

                # Render the individual small multiple
                my_params = most_params.copy()
                my_params['df']        = key_df
                my_params['x_view']    = x_sm
                my_params['y_view']    = y_sm
                my_params['widget_id'] = widget_id + "_" + str(tile_i)

                if customize_params_fn is not None:
                    customize_params_fn(key, my_params)

                sm_svg = widget_func(**my_params)
                svg += sm_svg._repr_svg_()

                # Add the labels
                if shrink_wrap_rows and draw_labels:
                    cropped_key_str = self.cropText(str(key[1]), txt_h, w_sm - 0.1*w_sm)
                    svg += self.svgText(cropped_key_str, x_sm+w_sm/2, y_sm+h_sm+txt_h-2, txt_h, anchor='middle')

                if faded_sm_set is not None and key_str in faded_sm_set:
                    svg += f'<rect x="{x_sm}" y="{y_sm}" width="{w_sm}" height="{h_sm}" fill="{background_color}" fill-opacity="{faded_opacity}" stroke="None" />'

                # Save the small multiple svg if the parameter was passed to the method
                if category_to_sm is not None:
                    category_to_sm[key] = sm_svg
                if category_to_instance is not None:
                    instance_params = my_params.copy()
                    category_to_instance[key] = widget_func(**instance_params)
                if category_to_df is not None:
                    category_to_df[key] = key_df

                tile_i += 1

            if draw_grid_column_header:
                for i in range(0,len(grid_cols)):
                    s = str(grid_cols[i])
                    s = self.cropText(s, txt_h, w_sm - 0.1*w_sm)

                    if draw_grid_row_header:
                        x = x_ins + txt_h + (w_sm + x_inter)*i + w_sm/2
                    else:
                        x = x_ins +         (w_sm + x_inter)*i + w_sm/2
                    y = y_ins + txt_h - 2
                    svg += f'<text x="{x}" y="{y}" text-anchor="middle" '
                    svg += f'font-family="{self.default_font}" fill="{text_fg}" font-size="{txt_h}px">'
                    svg += f'{s}</text>'
            if draw_grid_row_header:
                for i in range(0,len(grid_rows)):
                    s = str(grid_rows[i])

                    if len(s) > int(h_sm/txt_h):
                        s = s[:int(h_sm/txt_h)] + '...'

                    if draw_grid_column_header:
                        y = y_ins + txt_h + (h_sm + y_inter)*i + h_sm/2
                    else:
                        if shrink_wrap_rows and draw_labels:
                            y = y_ins + (h_sm + txt_h + y_inter)*i + h_sm/2
                        else:
                            y = y_ins + (h_sm +         y_inter)*i + h_sm/2
                    x = x_ins + txt_h - 2
                    svg += f'<text x="{x}" y="{y}" text-anchor="middle" '
                    svg += f'font-family="{self.default_font}" fill="{text_fg}" font-size="{txt_h}px" '
                    svg += f'transform="rotate(-90,{x},{y})">{s}</text>'

        # Draw the border
        if draw_border:
            border_color = self.co_mgr.getTVColor('border','default')
            svg += f'<rect width="{w-1}" height="{h-1}" x="0" y="0" fill="none" fill-opacity="0.0" stroke="{border_color}" />'
            
        svg += '</svg>'
        return svg

    #
    # smallMultiples() - return an instance of a Small Multiple RT Component.
    #
    def smallMultiples(self,
                       df,                                 # Dataframe
                       category_by,                        # Field(s) to separate small multiples by
                       sm_type,                            # Visualization type (e.g., 'xy', 'linkNode', ...)
                       #-----------------------------------# Defaults after this line
                       sm_params             = {},         # Dictionary for customizing widget
                       customize_params_fn   = None,       # Customize the parameters function
                       ts_field              = None,       # For any temporal components
                       count_by              = None,       # Passed to the widgets
                       color_by              = None,       # Passed to the widgets
                       global_color_order    = None,       # color by ordering... if none (default), will be calculated
                       count_by_set          = False,      # count by using a set operation
                       temporal_granularity  = None,       # Minimum temporal granularity for the temporalBarChart component
                       #-----------------------------------# Small multiple params
                       show_df_multiple      = False,      # Show the "all data" version small multiple // note issues with xy scatterplots when data is aggregated
                       max_categories        = None,       # Limit the number of small multiples shown
                       grid_view             = False,      # For two category fields, make it into a grid
                       shrink_wrap_rows      = False,      # For a grid view, shrink wrap rows
                       sort_by               = 'records',  # 'records','alpha','field', 'similarity', or a list in the category_by schema
                       sort_by_field         = None,       # For sort_by == 'field', the field name... for 'similarity', the exemplar key
                       faded_sm_set          = None,       # small multiple labels to render as faded -- stored in a set as the string label (not the index tuple)
                       faded_opacity         = 0.7,        # ... opacity to use when fading
                       x_axis_independent    = True,       # Use independent axis for x (xy, temporal, and linkNode)
                       y_axis_independent    = True,       # Use independent axis for y (xy, temporal, periodic, pie)
                       #-----------------------------------# Render-specific params
                       widget_id             = None,       # Uniquely identify this widget -- embedded into svg element ids
                       track_state           = False,      # State tracking... unsure... but state tracking is done in another way for this widget...
                       x_view                = 0,          # View coordinates
                       y_view                = 0,
                       w                     = 768,        # Width of the sm container
                       h                     = 768,        # Height of the sm container
                       w_sm_override         = None,       # Override the small multiple width
                       h_sm_override         = None,       # Override the small multiple height
                       txt_h                 = 14,         # Text height for the small multiple captions
                       x_ins                 = 2,          # Left/right inserts
                       y_ins                 = 2,          # Top/bottom inserts
                       x_inter               = 2,          # Horizontal spacing between small multiples
                       y_inter               = 4,          # Vertical spacing between small multiples
                       background_override   = None,       # Override the background color
                       draw_labels           = True,       # Draw label under each small multiple
                       draw_border           = True):      # Draw border around the whole chart
        ''' smallMultiples() - create a small multiples representation of the specified visualization component.

            Parameters
            ----------
            df                    : Pandas  or Polars DataFrame
            category_by           : Field(s) to separate small multiples by
            sm_type               : Visualization type (e.g., 'xy', 'linkNode', ...)
            sm_params             : Dictionary for customizing widget -- should be parameters for the specific component

            ts_field              : For any temporal components
            count_by              : Passed to the widgets (field to count by)
            count_by_set          : Passed to the widgets (use a set operation for counting)
            color_by              : Passed to the widgets (field to color_by)

            h_sm_override         : Override the small multiple height
            w_sm_override         : Override the small multiple width
            x_axis_independent    : Use independent axis for x (xy, temporal, and linkNode)
            y_axis_independent    : Use independent axis for y (xy, temporal, periodic, pie)

            draw_labels           : Draw label under each small multiple

            show_df_multiple      : Show the "all data" version small multiple
            max_categories        : Limit the number of small multiples shown
            grid_view             : For two category fields, make it into a grid
            shrink_wrap_rows      : For a grid view, shrink wrap rows
            sort_by               : 'records','alpha','field', 'similarity', or a list in the category_by schema
            sort_by_field         : For sort_by == 'field', the field name... for 'similarity', the exemplar key

            global_color_order    : color by ordering... if none (default), will be calculated
            customize_params_fn   : Customize the parameters function
            faded_sm_set          : small multiple labels to render as faded -- stored in a set as the string label (not the index tuple)
            faded_opacity         : ... opacity to use when fading
            temporal_granularity  : Minimum temporal granularity for the temporalBarChart component            
        '''
        _params_ = locals().copy()
        _params_.pop('self')
        return self.RTSmallMultiples(self, **_params_)

    #
    # RTSmallMultiples Class
    #
    class RTSmallMultiples(RTComponent):
        def __init__(self,
                     rt_self,
                     **kwargs):
            self.parms     = locals().copy()
            self.rt_self   = rt_self
            self.df        = rt_self.copyDataFrame(kwargs['df'])
            self.widget_id = kwargs['widget_id']
            if self.widget_id is None:
                self.widget_id = "smallMultiples_"+str(random.randint(0,65535))
            self.category_by           = kwargs['category_by']
            self.sm_type               = kwargs['sm_type']
            self.sm_params             = kwargs['sm_params']
            self.customize_params_fn   = kwargs['customize_params_fn']
            self.ts_field              = kwargs['ts_field']
            self.count_by              = kwargs['count_by']
            self.color_by              = kwargs['color_by']
            self.global_color_order    = kwargs['global_color_order']
            self.count_by_set          = kwargs['count_by_set']
            self.temporal_granularity  = kwargs['temporal_granularity']
            self.show_df_multiple      = kwargs['show_df_multiple']
            self.max_categories        = kwargs['max_categories']
            self.grid_view             = kwargs['grid_view']
            self.shrink_wrap_rows      = kwargs['shrink_wrap_rows']
            self.sort_by               = kwargs['sort_by']
            self.sort_by_field         = kwargs['sort_by_field']
            self.faded_sm_set          = kwargs['faded_sm_set']
            self.faded_opacity         = kwargs['faded_opacity']
            self.x_axis_independent    = kwargs['x_axis_independent']
            self.y_axis_independent    = kwargs['y_axis_independent']
            self.track_state           = kwargs['track_state']
            self.x_view                = kwargs['x_view']
            self.y_view                = kwargs['y_view']
            self.w                     = kwargs['w']
            self.h                     = kwargs['h']
            self.w_sm_override         = kwargs['w_sm_override']
            self.h_sm_override         = kwargs['h_sm_override']
            self.txt_h                 = kwargs['txt_h']
            self.x_ins                 = kwargs['x_ins']
            self.y_ins                 = kwargs['y_ins']
            self.x_inter               = kwargs['x_inter']
            self.y_inter               = kwargs['y_inter']
            self.background_override   = kwargs['background_override']
            self.draw_labels           = kwargs['draw_labels']
            self.draw_border           = kwargs['draw_border']

            # Ensure the timestamp field (ts_field) is set
            if (self.sm_type == 'temporalBarChart' or \
                self.sm_type == 'periodicBarChart' or \
                self.sm_type == 'calendarHeatmap') and self.ts_field is None:
                self.ts_field = self.rt_self.guessTimestampField(self.df)

            # Calculate temporal_granulaity if needed
            if kwargs['sm_type'] == 'temporalBarChart' and self.temporal_granularity is None:
                self.temporal_granularity = self.rt_self.temporalGranularity(self.df, self.ts_field)

            # Geometry lookup for tracking state
            self.last_render          = None
            self.category_to_sm       = {}
            self.category_to_instance = {}
            self.category_to_df       = {}

        #
        # SVG Representation Renderer
        #
        def _repr_svg_(self):
            if self.last_render is None:
                self.renderSVG()
            return self.last_render
        
        #
        # renderSVG() - create the SVG
        #
        def renderSVG(self, just_calc_max=False):
            self.category_to_sm       = {} # <== Reset the state tracking for all three dictionaries...
            self.category_to_instance = {}
            self.category_to_df       = {}
            self.last_render = self.rt_self.__smallMultiples_impl__(self.df,
                                                                    category_by           = self.category_by,
                                                                    sm_type               = self.sm_type,
                                                                    sm_params             = self.sm_params,
                                                                    customize_params_fn   = self.customize_params_fn,
                                                                    ts_field              = self.ts_field,
                                                                    count_by              = self.count_by,
                                                                    color_by              = self.color_by,
                                                                    global_color_order    = self.global_color_order,
                                                                    count_by_set          = self.count_by_set,
                                                                    temporal_granularity  = self.temporal_granularity,
                                                                    show_df_multiple      = self.show_df_multiple,
                                                                    max_categories        = self.max_categories,
                                                                    grid_view             = self.grid_view,
                                                                    shrink_wrap_rows      = self.shrink_wrap_rows,
                                                                    sort_by               = self.sort_by,
                                                                    sort_by_field         = self.sort_by_field,
                                                                    faded_sm_set          = self.faded_sm_set,
                                                                    faded_opacity         = self.faded_opacity,
                                                                    x_axis_independent    = self.x_axis_independent,
                                                                    y_axis_independent    = self.y_axis_independent,
                                                                    category_to_sm        = self.category_to_sm,        # <== State Tracking
                                                                    category_to_instance  = self.category_to_instance,  # <== State Tracking
                                                                    category_to_df        = self.category_to_df,        # <== State Tracking
                                                                    widget_id             = self.widget_id,
                                                                    x_view                = self.x_view,
                                                                    y_view                = self.y_view,
                                                                    w                     = self.w,
                                                                    h                     = self.h,
                                                                    w_sm_override         = self.w_sm_override,
                                                                    h_sm_override         = self.h_sm_override,
                                                                    txt_h                 = self.txt_h,
                                                                    x_ins                 = self.x_ins,
                                                                    y_ins                 = self.y_ins,
                                                                    x_inter               = self.x_inter,
                                                                    y_inter               = self.y_inter,
                                                                    background_override   = self.background_override,
                                                                    draw_labels           = self.draw_labels,
                                                                    draw_border           = self.draw_border)
            return self.last_render

        #
        # overlappingDataFrames() - Determine which dataframe geometris overlap with a specific one
        #
        def overlappingDataFrames(self, to_intersect):
            _dfs = []
            for _category in self.category_to_sm.keys():
                _sm   = self.category_to_sm[_category]._repr_svg_()
                x,y   = self.rt_self.__extractSVGXAndY__(_sm)
                w,h   = self.rt_self.__extractSVGWidthAndHeight__(_sm)
                _poly = Polygon([[x,y],[x,y+h],[x+w,y+h],[x+w,y]])
                if _poly.intersects(to_intersect):
                    _dfs.append(self.category_to_df[_category])
            if len(_dfs) > 0:
                _dfs_together = self.rt_self.concatDataFrames(_dfs)
                if   self.rt_self.isPandas(self.df):
                    _dfs_together = _dfs_together.drop_duplicates()
                elif self.rt_self.isPolars(self.df):
                    _dfs_together = _dfs_together.unique()                
                else:
                    raise Exception('RTSmallMultiples.overlappingDataFrames() - only pandas and polars are supported')
                return _dfs_together
            else:
                return None

    #
    # __alignDataFrames__()
    # ... create a single dataframe with all the required columns
    # ... for dataframes that don't have the columns, don't add them
    # ... no idea what the performance of this looks like (or if it impacts the original passed in dataframes)
    #
    def __alignDataFrames__(self, 
                            df,                     # single dataframe or a list of dataframes
                            required_columns):      # required columns as a set
        # if it's already one, just return it...
        if self.isPandas(df) or self.isPolars(df) or type(df) != list:
            return df
        
        # Otherwise, concat together if the fields are ther... when wouldn't the fields be there? 2023-11-10
        _dfs_ = []                                
        for _df in df:
            if len(set(_df.columns) & required_columns) == len(required_columns):
                _dfs_.append(_df)

        return self.concatDataFrames(_dfs_)

    #
    # createSmallMultiple()
    # ... for use by other widgets...
    # ... returns a dictionary of the str keys to svg small multiples
    #
    def createSmallMultiples(self,
                             df,                       # Single dataframe or list of dataframes
                             str_to_df_list,           # things to their related dataframes
                             str_to_xy,                # placement of things for the SVG

                             count_by,                 # what to count by... None == rows
                             count_by_set,             # if counting by should be done by sets versus numerical summation
                             color_by,                 # how to color... None == no (default) color

                             ts_field,                 # timestamp field... if none, will be attempted to pull from sm_params

                             parent_id,                # parent widget id

                             sm_type,                  # widget type -- should be the exact string for the method
                             sm_params,                # dictionary of parameters to customize the widget

                             x_axis_independent,       # Use independent axis for x (xy, temporal, and linkNode)
                             y_axis_independent,       # Use independent axis for y (xy, temporal, periodic, pie)

                             sm_w,                     # width of the small multiple
                             sm_h):                    # heigh of the small multiple

        # Determine the required parameters for each small multiple
        sm_all_params = sm_params.copy()
        sm_all_params['count_by']     = count_by
        sm_all_params['count_by_set'] = count_by_set
        sm_all_params['color_by']     = color_by
        required_columns = getattr(self, f'{sm_type}RequiredFields')(**sm_all_params)

        # Align each individual dataframe list with the required columns ... then concatenate them together
        my_cat_column = 'my_cat_col_' + str(random.randint(0,65535))

        df_example = None
        if type(df) == list and len(df) > 0:
            df_example = df[0]
        else:
            df_example = df 

        if   self.isPandas(df_example):
            _dfs_ = []
            for k in str_to_df_list.keys():
                df_list    = str_to_df_list[k]
                aligned_df = self.__alignDataFrames__(df_list,required_columns)
                pd.set_option('mode.chained_assignment', None)    # verified that the operation occurs correctly 2023-01-11 20:00EST
                aligned_df[my_cat_column] = k
                pd.set_option('mode.chained_assignment', 'warn')
                _dfs_.append(aligned_df)
            master_df = pd.concat(_dfs_)
        elif self.isPolars(df_example):
            _dfs_ = []
            for k in str_to_df_list.keys():
                df_list    = str_to_df_list[k]
                aligned_df = self.__alignDataFrames__(df_list,required_columns)
                #aligned_df = aligned_df.with_columns(pl.lit(k).alias(my_cat_column)) # original
                if type(k) == str: aligned_df = aligned_df.with_columns(pl.lit(k).alias(my_cat_column)) # attempt to fix due to unhashable type...
                else:              aligned_df = aligned_df.with_columns(pl.lit(k).list.join('|').alias(my_cat_column)) # attempt to fix due to unhashable type...
                _dfs_.append(aligned_df)
            master_df = pl.concat(_dfs_)
        else:
            raise Exception('RTSmallMultiples.createSmallMultiples() - only pandas and polars supported', type(df))

        # Find the timestamp field... or figure out what to use...
        accepted_args = set(inspect.getfullargspec(getattr(self, sm_type)).args)
        if ('ts_field' in accepted_args and sm_type != 'linkNode' and sm_type != 'link') or \
           ((sm_type == 'linkNode' or sm_type == 'link') and 'timing_marks' in sm_params.keys() and sm_params['timing_marks'] == True):
            if 'ts_field' in sm_params.keys():     # precedence is sm_params ts_field
                ts_field = sm_params['ts_field']
            elif ts_field is None:                 # best guess from the columns // copied from temporalBarChart method
                ts_field = self.guessTimestampField(master_df)
            else:                                  # use the ts_field passed into this method
                pass

        # Call the original smallMultiples method with the lookup parameter present
        my_category_to_sm = {}
        self.__smallMultiples_impl__(master_df, 
                                     my_cat_column, 
                                     sm_type, 
                                     sm_params=sm_params,
                                     ts_field=ts_field,
                                     count_by=count_by,
                                     color_by=color_by,
                                     count_by_set=count_by_set,
                                     show_df_multiple=False,
                                     x_axis_independent=x_axis_independent,
                                     y_axis_independent=y_axis_independent,
                                     category_to_sm=my_category_to_sm,
                                     widget_id=parent_id,
                                     w_sm_override=sm_w,
                                     h_sm_override=sm_h)

        # Re-write the SVG for the xy coordinate... // seems kindof clunky to do it this way... fragile
        updated_category_to_sm = {}
        for k in my_category_to_sm:
            k_svg = my_category_to_sm[k]._repr_svg_()
            if k in str_to_xy.keys():
                _coords_ = str_to_xy[k]
            elif k[0] in str_to_xy.keys():  # This solves a polars problem about how keys are returned 2024-02-09
                _coords_ = str_to_xy[k[0]]
            else:
                print('createSmallMultiples() exception...')
                print('key',            k)
                print('category_to_sm', my_category_to_sm)
                print('str_to_xy',      str_to_xy)

            updated_category_to_sm[k] = self.__overwriteSVGOriginPosition__(k_svg, _coords_, sm_w, sm_h)

        return updated_category_to_sm

    #
    # __orderSmallMultiplesBySimilarity__()
    # ... produce an ordered list by the small multiple similiarity
    # ... meant to be called from the smallMultiples() method...  that should be the kwargs parameter space
    # ... produces the "sort_by" list of the keys derived from the category_by variable
    # ... if the sort_by_field is set, that's the exemplar key...
    #
    def __orderSmallMultiplesBySimilarity__(self, **kwargs):
        if kwargs['sm_type'] != 'pieChart'         and \
           kwargs['sm_type'] != 'temporalBarChart' and \
           kwargs['sm_type'] != 'periodicBarChart' and \
           kwargs['sm_type'] != 'histogram':
            raise Exception(f'__orderSmallMultiplesBySimilarity__() -- sm_type "{kwargs["sm_type"]}" does not support similarity metrics')
        
        # Find the base small multiples dimensions...
        # ... may cause some of the similarity calcs to not make sense in the final rendering...
        # ... i guess it really only affects the temporalBarChart... all the other feature vecs are render-resolution-independent...
        dim_sm = getattr(self, f'{kwargs["sm_type"]}SmallMultipleDimensions')(**kwargs['sm_params'])

        # Create class instances for all of the small multiples...
        params_copy = kwargs.copy()
        params_copy['category_to_instance'] = category_to_instance = {} # store the instances here
        if params_copy['sort_by_field'] is None:                        # if no exemplar provided, use the "all" version
            params_copy['show_df_multiple'] = True
        
        if 'w_sm_override' not in params_copy.keys():                   # set the override for the width & height
            params_copy['w_sm_override'] = dim_sm[0]
        if 'h_sm_override' not in params_copy.keys():
            params_copy['h_sm_override'] = dim_sm[1]

        if 'sort_by' in params_copy.keys():                             # remove sort_by... because otherwise we get infinite looping
            params_copy.pop('sort_by')

        if 'max_categories' in params_copy.keys():                      # remove max categories so that we can compare against everything
            params_copy.pop('max_categories')

        self.__smallMultiples_impl__(**params_copy)                     # perform the actual instance creation -- results in category_to_instance variable

        # Have the classes create the their feature vectors
        category_to_fv = {}
        for k in category_to_instance.keys():
            category_to_fv[k] = category_to_instance[k].smallMultipleFeatureVector()
        
        # Create master feature vector list
        master_features = set()
        for k in category_to_fv.keys():
            fv = category_to_fv[k]
            master_features |= set(fv.keys())
        master_features = list(master_features)
        master_features_lu = {}
        for i in range(0,len(master_features)):
            master_features_lu[master_features[i]] = i
        
        # Orient the individual small multiple features so that they are all the same
        norm_to_fv = {}
        for k in category_to_fv.keys():
            sm_fv = category_to_fv[k]
            as_np = np.zeros(len(master_features))
            for fv_name in sm_fv.keys():
                fv_name_i = master_features_lu[fv_name]
                as_np[fv_name_i] = sm_fv[fv_name]
            norm_to_fv[k] = as_np

        # Determine the exemplar small multiple
        exemplar_key = kwargs['sort_by_field']
        if exemplar_key is None:
            exemplar_key = '__show_df_multiple__'

        # Calculate the distance from all to the exemplar...
        index_values,values = [],[]
        for k in norm_to_fv.keys():
            if k != '__show_df_multiple__':
                index_values.append(k)
                if k == exemplar_key:
                    values.append(0.0)
                else:
                    values.append(np.linalg.norm(norm_to_fv[k]-norm_to_fv[exemplar_key]))
        
        # Sort them... and return the keyed sorted list...
        sorted_sm = list(pd.Series(values, index=index_values).sort_values().index)
        return sorted_sm

    #
    # __overwriteSVGOriginPosition__()
    # ... overwrite the position of an SVG element with a new x,y coordinate
    # ... really fragile... should only be used with SVG generated by this package...
    #
    def __overwriteSVGOriginPosition__(self, svg, xy_tuple, sm_w=0, sm_h=0):
        # Correct way to do this...
        # ... however, it errors out with a 'cannot serialize #.###... (type float64)'
        #my_tree = ET.fromstring(svg)
        #my_tree.set('x',xy_tuple[0])
        #my_tree.set('y',xy_tuple[1])
        #return ET.tostring(my_tree,encoding='utf8',method='xml')

        # Incorrect way to do this...
        i0 = svg.index('x="')
        i1 = svg.index('"',i0+3)
        svg = svg[:i0] + 'x="' + str(xy_tuple[0] - sm_w/2) + '" ' + svg[i1+1:]

        i0 = svg.index('y="')
        i1 = svg.index('"',i0+3)
        svg = svg[:i0] + 'y="' + str(xy_tuple[1] - sm_h/2) + '" ' + svg[i1+1:]
        
        return svg

    #
    # __overwriteSVGID__()
    # ... really fragile... if i were a better person, i'd be using the ET library...
    #
    def __overwriteSVGID__(self, svg, new_id):
        if svg.startswith('<svg') and 'id="' in svg and svg.index('id="') < svg.index('>'):
            i0 = svg.index('id="')
            i1 = svg.index('"',i0+3)
            svg = svg[:i0] + 'id="' + new_id + '" ' + svg[i1+1:]
        return svg

    #
    # __extractSVGXAndY__()
    # ... extract the x and y coordinate of the SVG
    # ... really fragile... should only be used with SVG generated by this package...
    #
    def __extractSVGXAndY__(self, svg):
        i0 = svg.index('x="')
        i1 = svg.index('"',i0+3)
        x  = float(svg[i0+3:i1])
        i0 = svg.index('y="')
        i1 = svg.index('"',i0+3)
        y  = float(svg[i0+3:i1])
        return x,y

    #
    # __overwriteSVGXAndY__()
    # ... overwrite the position of an SVG element with a new x,y coordinate
    # ... really fragile... should only be used with SVG generated by this package...
    #
    def __overwriteSVGXAndY__(self, svg, xy_tuple):
        i0 = svg.index('x="')
        i1 = svg.index('"',i0+3)
        svg = svg[:i0] + 'x="' + str(xy_tuple[0]) + '" ' + svg[i1+1:]
        i0 = svg.index('y="')
        i1 = svg.index('"',i0+3)
        svg = svg[:i0] + 'y="' + str(xy_tuple[1]) + '" ' + svg[i1+1:]
        return svg

    #
    # __extractSVGWidthAndHeight__()
    # ... extract the width and height of an SVG section
    # ... really fragile... should only be used with SVG generated by this package...
    #
    def __extractSVGWidthAndHeight__(self, svg):
        if hasattr(svg, 'w') and hasattr(svg, 'h'):
            _width, _height = svg.w, svg.h
        else:
            if type(svg) != str: svg = svg._repr_svg_()
            i0 = svg.index('width="')
            i1 = svg.index('"',i0+len('width="'))
            _width = float(svg[i0+len('width="'):i1])
            i0 = svg.index('height="')
            i1 = svg.index('"',i0+len('height="'))
            _height = float(svg[i0+len('height="'):i1])
        return _width,_height
    
    #
    # Add A Title
    # ... a lot of assumptions built into this one -- specifically 
    #     that the passed in SVG is at coordinates (0,0)
    #
    def addTitleToSVG(self, _svg_, _title_, txt_h=12, _color_=None, _font_=None):
        if type(_svg_) != str:
            _svg_ = _svg_._repr_svg_()
        w,h = self.__extractSVGWidthAndHeight__(_svg_)
        _co = self.co_mgr.getTVColor('background','default')
        _new_svg  =  f'<svg x="0" y="0" width="{w}" height="{h+txt_h+4}">'
        _new_svg += f'<rect x="0" y="0" width="{w}" height="{h+txt_h+4}" fill="{_co}" />'
        _new_svg += _svg_
        _cropped  = self.cropText(_title_, txt_h, w)
        _new_svg += self.svgText(_cropped, w/2, h + txt_h + 1, txt_h, anchor='middle', color=_color_, font=_font_)
        _new_svg += f'</svg>'
        return _new_svg
    
    #
    # Make a table out of SVG tiles
    # - for equal sized elements (doesn't really need to be...  but let's just assume)
    # - place into a grid
    #
    def table(self, svg_list, per_row=4, spacer=0, background_override=None):
        rows,so_far = [],[]
        for _svg_ in svg_list:
            so_far.append(_svg_)
            if len(so_far) >= per_row:
                rows.append(self.tile(so_far, spacer=spacer, background_override=background_override)._repr_svg_())
                so_far = []
        if len(so_far) > 0:
            rows.append(self.tile(so_far, spacer=spacer, background_override=background_override)._repr_svg_())
        return self.tile(rows, horz=False, spacer=spacer, background_override=background_override)

    #
    # svgObject() - simple container to return an svg string
    #
    class svgObject(object):
        def __init__(self, svg_str):
            self.my_svg_str = svg_str
        def _repr_svg_(self):
            return self.my_svg_str

    #
    # Tile a list of SVG's
    #
    def tile(self, svg_list, horz=True, spacer=0, background_override=None):
        if background_override is None: background_override = self.co_mgr.getTVColor('border','default')
        svg = []
        if horz:
            w_overall,h_max = 0,0
            for _svg in svg_list:
                if type(_svg) != str: _svg = _svg._repr_svg_()
                w,h       =  self.__extractSVGWidthAndHeight__(_svg)
                w_overall += w+spacer
                h_max     =  max(h_max, h)
            w_overall = w_overall - spacer # there will be an extra one that needs to be deleted
            svg.append(f'<svg width="{w_overall}" height="{h_max}" x="0" y="0" xmlns="http://www.w3.org/2000/svg">')
            svg.append(f'<rect width="{w_overall}" height="{h_max}" x="0" y="0" fill="{background_override}" />')
            w_overall = 0
            for _svg in svg_list:
                if type(_svg) != str: _svg = _svg._repr_svg_()
                w,h  = self.__extractSVGWidthAndHeight__(_svg)
                svg.append(self.__overwriteSVGOriginPosition__(_svg, (w_overall + w/2, h/2), w, h))
                w_overall += w+spacer
            svg.append('</svg>')
        else:
            w_max,h_overall = 0,0
            for _svg in svg_list:
                if type(_svg) != str: _svg = _svg._repr_svg_()
                w,h       =  self.__extractSVGWidthAndHeight__(_svg)
                h_overall += h+spacer
                w_max     =  max(w_max, w)
            h_overall = h_overall - spacer
            svg.append(f'<svg width="{w_max}" height="{h_overall}" x="0" y="0" xmlns="http://www.w3.org/2000/svg">')
            svg.append(f'<rect width="{w_max}" height="{h_overall}" x="0" y="0" fill="{background_override}" />')
            h_overall = 0
            for _svg in svg_list:
                if type(_svg) != str: _svg = _svg._repr_svg_()
                w,h  = self.__extractSVGWidthAndHeight__(_svg)
                svg.append(self.__overwriteSVGOriginPosition__(_svg, (w/2, h_overall + h/2), w, h))
                h_overall += h+spacer
            svg.append('</svg>')
        return self.svgObject(''.join(svg))

    #
    # xyGrid() in the style of the all parameters?
    #
    def xyGrid(self, 
               df, 
               color_by=None,
               dot_size='small',
               w_tile=128, 
               h_tile=128):
        _fields = df.columns
        _rows   = []
        for row_field in _fields:
            _row = []
            for col_field in _fields:
                if row_field == col_field:
                    _row.append(self.xy(df, x_field=row_field, y_field=col_field, dot_size=None,    w=w_tile, h=h_tile, 
                                        render_x_distribution=int(w_tile/3), distribution_style='inside',
                                        background_override='#e0e0e0'))
                else:
                    if color_by is None:
                        _row.append(self.xy(df, x_field=row_field, y_field=col_field, dot_size=dot_size, w=w_tile, h=h_tile, color_magnitude='stretch'))
                    else:
                        _row.append(self.xy(df, x_field=row_field, y_field=col_field, dot_size=dot_size, w=w_tile, h=h_tile, color_by=color_by))
            _rows.append(self.tile(_row)._repr_svg_())
        return self.tile(_rows, horz=False)

#
# Find optimal fit for small multiples
# 
#
def findOptimalFit(w_sm,  # Minimum width of small multiple 
                   h_sm,  # Minimum height of small multiple
                   txt_h, # If labeling, this should be non-zero
                   w,     # Width of widget
                   h,     # Height of widget
                   n):    # Number to fit
    
    # Base (worse) case... does the minimum fit? ... if not, just return the minimum
    if howManyFit(w_sm,h_sm+txt_h,w,h) < n:
        return w_sm, h_sm
    
    # Binary search to where they don't fit
    iters = 0
    w0    = w_sm
    w1    = w
    while int(w0) < int(w1) and iters < w:
        w_mid  = (w0+w1)/2
        h_prop = w_mid * h_sm/w_sm
        if howManyFit(w_mid,h_prop+txt_h,w,h) <= n:
            w1 = w_mid
        else:
            w0 = w_mid
        iters += 1
    
    return w_mid,h_prop

#
# How many fit?
#
def howManyFit(w_sm,h_sm,w,h):
    cols = int(w/w_sm)
    rows = int(h/h_sm)
    return rows*cols

#
# fieldOrder()
#
def fieldOrder(rt_self,
               df,
               field,
               sort_by,
               sort_by_field):
    if   rt_self.isPandas(df):
        return __fieldOrder_pandas__(rt_self, df, field, sort_by, sort_by_field)
    elif rt_self.isPolars(df):
        return __fieldOrder_polars__(rt_self, df, field, sort_by, sort_by_field)
    else:
        raise Exception('SmallMultiples.fieldOrder() -- only pandas or polars supported')

#
def __fieldOrder_pandas__(rt_self,
                          df, 
                          field, 
                          sort_by, 
                          sort_by_field):
    # Sort by rows
    if   sort_by == 'records' or (sort_by == 'field' and sort_by_field is None):
        return df.groupby(field).size().sort_values(ascending=False)
    # Sort by a field
    elif sort_by == 'field':        
        if rt_self.fieldIsArithmetic(df,sort_by_field):
            #print('by field (arithmetic)')
            return df.groupby(field)[sort_by_field].sum().sort_values(ascending=False)
        else:
            #print('by field (set operation)')
            _df = pd.DataFrame(df.groupby([field,sort_by_field]).size())
            return _df.groupby(field).size().sort_values(ascending=False)
    # Sort naturally
    elif sort_by == 'natural':
        if   field.startswith('|tr|month|'):
            _set,_arr = set(df[field]),[]
            for _mon in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']:
                if _mon in _set:
                    _arr.append(_mon)
            _series = pd.Series(_arr)
            _series.index = _arr
            return _series
        elif field.startswith('|tr|day_of_week|'):
            _set,_arr = set(df[field]),[]
            for _dow in ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']:
                if _dow in _set:
                    _arr.append(_dow)
            _series = pd.Series(_arr)
            _series.index = _arr
            return _series
        else:
            return df.groupby(field).count()
    # Alphabetical
    else:
        #print('by alpha')
        return df.groupby(field).count()

#
def __fieldOrder_polars__(rt_self,
                          df, 
                          field, 
                          sort_by, 
                          sort_by_field):
    # Sort by rows
    if   sort_by == 'records' or (sort_by == 'field' and sort_by_field is None):
        return df.drop(set(df.columns) - set([field]))   \
                 .rename({field:'field'})                \
                 .group_by('field', maintain_order=True) \
                 .agg(pl.count().alias('__count__'))     \
                 .sort('__count__', descending=True)
    # Sort by a field
    elif sort_by == 'field':        
        if rt_self.fieldIsArithmetic(df,sort_by_field):
            df_min = df.drop(set(df.columns) - set([field]) - set([sort_by_field])).rename({field:'field',sort_by_field:'__count__'})
            return df_min.group_by('field', maintain_order=True).agg(pl.sum('__count__')).sort('__count__', descending=True)
        else:
            df_min = df.drop(set(df.columns) - set(field) - set([sort_by_field])).rename({field:'field', sort_by_field:'__count__'})
            return df_min.group_by('field', maintain_order=True).n_unique().sort('__count__', descending=True)
    # Sort naturally
    elif sort_by == 'natural':
        if   field.startswith('|tr|month|'):
            _set,_arr = set(df[field]),[]
            for _mon in ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']:
                if _mon in _set:
                    _arr.append(_mon)
            return pl.DataFrame({'field':_arr})
        elif field.startswith('|tr|day_of_week|'):
            _set,_arr = set(df[field]),[]
            for _dow in ['Sun','Mon','Tue','Wed','Thu','Fri','Sat']:
                if _dow in _set:
                    _arr.append(_dow)
            return pl.DataFrame({'field':_arr})
        else:
            return df.groupby(field).count()
    # Alphabetical
    else:
        df = df.drop(df.columns - set([field])).rename({field:'field'}).unique()
        return df.sort('field')
