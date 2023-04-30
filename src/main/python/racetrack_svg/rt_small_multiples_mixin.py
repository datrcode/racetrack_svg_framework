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

import pandas as pd
import numpy as np
import random
import inspect

import xml.etree.ElementTree as ET

__name__ = 'rt_small_multiples_mixin'

#
# Small Multiples Mixin
#
class RTSmallMultiplesMixin(object):
    #
    # For future reference, to make this work with a new widget:
    #
    # - Add to list of widgets in the racetrack.py
    # - Add to widget check in the beginning of this method
    # - If the widget uses timestamps, add it to the 'guess the timestamp column' area
    # - Add to widget dependent axis parts
    #
    def smallMultiples(self,
                       df,                                 # Dataframe
                       category_by,                        # Field(s) to separate small multiples by
                       sm_type,                            # Visualization type (e.g., 'xy', 'linkNode', ...)

                       #-----------------------------------# Defaults after this line

                       sm_params             = {},         # Dictionary for customizing widget

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
                       draw_labels           = True,       # Draw label under each small multiple
                       draw_border           = True):      # Draw border around the whole chart
        
        my_params = locals().copy()

        # Preserve original
        df = df.copy()

        # Check widget ... since there's widget specific processing
        _implemented_types = ['boxplot', 'calendarHeatmap', 'histogram', 'linkNode', 'periodicBarChart', 'pieChart', 'temporalBarChart', 'xy']
        if (sm_type in _implemented_types) == False:
            raise Exception(f'smallMultipes: widget type "{sm_type}" not implemented (initial check)')
        
        # Generate a widget id if it's not already set
        if widget_id is None:
            widget_id = "smallmultiples_" + str(random.randint(0,65535))

        ### ***************************************************************************************************************************
        ### PARAMETERS
        ### ***************************************************************************************************************************

        # Make the categories into a list (if not already so)
        if type(category_by) != list:
            category_by = [category_by]

        # Organize by similarity...        
        if sort_by == 'similarity':
            params_copy = my_params.copy()
            params_copy.pop('self')
            sort_by = self.__orderSmallMultiplesBySimilarity__(**params_copy)

        # Transform the categories if necessary (and the count and color bys as well)
        df, category_by = self.transformFieldListAndDataFrame(df, category_by)
        df, color_bys = self.transformFieldListAndDataFrame(df, [color_by])
        color_by = color_bys[0]
        df, count_bys = self.transformFieldListAndDataFrame(df, [count_by])
        count_by = count_bys[0]

        # Transform any of the sm params...
        required_columns = getattr(self, f'{sm_type}RequiredFields')(**sm_params)
        for _field in required_columns:
            df, _throwaway = self.transformFieldListAndDataFrame(df, [_field])

        # Ensure the timestamp field (ts_field) is set
        if (sm_type == 'temporalBarChart' or \
            sm_type == 'periodicBarChart' or \
            sm_type == 'calendarHeatmap') and ts_field is None:
            choices = df.select_dtypes(np.datetime64).columns
            if len(choices) == 1:
                ts_field = choices[0]
            elif len(choices) > 1:
                print('multiple timestamp fields... choosing the first (smallMultiples)')
                ts_field = choices[0]
            else:
                raise Exception('no timestamp field supplied to smallMultiples(), cannot automatically determine field')
                
        # Calculate temporal_granulaity if needed
        if sm_type == 'temporalBarChart' and temporal_granularity is None:
            temporal_granularity = self.temporalGranularity(df, ts_field)

        # Determine categories and ordering // cat_order and cat_gb need to be set
        cat_gb = df.groupby(category_by)
        if   sort_by is None or sort_by == 'alpha':
            cat_order = cat_gb.count()
        elif type(sort_by) == list:            
            cat_order = pd.Series(np.zeros(len(sort_by)), index=sort_by)
        elif sort_by == 'records' or sort_by_field is None or sort_by_field in category_by:    
            cat_order = cat_gb.size().sort_values(ascending=False)
        elif sort_by == 'field':
            # Count by numeric summation
            if df[sort_by_field].dtypes == np.int64   or df[sort_by_field].dtypes == np.int32 or \
               df[sort_by_field].dtypes == np.float64 or df[sort_by_field].dtypes == np.float32:
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
        widget_create_class = getattr(self, 'RT' + sm_type[0].upper() + sm_type[1:])

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

        # Handle dependent axes ... unfortunately, this is widget dependent
        if x_axis_independent == False or y_axis_independent == False:
            #
            # xy and x-axis
            #
            if sm_type == 'xy' and x_axis_independent == False:
                sm_x_axis = widget_id + "_x"
                x_field_is_scalar = True # Default for the xy widget
                if 'x_field_is_scalar' in sm_params.keys():
                    x_field_is_scalar = sm_params['x_field_is_scalar']
                x_is_time,x_label_min,x_label_max,xTrans = self.xyCreateAxisColumn(df, sm_params['x_field'], x_field_is_scalar, sm_x_axis)
                most_params['x_axis_col']  = sm_x_axis
                most_params['x_is_time']   = x_is_time
                most_params['x_label_min'] = x_label_min
                most_params['x_label_max'] = x_label_max
                most_params['x_trans_func'] = xTrans

            #
            # xy and y-axis
            #
            if sm_type == 'xy' and y_axis_independent == False:
                sm_y_axis = widget_id + "_y"
                y_field_is_scalar = True # Default for the xy widget
                if 'y_field_is_scalar' in sm_params.keys():
                    y_field_is_scalar = sm_params['y_field_is_scalar']
                y_is_time,y_label_min,y_label_max,yTrans = self.xyCreateAxisColumn(df, sm_params['y_field'], y_field_is_scalar, sm_y_axis)
                most_params['y_axis_col']  = sm_y_axis
                most_params['y_is_time']   = y_is_time
                most_params['y_label_min'] = y_label_min
                most_params['y_label_max'] = y_label_max
                most_params['y_trans_func'] = yTrans
            
            #
            # temporalBarChart and x-axis
            #
            if x_axis_independent == False and sm_type == 'temporalBarChart':
                most_params['ts_min'] = df[ts_field].min()
                most_params['ts_max'] = df[ts_field].max()
            
            #
            # linkNode and position for bounds
            #
            if x_axis_independent == False and sm_type == 'linkNode':
                most_params['use_pos_for_bounds'] = False
            
            #
            # histogram/periodicBarChart/temporalBarChart/boxplot and y-axis
            #
            if y_axis_independent == False and (sm_type == 'histogram' or sm_type == 'periodicBarChart' or sm_type == 'temporalBarChart' or sm_type == 'boxplot'):
                global_min,global_max = None,None

                if max_categories is None:
                    max_categories = len(cat_order)

                for cat_i in range(0,max_categories):
                    key = cat_order.index[cat_i]
                    key_df = cat_gb.get_group(key)
                    my_params = most_params.copy()
                    my_params['df'] = key_df
                    my_params['just_calc_max'] = True
                    local_min,local_max = widget_func(**my_params)
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
                    key = cat_order.index[cat_i]
                    key_df = cat_gb.get_group(key)
                    my_params = most_params.copy()
                    my_params['df'] = key_df
                    my_params['just_calc_max'] = True
                    local_max,local_min = widget_func(**my_params)
                    if global_max is None:
                        global_max,global_min = local_max,local_min
                    else:
                        if local_max > global_max:
                            global_max = local_max
                        if local_min < global_min:
                            global_min = local_min
                most_params['global_max'] = global_max
                most_params['global_min'] = global_min

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
                    col_lu[col_sort.index[i]] = i
                    col_order.append(col_sort.index[i])

                row_lu,row_order = {},[]
                for i in range(0,len(row_sort)):
                    row_lu[row_sort.index[i]] = i
                    row_order.append(row_sort.index[i])

                for key,key_df in cat_gb:
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
                row_gb   = df.groupby(category_by[0])

                longest_row = 1

                row_lu,row_order,grid_rows = {},[],[]
                for i in range(0,len(row_sort)):
                    k = row_sort.index[i]
                    row_lu[k] = i
                    row_order.append(k)
                    grid_rows.append(k)
                    k_df = row_gb.get_group(k)
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
            if show_df_multiple:
                to_fit = max_categories + 1
            else:
                to_fit = max_categories

            # Determine the width and height of the small multiples themselves
            # Minimum dimensions for the widget
            dim_min  = getattr(self, f'{sm_type}MinimumDimensions')  (**most_params)
            
            # Binary search to find the best fit
            w_min_adj = dim_min[0] + x_inter
            h_min_adj = dim_min[1] + y_inter
            if draw_labels:
                my_txt_h = txt_h
            else:
                my_txt_h = 0
                
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
                h    = 2*y_ins + rows_needed * h_sm + y_inter*(rows_needed - 1)
                w    = 2*x_ins + sm_columns  * w_sm + x_inter*(sm_columns  - 1)
                if draw_labels:
                    h += txt_h * rows_needed

        most_params['w']           = w_sm
        most_params['h']           = h_sm

        ### ***************************************************************************************************************************
        ### RENDERING
        ### ***************************************************************************************************************************

        # Start the SVG return result
        svg = f'<svg id="{widget_id}" x="{x_view}" y="{y_view}" width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">'
        background_color = self.co_mgr.getTVColor('background','default')
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

            sm_svg = widget_func(**my_params)
            svg += sm_svg

            if category_to_instance is not None:
                instance_params = my_params.copy()
                instance_params['rt_self'] = self
                category_to_instance['__show_df_multiple__'] = widget_create_class(**instance_params)
                       
            if draw_labels:
                svg += f'<text x="{x_ins+w_sm/2}" y="{y_ins+h_sm+txt_h-2}" text-anchor="middle" '
                svg += f'font-family="{self.default_font}" fill="{text_fg}" font-size="{txt_h}px">Vis</text>'
            tile_i += 1
        
        # ... draw the non-grid version
        if grid_lu is None:
            for cat_i in range(0,max_categories):
                # Convert key to a category string
                key = cat_order.index[cat_i]

                if type(key) == tuple:
                    key_str = ''
                    for _part in key:
                        if len(key_str) > 0:
                            key_str += '|'
                        key_str += str(_part)
                else:
                    key_str = str(key)
                key_df = cat_gb.get_group(key)

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
                sm_svg = widget_func(**my_params)
                svg += sm_svg

                # Save the small multiple svg if the parameter was passed to the method
                if category_to_sm is not None:
                    category_to_sm[key] = sm_svg
                if category_to_instance is not None:
                    instance_params = my_params.copy()
                    instance_params['rt_self'] = self
                    category_to_instance[key] = widget_create_class(**instance_params)

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
                key_df = cat_gb.get_group(key)
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
                sm_svg = widget_func(**my_params)
                svg += sm_svg

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
                    instance_params['rt_self'] = self
                    category_to_instance[key] = widget_create_class(**instance_params)

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
            svg += f'<rect width="{w-1}" height="{h-1}" x="0" y="0" fill-opacity="0.0" stroke="{border_color}" />'
            
        svg += '</svg>'
        return svg

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
        if type(df) == pd.DataFrame or type(df) != list:
            return df

        # combined together if they meet the required columns
        combined_df = pd.DataFrame()
        for _df in df:
            if len(set(_df.columns) & required_columns) == len(required_columns):
                combined_df = pd.concat([combined_df,_df])
        return combined_df

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
        master_df = pd.DataFrame()
        for k in str_to_df_list.keys():
            df_list    = str_to_df_list[k]
            aligned_df = self.__alignDataFrames__(df_list,required_columns)
            pd.set_option('mode.chained_assignment', None)    # verified that the operation occurs correctly 2023-01-11 20:00EST
            aligned_df[my_cat_column] = k
            pd.set_option('mode.chained_assignment', 'warn')
            master_df = pd.concat([master_df, aligned_df])

        # Find the timestamp field... or figure out what to use...
        accepted_args = set(inspect.getfullargspec(getattr(self, sm_type)).args)
        if 'ts_field' in accepted_args:
            if 'ts_field' in sm_params.keys():     # precedence is sm_params ts_field
                ts_field = sm_params['ts_field']
            elif ts_field is None:                 # best guess from the columns // copied from temporalBarChart method
                choices = master_df.select_dtypes(np.datetime64).columns
                if len(choices) == 1:
                    ts_field = choices[0]
                elif len(choices) > 1:
                    print('multiple timestamp fields... choosing the first (createSmallMultiples)')
                    ts_field = choices[0]
                else:
                    raise Exception('no timestamp field supplied to createSmallMultiples(), cannot automatically determine field')            
            else:                                  # use the ts_field passed into this method
                pass

        # Call the original smallMultiples method with the lookup parameter present
        my_category_to_sm = {}
        self.smallMultiples(master_df, 
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
            k_svg = my_category_to_sm[k]
            updated_category_to_sm[k] = self.__overwriteSVGOriginPosition__(k_svg, str_to_xy.get(k), sm_w, sm_h)

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

        self.smallMultiples(**params_copy)                              # perform the actual instance creation -- results in category_to_instance variable

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
    def __overwriteSVGOriginPosition__(self, svg, xy_tuple, sm_w, sm_h):
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
    # __extractSVGWidthAndHeight__()
    # ... extract the width and height of an SVG section
    # ... really fragile... should only be used with SVG generated by this package...
    #
    def __extractSVGWidthAndHeight__(self, svg):
        i0 = svg.index('width="')
        i1 = svg.index('"',i0+len('width="'))
        _width = float(svg[i0+len('width="'):i1])

        i0 = svg.index('height="')
        i1 = svg.index('"',i0+len('height="'))
        _height = float(svg[i0+len('height="'):i1])

        return _width,_height
    
    #
    # Tile a list of SVG's
    #
    def tile(self, svg_list, horz=True):
        if horz:
            w_overall,h_max = 0,0
            for _svg in svg_list:
                w,h = self.__extractSVGWidthAndHeight__(_svg)
                w_overall += w
                if h > h_max:
                    h_max = h

            svg = f'<svg width="{w_overall}" height="{h_max}" x="0" y="0">'
            w_overall = 0
            for _svg in svg_list:
                w,h = self.__extractSVGWidthAndHeight__(_svg)
                svg += self.__overwriteSVGOriginPosition__(_svg, (w_overall + w/2, h/2), w, h)
                w_overall += w
            return svg + '</svg>'
        
        else:
            w_max,h_overall = 0,0
            for _svg in svg_list:
                w,h = self.__extractSVGWidthAndHeight__(_svg)
                h_overall += h
                if w > w_max:
                    w_max = w

            svg = f'<svg width="{w_max}" height="{h_overall}" x="0" y="0">'
            h_overall = 0
            for _svg in svg_list:
                w,h = self.__extractSVGWidthAndHeight__(_svg)
                svg += self.__overwriteSVGOriginPosition__(_svg, (w/2, h_overall + h/2), w, h)
                h_overall += h
            return svg + '</svg>'

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
    
    #
    # Sort by rows
    #
    if   sort_by == 'records' or (sort_by == 'field' and sort_by_field is None):
        #print('by records')
        return df.groupby(field).size().sort_values(ascending=False)
    
    #
    # Sort by a field
    #
    elif sort_by == 'field':        
        if rt_self.fieldIsArithmetic(df,sort_by_field):
            #print('by field (arithmetic)')
            return df.groupby(field)[sort_by_field].sum().sort_values(ascending=False)
        else:
            #print('by field (set operation)')
            _df = pd.DataFrame(df.groupby([field,sort_by_field]).size())
            return _df.groupby(field).size().sort_values(ascending=False)

    #
    # Sort naturally
    #
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

    #
    # Alphabetical
    #
    else:
        #print('by alpha')
        return df.groupby(field).count()

