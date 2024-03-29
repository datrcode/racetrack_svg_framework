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
import numpy as np

__name__ = 'rt_datamanip_mixin'

#
# Data Manipulation Mixin
# ... utilities for preparing a dataframe for the visualization components
#
class RTDataManipMixin(object):
    #
    # temporalStatsAggregation()
    # ... Produces a variety of stats based on specified temporal frequency, a list of fields, and stat names
    # ... Returns as a pandas dataframe with field_stat columns ... index is the temporal aggregation
    #
    # ... and yes, there's probably something that already does this in the pandas library...
    #
    def temporalStatsAggregation(self, df, ts_field=None, freq='YS', fields=[], stats=['sum','max','median','mean','min','stdev','rows','set_size']):
        # Convert parameters to a list if necessary
        if type(fields) != list:
            fields = [fields]
        if type(stats)  != list:
            stats  = [stats]

        # Determine the timestamp field
        if ts_field is None:
            ts_field = self.guessTimestampField(df)

        # Determine if a field is a categorical type (for set-based operations only)
        field_is_set = {}
        for field in fields:
            field_is_set[field] = self.countBySet(df, field)

        # Operations that can only be done if the field is all numbers
        numeric_ops = ['sum','max','median','mean','min','stdev']

        # Initialize the column contain
        _lu = {}
        for field in fields:
            for stat in stats:
                if stat in numeric_ops and field_is_set[field] == False:
                    _lu[field + '_' + stat] = []
                elif stat not in numeric_ops:
                    _lu[field + '_' + stat] = []

        # Produce the stats
        indices = []
        gb = df.groupby(pd.Grouper(key=ts_field, freq=freq))
        for k,k_df in gb:
            indices.append(k)
            for field in fields:
                # ================================================================= #
                if 'sum' in stats and field_is_set[field] == False:
                    _lu[field + '_sum']     .append(k_df[field].sum())
                if 'max' in stats and field_is_set[field] == False:
                    _lu[field + '_max']     .append(k_df[field].max())
                if 'median' in stats and field_is_set[field] == False:
                    _lu[field + '_median']  .append(k_df[field].median())
                if 'mean' in stats and field_is_set[field] == False:
                    _lu[field + '_mean']    .append(k_df[field].mean())
                if 'min' in stats and field_is_set[field] == False:
                    _lu[field + '_min']     .append(k_df[field].min())
                if 'stdev' in stats and field_is_set[field] == False:
                    _lu[field + '_stdev']   .append(k_df[field].std())

                # ================================================================= #
                if 'rows' in stats:
                    _lu[field + '_rows']    .append(len(k_df))
                if 'set_size' in stats:
                    _lu[field + '_set_size'].append(len(set(k_df[field])))

        # Return the dataframe
        _df = pd.DataFrame(_lu, index=indices)
        _df.index.name = ts_field
        return _df

    #
    # temporalStatsAggregationWithGBFields()
    # ... same as above but keeps the gb_fields separable
    #
    def temporalStatsAggregationWithGBFields(self, 
                                             df,                   # Dataframe to aggregate
                                             fields,               # Field or fields to aggregate
                                             ts_field=None,        # timestamp field... if none, method will use first one found
                                             gb_fields=[],         # Fields to keep separable
                                             flatten_index=True,   # Flatten the index before returning the aggregation
                                             fill_missing=False,   # Fill in missing timestamps 
                                             freq='YS',            # Frequency for the aggregation
                                             stats=['sum','max','median','mean','min','stdev','rows','set_size']):
        # Convert parameters to a list if necessary
        if type(fields) != list:
            fields = [fields]
        if type(stats)  != list:
            stats  = [stats]
        if type(gb_fields) != list:
            gb_fields = [gb_fields]

        # Determine the timestamp field
        if ts_field is None:
            ts_field = self.guessTimestampField(df)

        # Determine if a field is a categorical type (for set-based operations only)
        field_is_set = {}
        for field in fields:
            field_is_set[field] = self.countBySet(df, field)

        # Operations that can only be done if the field is all numbers
        numeric_ops = ['sum','max','median','mean','min','stdev']

        # Initialize the column contain
        _lu = {}
        for field in fields:
            for stat in stats:
                if stat in numeric_ops and field_is_set[field] == False:
                    _lu[field + '_' + stat] = []
                elif stat not in numeric_ops:
                    _lu[field + '_' + stat] = []

        # Produce the stats
        indices     = []
        complete_gb,complete_index = [],[]
        complete_gb.    append(pd.Grouper(key=ts_field, freq=freq))
        complete_index. append(ts_field)
        for x in gb_fields:
            complete_gb.   append(x)
            complete_index.append(x)

        earliest_seen,latest_seen = None,None

        tuples_seen = set()
        gb = df.groupby(complete_gb)
        for k,k_df in gb:

            # Keep earliest & latest
            if earliest_seen is None:
                earliest_seen = k[0]
            latest_seen=k[0]

            # Keep track of tuples seen and the separate index
            tuples_seen.add(k)
            indices.append(k)

            # Calculate the stats
            for field in fields:
                # ================================================================= #
                if 'sum' in stats and field_is_set[field] == False:
                    _lu[field + '_sum']     .append(k_df[field].sum())
                if 'max' in stats and field_is_set[field] == False:
                    _lu[field + '_max']     .append(k_df[field].max())
                if 'median' in stats and field_is_set[field] == False:
                    _lu[field + '_median']  .append(k_df[field].median())
                if 'mean' in stats and field_is_set[field] == False:
                    _lu[field + '_mean']    .append(k_df[field].mean())
                if 'min' in stats and field_is_set[field] == False:
                    _lu[field + '_min']     .append(k_df[field].min())
                if 'stdev' in stats and field_is_set[field] == False:
                    _lu[field + '_stdev']   .append(k_df[field].std())

                # ================================================================= #
                if 'rows' in stats:
                    _lu[field + '_rows']    .append(len(k_df))
                if 'set_size' in stats:
                    _lu[field + '_set_size'].append(len(set(k_df[field])))

        # Fill in missing values
        if fill_missing:
            if len(gb_fields) == 1:
                gb_separable = df.groupby(gb_fields[0])
            else:
                gb_separable = df.groupby(gb_fields)
            for k,k_df in gb_separable:
                for _date in pd.date_range(start=earliest_seen, end=latest_seen, freq=freq):
                    _k_as_list = list()
                    _k_as_list.append(_date)
                    if type(k) == tuple:
                        _k_as_list += k
                    else:
                        _k_as_list.append(k)
                    _tuple = tuple(_k_as_list)
                    if _tuple not in tuples_seen:
                        indices.append(_tuple)
                        for x in _lu.keys():
                            _lu[x].append(0)

        # Create the dataframe
        _df = pd.DataFrame(_lu, index=indices)

        # Flatten the index if requested
        if flatten_index:
            _df = _df.reset_index()
            _df = _df.join(pd.DataFrame(_df['index'].values.tolist(), columns=complete_index))\
                     .drop('index',axis=1)

        return _df

    #
    # rowContainsSubstring()
    # - use it as follows:  df[df.apply(lambda x: rt.rowContainsSubstring(x, 'sub'),axis=1)]
    #
    def rowContainsSubstring(self,
                             _row,
                             _substring,
                             _match_case=False):
        if _match_case == False:
            _substring = _substring.lower()
        for x in _row:
            if _match_case == False:
                x = str(x).lower()
            if _substring in x:
                return True
        return False
    