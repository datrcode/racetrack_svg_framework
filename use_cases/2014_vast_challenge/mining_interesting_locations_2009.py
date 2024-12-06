#
# (Partial) Implementation of the following paper:
#
# "Mining Interesting Locations and Travel Sequences from GPS Trajectories"
# Authors:  Yu Zheng, Lizhu Zhang, Xing Xie, Wei-Ying Ma
# WWW 2009 MADRID: (Track: User Interfaces and Mobile Web / Session: Mobile Web)
#
import polars as pl
import numpy as np
import hdbscan
import rtsvg

__name__ = 'mining_interesting_locations_2009'

class MiningInterestingLocations2009(object):
    #
    # __init__()
    #
    def __init__(self, 
                 df,  
                 id_field, 
                 rt        = None, 
                 lat_field = 'latitude', 
                 lon_field = 'longitude', 
                 ts_field  = 'timestamp',
                 d_thresh  = 150.0/5280.0, # in miles
                 t_thresh  = 20.0*60.0):   # in seconds
        self.rt        = rtsvg.RACETrack() if rt is None else rt
        self.df        = df
        self.id_field  = id_field
        self.lat_field = lat_field
        self.lon_field = lon_field
        self.ts_field  = ts_field
        self.d_thresh  = d_thresh
        self.t_thresh  = t_thresh

        print(f"MiningInterestingLocations2009 | Original Input Size {len(self.df):_}")

        #
        # DataFrame Preparation
        # ... mostly Defintion 2 in the paper
        #
        self.df = self.df.sort([self.id_field,self.ts_field])
        self.df = self.df.with_columns(pl.col(self.ts_field)  .diff() .alias('__t_delta__'),
                                       pl.col(self.lat_field) .shift().alias('__prev_lat__'),
                                       pl.col(self.lon_field) .shift().alias('__prev_lon__'),
                                       pl.col(self.id_field)  .shift().alias('__prev_id__'))
        self.df = self.df.with_columns((self.distanceMiles(pl.col(self.lon_field), pl.col(self.lat_field), 
                                                           pl.col('__prev_lon__'), pl.col('__prev_lat__'))).alias('__miles__'))
        self.df = self.df.with_columns(pl.col('__t_delta__').dt.total_seconds().alias('__t_delta_secs__'))
        self.df = self.df.with_columns(((60.0*60.0)*pl.col('__miles__')/(0.001 + pl.col('__t_delta_secs__'))).alias('__mph__')) # the 0.001 is to avoid divide by zero
        self.df = self.df.with_columns(pl.col('__mph__')         .fill_null(0.0),
                                       pl.col('__prev_id__')     .fill_null( -1),
                                       pl.col('__miles__')       .fill_null(0.0),
                                       pl.col('__t_delta_secs__').fill_null(0.0))
        self.df = self.df.with_columns(pl.when(pl.col(self.id_field) != pl.col('__prev_id__')).then(pl.lit(0.0)).otherwise(pl.col('__t_delta_secs__')).alias('__t_delta_secs__'),
                                       pl.when(pl.col(self.id_field) != pl.col('__prev_id__')).then(pl.lit(0.0)).otherwise(pl.col('__miles__'))       .alias('__miles__'),
                                       pl.when(pl.col(self.id_field) != pl.col('__prev_id__')).then(pl.lit(0.0)).otherwise(pl.col('__mph__'))         .alias('__mph__'))
        self.df = self.df.drop(['__t_delta__','__prev_lat__','__prev_lon__','__prev_id__'])
        if len(self.df.filter(pl.col('__t_delta_secs__') < 0.0)) > 0: print('ERROR: t_delta_secs < 0.0')
        if len(self.df.filter(pl.col('__miles__')        < 0.0)) > 0: print('ERROR: miles < 0.0')
        if len(self.df.filter(pl.col('__mph__')          < 0.0)) > 0: print('ERROR: mph < 0.0')
        self.df = self.df.filter(pl.col('__t_delta_secs__') > 0.0)
        self.df = self.df.with_columns(pl.col('__t_delta_secs__').cum_sum().alias('__t_delta_secs_sum__'))
        print(f"MiningInterestingLocations2009 | Processed DataFrame {len(self.df):_}")

        #
        # Stay Point Detection (Definition 3 in the paper)
        #
        # self.df_stay_points = self.__stayPointDetectionGolden__()
        self.df_stay_points = self.__stayPointDetection__()
        print(f"MiningInterestingLocations2009 | Stay Points {len(self.df_stay_points):_}")

        #
        # User Location History (Defintion 4 in the paper)
        # - the stay point dataframe captures this information
        #

        #
        # Tree-Based Hierarchy (Definition 5 in the paper)
        # - this construct just the bottom level of the tree -- and assigns the cluster labels to the stay points
        #
        self.stay_point_clustering             = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=2).fit(self.df_stay_points[['lon_ave', 'lat_ave']])
        self.df_stay_points                    = self.df_stay_points.with_columns(pl.Series(self.stay_point_clustering.labels_).alias('cluster_label'))
        unclustered_sub_df                     = self.df_stay_points.filter(pl.col('cluster_label') == -1)
        clustered_sub_df                       = self.df_stay_points.filter(pl.col('cluster_label') != -1)
        unclustered_stay_points                = len(unclustered_sub_df)
        _max_label_                            = max(self.stay_point_clustering.labels_)

        # Do a second pass w/ the unclustered stay points -- get a stay point into a cluster... even if it's just one point
        print(f"MiningInterestingLocations2009 | Bottom Layer Clusters {_max_label_:_} | Unclustered Stay Points {unclustered_stay_points:_}")
        self.stay_point_clustering_unclustered = hdbscan.HDBSCAN(min_cluster_size=2, min_samples=2).fit(unclustered_sub_df[['lon_ave', 'lat_ave']])
        unclustered_labels, label_lu = [], {}
        for _label_ in self.stay_point_clustering_unclustered.labels_:
            if _label_ == -1: 
                _max_label_ += 1
                unclustered_labels.append(_max_label_)
            else:
                if _label_ not in label_lu:
                    _max_label_ += 1
                    label_lu[_label_] = _max_label_
                unclustered_labels.append(label_lu[_label_])
        unclustered_sub_df = unclustered_sub_df.with_columns(pl.Series(unclustered_labels).alias('cluster_label'))
        _before_len_, _before_clustered_, _before_unclustered_ = len(self.df_stay_points), len(clustered_sub_df), len(unclustered_sub_df)
        self.df_stay_points = pl.concat([clustered_sub_df, unclustered_sub_df])
        _after_len_ = len(self.df_stay_points)
        if _before_len_ != _after_len_: print(f'ERROR: _before_len_ {_before_len_} != _after_len_ {_after_len_}')
        if _after_len_  != _before_clustered_ + _before_unclustered_: print(f'ERROR: _after_len_ {_after_len_} != _before_clustered_ {_before_clustered_} + _before_unclustered_ {_before_unclustered_}')
        print(f'MiningInterestingLocations2009 | Adjusted Bottom Layer Clusters {max(self.df_stay_points["cluster_label"]):_}')

        #
        # Tree-Based Hierarchical Graph (Definition 6 in the paper)
        # - for just the bottom layer
        #
        self.df_stay_points          = self.df_stay_points.sort(['id', 'ts_arrive'])
        self.df_cluster_locations    = self.df_stay_points.group_by('cluster_label').agg(pl.col('lon_ave').mean(), 
                                                                                         pl.col('lat_ave').mean(),
                                                                                         pl.len().alias('num_points'),
                                                                                         pl.col('id').alias('ids_seen'))
        self.df_cluster_locations_pos = {}
        for i in range(len(self.df_cluster_locations)): self.df_cluster_locations_pos[i] = self.df_cluster_locations['lon_ave'][i], self.df_cluster_locations['lat_ave'][i]
        
        self.df_stay_points_combined = self.rt.polarsGroupOverlappingTimeframes(self.df_stay_points, 'ts_arrive', 'ts_leave', ['id','cluster_label'])
        print(f'MiningInterestingLocations2009 | Stay Points {len(self.df_stay_points):_} | Combined Stay Points {len(self.df_stay_points_combined):_}')
        self.df_stay_points_combined = self.df_stay_points_combined.sort(['id', 'ts_arrive'])
        self.df_hierarchical_graph = self.df_stay_points_combined.sort(['id','ts_arrive']) \
                                                                 .with_columns(pl.col('cluster_label').shift(-1, fill_value=-1).alias('next_cluster_label'),
                                                                               pl.col('id')           .shift(-1, fill_value=-1).alias('next_id')) \
                                                                 .filter(pl.col('id') == pl.col('next_id')) \
                                                                 .drop(['run_nbr', 'next_id']) \
                                                                 .rename({'cluster_label':'fm_location', 'next_cluster_label':'to_location'})


    #
    # __stayPointDetectionGolden__()
    # - This is the basic implementation w/out optimization
    #
    def __stayPointDetectionGolden__(self):
        stay_lu = {'id':[], 'lat_ave':[], 'lon_ave':[], 'dur_secs':[], 'ts_arrive':[], 'ts_leave':[], 'rows':[], 'row_i':[]}
        i = 0
        while i < len(self.df):
            j = i
            while j                <  len(df)     and \
                  self.df[self.id_field][j] == self.df[self.id_field][i] and \
                  self.distanceMiles(self.df[self.lat_field][i], self.df[self.lon_field][i], 
                                     self.df[self.lat_field][j], self.df[self.lon_field][j]) <= self.d_thresh: j += 1
            # Rewind by one
            #if j           == len(df):     j = j - 1
            #if df['id'][i] != df['id'][j]: j = j - 1
            # Does it meet the duration threshold?
            if self.df['__t_delta_secs_sum__'][j-1] - self.df['__t_delta_secs_sum__'][i] >= self.d_thresh:
                stay_lu['id']        .append(self.df[self.id_field][i])
                stay_lu['lat_ave']   .append(self.df[self.lat_field][i:j+1].mean())
                stay_lu['lon_ave']   .append(self.df[self.lon_field][i:j+1].mean())
                stay_lu['dur_secs']  .append(self.df['__t_delta_secs_sum__'][j] - df['__t_delta_secs_sum__'][i])
                stay_lu['ts_arrive'] .append(self.df[self.ts_field][i])
                stay_lu['ts_leave']  .append(self.df[self.ts_field][j])
                stay_lu['rows']      .append(j - i)
                stay_lu['row_i']     .append(i)    
            i += 1
        df_stay_points_gold = pl.DataFrame(stay_lu)
        return df_stay_points_gold

    #
    # __stayPointDetection__()
    # - Slightly optimized to keep track of the minimum timeframe w/out recalculating it every time
    #
    def __stayPointDetection__(self):
        stay_lu = {'id':[], 'lat_ave':[], 'lon_ave':[], 'dur_secs':[], 'ts_arrive':[], 'ts_leave':[], 'rows':[], 'row_i':[]}
        i, j = 0, 0
        while i < len(self.df):
            # Move j to the first point that satisfies the minimum duration threshold
            while j < len(self.df) and \
                  self.df[self.id_field][j] == self.df[self.id_field][i] and \
                  (self.df['__t_delta_secs_sum__'][j] - self.df['__t_delta_secs_sum__'][i]) < self.t_thresh: j += 1
            # Rewind by one
            if j == len(self.df): j = j - 1
            # If we ran out of data for this id, then move to the next id
            if self.df[self.id_field][i] != self.df[self.id_field][j]:
                _id_ = self.df[self.id_field][i] 
                while self.df[self.id_field][i] == _id_: i += 1
                j = i
            else:
                # Does it meet the minimum duration threshold?  And the distance threshold for the last point?
                if (self.df['__t_delta_secs_sum__'][j] - self.df['__t_delta_secs_sum__'][i])    >= self.t_thresh and \
                   self.distanceMiles(self.df[self.lat_field][i], self.df[self.lon_field][i], self.df[self.lat_field][j], self.df[self.lon_field][j]) <= self.d_thresh:
                    # Verify all the points are within the distance threshold
                    k = i + 1
                    while k                         <  len(self.df)              and \
                          self.df[self.id_field][k] == self.df[self.id_field][i] and \
                          self.distanceMiles(self.df[self.lat_field][i], self.df[self.lon_field][i], self.df[self.lat_field][k], self.df[self.lon_field][k]) <= self.d_thresh: k += 1
                    # Rewind by one
                    if k == len(self.df): k = k - 1
                    #if df['id'][i] != df['id'][k]: k = k - 1
                    # Does it meet the duration threshold? (j marks the minimum threshold ... so compare k with j)
                    if k >= j:
                        stay_lu['id']        .append(self.df[self.id_field][i])
                        stay_lu['lat_ave']   .append(self.df[self.lat_field][i:k+1].mean())
                        stay_lu['lon_ave']   .append(self.df[self.lon_field][i:k+1].mean())
                        stay_lu['dur_secs']  .append(self.df['__t_delta_secs_sum__'][k] - self.df['__t_delta_secs_sum__'][i])
                        stay_lu['ts_arrive'] .append(self.df[self.ts_field][i])
                        stay_lu['ts_leave']  .append(self.df[self.ts_field][j])
                        stay_lu['rows']      .append(k - i)
                        stay_lu['row_i']     .append(i)
                i += 1
        df_stay_points = pl.DataFrame(stay_lu)
        return df_stay_points

    #
    # visualizeStayPointClusters()
    #
    def visualizeStayPointClusters(self, w=950, h=500, fix_aspect_ratio=True):
        return self.rt.xy(self.df_stay_points, x_field='lon_ave', y_field='lat_ave', color_by='cluster_label', dot_size='medium', w=w, h=h, fix_aspect_ratio=fix_aspect_ratio)

    #
    # visualizeStayPointClustersAsSmallMultiples()
    #
    def visualizeStayPointClustersAsSmallMultiples(self, w=1280, w_sm_override=300, h_sm_override=200):
        return self.rt.smallMultiples(self.df_stay_points, category_by='cluster_label', sm_type='xy', 
                                      sm_params={'x_field':'lon_ave', 'y_field':'lat_ave', 'dot_size':'medium', 'fix_aspect_ratio':True}, 
                                      w=w, w_sm_override=w_sm_override, h_sm_override=h_sm_override)

    #
    # Source:
    # https://stackoverflow.com/questions/76262681/i-need-to-create-a-column-with-the-distance-between-two-coordinates-in-polars/76265233#76265233
    #
    def distanceMiles(self, s_lat, s_lng, e_lat, e_lng):
        # Approximate radius of earth in miles
        R = 3963.1
        s_lat = s_lat * np.pi/180.0
        s_lng = np.deg2rad(s_lng)
        e_lat = np.deg2rad(e_lat)
        e_lng = np.deg2rad(e_lng)
        d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2
        return 2 * R * np.arcsin(np.sqrt(d))

if __name__ == "__main__":
    rt = rtsvg.RACETrack()
    df = pl.read_csv('../../../data/2014_vast/MC2/gps.csv')
    df = df.filter(pl.col('id') != 28) # this track requires additional processing 
    df = rt.columnsAreTimestamps(df, 'Timestamp')
    mil2009 = MiningInterestingLocations2009(df, 'id', rt=rt, lat_field='lat', lon_field='long', ts_field='Timestamp')
