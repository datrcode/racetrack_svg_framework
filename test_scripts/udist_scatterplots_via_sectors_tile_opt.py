#
# Polars implementation of the following:
#
# H. Rave, V. Molchanov and L. Linsen, "Uniform Sample Distribution in Scatterplots via Sector-based Transformation," 
# 2024 IEEE Visualization and Visual Analytics (VIS), St. Pete Beach, FL, USA, 2024, pp. 156-160, 
# doi: 10.1109/VIS55277.2024.00039. 
# keywords: {Data analysis;Visual analytics;Clutter;Scatterplot de-cluttering;spatial transformation},
#
# This version includes the tile optimization...
#
import polars  as     pl
import numpy   as     np
from   math    import pi, sin, cos, atan2, sqrt
from   shapely import Polygon
from   os.path import exists
import time

__name__ = 'udist_scatterplots_via_sectors_tile_opt'

class UDistScatterPlotsViaSectorsTileOpt(object):
    def __init__(self, x_vals=[], y_vals=[], weights=None, colors=None, vector_scalar=0.01, iterations=4, debug=False, num_of_tiles=32):
        self.vector_scalar = vector_scalar
        self.iterations    = iterations
        self.num_of_tiles  = num_of_tiles
        self.debug         = debug
        self.time_lu       = {'prepare_df':0.0, 'normalize':0.0, 'all_sectors':0.0, 'explode_points':0.0, 'arctangents':0.0, 'sector_sums':0.0, 
                              'add_missing_sectors':0.0, 'prepare_sector_angles':0.0, 'join_sector_angles':0.0, 'ray_segment_intersections':0.0,
                              'area_calc':0.0, 'sector_uv_summation':0.0, 'point_update':0.0, 'determine_tile':0.0, 'tile_sums':0.0,
                              'cross_join_tile_offsets':0.0, 'join_sector_info':0.0, 'hard_way_arctangents':0.0,
                              'medium_way_crossproducts':0.0,'sector_sums':0.0, 'separate_easy_hard_way (easy way)':0.0,
                              'separate_easy_hard_way (non-easy way)':0.0, 'separate_easy_hard_way (medium way)':0.0,
                              'separate_easy_hard_way (hard way)':0.0,}

        # Create the debugging structures
        self.df_weight_sums              = []
        self.df_at_iteration_start       = []
        self.df_all_sectors              = []
        self.df_tile_determinations      = []
        self.df_tile_sums                = []
        self.df_cross_join_tile_offsets  = []
        self.df_join_sector_info         = []
        self.df_separate_easy_way        = []
        self.df_separate_medium_way      = []
        self.df_separate_hard_way        = []
        self.df_medium_way_crossproducts = []
        self.df_hard_way_arctangents     = []
        self.df_sector_sums              = []
        self.df_sector_fill              = []
        self.df_sector_angles            = []
        self.df_sector_angles_joined     = []
        self.df_fully_filled             = []
        self.df_uv                       = []

        # Create weights if none were set
        if weights is None: weights = np.ones(len(x_vals))

        def rayIntersectsSegment(xy_ray, uv_ray, xy0_segment, xy1_segment, include_xy1_endpoint=False, epsilon=1e-9):
            x_r,  y_r  = xy_ray
            dx_r, dy_r = uv_ray
            x0,   y0   = xy0_segment
            x1,   y1   = xy1_segment    
            # Segment direction vector
            dx_s, dy_s = x1 - x0, y1 - y0
            # Compute determinant
            det = -dx_r * dy_s + dy_r * dx_s
            if abs(det) < 1e-10: return None # Lines are parallel or collinear
            # Compute parameters t and u
            t = ((x_r - x0) * dy_s - (y_r - y0) * dx_s) / det
            u = ((x_r - x0) * dy_r - (y_r - y0) * dx_r) / det
            # Check if intersection is valid (t >= 0 for ray, 0 <= u <= 1 for segment)
            if t >= 0.0:
                if include_xy1_endpoint:
                    if 0.0 <= u <= 1.0+epsilon: return (x_r + t * dx_r, y_r + t * dy_r)
                else:
                    if 0.0 <= u <  1.0-epsilon: return (x_r + t * dx_r, y_r + t * dy_r)
            return None

        #
        # Prepare the xo/yo dataframe
        #
        t = time.time()
        self.xoyo_filename = 'udist_scatterplots_via_sectors_tile_opt.parquet'
        if exists(self.xoyo_filename) == False: self.createXoYoDataframeFile()
        self.df_xoyo_sector = pl.read_parquet(self.xoyo_filename).filter(pl.col('num_of_tiles') == num_of_tiles).drop('num_of_tiles').unique()
        if len(self.df_xoyo_sector) == 0: raise Exception('No xo/yo sector data found for num_of_tiles = %d' % num_of_tiles)
        self.time_lu['xoyo_sector_creation'] = time.time() - t

        #
        # Prepare the initial dataframe
        #
        t = time.time()
        if colors is None: df = pl.DataFrame({'x':x_vals, 'y':y_vals, 'w':weights})            .with_row_index('__index__').with_columns(pl.lit('#000000').alias('c'))
        else:              df = pl.DataFrame({'x':x_vals, 'y':y_vals, 'w':weights, 'c':colors}).with_row_index('__index__')
        df_orig = df.clone()
        self.time_lu['prepare_df'] += (time.time() - t)

        #
        # Perform each iteration
        #
        for _iteration_ in range(iterations):
            # Determine the overall weight sums
            df_weight_sum = df['w'].sum()
            if debug: self.df_weight_sums.append(df_weight_sum)

            #
            # Normalize the points to 0.02 to 0.98 (want to give it a little space around the edges to that there are sectors to move into)
            #
            t = time.time()
            if set(['_u_', '_v_', 'x_right', 'y_right']) & set(df.columns) == set(['_u_', '_v_', 'x_right', 'y_right']): df = df.drop(['_u_', '_v_', 'x_right', 'y_right'])
            df = df.with_columns((0.02 + 0.96 * (pl.col('x') - pl.col('x').min())/(pl.col('x').max() - pl.col('x').min())).alias('x'), 
                                 (0.02 + 0.96 * (pl.col('y') - pl.col('y').min())/(pl.col('y').max() - pl.col('y').min())).alias('y'))
            if debug: self.df_at_iteration_start.append(df.clone())
            self.time_lu['normalize'] += (time.time() - t)

            #
            # All Sectors DataFrame
            #
            t = time.time()
            df_all_sectors = df.join(pl.DataFrame({'sector': [i for i in range(16)]}), how='cross').drop(['w','c'])
            if debug: self.df_all_sectors.append(df_all_sectors.clone())
            self.time_lu['all_sectors'] += (time.time() - t)
            
            #
            # vvv -- New Performant Version -- vvv
            #

            # Determine the x/y tile of each point (xi,yi)
            t = time.time()
            df_w_tile = df.with_columns((pl.col('x') * num_of_tiles).cast(pl.Int16).alias('xi'), 
                                        (pl.col('y') * num_of_tiles).cast(pl.Int16).alias('yi'))
            if debug: self.df_tile_determinations.append(df_w_tile.clone())
            self.time_lu['determine_tile'] += (time.time() - t)

            # Determine the sum of the weights in each tile (tile_sum)
            t = time.time()
            df_tile_sums = df_w_tile.group_by(['xi','yi']).agg(pl.col('w').sum().alias('tile_sum'))
            if debug: self.df_tile_sums.append(df_tile_sums.clone())
            self.time_lu['tile_sums'] += (time.time() - t)

            # Cross join all filled in tiles with the original points & then compute the x/y offset (xo,yo)
            t = time.time()
            df           = df_w_tile.join(df_tile_sums, how='cross').rename({'xi_right':'xi_tile_sums', 'yi_right':'yi_tile_sums'}). \
                                                                     with_columns((pl.col('xi_tile_sums') - pl.col('xi')).alias('xo'),
                                                                                  (pl.col('yi_tile_sums') - pl.col('yi')).alias('yo'))
            if debug: self.df_cross_join_tile_offsets.append(df.clone())
            self.time_lu['cross_join_tile_offsets'] += (time.time() - t)

            # Pull the sector information from the df_xoyo_sector dataframe ... any sector == -1 needs to be done the hard way
            # ... hard way == atan2 on the individual point level
            t = time.time()
            df          = df.join(self.df_xoyo_sector, on=['xo','yo'])
            if debug: self.df_join_sector_info.append(df.clone())
            self.time_lu['join_sector_info'] += (time.time() - t)

            # Separate into "easy way" and "hard way"
            t = time.time()
            df_easy_way   = df.filter(pl.col('sector') != -1)
            self.time_lu['separate_easy_hard_way (easy way)'] += (time.time() - t)

            t = time.time()
            df_hard_way   = df.filter(pl.col('sector') == -1) \
                              .join(df_w_tile.drop(['c']), left_on=['xi_tile_sums','yi_tile_sums'], right_on=['xi','yi']) \
                              .filter(pl.col('__index__') != pl.col('__index___right'))
            self.time_lu['separate_easy_hard_way (non-easy way)'] += (time.time() - t)
            
            t = time.time()
            df_medium_way = df_hard_way.filter(pl.col('u').is_not_null())
            self.time_lu['separate_easy_hard_way (medium way)'] += (time.time() - t)

            t = time.time()
            df_hard_way   = df_hard_way.filter(pl.col('u').is_null())
            self.time_lu['separate_easy_hard_way (hard way)'] += (time.time() - t)

            if debug: self.df_separate_easy_way  .append(df_easy_way.clone())
            if debug: self.df_separate_medium_way.append(df_medium_way.clone())
            if debug: self.df_separate_hard_way  .append(df_hard_way.clone())

            # Medium way calculation ... determine the sector based on crossproduct
            t = time.time()
            _x1_, _y1_ = pl.col('x') + pl.col('u'), pl.col('y') + pl.col('v')
            df_medium_way = df_medium_way.with_columns(pl.when((pl.col('x') - pl.col('x_right')) * (_y1_ - pl.col('y_right')) - 
                                                               (pl.col('y') - pl.col('y_right')) * (_x1_ - pl.col('x_right')) > 0.0)
                                                         .then     (pl.col('lsector'))
                                                         .otherwise(pl.col('rsector'))
                                                         .alias    ('sector'))
            if debug: self.df_medium_way_crossproducts.append(df_medium_way.clone())
            self.time_lu['medium_way_crossproducts'] += (time.time() - t)

            # Hard way calculation ... determine the sector on a per point basis
            t = time.time()
            _dx_, _dy_ = pl.col('x_right') - pl.col('x'), pl.col('y_right') - pl.col('y')
            df_hard_way = df_hard_way.with_columns(((16*(pl.arctan2(_dy_, _dx_) + pl.lit(pi))/(pl.lit(2*pi))).cast(pl.Int64)).alias('sector'))
            if debug: self.df_hard_way_arctangents.append(df_hard_way.clone())
            self.time_lu['hard_way_arctangents'] += (time.time() - t)

            # Sector Summation
            t = time.time()
            df_easy_way   = df_easy_way  .drop(set(df_easy_way  .columns) - set(['__index__','sector','tile_sum'])).rename({'tile_sum':'_w_sum_'})
            df_medium_way = df_medium_way.drop(set(df_medium_way.columns) - set(['__index__','sector','w'       ])).rename({'w':       '_w_sum_'})
            df_hard_way   = df_hard_way  .drop(set(df_hard_way  .columns) - set(['__index__','sector','w_right' ])).rename({'w_right': '_w_sum_'})
            df_hard_way = pl.DataFrame({'__index__': df_hard_way['__index__'], '_w_sum_': df_hard_way['_w_sum_'], 'sector': df_hard_way['sector']}) # Re-align the columns
            df            = pl.concat([df_easy_way, df_medium_way, df_hard_way])
            df            = df.group_by(['__index__','sector']).agg(pl.col('_w_sum_').sum()).with_columns((pl.col('_w_sum_') / df_weight_sum).alias('_w_ratio_'))
            if debug: self.df_sector_sums.append(df.clone())
            self.time_lu['sector_sums'] += (time.time() - t)

            #
            # ^^^ -- New Performant Version -- ^^^
            #

            #
            # Add the missing sectors back in...
            #
            t = time.time()
            df = df_all_sectors.join(df, on=['__index__','sector'], how='left').with_columns(pl.col('_w_sum_').fill_null(0), pl.col('_w_ratio_').fill_null(0))
            if debug: self.df_sector_fill.append(df.clone())
            self.time_lu['add_missing_sectors'] += (time.time() - t)

            #
            # Create the sector angle dataframe
            # ... this is a small dataframe that covers just 16 sectors ...
            # ... it will be joined with the points dataframe to calculate the area of each sector for each point
            #
            t = time.time()
            _lu_ = {'sector':[], 
                    'a0':[],       'a0u':[],       'a0v':[],                 # Ray 0 angle & uv components
                    'a1':[],       'a1u':[],       'a1v':[],                 # Ray 1 angle & uv components
                    'corner_x':[], 'corner_y':[],                            # Corner between segment0 and segment 1
                    'anchor_a':[], 'anchor_u':[],  'anchor_v':[],            # Anchor angle & uv components
                    's0x0':[],     's0x1':[],      's0y0':[],     's0y1':[], # Segment 0
                    's1x0':[],     's1x1':[],      's1y0':[],     's1y1':[]} # Segment 1
            for _sector_ in range(16):
                #_lu_['sector'].append(_sector_)
                _sector_align_ = (_sector_ + 8)%16
                _lu_['sector'].append(_sector_align_)
                _a0_ = _sector_*pi/8.0
                _lu_['a0'].append(_a0_), _lu_['a0u'].append(cos(_a0_)), _lu_['a0v'].append(sin(_a0_))
                _a1_ = (_sector_+1)*pi/8.0
                _lu_['a1'].append(_a1_), _lu_['a1u'].append(cos(_a1_)), _lu_['a1v'].append(sin(_a1_))
                #_anchor_ = _a0_ + pi / 2.0 + pi / 16.0 # half angle on top of that
                _anchor_ = _a0_ + pi + pi / 16.0 # half angle on top of that
                _lu_['anchor_a'].append(_anchor_), _lu_['anchor_u'].append(cos(_anchor_)), _lu_['anchor_v'].append(sin(_anchor_))
                if   _sector_ >= 0 and _sector_ <  4:
                    _lu_['s0x0']    .append(1.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(1.0) # segment 0 (x0,y0) -> (x1,y1) (1,0) -> (1,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(1.0), _lu_['s1y1'].append(1.0) # segment 1 (x0,y0) -> (x1,y1) (0,1) -> (1,1)
                    _lu_['corner_x'].append(1.0), _lu_['corner_y'].append(1.0)
                elif _sector_ >= 4 and _sector_ <  8:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(1.0), _lu_['s0y1'].append(1.0) # (0,1) -> (1,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(0.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(1.0) # (0,0) -> (0,1)
                    _lu_['corner_x'].append(0.0), _lu_['corner_y'].append(1.0)
                elif _sector_ >= 8 and _sector_ < 12:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(0.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(1.0) # (0,0) -> (0,1)
                    _lu_['s1x0']    .append(0.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(0.0) # (0,0) -> (1,0)
                    _lu_['corner_x'].append(0.0), _lu_['corner_y'].append(0.0)
                else:
                    _lu_['s0x0']    .append(0.0), _lu_['s0x1']    .append(1.0), _lu_['s0y0'].append(0.0), _lu_['s0y1'].append(0.0) # (0,0) -> (1,0)
                    _lu_['s1x0']    .append(1.0), _lu_['s1x1']    .append(1.0), _lu_['s1y0'].append(0.0), _lu_['s1y1'].append(1.0) # (1,0) -> (1,1)
                    _lu_['corner_x'].append(1.0), _lu_['corner_y'].append(0.0)
            df_sector_angles = pl.DataFrame(_lu_)
            df_sector_angles = df_sector_angles.with_columns((pl.col('s0x1') - pl.col('s0x0')).alias('s0u'), (pl.col('s0y1') - pl.col('s0y0')).alias('s0v'),
                                                             (pl.col('s1x1') - pl.col('s1x0')).alias('s1u'), (pl.col('s1y1') - pl.col('s1y0')).alias('s1v'))
            if debug: self.df_sector_angles.append(df_sector_angles)
            self.time_lu['prepare_sector_angles'] += (time.time() - t)

            # Join w/ sector information
            t  = time.time()
            df = df.join(df_sector_angles, on='sector', how='left')
            if debug: self.df_sector_angles_joined.append(df)
            self.time_lu['join_sector_angles'] += (time.time() - t)

            # Create rays for each sector angles
            t  = time.time()
            df = df.with_columns((pl.col('a0').cos()).alias('xa0'), (pl.col('a0').sin()).alias('ya0'), # uv for angle 0
                                 (pl.col('a1').cos()).alias('xa1'), (pl.col('a1').sin()).alias('ya1')) # uv for angle 1

            # Intersect each ray with each segment (uses the multistep version of ray-segment intersection)
            # ... determinate "r0s0_det" (ray 0, segment 0) .... done four ways (ray 0 & 1 ... segment 0 & 1)
            df = df.with_columns((-pl.col('a0u') * pl.col('s0v') + pl.col('a0v') * pl.col('s0u')).alias('r0s0_det'),
                                 (-pl.col('a0u') * pl.col('s1v') + pl.col('a0v') * pl.col('s1u')).alias('r0s1_det'),
                                 (-pl.col('a1u') * pl.col('s0v') + pl.col('a1v') * pl.col('s0u')).alias('r1s0_det'),
                                 (-pl.col('a1u') * pl.col('s1v') + pl.col('a1v') * pl.col('s1u')).alias('r1s1_det'))
            # ... "t" ("r0s0_t") and "u" values ("r0s0_u") (ray 0, segment 0) for all four ways (ray 0 & 1 ... segment 0 & 1)
            df = df.with_columns((((pl.col('x') - pl.col('s0x0')) * pl.col('s0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('s0u')) / pl.col('r0s0_det')).alias('r0s0_t'),
                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('a0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('a0u')) / pl.col('r0s0_det')).alias('r0s0_u'),

                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('s1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('s1u')) / pl.col('r0s1_det')).alias('r0s1_t'),
                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('a0v') - (pl.col('y') - pl.col('s1y0')) * pl.col('a0u')) / pl.col('r0s1_det')).alias('r0s1_u'),

                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('s0v') - (pl.col('y') - pl.col('s0y0')) * pl.col('s0u')) / pl.col('r1s0_det')).alias('r1s0_t'),
                                 (((pl.col('x') - pl.col('s0x0')) * pl.col('a1v') - (pl.col('y') - pl.col('s0y0')) * pl.col('a1u')) / pl.col('r1s0_det')).alias('r1s0_u'),

                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('s1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('s1u')) / pl.col('r1s1_det')).alias('r1s1_t'),
                                 (((pl.col('x') - pl.col('s1x0')) * pl.col('a1v') - (pl.col('y') - pl.col('s1y0')) * pl.col('a1u')) / pl.col('r1s1_det')).alias('r1s1_u'),)
            # ... the x and y intersects (r0s0_xi, r0s0_yi) (ray 0, segment 0) for all four ways (ray 0 & 1) and segment (0 & 1)
            df = df.with_columns(pl.when((pl.col('r0s0_t') >= 0.0) & (pl.col('r0s0_u') >= 0.0) & (pl.col('r0s0_u') <= 1.0)).then(pl.col('x') + pl.col('r0s0_t') * pl.col('a0u')).otherwise(None).alias('r0s0_xi'),
                                 pl.when((pl.col('r0s0_t') >= 0.0) & (pl.col('r0s0_u') >= 0.0) & (pl.col('r0s0_u') <= 1.0)).then(pl.col('y') + pl.col('r0s0_t') * pl.col('a0v')).otherwise(None).alias('r0s0_yi'),

                                 pl.when((pl.col('r0s1_t') >= 0.0) & (pl.col('r0s1_u') >= 0.0) & (pl.col('r0s1_u') <= 1.0)).then(pl.col('x') + pl.col('r0s1_t') * pl.col('a0u')).otherwise(None).alias('r0s1_xi'),
                                 pl.when((pl.col('r0s1_t') >= 0.0) & (pl.col('r0s1_u') >= 0.0) & (pl.col('r0s1_u') <= 1.0)).then(pl.col('y') + pl.col('r0s1_t') * pl.col('a0v')).otherwise(None).alias('r0s1_yi'),

                                 pl.when((pl.col('r1s0_t') >= 0.0) & (pl.col('r1s0_u') >= 0.0) & (pl.col('r1s0_u') <= 1.0)).then(pl.col('x') + pl.col('r1s0_t') * pl.col('a1u')).otherwise(None).alias('r1s0_xi'),
                                 pl.when((pl.col('r1s0_t') >= 0.0) & (pl.col('r1s0_u') >= 0.0) & (pl.col('r1s0_u') <= 1.0)).then(pl.col('y') + pl.col('r1s0_t') * pl.col('a1v')).otherwise(None).alias('r1s0_yi'),

                                 pl.when((pl.col('r1s1_t') >= 0.0) & (pl.col('r1s1_u') >= 0.0) & (pl.col('r1s1_u') <= 1.0)).then(pl.col('x') + pl.col('r1s1_t') * pl.col('a1u')).otherwise(None).alias('r1s1_xi'),
                                 pl.when((pl.col('r1s1_t') >= 0.0) & (pl.col('r1s1_u') >= 0.0) & (pl.col('r1s1_u') <= 1.0)).then(pl.col('y') + pl.col('r1s1_t') * pl.col('a1v')).otherwise(None).alias('r1s1_yi'),)
            self.time_lu['ray_segment_intersections'] += (time.time() - t)

            #
            # Area Calculation using Shoelace Formula
            #
            # Case 0 ... which is X_X_ ... which is the first and second ray both hit the first segment
            t = time.time()
            _c0_0p_x_, _c0_0p_y_, _c0_0q_x_, _c0_0q_y_ = pl.col('r0s0_xi'), pl.col('r0s0_yi'), pl.col('r1s0_xi'), pl.col('r1s0_yi')
            _c0_1p_x_, _c0_1p_y_, _c0_1q_x_, _c0_1q_y_ = pl.col('r1s0_xi'), pl.col('r1s0_yi'), pl.col('x'),       pl.col('y')
            _c0_2p_x_, _c0_2p_y_, _c0_2q_x_, _c0_2q_y_ = pl.col('x'),       pl.col('y'),       pl.col('r0s0_xi'), pl.col('r0s0_yi')
            _c0_op_ = (((_c0_0p_x_*_c0_0q_y_ - _c0_0q_x_*_c0_0p_y_) + (_c0_1p_x_*_c0_1q_y_ - _c0_1q_x_*_c0_1p_y_) + (_c0_2p_x_*_c0_2q_y_ - _c0_2q_x_*_c0_2p_y_))/2.0).abs().alias('area')

            # Case 1 ... which is _X_X ... which is the first and second ray both hit the second segment
            _c1_0p_x_, _c1_0p_y_, _c1_0q_x_, _c1_0q_y_ = pl.col('r0s1_xi'), pl.col('r0s1_yi'), pl.col('r1s1_xi'), pl.col('r1s1_yi')
            _c1_1p_x_, _c1_1p_y_, _c1_1q_x_, _c1_1q_y_ = pl.col('r1s1_xi'), pl.col('r1s1_yi'), pl.col('x'),       pl.col('y')
            _c1_2p_x_, _c1_2p_y_, _c1_2q_x_, _c1_2q_y_ = pl.col('x'),       pl.col('y'),       pl.col('r0s1_xi'), pl.col('r0s1_yi')
            _c1_op_ = (((_c1_0p_x_*_c1_0q_y_ - _c1_0q_x_*_c1_0p_y_) + (_c1_1p_x_*_c1_1q_y_ - _c1_1q_x_*_c1_1p_y_) + (_c1_2p_x_*_c1_2q_y_ - _c1_2q_x_*_c1_2p_y_))/2.0).abs().alias('area')

            # Case 2 ... which is X__X ... which is the first and second ray both hit different segments... so needs the corner position
            _c2_0p_x_, _c2_0p_y_, _c2_0q_x_, _c2_0q_y_ = pl.col('r0s0_xi'),  pl.col('r0s0_yi'),  pl.col('corner_x'), pl.col('corner_y')
            _c2_1p_x_, _c2_1p_y_, _c2_1q_x_, _c2_1q_y_ = pl.col('corner_x'), pl.col('corner_y'), pl.col('r1s1_xi'),  pl.col('r1s1_yi')
            _c2_2p_x_, _c2_2p_y_, _c2_2q_x_, _c2_2q_y_ = pl.col('r1s1_xi'), pl.col('r1s1_yi'),   pl.col('x'),       pl.col('y')
            _c2_3p_x_, _c2_3p_y_, _c2_3q_x_, _c2_3q_y_ = pl.col('x'),       pl.col('y'),         pl.col('r0s0_xi'), pl.col('r0s0_yi')
            _c2_op_ = (((_c2_0p_x_*_c2_0q_y_ - _c2_0q_x_*_c2_0p_y_) + 
                        (_c2_1p_x_*_c2_1q_y_ - _c2_1q_x_*_c2_1p_y_) + 
                        (_c2_2p_x_*_c2_2q_y_ - _c2_2q_x_*_c2_2p_y_) +
                        (_c2_3p_x_*_c2_3q_y_ - _c2_3q_x_*_c2_3p_y_))/2.0).abs().alias('area')
            df = df.with_columns(pl.when(pl.col('r0s0_xi').is_not_null() & pl.col('r1s0_xi').is_not_null()).then(_c0_op_)
                                   .when(pl.col('r0s1_xi').is_not_null() & pl.col('r1s1_xi').is_not_null()).then(_c1_op_)
                                   .when(pl.col('r0s0_xi').is_not_null() & pl.col('r1s1_xi').is_not_null()).then(_c2_op_)
                                   .otherwise(pl.lit(None).alias('area')))
            if debug: self.df_fully_filled.append(df)
            self.time_lu['area_calc'] += (time.time() - t)

            #
            # With the sector sums, adjust the point based on the ratio of the sector area / sector density...
            # ... results of this iteration will be stored in the _xnext_ and _ynext_ fields of the dataframe
            # ... really important:  empty sectors need to be included in the calculation...
            #
            t = time.time()
            _diff_op_ = (pl.col('_w_ratio_') - pl.col('area')) # diff = sector_sum/weight_sum - sector_area/area_total # area_total == 1.0 since that was the normalization
            df_uv     = df.group_by(['__index__','x','y']).agg( (vector_scalar * _diff_op_ * pl.col('anchor_u')).sum().alias('_u_'),
                                                                (vector_scalar * _diff_op_ * pl.col('anchor_v')).sum().alias('_v_'))
            if debug: self.df_uv.append(df_uv.clone())
            self.time_lu['sector_uv_summation'] += (time.time() - t)

            t = time.time()
            df_uv     = df_uv.with_columns((pl.col('x') + pl.col('_u_')).alias('x'), 
                                           (pl.col('y') + pl.col('_v_')).alias('y'))
            df        = df_uv.join(df_orig, on=['__index__'], how='left') # add the weight back in
            self.time_lu['point_update'] += (time.time() - t)

        self.df_results = df

    #
    # svgOfPoint() // debugging
    #
    def svgOfPoint(self, rt, point_i=0, iteration=0):
        if self.debug == False: raise Exception('This function is for debugging only')
        svg  = [f'<svg x="0" y="0" width="768" height="768" viewBox="0 0 1 1" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1" height="1" x="0" y="0" fill="#ffffff" />')
        # Draw the sectors
        df_points_with_sectors = self.df_fully_filled[iteration]
        _df_ = df_points_with_sectors.filter(pl.col('__index__') == point_i)
        x, y = _df_['x'][0], _df_['y'][0]
        for i in range(len(_df_)):
            r0s0_xi,  r0s0_yi  = _df_['r0s0_xi'][i],  _df_['r0s0_yi'][i]
            r0s1_xi,  r0s1_yi  = _df_['r0s1_xi'][i],  _df_['r0s1_yi'][i]
            r1s0_xi,  r1s0_yi  = _df_['r1s0_xi'][i],  _df_['r1s0_yi'][i]
            r1s1_xi,  r1s1_yi  = _df_['r1s1_xi'][i],  _df_['r1s1_yi'][i]
            corner_x, corner_y = _df_['corner_x'][i], _df_['corner_y'][i]
            d = None
            if   r0s0_xi is not None and r1s0_xi is not None: d = f"M {x} {y} L {r0s0_xi} {r0s0_yi} L {r1s0_xi} {r1s0_yi} Z"
            elif r0s1_xi is not None and r1s1_xi is not None: d = f"M {x} {y} L {r0s1_xi} {r0s1_yi} L {r1s1_xi} {r1s1_yi} Z"
            else:                                             d = f"M {x} {y} L {r0s0_xi} {r0s0_yi} L {corner_x} {corner_y} L {r1s1_xi} {r1s1_yi} Z"
            if d is not None: svg.append(f'<path d="{d}" stroke="{rt.co_mgr.getColor(_df_['sector'][i])}" fill="{rt.co_mgr.getColor(_df_['sector'][i])}" fill-opacity="0.3" stroke-width="0.002" />')
            anchor_u, anchor_v = _df_['anchor_u'][i], _df_['anchor_v'][i]
            svg.append(f'<line x1="{x}" y1="{y}" x2="{x + anchor_u*0.1}" y2="{y + anchor_v*0.1}" stroke="{rt.co_mgr.getColor(_df_['sector'][i])}" stroke-width="0.002" />')

        # Draw the points
        _df_ = self.df_sector_determinations[iteration].filter(pl.col('__index__') == point_i)
        for i in range(len(_df_)): svg.append(f'<circle cx="{_df_['_xo_'][i]}" cy="{_df_['_yo_'][i]}" r="{0.002}" fill="{rt.co_mgr.getColor(_df_['sector'][i])}" />')

        svg.append('</svg>')
        return ''.join(svg)

    #
    # renderSector() // debugging
    #
    def renderSector(self, rt, point_i=0, sector=0, iteration=0):
        if self.debug == False: raise Exception('This function is for debugging only')
        svg  = [f'<svg x="0" y="0" width="256" height="256" viewBox="-0.01 -0.01 1.02 1.02" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1" height="1" x="0" y="0" fill="#ffffff" />')
        # Draw the sectors
        _df_ = self.df_fully_filled[iteration].filter(pl.col('__index__') == point_i)
        x, y = _df_['x'][0], _df_['y'][0]
        for i in range(len(_df_)):
            if _df_['sector'][i] != sector: continue
            r0s0_xi,  r0s0_yi  = _df_['r0s0_xi'][i],  _df_['r0s0_yi'][i]
            r0s1_xi,  r0s1_yi  = _df_['r0s1_xi'][i],  _df_['r0s1_yi'][i]
            r1s0_xi,  r1s0_yi  = _df_['r1s0_xi'][i],  _df_['r1s0_yi'][i]
            r1s1_xi,  r1s1_yi  = _df_['r1s1_xi'][i],  _df_['r1s1_yi'][i]
            corner_x, corner_y = _df_['corner_x'][i], _df_['corner_y'][i]
            d = None
            if   r0s0_xi is not None and r1s0_xi is not None: d = f"M {x} {y} L {r0s0_xi} {r0s0_yi} L {r1s0_xi} {r1s0_yi} Z"
            elif r0s1_xi is not None and r1s1_xi is not None: d = f"M {x} {y} L {r0s1_xi} {r0s1_yi} L {r1s1_xi} {r1s1_yi} Z"
            else:                                             d = f"M {x} {y} L {r0s0_xi} {r0s0_yi} L {corner_x} {corner_y} L {r1s1_xi} {r1s1_yi} Z"
            if d is not None: svg.append(f'<path d="{d}" stroke="{rt.co_mgr.getColor(_df_['sector'][i])}" fill="{rt.co_mgr.getColor(_df_['sector'][i])}" fill-opacity="0.3" stroke-width="0.002" />')
            anchor_u, anchor_v = _df_['anchor_u'][i], _df_['anchor_v'][i]
            svg.append(f'<line x1="{x}" y1="{y}" x2="{x + anchor_u*0.1}" y2="{y + anchor_v*0.1}" stroke="{rt.co_mgr.getColor(_df_['sector'][i])}" stroke-width="0.002" />')
        svg.append('</svg>')
        return ''.join(svg)

    #
    # _repr_svg_() -- simple SVG representation of the results
    #
    def _repr_svg_(self):
        x0, y0, x1, y1 = self.df_results['x'].min(), self.df_results['y'].min(), self.df_results['x'].max(), self.df_results['y'].max()
        xperc, yperc   = (x1-x0)*0.01, (y1-y0)*0.01
        x0, y0, x1, y1 = x0-xperc, y0-yperc, x1+xperc, y1+yperc
        svg = []
        svg.append(f'<svg x="0" y="0" width="256" height="256" viewBox="{x0} {y0} {x1-x0} {y1-y0}" xmlns="http://www.w3.org/2000/svg">')
        svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" x="0" y="0" fill="#ffffff" />')
        if 'c' in self.df_results.columns:
            for i in range(len(self.df_results)): svg.append(f'<circle cx="{self.df_results["x"][i]}" cy="{self.df_results["y"][i]}" r="{0.005}" fill="{self.df_results["c"][i]}" />')
        else:
            for i in range(len(self.df_results)): svg.append(f'<circle cx="{self.df_results["x"][i]}" cy="{self.df_results["y"][i]}" r="{0.005}" fill="#404040" />')
        svg.append('</svg>')
        return ''.join(svg)

    #
    # animateIterations()
    #
    def animateIterations(self, r=0.004, w=512, h=512, animation_dur="10s"):
        if self.debug == False: raise Exception('This function is for debugging only')
        x0, y0, x1, y1 = -0.01, -0.01, 1.02, 1.02
        svg = [f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="{x0} {y0} {x1-x0} {y1-y0}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" x="0" y="0" fill="#ffffff" />')
        _df_ = self.df_uv[0].drop(['_u_','_v_'])
        x_ops, y_ops = [pl.col('x_0')], [pl.col('y_0')]
        for i in range(1, len(self.df_uv)): 
            _df_ = _df_.join(self.df_uv[i].drop(['_u_', '_v_']), on='__index__', suffix=f'_{i}')
            x_ops.append(pl.col(f'x_{i}')), y_ops.append(pl.col(f'y_{i}'))
        x_ops.extend(x_ops[::-1]), y_ops.extend(y_ops[::-1])
        x_ops.extend(['x_0', 'x_0', 'x_0']), y_ops.extend(['y_0', 'y_0', 'y_0']) # so there's a slight delay before it starts all over again
        _df_  = _df_.rename({'x':'x_0','y':'y_0'})
        _df_ = _df_.with_columns(pl.concat_str(x_ops, separator=';').alias('x_anim'), pl.concat_str(y_ops, separator=';').alias('y_anim'))
        if 'c' in self.df_at_iteration_start[0].columns: _df_ = _df_.join(self.df_at_iteration_start[0].drop(['x','y','w']), on='__index__')
        else:                                            _df_ = _df_.with_columns(pl.lit('#000000').alias('c'))
        for i in range(len(_df_)):
            _color_ = _df_['c'][i]
            svg.append(f'<circle cx="{_df_["x_0"][i]}" cy="{_df_["y_0"][i]}" r="{r}" fill="{_color_}">')
            svg.append(f'<animate attributeName="cx" values="{_df_["x_anim"][i]}" dur="{animation_dur}" repeatCount="indefinite" />')
            svg.append(f'<animate attributeName="cy" values="{_df_["y_anim"][i]}" dur="{animation_dur}" repeatCount="indefinite" />')
            svg.append(f'</circle>')
        svg.append('</svg>')
        return ''.join(svg)

    def renderXOYO(self, rt):
        _df_ = self.df_xoyo_sector
        xo_min, yo_min, xo_max, yo_max = _df_['xo'].min(), _df_['yo'].min(), _df_['xo'].max(), _df_['yo'].max()
        dx, dy = xo_max - xo_min, yo_max - yo_min
        svg = [f'<svg x="0" y="0" width="900" height="900" viewBox="{xo_min-1} {yo_min-1} {dx+3} {dy+3}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="{xo_min-1}" y="{yo_min-1}" width="{dx+3}" height="{dy+3}" fill="#ffffff" />')
        for i in range(len(_df_)):
            xo, yo, sector, u, v, rsector, lsector = _df_['xo'][i], _df_['yo'][i], _df_['sector'][i], _df_['u'][i], _df_['v'][i], _df_['rsector'][i], _df_['lsector'][i]
            if sector != -1: 
                _color_ = rt.co_mgr.getColor(sector)
                svg.append(f'<rect x="{xo}" y="{yo}" width="1" height="1" fill="{_color_}" stroke="none" />')
            elif rsector is not None:
                _color_ = '#a0a0a0'
                svg.append(f'<rect x="{xo}" y="{yo}" width="1" height="1" fill="{_color_}" stroke="none" />')            
        svg.append('</svg>')
        return ''.join(svg)

    #
    # tileBounds()
    #
    def tileBounds(self, xi, yi):
        x0, x1 = xi/float(self.num_of_tiles), (xi+1)/float(self.num_of_tiles)
        y0, y1 = yi/float(self.num_of_tiles), (yi+1)/float(self.num_of_tiles)
        return (x0, y0, x1, y1)

    #
    # createXoYoDataframeFile() -- create the offset tiles dataframe
    # ... this is created ahead of time & saved off
    # ... it can be used for any values of the data
    # ... generation time is over 24m on 7900x 96gb
    #
    def createXoYoDataframeFile(self):

        def rayIntersectsSegment(xy_ray, uv_ray, xy0_segment, xy1_segment, include_xy1_endpoint=False, epsilon=1e-9):
            x_r,  y_r  = xy_ray
            dx_r, dy_r = uv_ray
            x0,   y0   = xy0_segment
            x1,   y1   = xy1_segment    
            # Segment direction vector
            dx_s, dy_s = x1 - x0, y1 - y0
            # Compute determinant
            det = -dx_r * dy_s + dy_r * dx_s
            if abs(det) < 1e-10: return None # Lines are parallel or collinear
            # Compute parameters t and u
            t = ((x_r - x0) * dy_s - (y_r - y0) * dx_s) / det
            u = ((x_r - x0) * dy_r - (y_r - y0) * dx_r) / det
            # Check if intersection is valid (t >= 0 for ray, 0 <= u <= 1 for segment)
            if t >= 0.0:
                if include_xy1_endpoint:
                    if 0.0 <= u <= 1.0+epsilon: return (x_r + t * dx_r, y_r + t * dy_r)
                else:
                    if 0.0 <= u <  1.0-epsilon: return (x_r + t * dx_r, y_r + t * dy_r)
            return None

        angles = []
        for i in range(16): angles.append(i*2*pi/16)

        dfs = []
        for num_of_tiles in [16, 32, 64, 128, 256, 512, 1024]:
            #
            # Prepare the xo/yo dataframe
            #
            _iota_ = 10e-6
            # Tile (xi,yi) --> (x0, y0, x1, y1) where the bounds are [x0,x1) and [y0,y1)
            tile_to_rect = {}
            for xi in range(num_of_tiles):
                x0, x1 = xi/float(num_of_tiles), (xi+1)/float(num_of_tiles)
                for yi in range(num_of_tiles):
                    y0, y1 = yi/float(num_of_tiles), (yi+1)/float(num_of_tiles)
                    tile_to_rect[(xi,yi)] = (x0, y0, x1, y1)
            # Determine which tiles (xo,yo) need to be checked for sector comparisons the hard way
            offtiles_intersected_by_rays = set()
            offtiles_to_sectors          = {}
            offtiles_to_uvs              = {}
            for _tile_ in [(0,0), (0, num_of_tiles-1), (num_of_tiles-1, num_of_tiles-1), (num_of_tiles-1, 0)]:
                xi,  yi  = _tile_
                x0,  y0, x1, y1 = tile_to_rect[_tile_]
                xm,  ym         = (x0+x1)/2.0, (y0+y1)/2.0
                _positions_ = [(x0+_iota_, y0+_iota_),(xm, y0+_iota_),(x1-_iota_, y0+_iota_),
                               (x0+_iota_, ym),       (xm, ym),       (x1-_iota_, ym),
                               (x0+_iota_, y1-_iota_),(xm, y1-_iota_),(x1-_iota_, y1-_iota_)]
                for _position_ in _positions_:
                    xpt, ypt = _position_
                    for _sector_ in range(16):
                        a0, a1         = angles[_sector_], angles[(_sector_+1)%16]
                        u0, v0, u1, v1 = cos(a0), sin(a0), cos(a1), sin(a1)
                        for _tile_ in tile_to_rect:
                            x0, y0, x1, y1 = tile_to_rect[_tile_]
                            _xo_, _yo_ = _tile_[0]-xi, _tile_[1]-yi
                            k          = (_xo_, _yo_)

                            if rayIntersectsSegment((xpt, ypt), (u0, v0), (x0, y0), (x0, y1)) or rayIntersectsSegment((xpt, ypt), (u0, v0), (x1, y0), (x1, y1)) or \
                               rayIntersectsSegment((xpt, ypt), (u0, v0), (x0, y0), (x1, y0)) or rayIntersectsSegment((xpt, ypt), (u0, v0), (x0, y1), (x1, y1)):
                                offtiles_intersected_by_rays.add(k)
                                if k not in offtiles_to_sectors: offtiles_to_sectors[k], offtiles_to_uvs[k] = set(), set()
                                offtiles_to_uvs[k].add((u0, v0))
                                offtiles_to_sectors[k].add((int(16*(atan2(y0-ypt, x0-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y0-ypt, x1-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y1-ypt, x1-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y1-ypt, x0-xpt)+pi)/(2*pi)) + 16)%16)

                            if rayIntersectsSegment((xpt, ypt), (u1, v1), (x0, y0), (x0, y1)) or rayIntersectsSegment((xpt, ypt), (u1, v1), (x1, y0), (x1, y1)) or \
                               rayIntersectsSegment((xpt, ypt), (u1, v1), (x0, y0), (x1, y0)) or rayIntersectsSegment((xpt, ypt), (u1, v1), (x0, y1), (x1, y1)):
                                offtiles_intersected_by_rays.add(k)
                                if k not in offtiles_to_sectors: offtiles_to_sectors[k], offtiles_to_uvs[k] = set(), set()
                                offtiles_to_uvs[k].add((u1, v1))
                                offtiles_to_sectors[k].add((int(16*(atan2(y0-ypt, x0-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y0-ypt, x1-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y1-ypt, x1-xpt)+pi)/(2*pi)) + 16)%16)
                                offtiles_to_sectors[k].add((int(16*(atan2(y1-ypt, x0-xpt)+pi)/(2*pi)) + 16)%16)

            # Determine the min and max x/y offsets
            xo_min, xo_max, yo_min, yo_max = 0, 0, 0, 0 # xi_min=-63 yi_min=-63 xi_max=63 yi_max=63 // for num_of_tiles = 64
            for _xyo_ in offtiles_intersected_by_rays:
                xo, yo = _xyo_
                xo_min, xo_max = min(xo, xo_min), max(xo, xo_max)
                yo_min, yo_max = min(yo, yo_min), max(yo, yo_max)
            xoyo_to_sector, sectors_seen = {}, set()
            # These tiles that are completely in a sector and do not touch the rays -- this determines the sector lookup
            for xo in range(xo_min, xo_max+1):
                for yo in range(yo_min, yo_max+1):
                    if (xo,yo) in offtiles_intersected_by_rays: continue
                    xm, ym   = xo, yo
                    _sector_ = int(16*(atan2(ym, xm)+pi)/(2*pi))
                    if _sector_ < 0: raise Exception('This should not happen // _sector_ < 0')
                    xoyo_to_sector[(xo,yo)] = _sector_
                    sectors_seen.add(_sector_)
            # Transform into a dataframe
            _lu_ = {'xo':[], 'yo':[], 'sector':[], 'u':[], 'v':[], 'rsector':[], 'lsector':[]}
            # These are the tiles that intersect the rays ... and so have to be figured out on a per point basis
            for _xyo_ in offtiles_intersected_by_rays:
                xo, yo = _xyo_
                _lu_['xo'].append(xo), _lu_['yo'].append(yo), _lu_['sector'].append(-1)
                # This shouldn't happen -- if there's an intersection, it should be at least two sectors (if not more)
                if   _xyo_ in offtiles_to_sectors and len(offtiles_to_sectors[_xyo_]) == 1: 
                    raise Exception('This should not happen // offtiles_to_sectors[_xyo_] == len(1)')
                # Two sectors intersected -- this can be used to determine the sector via the cross product
                elif _xyo_ in offtiles_to_sectors and len(offtiles_to_sectors[_xyo_]) == 2:
                    if len(offtiles_to_uvs[_xyo_]) != 1: # this shouldn't happen -- only one ray should have intersected if there are two sectors
                        print(_xyo_, offtiles_to_uvs[_xyo_])
                        raise Exception('This should not happen // len(offtiles_to_uvs[_xyo_]) != 1')
                    _sectors_ = sorted(list(offtiles_to_sectors[_xyo_]))
                    if _sectors_[0] == 0 and _sectors_[1] == 15: _sector0_, _sector1_ = 15,           0
                    else:                                        _sector0_, _sector1_ = _sectors_[0], _sectors_[1]
                    _lu_['u'].append(list(offtiles_to_uvs[_xyo_])[0][0]), _lu_['v'].append(list(offtiles_to_uvs[_xyo_])[0][1]), _lu_['rsector'].append(_sector0_), _lu_['lsector'].append(_sector1_)
                # Otherwise, we'll need to use the arctangent
                else:
                    _lu_['u'].append(None), _lu_['v'].append(None), _lu_['rsector'].append(None), _lu_['lsector'].append(None)
            # These are the tiles that don't intersect the rays ... and therefore can be optimized to a single sector
            for _xyo_ in xoyo_to_sector:
                xo, yo   = _xyo_
                _sector_ = xoyo_to_sector[_xyo_]
                _lu_['xo'].append(xo), _lu_['yo'].append(yo), _lu_['sector'].append(_sector_)
                _lu_['u'].append(None), _lu_['v'].append(None), _lu_['rsector'].append(None), _lu_['lsector'].append(None)
            # Create the dataframe and add it to the list of dataframes (each added one cover a specific num_of_tiles)
            df_xoyo_sector = pl.DataFrame(_lu_).with_columns(pl.lit(num_of_tiles).alias('num_of_tiles'))
            dfs.append(df_xoyo_sector)
        # Concatenate the dataframes together & write out to a file for use when the class starts up
        df = pl.concat(dfs)
        df.write_parquet(self.xoyo_filename)

    #
    # renderStages() -- rendered for debugging
    #
    def renderStages(self, rt, w_step=384, h_step=384, pt_i=0, iter=0):
        _tiles_ = []

        # Tile Determination
        _df_ = self.df_tile_determinations[iter]
        svg = [f'<svg x="0" y="0" width="{w_step}" height="{h_step}" viewBox="0.0 0.0 1.0 1.0" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')
        _xiyis_ = set()
        for i in range(len(_df_)):
            xi, yi      = _df_['xi'][i], _df_['yi'][i]
            _xiyis_.add((xi,yi))
        for _xy_ in _xiyis_:
            xi, yi = _xy_
            x0,y0,x1,y1 = self.tileBounds(xi,yi)
            svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#000000" stroke-width="0.001"/>')
        for i in range(len(_df_)):
            x, y, w, c = _df_['x'][i], _df_['y'][i], _df_['w'][i], _df_['c'][i]
            svg.append(f'<circle cx="{x}" cy="{y}" r="0.003" fill="{c}" />')
        svg.append(f'<text x="1.0" y="0.06" text-anchor="end" font-size="0.05" fill="#000000">{len(_xiyis_)} Tiles</text>')
        _df_ = _df_.filter(pl.col('__index__') == pt_i)
        svg.append('</svg>')

        _tiles_.append(''.join(svg))

        # Tile Sums
        _df_ = self.df_tile_sums[iter]
        svg = [f'<svg x="0" y="0" width="{w_step}" height="{h_step}" viewBox="0.0 0.0 1.0 1.0" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')
        _max_ = _df_['tile_sum'].max()
        for i in range(len(_df_)):
            xi, yi, tile_sum = _df_['xi'][i], _df_['yi'][i], _df_['tile_sum'][i]
            x0,y0,x1,y1 = self.tileBounds(xi,yi)
            _v_     = int(255 - 255 * tile_sum / _max_)
            _color_ = f'#{_v_:02x}{_v_:02x}{_v_:02x}'
            svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{_color_}" stroke="none"/>')
        svg.append(f'<text x="1.0" y="0.06" text-anchor="end" font-size="0.05" fill="#000000">{len(_df_)} Tiles</text>')
        svg.append('</svg>')
        _tiles_.append(''.join(svg))

        # Cross Joint Tile Offsets
        _df_ = self.df_cross_join_tile_offsets[iter].filter(pl.col('__index__') == pt_i)

        svg = [f'<svg x="0" y="0" width="{w_step}" height="{h_step}" viewBox="0.0 0.0 1.0 1.0" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')
        for i in range(len(_df_)):
            xi, yi      = _df_['xi'][i] + _df_['xo'][i],  _df_['yi'][i] + _df_['yo'][i]
            x0,y0,x1,y1 = self.tileBounds(xi,yi)
            svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#000000" stroke-width="0.002"/>')
        svg.append(f'<text x="1.0" y="0.06" text-anchor="end" font-size="0.05" fill="#000000">{len(_df_)} Tiles</text>')
        svg.append('</svg>')

        _tiles_.append(''.join(svg))

        # Sector Information
        _df_ = self.df_join_sector_info[iter].filter(pl.col('__index__') == pt_i)

        svg = [f'<svg x="0" y="0" width="{w_step}" height="{h_step}" viewBox="0.0 0.0 1.0 1.0" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')
        for xi in range(self.num_of_tiles):
            for yi in range(self.num_of_tiles):
                x0,y0,x1,y1 = self.tileBounds(xi,yi)
                svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#000000" stroke-width="0.0002"/>')
        seen_already = set()
        for i in range(len(_df_)):
            xi, yi, sector = _df_['xi_tile_sums'][i],  _df_['yi_tile_sums'][i], _df_['sector'][i]
            x0,y0,x1,y1    = self.tileBounds(xi,yi)
            _color_        = rt.co_mgr.getColor(sector)
            if sector == -1: _color_ = '#a0a0a0'
            if (xi,yi) in seen_already: _stroke_ = '#ff0000'
            else:                       _stroke_ = 'none'
            seen_already.add((xi,yi))
            svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{_color_}" stroke="{_stroke_}" stroke-width="0.005" />')
        x, y = _df_['x'][0], _df_['y'][0]
        svg.append(f'<circle cx="{x}" cy="{y}" r="0.01" fill="#000000" stroke="none"/>')
        for i in range(len(self.df_sector_angles[iter])):
            u, v = self.df_sector_angles[iter]['a0u'][i], self.df_sector_angles[iter]['a0v'][i]
            svg.append(f'<line x1="{x}" y1="{y}" x2="{x+2*u}" y2="{y+2*v}" stroke="#000000" stroke-width="0.003"/>')

        svg.append(f'<text x="1.0" y="0.06" text-anchor="end" font-size="0.05" fill="#000000">{len(_df_)} Tiles</text>')
        svg.append('</svg>')

        _tiles_.append(''.join(svg))

        # Separate Easy Way
        _df_ = self.df_separate_easy_way[iter].filter(pl.col('__index__') == pt_i)
        svg = [f'<svg x="0" y="0" width="{w_step}" height="{h_step}" viewBox="0.0 0.0 1.0 1.0" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')
        for xi in range(self.num_of_tiles):
            for yi in range(self.num_of_tiles):
                x0,y0,x1,y1 = self.tileBounds(xi,yi)
                svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#000000" stroke-width="0.0002"/>')
        seen_already = set()
        for i in range(len(_df_)):
            xi, yi, sector = _df_['xi_tile_sums'][i],  _df_['yi_tile_sums'][i], _df_['sector'][i]
            x0,y0,x1,y1    = self.tileBounds(xi,yi)
            _color_        = rt.co_mgr.getColor(sector)
            if sector == -1: _color_ = '#a0a0a0'
            if (xi,yi) in seen_already: _stroke_ = '#ff0000'
            else:                       _stroke_ = 'none'
            seen_already.add((xi,yi))
            svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{_color_}" stroke="{_stroke_}" stroke-width="0.005" />')
        x, y = _df_['x'][0], _df_['y'][0]
        svg.append(f'<circle cx="{x}" cy="{y}" r="0.01" fill="#000000" stroke="none"/>')
        for i in range(len(self.df_sector_angles[iter])):
            u, v = self.df_sector_angles[iter]['a0u'][i], self.df_sector_angles[iter]['a0v'][i]
            svg.append(f'<line x1="{x}" y1="{y}" x2="{x+2*u}" y2="{y+2*v}" stroke="#000000" stroke-width="0.003"/>')

        svg.append(f'<text x="1.0" y="0.06" text-anchor="end" font-size="0.05" fill="#000000">{len(_df_)} Tiles / Easy</text>')
        svg.append('</svg>')

        _tiles_.append(''.join(svg))

        # Medium Way Separation
        _df_ = self.df_separate_medium_way[iter].filter(pl.col('__index__') == pt_i)
        svg = [f'<svg x="0" y="0" width="{w_step}" height="{h_step}" viewBox="0.0 0.0 1.0 1.0" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')
        for xi in range(self.num_of_tiles):
            for yi in range(self.num_of_tiles):
                x0,y0,x1,y1 = self.tileBounds(xi,yi)
                svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#000000" stroke-width="0.0002"/>')
        for i in range(len(_df_)):
            xi, yi, sector = _df_['xi_tile_sums'][i],  _df_['yi_tile_sums'][i], _df_['sector'][i]
            x0,y0,x1,y1    = self.tileBounds(xi,yi)
            _color_        = rt.co_mgr.getColor(sector)
            if sector == -1: _color_ = '#a0a0a0'
            svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{_color_}" stroke="{_stroke_}" stroke-width="0.005" />')
        x, y = _df_['x'][0], _df_['y'][0]
        svg.append(f'<circle cx="{x}" cy="{y}" r="0.01" fill="#000000" stroke="none"/>')
        for i in range(len(self.df_sector_angles[iter])):
            u, v = self.df_sector_angles[iter]['a0u'][i], self.df_sector_angles[iter]['a0v'][i]
            svg.append(f'<line x1="{x}" y1="{y}" x2="{x+2*u}" y2="{y+2*v}" stroke="#000000" stroke-width="0.002"/>')
        _tile_count_ = 0
        for k, k_df in _df_.group_by(['xi_tile_sums', 'yi_tile_sums']): _tile_count_ += 1
        svg.append(f'<text x="1.0" y="0.06" text-anchor="end" font-size="0.05" fill="#000000">{len(_df_)} Points / Medium</text>')
        svg.append(f'<text x="1.0" y="0.12" text-anchor="end" font-size="0.05" fill="#000000">{_tile_count_} Tiles / Medium</text>')
        svg.append('</svg>')
        _tiles_.append(''.join(svg))

        # Hard Way Separation
        _df_ = self.df_separate_hard_way[iter].filter(pl.col('__index__') == pt_i)
        svg = [f'<svg x="0" y="0" width="{w_step}" height="{h_step}" viewBox="0.0 0.0 1.0 1.0" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')
        for xi in range(self.num_of_tiles):
            for yi in range(self.num_of_tiles):
                x0,y0,x1,y1 = self.tileBounds(xi,yi)
                svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#000000" stroke-width="0.0002"/>')
        for i in range(len(_df_)):
            xi, yi, sector = _df_['xi_tile_sums'][i],  _df_['yi_tile_sums'][i], _df_['sector'][i]
            x0,y0,x1,y1    = self.tileBounds(xi,yi)
            _color_        = rt.co_mgr.getColor(sector)
            if sector == -1: _color_ = '#a0a0a0'
            svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="{_color_}" stroke="{_stroke_}" stroke-width="0.005" />')
        if len(_df_) > 0:
            x, y = _df_['x'][0], _df_['y'][0]
            svg.append(f'<circle cx="{x}" cy="{y}" r="0.01" fill="#000000" stroke="none"/>')
            for i in range(len(self.df_sector_angles[iter])):
                u, v = self.df_sector_angles[iter]['a0u'][i], self.df_sector_angles[iter]['a0v'][i]
                svg.append(f'<line x1="{x}" y1="{y}" x2="{x+2*u}" y2="{y+2*v}" stroke="#000000" stroke-width="0.002"/>')
            _tile_count_ = 0
        for k, k_df in _df_.group_by(['xi_tile_sums', 'yi_tile_sums']): _tile_count_ += 1
        svg.append(f'<text x="1.0" y="0.06" text-anchor="end" font-size="0.05" fill="#000000">{len(_df_)} Points / Hard</text>')
        svg.append(f'<text x="1.0" y="0.12" text-anchor="end" font-size="0.05" fill="#000000">{_tile_count_} Tiles / Hard</text>')
        svg.append('</svg>')

        _tiles_.append(''.join(svg))

        # Medium Way Cross Products
        _df_ = self.df_medium_way_crossproducts[0].filter(pl.col('__index__') == pt_i)
        svg = [f'<svg x="0" y="0" width="{w_step}" height="{h_step}" viewBox="0.0 0.0 1.0 1.0" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="0" y="0" width="1.0" height="1.0" x="0" y="0" fill="#ffffff" />')
        already_seen = set()
        for i in range(len(_df_)):
            xi, yi        = _df_['xi_tile_sums'][i],  _df_['yi_tile_sums'][i]
            x0,y0,x1,y1   = self.tileBounds(xi,yi)
            _color_       = rt.co_mgr.getColor(sector)
            if (xi,yi) not in already_seen:
                svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#a0a0a0" stroke-width="0.001" />')
                already_seen.add((xi,yi))
        for i in range(len(self.df_sector_angles[iter])):
            u, v = self.df_sector_angles[iter]['a0u'][i], self.df_sector_angles[iter]['a0v'][i]
            svg.append(f'<line x1="{x}" y1="{y}" x2="{x+2*u}" y2="{y+2*v}" stroke="#000000" stroke-width="0.0003"/>')
        for i in range(len(_df_)):
            x, y, _sector_ = _df_['x_right'][i], _df_['y_right'][i], _df_['sector'][i]
            c = rt.co_mgr.getColor(_sector_)
            svg.append(f'<circle cx="{x}" cy="{y}" r="0.005" fill="{c}" />')
        svg.append(f'<text x="1.0" y="0.06" text-anchor="end" font-size="0.05" fill="#000000">{len(_df_)} Points / Medium</text>')
        svg.append(f'<text x="1.0" y="0.12" text-anchor="end" font-size="0.05" fill="#000000">{len(already_seen)} Tiles / Medium</text>')
        svg.append('</svg>')
        
        _tiles_.append(''.join(svg))

        return _tiles_

