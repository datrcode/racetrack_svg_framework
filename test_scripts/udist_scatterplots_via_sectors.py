#
# Polars implementation of the following:
#
# H. Rave, V. Molchanov and L. Linsen, "Uniform Sample Distribution in Scatterplots via Sector-based Transformation," 
# 2024 IEEE Visualization and Visual Analytics (VIS), St. Pete Beach, FL, USA, 2024, pp. 156-160, 
# doi: 10.1109/VIS55277.2024.00039. 
# keywords: {Data analysis;Visual analytics;Clutter;Scatterplot de-cluttering;spatial transformation},
#
import polars  as     pl
import numpy   as     np
from   math    import pi, sin, cos, atan2
from   shapely import Polygon
import time

__name__ = 'udist_scatterplots_via_sectors'

class UDistScatterPlotsViaSectors(object):
    def __init__(self, x_vals=[], y_vals=[], weights=None, colors=None, vector_scalar=0.01, iterations=4, debug=False):
        self.vector_scalar = vector_scalar
        self.iterations    = iterations
        self.debug         = debug
        self.time_lu       = {'prepare_df':0.0, 'normalize':0.0, 'all_sectors':0.0, 'explode_points':0.0, 'arctangents':0.0, 'sector_sums':0.0, 
                              'add_missing_sectors':0.0, 'prepare_sector_angles':0.0, 'join_sector_angles':0.0, 'ray_segment_intersections':0.0,
                              'area_calc':0.0, 'sector_uv_summation':0.0, 'point_update':0.0,}

        # Create the debugging structures
        self.df_at_iteration_start    = []
        self.df_sector_sums           = []
        self.df_sector_fill           = []
        self.df_sector_determinations = []
        self.df_sector_angles         = []
        self.df_sector_angles_joined  = []
        self.df_fully_filled          = []
        self.df_uv                    = []

        # Create weights if none were set
        if weights is None: weights = np.ones(len(x_vals))

        # Prepare the initial dataframe
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

            #
            # Normalize the points to 0.02 to 0.98 (want to give it a little space around the edges to that there are sectors to move into)
            #
            t = time.time()
            df = df.with_columns((0.02 + 0.96 * (pl.col('x') - pl.col('x').min())/(pl.col('x').max() - pl.col('x').min())).alias('x'), 
                                 (0.02 + 0.96 * (pl.col('y') - pl.col('y').min())/(pl.col('y').max() - pl.col('y').min())).alias('y'))
            if debug: self.df_at_iteration_start.append(df.clone())
            self.time_lu['normalize'] += (time.time() - t)

            #
            # All Sectors DataFrame
            #
            t = time.time()
            df_all_sectors = df.join(pl.DataFrame({'sector': [i for i in range(16)]}), how='cross').drop(['w','c'])
            self.time_lu['all_sectors'] += (time.time() - t)

            #
            # vvv -- PERFORMANCE ISSUE
            #

            #
            # Multiply out the points against all the other points
            # ... greatly explodes the dataframe
            #
            t = time.time()
            df = df.with_columns(pl.struct(['x','y','__index__']).implode().alias('_implode_')) \
                   .explode('_implode_')                                            \
                   .with_columns(pl.col('_implode_').struct.field('x')        .alias('_xo_'),
                                 pl.col('_implode_').struct.field('y')        .alias('_yo_'),
                                 pl.col('_implode_').struct.field('__index__').alias('_indexo_'))
            df = df.filter(pl.col('__index__') != pl.col('_indexo_')) # don't compare the point with itself
            self.time_lu['explode_points'] += (time.time() - t)

            #
            # Determine the sector for the other point in relationship to this point...
            #
            t = time.time()
            _dx_ = pl.col('_xo_') - pl.col('x')
            _dy_ = pl.col('_yo_') - pl.col('y')
            df   = df.with_columns(((16*(pl.arctan2(_dy_, _dx_) + pl.lit(pi))/(pl.lit(2*pi))).cast(pl.Int64)).alias('sector'))
            if debug: self.df_sector_determinations.append(df.clone())
            self.time_lu['arctangents'] += (time.time() - t)

            #
            # Sum the weights for each sector ... this is missing sectors (empty sectors (which are the ones missing) are needed later for the algo to work correctly)
            #
            t = time.time()
            df   = df.group_by(['__index__','x','y','sector']).agg((pl.col('w').sum()).alias('_w_sum_'), (pl.col('w').sum() / df_weight_sum).alias('_w_ratio_'))
            if debug: self.df_sector_sums.append(df.clone())
            self.time_lu['sector_sums'] += (time.time() - t)

            #
            # ^^^ -- PERFORMANCE ISSUE
            #

            #
            # Add the missing sectors back in...
            #
            t = time.time()
            df = df_all_sectors.join(df, on=['__index__','x','y','sector'], how='left').with_columns(pl.col('_w_sum_').fill_null(0), pl.col('_w_ratio_').fill_null(0))
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

#
#
# The following are reference implementations that are straightforward python
#
#

#
# xyUniformSampleDistributionSectorTransform() - implementation of the referenced paper
#
def xyUniformSampleDistributionSectorTransformDEBUG(rt, xvals, yvals, weights=None, colors=None, iterations=4, sectors=16, vector_scalar=0.01):
    svgs, svgs_for_sectors = [], []
    _fine_ = {'iteration':[], 'point_i':[], 'x':[], 'y':[], 's':[], 's_wgt':[], 's_area':[], 's_u':[], 's_v':[], 'u_sum':[], 'v_sum':[]}

    # Normalize the coordinates to be between 0.0 and 1.0
    def normalizeCoordinates(xs, ys):
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        if xmin == xmax: xmin -= 0.0001; xmax += 0.0001
        if ymin == ymax: ymin -= 0.0001; ymax += 0.0001
        xs_new, ys_new = [], []
        for x, y in zip(xs, ys):
            xs_new.append(0.02 + 0.96*(x-xmin)/(xmax-xmin))
            ys_new.append(0.02 + 0.96*(y-ymin)/(ymax-ymin))
        return xs_new, ys_new
    # Force all the coordinates to be between 0 and 1
    xvals, yvals = normalizeCoordinates(xvals, yvals)    
    xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
    # Determine the average density (used for expected density calculations)
    if weights is None: weights = np.ones(len(xvals))
    weight_sum   = sum(weights)
    area_total   = ((xmax-xmin)*(ymax-ymin))
    density_avg  = weight_sum / area_total
    iters        = 0
    # Determine the side and xy that a specific ray hits
    def sideAndXY(xy, uv):
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmin, ymin), (xmax, ymin))
        if _xyi_ is not None: return 0, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmax, ymin), (xmax, ymax))
        if _xyi_ is not None: return 1, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmax, ymax), (xmin, ymax))
        if _xyi_ is not None: return 2, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmin, ymax), (xmin, ymin))
        if _xyi_ is not None: return 3, _xyi_
        # hacking the corner cases ... literally the corners
        if xy[0] >= xmin and xy[0] <= xmax and xy[1] >= ymin and xy[1] <= ymax:
            if uv == (0.0, 0.0):
                print(xy, uv, (xmin,ymin,xmax,ymax))
                raise Exception('No Intersection Found for sideAndXY() ... ray is (0,0)')
            else:
                xp, yp, up, vp = round(xy[0], 2), round(xy[1], 2), round(uv[0], 2), round(uv[1], 2)
                if abs(xp) == abs(yp) and abs(up) == abs(vp):
                    if   up < 0.0 and vp < 0.0: return 0, (xmin, ymin)
                    elif up < 0.0 and vp > 0.0: return 1, (xmax, ymin)
                    elif up > 0.0 and vp > 0.0: return 2, (xmax, ymax)
                    elif up > 0.0 and vp < 0.0: return 3, (xmin, ymax)
                print(xy, uv, (xmin,ymin,xmax,ymax))
                raise Exception('No Intersection Found for sideAndXY() ... xy or uv are not equal to one another')
        else:
            print(xy, uv, (xmin,ymin,xmax,ymax))
            raise Exception('No Intersection Found for sideAndXY() ... point not within bounds')
    # Calculate the sector angles
    _sector_angles_, _sector_anchor_ = [], []
    a, ainc = 0.0, 2*pi/sectors
    for s in range(sectors):
        _sector_angles_.append((a, a+ainc))
        _sector_anchor_.append(a + pi + ainc/2.0)
        a += ainc
    # Calculate the UV vector for a specific point
    def ptUVVec(x, y, point_i):
        svg_sectors = [f'<svg x="0" y="0" width="512" height="512" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}" xmlns="http://www.w3.org/2000/svg">']
        svg_sectors.append(f'<rect x="{xmin}" y="{ymin}" width="{xmax-xmin}" height="{ymax-ymin}" fill="#ffffff" />')
        _sector_sum_ = {}
        for s in range(sectors): _sector_sum_[s] = 0.0
        # Iterate over all points ... adding to the sector sum for the correct sector
        for i in range(len(xvals)):
            _x_, _y_, _w_ = xvals[i], yvals[i], weights[i]
            if _x_ == x and _y_ == y: continue
            _dx_, _dy_ = _x_ - x, _y_ - y
            a = atan2(_dy_, _dx_)
            if a < 0.0: a += 2*pi
            _sector_found_ = False
            for s in range(sectors):
                if a >= _sector_angles_[s][0] and a < _sector_angles_[s][1]:
                    _sector_sum_[s] += _w_
                    _color_ = rt.co_mgr.getColor(s)
                    svg_sectors.append(f'<circle cx="{_x_}" cy="{_y_}" r="0.01" stroke="#000000" stroke-width="0.001" fill="{_color_}" />')
                    svg_sectors.append(f'<line x1="{x}" y1="{y}" x2="{_x_}" y2="{_y_}" stroke="#000000" stroke-width="0.001" />')
                    _sector_found_ = True
                    break
            if not _sector_found_: print('No sector found for point', _x_, _y_, a)
        # Determine the area for each sector (from this points perspective)
        _sector_area_, _poly_definition_ = {}, {}
        for s in range(sectors):
            uv          = (cos(_sector_angles_[s][0]), sin(_sector_angles_[s][0]))
            side_and_xy_a0 = sideAndXY((x,y), uv)
            uv = (cos(_sector_angles_[s][1]), sin(_sector_angles_[s][1]))
            side_and_xy_a1 = sideAndXY((x,y), uv)
            if side_and_xy_a0[0] == side_and_xy_a1[0]: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], side_and_xy_a1[1]] # same side
            else:
                if   side_and_xy_a0[0] == 0 and side_and_xy_a1[0] == 1: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmax,ymin), side_and_xy_a1[1]] # top 
                elif side_and_xy_a0[0] == 1 and side_and_xy_a1[0] == 2: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmax,ymax), side_and_xy_a1[1]] # right
                elif side_and_xy_a0[0] == 2 and side_and_xy_a1[0] == 3: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmin,ymax), side_and_xy_a1[1]] # bottom
                elif side_and_xy_a0[0] == 3 and side_and_xy_a1[0] == 0: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmin,ymin), side_and_xy_a1[1]] # left
            _poly_ = Polygon(_poly_definition_[s])
            _sector_area_[s] = _poly_.area
        # From the paper ... weight the anchor the difference between the expected and actual density
        _scalar_ = vector_scalar
        u, v = 0.0, 0.0
        for s in range(sectors):
            _diff_ = (_sector_sum_[s]/weight_sum) - (_sector_area_[s]/area_total)
            u, v   = u + _scalar_ * _diff_ * cos(_sector_anchor_[s]), v + _scalar_ * _diff_ * sin(_sector_anchor_[s])
            _poly_coords_ = _poly_definition_[s]
            d      = f'M {_poly_coords_[0][0]} {_poly_coords_[0][1]} '
            for i in range(1, len(_poly_coords_)): d += f'L {_poly_coords_[i][0]} {_poly_coords_[i][1]} '
            d += 'Z'
            if _diff_ < 0.0: _color_ = rt.co_mgr.getColor(s) # '#0000ff'
            else:            _color_ = rt.co_mgr.getColor(s) # '#ff0000'
            svg_sectors.append(f'<path d="{d}" stroke="{rt.co_mgr.getColor(s)}" fill="{_color_}" fill-opacity="0.3" stroke-width="0.002"/>')
            _fine_['iteration'].append(iters)
            _fine_['point_i'].append(point_i)
            _fine_['x'].append(x)
            _fine_['y'].append(y)
            _fine_['s'].append(s)
            _fine_['s_wgt'].append(_sector_sum_ [s])
            _fine_['s_area'].append(_sector_area_[s])
            _fine_['s_u'].append(cos(_sector_anchor_[s]))
            _fine_['s_v'].append(sin(_sector_anchor_[s]))
            _fine_['u_sum'].append(u)
            _fine_['v_sum'].append(v)

        # Return the value
        svg_sectors.append(f'<line x1="{x}" y1="{y}" x2="{x+3*u}" y2="{y+3*v}" stroke="#ff0000" stroke-width="0.01" />')
        svg_sectors.append('</svg>')
        svgs_for_sectors.append(''.join(svg_sectors))
        return u,v

    # Iterations...
    _df_  = pl.DataFrame({'x':xvals, 'y':yvals, 'c':colors})
    dfs = [_df_]
    while iters < iterations:
        svg = [f'<svg x="0" y="0" width="256" height="256" viewBox="{xmin} {ymin} {xmax-xmin} {ymax-ymin}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<rect x="{xmin}" y="{ymin}" width="{xmax-xmin}" height="{ymax-ymin}" x="0" y="0" fill="#ffffff" />')
        xvals_next, yvals_next = [], []
        for j in range(len(xvals)):
            _x_, _y_ = xvals[j], yvals[j]
            uv = ptUVVec(_x_, _y_, j)
            svg.append(f'<line x1="{_x_}" y1="{_y_}" x2="{_x_+uv[0]}" y2="{_y_+uv[1]}" stroke="#a0a0a0" stroke-width="0.001" />')
            _color_ = colors[j] if colors is not None else '#000000'
            svg.append(f'<circle cx="{_x_}" cy="{_y_}" r="0.004" fill="{_color_}" />')
            _x_next_, _y_next_ = _x_ + uv[0], _y_ + uv[1]
            xvals_next.append(_x_next_), yvals_next.append(_y_next_)
        svg.append('</svg>')
        svgs.append(''.join(svg))
        dfs.append(pl.DataFrame({'x':dfs[0]['x'], 'y':dfs[0]['y'], 'xn':xvals_next, 'yn':yvals_next}))
        xvals, yvals = xvals_next, yvals_next
        xvals, yvals = normalizeCoordinates(xvals, yvals)
        xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
        iters += 1

    # Copy of the animation creation from the class / modified to work here
    r, animation_dur, w, h = 0.004, "4s", 512, 512
    x0, y0, x1, y1 = -0.01, -0.01, 1.02, 1.02
    svg = [f'<svg x="0" y="0" width="{w}" height="{h}" viewBox="{x0} {y0} {x1-x0} {y1-y0}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" x="0" y="0" fill="#ffffff" />')
    _df_ = dfs[0]
    x_ops, y_ops = [pl.col('xn_0')], [pl.col('yn_0')]
    for i in range(1, len(dfs)): 
        _df_ = _df_.join(dfs[i], on=['x','y'], suffix=f'_{i}')
        x_ops.append(pl.col(f'xn_{i}')), y_ops.append(pl.col(f'yn_{i}'))
    x_ops.extend(x_ops[::-1]), y_ops.extend(y_ops[::-1])
    x_ops.extend(['xn_0', 'xn_0', 'xn_0']), y_ops.extend(['yn_0', 'yn_0', 'yn_0']) # so there's a slight delay before it starts all over again
    _df_  = _df_.rename({'x':'xn_0','y':'yn_0', 'xn':'xn_1', 'yn':'yn_1'})
    _df_ = _df_.with_columns(pl.concat_str(x_ops, separator=';').alias('x_anim'), pl.concat_str(y_ops, separator=';').alias('y_anim'))
    for i in range(len(_df_)):
        _color_ = _df_['c'][i]
        svg.append(f'<circle cx="{_df_["xn_0"][i]}" cy="{_df_["yn_0"][i]}" r="{r}" fill="{_color_}">')
        svg.append(f'<animate attributeName="cx" values="{_df_["x_anim"][i]}" dur="{animation_dur}" repeatCount="indefinite" />')
        svg.append(f'<animate attributeName="cy" values="{_df_["y_anim"][i]}" dur="{animation_dur}" repeatCount="indefinite" />')
        svg.append(f'</circle>')
    svg.append('</svg>')
    svg_animation = ''.join(svg)

    # Return
    return xvals, yvals, svgs, svgs_for_sectors, svg_animation, pl.DataFrame(_fine_)

#
# xyUniformSampleDistributionSectorTransform() - implementation of the referenced paper
# ... the above version is debug ... this removes all of the svg creation
# ... may have messed this one up in the conversion process...
#
def xyUniformSampleDistributionSectorTransform(rt, xvals, yvals, weights=None, colors=None, iterations=4, sectors=16, vector_scalar=0.01):
    # Normalize the coordinates to be between 0.0 and 1.0
    def normalizeCoordinates(xs, ys):
        xmin, ymin, xmax, ymax = min(xs), min(ys), max(xs), max(ys)
        if xmin == xmax: xmin -= 0.0001; xmax += 0.0001
        if ymin == ymax: ymin -= 0.0001; ymax += 0.0001
        xs_new, ys_new = [], []
        for x, y in zip(xs, ys):
            xs_new.append(0.02 + 0.96*(x-xmin)/(xmax-xmin))
            ys_new.append(0.02 + 0.96*(y-ymin)/(ymax-ymin))
        return xs_new, ys_new
    # Force all the coordinates to be between 0 and 1
    xvals, yvals = normalizeCoordinates(xvals, yvals)    
    xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
    # Determine the average density (used for expected density calculations)
    if weights is None: weights = np.ones(len(xvals))
    weight_sum   = sum(weights)
    area_total   = ((xmax-xmin)*(ymax-ymin))
    density_avg  = weight_sum / area_total
    iters        = 0

    def sideAndXY(xy, uv):
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmin, ymin), (xmax, ymin))
        if _xyi_ is not None: return 0, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmax, ymin), (xmax, ymax))
        if _xyi_ is not None: return 1, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmax, ymax), (xmin, ymax))
        if _xyi_ is not None: return 2, _xyi_
        _xyi_ = rt.rayIntersectsSegment(xy, uv, (xmin, ymax), (xmin, ymin))
        if _xyi_ is not None: return 3, _xyi_
        # hacking the corner cases ... literally the corners
        if xy[0] >= xmin and xy[0] <= xmax and xy[1] >= ymin and xy[1] <= ymax:
            if uv == (0.0, 0.0):
                print(xy, uv, (xmin,ymin,xmax,ymax))
                raise Exception('No Intersection Found for sideAndXY() ... ray is (0,0)')
            else:
                xp, yp, up, vp = round(xy[0], 2), round(xy[1], 2), round(uv[0], 2), round(uv[1], 2)
                if abs(xp) == abs(yp) and abs(up) == abs(vp):
                    if   up < 0.0 and vp < 0.0: return 0, (xmin, ymin)
                    elif up < 0.0 and vp > 0.0: return 1, (xmax, ymin)
                    elif up > 0.0 and vp > 0.0: return 2, (xmax, ymax)
                    elif up > 0.0 and vp < 0.0: return 3, (xmin, ymax)
                print(xy, uv, (xmin,ymin,xmax,ymax))
                raise Exception('No Intersection Found for sideAndXY() ... xy or uv are not equal to one another')
        else:
            print(xy, uv, (xmin,ymin,xmax,ymax))
            raise Exception('No Intersection Found for sideAndXY() ... point not within bounds')
    # Calculate the sector angles
    _sector_angles_, _sector_anchor_ = [], []
    a, ainc = 0.0, 2*pi/sectors
    for s in range(sectors):
        _sector_angles_.append((a, a+ainc))
        _sector_anchor_.append(a + pi + ainc/2.0)
        a += ainc
    # Calculate the UV vector for a specific point
    def ptUVVec(x,y):
        _sector_sum_ = {}
        for s in range(sectors): _sector_sum_[s] = 0.0
        # Iterate over all points ... adding to the sector sum for the correct sector
        for i in range(len(xvals)):
            _x_, _y_, _w_ = xvals[i], yvals[i], weights[i]
            if _x_ == x and _y_ == y: continue
            _dx_, _dy_ = _x_ - x, _y_ - y
            a = atan2(_dy_, _dx_)
            if a < 0.0: a += 2*pi
            _sector_found_ = False
            for s in range(sectors):
                if a >= _sector_angles_[s][0] and a < _sector_angles_[s][1]:
                    _sector_sum_[s] += _w_
                    _sector_found_   = True
                    break
            if not _sector_found_: print('No sector found for point', _x_, _y_, a)
        # Determine the area for each sector (from this points perspective)
        _sector_area_, _poly_definition_ = {}, {}
        for s in range(sectors):
            uv          = (cos(_sector_angles_[s][0]), sin(_sector_angles_[s][0]))
            side_and_xy_a0 = sideAndXY((x,y), uv)
            uv = (cos(_sector_angles_[s][1]), sin(_sector_angles_[s][1]))
            side_and_xy_a1 = sideAndXY((x,y), uv)
            if side_and_xy_a0[0] == side_and_xy_a1[0]: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], side_and_xy_a1[1]] # same side
            else:
                if   side_and_xy_a0[0] == 0 and side_and_xy_a1[0] == 1: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmax,ymin), side_and_xy_a1[1]] # top 
                elif side_and_xy_a0[0] == 1 and side_and_xy_a1[0] == 2: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmax,ymax), side_and_xy_a1[1]] # right
                elif side_and_xy_a0[0] == 2 and side_and_xy_a1[0] == 3: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmin,ymax), side_and_xy_a1[1]] # bottom
                elif side_and_xy_a0[0] == 3 and side_and_xy_a1[0] == 0: _poly_definition_[s] = [(x,y), side_and_xy_a0[1], (xmin,ymin), side_and_xy_a1[1]] # left
            _poly_ = Polygon(_poly_definition_[s])
            _sector_area_[s] = _poly_.area
        # From the paper ... weight the anchor the difference between the expected and actual density
        _scalar_ = vector_scalar
        u, v = 0.0, 0.0
        for s in range(sectors):
            _diff_ = (_sector_sum_[s]/weight_sum) - (_sector_area_[s]/area_total)
            u, v   = u + _scalar_ * _diff_ * cos(_sector_anchor_[s]), v + _scalar_ * _diff_ * sin(_sector_anchor_[s])
        return u,v
    
    # Iterations...
    _df_  = pl.DataFrame({'x':xvals, 'y':yvals, 'c':colors})
    dfs = [_df_]
    while iters < iterations:
        xvals_next, yvals_next = [], []
        for j in range(len(xvals)):
            _x_, _y_ = xvals[j], yvals[j]
            uv = ptUVVec(_x_, _y_)
            _x_next_, _y_next_ = _x_ + uv[0], _y_ + uv[1]
            xvals_next.append(_x_next_), yvals_next.append(_y_next_)
        xvals, yvals = xvals_next, yvals_next
        xvals, yvals = normalizeCoordinates(xvals, yvals)
        xmin, ymin, xmax, ymax = 0.0, 0.0, 1.0, 1.0
        iters += 1

    # Return
    return xvals, yvals

